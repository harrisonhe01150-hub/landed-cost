"""Landed Cost: photo of a supplier delivery note -> landed cost per unit in rand -> wholesale price per bale.

Run:  python app.py      then open http://127.0.0.1:5060
Everything (photos, prices, settings) stays in ./data on this computer. The vision models run in local Ollama.
"""
import json
import math
import os
import re
import shutil
import tempfile
import threading
import unicodedata
import uuid
import xml.etree.ElementTree as ET
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

import requests
from flask import Flask, abort, jsonify, redirect, render_template_string, request, send_file, url_for

from lc import classify, extract
from lc.cost import DEFAULT_CATEGORY, Settings, landed

PORT = 5060
DATA = Path(os.environ.get("LC_DATA", Path(__file__).resolve().parent / "data"))
DOCS = DATA / "docs"
SETTINGS = DATA / "settings.json"
REFERENCE = json.loads((Path(__file__).resolve().parent / "lc" / "reference" / "sars_duty_2026-08-28.json").read_text(encoding="utf-8"))
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
app.config["TRUSTED_HOSTS"] = ["127.0.0.1", "localhost"]  # blocks DNS-rebinding pages from reading the docs
_IO = threading.Lock()


# ---------- storage: one lock, atomic writes, tolerant reads ----------

def read_json(path: Path, default=None):
    with _IO:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return default
        except (OSError, ValueError):
            app.logger.warning("unreadable %s, using defaults", path)
            return default


def save_json(path: Path, obj) -> None:
    data = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with _IO:
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)  # a power cut leaves either the old file or the new one, never half of one
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)


def load_settings() -> Settings:
    raw = read_json(SETTINGS, {})
    raw = raw if isinstance(raw, dict) else {}
    known = {f.name for f in fields(Settings)}
    return Settings(**{"fx_zar_per_cny": 0.0, **{k: v for k, v in raw.items() if k in known}})


def doc_path(doc_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{10}", doc_id or ""):
        abort(404)
    return DOCS / doc_id


def load_doc(doc_id: str) -> tuple[Path, dict]:
    p = doc_path(doc_id) / "doc.json"
    doc = read_json(p)
    if not isinstance(doc, dict):
        abort(404)
    return p, doc


# ---------- input parsing ----------

class Bad(ValueError):
    pass


def num(v, decimal_comma: bool = True):
    """Parse what people actually type: '2,46' (SA decimal comma), '1 450', '1,200.50', '￥12.5', '45％', 'R15 000'.
    Empty -> None. Anything else unparseable -> raises Bad, so a typo never silently becomes 0."""
    s = unicodedata.normalize("NFKC", str(v or "")).replace("。", ".").strip()
    s = re.sub(r"(?i)\b(zar|cny|rmb)\b|[r¥元兰特%\s]", "", s)
    if not s:
        return None
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s):
        s = s.replace(",", "")  # thousands separators
    elif s.count(",") == 1 and "." not in s and decimal_comma:
        s = s.replace(",", ".")  # 2,46
    try:
        x = float(s)
    except ValueError:
        raise Bad(v) from None
    if not math.isfinite(x):
        raise Bad(v)
    return x


def num_or_none(v):
    """For the delivery-note table: a value that can't be read becomes empty (and the cross-check flags the row)."""
    try:
        return num(v, decimal_comma=False)
    except Bad:
        return None


# ---------- CSRF / origin guard: only this page may change things ----------

@app.before_request
def same_origin_posts_only():
    if request.method != "POST":
        return
    site = request.headers.get("Sec-Fetch-Site")
    if site is not None and site not in ("same-origin", "none"):
        abort(403)
    origin = request.headers.get("Origin")
    if origin is None and request.referrer:
        p = urlsplit(request.referrer)
        origin = f"{p.scheme}://{p.netloc}"
    if origin is not None and origin not in {f"http://127.0.0.1:{PORT}", f"http://localhost:{PORT}", request.host_url.rstrip("/")}:
        abort(403)


def safe_next(target: str | None) -> str:
    return target if target and target.startswith("/") and not target.startswith("//") else url_for("home")


# ---------- WhatsApp summary ----------

def issues_of(doc: dict, results: list[dict]) -> list[str]:
    out = list(doc.get("problems", []))
    for r in results:
        name = f"{r.get('sku', '')} {r.get('name', '')}".strip()
        if r.get("error"):
            out.append(f"{name}：{r['error'].split(' · ')[0]}")
        for w in r.get("warnings", []):
            out.append(f"{name}：{w}")
        if r.get("cbm_estimated"):
            out.append(f"{name}：体积是按金额从总体积里估的")
        if r.get("second_opinion"):
            out.append(f"{name}：这行的数是第二个模型重读的，请看一眼")
    return out


def summary_text(doc: dict, results: list[dict], s: Settings) -> str:
    if s.mode == "allin":
        how = f"包税 {'按类别' if s.allin_rates else f'{s.allin_rate_per_cbm:g} {s.allin_rate_currency}/方'}"
    else:
        how = f"海运 R{s.freight_per_cbm_zar:g}/方 · 关税按类别" + (" · 清关按本单比例" if s.clearing_fixed_zar else "")
    issues = issues_of(doc, results)
    lines = []
    if issues:
        lines.append(f"⚠️ 还有 {len(issues)} 处没核对，价格仅供参考")
    lines.append(f"📦 {doc.get('supplier', '')} {doc.get('doc_no', '')} {doc.get('date', '')}".strip())
    lines.append(f"汇率 {s.fx_zar_per_cny:g} · {how} · 加价 ×{s.markup:g}（内部用，含成本）")
    for r in results:
        name = f"{r.get('sku', '')} {r.get('name', '')}".strip()
        if r.get("error"):
            lines.append(f"⚠️ {name}：没算价（{r['error'].split(' · ')[0]}）")
            continue
        u = r.get("unit_label") or "件"
        bale = f" · R{r['suggested_per_bale']:,.0f}/包({r['pcs_per_bale']:g}{u})" if r.get("suggested_per_bale") else ""
        flag = "⚠️ " if (r.get("problems") or r.get("warnings") or r.get("second_opinion") or r.get("cbm_estimated")) else ""
        lines.append(f"{flag}{name}：成本 R{r['landed_per_piece']:.2f}/{u} → 建议批发 R{r['suggested_per_piece']:.2f}/{u}{bale}")
    if issues:
        lines.append("待核对：")
        lines += [f"· {i}" for i in issues]
    return "\n".join(lines)


# ---------- pages ----------

def sort_categories(doc: dict, s: Settings) -> bool:
    cats = s.choices()
    if not cats:
        return True
    got = classify.categories([f"{r.get('name', '')} {r.get('spec', '')}" for r in doc["rows"]], cats)
    if got is None:
        return False
    for r, c in zip(doc["rows"], got):
        if c:
            r["category"] = c
    return True


@app.get("/")
def home():
    docs = []
    for d in sorted(DOCS.glob("*/doc.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:12]:
        j = read_json(d)
        if not isinstance(j, dict):
            continue
        title = f"{j.get('supplier', '')} {j.get('doc_no', '')}".strip() or f"未命名单据 {j.get('read_at', '')[:16]}"
        docs.append({"id": d.parent.name, "title": title, "date": j.get("date", ""), "rows": len(j.get("rows", [])),
                     "problems": len(j.get("problems", []))})
    s = load_settings()
    return render_template_string(HOME_HTML, docs=docs, missing=s.missing(), error=request.args.get("error", ""))


@app.post("/read")
def read():
    f = request.files.get("photo")
    if not f or not f.filename:
        return redirect(url_for("home"))
    ext = Path(f.filename).suffix.lower()
    ext = ext if ext in IMAGE_SUFFIXES else ".jpg"
    doc_id = uuid.uuid4().hex[:10]
    folder = DOCS / doc_id
    folder.mkdir(parents=True)
    photo = folder / ("photo" + ext)
    f.save(photo)
    try:
        doc = extract.read_checked(photo)
    except requests.RequestException as e:
        shutil.rmtree(folder, ignore_errors=True)
        resp = getattr(e, "response", None)
        why = (f"本地模型 {extract.MODEL} 还没下载：先运行 ollama pull {extract.MODEL}" if resp is not None and resp.status_code == 404
               else "连不上本地 Ollama：请先打开 Ollama 再读 · Ollama is not running" if isinstance(e, requests.ConnectionError)
               else f"本地模型出错或超时：{str(e)[:120]}")
        return redirect(url_for("home", error=why))
    except (ValueError, KeyError) as e:
        shutil.rmtree(folder, ignore_errors=True)
        return redirect(url_for("home", error=f"模型返回的内容读不懂，请再试一次：{str(e)[:120]}"))
    doc.update(photo=photo.name, read_at=datetime.now().isoformat(timespec="seconds"))
    sort_categories(doc, load_settings())
    save_json(folder / "doc.json", doc)
    return redirect(url_for("show", doc_id=doc_id))


@app.get("/doc/<doc_id>/photo")
def photo(doc_id):
    _, doc = load_doc(doc_id)
    p = doc_path(doc_id) / doc.get("photo", "")
    if p.suffix.lower() not in IMAGE_SUFFIXES or not p.is_file():
        abort(404)
    return send_file(p)


@app.post("/doc/<doc_id>/classify")
def reclassify(doc_id):
    p, doc = load_doc(doc_id)
    if not sort_categories(doc, load_settings()):
        return redirect(url_for("show", doc_id=doc_id, note="model_down"))
    save_json(p, doc)
    return redirect(url_for("show", doc_id=doc_id))


EDITABLE = {"sku": "sku", "name": "name", "spec": "spec", "per": "pcs_per_bale", "bales": "bales", "qty": "qty_pcs",
            "price": "unit_price_rmb", "amount": "amount_rmb", "carton": "carton_cm", "cbm": "cbm", "unit": "unit",
            "cat": "category"}
NUMERIC = {"pcs_per_bale", "bales", "qty_pcs", "unit_price_rmb", "amount_rmb", "cbm"}


@app.route("/doc/<doc_id>", methods=["GET", "POST"])
def show(doc_id):
    p, doc = load_doc(doc_id)
    if request.method == "POST":
        try:
            n = int(request.form.get("n", "0"))
        except ValueError:
            abort(400)
        old_rows, rows = doc.get("rows", []), []
        for i in range(min(n, len(old_rows) + 2)):  # existing rows plus the two blank ones
            if request.form.get(f"del{i}"):
                continue
            g = {field: request.form.get(f"{form}{i}", "") for form, field in EDITABLE.items()}
            if i >= len(old_rows) and not any(v.strip() for k, v in g.items() if k != "category"):
                continue  # untouched blank row
            new = {k: (num_or_none(v) if k in NUMERIC else v.strip()) for k, v in g.items()}
            old = old_rows[i] if i < len(old_rows) else {}
            keep = {k: v for k, v in old.items() if k not in ("problems", "second_opinion")}
            row = {**keep, **new}
            if old.get("second_opinion") and all(old.get(k) == row.get(k) for k in NUMERIC):
                row["second_opinion"] = old["second_opinion"]  # still the second read's numbers
            rows.append(row)
        doc["rows"] = rows
        doc["total_amount_rmb"] = num_or_none(request.form.get("total"))
        doc["total_cbm"] = num_or_none(request.form.get("total_cbm"))
        share = num_or_none(request.form.get("share"))
        doc["shipment_share_pct"] = share if share is not None and 0 < share <= 100 else None
        doc["edited_at"] = datetime.now().isoformat(timespec="seconds")
        save_json(p, extract.check(doc))
        return redirect(url_for("show", doc_id=doc_id))
    s = load_settings()
    missing = s.missing()
    share = doc.get("shipment_share_pct")
    results = [] if missing else landed(doc["rows"], s, doc.get("total_cbm"), share / 100 if share else None)
    return render_template_string(
        DOC_HTML, d=doc, id=doc_id, s=s, results=results, cats=s.choices(), missing=missing,
        issues=issues_of(doc, results), text=summary_text(doc, results, s) if results else "",
        note=request.args.get("note", ""), blank=range(len(doc["rows"]), len(doc["rows"]) + 2))


SETTING_FIELDS = [  # form name, settings key, label for errors, allowed range
    ("fx", "fx_zar_per_cny", "汇率", (1, 5)), ("markup", "markup", "加价倍数", (1, 5)),
    ("allin", "allin_rate_per_cbm", "包税价", (0, 1e6)), ("agent", "agent_commission_pct", "代理费 %", (0, 20)),
    ("round", "round_to_zar", "取整", (0, 100)), ("freight", "freight_per_cbm_zar", "海运", (0, 1e6)),
    ("clearing", "clearing_fixed_zar", "清关", (0, 1e7)), ("local", "local_transport_per_cbm_zar", "陆运", (0, 1e6)),
    ("vat", "vat_rate", "增值税 %", (0, 30)), ("uplift", "vat_uplift", "上浮 %", (0, 30)),
]
PERCENT = {"vat_rate", "vat_uplift"}


def parse_table(text: str, pct: bool, label: str, errors: list) -> dict:
    out = {}
    for line in (text or "").splitlines():
        line = unicodedata.normalize("NFKC", line).strip()
        if not line:
            continue
        if "=" not in line:
            errors.append(f"{label}：「{line}」少了 =")
            continue
        k, v = line.rsplit("=", 1)
        try:
            x = num(v)
        except Bad:
            x = None
        if not k.strip() or x is None or x < 0:
            errors.append(f"{label}：「{line}」读不懂")
            continue
        out[k.strip()] = x / 100 if pct else x
    return out


@app.route("/settings", methods=["GET", "POST"])
def settings():
    s = load_settings()
    raw = read_json(SETTINGS, {}) or {}
    errors = []
    if request.method == "POST":
        f = request.form
        new = {"mode": f.get("mode") if f.get("mode") in ("allin", "itemized") else "allin",
               "allin_rate_currency": f.get("allin_cur") if f.get("allin_cur") in ("CNY", "ZAR") else "CNY",
               "vat_registered": f.get("vat_registered") == "on", "fx_note": f.get("fx_note", "")[:200]}
        for form, key, label, (lo, hi) in SETTING_FIELDS:
            try:
                x = num(f.get(form))
            except Bad:
                errors.append(f"{label}：「{f.get(form)}」读不懂")
                continue
            if x is None:
                x = getattr(Settings(fx_zar_per_cny=0), key) * (100 if key in PERCENT else 1)
            if not lo <= x <= hi:
                hint = "（1 元大约 2.4–2.6 兰特）" if key == "fx_zar_per_cny" else "（比如 1.3，不是 30%）" if key == "markup" else ""
                errors.append(f"{label} {x:g} 不在 {lo:g}–{hi:g} 之间{hint}")
                continue
            new[key] = x / 100 if key in PERCENT else x
        new["allin_rates"] = parse_table(f.get("allin_rates"), False, "包税分类价", errors)
        new["duty_rates"] = parse_table(f.get("duty_rates"), True, "关税率", errors)
        if not errors:
            save_json(SETTINGS, new)
            return redirect(safe_next(request.args.get("next")))
        form = dict(f)
    else:
        form = {
            "fx": f"{s.fx_zar_per_cny:g}" if s.fx_zar_per_cny else "", "fx_note": raw.get("fx_note", ""),
            "mode": s.mode, "markup": f"{s.markup:g}", "allin": f"{s.allin_rate_per_cbm:g}" if s.allin_rate_per_cbm else "",
            "allin_cur": s.allin_rate_currency, "agent": f"{s.agent_commission_pct:g}" if s.agent_commission_pct else "",
            "round": f"{s.round_to_zar:g}", "freight": f"{s.freight_per_cbm_zar:g}" if s.freight_per_cbm_zar else "",
            "local": f"{s.local_transport_per_cbm_zar:g}" if s.local_transport_per_cbm_zar else "",
            "clearing": f"{s.clearing_fixed_zar:g}" if s.clearing_fixed_zar else "",
            "vat": f"{s.vat_rate * 100:g}", "uplift": f"{s.vat_uplift * 100:g}",
            "vat_registered": "on" if s.vat_registered else "",
            "allin_rates": "\n".join(f"{k} = {v:g}" for k, v in s.allin_rates.items()),
            "duty_rates": "\n".join(f"{k} = {v * 100:g}%" for k, v in s.duty_rates.items()),
        }
    return render_template_string(SETTINGS_HTML, f=form, errors=errors, ref=REFERENCE)


@app.get("/fx/ecb")
def fx_ecb():
    """ZAR per CNY from the ECB's free daily reference rates (only called when the owner clicks the button)."""
    try:
        root = ET.fromstring(requests.get("https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml", timeout=15).text)
        rates = {c.get("currency"): float(c.get("rate")) for c in root.iter() if c.get("currency")}
        day = next(c.get("time") for c in root.iter() if c.get("time"))
        return jsonify(ok=True, rate=round(rates["ZAR"] / rates["CNY"], 4), date=day)
    except (requests.RequestException, ET.ParseError, KeyError, StopIteration, ValueError) as e:
        return jsonify(ok=False, error=str(e)[:120])


# ---------- templates (Jinja autoescapes everything below) ----------

CSS = """
:root{--bg:#f5f4ef;--card:#fff;--ink:#1d2320;--mute:#66706a;--line:#dfe1da;--acc:#1f6f5f;--bad:#b3402e;--badbg:#fbeae6;--warnbg:#fdf6dc}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,'Segoe UI','Microsoft YaHei',sans-serif}
main{max-width:1240px;margin:0 auto;padding:24px 16px 60px}h1{font-size:24px;margin:0}h2{font-size:16px;margin:24px 0 8px}
.sub,.mute{color:var(--mute)}.mute{font-size:13px}.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}
a{color:var(--acc)}button,.btn{font:inherit;border:0;border-radius:9px;padding:10px 18px;background:var(--acc);color:#fff;cursor:pointer;text-decoration:none;display:inline-block}
.btn2{background:#5f6b63;padding:6px 12px}
input,select,textarea{font:inherit;border:1px solid var(--line);border-radius:7px;padding:6px 8px;background:#fff;width:100%}
table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid var(--line);padding:6px 6px;text-align:left;vertical-align:top}
th{font-size:12px;color:var(--mute);font-weight:500;white-space:nowrap}.num{text-align:right;font-variant-numeric:tabular-nums}
.bad{background:var(--badbg)}.badtxt{color:var(--bad)}.warn{background:var(--warnbg)}.top{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}
.scroll{overflow-x:auto}
"""

HOME_HTML = """<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Landed Cost · 落地成本</title><style>""" + CSS + """.drop{border:2px dashed var(--line);border-radius:12px;padding:28px;text-align:center;background:#fff}</style></head><body><main>
<div class=top><div><h1>落地成本 · Landed Cost</h1><div class=sub>拍一张供应商送货单 → 每件到约堡的成本 → 每包批发价。全部在这台电脑上算。</div></div><a href=/settings>设置 Settings</a></div>
{% if error %}<p class="card bad badtxt">{{error}}</p>{% endif %}
{% if missing %}<p class="card badtxt">还没填：{{missing|join('、')}}。先去 <a href=/settings>设置</a> 填好，才能算成本。 Fill these in Settings first.</p>{% endif %}
<h2>新单据 New delivery note</h2>
<form class=drop method=post action=/read enctype=multipart/form-data><p>送货单 / 装箱单照片 · photo of a 送货单 or packing list</p>
<input type=file name=photo accept="image/*" required style="max-width:340px"><p><button>读取 Read it</button></p>
<p class=mute>本地 Gemma 4 读表，大约 10–30 秒。Read by Gemma 4 on this computer.</p></form>
<h2>最近 Recent</h2><div class=card>{% for d in docs %}<div><a href="/doc/{{d.id}}">{{d.title}}</a> <span class=mute>{{d.date}} · {{d.rows}} 行{% if d.problems %} · <span class=badtxt>{{d.problems}} 处要核对</span>{% endif %}</span></div>{% else %}<span class=mute>还没有单据。</span>{% endfor %}</div>
</main></body></html>"""

DOC_HTML = """<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>{{d.doc_no or 'Delivery note'}} · Landed Cost</title><style>""" + CSS + """
.two{display:grid;grid-template-columns:minmax(0,4fr) minmax(0,8fr);gap:16px}.photo img{width:100%;border-radius:8px;border:1px solid var(--line)}
.edit td{padding:3px}.edit input,.edit select{padding:4px 5px;font-size:13px;min-width:4.5em}.edit input[type=checkbox]{min-width:0;width:auto}
.edit input[name^=name]{min-width:10em}.res td{font-size:14px}.res td:first-child{min-width:9em}.big{font-weight:700}.bd{font-size:12px;color:var(--mute)}
@media(max-width:900px){.two{grid-template-columns:minmax(0,1fr)}}</style></head><body><main>
<div class=top><div><a href=/>← 全部</a><h1>{{d.supplier or '未命名单据'}} <span class=mute>{{d.doc_no}} · {{d.date}}</span></h1></div><a href="/settings?next=/doc/{{id}}">设置 Settings</a></div>
{% if note == 'model_down' %}<p class="card bad badtxt">本地模型没响应，类别没改。The local model didn't answer; categories unchanged.</p>{% endif %}
{% for n in d.notes or [] %}<p class="card warn" style="margin-top:12px">{{n}}</p>{% endfor %}
{% if d.problems %}<div class="card bad" style="margin-top:12px"><b class=badtxt>先核对这些 Check these first:</b><ul>{% for p in d.problems %}<li>{{p}}</li>{% endfor %}</ul></div>{% endif %}
<div class=two style="margin-top:12px"><div class=photo><img src="/doc/{{id}}/photo" alt="delivery note photo"><div class=mute>读取：{{d.model}} · {{d.read_at}}</div></div>
<div class=card><form method=post><b>读到的内容（可以直接改，空行可以补漏读的行）What was read — edit anything, use the blank rows for missed lines</b><input type=hidden name=n value="{{d.rows|length + 2}}">
<div class=scroll><table class=edit><tr><th>款号</th><th>品名</th><th>件/包</th><th>包数</th><th>数量</th><th>单位</th><th>单价¥</th><th>金额¥</th><th>箱规cm</th><th>体积m³</th>{% if cats %}<th>类别</th>{% endif %}<th>删</th></tr>
{% for r in d.rows %}{% set i = loop.index0 %}<tr {% if r.problems %}class=bad title="{{r.problems|join('; ')}}"{% elif r.second_opinion %}class=warn title="第二个模型 {{r.second_opinion}} 重读的数 · numbers from a second read"{% endif %}>
<td><input name=sku{{i}} value="{{r.sku}}"></td><td><input name=name{{i}} value="{{r.name}}"><input type=hidden name=spec{{i}} value="{{r.spec}}"></td>
{% for k, form in [('pcs_per_bale','per'),('bales','bales'),('qty_pcs','qty')] %}<td><input name={{form}}{{i}} value="{{r[k] if r[k] is not none else ''}}"></td>{% endfor %}
<td><input name=unit{{i}} value="{{r.unit or ''}}" placeholder="件"></td>
{% for k, form in [('unit_price_rmb','price'),('amount_rmb','amount')] %}<td><input name={{form}}{{i}} value="{{r[k] if r[k] is not none else ''}}"></td>{% endfor %}
<td><input name=carton{{i}} value="{{r.carton_cm}}"></td><td><input name=cbm{{i}} value="{{r.cbm if r.cbm is not none else ''}}"></td>
{% if cats %}<td><select name=cat{{i}}><option value="">未分类</option>{% for c in cats %}<option {% if r.category==c %}selected{% endif %}>{{c}}</option>{% endfor %}</select></td>{% else %}<input type=hidden name=cat{{i}} value="{{r.category or ''}}">{% endif %}
<td><input type=checkbox name=del{{i}}></td></tr>{% endfor %}
{% for i in blank %}<tr><td><input name=sku{{i}} placeholder="+ 补一行"></td><td><input name=name{{i}}><input type=hidden name=spec{{i}} value=""></td>
<td><input name=per{{i}}></td><td><input name=bales{{i}}></td><td><input name=qty{{i}}></td><td><input name=unit{{i}} placeholder="件"></td><td><input name=price{{i}}></td><td><input name=amount{{i}}></td>
<td><input name=carton{{i}}></td><td><input name=cbm{{i}}></td>{% if cats %}<td><select name=cat{{i}}><option value="">未分类</option>{% for c in cats %}<option>{{c}}</option>{% endfor %}</select></td>{% else %}<input type=hidden name=cat{{i}} value="">{% endif %}<td></td></tr>{% endfor %}
</table></div>
<div style="display:flex;gap:10px;margin-top:8px;align-items:end;flex-wrap:wrap"><label style="flex:1;min-width:8em">合计¥ total<input name=total value="{{d.total_amount_rmb if d.total_amount_rmb is not none else ''}}"></label>
<label style="flex:1;min-width:8em">总体积 方 CBM<input name=total_cbm value="{{d.total_cbm if d.total_cbm is not none else ''}}"></label>
{% if s.mode == 'itemized' and s.clearing_fixed_zar %}<label style="flex:1;min-width:10em">本单占整票货 %（分摊清关费）<input name=share value="{{d.shipment_share_pct if d.shipment_share_pct else ''}}" placeholder="例如 25"></label>{% else %}<input type=hidden name=share value="{{d.shipment_share_pct or ''}}">{% endif %}
<button>保存并重算 Save</button></div>
<div class=mute style="margin-top:6px">各行加起来 ¥{{'{:,.2f}'.format(d.rows_total_rmb or 0)}}{% if d.rows|selectattr('second_opinion')|list %} · <span class=warn style="padding:0 6px">黄色行</span> 第一遍没对上，用第二个本地模型重读后通过了核对，请看一眼。{% endif %}</div></form>
{% if cats %}<form method=post action="/doc/{{id}}/classify" style="margin-top:6px"><button class=btn2>按我的类别重新分类 Re-sort categories (Gemma)</button></form>{% endif %}</div></div>
<h2>落地成本 Landed cost</h2><div class=card>{% if missing %}<span class=badtxt>还没填：{{missing|join('、')}}。请先去 <a href="/settings?next=/doc/{{id}}">设置</a>。</span>{% else %}
<div class=scroll><table class=res><tr><th>款号 品名</th><th class=num>进价 R</th><th class=num>运费/税/费 R</th><th class=num>落地成本 R</th><th class=num>成本 R/包</th><th class=num>建议批发 R</th><th class=num>建议 R/包</th><th class=num>毛利</th></tr>
{% for r in results %}{% if r.error %}<tr class=bad><td colspan=8><b>{{r.sku}} {{r.name}}</b>：{{r.error}}</td></tr>{% else %}{% set u = r.unit_label or '件' %}{% set b = r.breakdown_per_piece %}
<tr class="{% if r.problems %}bad{% elif r.warnings or r.second_opinion or r.cbm_estimated %}warn{% endif %}">
<td>{{r.sku}} {{r.name}}<div class=bd>每{{u}}：货价 {{'%.2f'|format(b.goods)}}{% if b.agent %} · 代理费 {{'%.2f'|format(b.agent)}}{% endif %}{% if 'shipping_allin' in b %} · 包税运费 {{'%.2f'|format(b.shipping_allin)}}{% endif %}{% if 'freight' in b %} · 运费 {{'%.2f'|format(b.freight)}}{% endif %}{% if 'clearing' in b %} · 清关 {{'%.2f'|format(b.clearing)}}{% endif %}{% if 'duty' in b %} · 关税 {{'%.2f'|format(b.duty)}}{% endif %}{% if 'vat' in b %} · 增值税 {{'%.2f'|format(b.vat)}}{% endif %}{% if r.vat_cashflow_per_piece %} · 进口增值税 {{'%.2f'|format(r.vat_cashflow_per_piece)}} 可抵扣{% endif %}{% if r.cbm_estimated %} · 体积按金额估{% endif %}</div>
{% for w in r.warnings %}<div class=badtxt>{{w}}</div>{% endfor %}</td>
<td class=num>{{'%.2f'|format(b.goods)}}/{{u}}</td><td class=num>{{'%.2f'|format(r.landed_per_piece - b.goods)}}</td>
<td class="num big">{{'%.2f'|format(r.landed_per_piece)}}/{{u}}</td><td class=num>{{'{:,.0f}'.format(r.landed_per_bale) if r.landed_per_bale else ''}}</td>
<td class="num big">{{'%.2f'|format(r.suggested_per_piece)}}/{{u}}</td><td class=num>{{'{:,.0f}'.format(r.suggested_per_bale) if r.suggested_per_bale else ''}}</td><td class=num>{{r.margin_pct}}%</td></tr>{% endif %}{% endfor %}</table></div>
<p class=mute>汇率 {{s.fx_zar_per_cny}} · {% if s.mode=='allin' %}包税 {% if s.allin_rates %}按类别{% else %}{{s.allin_rate_per_cbm}} {{s.allin_rate_currency}}/方{% endif %}{% else %}海运 R{{s.freight_per_cbm_zar}}/方 · 关税按类别{% if s.vat_registered %} · 进口增值税可抵扣，不算成本{% endif %}{% endif %} · 加价 ×{{s.markup}}</p>
<h2 style="margin-top:12px">发到 WhatsApp / 微信（内部用，含成本）</h2>{% if issues %}<p class=badtxt>⚠️ 还有 {{issues|length}} 处没核对，文字里已经标出来了。</p>{% endif %}
<textarea id=t rows=9 readonly>{{text}}</textarea><p><button type=button onclick="navigator.clipboard.writeText(document.getElementById('t').value);this.textContent='已复制 Copied'">复制 Copy</button></p>{% endif %}</div>
<p class=mute>照片、价格和设置都只存在这台电脑的 data/ 文件夹里。Photos, prices and settings stay in data/ on this computer.</p>
</main></body></html>"""

SETTINGS_HTML = """<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Settings · Landed Cost</title><style>""" + CSS + """label{display:block;margin:10px 0}.grid{display:grid;grid-template-columns:1fr 1fr;gap:0 18px}@media(max-width:700px){.grid{grid-template-columns:1fr}}</style></head><body><main>
<a href=/>← 返回</a><h1>设置 Settings</h1><p class=sub>按你自己的账单填。税率和运价都不会自动预填。Fill these from your own bills; nothing is pre-filled.</p>
{% if errors %}<div class="card bad"><b class=badtxt>没保存，请改一下 Not saved:</b><ul>{% for e in errors %}<li>{{e}}</li>{% endfor %}</ul></div>{% endif %}
<form method=post class=card><div class=grid>
<label>汇率：1 元 = ? 兰特 (ZAR per CNY)<input name=fx id=fx value="{{f.fx}}" required>
<button type=button id=ecb class=btn2 style="margin-top:6px" onclick="getEcb()">用欧洲央行参考汇率 Use ECB reference rate</button></label>
<label>汇率来源/日期 note<input name=fx_note id=fx_note value="{{f.fx_note}}" placeholder="e.g. bank rate 2026-10-13"></label>
<label>怎么运 Shipping<select name=mode><option value=allin {% if f.mode=='allin' %}selected{% endif %}>货代包税，按方收 (all-in per CBM)</option><option value=itemized {% if f.mode=='itemized' %}selected{% endif %}>自己清关，分项算 (itemized)</option></select></label>
<label>加价倍数 markup（批发价 = 成本 × ?，比如 1.3）<input name=markup value="{{f.markup}}"></label>
<label>包税默认价 / 方 all-in rate per CBM<input name=allin value="{{f.allin}}"></label>
<label>包税价的币种 currency<select name=allin_cur><option value=CNY {% if f.allin_cur!='ZAR' %}selected{% endif %}>¥ 人民币/方 CNY</option><option value=ZAR {% if f.allin_cur=='ZAR' %}selected{% endif %}>R 兰特/方 ZAR</option></select></label>
<label>代理/付款手续费 % agent fee on goods<input name=agent value="{{f.agent}}"></label>
<label>价格取整到 R round price up to<input name=round value="{{f.round}}"></label>
</div><label>包税分类价（可选，币种同上）all-in rate per category — one per line, e.g. <code>服装 clothing = 3450</code>。没列出的类别和「{{default}}」用上面的默认价。
<textarea name=allin_rates rows=4>{{f.allin_rates}}</textarea></label>
<p class=mute>例：{{ref.allin_example.note}} <a href="{{ref.allin_example.source}}">来源</a></p>
<h2>自己清关时 (itemized only)</h2><div class=grid>
<label>海运 R/方 freight per CBM<input name=freight value="{{f.freight}}"></label>
<label>德班→约堡 R/方 local transport per CBM<input name=local value="{{f.local}}"></label>
<label>清关+港杂 R/票（整票货，单据页按比例分摊）clearing per whole shipment<input name=clearing value="{{f.clearing}}"></label>
<label>进口增值税 % VAT<input name=vat value="{{f.vat}}"></label>
<label>增值税基数上浮 % VAT uplift on FOB<input name=uplift value="{{f.uplift}}"></label>
<label style="display:flex;gap:8px;align-items:center"><input type=checkbox name=vat_registered style="width:auto" {% if f.vat_registered %}checked{% endif %}> 已注册增值税（进口增值税可抵扣）VAT-registered</label>
</div><label>关税率，每行一个类别 duty rate per category (one per line, e.g. <code>棉袜 cotton socks = 45%</code>)<textarea name=duty_rates id=duty rows=8>{{f.duty_rates}}</textarea></label>
<button type=button class=btn2 onclick="loadSars()">填入 SARS 参考关税率（{{ref.edition}} 版）Load SARS reference rates</button>
<p class=mute>{{ref.title}} · <a href="{{ref.source}}">SARS</a> · {{ref.anti_dumping_check}}<br>参考而已：税率取决于具体归类（棉还是化纤、针织还是梭织、橡胶还是 PVC），请和清关代理核对。{{ref.caveat}}<br>VAT: {{ref.vat.base}} × 15% (<a href="{{ref.vat.source}}">SARS</a>) · Valuation: {{ref.valuation.basis}} (<a href="{{ref.valuation.source}}">SARS</a>)</p>
<p><button>保存 Save</button></p></form>
<script>
const SARS = {{ ref.rates | tojson }};
function loadSars(){const t=document.getElementById('duty');const have=t.value.trim();
 const lines=SARS.map(r=>`${r.category} = ${+(r.rate*100).toFixed(2)}%`);t.value=(have?have+'\\n':'')+lines.join('\\n');}
async function getEcb(){const b=document.getElementById('ecb');b.textContent='…';
 try{const j=await (await fetch('/fx/ecb')).json();if(!j.ok)throw new Error(j.error);
  document.getElementById('fx').value=j.rate;document.getElementById('fx_note').value=`ECB reference cross rate ${j.date} (SARS customs rate on the shipment date can differ by ~1–2%)`;b.textContent='✓ '+j.date;}
 catch(e){b.textContent='没取到 failed: '+e.message;}}
</script></main></body></html>""".replace("{{default}}", DEFAULT_CATEGORY)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=PORT, debug=False)
