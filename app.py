"""Landed Cost: photo of a supplier delivery note -> landed cost per piece in rand -> wholesale price per bale.

Run:  python app.py      then open http://127.0.0.1:5060
Everything (photos, prices, settings) stays in ./data on this computer. The vision model runs in local Ollama.
"""
import json
import os
import uuid
import xml.etree.ElementTree as ET
from dataclasses import fields
from datetime import datetime
from pathlib import Path

import requests
from flask import Flask, abort, jsonify, redirect, render_template_string, request, send_file, url_for

from lc import classify, extract
from lc.cost import Settings, landed

DATA = Path(os.environ.get("LC_DATA", Path(__file__).resolve().parent / "data"))
DOCS = DATA / "docs"
SETTINGS = DATA / "settings.json"
REFERENCE = json.loads((Path(__file__).resolve().parent / "lc" / "reference" / "sars_duty_2026-08-28.json").read_text(encoding="utf-8"))
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

DEFAULTS = {"fx_zar_per_cny": 0.0, "mode": "allin", "duty_rates": {}}


def load_settings() -> Settings:
    raw = json.loads(SETTINGS.read_text(encoding="utf-8")) if SETTINGS.exists() else DEFAULTS
    known = {f.name for f in fields(Settings)}
    return Settings(**{k: v for k, v in {**DEFAULTS, **raw}.items() if k in known})


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def doc_path(doc_id: str) -> Path:
    if not doc_id.isalnum():
        abort(404)
    return DOCS / doc_id


def num(v):
    v = (v or "").strip().replace(",", "").replace("¥", "")
    try:
        return float(v) if v else None
    except ValueError:
        return None


def summary_text(doc: dict, results: list[dict], s: Settings) -> str:
    how = (f"包税 {s.allin_rate_per_cbm:g} {s.allin_rate_currency}/方" if s.mode == "allin"
           else f"运费 R{s.freight_per_cbm_zar:g}/方 + 清关 R{s.clearing_fixed_zar:g} + 关税")
    head = f"📦 {doc.get('supplier', '')} {doc.get('doc_no', '')} {doc.get('date', '')}\n汇率 {s.fx_zar_per_cny:g} · {how} · 加价 ×{s.markup:g}"
    lines = [head]
    for r in results:
        if r.get("error"):
            continue
        bale = f" · R{r['suggested_per_bale']:,.0f}/包({r['pcs_per_bale']:g}件)" if r.get("suggested_per_bale") else ""
        lines.append(f"{r.get('sku', '')} {r.get('name', '')}: 成本 R{r['landed_per_piece']:.2f}/件 → 建议批发 R{r['suggested_per_piece']:.2f}/件{bale}")
    return "\n".join(lines)


@app.get("/")
def home():
    docs = []
    for d in sorted(DOCS.glob("*/doc.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:12]:
        j = json.loads(d.read_text(encoding="utf-8"))
        docs.append({"id": d.parent.name, "title": f"{j.get('supplier', '')} {j.get('doc_no', '')}", "date": j.get("date", ""),
                     "rows": len(j.get("rows", [])), "problems": len(j.get("problems", []))})
    s = load_settings()
    return render_template_string(HOME_HTML, docs=docs, s=s, ready=s.fx_zar_per_cny > 0)


@app.post("/read")
def read():
    f = request.files.get("photo")
    if not f or not f.filename:
        return redirect(url_for("home"))
    doc_id = uuid.uuid4().hex[:10]
    folder = doc_path(doc_id)
    folder.mkdir(parents=True)
    photo = folder / ("photo" + (Path(f.filename).suffix.lower() or ".jpg"))
    f.save(photo)
    doc = extract.read_checked(photo)
    doc.update(photo=photo.name, read_at=datetime.now().isoformat(timespec="seconds"))
    sort_categories(doc, load_settings())
    save_json(folder / "doc.json", doc)
    return redirect(url_for("show", doc_id=doc_id))


def sort_categories(doc: dict, s: Settings) -> None:
    cats = s.categories()
    if cats:
        for r, c in zip(doc["rows"], classify.categories([f"{r['name']} {r.get('spec', '')}" for r in doc["rows"]], cats)):
            r["category"] = c


@app.post("/doc/<doc_id>/classify")
def reclassify(doc_id):
    p = doc_path(doc_id) / "doc.json"
    doc = json.loads(p.read_text(encoding="utf-8"))
    sort_categories(doc, load_settings())
    save_json(p, doc)
    return redirect(url_for("show", doc_id=doc_id))


@app.get("/fx/ecb")
def fx_ecb():
    """ZAR per CNY from the ECB's free daily reference rates (only called when the owner clicks the button)."""
    try:
        xml = requests.get("https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml", timeout=15).text
        root = ET.fromstring(xml)
        rates = {c.get("currency"): float(c.get("rate")) for c in root.iter() if c.get("currency")}
        day = next(c.get("time") for c in root.iter() if c.get("time"))
        return jsonify(ok=True, rate=round(rates["ZAR"] / rates["CNY"], 4), date=day)
    except (requests.RequestException, ET.ParseError, KeyError, StopIteration) as e:
        return jsonify(ok=False, error=str(e)[:120])


@app.get("/doc/<doc_id>/photo")
def photo(doc_id):
    doc = json.loads((doc_path(doc_id) / "doc.json").read_text(encoding="utf-8"))
    return send_file(doc_path(doc_id) / doc["photo"])


@app.route("/doc/<doc_id>", methods=["GET", "POST"])
def show(doc_id):
    p = doc_path(doc_id) / "doc.json"
    if not p.exists():
        abort(404)
    doc = json.loads(p.read_text(encoding="utf-8"))
    if request.method == "POST":
        rows = []
        for i in range(int(request.form["n"])):
            if request.form.get(f"del{i}"):
                continue
            g = lambda k: request.form.get(f"{k}{i}", "")  # noqa: E731
            rows.append({"sku": g("sku"), "name": g("name"), "spec": g("spec"), "pcs_per_bale": num(g("per")),
                         "bales": num(g("bales")), "qty_pcs": num(g("qty")), "unit_price_rmb": num(g("price")),
                         "amount_rmb": num(g("amount")), "carton_cm": g("carton"), "category": g("cat")})
        doc["rows"] = rows
        doc["total_amount_rmb"] = num(request.form.get("total"))
        doc["total_cbm"] = num(request.form.get("total_cbm"))
        doc["edited_at"] = datetime.now().isoformat(timespec="seconds")
        save_json(p, extract.check(doc))
        return redirect(url_for("show", doc_id=doc_id))
    s = load_settings()
    results = landed(doc["rows"], s, doc.get("total_cbm")) if s.fx_zar_per_cny > 0 else []
    return render_template_string(DOC_HTML, d=doc, id=doc_id, s=s, results=results,
                                  cats=s.categories(), text=summary_text(doc, results, s) if results else "")


@app.route("/settings", methods=["GET", "POST"])
def settings():
    s = load_settings()
    if request.method == "POST":
        f = request.form
        def table(text: str, pct: bool) -> dict:
            out = {}
            for line in (text or "").splitlines():
                if "=" in line:
                    k, v = line.rsplit("=", 1)
                    v = num(v.strip().rstrip("%"))
                    if k.strip() and v is not None:
                        out[k.strip()] = v / 100 if pct else v
            return out

        rates = table(f.get("duty_rates"), pct=True)
        new = {
            "fx_zar_per_cny": num(f.get("fx")) or 0.0, "mode": f.get("mode", "allin"),
            "allin_rate_per_cbm": num(f.get("allin")) or 0.0, "allin_rate_currency": f.get("allin_cur", "ZAR"),
            "allin_rates": table(f.get("allin_rates"), pct=False),
            "freight_per_cbm_zar": num(f.get("freight")) or 0.0, "clearing_fixed_zar": num(f.get("clearing")) or 0.0,
            "local_transport_per_cbm_zar": num(f.get("local")) or 0.0, "duty_rates": rates,
            "vat_rate": (num(f.get("vat")) or 15) / 100, "vat_uplift": (num(f.get("uplift")) or 10) / 100,
            "vat_registered": f.get("vat_registered") == "on", "agent_commission_pct": num(f.get("agent")) or 0.0,
            "markup": num(f.get("markup")) or 1.3, "round_to_zar": num(f.get("round")) or 0.5,
            "fx_note": f.get("fx_note", ""),
        }
        save_json(SETTINGS, new)
        return redirect(request.args.get("next") or url_for("home"))
    raw = json.loads(SETTINGS.read_text(encoding="utf-8")) if SETTINGS.exists() else {}
    return render_template_string(SETTINGS_HTML, s=s, fx_note=raw.get("fx_note", ""), ref=REFERENCE,
                                  rates="\n".join(f"{k} = {v * 100:g}%" for k, v in s.duty_rates.items()),
                                  allin_rates="\n".join(f"{k} = {v:g}" for k, v in s.allin_rates.items()))


CSS = """
:root{--bg:#f5f4ef;--card:#fff;--ink:#1d2320;--mute:#66706a;--line:#dfe1da;--acc:#1f6f5f;--bad:#b3402e;--badbg:#fbeae6}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,'Segoe UI','Microsoft YaHei',sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 60px}h1{font-size:24px;margin:0}h2{font-size:16px;margin:24px 0 8px}
.sub,.mute{color:var(--mute)}.mute{font-size:13px}.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}
a{color:var(--acc)}button,.btn{font:inherit;border:0;border-radius:9px;padding:10px 18px;background:var(--acc);color:#fff;cursor:pointer;text-decoration:none;display:inline-block}
input,select,textarea{font:inherit;border:1px solid var(--line);border-radius:7px;padding:6px 8px;background:#fff;width:100%}
table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid var(--line);padding:6px 6px;text-align:left;vertical-align:top}
th{font-size:12px;color:var(--mute);font-weight:500}.num{text-align:right;font-variant-numeric:tabular-nums}
.bad{background:var(--badbg)}.second{background:#fdf6dc}.badtxt{color:var(--bad)}.top{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}
"""

HOME_HTML = """<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Landed Cost · 落地成本</title><style>""" + CSS + """.drop{border:2px dashed var(--line);border-radius:12px;padding:28px;text-align:center;background:#fff}</style></head><body><main>
<div class=top><div><h1>落地成本 · Landed Cost</h1><div class=sub>拍一张供应商送货单 → 每件到约堡的成本 → 每包批发价。全部在这台电脑上算。</div></div><a href=/settings>设置 Settings</a></div>
{% if not ready %}<p class="card badtxt">先去 <a href=/settings>设置</a> 填汇率和运费，才能算成本。 Set the exchange rate and shipping first.</p>{% endif %}
<h2>新单据 New delivery note</h2>
<form class=drop method=post action=/read enctype=multipart/form-data><p>送货单 / 装箱单照片 · photo of a 送货单 or packing list</p>
<input type=file name=photo accept="image/*" required style="max-width:340px"><p><button>读取 Read it</button></p>
<p class=mute>本地 Gemma 4 读表，大约 10–30 秒。Read by Gemma 4 on this computer.</p></form>
<h2>最近 Recent</h2><div class=card>{% for d in docs %}<div><a href="/doc/{{d.id}}">{{d.title or d.id}}</a> <span class=mute>{{d.date}} · {{d.rows}} 行{% if d.problems %} · <span class=badtxt>{{d.problems}} 处要核对</span>{% endif %}</span></div>{% else %}<span class=mute>还没有单据。</span>{% endfor %}</div>
</main></body></html>"""

DOC_HTML = """<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>{{d.doc_no or 'Delivery note'}} · Landed Cost</title><style>""" + CSS + """
.two{display:grid;grid-template-columns:minmax(0,5fr) minmax(0,7fr);gap:16px}.photo img{width:100%;border-radius:8px;border:1px solid var(--line)}
.edit td{padding:3px}.edit input{padding:4px 5px;font-size:13px}.res td{font-size:14px}.big{font-weight:700}
@media(max-width:900px){.two{grid-template-columns:1fr}.edit{display:block;overflow-x:auto}}</style></head><body><main>
<div class=top><div><a href=/>← 全部</a><h1>{{d.supplier}} <span class=mute>{{d.doc_no}} · {{d.date}}</span></h1></div><a href="/settings?next=/doc/{{id}}">设置 Settings</a></div>
{% if d.problems %}<div class="card bad" style="margin-top:12px"><b class=badtxt>先核对这些 Check these first:</b><ul>{% for p in d.problems %}<li>{{p}}</li>{% endfor %}</ul></div>{% endif %}
<div class=two style="margin-top:12px"><div class=photo><img src="/doc/{{id}}/photo" alt="delivery note photo"><div class=mute>读取：{{d.model}} · {{d.read_at}}</div></div>
<form method=post class=card><b>读到的内容（可以直接改）What was read — edit anything</b><input type=hidden name=n value="{{d.rows|length}}">
<table class=edit><tr><th>款号</th><th>品名</th><th>件/包</th><th>包数</th><th>数量</th><th>单价¥</th><th>金额¥</th><th>箱规cm</th>{% if cats %}<th>类别</th>{% endif %}<th>删</th></tr>
{% for r in d.rows %}{% set i = loop.index0 %}<tr {% if r.problems %}class=bad title="{{r.problems|join('; ')}}"{% elif r.second_opinion %}class=second title="第一遍没对上，这行用第二个模型 {{r.second_opinion}} 读的数，请看一眼 · numbers from a second read"{% endif %}>
<td><input name=sku{{i}} value="{{r.sku}}"></td><td><input name=name{{i}} value="{{r.name}}"><input type=hidden name=spec{{i}} value="{{r.spec}}"></td>
<td><input name=per{{i}} value="{{r.pcs_per_bale if r.pcs_per_bale is not none else ''}}"></td><td><input name=bales{{i}} value="{{r.bales if r.bales is not none else ''}}"></td>
<td><input name=qty{{i}} value="{{r.qty_pcs if r.qty_pcs is not none else ''}}"></td><td><input name=price{{i}} value="{{r.unit_price_rmb if r.unit_price_rmb is not none else ''}}"></td>
<td><input name=amount{{i}} value="{{r.amount_rmb if r.amount_rmb is not none else ''}}"></td><td><input name=carton{{i}} value="{{r.carton_cm}}"></td>
{% if cats %}<td><select name=cat{{i}}><option value=""></option>{% for c in cats %}<option {% if r.category==c %}selected{% endif %}>{{c}}</option>{% endfor %}</select></td>{% else %}<input type=hidden name=cat{{i}} value="{{r.category or ''}}">{% endif %}
<td><input type=checkbox name=del{{i}} style="width:auto"></td></tr>{% endfor %}</table>
<div style="display:flex;gap:10px;margin-top:8px;align-items:end"><label style="flex:1">合计¥ total<input name=total value="{{d.total_amount_rmb if d.total_amount_rmb is not none else ''}}"></label>
<label style="flex:1">总体积 方 CBM<input name=total_cbm value="{{d.total_cbm if d.total_cbm is not none else ''}}"></label><button>保存并重算 Save</button></div>
{% if d.rows|selectattr('second_opinion')|list %}<div class=mute style="margin-top:6px"><span class=second style="padding:0 6px">黄色行</span> 第一遍没对上，用第二个本地模型重读后通过了核对，请看一眼。Yellow rows were re-read by a second local model and now add up.</div>{% endif %}
<div class=mute style="margin-top:6px">各行加起来 ¥{{'{:,.2f}'.format(d.rows_total_rmb or 0)}}</div></form>
{% if cats %}<form method=post action="/doc/{{id}}/classify" style="margin-top:6px"><button style="background:#5f6b63;padding:6px 12px">按我的类别重新分类 Re-sort categories (Gemma)</button></form>{% endif %}</div>
<h2>落地成本 Landed cost</h2><div class=card>{% if not results %}<span class=badtxt>先在 <a href="/settings?next=/doc/{{id}}">设置</a> 填汇率和运费。</span>{% else %}
<table class=res><tr><th>款号 品名</th><th class=num>进价 R/件</th><th class=num>运费+税 R/件</th><th class=num>落地成本 R/件</th><th class=num>R/包</th><th class=num>建议批发 R/件</th><th class=num>建议 R/包</th><th class=num>毛利</th></tr>
{% for r in results %}{% if r.error %}<tr class=bad><td colspan=8>{{r.sku}} {{r.name}}: {{r.error}}</td></tr>{% else %}<tr {% if r.problems or r.warning %}class=bad{% endif %}>
<td>{{r.sku}} {{r.name}}{% if r.cbm_estimated %} <span class=mute>(体积按金额估)</span>{% endif %}{% if r.warning %}<div class=badtxt>{{r.warning}}</div>{% endif %}</td>
<td class=num>{{'%.2f'|format(r.breakdown_per_piece.goods)}}</td><td class=num title="{{r.breakdown_per_piece}}">{{'%.2f'|format(r.landed_per_piece - r.breakdown_per_piece.goods)}}</td>
<td class="num big">{{'%.2f'|format(r.landed_per_piece)}}</td><td class=num>{{'{:,.0f}'.format(r.landed_per_bale) if r.landed_per_bale else ''}}</td>
<td class="num big">{{'%.2f'|format(r.suggested_per_piece)}}</td><td class=num>{{'{:,.0f}'.format(r.suggested_per_bale) if r.suggested_per_bale else ''}}</td><td class=num>{{r.margin_pct}}%</td></tr>{% endif %}{% endfor %}</table>
<p class=mute>汇率 {{s.fx_zar_per_cny}} · {% if s.mode=='allin' %}包税 {{s.allin_rate_per_cbm}} {{s.allin_rate_currency}}/方{% else %}运费 R{{s.freight_per_cbm_zar}}/方 · 清关 R{{s.clearing_fixed_zar}} · 关税按类别{% if s.vat_registered %} · 进口增值税可抵扣，不算成本{% endif %}{% endif %} · 加价 ×{{s.markup}}</p>
<h2 style="margin-top:12px">发到 WhatsApp / 微信</h2><textarea id=t rows=8 readonly>{{text}}</textarea><p><button onclick="navigator.clipboard.writeText(document.getElementById('t').value);this.textContent='已复制 Copied'">复制 Copy</button></p>{% endif %}</div>
<p class=mute>照片、价格和设置都只存在这台电脑的 data/ 文件夹里。Photos, prices and settings stay in data/ on this computer.</p>
</main></body></html>"""

SETTINGS_HTML = """<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Settings · Landed Cost</title><style>""" + CSS + """label{display:block;margin:10px 0}.grid{display:grid;grid-template-columns:1fr 1fr;gap:0 18px}@media(max-width:700px){.grid{grid-template-columns:1fr}}</style></head><body><main>
<a href=/>← 返回</a><h1>设置 Settings</h1><p class=sub>按你自己的账单填。这里不预设任何税率或运价。Fill these from your own bills; nothing is pre-filled.</p>
<form method=post class=card><div class=grid>
<label>汇率：1 元 = ? 兰特 (ZAR per CNY)<input name=fx id=fx value="{{s.fx_zar_per_cny or ''}}" required>
<button type=button id=ecb style="margin-top:6px;padding:5px 10px;background:#5f6b63" onclick="getEcb()">用欧洲央行参考汇率 Use ECB reference rate</button></label>
<label>汇率来源/日期 note<input name=fx_note id=fx_note value="{{fx_note}}" placeholder="e.g. bank rate 2026-10-13"></label>
<label>怎么运 Shipping<select name=mode><option value=allin {% if s.mode=='allin' %}selected{% endif %}>货代包税，按方收 (all-in per CBM)</option><option value=itemized {% if s.mode=='itemized' %}selected{% endif %}>自己清关，分项算 (itemized)</option></select></label>
<label>加价倍数 markup (批发价 = 成本 × ?)<input name=markup value="{{s.markup}}"></label>
<label>包税价 / 方 all-in rate per CBM<input name=allin value="{{s.allin_rate_per_cbm or ''}}"></label>
<label>币种 currency<select name=allin_cur><option {% if s.allin_rate_currency=='ZAR' %}selected{% endif %}>ZAR</option><option {% if s.allin_rate_currency=='CNY' %}selected{% endif %}>CNY</option></select></label>
<label>代理/付款手续费 % agent fee on goods<input name=agent value="{{s.agent_commission_pct or ''}}"></label>
<label>价格取整到 R round price up to<input name=round value="{{s.round_to_zar}}"></label>
</div><label>包税分类价（可选）all-in rate per category, optional — one per line, e.g. <code>服装 clothing = 3450</code>. 没列出的类别用上面的默认价。
<textarea name=allin_rates rows=4>{{allin_rates}}</textarea></label>
<p class=mute>例：{{ref.allin_example.note}} <a href="{{ref.allin_example.source}}">来源</a></p>
<h2>自己清关时 (itemized only)</h2><div class=grid>
<label>海运 R/方 freight per CBM<input name=freight value="{{s.freight_per_cbm_zar or ''}}"></label>
<label>德班→约堡 R/方 local transport per CBM<input name=local value="{{s.local_transport_per_cbm_zar or ''}}"></label>
<label>清关+港杂 R/票 clearing per shipment<input name=clearing value="{{s.clearing_fixed_zar or ''}}"></label>
<label>进口增值税 % VAT<input name=vat value="{{s.vat_rate * 100}}"></label>
<label>增值税基数上浮 % VAT uplift on FOB<input name=uplift value="{{s.vat_uplift * 100}}"></label>
<label style="display:flex;gap:8px;align-items:center"><input type=checkbox name=vat_registered style="width:auto" {% if s.vat_registered %}checked{% endif %}> 已注册增值税（进口增值税可抵扣）VAT-registered</label>
</div><label>关税率，每行一个类别 duty rate per category (one per line, e.g. <code>clothing = 45%</code>)<textarea name=duty_rates id=duty rows=8>{{rates}}</textarea></label>
<button type=button style="padding:5px 10px;background:#5f6b63" onclick="loadSars()">填入 SARS 参考关税率（{{ref.edition}} 版）Load SARS reference rates</button>
<p class=mute>{{ref.title}} · <a href="{{ref.source}}">SARS</a> · {{ref.anti_dumping_check}}<br>{{ref.caveat}}<br>VAT: {{ref.vat.base}} × 15% (<a href="{{ref.vat.source}}">SARS</a>) · Valuation: {{ref.valuation.basis}} (<a href="{{ref.valuation.source}}">SARS</a>)</p>
<p><button>保存 Save</button></p></form>
<script>
const SARS = {{ ref.rates | tojson }};
function loadSars(){const t=document.getElementById('duty');const have=t.value.trim();
 const lines=SARS.map(r=>`${r.category} = ${+(r.rate*100).toFixed(2)}%`);t.value=(have?have+'\\n':'')+lines.join('\\n');}
async function getEcb(){const b=document.getElementById('ecb');b.textContent='…';
 try{const j=await (await fetch('/fx/ecb')).json();if(!j.ok)throw new Error(j.error);
  document.getElementById('fx').value=j.rate;document.getElementById('fx_note').value=`ECB reference cross rate ${j.date} (SARS customs rate on the shipment date can differ by ~1–2%)`;b.textContent='✓ '+j.date;}
 catch(e){b.textContent='没取到 failed: '+e.message;}}
</script></main></body></html>"""

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5060, debug=False)
