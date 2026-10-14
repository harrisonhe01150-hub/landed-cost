"""Read a supplier delivery note / packing list photo into rows, with a local vision model (Ollama).

The model only transcribes. Every number it returns is cross-checked in code (qty = pcs/bale x bales,
amount = qty x unit price, the rows add up to the printed totals), and anything that does not add up is flagged
for the owner to fix before any cost is trusted.
"""
import base64
import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path

import requests

OLLAMA = "http://127.0.0.1:11434/api/chat"
MODEL = os.environ.get("LC_MODEL", "gemma4:12b")
SECOND = os.environ.get("LC_SECOND_MODEL", "gemma4:e4b")

ROW = {
    "type": "object",
    "properties": {
        # Field order matters: the model fills these left to right like table columns. Putting carton_no first
        # shifted every value one column over on delivery notes (accuracy ~90% -> ~10%), so extras go last.
        "sku": {"type": "string", "description": "款号 / 货号 / item code, exactly as printed (NOT the 箱号 carton number)"},
        "name": {"type": "string", "description": "品名, exactly as printed (Chinese is fine)"},
        "spec": {"type": "string", "description": "规格 / sizes / colours, as printed, or empty"},
        "pcs_per_bale": {"type": ["number", "null"], "description": "件/包 or pieces per carton; null if not printed"},
        "bales": {"type": ["number", "null"], "description": "包数 / cartons; null if not printed"},
        "qty_pcs": {"type": ["number", "null"], "description": "数量 total; null if not printed"},
        "unit_price_rmb": {"type": ["number", "null"], "description": "单价 in yuan"},
        "amount_rmb": {"type": ["number", "null"], "description": "金额 in yuan for the row"},
        "carton_cm": {"type": "string", "description": "箱规 carton size like 60x50x45, or empty"},
        "cbm": {"type": ["number", "null"], "description": "体积 / 立方 / m³ printed for this row, or null"},
        "carton_no": {"type": "string", "description": "箱号 / carton numbers like 1-5 (packing lists only), or empty"},
        "unit": {"type": "string", "description": "单位 if printed for 数量/单价 (件 双 个 条 顶 打 套 ...), or empty"},
    },
    "required": ["sku", "name", "spec", "pcs_per_bale", "bales", "qty_pcs", "unit_price_rmb", "amount_rmb", "carton_cm",
                 "cbm", "carton_no", "unit"],
}
DOC = {
    "type": "object",
    "properties": {
        "supplier": {"type": "string"},
        "date": {"type": "string"},
        "doc_no": {"type": "string"},
        "rows": {"type": "array", "items": ROW},
        "total_amount_rmb": {"type": ["number", "null"], "description": "合计 金额 as printed"},
        "total_cbm": {"type": ["number", "null"], "description": "total volume in cubic metres (方) if printed"},
    },
    "required": ["supplier", "date", "doc_no", "rows", "total_amount_rmb", "total_cbm"],
}

PROMPT = """This is a photo of a Chinese wholesale supplier's delivery note (送货单) or packing list (装箱单).
Transcribe every product row of the table exactly, once each. Do not calculate, guess or fill in anything that is not
printed; use null for a number that is not printed. A comma inside a number is a thousands separator: 1,680 is 1680
and 28,941.20 is 28941.2. Skip the 合计 (total) row in "rows" and put its amount in total_amount_rmb."""

PIECE_UNITS = {"", "件", "双", "个", "条", "顶", "只", "支", "张", "pcs", "pc", "pr", "prs", "pair", "pairs"}


def _b64(path: Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


def read_document(path: Path, model: str = MODEL, timeout: int = 300, retries: int = 1) -> dict:
    payload = {
        "model": model,
        "stream": False,
        "think": False,  # with thinking on, Gemma 4 can ignore the schema (ollama#18774)
        "format": DOC,
        "options": {"temperature": 0, "num_ctx": 8192},
        "messages": [{"role": "user", "content": PROMPT, "images": [_b64(path)]}],
    }
    for attempt in range(retries + 1):
        r = requests.post(OLLAMA, json=payload, timeout=timeout)
        # A cold model load right after another model was unloaded sometimes dies with a CUDA error (HTTP 500);
        # the second attempt, once the GPU memory is really free, works.
        if r.status_code >= 500 and attempt < retries:
            time.sleep(3)
            continue
        r.raise_for_status()
        return json.loads(r.json()["message"]["content"])


def carton_cbm(carton: str) -> float | None:
    """'60×50×45' or '60x50x45cm' -> cubic metres."""
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", carton or "")]
    return round(nums[0] * nums[1] * nums[2] / 1e6, 4) if len(nums) >= 3 else None


def unit_label(unit: str) -> str:
    """'件' for anything that means one piece; otherwise the printed unit (e.g. 打), so prices say R/打, not R/件."""
    u = (unit or "").strip().lower()
    return "件" if u in PIECE_UNITS else unit.strip()


def row_volume(r: dict) -> float | None:
    """Volume of one row: the printed 体积 if there is one, else carton size x number of bales/cartons
    (bales derived from quantity / pieces per bale if only those are printed; a partial carton is a whole carton)."""
    if r.get("cbm"):
        return r["cbm"]
    per_bale = r.get("cbm_per_bale") or carton_cbm(r.get("carton_cm", ""))
    if not per_bale:
        return None
    if r.get("bales"):
        return per_bale * r["bales"]
    if r.get("qty_pcs") and r.get("pcs_per_bale"):
        return per_bale * math.ceil(r["qty_pcs"] / r["pcs_per_bale"] - 1e-9)
    return None


def check(doc: dict) -> dict:
    """Cross-check the transcription. Returns the doc with per-row 'problems' and a top-level 'problems' list."""
    problems, total = [], 0.0
    rows = doc.get("rows", [])
    for i, r in enumerate(rows, 1):
        rp = []
        per, n, q, p, a = (r.get(k) for k in ("pcs_per_bale", "bales", "qty_pcs", "unit_price_rmb", "amount_rmb"))
        if per and n and q and abs(per * n - q) > 0.5:
            rp.append(f"件/包 {per:g} × 包数 {n:g} = {per * n:g}，但数量写 {q:g}")
        if q is None and per and n:
            r["qty_pcs"], q = per * n, per * n
        if p is None and q and a is not None:
            r["unit_price_rmb"], p = round(a / q, 4), round(a / q, 4)
            rp.append(f"单价没读到，用 金额÷数量 = {p:g}，请核对")
        elif p is None:
            rp.append("缺单价")
        # A printed price is rounded to at most half a fen, so q x p can differ from 金额 by at most q x 0.005.
        if q and p is not None and a is not None and abs(q * p - a) > q * 0.005 + 0.01:
            rp.append(f"数量 {q:g} × 单价 {p:g} = {q * p:,.2f}，但金额写 {a:,.2f}")
        if a is None and q and p is not None:
            r["amount_rmb"], a = round(q * p, 2), round(q * p, 2)
        total += a or 0
        r["cbm_per_bale"] = carton_cbm(r.get("carton_cm", ""))
        if r["cbm_per_bale"] and r["cbm_per_bale"] > 1:
            rp.append(f"箱规 {r.get('carton_cm')} 算出来一箱 {r['cbm_per_bale']:g} 方，可能是毫米或读错了")
        r["unit_label"] = unit_label(r.get("unit", ""))
        r["problems"] = rp
        problems += [f"第 {i} 行 ({r.get('sku') or r.get('name')})：{x}" for x in rp]
    printed = doc.get("total_amount_rmb")
    # The printed row amounts add up exactly to 合计; allow only rounding (a whole-yuan 合计), never a whole row.
    if printed is not None and abs(printed - total) > max(0.5, 0.01 * len(rows)):
        problems.append(f"各行加起来 ¥{total:,.2f}，但合计写 ¥{printed:,.2f}（可能漏读或读错了一行）")
    printed_cbm = doc.get("total_cbm")
    vols = [row_volume(r) for r in rows]
    if printed_cbm and vols:
        known = sum(v for v in vols if v)
        tol = max(0.03 * printed_cbm, 0.002)
        if all(vols) and abs(known - printed_cbm) > tol:
            problems.append(f"各行体积加起来 {known:.3f} 方，但总体积写 {printed_cbm:.3f} 方")
        elif not all(vols) and known > printed_cbm + tol:
            problems.append(f"有体积的几行已经 {known:.3f} 方，超过了总体积 {printed_cbm:.3f} 方")
    doc["rows_total_rmb"] = round(total, 2)
    doc["problems"] = problems
    return doc


def _unload(model: str, wait: float = 15) -> None:
    """A 16 GB card can't hold both vision models at once, so swap them explicitly, and wait until the model has
    really left the GPU: loading the next one straight away sometimes crashed with a CUDA error."""
    base = OLLAMA.rsplit("/api/", 1)[0]
    try:
        requests.post(f"{base}/api/generate", json={"model": model, "keep_alive": 0}, timeout=60)
        deadline = time.time() + wait
        while time.time() < deadline:
            loaded = [m["name"] for m in requests.get(f"{base}/api/ps", timeout=10).json().get("models", [])]
            if model not in loaded:
                return
            time.sleep(0.5)
    except (requests.RequestException, ValueError, KeyError):
        pass


def _key(r: dict) -> str:
    return (r.get("sku") or "").upper().replace(" ", "")


def _row_problems(d: dict) -> int:
    return sum(len(r["problems"]) for r in d["rows"])


def _amount_gap(d: dict) -> float:
    printed = d.get("total_amount_rmb")
    return 0.0 if printed is None else abs(d["rows_total_rmb"] - printed)


def _volume_gap(d: dict) -> float:
    printed = d.get("total_cbm")
    vols = [row_volume(r) for r in d["rows"]]
    if not printed or not any(vols):
        return 0.0
    return abs(sum(v or 0 for v in vols) - printed) + (printed if not all(vols) else 0)


NUMBERS = ("pcs_per_bale", "bales", "qty_pcs", "unit_price_rmb", "amount_rmb", "cbm")


def read_checked(path: Path, first: str = MODEL, second: str | None = SECOND) -> dict:
    """Read with the first model; if the cross-check flags rows, ask a second model and take its numbers for a
    flagged row only when that row is complete, passes every check, and the whole document then has fewer row
    problems without the amount or volume totals getting any further off. Such rows are marked for the owner."""
    doc = check(read_document(path, model=first))
    doc["model"] = first
    if not doc["problems"] or not second or second == first:
        return doc
    _unload(first)
    try:
        alt = check(read_document(path, model=second))
    except (requests.RequestException, ValueError, KeyError):
        return doc
    finally:
        _unload(second)
    # A second read that lists the very same line twice is the model repeating itself, not two products:
    # drop exact duplicates (two colours under one code differ in their numbers and are kept, and stay ambiguous).
    seen, rows = set(), []
    for r in alt["rows"]:
        sig = (_key(r), r.get("name"), r.get("spec")) + tuple(r.get(f) for f in NUMBERS)
        if sig not in seen:
            seen.add(sig)
            rows.append(r)
    alt = {**alt, "rows": rows}
    alt_count, doc_count = Counter(map(_key, alt["rows"])), Counter(map(_key, doc["rows"]))
    by_key = {k: r for r in alt["rows"] if (k := _key(r)) and alt_count[k] == 1}  # a repeated code is ambiguous
    same_shape = len(alt["rows"]) == len(doc["rows"])
    merged = []
    for i, r in enumerate(doc["rows"]):
        k = _key(r)
        cand = (by_key.get(k) if k and doc_count[k] == 1 else None) or (alt["rows"][i] if same_shape else None)
        complete = cand is not None and all(cand.get(f) is not None for f in ("qty_pcs", "unit_price_rmb", "amount_rmb"))
        if r["problems"] and complete and not cand["problems"]:
            # Identity (code, name, spec) from the first read, which gets those right far more often;
            # the numbers from the second read, which passed every check.
            merged.append({**r, **{f: cand.get(f) for f in NUMBERS + ("carton_cm", "unit")}, "second_opinion": second})
        else:
            merged.append(r)
    clean = [{k: v for k, v in r.items() if k != "problems"} for r in merged]
    # The first read may have misread the printed 合计 itself. If the second read's own 合计 matches the (merged) rows
    # exactly — two independent reads agree and every check passes — use it, and say so on the page.
    if alt.get("total_amount_rmb") is not None and alt.get("total_amount_rmb") != doc.get("total_amount_rmb"):
        both = check({**doc, "rows": [dict(r) for r in clean], "total_amount_rmb": alt["total_amount_rmb"],
                      "total_cbm": alt.get("total_cbm") if alt.get("total_cbm") is not None else doc.get("total_cbm")})
        if not both["problems"]:
            both["model"] = f"{first} + {second}"
            both["notes"] = [f"合计按第二遍读的 ¥{alt['total_amount_rmb']:,.2f}（第一遍读成 ¥{doc.get('total_amount_rmb') or 0:,.2f}），"
                             "两遍和各行都对得上"]
            return both
    if merged == doc["rows"]:
        return doc
    trial = check({**doc, "rows": clean})
    if (_row_problems(trial) < _row_problems(doc) and _amount_gap(trial) <= _amount_gap(doc) + 0.05
            and _volume_gap(trial) <= _volume_gap(doc) + 0.002):
        trial["model"] = f"{first} + {second}"
        return trial
    return doc
