"""Read a supplier delivery note / packing list photo into rows, with a local vision model (Ollama).

The model only transcribes. Every number it returns is cross-checked in code (qty = pcs/bale x bales,
amount = qty x unit price, the rows add up to the printed total), and anything that does not add up is flagged
for the owner to fix before any cost is computed.
"""
import base64
import json
import os
from pathlib import Path

import requests

OLLAMA = "http://127.0.0.1:11434/api/chat"
MODEL = os.environ.get("LC_MODEL", "gemma4:12b")

ROW = {
    "type": "object",
    "properties": {
        # Field order matters: the model fills these left to right like table columns. Putting carton_no first
        # shifted every value one column over on delivery notes (accuracy ~90% -> ~10%), so it goes last.
        "sku": {"type": "string", "description": "款号 / 货号 / item code, exactly as printed (NOT the 箱号 carton number)"},
        "name": {"type": "string", "description": "品名, exactly as printed (Chinese is fine)"},
        "spec": {"type": "string", "description": "规格 / sizes / colours, as printed, or empty"},
        "pcs_per_bale": {"type": ["number", "null"], "description": "件/包 or pieces per carton; null if not printed"},
        "bales": {"type": ["number", "null"], "description": "包数 / cartons; null if not printed"},
        "qty_pcs": {"type": ["number", "null"], "description": "数量 total pieces; null if not printed"},
        "unit_price_rmb": {"type": ["number", "null"], "description": "单价 in yuan per piece"},
        "amount_rmb": {"type": ["number", "null"], "description": "金额 in yuan for the row"},
        "carton_cm": {"type": "string", "description": "箱规 carton size like 60x50x45, or empty"},
        "cbm": {"type": ["number", "null"], "description": "体积 / 立方 / m³ printed for this row, or null"},
        "carton_no": {"type": "string", "description": "箱号 / carton numbers like 1-5 (packing lists only), or empty"},
    },
    "required": ["sku", "name", "spec", "pcs_per_bale", "bales", "qty_pcs", "unit_price_rmb", "amount_rmb", "carton_cm",
                 "cbm", "carton_no"],
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
Transcribe every product row of the table exactly. Do not calculate, guess or fill in anything that is not printed;
use null for a number that is not printed. Skip the 合计 (total) row in "rows" and put its amount in total_amount_rmb."""


def _b64(path: Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


def read_document(path: Path, model: str = MODEL, timeout: int = 300) -> dict:
    r = requests.post(
        OLLAMA,
        json={
            "model": model,
            "stream": False,
            "think": False,  # with thinking on, Gemma 4 can ignore the schema (ollama#18774)
            "format": DOC,
            "options": {"temperature": 0, "num_ctx": 8192},
            "messages": [{"role": "user", "content": PROMPT, "images": [_b64(path)]}],
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return json.loads(r.json()["message"]["content"])


def carton_cbm(carton: str) -> float | None:
    """'60×50×45' or '60x50x45cm' -> cubic metres."""
    import re

    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", carton or "")]
    return round(nums[0] * nums[1] * nums[2] / 1e6, 4) if len(nums) >= 3 else None


def check(doc: dict, tol: float = 0.011) -> dict:
    """Cross-check the transcription. Returns the doc with per-row 'problems' and a top-level 'problems' list."""
    problems, total = [], 0.0
    for i, r in enumerate(doc.get("rows", []), 1):
        rp = []
        per, n, q, p, a = (r.get(k) for k in ("pcs_per_bale", "bales", "qty_pcs", "unit_price_rmb", "amount_rmb"))
        if per and n and q and abs(per * n - q) > 0.5:
            rp.append(f"件/包 {per:g} × 包数 {n:g} = {per * n:g}, but 数量 says {q:g}")
        if q is None and per and n:
            r["qty_pcs"], q = per * n, per * n
        if p is None:
            rp.append("单价 missing")
        if q and p is not None and a is not None and abs(q * p - a) > max(tol * a, 0.05):
            rp.append(f"数量 {q:g} × 单价 {p:g} = {q * p:,.2f}, but 金额 says {a:,.2f}")
        if a is None and q and p is not None:
            r["amount_rmb"], a = round(q * p, 2), round(q * p, 2)
        total += a or 0
        r["cbm_per_bale"] = carton_cbm(r.get("carton_cm", ""))
        r["problems"] = rp
        problems += [f"row {i} ({r.get('sku') or r.get('name')}): {x}" for x in rp]
    printed = doc.get("total_amount_rmb")
    if printed is not None and abs(printed - total) > max(tol * printed, 0.05):
        problems.append(f"rows add up to ¥{total:,.2f}, but 合计 says ¥{printed:,.2f}")
    vols = [r.get("cbm") for r in doc.get("rows", [])]
    printed_cbm = doc.get("total_cbm")
    if printed_cbm and any(v is not None for v in vols):
        vol = sum(v or 0 for v in vols)
        if None in vols or abs(vol - printed_cbm) > max(0.011 * printed_cbm, 0.002):
            problems.append(f"row volumes add up to {vol:.3f} m³, but the total says {printed_cbm:.3f} m³")
    doc["rows_total_rmb"] = round(total, 2)
    doc["problems"] = problems
    return doc


SECOND = os.environ.get("LC_SECOND_MODEL", "gemma4:e4b")


def _unload(model: str) -> None:
    """A 16 GB card can't hold both vision models at once (it crashed a load once), so swap them explicitly."""
    try:
        requests.post(OLLAMA.replace("/api/chat", "/api/generate"), json={"model": model, "keep_alive": 0}, timeout=60)
    except requests.RequestException:
        pass


def _key(r: dict) -> str:
    return (r.get("sku") or "").upper().replace(" ", "")


def read_checked(path: Path, first: str = MODEL, second: str | None = SECOND) -> dict:
    """Read with the first model; if the cross-check flags rows, ask a second model and take its version of a
    flagged row only when that row passes every check and the whole document then reconciles at least as well.
    Rows taken from the second model are marked so the owner looks at them."""
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
    by_key = {_key(r): r for r in alt["rows"] if _key(r)}
    same_shape = len(alt["rows"]) == len(doc["rows"])
    merged = []
    for i, r in enumerate(doc["rows"]):
        cand = by_key.get(_key(r)) or (alt["rows"][i] if same_shape else None)
        if r["problems"] and cand is not None and not cand["problems"]:
            # Identity (code, name, spec) from the first read, which gets those right far more often;
            # the numbers from the second read, which passed every check.
            merged.append({**cand, "sku": r["sku"], "name": r["name"], "spec": r["spec"], "second_opinion": second})
        else:
            merged.append(r)
    if merged == doc["rows"]:
        return doc
    trial = check({**doc, "rows": [{k: v for k, v in r.items() if k != "problems"} for r in merged]})
    if len(trial["problems"]) < len(doc["problems"]):
        trial["model"] = f"{first} + {second}"
        return trial
    return doc
