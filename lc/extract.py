"""Read a supplier delivery note / packing list photo into rows, with a local vision model (Ollama).

The model only transcribes. Every number it returns is cross-checked in code (qty = pcs/bale x bales,
amount = qty x unit price, the rows add up to the printed total), and anything that does not add up is flagged
for the owner to fix before any cost is computed.
"""
import base64
import json
from pathlib import Path

import requests

OLLAMA = "http://127.0.0.1:11434/api/chat"
MODEL = "gemma4:e4b"

ROW = {
    "type": "object",
    "properties": {
        "sku": {"type": "string", "description": "款号 / item code, exactly as printed"},
        "name": {"type": "string", "description": "品名, exactly as printed (Chinese is fine)"},
        "spec": {"type": "string", "description": "规格 / sizes / colours, as printed, or empty"},
        "pcs_per_bale": {"type": ["number", "null"], "description": "件/包 or pieces per carton; null if not printed"},
        "bales": {"type": ["number", "null"], "description": "包数 / cartons; null if not printed"},
        "qty_pcs": {"type": ["number", "null"], "description": "数量 total pieces; null if not printed"},
        "unit_price_rmb": {"type": ["number", "null"], "description": "单价 in yuan per piece"},
        "amount_rmb": {"type": ["number", "null"], "description": "金额 in yuan for the row"},
        "carton_cm": {"type": "string", "description": "箱规 carton size like 60x50x45, or empty"},
    },
    "required": ["sku", "name", "spec", "pcs_per_bale", "bales", "qty_pcs", "unit_price_rmb", "amount_rmb", "carton_cm"],
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
            "format": DOC,
            "options": {"temperature": 0},
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
    doc["rows_total_rmb"] = round(total, 2)
    doc["problems"] = problems
    return doc
