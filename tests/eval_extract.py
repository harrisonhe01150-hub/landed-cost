"""Measure transcription accuracy on the synthetic test set, per model.

For every document: rows found, numeric fields exactly right, SKUs right, name character accuracy,
and the number that matters most: SILENT errors, i.e. wrong numbers in rows the cross-check did NOT flag.
Usage: python tests/eval_extract.py <testset_dir> <model> [<model> ...]  -> prints a table, writes eval_<model>.json
"""
import difflib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lc import extract  # noqa: E402

NUM = ["pcs_per_bale", "bales", "qty_pcs", "unit_price_rmb", "amount_rmb", "cbm"]


def same(a, b) -> bool:
    if b is None:
        return True  # field not printed: anything (usually null or computed) is fine
    return a is not None and abs(float(a) - float(b)) <= max(0.006 * abs(b), 0.005)


def norm_sku(s: str) -> str:
    return (s or "").upper().replace(" ", "").replace("—", "-").replace("–", "-").replace("_", "-")


def align(got: list, want: list) -> list[tuple]:
    sm = difflib.SequenceMatcher(a=[norm_sku(r.get("sku")) for r in got], b=[norm_sku(r["sku"]) for r in want], autojunk=False)
    pairs = []
    for blk in sm.get_matching_blocks():
        pairs += [(got[blk.a + k], want[blk.b + k]) for k in range(blk.size)]
    used = {id(w) for _, w in pairs}
    left = [r for r in got if not any(r is g for g, _ in pairs)]
    for w in want:  # pair the rest by order, so a misread SKU still gets its numbers scored
        if id(w) not in used and left:
            pairs.append((left.pop(0), w))
    return pairs


def score(doc: dict, truth: dict) -> dict:
    pairs = align(doc["rows"], truth["rows"])
    fields = right = silent = sku_ok = 0
    name_sim = []
    for g, w in pairs:
        wrong = [k for k in NUM if w.get(k) is not None and not same(g.get(k), w[k])]
        fields += sum(1 for k in NUM if w.get(k) is not None)
        right += sum(1 for k in NUM if w.get(k) is not None) - len(wrong)
        if wrong and not g.get("problems"):
            silent += len(wrong)
        sku_ok += norm_sku(g.get("sku")) == norm_sku(w["sku"])
        name_sim.append(difflib.SequenceMatcher(a=g.get("name", "").replace(" ", ""), b=w["name"].replace(" ", "")).ratio())
    missing = len(truth["rows"]) - len(pairs)
    return {
        "rows_want": len(truth["rows"]), "rows_got": len(doc["rows"]), "rows_missing": missing,
        "fields": fields, "fields_right": right, "silent_errors": silent,
        "sku_right": sku_ok, "name_char_acc": round(sum(name_sim) / len(name_sim), 3) if name_sim else 0,
        "total_right": same(doc.get("total_amount_rmb"), truth["total_amount_rmb"]),
        "flagged": len(doc.get("problems", [])),
    }


def unload_others(model: str) -> None:
    """Two vision models resident at once overflow a 16 GB card (that crashed a load), so evaluate one at a time."""
    import requests

    for m in requests.get("http://127.0.0.1:11434/api/ps", timeout=10).json().get("models", []):
        if m["name"] not in model.split("+"):
            requests.post("http://127.0.0.1:11434/api/generate", json={"model": m["name"], "keep_alive": 0}, timeout=60)


def run(testset: Path, model: str) -> list[dict]:
    unload_others(model)
    out = []
    for truth_file in sorted(testset.glob("*.truth.json")):
        name = truth_file.name.removesuffix(".truth.json")
        img = next(p for p in testset.glob(name + ".*") if p.suffix in (".png", ".jpg"))
        truth = json.loads(truth_file.read_text(encoding="utf-8"))
        t = time.time()
        try:
            if "+" in model:  # "first+second": the two-pass reader the app uses
                first, second = model.split("+")
                doc = extract.read_checked(img, first=first, second=second)
            else:
                doc = extract.check(extract.read_document(img, model=model))
            s = score(doc, truth)
        except Exception as e:  # noqa: BLE001 - a crash is a result too
            s = {"error": repr(e)[:200], "rows_want": len(truth["rows"])}
        s.update(doc=name, model=model, secs=round(time.time() - t, 1))
        out.append(s)
        print(json.dumps(s, ensure_ascii=False))
    return out


if __name__ == "__main__":
    testset = Path(sys.argv[1])
    for m in sys.argv[2:]:
        res = run(testset, m)
        ok = [r for r in res if "error" not in r]
        f, fr = sum(r["fields"] for r in ok), sum(r["fields_right"] for r in ok)
        summary = {
            "model": m, "docs": len(res), "crashed": len(res) - len(ok),
            "numeric_field_acc": round(fr / f, 4) if f else None, "silent_errors": sum(r["silent_errors"] for r in ok),
            "rows_missing": sum(r["rows_missing"] for r in ok), "totals_right": sum(r["total_right"] for r in ok),
            "sku_acc": round(sum(r["sku_right"] for r in ok) / max(1, sum(r["rows_want"] for r in ok)), 4),
            "name_char_acc": round(sum(r["name_char_acc"] for r in ok) / max(1, len(ok)), 3),
            "median_secs": sorted(r["secs"] for r in res)[len(res) // 2],
        }
        print("SUMMARY", json.dumps(summary, ensure_ascii=False))
        (testset / f"eval_{m.replace(':', '_').replace('+', '_then_')}.json").write_text(
            json.dumps({"summary": summary, "docs": res}, ensure_ascii=False, indent=1), encoding="utf-8")
