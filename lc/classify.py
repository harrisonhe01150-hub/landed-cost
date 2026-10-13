"""Put each product into one of the owner's duty categories (only used in itemized mode).

The model can only answer with a category from the owner's own list (enforced by the JSON schema),
and the owner can change any of them on the page.
"""
import json

import requests

from lc.extract import MODEL, OLLAMA


def categories(names: list[str], cats: list[str], model: str = MODEL, timeout: int = 120) -> list[str]:
    if not names or not cats:
        return ["" for _ in names]
    schema = {
        "type": "object",
        "properties": {"items": {"type": "array", "items": {"type": "string", "enum": cats}}},
        "required": ["items"],
    }
    prompt = (
        "For each product below, pick the best category from this list: " + ", ".join(cats) + ".\n"
        "Answer with one category per product, in the same order.\n"
        + "\n".join(f"{i + 1}. {n}" for i, n in enumerate(names))
    )
    try:
        r = requests.post(
            OLLAMA,
            json={"model": model, "stream": False, "format": schema, "options": {"temperature": 0},
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=timeout,
        )
        r.raise_for_status()
        out = json.loads(r.json()["message"]["content"])["items"]
    except (requests.RequestException, KeyError, ValueError):
        return ["" for _ in names]
    return [c if c in cats else "" for c in out[: len(names)]] + [""] * max(0, len(names) - len(out))
