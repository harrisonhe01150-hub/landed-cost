"""Put each product into one of the owner's own categories (duty rates or all-in shipping rates).

The model can only answer with a category from the owner's list (enforced by the JSON schema), and the owner can
change any of them on the page. If the model doesn't answer, nothing the owner set is overwritten.
"""
import json

import requests

from lc.extract import MODEL, OLLAMA


def categories(names: list[str], cats: list[str], model: str = MODEL, timeout: int = 120) -> list[str] | None:
    """One category per name, '' where the model gave nothing usable; None if the model could not be reached."""
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
            json={"model": model, "stream": False, "think": False, "format": schema, "options": {"temperature": 0},
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=timeout,
        )
        r.raise_for_status()
        out = json.loads(r.json()["message"]["content"])["items"]
    except (requests.RequestException, KeyError, ValueError):
        return None
    return [c if c in cats else "" for c in out[: len(names)]] + [""] * max(0, len(names) - len(out))
