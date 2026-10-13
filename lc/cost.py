"""Landed cost per piece in ZAR, and a suggested wholesale price.

Two ways goods actually get from Yiwu to a Johannesburg stall:
  - "allin":    a forwarder quotes one door-to-door, duty-inclusive price per CBM (常见的「包税」).
  - "itemized": you pay freight per CBM, customs duty on the FOB value, import VAT and clearing yourself.
Every rate is a setting the owner fills in from their own bills; nothing here is a tax opinion.
"""
import math
from dataclasses import asdict, dataclass, field


@dataclass
class Settings:
    fx_zar_per_cny: float  # rand per yuan, from the owner's bank or a reference rate
    mode: str = "allin"  # "allin" | "itemized"
    # allin
    allin_rate_per_cbm: float = 0.0
    allin_rate_currency: str = "ZAR"  # "ZAR" | "CNY"
    # itemized
    freight_per_cbm_zar: float = 0.0
    clearing_fixed_zar: float = 0.0  # broker + port charges for the whole shipment
    local_transport_per_cbm_zar: float = 0.0
    duty_rates: dict = field(default_factory=dict)  # category -> ad valorem rate, e.g. {"clothing": 0.45}
    vat_rate: float = 0.15
    vat_uplift: float = 0.10  # VAT base = FOB x (1 + uplift) + duty
    vat_registered: bool = True  # registered importers claim import VAT back, so it's cash flow, not cost
    # both
    agent_commission_pct: float = 0.0  # Yiwu buying agent / payment fees, % of goods value
    markup: float = 1.3  # wholesale price = landed cost x markup
    round_to_zar: float = 0.5  # round the suggested price per piece up to this step

    def to_dict(self) -> dict:
        return asdict(self)


def _round_up(x: float, step: float) -> float:
    return math.ceil(x / step - 1e-9) * step if step else x


def row_cbm(row: dict) -> float | None:
    """Volume of one row's goods: carton size x number of bales/cartons."""
    if row.get("cbm_per_bale") and row.get("bales"):
        return row["cbm_per_bale"] * row["bales"]
    return None


def allocate_cbm(rows: list[dict], doc_total_cbm: float | None) -> list[tuple[float, bool]]:
    """(cbm, estimated?) per row. Rows without carton sizes share the printed total CBM by goods value."""
    known = [row_cbm(r) for r in rows]
    if all(k is not None for k in known):
        return [(k, False) for k in known]
    rest = (doc_total_cbm or 0) - sum(k for k in known if k is not None)
    unknown_value = sum((r.get("amount_rmb") or 0) for r, k in zip(rows, known) if k is None)
    out = []
    for r, k in zip(rows, known):
        if k is not None:
            out.append((k, False))
        elif rest > 0 and unknown_value > 0:
            out.append((rest * (r.get("amount_rmb") or 0) / unknown_value, True))
        else:
            out.append((0.0, True))
    return out


def landed(rows: list[dict], s: Settings, doc_total_cbm: float | None = None) -> list[dict]:
    """Per-row landed cost per piece (ZAR), with a breakdown and a suggested wholesale price."""
    vols = allocate_cbm(rows, doc_total_cbm)
    goods_zar_total = sum((r.get("amount_rmb") or 0) for r in rows) * s.fx_zar_per_cny
    out = []
    for r, (cbm, est) in zip(rows, vols):
        q = r.get("qty_pcs") or 0
        price_rmb = r.get("unit_price_rmb") or 0
        if not q:
            out.append({**r, "error": "no quantity"})
            continue
        goods = price_rmb * s.fx_zar_per_cny
        parts = {"goods": goods, "agent": goods * s.agent_commission_pct / 100}
        if s.mode == "allin":
            rate = s.allin_rate_per_cbm * (s.fx_zar_per_cny if s.allin_rate_currency == "CNY" else 1)
            parts["shipping_allin"] = rate * cbm / q
        else:
            duty_rate = s.duty_rates.get(r.get("category", ""), None)
            parts["freight"] = (s.freight_per_cbm_zar + s.local_transport_per_cbm_zar) * cbm / q
            share = (goods * q / goods_zar_total) if goods_zar_total else 0
            parts["clearing"] = s.clearing_fixed_zar * share / q
            parts["duty"] = goods * (duty_rate or 0)
            vat = (goods * (1 + s.vat_uplift) + parts["duty"]) * s.vat_rate
            if s.vat_registered:
                r = {**r, "vat_cashflow_per_piece": round(vat, 2)}
            else:
                parts["vat"] = vat
            if duty_rate is None:
                r = {**r, "warning": f"no duty rate set for category '{r.get('category', '')}'"}
        cost = sum(parts.values())
        suggest = _round_up(cost * s.markup, s.round_to_zar)
        out.append(
            {
                **r,
                "cbm": round(cbm, 4),
                "cbm_estimated": est,
                "breakdown_per_piece": {k: round(v, 2) for k, v in parts.items()},
                "landed_per_piece": round(cost, 2),
                "landed_per_bale": round(cost * (r.get("pcs_per_bale") or 0), 2) if r.get("pcs_per_bale") else None,
                "suggested_per_piece": suggest,
                "suggested_per_bale": round(suggest * r["pcs_per_bale"], 2) if r.get("pcs_per_bale") else None,
                "margin_pct": round((suggest - cost) / suggest * 100, 1) if suggest else None,
            }
        )
    return out
