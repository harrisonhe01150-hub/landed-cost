"""Landed cost per unit in ZAR, and a suggested wholesale price.

Two ways goods actually get from Yiwu to a Johannesburg stall:
  - "allin":    a forwarder quotes one door-to-door, duty-inclusive price per CBM (常见的「包税」).
  - "itemized": you pay freight per CBM, customs duty on the FOB value, import VAT and clearing yourself.
Every rate is a setting the owner fills in from their own bills; nothing here is a tax opinion.

Rule: a number that is missing never silently becomes 0 in a cost. A row either gets a full cost, a cost with an
explicit warning (something was counted as 0 on purpose and says so), or an error and no price at all.
"""
import math
from dataclasses import asdict, dataclass, field

from lc.extract import row_volume

DEFAULT_CATEGORY = "其他 · 默认价"


@dataclass
class Settings:
    fx_zar_per_cny: float  # rand per yuan, from the owner's bank or a reference rate
    mode: str = "allin"  # "allin" | "itemized"
    # allin
    allin_rate_per_cbm: float = 0.0  # default rate, used for 其他 / categories without their own
    allin_rate_currency: str = "CNY"  # "ZAR" | "CNY"; Yiwu forwarders usually quote yuan per 方
    allin_rates: dict = field(default_factory=dict)  # category -> rate per CBM (forwarders charge clothing more)
    # itemized
    freight_per_cbm_zar: float = 0.0
    clearing_fixed_zar: float = 0.0  # broker + port charges for the WHOLE shipment; each note pays its share
    local_transport_per_cbm_zar: float = 0.0
    duty_rates: dict = field(default_factory=dict)  # category -> ad valorem rate, e.g. {"棉袜 cotton socks": 0.45}
    vat_rate: float = 0.15
    vat_uplift: float = 0.10  # VAT base = FOB x (1 + uplift) + duty
    vat_registered: bool = True  # registered importers claim import VAT back, so it's cash flow, not cost
    # both
    agent_commission_pct: float = 0.0  # Yiwu buying agent / payment fees, % of goods value
    markup: float = 1.3  # wholesale price = landed cost x markup
    round_to_zar: float = 0.5  # round the suggested price per unit up to this step

    def to_dict(self) -> dict:
        return asdict(self)

    def categories(self) -> list[str]:
        """The owner's own category list for this mode (what the model may choose from)."""
        return list(self.allin_rates if self.mode == "allin" else self.duty_rates)

    def choices(self) -> list[str]:
        """Categories plus an explicit 'default' choice, so 'deliberately default' differs from 'never sorted'."""
        cats = self.categories()
        return cats + [DEFAULT_CATEGORY] if (cats and self.mode == "allin") else cats

    def missing(self) -> list[str]:
        """What still has to be filled in before prices mean anything (Chinese, shown to the owner)."""
        out = []
        if not self.fx_zar_per_cny > 0:
            out.append("汇率")
        if self.mode == "allin" and not (self.allin_rate_per_cbm > 0 or self.allin_rates):
            out.append("包税价/方")
        if self.mode == "itemized":
            if not self.freight_per_cbm_zar > 0:
                out.append("海运 R/方")
            if not self.duty_rates:
                out.append("关税率")
        if not self.markup >= 1:
            out.append("加价倍数")
        return out


def _round_up(x: float, step: float) -> float:
    return math.ceil(x / step - 1e-9) * step if step else x


def allocate_cbm(rows: list[dict], doc_total_cbm: float | None) -> list[tuple[float | None, bool]]:
    """(cbm, estimated?) per row. A row with no carton size or printed volume gets a share of the printed total
    CBM by goods value, but only if there is volume left to share; otherwise None (unknown), never 0."""
    known = [row_volume(r) for r in rows]
    if all(k is not None for k in known):
        return [(k, False) for k in known]
    rest = (doc_total_cbm or 0) - sum(k for k in known if k is not None)
    unknown_value = sum((r.get("amount_rmb") or 0) for r, k in zip(rows, known) if k is None)
    out = []
    for r, k in zip(rows, known):
        if k is not None:
            out.append((k, False))
        elif rest > 0.001 and unknown_value > 0 and (r.get("amount_rmb") or 0) > 0:
            out.append((rest * r["amount_rmb"] / unknown_value, True))
        else:
            out.append((None, False))
    return out


def landed(rows: list[dict], s: Settings, doc_total_cbm: float | None = None,
           shipment_share: float | None = None) -> list[dict]:
    """Per-row landed cost per unit (ZAR), with a breakdown and a suggested wholesale price.

    shipment_share: this note's share (0-1) of the whole shipment, used to split the per-shipment clearing fee.
    """
    vols = allocate_cbm(rows, doc_total_cbm)
    goods_zar_total = sum((r.get("amount_rmb") or 0) for r in rows) * s.fx_zar_per_cny
    volume_costs = (s.mode == "allin") or (s.freight_per_cbm_zar + s.local_transport_per_cbm_zar > 0)
    out = []
    for r, (cbm, est) in zip(rows, vols):
        q, price_rmb = r.get("qty_pcs"), r.get("unit_price_rmb")
        unit = r.get("unit_label") or "件"
        if not q:
            out.append({**r, "error": "缺数量，没算 · no quantity"})
            continue
        if price_rmb is None:
            out.append({**r, "error": "缺单价，没算 · no unit price"})
            continue
        if cbm is None and volume_costs:
            out.append({**r, "error": "缺体积：请填这行的箱规，或整单的总体积 · no volume, shipping not costed"})
            continue
        warnings = []
        goods = price_rmb * s.fx_zar_per_cny
        parts = {"goods": goods, "agent": goods * s.agent_commission_pct / 100}
        cat = r.get("category") or ""
        if s.mode == "allin":
            if cat in s.allin_rates:
                base = s.allin_rates[cat]
            else:
                base = s.allin_rate_per_cbm
                if s.allin_rates and cat != DEFAULT_CATEGORY:
                    warnings.append(("未分类" if not cat else f"类别「{cat}」没有包税价") + f"，按默认价 {base:g}/方 算了，请选类别")
            if not base:
                out.append({**r, "error": "没有包税价，运费没算 · no all-in rate"})
                continue
            rate = base * (s.fx_zar_per_cny if s.allin_rate_currency == "CNY" else 1)
            parts["shipping_allin"] = rate * cbm / q
        else:
            duty_rate = s.duty_rates.get(cat)
            parts["freight"] = (s.freight_per_cbm_zar + s.local_transport_per_cbm_zar) * (cbm or 0) / q
            if s.clearing_fixed_zar:
                if shipment_share is None:
                    warnings.append("清关费没算：请在单据页填这张单占整票货的比例")
                else:
                    share = (goods * q / goods_zar_total) if goods_zar_total else 0
                    parts["clearing"] = s.clearing_fixed_zar * shipment_share * share / q
            if duty_rate is None:
                warnings.append(("未分类" if not cat else f"类别「{cat}」没有关税率") + "，关税按 0 算了，请选类别")
            parts["duty"] = goods * (duty_rate or 0)
            vat = (goods * (1 + s.vat_uplift) + parts["duty"]) * s.vat_rate
            if s.vat_registered:
                r = {**r, "vat_cashflow_per_piece": round(vat, 2)}
            else:
                parts["vat"] = vat
        cost = sum(parts.values())
        suggest = _round_up(cost * s.markup, s.round_to_zar)
        per = r.get("pcs_per_bale")
        out.append(
            {
                **r,
                "unit_label": unit,
                "cbm": round(cbm, 4) if cbm is not None else None,
                "cbm_estimated": est,
                "warnings": warnings,
                "breakdown_per_piece": {k: round(v, 2) for k, v in parts.items()},
                "landed_per_piece": round(cost, 2),
                "landed_per_bale": round(cost * per, 2) if per else None,
                "suggested_per_piece": suggest,
                "suggested_per_bale": round(suggest * per, 2) if per else None,
                "margin_pct": round((suggest - cost) / suggest * 100, 1) if suggest else None,
            }
        )
    return out
