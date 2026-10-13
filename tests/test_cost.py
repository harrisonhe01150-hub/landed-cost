"""Hand-checked cost arithmetic. Numbers are made up for the test, not real rates."""
import pytest

from lc import extract
from lc.cost import Settings, allocate_cbm, landed

ROW = {"sku": "SF-320", "pcs_per_bale": 120, "bales": 2, "qty_pcs": 240, "unit_price_rmb": 23.5,
       "amount_rmb": 5640.0, "carton_cm": "60x50x45"}


def _row(**kw):
    r = {**ROW, **kw}
    r["cbm_per_bale"] = extract.carton_cbm(r["carton_cm"])
    return r


def test_allin_per_cbm():
    s = Settings(fx_zar_per_cny=2.5, mode="allin", allin_rate_per_cbm=6000, markup=1.3, round_to_zar=0.5)
    (r,) = landed([_row()], s)
    # goods 23.5*2.5 = 58.75; volume 0.135*2 = 0.27 CBM; shipping 6000*0.27/240 = 6.75
    assert r["breakdown_per_piece"] == {"goods": 58.75, "agent": 0.0, "shipping_allin": 6.75}
    assert r["landed_per_piece"] == 65.5
    assert r["landed_per_bale"] == 7860.0
    assert r["suggested_per_piece"] == 85.5  # 65.5*1.3 = 85.15 -> up to 85.5
    assert r["suggested_per_bale"] == 10260.0


def test_allin_rate_in_yuan_is_converted():
    s = Settings(fx_zar_per_cny=2.5, allin_rate_per_cbm=2400, allin_rate_currency="CNY")
    (r,) = landed([_row()], s)
    assert r["breakdown_per_piece"]["shipping_allin"] == 6.75  # 2400 CNY = 6000 ZAR


def test_itemized_duty_on_fob_and_vat_as_cashflow_when_registered():
    s = Settings(fx_zar_per_cny=2.5, mode="itemized", freight_per_cbm_zar=2000, clearing_fixed_zar=480,
                 duty_rates={"clothing": 0.45}, vat_registered=True)
    (r,) = landed([_row(category="clothing")], s)
    b = r["breakdown_per_piece"]
    assert b["duty"] == pytest.approx(26.44, abs=0.01)  # 58.75 * 0.45
    assert b["freight"] == 2.25  # 2000*0.27/240
    assert b["clearing"] == 2.0  # the only row carries all 480 over 240 pcs
    assert "vat" not in b
    assert r["vat_cashflow_per_piece"] == pytest.approx((58.75 * 1.1 + 26.4375) * 0.15, abs=0.01)


def test_itemized_unregistered_vat_is_a_cost_and_missing_duty_rate_warns():
    s = Settings(fx_zar_per_cny=2.5, mode="itemized", vat_registered=False)
    (r,) = landed([_row(category="mystery")], s)
    assert "vat" in r["breakdown_per_piece"] and "warning" in r


def test_rows_without_cartons_share_printed_total_by_value():
    rows = [_row(), {**ROW, "carton_cm": "", "cbm_per_bale": None, "amount_rmb": 1000.0},
            {**ROW, "carton_cm": "", "cbm_per_bale": None, "amount_rmb": 3000.0}]
    vols = allocate_cbm(rows, doc_total_cbm=1.27)
    assert vols[0] == (pytest.approx(0.27), False)
    assert vols[1] == (pytest.approx(0.25), True) and vols[2] == (pytest.approx(0.75), True)


def test_check_flags_rows_that_do_not_add_up():
    doc = {"rows": [dict(ROW, amount_rmb=5460.0)], "total_amount_rmb": 5640.0}
    out = extract.check(doc)
    assert any("金额" in p for p in out["problems"])
    assert any("合计" in p for p in out["problems"])
