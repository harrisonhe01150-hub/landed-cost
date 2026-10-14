"""Hand-checked cost arithmetic and the cross-checks. Numbers are made up for the tests, not real rates."""
import pytest

from lc import extract
from lc.cost import DEFAULT_CATEGORY, Settings, allocate_cbm, landed

ROW = {"sku": "SF-320", "pcs_per_bale": 120, "bales": 2, "qty_pcs": 240, "unit_price_rmb": 23.5,
       "amount_rmb": 5640.0, "carton_cm": "60x50x45"}


def _row(**kw):
    r = {**ROW, **kw}
    r["cbm_per_bale"] = extract.carton_cbm(r.get("carton_cm", ""))
    return r


def zar(**kw):
    return Settings(**{"fx_zar_per_cny": 2.5, "allin_rate_currency": "ZAR", **kw})


def test_allin_per_cbm():
    (r,) = landed([_row()], zar(allin_rate_per_cbm=6000, markup=1.3, round_to_zar=0.5))
    # goods 23.5*2.5 = 58.75; volume 0.135*2 = 0.27 CBM; shipping 6000*0.27/240 = 6.75
    assert r["breakdown_per_piece"] == {"goods": 58.75, "agent": 0.0, "shipping_allin": 6.75}
    assert r["landed_per_piece"] == 65.5 and r["landed_per_bale"] == 7860.0
    assert r["suggested_per_piece"] == 85.5 and r["suggested_per_bale"] == 10260.0  # 85.15 -> up to 85.5


def test_allin_rate_in_yuan_is_converted_and_yuan_is_the_default():
    (r,) = landed([_row()], Settings(fx_zar_per_cny=2.5, allin_rate_per_cbm=2400))
    assert r["breakdown_per_piece"]["shipping_allin"] == 6.75  # 2400 CNY = 6000 ZAR


def test_itemized_duty_on_fob_vat_as_cashflow_and_clearing_split_by_this_notes_share():
    s = zar(mode="itemized", freight_per_cbm_zar=2000, clearing_fixed_zar=4800, duty_rates={"clothing": 0.45})
    (r,) = landed([_row(category="clothing")], s, shipment_share=0.1)
    b = r["breakdown_per_piece"]
    assert b["duty"] == pytest.approx(26.44, abs=0.01)  # 58.75 * 0.45
    assert b["freight"] == 2.25  # 2000*0.27/240
    assert b["clearing"] == 2.0  # this note carries 10% of R4800 over 240 pcs
    assert "vat" not in b and r["warnings"] == []
    assert r["vat_cashflow_per_piece"] == pytest.approx((58.75 * 1.1 + 26.4375) * 0.15, abs=0.01)


def test_clearing_is_not_charged_in_full_to_one_note():  # review C21
    s = zar(mode="itemized", freight_per_cbm_zar=2000, clearing_fixed_zar=15000, duty_rates={"clothing": 0.45})
    (r,) = landed([_row(category="clothing")], s)
    assert "clearing" not in r["breakdown_per_piece"] and any("清关费没算" in w for w in r["warnings"])


def test_itemized_unregistered_vat_is_a_cost_and_missing_duty_rate_warns():
    (r,) = landed([_row(category="mystery")], zar(mode="itemized", freight_per_cbm_zar=1, vat_registered=False))
    assert "vat" in r["breakdown_per_piece"] and any("关税按 0" in w for w in r["warnings"])


def test_no_volume_anywhere_is_an_error_not_free_shipping():  # review C1/C14
    (r,) = landed([_row(carton_cm="", cbm_per_bale=None)], zar(allin_rate_per_cbm=6000))
    assert "error" in r and "体积" in r["error"]


def test_known_volumes_exceeding_the_printed_total_leave_unknown_rows_unknown():  # review C1 case (a)
    rows = [_row(bales=10), {**ROW, "carton_cm": "", "cbm_per_bale": None, "amount_rmb": 1000.0}]
    assert allocate_cbm(rows, doc_total_cbm=1.3)[1] == (None, False)


def test_volume_from_quantity_when_bales_are_not_printed():  # review C1 case (b)
    (r,) = landed([_row(bales=None, qty_pcs=250)], zar(allin_rate_per_cbm=6000))
    assert r["cbm"] == pytest.approx(0.135 * 3)  # 250 pcs at 120/bale = 3 cartons


def test_rows_without_cartons_share_printed_total_by_value():
    rows = [_row(), {**ROW, "carton_cm": "", "cbm_per_bale": None, "amount_rmb": 1000.0},
            {**ROW, "carton_cm": "", "cbm_per_bale": None, "amount_rmb": 3000.0}]
    vols = allocate_cbm(rows, doc_total_cbm=1.27)
    assert vols[0] == (pytest.approx(0.27), False)
    assert vols[1] == (pytest.approx(0.25), True) and vols[2] == (pytest.approx(0.75), True)


def test_missing_unit_price_is_never_costed_as_zero():  # review C5/C17
    (r,) = landed([_row(unit_price_rmb=None)], zar(allin_rate_per_cbm=6000))
    assert "error" in r and "单价" in r["error"]


def test_check_derives_missing_price_from_amount_and_keeps_the_row_flagged():  # review C5
    doc = extract.check({"rows": [dict(ROW, unit_price_rmb=None)], "total_amount_rmb": 5640.0})
    r = doc["rows"][0]
    assert r["unit_price_rmb"] == 23.5 and r["problems"] and "金额÷数量" in r["problems"][0]


def test_check_flags_rows_that_do_not_add_up():
    out = extract.check({"rows": [dict(ROW, amount_rmb=5460.0)], "total_amount_rmb": 5640.0})
    assert any("金额" in p for p in out["problems"]) and any("合计" in p for p in out["problems"])


def test_a_small_missing_row_is_not_hidden_by_a_percentage_tolerance():  # review C6
    rows = [dict(ROW, sku=f"A{i}", pcs_per_bale=100, bales=1, qty_pcs=100, unit_price_rmb=88.0, amount_rmb=8800.0)
            for i in range(9)]
    out = extract.check({"rows": rows, "total_amount_rmb": 80000.0})  # the ¥800 row was skipped
    assert any("合计" in p for p in out["problems"])


def test_unit_price_misread_by_one_percent_is_caught():  # review low: tolerance was 1.1% of the amount
    out = extract.check({"rows": [dict(ROW, unit_price_rmb=23.25)], "total_amount_rmb": 5640.0})
    assert out["rows"][0]["problems"]


def test_carton_volumes_are_checked_against_the_printed_total():  # review C7 (a)
    out = extract.check({"rows": [dict(ROW, carton_cm="80x50x45")], "total_amount_rmb": 5640.0, "total_cbm": 0.27})
    assert any("体积" in p for p in out["problems"])


def test_swapped_bales_and_pieces_per_bale_trip_the_volume_check():  # review C7 (b)
    out = extract.check({"rows": [dict(ROW, pcs_per_bale=2, bales=120)], "total_amount_rmb": 5640.0, "total_cbm": 0.27})
    assert any("体积" in p for p in out["problems"])


def test_carton_in_millimetres_is_a_row_problem():  # review C7 (c)
    out = extract.check({"rows": [dict(ROW, carton_cm="600x500x450")], "total_amount_rmb": 5640.0})
    assert any("毫米" in p for p in out["rows"][0]["problems"])


def test_check_flags_row_volumes_that_do_not_add_up_to_the_printed_total():
    rows = [dict(ROW, cbm=0.27), dict(ROW, cbm=25.1)]  # 25.1 is a gross weight read into the volume column
    out = extract.check({"rows": rows, "total_amount_rmb": 11280.0, "total_cbm": 0.55})
    assert any("体积" in p for p in out["problems"])


def test_per_dozen_prices_are_labelled_per_dozen():  # review C23
    doc = extract.check({"rows": [dict(ROW, unit="打")], "total_amount_rmb": 5640.0})
    (r,) = landed(doc["rows"], zar(allin_rate_per_cbm=6000))
    assert r["unit_label"] == "打"
    assert extract.unit_label("双") == "件" and extract.unit_label("") == "件"


def test_allin_rate_per_category_with_explicit_default():
    s = Settings(fx_zar_per_cny=2.5, allin_rate_per_cbm=1450, allin_rates={"服装 clothing": 3450})
    clothing, general = landed([_row(category="服装 clothing"), _row(category=DEFAULT_CATEGORY)], s)
    assert clothing["breakdown_per_piece"]["shipping_allin"] == pytest.approx(3450 * 2.5 * 0.27 / 240, abs=0.01)
    assert general["breakdown_per_piece"]["shipping_allin"] == pytest.approx(1450 * 2.5 * 0.27 / 240, abs=0.01)
    assert general["warnings"] == [] and s.choices() == ["服装 clothing", DEFAULT_CATEGORY]


def test_unsorted_row_with_category_rates_warns_instead_of_silently_taking_the_cheap_rate():  # review C19
    s = Settings(fx_zar_per_cny=2.5, allin_rate_per_cbm=1450, allin_rates={"服装 clothing": 3450})
    (r,) = landed([_row(category="")], s)
    assert any("未分类" in w for w in r["warnings"])


def test_no_allin_rate_at_all_is_an_error():
    (r,) = landed([_row(category="x")], zar())
    assert "error" in r


def test_printed_row_volume_beats_carton_size():
    (r,) = landed([_row(cbm=0.5)], zar(allin_rate_per_cbm=6000))
    assert r["cbm"] == 0.5


def test_readiness_lists_what_is_missing_per_mode():  # review C18
    assert Settings(fx_zar_per_cny=2.5).missing() == ["包税价/方"]
    assert Settings(fx_zar_per_cny=2.5, mode="itemized").missing() == ["海运 R/方", "关税率"]
    assert Settings(fx_zar_per_cny=2.5, allin_rates={"x": 1}).missing() == []
