"""The two-pass reader: a second model may only replace rows that the first read got wrong, and only with rows
that pass every check. No model is called here."""
from lc import extract

GOOD = {"sku": "A1", "name": "袜子", "spec": "", "pcs_per_bale": 600, "bales": 2, "qty_pcs": 1200, "unit_price_rmb": 1.5,
        "amount_rmb": 1800.0, "carton_cm": "", "cbm": None, "carton_no": ""}
OTHER = {**GOOD, "sku": "B2", "name": "睡衣", "pcs_per_bale": 120, "bales": 1, "qty_pcs": 120, "unit_price_rmb": 20.0,
         "amount_rmb": 2400.0}


def fake_reader(by_model):
    return lambda path, model: {"supplier": "", "date": "", "doc_no": "", "total_amount_rmb": 4200.0, "total_cbm": None,
                                "rows": [dict(r) for r in by_model[model]]}


def test_clean_first_read_never_calls_the_second_model(monkeypatch):
    calls = []
    monkeypatch.setattr(extract, "_unload", lambda m: calls.append(m))
    monkeypatch.setattr(extract, "read_document", fake_reader({"first": [GOOD, OTHER]}))
    doc = extract.read_checked("x.jpg", first="first", second="second")
    assert doc["problems"] == [] and doc["model"] == "first" and calls == []


def test_flagged_row_is_replaced_by_a_passing_row_keeping_the_first_reads_identity(monkeypatch):
    monkeypatch.setattr(extract, "_unload", lambda m: None)
    shifted = {**OTHER, "pcs_per_bale": 1, "bales": 120, "qty_pcs": 20.0}  # columns shifted: doesn't add up
    second_row = {**OTHER, "sku": "1-1", "name": "睡衣"}  # right numbers, carton number read as the code
    monkeypatch.setattr(extract, "read_document", fake_reader({"first": [GOOD, shifted], "second": [GOOD, second_row]}))
    doc = extract.read_checked("x.jpg", first="first", second="second")
    assert doc["problems"] == []
    assert doc["rows"][1]["sku"] == "B2" and doc["rows"][1]["qty_pcs"] == 120 and doc["rows"][1]["second_opinion"] == "second"
    assert "second_opinion" not in doc["rows"][0]


def test_second_read_that_also_fails_changes_nothing(monkeypatch):
    monkeypatch.setattr(extract, "_unload", lambda m: None)
    bad = {**OTHER, "amount_rmb": 9999.0}
    monkeypatch.setattr(extract, "read_document", fake_reader({"first": [GOOD, bad], "second": [GOOD, bad]}))
    doc = extract.read_checked("x.jpg", first="first", second="second")
    assert doc["model"] == "first" and doc["problems"]
