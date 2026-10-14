"""The two-pass reader: a second model may only replace rows that the first read got wrong, and only with complete
rows that pass every check without making the totals any worse. No model is called here."""
from lc import extract

GOOD = {"sku": "A1", "name": "袜子", "spec": "", "pcs_per_bale": 600, "bales": 2, "qty_pcs": 1200, "unit_price_rmb": 1.5,
        "amount_rmb": 1800.0, "carton_cm": "", "cbm": None, "carton_no": "", "unit": ""}
OTHER = {**GOOD, "sku": "B2", "name": "睡衣", "pcs_per_bale": 120, "bales": 1, "qty_pcs": 120, "unit_price_rmb": 20.0,
         "amount_rmb": 2400.0}


def fake_reader(by_model, total=4200.0):
    return lambda path, model: {"supplier": "", "date": "", "doc_no": "", "total_amount_rmb": total, "total_cbm": None,
                                "rows": [dict(r) for r in by_model[model]]}


def setup(monkeypatch, by_model, total=4200.0):
    calls = []
    monkeypatch.setattr(extract, "_unload", lambda m: calls.append(m))
    monkeypatch.setattr(extract, "read_document", fake_reader(by_model, total))
    return calls


def test_clean_first_read_never_calls_the_second_model(monkeypatch):
    calls = setup(monkeypatch, {"first": [GOOD, OTHER]})
    doc = extract.read_checked("x.jpg", first="first", second="second")
    assert doc["problems"] == [] and doc["model"] == "first" and calls == []


def test_flagged_row_is_replaced_by_a_passing_row_keeping_the_first_reads_identity(monkeypatch):
    shifted = {**OTHER, "pcs_per_bale": 1, "bales": 120, "qty_pcs": 20.0}  # columns shifted: doesn't add up
    second_row = {**OTHER, "sku": "1-1", "name": "睡衣"}  # right numbers, carton number read as the code
    setup(monkeypatch, {"first": [GOOD, shifted], "second": [GOOD, second_row]})
    doc = extract.read_checked("x.jpg", first="first", second="second")
    assert doc["problems"] == []
    assert doc["rows"][1]["sku"] == "B2" and doc["rows"][1]["qty_pcs"] == 120 and doc["rows"][1]["second_opinion"] == "second"
    assert "second_opinion" not in doc["rows"][0]


def test_second_read_that_also_fails_changes_nothing(monkeypatch):
    bad = {**OTHER, "amount_rmb": 9999.0}
    setup(monkeypatch, {"first": [GOOD, bad], "second": [GOOD, bad]})
    doc = extract.read_checked("x.jpg", first="first", second="second")
    assert doc["model"] == "first" and doc["problems"]


def test_duplicate_code_never_takes_the_other_colours_numbers(monkeypatch):  # review C3
    black = {**GOOD, "sku": "X9", "spec": "黑", "pcs_per_bale": 50, "bales": 2, "qty_pcs": 100, "unit_price_rmb": 10.0,
             "amount_rmb": 1000.0}
    white = {**GOOD, "sku": "X9", "spec": "白", "pcs_per_bale": 100, "bales": 2, "qty_pcs": 200, "unit_price_rmb": 5.0,
             "amount_rmb": 1000.0}
    setup(monkeypatch, {"first": [{**black, "qty_pcs": 10}, white], "second": [black, white]}, total=2000.0)
    doc = extract.read_checked("x.jpg", first="first", second="second")
    assert doc["rows"][0]["unit_price_rmb"] == 10.0 and doc["rows"][0]["qty_pcs"] == 100


def test_merge_that_breaks_a_reconciled_total_is_rejected(monkeypatch):  # review C4
    misread_qty = {**OTHER, "qty_pcs": 1200}  # price and amount right, total still reconciles
    wrong_but_consistent = {**OTHER, "unit_price_rmb": 2.0, "amount_rmb": 240.0}
    setup(monkeypatch, {"first": [GOOD, misread_qty], "second": [GOOD, wrong_but_consistent]})
    doc = extract.read_checked("x.jpg", first="first", second="second")
    assert doc["model"] == "first" and doc["rows"][1]["unit_price_rmb"] == 20.0


def test_incomplete_second_read_row_is_never_taken(monkeypatch):  # review low
    shifted = {**OTHER, "qty_pcs": 20.0}
    empty = {**OTHER, "pcs_per_bale": None, "bales": None, "qty_pcs": None, "unit_price_rmb": None, "amount_rmb": None}
    setup(monkeypatch, {"first": [GOOD, shifted], "second": [GOOD, empty]})
    doc = extract.read_checked("x.jpg", first="first", second="second")
    assert doc["model"] == "first"


def test_misread_total_is_replaced_only_when_the_second_reads_total_matches_every_row(monkeypatch):
    shifted = {**OTHER, "qty_pcs": 20.0}
    monkeypatch.setattr(extract, "_unload", lambda m: None)
    monkeypatch.setattr(extract, "read_document", lambda path, model: {
        "supplier": "", "date": "", "doc_no": "", "total_cbm": None,
        "total_amount_rmb": 9999.0 if model == "first" else 4200.0,
        "rows": [dict(r) for r in ([GOOD, shifted] if model == "first" else [GOOD, OTHER])]})
    doc = extract.read_checked("x.jpg", first="first", second="second")
    assert doc["problems"] == [] and doc["total_amount_rmb"] == 4200.0 and "第二遍" in doc["notes"][0]


def test_second_reads_total_is_not_used_when_it_does_not_reconcile(monkeypatch):
    shifted = {**OTHER, "qty_pcs": 20.0}
    monkeypatch.setattr(extract, "_unload", lambda m: None)
    monkeypatch.setattr(extract, "read_document", lambda path, model: {
        "supplier": "", "date": "", "doc_no": "", "total_cbm": None,
        "total_amount_rmb": 9999.0 if model == "first" else 5000.0,
        "rows": [dict(r) for r in ([GOOD, shifted] if model == "first" else [GOOD, OTHER])]})
    doc = extract.read_checked("x.jpg", first="first", second="second")
    # the row fix is still taken, but the printed 合计 is kept and stays flagged for the owner
    assert doc["total_amount_rmb"] == 9999.0 and any("合计" in p for p in doc["problems"])
    assert doc["rows"][1]["qty_pcs"] == 120


def test_only_the_total_misread_is_fixed_when_both_reads_agree_on_every_row(monkeypatch):
    monkeypatch.setattr(extract, "_unload", lambda m: None)
    monkeypatch.setattr(extract, "read_document", lambda path, model: {
        "supplier": "", "date": "", "doc_no": "", "total_cbm": None,
        "total_amount_rmb": 4290.0 if model == "first" else 4200.0, "rows": [dict(GOOD), dict(OTHER)]})
    doc = extract.read_checked("x.jpg", first="first", second="second")
    assert doc["problems"] == [] and doc["total_amount_rmb"] == 4200.0


def test_a_line_the_second_model_repeated_is_not_treated_as_two_products(monkeypatch):
    misread_amount = {**OTHER, "amount_rmb": 2.0}  # "2,400" read as 2
    monkeypatch.setattr(extract, "_unload", lambda m: None)
    monkeypatch.setattr(extract, "read_document", lambda path, model: {
        "supplier": "", "date": "", "doc_no": "", "total_cbm": None,
        "total_amount_rmb": 4.0 if model == "first" else 4200.0,
        "rows": [dict(r) for r in ([GOOD, misread_amount] if model == "first" else [GOOD, OTHER, OTHER])]})
    doc = extract.read_checked("x.jpg", first="first", second="second")
    assert doc["problems"] == [] and doc["rows"][1]["amount_rmb"] == 2400.0 and len(doc["rows"]) == 2


def test_a_crashed_cold_load_is_retried_once(monkeypatch, tmp_path):
    img = tmp_path / "x.jpg"
    img.write_bytes(b"x")
    calls = []

    class Resp:
        def __init__(self, code):
            self.status_code = code

        def raise_for_status(self):
            if self.status_code >= 400:
                raise extract.requests.HTTPError(str(self.status_code))

        def json(self):
            return {"message": {"content": '{"supplier": "", "date": "", "doc_no": "", "rows": [], '
                                           '"total_amount_rmb": null, "total_cbm": null}'}}

    def post(url, json, timeout):
        calls.append(1)
        return Resp(500 if len(calls) == 1 else 200)

    monkeypatch.setattr(extract.requests, "post", post)
    monkeypatch.setattr(extract.time, "sleep", lambda s: None)
    assert extract.read_document(img, model="m")["rows"] == [] and len(calls) == 2
