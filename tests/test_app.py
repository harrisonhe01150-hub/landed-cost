"""Page behaviour, without calling the model. Each test that names a review finding failed before its fix."""
import io
import json
import os
import tempfile

import pytest
import requests

os.environ["LC_DATA"] = tempfile.mkdtemp()

import app as lc_app  # noqa: E402
from lc import classify, extract  # noqa: E402

DOC_ID = "abc1234567"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(lc_app, "DATA", tmp_path)
    monkeypatch.setattr(lc_app, "DOCS", tmp_path / "docs")
    monkeypatch.setattr(lc_app, "SETTINGS", tmp_path / "settings.json")
    return lc_app.app.test_client()


def settings(c, **kw):
    data = {"fx": "2.46", "mode": "allin", "allin": "1450", "allin_cur": "CNY", "markup": "1.35", "round": "1",
            "vat": "15", "uplift": "10", **kw}
    return c.post("/settings", data=data)


def put_doc(rows, **doc):
    folder = lc_app.DOCS / DOC_ID
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "photo.jpg").write_bytes(b"x")
    d = {"supplier": "样例", "doc_no": "1", "date": "", "photo": "photo.jpg", "model": "m", "read_at": "",
         "total_amount_rmb": sum(r["amount_rmb"] or 0 for r in rows), "total_cbm": None, **doc}
    d["rows"] = [dict(r) for r in rows]
    lc_app.save_json(folder / "doc.json", extract.check(d))
    return folder / "doc.json"


SOCKS = {"sku": "A1", "name": "袜子", "spec": "", "pcs_per_bale": 600, "bales": 1, "qty_pcs": 600, "unit_price_rmb": 1.0,
         "amount_rmb": 600.0, "carton_cm": "50x40x30", "cbm": None, "carton_no": "", "unit": "", "category": "服装 clothing"}


def form_of(doc):
    """What the browser would post back unchanged."""
    f = {"n": str(len(doc["rows"]) + 2), "total": str(doc["total_amount_rmb"] or ""), "total_cbm": str(doc["total_cbm"] or "")}
    keys = {"sku": "sku", "name": "name", "spec": "spec", "per": "pcs_per_bale", "bales": "bales", "qty": "qty_pcs",
            "price": "unit_price_rmb", "amount": "amount_rmb", "carton": "carton_cm", "cbm": "cbm", "unit": "unit",
            "cat": "category"}
    for i, r in enumerate(doc["rows"]):
        for form, key in keys.items():
            v = r.get(key)
            f[f"{form}{i}"] = "" if v is None else str(v)
    return f


def test_settings_round_trip_and_reference_rates_shown(client):
    assert settings(client, allin_rates="服装 clothing = 3450", duty_rates="棉袜 cotton socks = 45%").status_code == 302
    s = lc_app.load_settings()
    assert s.fx_zar_per_cny == 2.46 and s.allin_rates == {"服装 clothing": 3450.0} and s.duty_rates == {"棉袜 cotton socks": 0.45}
    page = client.get("/settings").get_data(as_text=True)
    assert "2026-08-28" in page and "4014.90.50" in page and "服装 clothing = 3450" in page


@pytest.mark.parametrize("typed,expected", [("2,46", 2.46), ("2.46", 2.46), ("２．４６", 2.46), ("R2.46", 2.46)])
def test_settings_accept_how_people_actually_type_numbers(client, typed, expected):  # review C13/C22
    assert settings(client, fx=typed).status_code == 302
    assert lc_app.load_settings().fx_zar_per_cny == expected


def test_settings_parse_rand_with_spaces_and_full_width_symbols(client):
    assert settings(client, allin="1 450", allin_rates="服装＝￥3,450", duty_rates="棉袜 = 45％").status_code == 302
    s = lc_app.load_settings()
    assert s.allin_rate_per_cbm == 1450 and s.allin_rates == {"服装": 3450.0} and s.duty_rates == {"棉袜": 0.45}


@pytest.mark.parametrize("field,typed", [("fx", "24.6"), ("fx", "abc"), ("markup", "30"), ("allin", "inf")])
def test_settings_reject_impossible_values_instead_of_saving_them(client, field, typed):  # review C22, low inf
    r = settings(client, **{field: typed})
    assert r.status_code == 200 and "没保存" in r.get_data(as_text=True) and not lc_app.SETTINGS.exists()


def test_settings_can_store_zero_vat_and_bad_table_lines_are_reported(client):  # review low
    r = settings(client, vat="0", duty_rates="棉袜 45%\n化纤袜 = -5%")
    assert r.status_code == 200 and "少了 =" in r.get_data(as_text=True)
    assert settings(client, vat="0").status_code == 302 and lc_app.load_settings().vat_rate == 0


def test_saving_without_changes_keeps_printed_row_volumes(client):  # review C2/C8/C15
    settings(client)
    p = put_doc([dict(SOCKS, carton_cm="", cbm=0.9), dict(SOCKS, sku="B2", carton_cm="", cbm=0.1)])
    before = client.get(f"/doc/{DOC_ID}").get_data(as_text=True)
    doc = json.loads(p.read_text(encoding="utf-8"))
    assert client.post(f"/doc/{DOC_ID}", data=form_of(doc)).status_code == 302
    after = json.loads(p.read_text(encoding="utf-8"))
    assert [r["cbm"] for r in after["rows"]] == [0.9, 0.1]
    assert client.get(f"/doc/{DOC_ID}").get_data(as_text=True).count("建议批发") == before.count("建议批发")


def test_blank_rows_let_the_owner_add_a_missed_line(client):  # review C25
    settings(client)
    p = put_doc([SOCKS], total_amount_rmb=1200.0)
    f = form_of(json.loads(p.read_text(encoding="utf-8")))
    f.update({"sku1": "C3", "name1": "帽子", "per1": "600", "bales1": "1", "qty1": "600", "price1": "1", "amount1": "600",
              "carton1": "50x40x30"})
    client.post(f"/doc/{DOC_ID}", data=f)
    doc = json.loads(p.read_text(encoding="utf-8"))
    assert [r["sku"] for r in doc["rows"]] == ["A1", "C3"] and doc["problems"] == []


def test_doc_page_renders_results_and_the_summary_carries_every_warning(client):  # review C16
    settings(client)
    put_doc([SOCKS, dict(SOCKS, sku="B2", unit_price_rmb=None, amount_rmb=None)], total_amount_rmb=900.0)
    html = client.get(f"/doc/{DOC_ID}").get_data(as_text=True)
    text = html.split("<textarea id=t rows=9 readonly>")[1].split("</textarea>")[0]
    assert "⚠️ 还有" in text and "没算价" in text and "待核对" in text
    # 1 yuan * 2.46 + 1450 CNY * 2.46 * 0.06 / 600 = 2.82; x1.35 = 3.80 -> R4 (round to 1)
    assert "建议批发 R4.00/件" in text


def test_missing_settings_block_prices_and_say_what_to_fill(client):  # review C18
    settings(client, allin="")
    put_doc([SOCKS])
    assert "还没填：包税价/方" in client.get("/").get_data(as_text=True)
    assert "建议批发" not in client.get(f"/doc/{DOC_ID}").get_data(as_text=True)


def test_reading_with_ollama_down_explains_and_leaves_no_orphan_folder(client, monkeypatch):  # review C10/C20
    def down(path):
        raise requests.ConnectionError("refused")
    monkeypatch.setattr(extract, "read_checked", down)
    r = client.post("/read", data={"photo": (io.BytesIO(b"img"), "a.jpg")}, content_type="multipart/form-data")
    assert r.status_code == 302 and "error=" in r.headers["Location"]
    assert "Ollama" in client.get(r.headers["Location"]).get_data(as_text=True)
    assert not any(lc_app.DOCS.glob("*")) if lc_app.DOCS.exists() else True


def test_resort_with_the_model_down_keeps_the_owners_categories(client, monkeypatch):  # review C9
    settings(client, allin_rates="服装 clothing = 3450")
    p = put_doc([SOCKS])
    monkeypatch.setattr(classify, "categories", lambda names, cats: None)
    r = client.post(f"/doc/{DOC_ID}/classify")
    assert "model_down" in r.headers["Location"]
    assert json.loads(p.read_text(encoding="utf-8"))["rows"][0]["category"] == "服装 clothing"


def test_cross_site_posts_are_refused_and_next_cannot_leave_the_app(client):  # review C12
    r = client.post("/settings", data={"fx": "24.6"}, headers={"Origin": "https://evil.example"})
    assert r.status_code == 403
    r = client.post("/settings?next=https://evil.example/x", data={"fx": "2.46", "mode": "allin", "allin": "1450",
                                                                    "markup": "1.3", "round": "1", "vat": "15", "uplift": "10"})
    assert r.headers["Location"] == "/"
    assert client.get("/", headers={"Host": "attacker.example:5060"}).status_code == 400


def test_corrupt_settings_file_does_not_lock_the_owner_out(client):  # review C11
    lc_app.SETTINGS.write_text('{"fx_zar_per_cny": 2.4', encoding="utf-8")
    assert client.get("/").status_code == 200 and client.get("/settings").status_code == 200


def test_unknown_docs_are_404_and_ids_cannot_escape_the_data_folder(client):  # review low
    assert client.get("/doc/..%2Fsettings").status_code == 404
    assert client.get("/doc/abcdef0123/photo").status_code == 404
    assert client.post("/doc/abcdef0123/classify").status_code == 404


def test_uploads_are_only_ever_served_as_images(client, monkeypatch):  # review low
    monkeypatch.setattr(extract, "read_checked", lambda p: extract.check({"rows": [], "total_amount_rmb": None}))
    monkeypatch.setattr(lc_app, "sort_categories", lambda d, s: True)
    r = client.post("/read", data={"photo": (io.BytesIO(b"<script>"), "x.html")}, content_type="multipart/form-data")
    doc_id = r.headers["Location"].rsplit("/", 1)[1]
    assert json.loads((lc_app.DOCS / doc_id / "doc.json").read_text(encoding="utf-8"))["photo"] == "photo.jpg"


def test_default_port_is_one_browsers_will_open():
    # Chrome/Firefox refuse these "unsafe" ports outright (ERR_UNSAFE_PORT); 5060 was the first default.
    blocked = {1, 7, 9, 11, 13, 15, 17, 19, 20, 21, 22, 23, 25, 37, 42, 43, 53, 69, 77, 79, 87, 95, 101, 102, 103, 104,
               109, 110, 111, 113, 115, 117, 119, 123, 135, 137, 139, 143, 161, 179, 389, 427, 465, 512, 513, 514, 515,
               526, 530, 531, 532, 540, 548, 554, 556, 563, 587, 601, 636, 989, 990, 993, 995, 1719, 1720, 1723, 2049,
               3659, 4045, 4190, 5060, 5061, 6000, 6566, 6665, 6666, 6667, 6668, 6669, 6679, 6697, 10080}
    assert lc_app.PORT not in blocked
