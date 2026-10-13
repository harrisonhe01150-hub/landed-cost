"""Settings round-trip and page rendering, without calling the model."""
import json
import os
import tempfile

os.environ["LC_DATA"] = tempfile.mkdtemp()

import app as lc_app  # noqa: E402


def test_settings_round_trip_and_reference_rates_shown():
    c = lc_app.app.test_client()
    r = c.post("/settings", data={"fx": "2.46", "mode": "allin", "allin": "1450", "allin_cur": "CNY", "markup": "1.35",
                                  "round": "1", "vat": "15", "uplift": "10", "allin_rates": "服装 clothing = 3450\nbad line",
                                  "duty_rates": "棉袜 cotton socks = 45%\n橡胶热水袋 rubber hot water bottles = 0%"})
    assert r.status_code == 302
    s = lc_app.load_settings()
    assert s.fx_zar_per_cny == 2.46 and s.allin_rates == {"服装 clothing": 3450.0}
    assert s.duty_rates == {"棉袜 cotton socks": 0.45, "橡胶热水袋 rubber hot water bottles": 0.0}
    page = c.get("/settings").get_data(as_text=True)
    assert "2026-08-28" in page and "4014.90.50" in page and "服装 clothing = 3450" in page


def test_doc_page_renders_results_and_summary():
    folder = lc_app.DOCS / "abc123"
    folder.mkdir(parents=True)
    (folder / "photo.jpg").write_bytes(b"x")
    doc = {"supplier": "样例", "doc_no": "1", "date": "", "photo": "photo.jpg", "model": "m", "read_at": "",
           "rows": [{"sku": "A1", "name": "袜子", "spec": "", "pcs_per_bale": 600, "bales": 1, "qty_pcs": 600,
                     "unit_price_rmb": 1.0, "amount_rmb": 600.0, "carton_cm": "50x40x30", "cbm": None,
                     "cbm_per_bale": 0.06, "problems": [], "category": "服装 clothing"}],
           "total_amount_rmb": 600.0, "total_cbm": None, "problems": [], "rows_total_rmb": 600.0}
    (folder / "doc.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    html = lc_app.app.test_client().get("/doc/abc123").get_data(as_text=True)
    # 1 yuan * 2.46 + 3450 CNY * 2.46 * 0.06 / 600 = 2.46 + 0.8487 = 3.31; x1.35 = 4.47 -> R5 (round to 1)
    assert "3.31" in html and "建议批发 R5.00/件" in html


def test_doc_ids_cannot_escape_the_data_folder():
    assert lc_app.app.test_client().get("/doc/..%2Fsettings").status_code == 404
