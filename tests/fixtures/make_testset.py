"""Generate a SYNTHETIC test set of Yiwu-style delivery notes and packing lists, with ground truth.

Every document is invented and watermarked 样例 / SAMPLE. Used only to measure how well a local vision model
transcribes the kind of paperwork a stall receives; it says nothing about real prices.
Usage: python make_testset.py <out_dir>
"""
import json
import math
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter
from playwright.sync_api import sync_playwright

GOODS = [  # name, spec, pcs per bale choices, price range (yuan/piece), carton dims
    ("珊瑚绒卡通睡衣套装", "M-XXL 混色", (60, 120), (18, 32), "60×50×45"),
    ("坑条休闲运动套装", "S-XL 4色", (100, 120), (14, 26), "60×45×40"),
    ("男士纯棉中筒袜", "均码 黑白灰", (300, 600, 1200), (0.9, 2.2), "55×40×35"),
    ("女士船袜", "均码 混色", (600, 1200), (0.6, 1.4), "50×40×30"),
    ("2L 加厚橡胶热水袋 带针织套", "2000ml", (60, 100), (5, 9), "50×40×38"),
    ("高腰收腹打底短裤", "均码 黑肤", (120, 240), (3.5, 7), "50×35×30"),
    ("加绒连帽外套", "L-4XL", (30, 50), (35, 60), "70×50×50"),
    ("瑜伽健身裤", "S-XL 6色", (100, 200), (8, 15), "55×40×35"),
    ("针织毛线帽", "均码 混色", (240, 480), (2.5, 5), "60×40×40"),
    ("PVC 印花桌布", "137×180cm", (100, 200), (3, 6.5), "50×40×30"),
    ("陶瓷马克杯 带盖勺", "350ml", (48, 72), (4, 8), "45×35×30"),
    ("儿童卡通棉袜", "S-L 5双装", (200, 400), (1.2, 2.8), "50×40×35"),
]

WM = "<div class=wm>样例 SAMPLE</div>"
STYLE = """<style>body{margin:0;background:#fff;color:#111}.page{width:1000px;padding:26px 32px;position:relative}
h1{text-align:center;font-size:30px;letter-spacing:8px;margin:0 0 6px}.sub{text-align:center;font-size:14px;margin-bottom:12px}
.meta{display:flex;justify-content:space-between;font-size:15px;margin:6px 0}table{border-collapse:collapse;width:100%;font-size:15px;margin-top:8px}
td,th{border:1.5px solid #222;padding:7px 5px;text-align:center}th{background:#f1f1f1}
.wm{position:absolute;top:45%;left:16%;font-size:110px;color:rgba(200,0,0,.12);transform:rotate(-18deg);font-weight:700}
.hand td{font-family:'Ink Free','Segoe Print',cursive;font-size:19px;color:#173a8c}.hand td.cn{font-family:KaiTi,'楷体',serif;font-size:19px}
.hand th,.hand h1,.hand .meta,.hand .sub{font-family:KaiTi,'楷体',serif}.hand td,.hand th{border-color:#9a7}</style>"""


def fmt(x: float) -> str:
    return f"{x:,.2f}" if x != int(x) else f"{int(x):,}"


def build(kind: str, rng: random.Random, n_rows: int, hand: bool, with_amount: bool = True):
    picks = rng.sample(GOODS, n_rows)
    rows = []
    for i, (name, spec, pers, (lo, hi), dims) in enumerate(picks):
        per, bales = rng.choice(pers), rng.randint(1, 5)
        price = round(rng.uniform(lo, hi) * 20) / 20  # to 0.05
        a, b, c = (float(x) for x in dims.split("×"))
        sku = f"{rng.choice(['SF', 'W', 'HB', 'K', 'YG', 'MZ', 'TB', 'BZ'])}-{rng.randint(100, 999)}"
        rows.append({"sku": sku, "name": name, "spec": spec, "pcs_per_bale": per, "bales": bales, "qty_pcs": per * bales,
                     "unit_price_rmb": price, "amount_rmb": round(per * bales * price, 2), "carton_cm": dims,
                     "cbm": round(a * b * c / 1e6 * bales, 3)})
    total = round(sum(r["amount_rmb"] for r in rows), 2)
    no = rng.randint(1000, 9999)
    cls = " hand" if hand else ""
    if kind == "delivery":
        head = "<tr><th>款号</th><th>品名</th><th>规格</th><th>件/包</th><th>包数</th><th>数量(件)</th><th>单价(元)</th>" + \
               ("<th>金额(元)</th>" if with_amount else "") + "<th>箱规(cm)</th></tr>"
        body = "".join(
            f"<tr><td>{r['sku']}</td><td class=cn>{r['name']}</td><td class=cn>{r['spec']}</td><td>{r['pcs_per_bale']}</td>"
            f"<td>{r['bales']}</td><td>{r['qty_pcs']}</td><td>{r['unit_price_rmb']:g}</td>"
            + (f"<td>{fmt(r['amount_rmb'])}</td>" if with_amount else "") + f"<td>{r['carton_cm']}</td></tr>" for r in rows)
        # 9 columns with 金额 (the total goes under 金额); 8 without (written across the right-hand cells)
        foot = f"<tr><td colspan=7>合计</td><td>{fmt(total)}</td><td></td></tr>" if with_amount else \
               f"<tr><td colspan=5>合计</td><td colspan=3>{fmt(total)}</td></tr>"
        html = f"""<div class="page{cls}"><h1>送 货 单</h1><div class=sub>义乌示例商行（虚构样例）</div>
<div class=meta><span>客户：样例客户</span><span>日期：2026年10月{rng.randint(1, 12):02d}日</span><span>No. {no}</span></div>
<table>{head}{body}{foot}</table>{WM}</div>"""
        truth_rows = [{k: v for k, v in r.items() if k != "cbm"} | {"cbm": None} for r in rows]
        if not with_amount:
            for r in truth_rows:
                r["amount_rmb"] = None
    else:  # packing list: cartons, pieces per carton, gross weight, volume per row
        head = "<tr><th>箱号</th><th>款号</th><th>品名</th><th>每箱件数</th><th>箱数</th><th>总件数</th><th>单价</th><th>金额</th><th>毛重kg</th><th>体积m³</th></tr>"
        body, box = "", 1
        for r in rows:
            gw = round(r["bales"] * rng.uniform(12, 28), 1)
            body += (f"<tr><td>{box}-{box + r['bales'] - 1}</td><td>{r['sku']}</td><td class=cn>{r['name']}</td><td>{r['pcs_per_bale']}</td>"
                     f"<td>{r['bales']}</td><td>{r['qty_pcs']}</td><td>{r['unit_price_rmb']:g}</td><td>{fmt(r['amount_rmb'])}</td>"
                     f"<td>{gw}</td><td>{r['cbm']:.3f}</td></tr>")
            box += r["bales"]
        tcbm = round(sum(r["cbm"] for r in rows), 3)
        foot = f"<tr><td colspan=7>TOTAL 合计</td><td>{fmt(total)}</td><td></td><td>{tcbm:.3f}</td></tr>"
        html = f"""<div class="page{cls}"><h1>装 箱 单</h1><div class=sub>PACKING LIST · 示例贸易（虚构样例）</div>
<div class=meta><span>INVOICE NO. YW{no}</span><span>DATE: 2026-10-{rng.randint(1, 12):02d}</span></div>
<table>{head}{body}{foot}</table>{WM}</div>"""
        truth_rows = [dict(r, carton_cm="") for r in rows]
    truth = {"rows": truth_rows, "total_amount_rmb": total,
             "total_cbm": round(sum(r["cbm"] for r in rows), 3) if kind == "packing" else None}
    return "<!doctype html><html><head><meta charset=utf-8>" + STYLE + "</head><body>" + html + "</body></html>", truth


def photo_effects(src: Path, dst: Path, rng: random.Random, strength: float) -> None:
    """Rotation, perspective-ish shear, uneven light, blur and JPEG: a rough phone photo of paper on a desk."""
    im = Image.open(src).convert("RGB")
    w, h = im.size
    im = im.rotate(rng.uniform(-3, 3) * strength, expand=True, fillcolor=(196, 188, 170))
    sx = rng.uniform(-0.04, 0.04) * strength
    im = im.transform(im.size, Image.AFFINE, (1, sx, -sx * im.size[1] / 2, 0, 1, 0), fillcolor=(196, 188, 170))
    shade = Image.new("L", im.size, 0)
    d = ImageDraw.Draw(shade)
    cx, cy = rng.uniform(0, im.size[0]), rng.uniform(0, im.size[1])
    for r in range(0, int(math.hypot(*im.size)), 40):
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=min(255, int(r / 12 * strength)), width=40)
    im = Image.composite(Image.new("RGB", im.size, (60, 50, 40)), im, shade.point(lambda v: int(v * 0.35)))
    im = im.filter(ImageFilter.GaussianBlur(0.6 + 0.6 * strength))
    im.save(dst, quality=int(80 - 15 * strength))


CASES = [  # name, kind, rows, handwritten, with_amount, photo strength (0 = clean scan)
    ("t01_delivery_clean", "delivery", 5, False, True, 0),
    ("t02_delivery_photo", "delivery", 6, False, True, 1.0),
    ("t03_packing_clean", "packing", 6, False, True, 0),
    ("t04_packing_photo", "packing", 7, False, True, 1.0),
    ("t05_hand_delivery", "delivery", 5, True, True, 0.6),
    ("t06_hand_delivery_photo", "delivery", 6, True, True, 1.2),
    ("t07_no_amount_column", "delivery", 5, False, False, 0.8),
    ("t08_long_list", "delivery", 12, False, True, 0.8),
]


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20261013)
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        page = b.new_page(viewport={"width": 1000, "height": 700}, device_scale_factor=1.5)
        for name, kind, n, hand, amt, strength in CASES:
            html, truth = build(kind, rng, n, hand, amt)
            page.set_content(html)
            clean = out / f"{name}.png"
            page.locator(".page").screenshot(path=str(clean))
            img = clean
            if strength:
                img = out / f"{name}.jpg"
                photo_effects(clean, img, rng, strength)
                clean.unlink()
            (out / f"{name}.truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=1), encoding="utf-8")
            print(name, img.name, len(truth["rows"]), "rows")
        b.close()


if __name__ == "__main__":
    main(Path(sys.argv[1]))
