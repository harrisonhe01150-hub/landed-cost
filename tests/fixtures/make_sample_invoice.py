"""Render a FAKE Yiwu-style delivery note (送货单) for development and tests.

Everything on it is invented and it is watermarked 样例 / SAMPLE. It is not a real supplier document.
Usage: python make_sample_invoice.py <out_dir>
"""
import sys
from pathlib import Path

from PIL import Image, ImageFilter
from playwright.sync_api import sync_playwright

ROWS = [
    # 款号, 品名, 规格, 件/包, 包数, 单价(元/件), 箱规 cm
    ("SF-320", "珊瑚绒卡通睡衣套装", "M-XXL 混色", 120, 2, 23.5, "60×50×45"),
    ("SF-204", "坑条休闲运动套装", "S-XL 4色", 120, 1, 18.0, "60×45×40"),
    ("W-118", "男士纯棉中筒袜", "均码 黑白灰", 600, 3, 1.35, "55×40×35"),
    ("HB-2L", "2L 加厚橡胶热水袋 带针织套", "2000ml", 100, 2, 6.8, "50×40×38"),
    ("K-77", "高腰收腹打底短裤", "均码 黑肤", 240, 1, 5.2, "50×35×30"),
]

HTML = """<!doctype html><html><head><meta charset=utf-8><style>
body{margin:0;background:#fff;font-family:'Microsoft YaHei','SimSun',sans-serif;color:#111}
.page{width:1000px;padding:28px 34px;position:relative}
h1{text-align:center;font-size:30px;letter-spacing:8px;margin:0 0 6px}
.sub{text-align:center;font-size:14px;margin-bottom:14px}
.meta{display:flex;justify-content:space-between;font-size:15px;margin:6px 0}
table{border-collapse:collapse;width:100%;font-size:15px;margin-top:8px}
td,th{border:1.5px solid #222;padding:7px 6px;text-align:center}
th{background:#f1f1f1}
.foot{display:flex;justify-content:space-between;margin-top:14px;font-size:15px}
.wm{position:absolute;top:300px;left:180px;font-size:120px;color:rgba(200,0,0,.13);transform:rotate(-18deg);font-weight:700}
</style></head><body><div class=page>
<h1>送 货 单</h1><div class=sub>义乌市示例针织服饰商行（虚构样例） · 电话：0579-0000000</div>
<div class=meta><span>客户：Lifong（样例）</span><span>日期：2026年10月09日</span><span>No. 0001234</span></div>
<table><tr><th>款号</th><th>品名</th><th>规格</th><th>件/包</th><th>包数</th><th>数量(件)</th><th>单价(元)</th><th>金额(元)</th><th>箱规(cm)</th></tr>
{rows}
<tr><td colspan=4>合计</td><td>{bales}</td><td>{pcs}</td><td></td><td>{total}</td><td>{cbm}</td></tr></table>
<div class=foot><span>付款方式：现金 / 微信</span><span>送货人：____</span><span>收货人签字：____</span></div>
<div class=wm>样例 SAMPLE</div></div></body></html>"""


def cbm(dims: str) -> float:
    a, b, c = (float(x) for x in dims.split("×"))
    return a * b * c / 1e6


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows, bales, pcs, total, vol = [], 0, 0, 0.0, 0.0
    for sku, name, spec, per, n, price, dims in ROWS:
        q, amt = per * n, per * n * price
        bales, pcs, total, vol = bales + n, pcs + q, total + amt, vol + cbm(dims) * n
        rows.append(f"<tr><td>{sku}</td><td>{name}</td><td>{spec}</td><td>{per}</td><td>{n}</td><td>{q}</td>"
                    f"<td>{price:g}</td><td>{amt:,.2f}</td><td>{dims}</td></tr>")
    html = HTML.replace("{rows}", "\n".join(rows)).replace("{bales}", str(bales)).replace("{pcs}", str(pcs))
    html = html.replace("{total}", f"{total:,.2f}").replace("{cbm}", f"约 {vol:.2f} 方")
    (out / "sample_invoice.html").write_text(html, encoding="utf-8")
    clean = out / "sample_invoice_clean.png"
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        p = b.new_page(viewport={"width": 1000, "height": 700}, device_scale_factor=1.5)
        p.set_content(html)
        p.locator(".page").screenshot(path=str(clean))
        b.close()
    # A rough "phone photo": slight rotation, blur, warm tint, JPEG.
    im = Image.open(clean).convert("RGB")
    im = im.rotate(-2.2, expand=True, fillcolor=(205, 198, 182)).filter(ImageFilter.GaussianBlur(0.8))
    tint = Image.new("RGB", im.size, (255, 240, 215))
    Image.blend(im, tint, 0.12).save(out / "sample_invoice_photo.jpg", quality=72)
    print(f"{len(ROWS)} rows, {bales} bales, {pcs} pcs, ¥{total:,.2f}, {vol:.3f} CBM")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
