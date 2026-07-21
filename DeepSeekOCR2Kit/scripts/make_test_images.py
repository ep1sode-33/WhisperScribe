#!/usr/bin/env python3
"""Deterministic test images for parity fixtures. PIL only."""
import pathlib
from PIL import Image, ImageDraw, ImageFont

OUT = pathlib.Path(__file__).parent.parent / "Fixtures" / "images"
OUT.mkdir(parents=True, exist_ok=True)

# Font availability varies by macOS build/locale pack. Try a few known-good
# system paths in order and fall back to PIL's default bitmap font (which
# renders CJK as tofu, so real CJK coverage depends on one of the .ttc paths
# below existing on the host).
LATIN_FONT_CANDIDATES = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/HelveticaNeue.ttc",
]
CJK_FONT_CANDIDATES = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
]


def _first_existing(paths):
    for p in paths:
        if pathlib.Path(p).exists():
            return p
    return None


def font(size, cjk=False):
    candidates = CJK_FONT_CANDIDATES if cjk else LATIN_FONT_CANDIDATES
    path = _first_existing(candidates)
    if path is None:
        return ImageFont.load_default()
    return ImageFont.truetype(path, size)


def page(name, size, lines, cjk=False, fsize=28, margin=60, spacing=14):
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    y = margin
    for ln in lines:
        d.text((margin, y), ln, fill="black", font=font(fsize, cjk))
        y += fsize + spacing
    img.save(OUT / f"{name}.png")


page("doc_page", (1200, 1600), [
    "DeepSeekOCR2Kit Parity Test Document",
    "", "1. Introduction",
    "This fixed document verifies OCR output parity",
    "between the Python reference and the Swift port.",
    "", "2. Requirements",
    "- Deterministic rendering", "- No external assets",
    "The quick brown fox jumps over the lazy dog. 0123456789.",
])
page("receipt", (600, 900), [
    "RECEIPT #004217", "----------------",
    "Espresso        4.50", "Croissant       3.25",
    "Total           7.75", "VISA ****1234  PAID",
], fsize=24)
page("cjk_dense", (1000, 1400), [
    "语音转写与光学识别的本地化验证文档",
    "第一段：本页用于验证中日韩文字的识别一致性，",
    "包含标点、数字 3.1415、以及混排 English words。",
    "第二段：滚动截图、书页拍照、演示文稿截屏等场景",
    "均应产生逐字节一致的贪心解码序列。",
], cjk=True)
page("tall_scroll", (700, 2800),
     [f"第 {i} 行：这是一张模拟滚动截图的超长图片。Line {i}." for i in range(1, 60)],
     cjk=True, fsize=24)
page("grounding_menu", (900, 1200), [
    "MENU", "Latte ......... 5.00", "Mocha ......... 5.50",
    "Tea ........... 3.00", "Find the price of Mocha.",
], fsize=32)
print("wrote", sorted(p.name for p in OUT.iterdir()))
