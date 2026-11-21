"""Shared style for the manuscript figures (print: white canvas, vector, >= 7 pt at 6.75 in).

A 1000-unit-wide viewBox printed at 6.75 in gives 1 unit = 0.486 pt, so 15 units ~ 7.3 pt.
Colours are from BRAND_KIT.md; the darker variants are for strokes and text on white.
"""
import html
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ONYX, SLATE, ORANGE, SAGE = "#0F0F0F", "#2C3E50", "#F39C12", "#87A96B"
ORANGE_DARK = "#B36B00"   # strokes/text on white
SAGE_TEXT = "#4F6B3A"
GREY, LIGHT, BAND = "#6B6B6B", "#EEF1F4", "#E3EBDD"
SANS = "Helvetica, Arial, sans-serif"
MONO = "Menlo, 'DejaVu Sans Mono', 'Courier New', monospace"
MIN_FONT = 15

# One colour per model, used in every results figure. The palette has four brand colours, so the three
# API models (Addendum B) share the sage family and are told apart by marker shape: "col" is the stroke
# and text colour, "fill" the marker fill, "marker" the shape (circle if absent).
MODEL = {
    "qwen3.8-27b-fp8": {"label": "Qwen3.8 27B", "col": SLATE},
    "gemma-4-31b-it": {"label": "Gemma 4 31B", "col": ORANGE_DARK},
    "gpt-6-luna": {"label": "gpt-6-luna", "col": SAGE_TEXT, "fill": SAGE, "marker": "square"},
    "claude-haiku-5.5": {"label": "claude-haiku-5.5", "col": SAGE_TEXT, "fill": SAGE, "marker": "diamond"},
    "gemini-3.5-flash-lite": {"label": "gemini-3.5-flash-lite", "col": SAGE_TEXT, "fill": SAGE, "marker": "triangle"},
}


def esc(s):
    return html.escape(str(s), quote=True)


def text(x, y, s, size=MIN_FONT, fill=ONYX, weight="normal", anchor="start", family=SANS, italic=False, rotate=None):
    assert size >= MIN_FONT - 0.01, f"font {size} below the print minimum"
    extra = " font-style='italic'" if italic else ""
    if rotate is not None:
        extra += f" transform='rotate({rotate} {x:.1f} {y:.1f})'"
    return (f"<text x='{x:.1f}' y='{y:.1f}' font-size='{size}' font-family=\"{family}\" fill='{fill}' "
            f"font-weight='{weight}' text-anchor='{anchor}'{extra}>{esc(s)}</text>")


def line(x1, y1, x2, y2, col=GREY, w=1.5, dash=None, opacity=None):
    d = f" stroke-dasharray='{dash}'" if dash else ""
    o = f" stroke-opacity='{opacity}'" if opacity is not None else ""
    return f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' stroke='{col}' stroke-width='{w}'{d}{o}/>"


def svg_doc(w, h, title, desc, body):
    return "\n".join([f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' viewBox='0 0 {w} {h}' "
                      f"role='img' aria-labelledby='t d'><title id='t'>{esc(title)}</title><desc id='d'>{esc(desc)}</desc>"
                      f"<rect width='{w}' height='{h}' fill='white'/>"] + body + ["</svg>"])


def write(path_svg, doc):
    with open(path_svg, "w") as fh:
        fh.write(doc)
    return path_svg
