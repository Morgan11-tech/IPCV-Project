"""
Game text with real fonts (Pillow), drawn onto OpenCV frames.

OpenCV can only draw its built-in Hershey fonts, so each piece of text is
rendered once by Pillow into a small transparent image (a "sprite") and then
alpha-blended onto the frame. Sprites are cached by (text, font, size, colour),
so a label that stays the same costs only the blend (well under 1 ms).

Fonts (both SIL Open Font License, see assets/fonts/):
    "display"  Lilita One - chunky arcade headings, words, numbers
    "body"     small labels and instructions: Press Start 2P (pixel font),
               sized down by BODY_SCALE so callers can keep Nunito-sized
               numbers (the pixel font is much wider per point)
    "pixel"    Press Start 2P at its real size (title screen)
    "nunito"   Nunito ExtraBold (the former body font)

All text is rendered without anti-aliasing (hard pixel edges, retro look).

Pixel-font text ("body", "pixel") is already pixel art; re-pixelating it in
render/pixelate.py would break its letters. While defer(True) is on, it is
collected instead of drawn, and flush(frame) draws it after that pass.
"""
import os
from collections import OrderedDict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import bootstrap
import theme

FONT_FILES = {"display": ("LilitaOne-Regular.ttf", None),
              "body": ("PressStart2P-Regular.ttf", None),
              "pixel": ("PressStart2P-Regular.ttf", None),
              "nunito": ("Nunito.ttf", "ExtraBold")}
BODY_SCALE = 0.6            # body size 20 -> pixel font 12
PIXEL_SIZES = (8, 12, 16, 24)   # sizes where Press Start 2P's pixels stay even
CACHE_SIZE = 600

_fonts = {}
_sprites = OrderedDict()
_deferred = None            # list of pending pixel-font draws while deferring


def _font(name, size):
    key = (name, size)
    if key not in _fonts:
        filename, variation = FONT_FILES[name]
        font = ImageFont.truetype(os.path.join(bootstrap.ASSETS, "fonts", filename), size)
        if variation:
            font.set_variation_by_name(variation)
        _fonts[key] = font
    return _fonts[key]


def _sprite(s, font, size, color, outline):
    """(BGR image, alpha 0..1, left, top) for this text; left/top are offsets
    from the text's baseline start point."""
    key = (s, font, size, color, outline)
    hit = _sprites.get(key)
    if hit is not None:
        _sprites.move_to_end(key)
        return hit
    if font == "body":
        size = min(PIXEL_SIZES, key=lambda p: abs(p - size * BODY_SCALE))
    f = _font(font, size)
    stroke = max(2, size // 12) if outline else 0
    left, top, right, bottom = f.getbbox(s, anchor="ls", stroke_width=stroke)
    w, h = max(1, right - left), max(1, bottom - top)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    b, g, r = color
    ob, og, orr = theme.OUTLINE
    d = ImageDraw.Draw(img)
    d.fontmode = "1"                                    # no anti-aliasing: hard pixels
    d.text((-left, -top), s, font=f, anchor="ls", fill=(r, g, b, 255),
           stroke_width=stroke, stroke_fill=(orr, og, ob, 255))
    rgba = np.asarray(img, dtype=np.float32)
    hit = (rgba[:, :, 2::-1].copy(), rgba[:, :, 3:4] / 255.0, left, top)
    _sprites[key] = hit
    if len(_sprites) > CACHE_SIZE:
        _sprites.popitem(last=False)
    return hit


def measure(s, size, font="display", outline=True):
    """(width, height) in pixels."""
    bgr, _, _, _ = _sprite(s, font, size, theme.WHITE, outline)
    return bgr.shape[1], bgr.shape[0]


def defer(on):
    """Start (True) or stop (False) collecting pixel-font text instead of drawing it."""
    global _deferred
    _deferred = [] if on else None


def flush(frame):
    """Draw the collected pixel-font text and stop deferring."""
    global _deferred
    pending, _deferred = _deferred or [], None
    for args, kwargs in pending:
        draw(frame, *args, **kwargs)


def draw(frame, s, org, size, color=theme.WHITE, font="display", align="left", outline=True,
         alpha=1.0):
    """Draw text with its baseline at org. align: "left", "center", "right".
    alpha (0..1) fades the whole label. Returns (width, height)."""
    if not s or alpha <= 0:
        return 0, 0
    if _deferred is not None and font in ("body", "pixel"):
        _deferred.append(((s, org, size), dict(color=color, font=font, align=align,
                                                outline=outline, alpha=alpha)))
        bgr, _, _, _ = _sprite(s, font, max(1, int(size)), tuple(int(c) for c in color), outline)
        return bgr.shape[1], bgr.shape[0]
    bgr, mask, left, top = _sprite(s, font, max(1, int(size)), tuple(int(c) for c in color),
                                   outline)
    h, w = mask.shape[:2]
    x, y = org
    x = x + left if align == "left" else x - w / 2 if align == "center" else x - w
    x, y = int(round(x)), int(round(y + top))

    H, W = frame.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return w, h
    sx, sy = x0 - x, y0 - y
    a = mask[sy:sy + y1 - y0, sx:sx + x1 - x0] * alpha
    roi = frame[y0:y1, x0:x1]
    roi[:] = (roi * (1 - a) + bgr[sy:sy + y1 - y0, sx:sx + x1 - x0] * a).astype(np.uint8)
    return w, h
