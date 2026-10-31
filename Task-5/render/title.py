"""
Title screen artwork, retro pixel style: the SIGN RACE logo over a pixel-art
sunset, with a segmented loading bar underneath.

Everything except the small labels is drawn on a canvas at 1/PIXEL of the
window size and scaled up with nearest-neighbour interpolation, so the whole
picture shares one coarse pixel grid (no smooth edges anywhere).

  * scene   - the sunset landscape from pixel_scene.py (same as the lobby)
  * logo    - "SI" blue + "GN" orange, "RACE" yellow, in the Racing Sans One
              racing font; rendered by Pillow with
              anti-aliasing off at canvas size, so the letters are made of
              hard pixels; white outline (gaps between letters closed with a
              morphological closing, so each word reads as one solid
              sticker) and a navy offset shadow
  * labels  - Press Start 2P pixel font at full size (readable)
The logo is rendered once.
"""
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import bootstrap
import theme

from render.anim import clamp01, ease_out_back, progress
from render.pixel_scene import NAVY, PIXEL, SCENES, PixelScene
from render.text import draw as text

FONT = os.path.join(bootstrap.ASSETS, "fonts", "RacingSansOne-Regular.ttf")   # italic racing style


def _rgb(bgr):
    return (bgr[2], bgr[1], bgr[0], 255)


def _word_sprite(parts, size, shadow_px):
    """parts: [(text, BGR)] side by side, at canvas size with hard pixel edges.
    Layers, back to front: navy shadow, white outline, coloured letters.
    Returns (BGR uint8, bool mask)."""
    font = ImageFont.truetype(FONT, size)
    stroke = max(1, size // 14)
    full = "".join(p for p, _ in parts)
    left, top, right, bottom = font.getbbox(full, anchor="ls", stroke_width=stroke)
    pad = stroke + shadow_px + 2
    w, h = right - left + 2 * pad, bottom - top + 2 * pad
    o = (pad - left, pad - top)

    def layer(draw_fn):
        img = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(img)
        d.fontmode = "1"                      # no anti-aliasing: hard pixels
        draw_fn(d)
        return np.asarray(img) > 127

    outline = layer(lambda d: d.text(o, full, font=font, anchor="ls", fill=255,
                                     stroke_width=stroke, stroke_fill=255))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * stroke + 3, 2 * stroke + 3))
    outline = cv2.morphologyEx(outline.astype(np.uint8), cv2.MORPH_CLOSE, k) > 0
    shadow = np.zeros_like(outline)
    shadow[shadow_px:, shadow_px:] = outline[:-shadow_px, :-shadow_px]

    bgr = np.zeros((h, w, 3), np.uint8)
    bgr[shadow] = NAVY
    bgr[outline] = theme.WHITE
    x = o[0]
    for part, color in parts:
        fill = layer(lambda d: d.text((x, o[1]), part, font=font, anchor="ls", fill=255))
        bgr[fill] = color
        x += font.getlength(part)
    return bgr, outline | shadow


def _blit(canvas, sprite, cx, cy, scale):
    """Paste a sprite centred at (cx, cy), scaled with nearest neighbour."""
    bgr, mask = sprite
    if scale <= 0.05:
        return
    if abs(scale - 1) > 1e-3:
        size = (max(1, int(bgr.shape[1] * scale)), max(1, int(bgr.shape[0] * scale)))
        bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    h, w = mask.shape
    H, W = canvas.shape[:2]
    x, y = int(cx - w / 2), int(cy - h / 2)
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    m = mask[y0 - y:y1 - y, x0 - x:x1 - x]
    canvas[y0:y1, x0:x1][m] = bgr[y0 - y:y1 - y, x0 - x:x1 - x][m]


class TitleScreen:
    def __init__(self):
        self.scene = PixelScene(SCENES["lobby"])
        self.sign = _word_sprite([("SI", theme.player_color(1)), ("GN", theme.player_color(2))],
                                 58, 3)
        self.race = _word_sprite([("RACE", theme.ACCENT)], 30, 2)

    def draw(self, W, H, t, frac, label):
        """One frame: pixel scene, logo, segmented bar at frac (0..1), status label."""
        w, h = W // PIXEL, H // PIXEL
        c = self.scene.render(w, h, t)

        cx, cy = w // 2, int(h * 0.33)
        bob = int(round(1.5 * np.sin(t / 500.0)))            # whole-pixel bob
        _blit(c, self.sign, cx, cy + bob, ease_out_back(progress(t, 0, 650)))
        _blit(c, self.race, cx + 14, cy + self.sign[0].shape[0] * 0.48 + bob,
              ease_out_back(progress(t, 250, 650)))

        a = clamp01((t - 500) / 300)
        bx, by, bw, bh = cx - 70, int(h * 0.80), 140, 7
        if a > 0:                                             # segmented loading bar
            cv2.rectangle(c, (bx - 2, by - 2), (bx + bw + 1, by + bh + 1), theme.WHITE, -1)
            cv2.rectangle(c, (bx - 1, by - 1), (bx + bw, by + bh), NAVY, -1)
            for sx in range(0, int(bw * clamp01(frac)) - 2, 4):
                cv2.rectangle(c, (bx + sx, by), (bx + sx + 2, by + bh - 1), theme.ACCENT, -1)

        frame = cv2.resize(c, (W, H), interpolation=cv2.INTER_NEAREST)
        if a > 0:
            ty = (by + bh + 8) * PIXEL
            text(frame, label.upper(), (bx * PIXEL, ty), 16, theme.WHITE, font="pixel", alpha=a)
            text(frame, f"{int(round(frac * 100))}%", ((bx + bw) * PIXEL, ty), 16, theme.ACCENT,
                 font="pixel", align="right", alpha=a)
        return frame
