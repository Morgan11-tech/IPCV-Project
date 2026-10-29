"""
Drawing building blocks shared by every screen: translucent rounded panels
with a soft drop shadow, progress bars and hearts. Text lives in render/text.py.

Translucent shapes blend only their own region (not the whole frame), which
keeps each one well under a millisecond at 1280x720.
"""
import cv2
import numpy as np

import theme

SHADOW_OFFSET = 5
SHADOW_ALPHA = 0.35


def rounded_rect(img, x0, y0, x1, y1, color, thick=-1, radius=12):
    """Rectangle with rounded corners; thick=-1 fills it."""
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    r = int(max(0, min(radius, (x1 - x0) // 2, (y1 - y0) // 2)))
    if thick < 0:
        cv2.rectangle(img, (x0 + r, y0), (x1 - r, y1), color, -1)
        cv2.rectangle(img, (x0, y0 + r), (x1, y1 - r), color, -1)
        for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
            cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
        return
    cv2.line(img, (x0 + r, y0), (x1 - r, y0), color, thick, cv2.LINE_AA)
    cv2.line(img, (x0 + r, y1), (x1 - r, y1), color, thick, cv2.LINE_AA)
    cv2.line(img, (x0, y0 + r), (x0, y1 - r), color, thick, cv2.LINE_AA)
    cv2.line(img, (x1, y0 + r), (x1, y1 - r), color, thick, cv2.LINE_AA)
    for (cx, cy), angle in (((x0 + r, y0 + r), 180), ((x1 - r, y0 + r), 270),
                            ((x1 - r, y1 - r), 0), ((x0 + r, y1 - r), 90)):
        cv2.ellipse(img, (cx, cy), (r, r), angle, 0, 90, color, thick, cv2.LINE_AA)


def blend_rounded_rect(img, x, y, w, h, color, alpha, radius=14):
    """Filled rounded rectangle mixed with what is underneath (alpha 0..1)."""
    H, W = img.shape[:2]
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(W, int(x + w)), min(H, int(y + h))
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    layer = roi.copy()
    rounded_rect(layer, x - x0, y - y0, x + w - x0 - 1, y + h - y0 - 1, color, -1, radius)
    cv2.addWeighted(layer, alpha, roi, 1 - alpha, 0, dst=roi)


def panel(img, x, y, w, h, fill=theme.PANEL, alpha=theme.PANEL_ALPHA, border=None,
          border_thick=2, radius=14, shadow=True):
    """Translucent rounded panel with a drop shadow and optional coloured border."""
    if shadow:
        blend_rounded_rect(img, x, y + SHADOW_OFFSET, w, h, theme.OUTLINE, SHADOW_ALPHA, radius)
    blend_rounded_rect(img, x, y, w, h, fill, alpha, radius)
    if border is not None:
        rounded_rect(img, x, y, x + w - 1, y + h - 1, border, border_thick, radius)


def bar(img, x, y, w, h, frac, color, back=theme.TRACK):
    """Horizontal progress bar, frac in 0..1."""
    frac = min(1.0, max(0.0, frac))
    rounded_rect(img, x, y, x + w, y + h, back, -1, h // 2)
    if frac > 0:
        rounded_rect(img, x, y, x + max(h, int(w * frac)), y + h, color, -1, h // 2)


def heart(img, cx, cy, size, color, outline=True):
    """Filled heart centred on (cx, cy), about `size` px wide."""
    cx, cy, r = int(cx), int(cy), max(2, int(size * 0.27))
    pts = np.array([[cx - 2 * r, cy - r // 6], [cx + 2 * r, cy - r // 6], [cx, cy + int(1.9 * r)]],
                   dtype=np.int32)
    if outline:
        for dx in (-r, r):
            cv2.circle(img, (cx + dx, cy - r // 2), r + 2, theme.OUTLINE, -1, cv2.LINE_AA)
        cv2.polylines(img, [pts], True, theme.OUTLINE, 4, cv2.LINE_AA)
    for dx in (-r, r):
        cv2.circle(img, (cx + dx, cy - r // 2), r, color, -1, cv2.LINE_AA)
    cv2.fillPoly(img, [pts], color, cv2.LINE_AA)
