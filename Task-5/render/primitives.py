"""
Drawing building blocks shared by every screen: outlined text, translucent
rounded panels, progress bars and hearts.

Translucent panels blend only the panel's own region (not the whole frame),
which keeps each one well under a millisecond at 1280x720.
"""
import cv2
import numpy as np

import theme

FONT = cv2.FONT_HERSHEY_DUPLEX


def text_size(s, scale, thick=2):
    (w, h), _ = cv2.getTextSize(s, FONT, scale, thick)
    return w, h


def text(img, s, org, scale=0.8, color=theme.WHITE, thick=2, align="left", outline=True):
    """Draw text at org (baseline). align: "left", "center" or "right"."""
    w, h = text_size(s, scale, thick)
    x, y = org
    if align == "center":
        x -= w / 2
    elif align == "right":
        x -= w
    x, y = int(x), int(y)
    if outline:
        cv2.putText(img, s, (x, y), FONT, scale, theme.OUTLINE, thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), FONT, scale, color, thick, cv2.LINE_AA)
    return w, h


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


def panel(img, x, y, w, h, fill=theme.PANEL, alpha=theme.PANEL_ALPHA, border=None,
          border_thick=2, radius=14):
    """Translucent rounded panel, optionally with a coloured border."""
    H, W = img.shape[:2]
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(W, int(x + w)), min(H, int(y + h))
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    layer = roi.copy()
    rounded_rect(layer, x - x0, y - y0, x + w - x0 - 1, y + h - y0 - 1, fill, -1, radius)
    cv2.addWeighted(layer, alpha, roi, 1 - alpha, 0, dst=roi)
    if border is not None:
        rounded_rect(img, x, y, x + w - 1, y + h - 1, border, border_thick, radius)


def bar(img, x, y, w, h, frac, color, back=theme.TRACK):
    """Horizontal progress bar, frac in 0..1."""
    frac = min(1.0, max(0.0, frac))
    rounded_rect(img, x, y, x + w, y + h, back, -1, h // 2)
    if frac > 0:
        rounded_rect(img, x, y, x + max(h, int(w * frac)), y + h, color, -1, h // 2)


def heart(img, cx, cy, size, color):
    """Filled heart centred on (cx, cy), about `size` px wide."""
    cx, cy, r = int(cx), int(cy), max(2, int(size * 0.27))
    cv2.circle(img, (cx - r, cy - r // 2), r, color, -1, cv2.LINE_AA)
    cv2.circle(img, (cx + r, cy - r // 2), r, color, -1, cv2.LINE_AA)
    pts = [[cx - 2 * r, cy - r // 6], [cx + 2 * r, cy - r // 6], [cx, cy + int(1.9 * r)]]
    cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], color, cv2.LINE_AA)
