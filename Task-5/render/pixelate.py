"""
Retro look for everything drawn over the camera, in one pass.

The renderer keeps a copy of the frame before the overlay (face rings, name
tags, particles, effects, HUD, game-over screen) is drawn. Afterwards:

    1. changed  = |after - before| > THRESH         which pixels the overlay touched
    2. blocks   = sample `after` every PX pixels     one colour per PX x PX block
    3. coverage = share of each block that changed   (area resize of the mask)
    4. blocks with coverage > MIN_COVER are pasted back, scaled up with
       nearest neighbour; thin anti-aliased fringes (low coverage) are dropped

So the whole overlay sits on one coarse pixel grid with hard edges, while
the players (the camera picture under it) stay sharp.
"""
import cv2
import numpy as np

PX = 2              # one overlay pixel = 2x2 screen pixels
THRESH = 6          # grey-level change that counts as "drawn on"
MIN_COVER = 0.35    # keep a block if at least this share of it was drawn on


def pixelate_overlay(before, after, px=PX):
    """Return `after` with the pixels changed since `before` turned into blocks."""
    H, W = after.shape[:2]
    h, w = H // px, W // px
    if h == 0 or w == 0:
        return after
    Hc, Wc = h * px, w * px
    changed = cv2.cvtColor(cv2.absdiff(after[:Hc, :Wc], before[:Hc, :Wc]), cv2.COLOR_BGR2GRAY)
    cover = cv2.resize((changed > THRESH).astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    keep = cv2.resize((cover > MIN_COVER).astype(np.uint8), (Wc, Hc), interpolation=cv2.INTER_NEAREST)
    blocks = cv2.resize(after[px // 2:Hc:px, px // 2:Wc:px], (Wc, Hc), interpolation=cv2.INTER_NEAREST)
    out = before.copy()
    cv2.copyTo(blocks, keep, out[:Hc, :Wc])
    if Hc < H or Wc < W:                       # leftover edge rows/columns: unchanged overlay
        out[Hc:], out[:, Wc:] = after[Hc:], after[:, Wc:]
    return out
