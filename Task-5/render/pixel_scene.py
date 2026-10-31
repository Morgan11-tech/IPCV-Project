"""
Pixel-art landscape used by the title screen and as the in-game arena.

Drawn on a small canvas (1/PIXEL of the window) and scaled up with
nearest-neighbour interpolation by the caller, so it has hard, coarse pixels.

  * sky     - colour bands blended with ordered (Bayer 4x4) dithering, the
              classic way to fake a gradient with few colours
  * sun     - vertical colour ramp, retro stripes in its lower half (optional)
  * hills   - two ranges from random-walk ridges, the far one with a rim light
  * lake    - darkens towards the camera; the sun's reflection shimmers on it
  * extras  - drifting blocky clouds, twinkling stars (optional)

One palette per scene (SCENES); the game-over palette is built from the
winner's colour. The static part is rendered once per palette and size.
"""
import cv2
import numpy as np

import theme

PIXEL = 4                                   # one art pixel = 4x4 screen pixels
HORIZON = 0.66
NAVY = (62, 26, 24)                         # BGR
BAYER = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32) / 16

SUNSET = {      # title screen and lobby: the logo's colours
    "sky": [NAVY, (120, 48, 30), (185, 95, 40), theme.player_color(1), (160, 80, 125),
            (120, 70, 205), theme.player_color(2), (70, 180, 255)],
    "sun": (theme.ACCENT, (30, 120, 250)), "stripes": True,
    "far": (105, 45, 80), "rim": (40, 120, 250), "near": (70, 28, 42),
    "water": ((95, 42, 62), (48, 20, 30)), "cloud": ((235, 215, 250), (190, 150, 225)),
    "stars": True,
}
SCENES = {
    "lobby": SUNSET,
    1: {        # calm daytime
        "sky": [(190, 120, 50), (210, 145, 70), (225, 170, 95), (235, 195, 130), (235, 220, 175)],
        "sun": ((90, 235, 255), (40, 210, 255)), "stripes": False,
        "far": (120, 150, 80), "rim": (170, 220, 160), "near": (60, 135, 55),
        "water": ((185, 135, 60), (120, 80, 35)), "cloud": ((255, 252, 250), (225, 205, 200)),
        "stars": False,
    },
    2: {        # golden dusk (amber level colour)
        "sky": [(110, 45, 40), (120, 60, 110), (90, 90, 200), (50, 140, 245), (60, 190, 255),
                (120, 225, 255)],
        "sun": ((150, 245, 255), (30, 160, 255)), "stripes": True,
        "far": (80, 50, 110), "rim": (60, 200, 255), "near": (50, 30, 70),
        "water": ((90, 60, 110), (40, 22, 40)), "cloud": ((170, 200, 255), (110, 140, 230)),
        "stars": False,
    },
    3: {        # red night (danger)
        "sky": [(40, 10, 25), (50, 15, 70), (50, 20, 120), (50, 30, 170), (40, 50, 220),
                (60, 110, 250)],
        "sun": ((60, 80, 250), (30, 30, 200)), "stripes": True,
        "far": (40, 15, 60), "rim": (60, 90, 255), "near": (25, 10, 35),
        "water": ((50, 20, 60), (20, 8, 18)), "cloud": ((120, 90, 200), (80, 60, 150)),
        "stars": True,
    },
}
SPEED = {"lobby": 1.0, 1: 1.0, 2: 1.8, 3: 3.0, "game_over": 1.2}   # clouds and shimmer


def winner_palette(color):
    """Night sky fading into the winner's colour (game over)."""
    c = np.array(color, np.float32)
    navy = np.array(NAVY, np.float32)
    sky = [tuple(int(v) for v in navy + (c - navy) * k) for k in (0, 0.2, 0.4, 0.6, 0.8, 1.0)]
    sky.append(tuple(int(v) for v in np.minimum(255, c * 1.3 + 30)))
    return dict(SUNSET, sky=sky, rim=tuple(int(v) for v in np.minimum(255, c * 1.2 + 40)))


def palette_for(scene, winner=None):
    if scene == "game_over":
        return winner_palette(theme.player_color(winner)) if winner else SUNSET
    return SCENES.get(scene, SUNSET)


def _dithered(h, w, colors):
    """Vertical gradient through `colors` with ordered dithering between bands."""
    ys, xs = np.mgrid[0:h, 0:w]
    f = ys / max(h - 1, 1) * (len(colors) - 1)
    idx = np.floor(f).astype(int)
    idx += ((f - idx) > BAYER[ys % 4, xs % 4]).astype(int)
    return np.array(colors, np.uint8)[np.clip(idx, 0, len(colors) - 1)]


def _ridge(w, base, rough, rng):
    """Mountain ridge heights: a random walk, lightly smoothed."""
    y = np.cumsum(rng.normal(0, rough, w + 8))
    y = np.convolve(y - y.mean(), np.ones(9) / 9, mode="valid")[:w]
    return (base + y).astype(int)


def _render_static(w, h, pal, seed=11):
    hz = int(h * HORIZON)
    img = np.empty((h, w, 3), np.uint8)
    img[:hz] = _dithered(hz, w, pal["sky"])
    img[hz:] = _dithered(h - hz, w, list(pal["water"]))

    cx, r = w // 2, int(h * 0.20)
    cy = hz - int(r * 0.35)
    top, bottom = pal["sun"]
    for y in range(cy - r, cy + r + 1):
        u = (y - (cy - r)) / (2 * r)
        if pal["stripes"] and u > 0.5 and int((u - 0.5) * 20) % 3 == 2:
            continue                          # stripe gap
        half = int(np.sqrt(max(0, r * r - (y - cy) ** 2)))
        if 0 <= y < hz:
            img[y, max(0, cx - half):cx + half + 1] = [int(a + (b - a) * u) for a, b in zip(top, bottom)]

    rng = np.random.default_rng(seed)
    far = _ridge(w, hz - int(h * 0.10), 1.6, rng)
    near = _ridge(w, hz - int(h * 0.03), 1.2, rng)
    for x in range(w):
        img[max(0, far[x]):hz, x] = pal["far"]
        img[max(0, far[x]), x] = pal["rim"]
        img[max(0, near[x]):hz, x] = pal["near"]
    return img


class PixelScene:
    def __init__(self, palette, speed=1.0):
        self.pal, self.speed = palette, speed
        self._static, self._size = None, None
        rng = np.random.default_rng(5)
        self._clouds = rng.random((6, 4))     # x, y, speed, size
        self._stars = rng.random((40, 3))     # x, y, twinkle phase

    def render(self, w, h, t):
        """The canvas (h, w) for time t (ms)."""
        if self._size != (w, h):
            self._static, self._size = _render_static(w, h, self.pal), (w, h)
        c = self._static.copy()
        hz = int(h * HORIZON)
        if self.pal["stars"]:
            for x, y, ph in self._stars:
                if (t / 700.0 + ph * 7) % 3 > 0.6:   # twinkle: briefly off
                    c[int(y * hz * 0.35), int(x * w)] = theme.WHITE
        cloud, shade = self.pal["cloud"]
        for x, y, s, size in self._clouds:
            cw = int(14 + size * 18)
            px = int((x * (w + cw) + t / 1000.0 * (1 + 3 * s) * self.speed) % (w + cw)) - cw
            py = int(8 + y * hz * 0.45)
            for dx, dy, bw in ((0, 2, cw), (cw // 5, 0, cw // 2), (cw // 2, -2, cw // 3)):
                cv2.rectangle(c, (px + dx, py + dy), (px + dx + bw, py + dy + 3), cloud, -1)
            cv2.rectangle(c, (px, py + 5), (px + cw, py + 5), shade, -1)
        top, bottom = self.pal["sun"]                # reflection on the lake
        for k, y in enumerate(range(hz + 2, int(hz + (h - hz) * 0.36), 2)):
            depth = (y - hz) / (h - hz)
            half = int(w * 0.11 * (1 - depth * 1.6)) + 2
            shift = int(round(2.5 * np.sin(t / 350.0 * self.speed + k * 2.1)))
            cv2.line(c, (w // 2 - half + shift, y), (w // 2 + half + shift, y),
                     top if k % 3 else bottom, 1)
        return c
