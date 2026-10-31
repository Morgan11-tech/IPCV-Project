"""
Particles: bursts of sparks when a player scores and falling confetti at game
over. All particles live in NumPy arrays and move together each frame
(velocity + gravity + drag), so a few hundred cost well under a millisecond.
"""
import cv2
import numpy as np

import theme

MAX_PARTICLES = 600
GRAVITY = 900.0          # px / s^2
DRAG = 1.6               # 1 / s, slows sparks down


class Particles:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.pos = np.zeros((0, 2), np.float32)
        self.vel = np.zeros((0, 2), np.float32)
        self.life = np.zeros(0, np.float32)        # seconds left
        self.max_life = np.zeros(0, np.float32)
        self.size = np.zeros(0, np.float32)
        self.color = np.zeros((0, 3), np.uint8)
        self.gravity = np.zeros(0, np.float32)     # per particle (confetti falls slower)
        self._last_t = None

    def _add(self, pos, vel, life, size, colors, gravity):
        n = len(pos)
        idx = self.rng.integers(0, len(colors), n)
        self.pos = np.vstack([self.pos, pos]).astype(np.float32)
        self.vel = np.vstack([self.vel, vel]).astype(np.float32)
        self.life = np.concatenate([self.life, life]).astype(np.float32)
        self.max_life = np.concatenate([self.max_life, life]).astype(np.float32)
        self.size = np.concatenate([self.size, size]).astype(np.float32)
        self.color = np.vstack([self.color, np.array(colors, np.uint8)[idx]])
        self.gravity = np.concatenate([self.gravity, np.full(n, gravity, np.float32)])
        if len(self.life) > MAX_PARTICLES:                 # drop the oldest
            keep = slice(len(self.life) - MAX_PARTICLES, None)
            for name in ("pos", "vel", "life", "max_life", "size", "color", "gravity"):
                setattr(self, name, getattr(self, name)[keep])

    def burst(self, center, colors, n=40, speed=520.0):
        """Sparks flying out from `center` in all directions."""
        ang = self.rng.uniform(0, 2 * np.pi, n)
        spd = self.rng.uniform(0.35, 1.0, n) * speed
        vel = np.stack([np.cos(ang) * spd, np.sin(ang) * spd - 0.3 * speed], axis=1)
        pos = np.repeat(np.array([center], np.float32), n, axis=0)
        self._add(pos, vel, self.rng.uniform(0.5, 1.0, n), self.rng.uniform(3, 7, n), colors,
                  GRAVITY)

    def confetti(self, W, colors, n=4):
        """Confetti dropping from the top edge (call every frame while it should rain)."""
        pos = np.stack([self.rng.uniform(0, W, n), np.full(n, -10.0)], axis=1)
        vel = np.stack([self.rng.normal(0, 60, n), self.rng.uniform(80, 200, n)], axis=1)
        self._add(pos, vel, self.rng.uniform(2.5, 4.0, n), self.rng.uniform(4, 8, n), colors,
                  120.0)

    def clear(self):
        self.__init__()

    def update(self, t):
        dt = 0.0 if self._last_t is None else min(0.1, (t - self._last_t) / 1000.0)
        self._last_t = t
        if not len(self.life) or dt == 0:
            return
        self.vel[:, 1] += self.gravity * dt
        self.vel *= np.float32(max(0.0, 1 - DRAG * dt))
        self.pos += self.vel * dt
        self.life -= dt
        alive = self.life > 0
        if not alive.all():
            for name in ("pos", "vel", "life", "max_life", "size", "color", "gravity"):
                setattr(self, name, getattr(self, name)[alive])

    def draw(self, frame):
        H, W = frame.shape[:2]
        shrink = np.clip(self.life / np.maximum(self.max_life, 1e-3) * 1.5, 0, 1)
        for (x, y), s, c, k in zip(self.pos, self.size, self.color, shrink):
            r = int(s * k)
            if r >= 1 and -10 < x < W + 10 and -10 < y < H + 10:
                cv2.circle(frame, (int(x), int(y)), r, tuple(int(v) for v in c), -1, cv2.LINE_AA)


def spark_colors(pid):
    return [theme.player_color(pid), theme.scaled(theme.player_color(pid), 1.4), theme.WHITE]


def confetti_colors(winner):
    base = [theme.ACCENT, theme.GOOD, theme.WHITE]
    return base + [theme.player_color(winner)] * 3 if winner else base
