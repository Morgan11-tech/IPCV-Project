"""Temporal smoothing for hand landmarks.

One-Euro filter (Casiez, Roussel, Vogel, CHI 2012): an adaptive low-pass
filter. When the hand is nearly still the cutoff is low (strong smoothing,
kills jitter); when it moves fast the cutoff rises with speed (little
smoothing, so there is almost no lag). That is exactly the
smoothness-versus-responsiveness trade-off the brief asks Task 2 to balance,
with two intuitive knobs:

    min_cutoff  Hz, jitter suppression when still   (lower = smoother, laggier)
    beta        how fast the cutoff opens up with speed (higher = less lag in motion)
"""
import math

import numpy as np


class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=20.0, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x = None      # previous filtered value
        self._dx = None     # previous filtered derivative
        self._t = None      # previous timestamp (seconds)

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self._x = self._dx = self._t = None

    def __call__(self, x, t):
        """Filter one sample. `x` is an array, `t` its timestamp in seconds."""
        x = np.asarray(x, dtype=float)
        if self._t is None:
            self._x, self._dx, self._t = x.copy(), np.zeros_like(x), float(t)
            return x
        dt = max(float(t) - self._t, 1e-3)
        dx = (x - self._x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x
        self._x, self._dx, self._t = x_hat, dx_hat, float(t)
        return x_hat
