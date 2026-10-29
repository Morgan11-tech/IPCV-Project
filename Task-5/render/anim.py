"""
Easing curves: turn linear progress u (0 -> 1) into motion that feels
physical. Every animation in the game is "progress over a duration" pushed
through one of these.
"""
import math


def clamp01(u):
    return 0.0 if u < 0 else 1.0 if u > 1 else u


def progress(t, t0, dur):
    """0 -> 1 over [t0, t0 + dur] (ms)."""
    return clamp01((t - t0) / max(dur, 1))


def ease_out_cubic(u):
    """Fast start, gentle stop - things sliding or rising into place."""
    u = clamp01(u)
    return 1 - (1 - u) ** 3


def ease_in_cubic(u):
    """Gentle start, fast end - things leaving."""
    u = clamp01(u)
    return u ** 3


def ease_out_back(u, overshoot=1.7):
    """Overshoots a little then settles - pop-ins and bumps."""
    u = clamp01(u)
    return 1 + (overshoot + 1) * (u - 1) ** 3 + overshoot * (u - 1) ** 2


def smoothstep(u):
    u = clamp01(u)
    return u * u * (3 - 2 * u)


def fade_out(u, start=0.7):
    """1 until `start`, then linearly to 0 at u = 1."""
    return 1.0 if u < start else clamp01((1 - u) / (1 - start))


def pulse(t, period_ms):
    """0 -> 1 -> 0 once per period (sine), for heartbeats and blinking."""
    return 0.5 - 0.5 * math.cos(2 * math.pi * (t % period_ms) / period_ms)


def bump(t, t0, dur=300, amount=0.35):
    """Scale factor for a quick 'boing' after an event at t0 (1.0 when idle)."""
    if t0 is None or t - t0 >= dur or t < t0:
        return 1.0
    u = (t - t0) / dur
    return 1 + amount * math.sin(math.pi * u) * (1 - u)
