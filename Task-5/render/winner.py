"""
Game-over screen: "<NAME> WINS!" (or "DRAW!") pops in and gently bounces in
the winner's colour, the feedback line underneath, then a stats table that
slides in row by row:

            P1 value   ROW NAME   P2 value
    score, correct signs, wrong signs, fastest, average time, best streak,
    and games won so far. The better value of each row is drawn bigger.

The confetti rain behind it comes from the particle system (render/__init__.py).
"""
import numpy as np

import theme

from render.anim import clamp01, ease_out_back, ease_out_cubic, progress
from render.primitives import panel
from render.text import draw as text

TABLE_W, ROW_H = 440, 34
ROWS_START_MS = 600            # table rows start after the title has popped in
ROW_STAGGER_MS = 120


def _secs(ms):
    return "-" if ms is None else f"{ms / 1000:.1f}s"


def stat_rows(stats, scores):
    """(label, {pid: shown text}, {pid: number to compare}, higher_is_better) per row."""
    p = stats.players
    rows = [
        ("SCORE", {i: scores.get(i, 0) for i in p}, True),
        ("CORRECT", {i: p[i].correct for i in p}, True),
        ("WRONG", {i: p[i].wrong for i in p}, False),
        ("FASTEST", {i: p[i].fastest_ms for i in p}, False),
        ("AVERAGE", {i: p[i].avg_ms for i in p}, False),
        ("BEST STREAK", {i: p[i].best_streak for i in p}, True),
        ("GAMES WON", dict(stats.games_won), True),
    ]
    out = []
    for label, values, higher in rows:
        timed = label in ("FASTEST", "AVERAGE")
        shown = {i: _secs(v) if timed else str(v) for i, v in values.items()}
        out.append((label, shown, values, higher))
    return out


def _best(values, higher):
    """Player with the better value, or None on a tie / missing values."""
    known = {i: v for i, v in values.items() if v is not None}
    if len(known) < 2 or len(set(known.values())) == 1:
        return None
    return (max if higher else min)(known, key=known.get)


class WinnerScreen:
    def __init__(self):
        self.start_t, self.winner, self.line, self.rows = None, None, None, []

    def start(self, t, winner_id, line=None, stats=None, scores=None):
        self.start_t, self.winner, self.line = t, winner_id, line
        self.rows = stat_rows(stats, scores or {}) if stats is not None else []

    def stop(self):
        self.start_t, self.winner = None, None

    def draw(self, frame, t):
        if self.start_t is None:
            return
        H, W = frame.shape[:2]
        c = theme.player_color(self.winner) if self.winner else theme.ACCENT
        u = (t - self.start_t) / 1000.0
        pop = ease_out_back(min(1.0, u / 0.5))
        y = 205 + int(6 * np.sin(u * 4.0))                                   # gentle bounce

        title = f"{theme.PLAYER_NAMES[self.winner]} WINS!" if self.winner else "DRAW!"
        text(frame, title, (W // 2, y), 72 * (0.4 + 0.6 * pop), c, align="center")
        if self.line:
            text(frame, self.line, (W // 2, y + 44), 34, theme.ACCENT, align="center",
                 alpha=clamp01((u - 0.3) / 0.4))
        self._table(frame, t, W // 2, y + 70)

    def _table(self, frame, t, cx, y):
        if not self.rows:
            return
        h = 16 + ROW_H * len(self.rows)
        grow = ease_out_cubic(progress(t, self.start_t + ROWS_START_MS - 200, 300))
        if grow <= 0:
            return
        panel(frame, cx - TABLE_W // 2, y, TABLE_W, max(1, int(h * grow)))
        for i, (label, shown, values, higher) in enumerate(self.rows):
            v = progress(t, self.start_t + ROWS_START_MS + ROW_STAGGER_MS * i, 250)
            if v <= 0:
                break
            a = ease_out_cubic(v)
            row = y + 8 + ROW_H * (i + 1) - 9
            best = _best(values, higher)
            text(frame, label, (cx, row), 18, theme.WHITE, font="body", align="center",
                 alpha=0.85 * a)
            for pid, x, align in ((1, cx - TABLE_W // 2 + 24, "left"),
                                  (2, cx + TABLE_W // 2 - 24, "right")):
                slide = (1 - a) * (-30 if pid == 1 else 30)
                text(frame, shown[pid], (x + slide, row + 2), 30 if pid == best else 24,
                     theme.player_color(pid), align=align, alpha=a)
