"""
Task 5 - feedback lines: what the game "says" when something happens.

for_events() returns one line (or None) per game event. Lines are picked at
random from a pool for that moment, never repeating one of the last few lines
of the same pool, so the game does not sound like a broken record.

Which pool is used:
    correct    fast answer (quick for this level)   -> FAST
               player was behind before this point  -> COMEBACK
               3+ correct in a row                  -> STREAK
               otherwise                            -> CORRECT
    wrong      second wrong in a row                -> WRONG_AGAIN, else WRONG
    clash / time's up / level up / game over        -> their own pools
"""
import random
from collections import deque

CORRECT = ["Nailed it!", "Smooth!", "Clean sign!", "Textbook!", "Beautiful!", "Spot on!",
           "Too easy!", "Yes!"]
FAST = ["Lightning hands!", "Speed demon!", "Blink and you miss it!", "Quick draw!",
        "Whoa, fast!"]
STREAK = ["On fire!", "Unstoppable!", "Can't be stopped!", "Hat trick!", "Rolling!"]
COMEBACK = ["Comeback!", "Back in it!", "Not over yet!", "Closing the gap!"]
WRONG = ["So close!", "Not quite!", "Shake it off!", "Almost!", "Try again!",
         "Skill issue!", "Failure!", "Go home!"]
WRONG_AGAIN = ["What are you doing?", "Get lost!", "Come on man!",
               "Watch the word!"]
CLASH = ["Great minds!", "Dead heat!", "Photo finish!"]
TIMEOUT = ["Too slow, both of you!", "Clock wins this one!", "Tick tock...",
           "Faster next time!"]
LAST_LIFE = ["Last life!", "One heart left!", "Now or never!"]
LEVEL_UP = ["Getting harder!", "Less time now!", "Turn it up!", "Pressure's on!"]
WIN = ["What a match!", "Champion!", "Total legend!", "Untouchable!"]
DRAW = ["Dead even!", "Nobody blinked!", "Too close to call!"]

FAST_FRACTION = 0.3        # answered within 30% of the round time ...
FAST_MIN_MS = 2500         # ... or within 2.5 s, whichever is longer
STREAK_MIN = 3
NO_REPEAT = 3              # never repeat any of the last 3 lines of a pool


class Feedback:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self._recent = {}          # pool id -> recent lines
        self._wrong_streak = {}    # pid -> wrong signs in a row

    def pick(self, pool):
        recent = self._recent.setdefault(id(pool), deque(maxlen=min(NO_REPEAT, len(pool) - 1)))
        line = self.rng.choice([s for s in pool if s not in recent] or pool)
        recent.append(line)
        return line

    def for_events(self, events, snap):
        """One line (or None) per event, in the same order."""
        scores = {p["player_id"]: p["score"] for p in snap["players"]}
        hearts = {p["player_id"]: p.get("hearts") for p in snap["players"]}
        lines = []
        for e in events:
            k = e["type"]
            line = None
            if k == "score_event":
                line = self._correct(e, snap, scores)
            elif k == "wrong_sign":
                pid = e["player_id"]
                self._wrong_streak[pid] = self._wrong_streak.get(pid, 0) + 1
                line = self.pick(WRONG_AGAIN if self._wrong_streak[pid] >= 2 else WRONG)
            elif k == "clash":
                line = self.pick(CLASH)
            elif k == "round_timeout":
                low = [h for h in hearts.values() if h is not None]
                line = self.pick(LAST_LIFE if low and min(low) == 1 else TIMEOUT)
            elif k == "level_up":
                line = self.pick(LEVEL_UP)
            elif k == "game_over":
                line = self.pick(WIN if e["winner"] is not None else DRAW)
            elif k == "game_start":
                self._wrong_streak.clear()
            lines.append(line)
        return lines

    def _correct(self, e, snap, scores):
        pid = e["player_id"]
        self._wrong_streak[pid] = 0
        before = e["score"] - e["points"]
        rivals = [s for p, s in scores.items() if p != pid]
        target = snap.get("target") or {}
        window = target.get("expires_ms", 0) - target.get("spawned_ms", 0)
        took = e["timestamp_ms"] - target.get("spawned_ms", e["timestamp_ms"])
        if window > 0 and took <= max(FAST_MIN_MS, FAST_FRACTION * window):
            return self.pick(FAST)
        if rivals and before < max(rivals):
            return self.pick(COMEBACK)
        if e.get("streak", 0) >= STREAK_MIN:
            return f"{self.pick(STREAK)} x{e['streak']}"
        return self.pick(CORRECT)
