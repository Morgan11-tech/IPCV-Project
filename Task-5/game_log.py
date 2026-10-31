"""
Task 5 - per-round log of the game rules (Task 4), for the report (--log).

One CSV per program run in logs/, one row per finished round:
    game, level, word        which game of this run, level, target word
    outcome                  score / clash / timeout
    winner                   player who scored (empty otherwise)
    time_to_score_ms         word shown -> sign accepted
    feedback_ms              sign accepted -> score visible in the window
    wrong_p1, wrong_p2       wrong signs in this round        (false triggers)
    near_p1, near_p2         target word recognised, but below Task 4's 0.50
                             confidence, so not counted        (missed actions)
    target_conf              confidence of every sign of the target word that
                             Task 2 confirmed, space-separated
    other_conf               the same for confirmed signs of other words
    both_ms                  both players signed the word (accepted confidence):
                             time between their two signs; empty if only one did.
                             A clash is <= 150 ms. The other player's sign is
                             still watched for BOTH_WINDOW_MS after the round ends
The two confidence lists show whether Task 4's threshold (TARGET_CONFIDENCE)
is right: the summary counts what other thresholds would have accepted.

Rows are written 1 s after a round ends (or on quit), so quitting loses nothing.
Several games in one run are fine (rematch or R); "game" tells them apart.

Summary table for the report:
    python Task-5/game_log.py logs/*.csv
"""
import csv
import os
import sys
import time
from statistics import median

import bootstrap                     # task folders importable, also when run on its own
from task4_game import TARGET_CONFIDENCE

LOG_DIR = os.path.join(bootstrap.ROOT, "logs")
COLUMNS = ["game", "level", "word", "outcome", "winner", "time_to_score_ms", "feedback_ms",
           "wrong_p1", "wrong_p2", "near_p1", "near_p2", "target_conf", "other_conf", "both_ms"]
BOTH_WINDOW_MS = 1000    # shorter than the 1.5 s pause before the next word
OUTCOMES = {"score_event": "score", "clash": "clash", "round_timeout": "timeout"}


class GameLog:
    def __init__(self, folder=LOG_DIR):
        os.makedirs(folder, exist_ok=True)
        self.path = os.path.join(folder, time.strftime("rounds_%Y%m%d_%H%M%S.csv"))
        self._file = open(self.path, "w", newline="")
        self._csv = csv.DictWriter(self._file, COLUMNS)
        self._csv.writeheader()
        self.game = 0
        self._round = None       # the round being played
        self._done = None        # finished round, written once BOTH_WINDOW_MS has passed

    def observe(self, t_ms, events, snap, inputs):
        """Once per frame, after Task 4's update."""
        r = self._round
        if r is not None and snap["round_open"]:
            for pid, inp in inputs.items():
                unsure = getattr(inp, "unsure", None)
                if unsure == r["word"]:
                    r[f"near_p{pid}"] += 1
                confirmed = inp.word if inp.locked else unsure
                if confirmed is not None:
                    side = "target_conf" if confirmed == r["word"] else "other_conf"
                    r[side].append(round(inp.conf, 2))
        # both_ms: each player's first accepted sign of the word, also shortly
        # after the round ended (the other player's sign no longer scores then)
        for r in (self._round, self._done):
            if r is not None:
                for pid, inp in inputs.items():
                    if inp.locked and inp.word == r["word"]:
                        r["_signed"].setdefault(pid, t_ms)
        for e in events:
            k = e["type"]
            if k == "game_start":
                self._write_done()
                self.game += 1
                self._round = None
            elif k == "round_start":
                self._write_done()
                self._round = {"game": self.game, "level": snap["level"], "word": e["word"],
                               "wrong_p1": 0, "wrong_p2": 0, "near_p1": 0, "near_p2": 0,
                               "target_conf": [], "other_conf": [], "_signed": {},
                               "_start": e["timestamp_ms"]}
            elif self._round is None:
                continue
            elif k == "wrong_sign":
                self._round[f"wrong_p{e['player_id']}"] += 1
            elif k in OUTCOMES:
                r, self._round = self._round, None
                r["outcome"] = OUTCOMES[k]
                r["_end"] = e["timestamp_ms"]
                if k == "score_event":
                    r["winner"] = e["player_id"]
                    r["_accepted"] = e["timestamp_ms"]
                    r["time_to_score_ms"] = e["timestamp_ms"] - r["_start"]
                self._done = r

    def frame_shown(self, t_ms):
        """Call right after the frame is on screen: notes the feedback time and
        writes a finished round once BOTH_WINDOW_MS has passed."""
        r = self._done
        if r is None:
            return
        if "_accepted" in r and "feedback_ms" not in r:
            r["feedback_ms"] = t_ms - r["_accepted"]
        if t_ms >= r["_end"] + BOTH_WINDOW_MS:
            self._write_done()

    def _write_done(self):
        r, self._done = self._done, None
        if r is None:
            return
        signed = r["_signed"]
        if len(signed) == 2:
            r["both_ms"] = abs(signed[1] - signed[2])
        for c in ("target_conf", "other_conf"):
            r[c] = " ".join(f"{v:.2f}" for v in r[c])
        self._csv.writerow({c: r.get(c, "") for c in COLUMNS})
        self._file.flush()

    def reset(self):
        """R pressed: a finished round is kept, the open one is not logged."""
        self._write_done()
        self._round = None

    def close(self):
        self._write_done()
        self._file.close()


def summary(paths):
    """Per level: rounds and outcomes, median times, wrong signs and near
    misses per round (both players together)."""
    rows = [r for p in paths for r in csv.DictReader(open(p, newline=""))]
    print(f"{len(rows)} rounds from {len(paths)} file(s)\n")
    print("time to score = word shown -> sign accepted, feedback delay = sign accepted -> on screen\n"
          "(medians); wrong signs and near misses are averages per round, both players together\n")
    print(f"{'level':>5} {'rounds':>6} {'scored':>6} {'clashes':>7} {'timeouts':>8} "
          f"{'time to score':>13} {'feedback delay':>14} {'wrong signs':>11} {'near misses':>11}")
    for level in sorted({r["level"] for r in rows}) + ["all"]:
        rs = [r for r in rows if level == "all" or r["level"] == level]
        n = len(rs)
        count = lambda o: sum(r["outcome"] == o for r in rs)
        med = lambda c: median(int(r[c]) for r in rs if r[c]) if any(r[c] for r in rs) else float("nan")
        per = lambda a, b: sum(int(r[a]) + int(r[b]) for r in rs) / n
        print(f"{level:>5} {n:>6} {count('score') / n:>6.0%} {count('clash'):>7} {count('timeout'):>8} "
              f"{med('time_to_score_ms') / 1000:>12.2f}s {med('feedback_ms'):>12.0f}ms "
              f"{per('wrong_p1', 'wrong_p2'):>11.2f} {per('near_p1', 'near_p2'):>11.2f}")
    thresholds(rows)
    simultaneous(rows)


def thresholds(rows):
    """What each threshold would have done with the confirmed signs: share of
    target signs accepted, and wrong-word signs per round that would cost a
    lockout. Rounds end at the first accepted sign, so signs after it were
    never made: higher thresholds are an estimate."""
    rows = [r for r in rows if r.get("target_conf") is not None]   # logs older than these columns
    confs = lambda c: [float(v) for r in rows for v in r[c].split()]
    target, other = confs("target_conf"), confs("other_conf")
    if not target and not other:
        return
    print(f"\nthreshold check, {len(rows)} rounds ({len(target)} target signs, "
          f"{len(other)} other signs confirmed by Task 2)")
    print(f"{'threshold':>9} {'target signs accepted':>21} {'wrong signs per round':>21}")
    for th in (0.30, 0.40, 0.50, 0.60, 0.70):
        ok = sum(c >= th for c in target) / len(target) if target else float("nan")
        bad = sum(c >= th for c in other) / len(rows)
        mark = "  <- current" if abs(th - TARGET_CONFIDENCE) < 1e-9 else ""
        print(f"{th:>9.2f} {ok:>21.0%} {bad:>21.2f}{mark}")


def simultaneous(rows):
    """Rounds where both players signed the word, and how Task 4 decided them."""
    rows = [r for r in rows if r.get("both_ms") is not None]       # logs older than this column
    both = [r for r in rows if r["both_ms"]]
    if not both:
        return
    clashes = sum(r["outcome"] == "clash" for r in both)
    print(f"\nsimultaneous signing: both players signed the word in {len(both)} of {len(rows)} rounds")
    print(f"    {clashes} clash (within 150 ms), {len(both) - clashes} scored by the first signer; "
          f"median gap {median(int(r['both_ms']) for r in both):.0f} ms")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python Task-5/game_log.py logs/*.csv")
    summary(sys.argv[1:])

