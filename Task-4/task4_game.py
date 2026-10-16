# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 23:13:27 2026

@author: ruvah
"""

"""
Task 4 — Player interaction and game control  (sign race)
=========================================================
DELIVERABLE MODULE for Task 5 integration.

This file contains ONLY game logic and drawing.  It does NOT touch the
webcam, MediaPipe, Task 2's model files, or any external resource.
Task 5 owns the main loop and feeds this module `PlayerInput` objects
once per frame.

ACTIONS (per player)
  1. SIGN    - sign the target word.  FIRST correct locked sign wins the round.
  2. READY   - hold an open palm still ~0.7 s to start / restart.
  3. DEFEND  - cross both arms over your chest for 0.3 s.
               While defending you take only HALF points from a loss.

LEVEL PROGRESSION (5 words per level, 15 words total)
  Level 1 -> 15 s : please, no, yes, hello, more
  Level 2 -> 10 s : bathroom, apple, color, boy, glasses
  Level 3 ->  5 s : yesterday, brown, football, water, red
  Advance when either player hits LEVEL_UP_SCORE (default 20 pts = 2 wins).

PUBLIC API (what Task 5 uses)
  PlayerInput        - one per player per frame
  GameController     - state machine; call .update(t_ms, inputs) each frame
  EffectsLayer       - visual feedback anchored to player positions
  WinnerDancer       - victory animation
  VictoryJingle      - synthesised victory sound
  draw_hud           - scores, timer, level, phase
  anchors_from_inputs- helper to compute {pid: {center, r}} for effects
  detect_defending   - pure pose -> bool helper (Task 2 calls it)

EVENTS emitted by GameController.update():
  {"type": "game_start"}
  {"type": "score_event",  "player_id", "points", "target", ...}
  {"type": "wrong_sign",   "player_id", "word", "target"}
  {"type": "clash",        "player_ids", "target", "points"}
  {"type": "round_timeout","target"}
  {"type": "level_up",     "level", "window_ms"}
  {"type": "defend_start", "player_id"}
  {"type": "defend_end",   "player_id"}
  {"type": "game_over",    "winner", "scores"}
  {"type": "action_event", "player_id", "action"}
"""

import math
import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# =====================================================================
#  CONSTANTS
# =====================================================================
PLAYER_IDS = (1, 2)
POINTS_PER_WIN = 10
WIN_SCORE = 50

TIE_WINDOW_MS = 150
WRONG_LOCKOUT_MS = 1500
REARM_RELEASE_MS = 250

REQUIRE_READY_TO_START = True
READY_HOLD_MS = 700
STILL_SPEED = 1.5
SIGNING_CONF = 0.5
GRACE_MS = 250

LEVEL_TIMES_MS = {1: 15000, 2: 10000, 3: 5000}
LEVEL_UP_SCORE = 20
MAX_LEVEL = 3
WORDS_PER_LEVEL = 5

LEVEL_WORDS = {
    1: ["please", "no", "yes", "hello", "more"],
    2: ["bathroom", "apple", "color", "boy", "glasses"],
    3: ["yesterday", "brown", "football", "water", "red"],
}

JINGLE_ENABLED = True
JINGLE_VOLUME = 0.35

DEFEND_HOLD_MS = 300
DEFEND_COOLDOWN_MS = 1500
DEFEND_SCORE_MULTIPLIER = 0.5

STREAK_GLOW_THRESHOLD = 2


# =====================================================================
#  PUBLIC DATA CONTRACTS
# =====================================================================
@dataclass
class PlayerInput:
    """Everything the controller needs about ONE player in ONE frame.

    Task 5 fills this in from Task 1 / 2 / 3 outputs.
    """
    present: bool = True
    hands: list = field(default_factory=list)         # (21,3) pixel arrays (for READY + display)
    word: Optional[str] = None                        # Task 2: confirmed or best guess
    conf: float = 0.0                                 # Task 2: confidence
    locked: bool = False                              # Task 2: True only on confirm frame
    face_box: Optional[tuple] = None                  # Task 1: (x, y, w, h)
    defending: bool = False                           # Task 2 pose: in defend pose


@dataclass
class PlayerState:
    id: int
    score: int = 0
    ready: bool = False
    lockout_until: int = 0
    needs_release: bool = False
    release_since: Optional[int] = None
    prev_wrist: Optional[tuple] = None
    ready_dwell: "Dwell" = field(default_factory=lambda: Dwell())
    defending_since: Optional[int] = None
    defend_active: bool = False
    defend_cooldown_until: int = 0
    win_streak: int = 0


class Dwell:
    """'Has this condition been true long enough?' with a short grace period."""
    def __init__(self, grace_ms=GRACE_MS):
        self.grace, self.since, self.last_true = grace_ms, None, None

    def update(self, cond, t):
        if cond:
            if self.since is None:
                self.since = t
            self.last_true = t
        elif self.since is not None and t - self.last_true > self.grace:
            self.since = None
        return 0 if self.since is None else t - self.since

    def reset(self):
        self.since = self.last_true = None


# =====================================================================
#  PURE HELPERS  (no external deps beyond numpy)
# =====================================================================
def hand_size(pts):
    pts = np.asarray(pts, dtype=np.float32).reshape(21, 3)
    return float(np.max(np.linalg.norm(pts[:, :2] - pts[0, :2], axis=1)))


def _finger_extended(pts, tip, pip):
    wrist = pts[0, :2]
    return np.linalg.norm(pts[tip, :2] - wrist) > 1.15 * np.linalg.norm(pts[pip, :2] - wrist)


def is_open_palm(points):
    """True when all four fingers and the thumb are stretched out."""
    pts = np.asarray(points, dtype=np.float32).reshape(21, 3)
    four = all(_finger_extended(pts, t, p) for t, p in ((8, 6), (12, 10), (16, 14), (20, 18)))
    thumb = (np.linalg.norm(pts[4, :2] - pts[17, :2]) >
             np.linalg.norm(pts[3, :2] - pts[17, :2]))
    return bool(four and thumb)


def landmarks_to_xyz(lm, W, H):
    """Convert MediaPipe NormalizedLandmark list to a (21,3) pixel array.
    Convenience helper for Task 5; not used internally."""
    pts = np.zeros((21, 3), dtype=np.float32)
    for i in range(21):
        pts[i, 0] = lm[i].x * W
        pts[i, 1] = lm[i].y * H
        pts[i, 2] = getattr(lm[i], "z", 0.0)
    return pts


def synthetic_hand(center, size, open_palm=True):
    """Fake 21-point hand for tests. Fingers point up."""
    P = np.zeros((21, 3), dtype=np.float32)
    for k, (x, y) in enumerate([(-0.2, -0.2), (-0.35, -0.3), (-0.5, -0.4), (-0.65, -0.5)]):
        P[1 + k, :2] = (x, y)
    for f, x in enumerate((-0.1, 0.0, 0.15, 0.3)):
        ys = (-0.5, -0.75, -0.9, -1.0) if open_palm else (-0.5, -0.65, -0.5, -0.3)
        for j, y in enumerate(ys):
            P[5 + 4 * f + j, :2] = (x, y)
    if not open_palm:
        P[2:5, :2] = [(-0.1, -0.3), (0.0, -0.4), (0.1, -0.45)]
    P[:, :2] = P[:, :2] * size + np.array(center, dtype=np.float32)
    return P


def detect_defending(pose_landmarks, W, H):
    """Return True when both wrists are near the chest and above the elbows.

    `pose_landmarks` is a sequence of 33 (x, y[, z]) triples (MediaPipe Pose order).
    Accepts normalised (0..1) or pixel coordinates.
    """
    if pose_landmarks is None or len(pose_landmarks) < 25:
        return False
    pts = np.asarray(pose_landmarks, dtype=np.float32)
    if pts[:, :2].max() > 1.5:
        pts = pts.copy()
        pts[:, 0] /= max(W, 1)
        pts[:, 1] /= max(H, 1)

    ls, rs = pts[11, :2], pts[12, :2]
    le, re = pts[13, :2], pts[14, :2]
    lw, rw = pts[15, :2], pts[16, :2]
    lh, rh = pts[23, :2], pts[24, :2]

    chest = 0.5 * (ls + rs)
    hip_mid = 0.5 * (lh + rh)
    torso_h = max(np.linalg.norm(chest - hip_mid), 1e-3)

    near_chest = (np.linalg.norm(lw - chest) < 0.55 * torso_h and
                  np.linalg.norm(rw - chest) < 0.55 * torso_h)
    above_elbows = lw[1] < le[1] and rw[1] < re[1]
    return bool(near_chest and above_elbows)


def anchors_from_inputs(inputs, defaults=None):
    """Compute {pid: {"center": (x,y), "r": r}} for effects.

    Prefers face_box (Task 1), falls back to hands (Task 2), then defaults.
    """
    out = {}
    for pid, inp in inputs.items():
        if inp.face_box is not None:
            x, y, w, h = inp.face_box
            out[pid] = {"center": (x + w / 2, y + h / 2), "r": 0.9 * max(w, h)}
        elif inp.hands:
            c = np.mean([hh[0, :2] for hh in inp.hands], axis=0)
            out[pid] = {"center": (float(c[0]), float(c[1])), "r": 80.0}
        elif defaults and pid in defaults:
            out[pid] = {"center": tuple(defaults[pid]), "r": 80.0}
    return out


# =====================================================================
#  GAME CONTROLLER
# =====================================================================
class GameController:
    """Pure state machine. No I/O, no rendering, no file loading.

    Usage:
        gc = GameController(player_ids=(1, 2), words=None)
        while running:
            events = gc.update(t_ms, {1: PlayerInput(...), 2: PlayerInput(...)})
            for e in events:
                if e["type"] in ("score_event", "clash", "round_timeout"):
                    gc.next_word(t_ms + 1500)   # schedule next round
            snap = gc.snapshot(t_ms)
    """

    def __init__(self, player_ids=PLAYER_IDS, words=None):
        self.ids = list(player_ids)
        if words is None:
            words = sorted({w for block in LEVEL_WORDS.values() for w in block})
        self.words = [str(w).strip().lower() for w in words]
        self.reset()

    def reset(self):
        self.players = {pid: PlayerState(pid) for pid in self.ids}
        self.phase = "waiting" if REQUIRE_READY_TO_START else "playing"
        self.target, self.round_open, self.pending = None, False, {}
        self.winner, self.log, self._frame_id = None, [], 0
        self.level = 1
        self.level_up_at = None
        self._level_queues = {lvl: random.sample(words, len(words))
                              for lvl, words in LEVEL_WORDS.items()}
        self._queue_pos = {lvl: 0 for lvl in LEVEL_WORDS}

    # --- target spawning ---
    def set_target(self, word, spawned_ms):
        window = LEVEL_TIMES_MS[self.level]
        self.target = {"word": word, "spawned_ms": spawned_ms,
                       "expires_ms": spawned_ms + window, "level": self.level}
        self.round_open, self.pending = True, {}
        for st in self.players.values():
            st.lockout_until = 0

    def next_word(self, t_ms):
        """Pick the next word from the current level's 5-word queue."""
        lvl = self.level
        q = self._level_queues[lvl]
        pos = self._queue_pos[lvl]
        if pos >= len(q):
            self._level_queues[lvl] = random.sample(LEVEL_WORDS[lvl], len(LEVEL_WORDS[lvl]))
            q = self._level_queues[lvl]
            pos = 0
        word = q[pos]
        self._queue_pos[lvl] = pos + 1
        self.set_target(word, t_ms)

    def _ev(self, kind, t, **kw):
        return {"type": kind, "frame_id": self._frame_id, "timestamp_ms": int(t), **kw}

    def update(self, t_ms, inputs, frame_id=0):
        """Advance the state machine by one frame. Returns a list of events."""
        ev = []
        self._frame_id = frame_id
        if self.phase in ("waiting", "game_over"):
            self._lobby(t_ms, inputs, ev)
        else:
            for pid in self.ids:
                inp, st = inputs.get(pid), self.players[pid]
                if inp is not None and inp.present:
                    self._defend(st, inp, t_ms, ev)
                    self._sign(st, inp, t_ms, ev)
            self._resolve_round(t_ms, ev)
            self._check_end(t_ms, ev)
        self.log += ev
        return ev

    def _palm_still(self, st, inp, t):
        palm = next((h for h in inp.hands if is_open_palm(h)), None)
        if palm is None:
            st.prev_wrist = None
            return False
        wrist, size = palm[0, :2].copy(), max(hand_size(palm), 1e-6)
        still = False
        if st.prev_wrist is not None:
            pw, pt = st.prev_wrist
            still = np.linalg.norm(wrist - pw) / size / (max(t - pt, 1) / 1000.0) < STILL_SPEED
        st.prev_wrist = (wrist, t)
        return still

    def _lobby(self, t, inputs, ev):
        for pid in self.ids:
            inp, st = inputs.get(pid), self.players[pid]
            if inp is None or not inp.present:
                st.ready_dwell.reset(); st.prev_wrist = None; st.ready = False
                continue
            mid_sign = inp.word is not None and inp.conf >= SIGNING_CONF
            held = st.ready_dwell.update(self._palm_still(st, inp, t) and not mid_sign, t)
            if held >= READY_HOLD_MS and not st.ready:
                st.ready = True
                ev.append(self._ev("action_event", t, player_id=pid, action="ready"))
        if all(self.players[p].ready for p in self.ids):
            self.reset()
            self.phase = "playing"
            ev.append(self._ev("game_start", t))

    def _defend(self, st, inp, t, ev):
        if not inp.defending:
            if st.defend_active:
                st.defend_active = False
                ev.append(self._ev("defend_end", t, player_id=st.id))
            st.defending_since = None
            return
        if st.defending_since is None:
            st.defending_since = t
        held = t - st.defending_since
        if (not st.defend_active and held >= DEFEND_HOLD_MS
                and t >= st.defend_cooldown_until):
            st.defend_active = True
            ev.append(self._ev("defend_start", t, player_id=st.id))

    def _sign(self, st, inp, t, ev):
        if not inp.locked:
            if inp.word is None:
                st.release_since = t if st.release_since is None else st.release_since
                if t - st.release_since >= REARM_RELEASE_MS:
                    st.needs_release = False
            else:
                st.release_since = None
            return
        st.release_since = None
        if st.needs_release or t < st.lockout_until:
            return
        st.needs_release = True
        if not self.round_open or self.target is None or t > self.target["expires_ms"]:
            return
        if inp.word and inp.word.strip().lower() == self.target["word"].strip().lower():
            self.pending.setdefault(st.id, (t, inp.conf))
        else:
            st.lockout_until = t + WRONG_LOCKOUT_MS
            ev.append(self._ev("wrong_sign", t, player_id=st.id, word=inp.word,
                               target=self.target["word"], confidence=inp.conf))

    def _resolve_round(self, t, ev):
        if not self.pending:
            if self.round_open and self.target and t > self.target["expires_ms"]:
                self.round_open = False
                ev.append(self._ev("round_timeout", t, target=self.target["word"]))
            return
        first_t = min(v[0] for v in self.pending.values())
        if t < first_t + TIE_WINDOW_MS:
            return
        cands = sorted((tt, pid, c) for pid, (tt, c) in self.pending.items()
                       if tt <= first_t + TIE_WINDOW_MS)
        self.round_open, self.pending = False, {}
        word = self.target["word"]
        if len(cands) > 1:
            for _, pid, _ in cands:
                self.players[pid].score += POINTS_PER_WIN // 2
            ev.append(self._ev("clash", t, player_ids=[p for _, p, _ in cands],
                               target=word, points=POINTS_PER_WIN // 2))
            return
        tt, winner, conf = cands[0]
        reward = POINTS_PER_WIN
        if self.players[winner].defend_active:
            reward = int(round(POINTS_PER_WIN * DEFEND_SCORE_MULTIPLIER))
            self.players[winner].defend_cooldown_until = t + DEFEND_COOLDOWN_MS
        self.players[winner].score += reward
        for pid in self.ids:
            if pid == winner:
                self.players[pid].win_streak += 1
            else:
                self.players[pid].win_streak = 0
        ev.append(self._ev("score_event", tt, player_id=winner, word=word, letter=word,
                           target=word, correct=True, first=True, confidence=conf,
                           points=reward, score=self.players[winner].score,
                           defended=self.players[winner].defend_active,
                           streak=self.players[winner].win_streak))
        top = max(self.players[p].score for p in self.ids)
        new_level = min(MAX_LEVEL, 1 + top // LEVEL_UP_SCORE)
        if new_level > self.level:
            self.level = new_level
            self.level_up_at = t
            ev.append(self._ev("level_up", t, level=self.level,
                               window_ms=LEVEL_TIMES_MS[self.level]))

    def _check_end(self, t, ev):
        if WIN_SCORE is None:
            return
        scores = {p: self.players[p].score for p in self.ids}
        top = max(scores.values())
        leaders = [p for p, s in scores.items() if s == top]
        if top >= WIN_SCORE and len(leaders) == 1:
            self.phase, self.winner, self.round_open = "game_over", leaders[0], False
            for st in self.players.values():
                st.ready = False; st.ready_dwell.reset(); st.prev_wrist = None
                st.defend_active = False; st.defending_since = None
            ev.append(self._ev("game_over", t, winner=self.winner, scores=scores))

    def snapshot(self, t_ms):
        """Current state for Task 5's scene renderer and Task 4's HUD."""
        return {"phase": self.phase, "target": self.target, "round_open": self.round_open,
                "winner": self.winner, "level": self.level,
                "level_window_ms": LEVEL_TIMES_MS[self.level],
                "players": [{"player_id": p.id, "score": p.score, "ready": p.ready,
                             "locked_out": t_ms < p.lockout_until,
                             "defending": p.defend_active,
                             "streak": p.win_streak}
                            for p in self.players.values()]}


# =====================================================================
#  VISUAL FEEDBACK  (drawing only)
# =====================================================================
def _txt(img, text, org, scale=0.8, color=(255, 255, 255), thick=2):
    import cv2
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


class EffectsLayer:
    """Turns game events into short animations anchored to the right player.

    Usage:
        fx = EffectsLayer()
        fx.on_events(events, t_ms)          # register animations
        fx.draw(frame, t_ms, anchors)       # draw them
    """
    def __init__(self):
        self.fx = []

    def _add(self, kind, pid, t, dur, text=""):
        self.fx.append({"kind": kind, "pid": pid, "t0": t, "dur": dur, "text": text})

    def on_events(self, events, t):
        for e in events:
            k = e["type"]
            if k == "score_event":
                self._add("score", e["player_id"], t, 1300, f"+{e['points']}")
            elif k == "wrong_sign":
                self._add("wrong", e["player_id"], t, 900, "wrong sign")
            elif k == "clash":
                self._add("banner", None, t, 1300, "CLASH!")
            elif k == "round_timeout":
                self._add("banner", None, t, 1000, "TIME'S UP")
            elif k == "action_event" and e["action"] == "ready":
                self._add("ready", e["player_id"], t, 1000, "READY!")
            elif k == "game_start":
                self._add("banner", None, t, 1200, "GO!")
            elif k == "level_up":
                self._add("banner", None, t, 1500, f"LEVEL {e['level']}!")
            elif k == "defend_start":
                self._add("ready", e["player_id"], t, 900, "DEFEND!")

    def draw(self, frame, t, anchors):
        import cv2
        self.fx = [f for f in self.fx if t - f["t0"] < f["dur"]]
        H, W = frame.shape[:2]
        for f in self.fx:
            a = anchors.get(f["pid"]) if f["pid"] is not None else None
            if f["pid"] is not None and a is None:
                continue
            u = (t - f["t0"]) / f["dur"]
            kind, text = f["kind"], f["text"]
            cx, cy, r, top = 0, 0, 0, 100
            if a:
                cx, cy, r = int(a["center"][0]), int(a["center"][1]), int(a["r"])
                top = max(int(cy - r - 25 - 60 * u), 100)
            if kind == "score":
                cv2.circle(frame, (cx, cy), int(r * (1 + 0.6 * u)), (0, 255, 255), 4, cv2.LINE_AA)
                _txt(frame, text, (cx - 45, top), 1.8, (0, 255, 255), 4)
            elif kind == "wrong":
                _txt(frame, "X " + text, (cx - 70, top), 1.0, (60, 60, 255), 2)
            elif kind == "ready":
                _txt(frame, text, (cx - 60, top), 1.3, (80, 255, 80), 3)
            elif kind == "banner":
                _txt(frame, text, (W // 2 - 110, H // 2 - 40 - int(30 * u)), 1.8, (0, 255, 255), 4)


def draw_hud(frame, snap, inputs, t, show_scores=True):
    """Scores, live guesses, target word + timer, level, phase messages.

    Safe to call on any BGR frame. Task 5 can call this last, after their scene.
    """
    import cv2
    H, W = frame.shape[:2]
    n = len(snap["players"])
    for i, p in enumerate(snap["players"]):
        x0, y0 = 15 + i * (W // n), H - 85
        _txt(frame, f"P{p['player_id']}", (x0, y0), 0.9)
        if show_scores:
            _txt(frame, f"{p['score']}", (x0 + 70, y0 + 5), 1.6, (0, 255, 255), 3)
        inp = inputs.get(p["player_id"])
        if p.get("defending"):
            _txt(frame, "DEFENDING", (x0, y0 + 40), 0.7, (255, 200, 0), 2)
        elif snap["phase"] != "playing":
            _txt(frame, "READY" if p["ready"] else "hold open palm", (x0, y0 + 40), 0.65,
                 (80, 255, 80) if p["ready"] else (200, 200, 200), 2)
        elif p["locked_out"]:
            _txt(frame, "locked out", (x0, y0 + 40), 0.6, (60, 60, 255), 2)
        elif inp is not None and inp.word:
            _txt(frame, f"guess: {inp.word} {inp.conf:.2f}" + ("  LOCKED" if inp.locked else ""),
                 (x0, y0 + 40), 0.6, (0, 255, 0) if inp.locked else (0, 200, 255), 2)

    for i, p in enumerate(snap["players"]):
        if p.get("streak", 0) >= STREAK_GLOW_THRESHOLD:
            x0 = 15 + i * (W // n) + 130
            _txt(frame, f"STREAK x{p['streak']}", (x0, H - 60), 0.65, (0, 180, 255), 2)

    lvl_colors = {1: (80, 255, 80), 2: (0, 200, 255), 3: (60, 60, 255)}
    _txt(frame, f"LEVEL {snap['level']}", (15, 30), 1.0,
         lvl_colors.get(snap["level"], (255, 255, 255)), 3)

    if snap["phase"] == "waiting":
        _txt(frame, "Both players: hold an open palm to start", (W // 2 - 280, 45), 0.9)
    elif snap["phase"] == "game_over":
        _txt(frame, f"GAME OVER - PLAYER {snap['winner']} WINS", (W // 2 - 300, 45), 1.1, (0, 255, 255), 3)
        _txt(frame, "hold open palm to play again", (W // 2 - 200, 80), 0.7)
    elif snap["target"]:
        tg = snap["target"]
        _txt(frame, f"SIGN:  {tg['word'].upper()}", (W // 2 - 140, 40), 1.2, (255, 255, 255), 3)
        if snap["round_open"]:
            span = max(tg["expires_ms"] - tg["spawned_ms"], 1)
            frac = float(np.clip((tg["expires_ms"] - t) / span, 0, 1))
            cv2.rectangle(frame, (W // 2 - 150, 55), (W // 2 + 150, 68), (60, 60, 60), -1)
            cv2.rectangle(frame, (W // 2 - 150, 55), (W // 2 - 150 + int(300 * frac), 68),
                          (0, 200, 255), -1)
            secs = max(0, (tg["expires_ms"] - t) / 1000.0)
            _txt(frame, f"{secs:4.1f}s", (W // 2 + 170, 68), 0.7,
                 (0, 255, 255) if secs > 3 else (60, 60, 255), 2)


# =====================================================================
#  WINNER DANCER
# =====================================================================
class WinnerDancer:
    """Stick-figure that dances in the middle of the screen when someone wins.

    Usage:
        dancer = WinnerDancer()
        # on game_over event:
        dancer.start(t_ms, winner_id, winner_color)
        # each frame while phase == "game_over":
        dancer.draw(frame, t_ms)
    """
    def __init__(self):
        self.start_t = None
        self.winner = None
        self.color = (0, 255, 255)

    def start(self, t, winner_id, winner_color):
        self.start_t = t
        self.winner = winner_id
        self.color = winner_color

    def stop(self):
        self.start_t = None
        self.winner = None

    def draw(self, frame, t):
        if self.start_t is None:
            return
        import cv2
        H, W = frame.shape[:2]
        cx, cy = W // 2, int(H * 0.55)
        u = (t - self.start_t) / 1000.0
        cx += int(10 * np.sin(u * 3.0))
        cy += int(12 * np.sin(u * 6.0))
        c = self.color
        cv2.line(frame, (cx, cy), (cx, cy + 70), c, 6, cv2.LINE_AA)
        for sign, phase in ((-1, 0.0), (1, np.pi)):
            hip = np.array([cx, cy + 70])
            knee = hip + np.array([sign * (28 + 10 * np.sin(u * 6 + phase)),
                                   45 - 15 * np.sin(u * 6 + phase)])
            foot = knee + np.array([sign * 20, 35])
            cv2.line(frame, tuple(hip.astype(int)), tuple(knee.astype(int)), c, 5, cv2.LINE_AA)
            cv2.line(frame, tuple(knee.astype(int)), tuple(foot.astype(int)), c, 5, cv2.LINE_AA)
        for sign, phase in ((-1, 0.0), (1, np.pi)):
            shoulder = np.array([cx, cy + 5])
            elbow = shoulder + np.array([sign * (35 + 15 * np.sin(u * 6 + phase)),
                                         -20 + 20 * np.cos(u * 6 + phase)])
            hand = elbow + np.array([sign * (25 + 10 * np.sin(u * 6 + phase)),
                                     -25 + 15 * np.cos(u * 6 + phase)])
            cv2.line(frame, tuple(shoulder.astype(int)), tuple(elbow.astype(int)), c, 5, cv2.LINE_AA)
            cv2.line(frame, tuple(elbow.astype(int)), tuple(hand.astype(int)), c, 5, cv2.LINE_AA)
        head_tilt = int(8 * np.sin(u * 4.0))
        cv2.circle(frame, (cx + head_tilt, cy - 20), 22, c, -1, cv2.LINE_AA)
        for dx in (-7, 7):
            cv2.circle(frame, (cx + head_tilt + dx, cy - 25), 3, (0, 0, 0), -1)
        cv2.ellipse(frame, (cx + head_tilt, cy - 14), (9, 5), 0, 0, 180, (0, 0, 0), 2, cv2.LINE_AA)
        for i in range(14):
            ang = u * 2.0 + i * (2 * np.pi / 14)
            rad = 90 + 20 * np.sin(u * 5 + i)
            px = int(cx + rad * np.cos(ang)); py = int(cy + rad * np.sin(ang))
            col = [(0, 255, 255), (0, 255, 0), (255, 0, 255), (255, 255, 0)][i % 4]
            cv2.circle(frame, (px, py), 5, col, -1, cv2.LINE_AA)
        _txt(frame, f"PLAYER {self.winner} WINS!", (cx - 170, cy - 90), 1.3, c, 3)
        _txt(frame, "press R to play again", (cx - 140, cy + 140), 0.7, (220, 220, 220), 2)


# =====================================================================
#  VICTORY JINGLE
# =====================================================================
class VictoryJingle:
    """Synthesises a short ascending arpeggio and plays it non-blocking.

    Tries winsound (Windows stdlib) first; falls back to pygame.mixer.
    If neither is available, .play() is a no-op.
    """
    def __init__(self, enabled=JINGLE_ENABLED, volume=JINGLE_VOLUME):
        self.enabled = enabled
        self.volume = float(np.clip(volume, 0.0, 1.0))
        self.backend = None
        self._winsound = None
        if not enabled:
            return
        try:
            import winsound
            self._winsound = winsound
            self.backend = "winsound"
        except Exception:
            try:
                import pygame
                pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
                self.backend = "pygame"
            except Exception:
                self.backend = None

    @staticmethod
    def _make_wav_bytes():
        sr = 44100
        notes = [523.25, 659.25, 783.99, 1046.50]   # C5 E5 G5 C6
        note_dur, gap = 0.18, 0.04
        samples = []
        for f in notes:
            n = int(sr * note_dur)
            t = np.arange(n) / sr
            env = np.clip(np.minimum(1.0, np.minimum(t / 0.01, (note_dur - t) / 0.05)), 0, 1)
            samples.append(0.5 * np.sin(2 * np.pi * f * t) * env)
            samples.append(np.zeros(int(sr * gap)))
        n = int(sr * 0.35)
        t = np.arange(n) / sr
        env = np.clip(np.minimum(1.0, np.minimum(t / 0.01, (0.35 - t) / 0.15)), 0, 1)
        samples.append(sum(0.18 * np.sin(2 * np.pi * f * t) for f in notes) * env)
        audio = np.clip(np.concatenate(samples), -1.0, 1.0)
        pcm = (audio * 32767).astype(np.int16)
        import struct
        data_bytes = pcm.tobytes()
        header = b"RIFF" + struct.pack("<I", 36 + len(data_bytes)) + b"WAVE"
        header += b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
        header += b"data" + struct.pack("<I", len(data_bytes))
        return header + data_bytes

    def play(self):
        if not self.enabled:
            return
        wav = self._make_wav_bytes()

        def _run():
            try:
                if self.backend == "winsound":
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                        f.write(wav); path = f.name
                    self._winsound.PlaySound(path, self._winsound.SND_FILENAME |
                                             self._winsound.SND_ASYNC)
                elif self.backend == "pygame":
                    import pygame, io
                    snd = pygame.mixer.Sound(io.BytesIO(wav))
                    snd.set_volume(self.volume); snd.play()
            except Exception as e:
                print(f"[jingle] playback failed: {e}")

        threading.Thread(target=_run, daemon=True).start()


# =====================================================================
#  SELF-TESTS  (no camera, no Task 2 files needed)
# =====================================================================
def run_self_tests(verbose=True):
    """Unit tests for the game rules. Pure, deterministic, fast."""
    results = []

    def check(name, cond):
        results.append((name, bool(cond)))
        if verbose:
            print(("PASS  " if cond else "FAIL  ") + name)

    PALM = synthetic_hand((300, 300), 100, True)
    FIST = synthetic_hand((300, 300), 100, False)

    def run(gc, t0, dur, fn, dt=33):
        evs, t = [], t0
        while t < t0 + dur:
            evs += gc.update(int(t), fn(t))
            t += dt
        return evs, t

    def of(evs, k):
        return [e for e in evs if e["type"] == k]

    def playing(target="please"):
        gc = GameController()
        gc.phase = "playing"
        gc.set_target(target, 0)
        return gc

    idle = PlayerInput()
    lock = lambda w: PlayerInput(word=w, conf=0.9, locked=True)

    gc = playing()
    _, t = run(gc, 0, 1000, lambda t: {1: idle, 2: idle})
    evs, t = run(gc, t, 400, lambda t: {1: lock("please"), 2: idle})
    evs += run(gc, t, 400, lambda t: {1: idle, 2: lock("please")})[0]
    check("first correct sign gets +10 (one score_event)",
          gc.players[1].score == POINTS_PER_WIN and len(of(evs, "score_event")) == 1)
    check("later correct sign in the same round is ignored", gc.players[2].score == 0)

    gc = playing()
    evs, _ = run(gc, 1000, 600, lambda t: {1: lock("please"), 2: lock("please")})
    check("simultaneous correct signs -> clash, half points each",
          len(of(evs, "clash")) == 1 and gc.players[1].score == gc.players[2].score == POINTS_PER_WIN // 2)

    gc = playing()
    evs, _ = run(gc, 1000, 300, lambda t: {1: lock("water"), 2: idle})
    check("wrong sign -> wrong_sign event, no points",
          len(of(evs, "wrong_sign")) == 1 and gc.players[1].score == 0)

    gc = playing()
    evs, _ = run(gc, 1000, 400, lambda t: {1: lock("PLEASE"), 2: idle})
    check("case-insensitive word match", gc.players[1].score == POINTS_PER_WIN)

    # defend: controller-level unit test
    gc = playing()
    gc.players[1].defend_active = True
    gc.pending = {1: (1200, 0.9)}
    gc._resolve_round(1400, [])
    check("defender gets half points",
          gc.players[1].score == int(POINTS_PER_WIN * DEFEND_SCORE_MULTIPLIER))

    hold = lambda t: {1: PlayerInput(hands=[PALM]), 2: PlayerInput(hands=[PALM])}
    gc = GameController()
    evs, _ = run(gc, 0, 1500, hold)
    check("both players hold open palm -> game_start",
          len(of(evs, "game_start")) == 1 and gc.phase == "playing")

    gc = GameController()
    evs, _ = run(gc, 0, 2000, lambda t: {1: PlayerInput(hands=[PALM]), 2: idle})
    check("only one player ready -> game does not start", gc.phase == "waiting")

    gc = GameController()
    run(gc, 0, 2000, lambda t: {p: PlayerInput(hands=[FIST]) for p in (1, 2)})
    check("closed fist does not count as ready", gc.phase == "waiting")

    n_ok = sum(ok for _, ok in results)
    print(f"\n{n_ok}/{len(results)} tests passed")
    return results


if __name__ == "__main__":
    run_self_tests()