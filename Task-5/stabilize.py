"""
Task 5 - steadier tracking on top of Task 1 (faces) and Task 3 (hands).
Uses the teams' own functions; none of their code is changed.

FaceTracks replaces tracking.update_all_tracks() with three changes:
  * Smoothing: a One Euro filter (Task 2's smoothing.py) on the face centre and
    size instead of a fixed EMA. Strong smoothing while still, little lag
    while moving. Centre and size have separate filters (size changes slowly).
  * Jump guard (identity hysteresis): a tracked player is only moved to a
    recognised face far away (> FAR face widths) after recognition has agreed
    for SWITCH_FRAMES frames in a row. One bad frame cannot swap identities.
  * Guarded overlap fallback: an unrecognised player may keep an overlapping
    face only if that face does not look more like the other player.

HandOwners makes Task 3's hand -> player decision sticky. It is keyed by
Task 2's track_id (stable while a hand is visible) instead of the hand's
position in the list, and a hand only changes owner after Task 3 has picked
the new owner HAND_SWITCH_FRAMES frames in a row.
"""
import math

import config
import tracking
from src.smoothing import OneEuroFilter

FACE_POS = {"min_cutoff": 0.3, "beta": 0.01}      # centre, pixels (tuned: see below)
FACE_SIZE = {"min_cutoff": 0.3, "beta": 0.005}    # width/height, pixels
# Tuned on a simulated face (4 px detector noise, 30 fps): versus Task 1's
# EMA (alpha 0.3): somewhat less jitter when still (0.8-1.2 vs 1.5 px) and a
# third of the lag when moving at 300 px/s (9 vs 23 px).
FAR = 1.5                  # face widths: further than this counts as a jump
SWITCH_FRAMES = 5          # frames of agreement needed before a jump is accepted
HAND_SWITCH_FRAMES = 5     # frames of agreement needed before a hand changes owner
HAND_HOLD_FRAMES = 10      # frames a hand keeps its owner while Task 3 gives none


def _center(box):
    x, y, w, h = box
    return x + w / 2, y + h / 2


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class FaceTracks:
    def __init__(self, players):
        self.players = players
        self._pos = {pid: OneEuroFilter(**FACE_POS) for pid in players}
        self._size = {pid: OneEuroFilter(**FACE_SIZE) for pid in players}
        self._box = {}                              # pid -> filtered (x, y, w, h)
        self._far = {pid: 0 for pid in players}     # frames a far face was recognised

    def update(self, faces, assignments, similarities, frame_id, t):
        was_visible = {pid: p.visible for pid, p in self.players.items()}
        matched, used = {}, set()
        for pid, (f, sim) in self._guard_jumps(faces, assignments).items():
            matched[pid] = (f, sim, "recognized")
            used.add(f)
        for pid, p in self.players.items():
            if pid in matched or not p.visible:
                continue
            f = self._overlap_face(pid, p, faces, used, similarities)
            if f is not None:
                sim = float(similarities[f, pid - 1]) if similarities is not None else p.similarity
                matched[pid] = (f, sim, "overlap")
                used.add(f)
        for pid, p in self.players.items():
            if pid in matched:
                f, sim, source = matched[pid]
                tracking.update_player_track(p, faces[f], sim, source, frame_id)
                self._smooth(pid, p, was_visible[pid], t)
            else:
                tracking.mark_player_missing(p)
            if p.visible and pid in self._box:
                p.smooth_box = self._box[pid]       # replaces Task 1's EMA box

    def _guard_jumps(self, faces, assignments):
        accepted = {}
        for pid, p in self.players.items():
            if pid not in assignments:
                self._far[pid] = 0
                continue
            f, sim = assignments[pid]
            if p.visible and _dist(_center(faces[f]["box"]), p.center) > FAR * p.face_width:
                self._far[pid] += 1
                if self._far[pid] < SWITCH_FRAMES:
                    continue                        # not convinced yet: ignore this frame
            self._far[pid] = 0
            accepted[pid] = (f, sim)
        return accepted

    @staticmethod
    def _overlap_face(pid, p, faces, used, similarities):
        best_f, best_iou = None, 0.0
        for f, face in enumerate(faces):
            if f in used:
                continue
            if similarities is not None and similarities[f, 2 - pid] > similarities[f, pid - 1]:
                continue                            # looks more like the other player
            iou = tracking.box_iou(p.raw_box, face["box"])
            if iou > best_iou:
                best_f, best_iou = f, iou
        return best_f if best_iou >= config.TRACK_IOU_THRESHOLD else None

    def _smooth(self, pid, p, was_visible, t):
        x, y, w, h = p.raw_box
        c = (x + w / 2, y + h / 2)
        last = self._box.get(pid)
        if (not was_visible or last is None
                or _dist(c, _center(last)) > config.SMOOTHING_RESET_DISTANCE * last[2]):
            self._pos[pid].reset()                  # new or re-found face: start fresh
            self._size[pid].reset()
        cx, cy = self._pos[pid](c, t)
        sw, sh = self._size[pid]((w, h), t)
        self._box[pid] = (cx - sw / 2, cy - sh / 2, sw, sh)


class HandOwners:
    def __init__(self):
        self._tracks = {}   # Task 2 track_id -> {"owner", "challenger", "count", "none"}

    def update(self, owners, signs):
        """owners: Task 3 {hand index: pid} this frame; signs: Task 2 rec.signs
        (same order as the hands). Returns the sticky {hand index: pid}."""
        stable, seen = {}, set()
        for i, s in enumerate(signs):
            tid, new = s.get("track_id"), owners.get(i)
            if tid is None:
                if new is not None:
                    stable[i] = new
                continue
            seen.add(tid)
            st = self._tracks.get(tid)
            if st is None:
                if new is not None:
                    self._tracks[tid] = {"owner": new, "challenger": None, "count": 0, "none": 0}
                    stable[i] = new
                continue
            if new == st["owner"]:
                st.update(challenger=None, count=0, none=0)
            elif new is None:
                st["none"] += 1
                if st["none"] > HAND_HOLD_FRAMES:
                    del self._tracks[tid]
                    continue
            else:
                st["none"] = 0
                st["count"] = st["count"] + 1 if st["challenger"] == new else 1
                st["challenger"] = new
                if st["count"] >= HAND_SWITCH_FRAMES:
                    st.update(owner=new, challenger=None, count=0)
            stable[i] = st["owner"]
        for tid in [tid for tid in self._tracks if tid not in seen]:
            del self._tracks[tid]                   # hand left the picture
        return stable
