"""
Task 3 - Player association + MediaPipe hand detection (IPCV Project).

Plug into Rush's Task 1 live loop. Provides:
  1. faces_from_rush_players(): adapter for Rush's PlayerState dict.
  2. MediaPipeHandsDetector: real hand detection with MediaPipe Hands
     (21 landmarks per hand). Same output shape as Nived's future module.
  3. SimpleHandDetector: skin-colour fallback, used only if mediapipe
     is not installed.
  4. PlayerAssociator: body shapes (circles) that claim hands and emit the
     Task 3 JSON contract.
  5. draw_debug(): draws the body-shape circles + hand landmarks on the frame.

Hand shape produced (and expected from Nived later):
  {"center": [x, y], "sign": {...} or None, "handedness": "Left"/"Right" or None}
"""

import math

# --- associator tunables (image pixels; frame is 1280x720 in Rush's config) ---
MIN_RADIUS = 110.0        # starting body radius
MAX_RADIUS = 300.0
GROW_RATE = 0.25
CAPTURE_MARGIN = 1.4
DECAY = 0.997


def _face_center(face_box):
    x, y, w, h = face_box
    return (x + w / 2.0, y + h / 2.0)


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ---------------------------------------------------------------------------
# 1. Adapter for Rush's output
# ---------------------------------------------------------------------------
def faces_from_rush_players(players):
    """{player_id: PlayerState} -> flat face list for the associator."""
    faces = []
    for pid, p in players.items():
        if getattr(p, "smooth_box", None) is None:
            continue
        faces.append({
            "player_id": pid,
            "face_box": tuple(p.smooth_box),
            "present": p.visible,
            "status": p.status,
        })
    return faces


# ---------------------------------------------------------------------------
# 2. MediaPipe hand detection (Tasks API - mediapipe >= 0.10 / 1.x)
# ---------------------------------------------------------------------------
HAND_LANDMARKER_URL = ("https://storage.googleapis.com/mediapipe-models/"
                       "hand_landmarker/hand_landmarker/float16/latest/"
                       "hand_landmarker.task")
HAND_LANDMARKER_PATH = "hand_landmarker.task"   # next to the script


class MediaPipeHandsDetector:
    """
    MediaPipe Hand Landmarker (the Tasks API). Finds up to `max_hands`
    hands per frame with 21 landmarks each. Runs on CPU in real time.

    Notes:
      - The model file (hand_landmarker.task) is downloaded automatically on
        first use (~7.5 MB) next to this script.
      - We feed it the same (already flipped) frame that Rush's face pipeline
        uses, so landmark coordinates match everything on screen.
      - The Landmarker's "handedness" label assumes an UNFLIPPED image. With
        Rush's MIRROR_VIEW flip the label is inverted, so we leave
        "handedness" as None and let the associator infer the side by
        comparing the hand center with the player's face x. Nived must
        account for the same flip in his module.
      - The hand "center" is the mean of all 21 landmarks; the associator
        only needs one stable point per hand.
    """

    def __init__(self, max_hands=4, min_confidence=0.5,
                 model_path=HAND_LANDMARKER_PATH):
        import os
        import urllib.request
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        if not os.path.exists(model_path):
            print(f"[task3] downloading hand_landmarker.task ...")
            urllib.request.urlretrieve(HAND_LANDMARKER_URL, model_path)

        self._mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            num_hands=max_hands,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame, face_boxes):
        """
        frame: BGR image (the same flipped frame the faces were found in).
        face_boxes: unused here (kept for the shared detector interface);
        the ML model does not confuse faces with hands.
        Returns [{"center": [x,y], "sign": None, "handedness": None,
                  "points": [[x,y], ...21]}]
        """
        import cv2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                  data=rgb)
        result = self._landmarker.detect(mp_image)

        hands = []
        h, w = frame.shape[:2]
        for hl in result.hand_landmarks:
            pts = [[p.x * w, p.y * h] for p in hl]
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            hands.append({
                "center": [float(cx), float(cy)],
                "sign": None,           # Nived's classifier fills this
                "handedness": None,     # inferred by the associator
                "points": pts,          # 21 landmarks, kept for drawing
            })
        return hands


# ---------------------------------------------------------------------------
# 2b. Skin-colour fallback (only if mediapipe is not installed)
# ---------------------------------------------------------------------------
class SimpleHandDetector:
    """Skin-colour blob detector. Same detect() interface, no ML."""

    def __init__(self):
        import cv2
        import numpy as np
        self.cv2, self.np = cv2, np
        self.cr_min, self.cr_max = 133, 180
        self.cb_min, self.cb_max = 77, 133
        self.min_area = 1500
        self.max_area = 60000

    def detect(self, frame, face_boxes):
        cv2, np = self.cv2, self.np
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb,
                           np.array([0, self.cr_min, self.cb_min]),
                           np.array([255, self.cr_max, self.cb_max]))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
        hands = []
        for c in contours:
            area = cv2.contourArea(c)
            if not (self.min_area <= area <= self.max_area):
                continue
            m = cv2.moments(c)
            if m["m00"] == 0:
                continue
            cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
            if any(fx <= cx <= fx + fw and fy <= cy <= fy + fh
                   for fx, fy, fw, fh in face_boxes):
                continue
            hands.append({"center": [float(cx), float(cy)],
                          "sign": None, "handedness": None,
                          "points": None})
        return hands


POSE_LANDMARKER_URL = ("https://storage.googleapis.com/mediapipe-models/"
                       "pose_landmarker/pose_landmarker_lite/float16/"
                       "latest/pose_landmarker_lite.task")
POSE_LANDMARKER_PATH = "pose_landmarker_lite.task"   # next to the script

# indices of useful landmarks in the 33-point pose skeleton
POSE_NOSE = 0
POSE_L_SHOULDER, POSE_R_SHOULDER = 11, 12
POSE_L_ELBOW, POSE_R_ELBOW = 13, 14
POSE_L_WRIST, POSE_R_WRIST = 15, 16
POSE_L_HIP, POSE_R_HIP = 23, 24


class PoseBodyShapes:
    """
    MediaPipe Pose Landmarker (Tasks API). Detects up to `num_poses` people
    and builds, per person, a body REGION (a polygon) plus their wrists.

    Per-person output (what the associator consumes):
      {
        "region":   [(x,y), ...]        polygon: convex hull of shoulders,
                                        elbows, hips and wrists
        "wrists":   {"Left": (x,y) or None, "Right": (x,y) or None}
                                        NOTE: MediaPipe's Left/Right labels
                                        are inverted by the mirror flip, so
                                        the associator only uses them for
                                        proximity, never for naming.
        "nose":     (x, y) or None      used to match the pose to a face
      }
    """

    def __init__(self, num_poses=2, min_confidence=0.5,
                 model_path=POSE_LANDMARKER_PATH):
        import os
        import urllib.request
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        if not os.path.exists(model_path):
            print(f"[task3] downloading pose_landmarker_lite.task ...")
            urllib.request.urlretrieve(POSE_LANDMARKER_URL, model_path)

        self._mp = mp
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            num_poses=num_poses,
            min_pose_detection_confidence=min_confidence,
            min_pose_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def detect(self, frame):
        """frame: BGR (the same flipped frame). Returns the list of bodies."""
        import cv2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                  data=rgb)
        result = self._landmarker.detect(mp_image)

        bodies = []
        for lm_list in result.pose_landmarks:
            pts = [(p.x * frame.shape[1], p.y * frame.shape[0])
                   for p in lm_list]
            body = {
                "region": _body_region(pts),
                "nose": pts[POSE_NOSE],
                "wrists": {"Left": pts[POSE_L_WRIST], "Right": pts[POSE_R_WRIST]},
            }
            bodies.append(body)
        return bodies


def _body_region(points, grow=1.15):
    """
    Convex hull of the torso + arms keypoints (shoulders, elbows, hips,
    wrists). Enlarged slightly so hands near the wrist stay inside even with
    landmark jitter. Uses cv2.convexHull; falls back to the raw points if
    cv2 is unavailable (e.g. in tests).
    """
    torso = [points[i] for i in (POSE_L_SHOULDER, POSE_R_SHOULDER,
                                 POSE_L_ELBOW, POSE_R_ELBOW,
                                 POSE_L_HIP, POSE_R_HIP,
                                 POSE_L_WRIST, POSE_R_WRIST)]
    try:
        import cv2
        import numpy as np
        arr = np.array([[int(x), int(y)] for x, y in torso], dtype=np.int32)
        hull = cv2.convexHull(arr).reshape(-1, 2)
        cx = hull[:, 0].mean()
        cy = hull[:, 1].mean()
        region = [((x - cx) * grow + cx, (y - cy) * grow + cy)
                  for x, y in hull]
    except ImportError:
        region = torso
    return [(float(x), float(y)) for x, y in region]


def _point_in_region(point, region):
    """Ray-casting point-in-polygon test. region is a convex hull list."""
    x, y = point
    inside = False
    n = len(region)
    for i in range(n):
        x1, y1 = region[i]
        x2, y2 = region[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            t = (y - y1) / (y2 - y1)
            if x < x1 + t * (x2 - x1):
                inside = not inside
    return inside


def _region_center(region):
    if not region:
        return (0.0, 0.0)
    return (sum(p[0] for p in region) / len(region),
            sum(p[1] for p in region) / len(region))


def _region_distance(point, region):
    """Distance from a point to the nearest edge/vertex of the polygon."""
    x, y = point
    n = len(region)
    best = float("inf")
    for i in range(n):
        x1, y1 = region[i]
        x2, y2 = region[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        length2 = dx * dx + dy * dy
        if length2 == 0:
            d = math.hypot(x - x1, y - y1)
        else:
            t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / length2))
            d = math.hypot(x - (x1 + t * dx), y - (y1 + t * dy))
        best = min(best, d)
    return best


def make_hand_detector():
    """Return the MediaPipe Tasks detector, or the skin-colour fallback."""
    try:
        return MediaPipeHandsDetector()
    except (ImportError, AttributeError, OSError, ValueError) as e:
        print(f"[task3] mediapipe unusable ({e}) - "
              "using skin-colour fallback.")
        return SimpleHandDetector()


# ---------------------------------------------------------------------------
# 3. Body shapes + association
# ---------------------------------------------------------------------------
class BodyShape:
    def __init__(self, center, radius=MIN_RADIUS):
        self.center = list(center)
        self.radius = radius

    def update_anchor(self, face_center, k=0.5):
        self.center[0] += k * (face_center[0] - self.center[0])
        self.center[1] += k * (face_center[1] - self.center[1])

    def absorb(self, point):
        r = _dist(self.center, point)
        if r > self.radius:
            self.radius = min(self.radius * (1 + GROW_RATE * r / self.radius),
                              MAX_RADIUS)

    def contains(self, point, margin=1.0):
        return _dist(self.center, point) <= self.radius * margin

    def distance(self, point):
        return _dist(self.center, point)

    def decay(self):
        self.radius = max(MIN_RADIUS, self.radius * DECAY)


class PlayerAssociator:
    def __init__(self):
        self.shapes = {}            # player_id -> BodyShape (fallback)
        self.body_regions = {}     # player_id -> polygon (pose, this frame)
        self._body_wrists = {}     # player_id -> wrist points (pose)
        self.last_owner = {}        # hand index -> player_id (stickiness)
        self.current_owners = {}    # hand index -> player_id (this frame, for drawing)

    def process(self, frame_id, faces, hands, bodies=None):
        """
        faces: from faces_from_rush_players; hands: from detector/Nived.
        bodies: optional list from PoseBodyShapes.detect(). When present,
        each player's body region is the pose polygon matched to their face,
        and hands are attributed by region + wrist proximity. When absent,
        the circular BodyShape fallback is used.
        """
        frame_faces = {}
        for f in faces:
            pid = f["player_id"]
            fc = _face_center(f["face_box"])
            frame_faces[pid] = fc
            if pid not in self.shapes:
                self.shapes[pid] = BodyShape(fc)
            else:
                self.shapes[pid].update_anchor(fc)

        # --- match poses to players via the nose/face ---
        self.body_regions = {}
        used_bodies = set()
        for pid, fc in frame_faces.items():
            best, best_d = None, None
            for b, body in enumerate(bodies or []):
                if b in used_bodies or body.get("nose") is None:
                    continue
                d = _dist(body["nose"], fc)
                if best_d is None or d < best_d:
                    best, best_d = b, d
            if best is not None:
                # accept the pose only if its nose is near the face
                fw = faces[0]["face_box"][2]
                if best_d <= max(150.0, 2.0 * fw):
                    self.body_regions[pid] = bodies[best]["region"]
                    self._body_wrists[pid] = bodies[best]["wrists"]
                    used_bodies.add(best)

        owners = {}
        self.current_owners = {}
        for i, hand in enumerate(hands):
            pid = self._owner_of(hand, i, frame_faces)
            if pid is not None:
                owners.setdefault(pid, []).append(i)
                self.current_owners[i] = pid

        players = []
        for pid in sorted(self.shapes):
            entry = next((f for f in faces if f["player_id"] == pid), None)
            present = bool(entry.get("present", True)) if entry else False
            shape = self.shapes[pid]
            hand_left, hand_right = self._split_hands(pid, owners, hands,
                                                      frame_faces)
            for hi in owners.get(pid, []):
                shape.absorb(hands[hi]["center"])
            shape.decay()

            players.append({
                "player_id": pid,
                "present": present,
                "position": [round(shape.center[0], 1),
                             round(shape.center[1], 1)],
                "hand_left": hand_left,
                "hand_right": hand_right,
            })

        return {"frame_id": frame_id, "players": players}

    def _owner_of(self, hand, index, frame_faces):
        point = hand["center"]

        # --- primary path: pose body regions ---
        if self.body_regions:
            inside = [pid for pid, region in self.body_regions.items()
                      if _point_in_region(point, region)]
            if len(inside) == 1:
                return inside[0]
            if len(inside) > 1:
                # inside both regions: previous owner wins, else nearest face
                sticky = self.last_owner.get(index)
                if sticky in inside:
                    return sticky
                if frame_faces:
                    return min(inside,
                               key=lambda p: _dist(frame_faces[p], point))
                return min(inside,
                           key=lambda p: _region_distance(point,
                                                          self.body_regions[p]))
            # inside no region: nearest wrist decides (hand slightly outside
            # its owner's hull is still attached to their arm)
            wrist_best, wrist_d = None, None
            for pid, wrists in self._body_wrists.items():
                for side, wp in wrists.items():
                    if wp is None:
                        continue
                    d = _dist(point, wp)
                    if wrist_d is None or d < wrist_d:
                        wrist_best, wrist_d = pid, d
            if wrist_best is not None and wrist_d <= 120.0:
                return wrist_best
            # no wrist nearby: fall through to the circle fallback

        # --- fallback path: circular body shapes ---
        cands = [pid for pid, sh in self.shapes.items()
                 if sh.contains(point, margin=CAPTURE_MARGIN)]
        if not cands:
            if not self.shapes:
                return None
            pid, sh = min(self.shapes.items(),
                          key=lambda kv: kv[1].distance(point))
            return pid if sh.contains(point, CAPTURE_MARGIN) else None
        if len(cands) == 1:
            return cands[0]
        sticky = self.last_owner.get(index)
        if sticky in cands:
            return sticky
        if frame_faces:
            return min(cands, key=lambda p: _dist(frame_faces[p], point))
        return min(cands, key=lambda p: self.shapes[p].distance(point))

    def _split_hands(self, pid, owners, hands, frame_faces):
        left = right = None
        face = frame_faces.get(pid)
        for i in owners.get(pid, []):
            hand = hands[i]
            h = hand.get("handedness")
            side = h if h in ("Left", "Right") else (
                "Left" if face is not None and hand["center"][0] < face[0]
                else "Right")
            payload = {"sign": hand.get("sign"),
                       "center": [round(hand["center"][0], 1),
                                  round(hand["center"][1], 1)]}
            if side == "Left" and left is None:
                left = payload
            elif side == "Right" and right is None:
                right = payload
        return left, right


# ---------------------------------------------------------------------------
# 4. Debug drawing (body-shape circles + hand landmarks)
# ---------------------------------------------------------------------------
PLAYER_DEBUG_COLORS = {1: (255, 140, 0), 2: (40, 40, 230)}   # BGR, matches Rush


def draw_debug(display, assoc, hands=None):
    """Draw body polygons (or circles), hand squares and skeletons."""
    import cv2
    import numpy as np
    for pid, shape in assoc.shapes.items():
        color = PLAYER_DEBUG_COLORS.get(pid, (0, 255, 0))
        region = assoc.body_regions.get(pid)
        if region:
            poly = np.array([[int(x), int(y)] for x, y in region],
                            dtype=np.int32)
            cv2.polylines(display, [poly], True, color, 2, cv2.LINE_AA)
        else:
            cv2.circle(display, (int(shape.center[0]), int(shape.center[1])),
                       int(shape.radius), color, 2, cv2.LINE_AA)
            cv2.circle(display, (int(shape.center[0]), int(shape.center[1])), 3,
                       color, -1)
    if hands:
        for i, hand in enumerate(hands):
            x, y = int(hand["center"][0]), int(hand["center"][1])
            pid = assoc.current_owners.get(i)
            color = PLAYER_DEBUG_COLORS.get(pid, (120, 120, 120))
            # filled square in the owning player's colour (grey = unassigned)
            size = 22
            cv2.rectangle(display, (x - size // 2, y - size // 2),
                          (x + size // 2, y + size // 2), color, -1, cv2.LINE_AA)
            cv2.rectangle(display, (x - size // 2, y - size // 2),
                          (x + size // 2, y + size // 2), (255, 255, 255), 1, cv2.LINE_AA)
            label = f"P{pid}" if pid is not None else "?"
            cv2.putText(display, label, (x + size // 2 + 4, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            if hand.get("points"):
                for px, py in hand["points"]:
                    cv2.circle(display, (int(px), int(py)), 2,
                               color, -1)


# ---------------------------------------------------------------------------
# Self-test (association logic only; no mediapipe/cv2 needed)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    class FakePlayerState:
        def __init__(self, pid, box, status):
            self.player_id = pid
            self.smooth_box = box
            self.status = status
            self.visible = status in ("TRACKED", "HELD")

    rush_players = {1: FakePlayerState(1, (200, 100, 80, 80), "TRACKED"),
                    2: FakePlayerState(2, (520, 120, 80, 80), "TRACKED")}
    faces = faces_from_rush_players(rush_players)
    hands = [
        {"center": [280, 260], "sign": None, "handedness": None, "points": None},
        {"center": [180, 270], "sign": None, "handedness": None, "points": None},
        {"center": [560, 280], "sign": None, "handedness": None, "points": None},
    ]

    # --- test 1: circle fallback (no pose) ---
    assoc = PlayerAssociator()
    for fid in range(810, 813):
        out = assoc.process(fid, faces, hands)
    assert out["players"][0]["hand_left"] is not None, "circle: P1 left hand"
    assert out["players"][0]["hand_right"] is not None, "circle: P1 right hand"
    assert out["players"][1]["hand_right"] is not None, "circle: P2 right hand"
    assert out["players"][0]["hand_left"]["center"][0] < 240, "left is left of face"
    print("test 1 OK: circle fallback")

    # --- test 2: pose regions (primary path) ---
    # body 1: hull around x 120..300, nose inside P1's face
    # body 2: hull around x 480..660, nose inside P2's face
    def fake_body(x0, x1, y0, y1, nose):
        region = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        return {"region": region, "nose": nose,
                "wrists": {"Left": (x0 + 30, y1 - 10),
                           "Right": (x1 - 30, y1 - 10)}}

    bodies = [fake_body(120, 300, 120, 400, (240, 140)),
              fake_body(480, 660, 140, 420, (560, 160))]
    hands_pose = [
        {"center": [280, 260], "sign": None, "handedness": None, "points": None},
        {"center": [180, 270], "sign": None, "handedness": None, "points": None},
        {"center": [560, 280], "sign": None, "handedness": None, "points": None},
    ]
    assoc2 = PlayerAssociator()
    out = assoc2.process(820, faces, hands_pose, bodies=bodies)
    assert assoc2.current_owners == {0: 1, 1: 1, 2: 2}, assoc2.current_owners
    assert out["players"][0]["hand_left"] is not None, "pose: P1 left"
    assert out["players"][0]["hand_right"] is not None, "pose: P1 right"
    assert out["players"][1]["hand_right"] is not None, "pose: P2 right"
    print("test 2 OK: pose regions attribute hands correctly")

    # --- test 3: hand outside both hulls, near P2's wrist ---
    hands_cross = [
        {"center": [430, 410], "sign": None, "handedness": None, "points": None},
    ]
    assoc3 = PlayerAssociator()
    out = assoc3.process(821, faces, hands_cross, bodies=bodies)
    assert assoc3.current_owners.get(0) == 2, assoc3.current_owners
    print("test 3 OK: outside-hull hand claimed by nearest wrist (P2)")

    # --- test 4: geometry helpers ---
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert _point_in_region((5, 5), square)
    assert not _point_in_region((15, 5), square)
    assert abs(_region_distance((15, 5), square) - 5) < 1e-6
    assert abs(_region_center(square)[0] - 5) < 1e-6
    print("test 4 OK: geometry helpers")

    print("\nAll tests passed. Sample output (pose path):")
    print(json.dumps(out, indent=2))