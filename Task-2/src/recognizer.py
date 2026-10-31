"""Task 2 module: hand tracking + smoothing + sign recognition for N hands.

Usage (by the integrator / Task 5):

    from src.recognizer import SignRecognizer
    rec = SignRecognizer()                      # loads models from <project>/models/
    out = rec.update(frame_bgr, frame_id)       # frame_bgr: OpenCV BGR image
    out  -> exactly the team's agreed Task 2 -> Task 3 format:

        {"frame_id": 812,
         "hands": [ {"points": [[x, y, z] x 21],   # smoothed, normalised
                     "confidence": 0.94,
                     "handedness": "Right",         # the person's real hand
                     "box": [x, y, w, h]},          # normalised, top-left origin
                    ... ]}

Coordinate conventions
  * x, y are normalised 0..1 across the frame you passed in (origin top-left,
    y grows downward), z is MediaPipe's relative depth (wrist = 0, smaller =
    closer to the camera).
  * By default the frame is assumed to be the RAW camera image (not mirrored)
    and output coordinates refer to that same image. The recogniser flips it
    internally only because the classifier was trained on mirrored frames.
    If your pipeline already mirrors the frame before handing it to every
    module, construct with mirror_input=True and coordinates will refer to
    the mirrored frame.
  * Hands are listed left-to-right (by box centre x) in the output frame.
  * "handedness" is the person's real hand (MediaPipe's label is inverted for
    mirrored views; this is corrected).
  * "confidence" is MediaPipe's handedness score for that detection.

The agreed format has no field for the recognised *sign*, so it is exposed
separately (the format itself is untouched):

    rec.signs  -> list aligned 1:1 with out["hands"], each entry
        {"track_id": int,          # stable while that hand stays visible
         "guess": str | None,      # current best word (None while idle / not moving)
         "guess_confidence": float,
         "confidence": float,      # model probability of `guess` (use this one on screen / for scoring)
         "timestamp": float,       # time of this frame (the `timestamp` you passed, else time.time())
         "sign_start": float|None, # on a confirmed frame: when the hand started moving (estimate)
         "detect_seconds": float|None, # on a confirmed frame: timestamp - sign_start (>= ~0.8 s by design)
         "confirmed": bool,        # True ONLY on the frame a sign is confirmed
         "label": str | None,      # the confirmed word on that frame, else None
         "hand", "mirrored", "top3"}   # debugging: voted hand, whether features were mirrored, top-3 guesses
    rec.timing -> {"detect_ms", "recognize_ms", "total_ms"} for the last frame

    SignRecognizer(embed_sign=True) additionally puts "hand_id" and a "sign" object
    {label, guess, confidence, confirmed, detect_seconds, sign_start, timestamp} inside each
    hand dict. It is OFF by default so the output stays exactly the agreed format.

Smoothing is applied to the OUTPUT points only. The classifier is fed the raw
landmarks, i.e. exactly what it was trained on.
"""
import os
import time
from collections import deque

import cv2
import joblib
import mediapipe as mp
import numpy as np

from .features import build_clip_feature, extract_frame, mirror_raw63, motion_energy
from .hand_model import corrected_handedness, load_hand_landmarker
from .smoothing import OneEuroFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_HAND_MODEL = os.path.join(ROOT, "models", "hand_landmarker.task")
DEFAULT_CLASSIFIER = os.path.join(ROOT, "models", "sign_classifier.joblib")


class _Track:
    """One continuously visible hand: smoothing filter + recognition state."""

    def __init__(self, track_id, wrist, handedness, t, min_cutoff, beta):
        self.id = track_id
        self.wrist = wrist
        self.handedness = handedness
        self.last_seen = t
        self.filter = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        self.buffer = deque()            # (t, raw63, hand_scale)
        self.votes = deque(maxlen=15)    # recent Left/Right labels -> stable handedness
        self.last_confirmed = -1e9
        self.streak_label = None
        self.streak_start = 0.0
        self.move_start = None           # when the hand started moving (sign-start estimate)
        self.last_moving = -1e9


class SignRecognizer:
    def __init__(
        self,
        hand_model_path=DEFAULT_HAND_MODEL,
        classifier_path=DEFAULT_CLASSIFIER,
        num_hands=4,
        mirror_input=False,
        # recognition (same meaning/values as notebooks/04_live_recognition.ipynb)
        buffer_seconds=1.2,
        motion_threshold=0.03,
        min_span_fraction=0.5,
        confidence_threshold=0.30,
        confirm_seconds=0.2,
        cooldown_seconds=1.0,
        idle_label="Idle",
        # left/right handling: "auto" = mirror hands labelled Left (vote over recent frames);
        # "never"/"always" force it (testing); "best" = auto, but also try the opposite
        # orientation and use it when it is clearly more confident (guards against a wrong label)
        mirror_mode="auto",
        mirror_margin=0.15,
        # embed_sign=True adds "hand_id" and a "sign" object INSIDE each hand dict (off by default so the
        # output stays exactly the agreed Task 2 -> Task 3 format)
        embed_sign=False,
        # tracking
        track_timeout=0.3,
        max_link_dist=0.25,
        # smoothing (One-Euro): ~2.6x jitter reduction for ~18 ms lag, see README
        smooth_min_cutoff=0.5,
        smooth_beta=20.0,
        # hand detection: VIDEO mode tracks hands from frame to frame (fewer lost hands,
        # e.g. both hands touching for "More"); False = old per-frame IMAGE mode
        video_mode=True,
        min_detection_confidence=0.5,
    ):
        self.mirror_input = mirror_input
        self.buffer_seconds = buffer_seconds
        self.motion_threshold = motion_threshold
        self.min_span_fraction = min_span_fraction
        self.confidence_threshold = confidence_threshold
        self.confirm_seconds = confirm_seconds
        self.cooldown_seconds = cooldown_seconds
        self.idle_label = idle_label
        self.mirror_mode = mirror_mode
        self.mirror_margin = mirror_margin
        self.embed_sign = embed_sign
        self.track_timeout = track_timeout
        self.max_link_dist = max_link_dist
        self.smooth_min_cutoff = smooth_min_cutoff
        self.smooth_beta = smooth_beta

        self.video_mode = video_mode
        self._landmarker = load_hand_landmarker(model_path=hand_model_path, num_hands=num_hands,
                                                min_hand_detection_confidence=min_detection_confidence,
                                                video=video_mode)
        self._last_ms = -1               # VIDEO mode needs strictly increasing timestamps
        bundle = joblib.load(classifier_path)
        self._clf = bundle["model"]
        self._target_frames = bundle["target_frames"]
        self.classes = list(bundle["classes"])

        self._tracks = []
        self._next_id = 1
        self.signs = []
        self.timing = {"detect_ms": 0.0, "recognize_ms": 0.0, "total_ms": 0.0}

    # ------------------------------------------------------------------ API
    def update(self, frame_bgr, frame_id, timestamp=None):
        """Process one camera frame. Returns the agreed Task 2 -> Task 3 dict.

        timestamp: seconds (any monotonic clock). Defaults to time.time(). Pass
        your own if you replay video or want frame-accurate timing.
        """
        t_start = time.perf_counter()
        aspect = frame_bgr.shape[1] / frame_bgr.shape[0]
        t = time.time() if timestamp is None else float(timestamp)

        img = frame_bgr if self.mirror_input else cv2.flip(frame_bgr, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if self.video_mode:
            # strictly increasing ms, even if the caller's clock restarts (replaying a new video)
            self._last_ms = max(self._last_ms + 1, int(t * 1000))
            result = self._landmarker.detect_for_video(image, self._last_ms)
        else:
            result = self._landmarker.detect(image)
        t_detect = time.perf_counter()

        dets = []
        for lms, hd in zip(result.hand_landmarks, result.handedness):
            dets.append({
                "lms": lms,
                "wrist": np.array([lms[0].x, lms[0].y]),
                "handedness": corrected_handedness(hd[0].category_name),
                "score": float(hd[0].score),
            })

        tracks = self._associate(dets, t)

        entries = []  # (center_x, hand_dict, sign_dict)
        for det, trk in zip(dets, tracks):
            raw = np.array([[l.x, l.y, l.z] for l in det["lms"]])
            smooth = trk.filter(raw.ravel(), t).reshape(21, 3)
            out = smooth.copy()
            if not self.mirror_input:
                out[:, 0] = 1.0 - out[:, 0]          # back into the caller's frame
            x0, y0 = out[:, 0].min(), out[:, 1].min()
            x1, y1 = out[:, 0].max(), out[:, 1].max()
            x0, y0 = max(0.0, float(x0)), max(0.0, float(y0))
            x1, y1 = min(1.0, float(x1)), min(1.0, float(y1))
            hand = {
                "points": np.round(out, 5).tolist(),
                "confidence": round(det["score"], 4),
                "handedness": det["handedness"],
                "box": [round(x0, 5), round(y0, 5), round(x1 - x0, 5), round(y1 - y0, 5)],
            }
            sign = self._recognize(trk, det["lms"], t, aspect)
            if self.embed_sign:
                hand["hand_id"] = trk.id
                hand["sign"] = {k: sign[k] for k in ("label", "guess", "confidence", "confirmed",
                                                      "detect_seconds", "sign_start", "timestamp")}
            entries.append(((x0 + x1) / 2.0, hand, sign))

        entries.sort(key=lambda e: e[0])             # left-to-right
        self.signs = [e[2] for e in entries]

        t_end = time.perf_counter()
        self.timing = {
            "detect_ms": (t_detect - t_start) * 1000.0,
            "recognize_ms": (t_end - t_detect) * 1000.0,
            "total_ms": (t_end - t_start) * 1000.0,
        }
        return {"frame_id": frame_id, "hands": [e[1] for e in entries]}

    def reset(self):
        """Forget all tracked hands and their histories (e.g. between game rounds
        or when replaying a new video)."""
        self._tracks = []
        self.signs = []

    def close(self):
        self._landmarker.close()

    # ------------------------------------------------------------- tracking
    def _associate(self, dets, t):
        """Match this frame's detections to live tracks (nearest wrist,
        preferring the same handedness); unmatched detections start new tracks;
        tracks unseen for track_timeout seconds are dropped."""
        self._tracks = [k for k in self._tracks if t - k.last_seen <= self.track_timeout]
        pairs = []
        for di, d in enumerate(dets):
            for ki, k in enumerate(self._tracks):
                dist = float(np.linalg.norm(k.wrist - d["wrist"]))
                if k.handedness != d["handedness"]:
                    dist += 0.15
                if dist < self.max_link_dist:
                    pairs.append((dist, di, ki))
        pairs.sort()
        assigned, used_d, used_k = [None] * len(dets), set(), set()
        for _, di, ki in pairs:
            if di in used_d or ki in used_k:
                continue
            assigned[di] = self._tracks[ki]
            used_d.add(di)
            used_k.add(ki)
        for di, d in enumerate(dets):
            if assigned[di] is None:
                trk = _Track(self._next_id, d["wrist"], d["handedness"], t,
                             self.smooth_min_cutoff, self.smooth_beta)
                self._next_id += 1
                self._tracks.append(trk)
                assigned[di] = trk
            trk = assigned[di]
            trk.wrist = d["wrist"]
            trk.handedness = d["handedness"]
            trk.votes.append(d["handedness"])
            trk.last_seen = t
        return assigned

    # --------------------------------------------------------- recognition
    def _recognize(self, trk, lms, t, aspect):
        info = {"track_id": trk.id, "guess": None, "guess_confidence": 0.0, "confidence": 0.0,
                "confirmed": False, "label": None,
                "timestamp": round(float(t), 4), "sign_start": None, "detect_seconds": None}
        raw63, scale = extract_frame(lms, aspect)
        trk.buffer.append((t, raw63, scale))
        while trk.buffer and t - trk.buffer[0][0] > self.buffer_seconds:
            trk.buffer.popleft()

        # sign-start estimate: the moment the hand began moving after being still (0.3 s hysteresis)
        if len(trk.buffer) >= 4:
            last4 = list(trk.buffer)[-4:]
            if motion_energy([b[1] for b in last4]) >= self.motion_threshold:
                if trk.move_start is None:
                    trk.move_start = last4[0][0]
                trk.last_moving = t
        if trk.move_start is not None and t - trk.last_moving > 0.3:
            trk.move_start = None

        if len(trk.buffer) < 5:
            trk.streak_label = None
            return info
        times = [b[0] for b in trk.buffer]
        feats = [b[1] for b in trk.buffer]
        scales = [b[2] for b in trk.buffer]
        span = times[-1] - times[0]
        if motion_energy(feats) < self.motion_threshold or span < self.buffer_seconds * self.min_span_fraction:
            trk.streak_label = None
            return info

        voted_left = sum(1 for v in trk.votes if v == "Left") * 2 > len(trk.votes)
        if self.mirror_mode == "never":
            mirrored = False
        elif self.mirror_mode == "always":
            mirrored = True
        else:
            mirrored = voted_left            # left hand -> right-hand form (training is mostly right-handed)

        def probs_for(m):
            f = [mirror_raw63(x) for x in feats] if m else feats
            seq = build_clip_feature(f, scales, times, self._target_frames)
            return None if seq is None else self._clf.predict_proba([seq])[0]

        probs = probs_for(mirrored)
        if probs is None:
            trk.streak_label = None
            return info
        if self.mirror_mode == "best":
            alt = probs_for(not mirrored)
            if alt is not None and alt.max() > probs.max() + self.mirror_margin:
                probs, mirrored = alt, not mirrored

        order = np.argsort(probs)[::-1][:3]
        best = int(order[0])
        label, conf = self._clf.classes_[best], float(probs[best])
        info["guess"], info["guess_confidence"] = str(label), round(conf, 4)
        info["confidence"] = round(conf, 4)
        info["hand"] = "Left" if voted_left else "Right"
        info["mirrored"] = bool(mirrored)
        info["top3"] = [(str(self._clf.classes_[i]), round(float(probs[i]), 3)) for i in order]

        if label != self.idle_label and conf >= self.confidence_threshold:
            if label == trk.streak_label:
                held = t - trk.streak_start
            else:
                trk.streak_label, trk.streak_start, held = label, t, 0.0
            if held >= self.confirm_seconds and (t - trk.last_confirmed) >= self.cooldown_seconds:
                trk.last_confirmed = t
                trk.streak_label = None
                start = trk.move_start if trk.move_start is not None else times[0]
                trk.buffer.clear()               # don't re-trigger on this gesture's tail
                trk.move_start = None            # the next sign starts a new attempt
                info["confirmed"], info["label"] = True, str(label)
                info["sign_start"] = round(float(start), 4)
                info["detect_seconds"] = round(max(0.0, float(t - start)), 3)
        else:
            trk.streak_label = None
        return info
