# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 13:17:03 2026

@author: ruvah
"""

"""
Sign WORD classifier (Task 4) for the WLASL video dataset
=========================================================
Words are signed with MOVEMENT, so one frame is not enough. Pipeline:

  video --MediaPipe HandLandmarker (every frame)--> up to 2 hands x 21 points
        --per-frame features (hand shape + hand position)-->
        --cut out the part where hands are visible ("the sign")-->
        --resample to exactly T frames--> one long feature vector
        --Random Forest / SVM--> word + confidence

Run it in Spyder CELL BY CELL (Ctrl+Enter). Cells start with "# %%".
Order: 0 settings -> 1 helpers -> 2 look at dataset -> 3 extract (slow, once)
       -> 4 train -> 5 live test

Install once:
    pip install mediapipe opencv-python scikit-learn pandas joblib matplotlib
"""

# %% Cell 0: SETTINGS  (edit these)
import os

DATASET_DIR = r"C:\Users\ruvah\OneDrive - University of Twente\Desktop\IPCV Assignments\IPCV Project\WLASL_Dataset"   # <-- CHANGE: folder with the json + videos

# Which words to learn. Option 1: choose yourself (look at Cell 2 output first).
WORDS = None            # e.g. ["book", "drink", "thank you"]   (None = automatic)
# Option 2 (used when WORDS is None): the N words that have the most videos.
N_WORDS = 10
MIN_VIDEOS_PER_WORD = 8

FRAME_STRIDE = 2        # process every 2nd frame (faster). 1 = every frame.
MAX_VIDEO_SECONDS = 6   # ignore anything after this
USE_JSON_TRIM = False   # True = cut videos with frame_start/frame_end from the json
N_STEPS = 16            # every sign is resampled to this many time steps
USE_MIRROR_AUGMENTATION = True   # also train on mirrored signs (left-handed signers)
REFIT_ON_ALL_DATA = True         # after testing, retrain on everything for the game

# Live / game behaviour
MIN_CONFIDENCE = 0.50   # ignore guesses below this
HOLD_MS = 300           # same word must stay the guess this long to "lock"
MIN_SIGN_MS = 400       # shorter hand appearances are ignored
GAP_MS = 400            # hands may vanish this long inside one sign
END_GRACE_MS = 700      # keep judging this long after the hands drop
PREDICT_EVERY_MS = 200  # how often the live loop re-classifies
BUFFER_MS = 5000

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
OUT_DIR = os.path.join(HERE, "outputs_words")
MODEL_TASK_FILE = os.path.join(OUT_DIR, "hand_landmarker.task")
SEQUENCES_FILE = os.path.join(OUT_DIR, "sequences.pkl")
CLASSIFIER_FILE = os.path.join(OUT_DIR, "word_classifier.joblib")
CONFUSION_PNG = os.path.join(OUT_DIR, "confusion_matrix_words.png")
os.makedirs(OUT_DIR, exist_ok=True)


# %% Cell 1: HELPERS (run this cell, nothing to change)
import json
import pickle
import time
import urllib.request
from collections import deque

import numpy as np
import joblib

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".webm", ".mkv")
N_FRAME_FEATURES = 2 * (63 + 2 + 1)    # per hand slot: 63 shape + 2 position + 1 flag


# ---------- MediaPipe ----------
def ensure_mediapipe_model():
    if not os.path.exists(MODEL_TASK_FILE):
        print("Downloading hand_landmarker.task ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_TASK_FILE)
    return MODEL_TASK_FILE


def make_detector(video_mode=True, num_hands=2):
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=ensure_mediapipe_model()),
        running_mode=vision.RunningMode.VIDEO if video_mode else vision.RunningMode.IMAGE,
        num_hands=num_hands,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5)
    return vision.HandLandmarker.create_from_options(options)


def landmarks_to_array(hand_landmarks, width, height):
    """MediaPipe landmarks (0..1) -> (21,3) in pixel units."""
    return np.array([[p.x * width, p.y * height, p.z * width] for p in hand_landmarks],
                    dtype=np.float32)


# ---------- features ----------
def normalise_points(points):
    """(21,3) -> 63 numbers independent of hand position and hand size."""
    pts = np.asarray(points, dtype=np.float32).reshape(21, 3)
    pts = pts - pts[0]
    scale = np.max(np.linalg.norm(pts[:, :2], axis=1))
    return (pts / (scale if scale > 1e-6 else 1.0)).reshape(-1)


def place_in_slots(hands_px):
    """List of up to 2 hands (each (21,3) pixels) -> raw (2,21,3) + present (2,).
    RULE (identical in training and live): hands are ordered by wrist x, smaller x
    first; a single hand always goes in slot 0. Frames must have the same
    orientation as the training videos (NOT mirrored)."""
    hands = sorted(hands_px, key=lambda p: float(p[0, 0]))[:2]
    raw = np.zeros((2, 21, 3), dtype=np.float32)
    present = np.zeros(2, dtype=bool)
    for i, h in enumerate(hands):
        raw[i] = h
        present[i] = True
    return raw, present


def frame_features(raw, present, width, height):
    """One frame -> 132 numbers: for each of 2 slots [shape(63), wrist x,y in 0..1, flag]."""
    out = []
    for s in range(2):
        if present[s]:
            out += [normalise_points(raw[s]),
                    np.array([raw[s][0, 0] / width, raw[s][0, 1] / height], dtype=np.float32),
                    np.array([1.0], dtype=np.float32)]
        else:
            out.append(np.zeros(66, dtype=np.float32))
    return np.concatenate(out)


def mirror_sequence(raw, present, width):
    """Mirror a whole sign left<->right (x -> width - x) and re-order the slots."""
    raw2 = np.zeros_like(raw)
    pres2 = np.zeros_like(present)
    for i in range(len(raw)):
        hands = []
        for s in range(2):
            if present[i, s]:
                h = raw[i, s].copy()
                h[:, 0] = width - h[:, 0]
                hands.append(h)
        r, p = place_in_slots(hands)
        raw2[i], pres2[i] = r, p
    return raw2, pres2


def find_runs(times_ms, present_any, gap_ms):
    """Group frames where hands are visible into 'signs'. Returns [(start, end), ...]
    (frame indices, inclusive). Gaps shorter than gap_ms do not split a sign."""
    idx = [i for i, p in enumerate(present_any) if p]
    if not idx:
        return []
    runs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if times_ms[i] - times_ms[prev] > gap_ms:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))
    return runs


def resample_sequence(frames, n_steps=N_STEPS):
    """(n, 132) -> flat vector of n_steps*132 by picking evenly spaced frames."""
    idx = np.linspace(0, len(frames) - 1, n_steps).round().astype(int)
    return np.asarray(frames)[idx].reshape(-1)


def record_to_vector(rec, mirror=False):
    """One stored video -> one feature vector (or None if no usable sign)."""
    raw, present, W, H = rec["raw"], rec["present"], rec["W"], rec["H"]
    if mirror:
        raw, present = mirror_sequence(raw, present, W)
    runs = find_runs(rec["times"], present.any(axis=1), GAP_MS)
    if not runs:
        return None
    s, e = max(runs, key=lambda r: r[1] - r[0])           # the longest sign in the video
    if rec["times"][e] - rec["times"][s] < MIN_SIGN_MS:
        return None
    feats = np.stack([frame_features(raw[i], present[i], W, H) for i in range(s, e + 1)])
    return resample_sequence(feats)


# ---------- dataset index ----------
def load_dataset_index(dataset_dir):
    """Return {word: [ {video_id, split, path}, ... ]} for videos that really exist.
    Works with the standard WLASL layout (WLASL_v0.3.json + videos/<video_id>.mp4).
    Fallback: one folder per word."""
    videos = {}
    for dp, _, files in os.walk(dataset_dir):
        for f in files:
            if f.lower().endswith(VIDEO_EXTS):
                videos[os.path.splitext(f)[0]] = os.path.join(dp, f)
    if not videos:
        raise FileNotFoundError(f"No video files found under {dataset_dir}. Check DATASET_DIR.")

    json_path = None
    for dp, _, files in os.walk(dataset_dir):
        for f in files:
            if f.lower().endswith(".json") and "wlasl" in f.lower() and "nslt" not in f.lower():
                json_path = os.path.join(dp, f)
    index = {}
    if json_path:
        with open(json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for entry in data:
            for inst in entry.get("instances", []):
                vid = str(inst.get("video_id"))
                if vid in videos:
                    index.setdefault(entry["gloss"], []).append({
                        "video_id": vid, "split": inst.get("split", "train"),
                        "path": videos[vid], "frame_start": inst.get("frame_start", 1),
                        "frame_end": inst.get("frame_end", -1)})
        print(f"Index: {json_path}\n  {len(videos)} video files, "
              f"{sum(len(v) for v in index.values())} matched to {len(index)} words")
    else:
        print("No WLASL json found: using the folder name of each video as the word.")
        for vid, path in videos.items():
            index.setdefault(os.path.basename(os.path.dirname(path)), []).append({
                "video_id": vid, "split": "train", "path": path,
                "frame_start": 1, "frame_end": -1})
    return index


def show_available_words(index, top=60):
    ranked = sorted(index.items(), key=lambda kv: -len(kv[1]))[:top]
    print(f"{'word':<18}videos")
    for word, inst in ranked:
        print(f"{word:<18}{len(inst)}")


def select_words(index, words=None, n_words=10, min_videos=8):
    if words:
        missing = [w for w in words if w not in index]
        if missing:
            print("Not available in your download:", missing)
        chosen = [w for w in words if w in index]
    else:
        ranked = sorted(index.items(), key=lambda kv: -len(kv[1]))
        chosen = [w for w, inst in ranked if len(inst) >= min_videos][:n_words]
    print("Selected words:", chosen)
    return {w: index[w] for w in chosen}


# ---------- extraction ----------
def extract_sequences(selection, out_pkl, stride=FRAME_STRIDE):
    """Run MediaPipe over every selected video and store the raw hand points."""
    import cv2
    import mediapipe as mp

    jobs = [(w, inst) for w, lst in selection.items() for inst in lst]
    print(f"{len(jobs)} videos to process (this takes a while; stop with the stop button, "
          f"nothing is lost until the end).")
    records, no_hands, t0 = [], {}, time.time()
    for n, (word, inst) in enumerate(jobs, 1):
        cap = cv2.VideoCapture(inst["path"])
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        detector = make_detector(video_mode=True, num_hands=2)   # fresh one per video
        raws, pres, times, W, H, i = [], [], [], None, None, 0
        first = max(inst["frame_start"] - 1, 0) if USE_JSON_TRIM else 0
        last = inst["frame_end"] if (USE_JSON_TRIM and inst["frame_end"] > 0) else 10 ** 9
        while True:
            ok, bgr = cap.read()
            if not ok or i / fps > MAX_VIDEO_SECONDS or i >= last:
                break
            if i >= first and (i - first) % stride == 0:
                h0, w0 = bgr.shape[:2]
                if max(h0, w0) > 480:                             # speed
                    sc = 480 / max(h0, w0)
                    bgr = cv2.resize(bgr, (int(w0 * sc), int(h0 * sc)))
                H, W = bgr.shape[:2]
                t_ms = int(i * 1000 / fps)
                res = detector.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), t_ms)
                hands = [landmarks_to_array(lm, W, H) for lm in res.hand_landmarks]
                r, p = place_in_slots(hands)
                raws.append(r); pres.append(p); times.append(t_ms)
            i += 1
        cap.release()
        detector.close()
        if len(raws) < 3 or not np.any(pres):
            no_hands[word] = no_hands.get(word, 0) + 1
            continue
        records.append({"word": word, "split": inst["split"], "video_id": inst["video_id"],
                        "raw": np.stack(raws), "present": np.stack(pres),
                        "times": np.array(times), "W": W, "H": H})
        if n % 10 == 0:
            el = time.time() - t0
            print(f"  {n}/{len(jobs)} videos   {el:.0f}s elapsed, ~{el / n * (len(jobs) - n):.0f}s left")

    with open(out_pkl, "wb") as fh:
        pickle.dump(records, fh)
    print(f"\nSaved {len(records)} videos to {out_pkl}")
    if no_hands:
        print("Videos where no hand was found / unreadable (per word):", no_hands)
    return records


# ---------- training ----------
def train_and_evaluate(pkl_path, model_path, confusion_png):
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    with open(pkl_path, "rb") as fh:
        records = pickle.load(fh)
    vecs = [record_to_vector(r) for r in records]
    keep = [i for i, v in enumerate(vecs) if v is not None]
    print(f"{len(keep)}/{len(records)} videos contain a usable sign")
    records = [records[i] for i in keep]
    X = np.stack([vecs[i] for i in keep])
    y = np.array([r["word"] for r in records])
    split = np.array([r["split"] for r in records])

    words, counts = np.unique(y, return_counts=True)
    print("Samples per word:", dict(zip(words, counts)))
    if counts.min() < 3:
        print("WARNING: some words have fewer than 3 samples - results will be unreliable.")

    # Prefer the dataset's own train/val/test split (different signers = honest score).
    test_mask = np.isin(split, ["val", "test"])
    ok_split = all(test_mask[y == w].any() and (~test_mask[y == w]).any() for w in words)
    if ok_split:
        tr, te = np.where(~test_mask)[0], np.where(test_mask)[0]
        print(f"Using the dataset split: {len(tr)} train / {len(te)} test videos")
    else:
        tr, te = train_test_split(np.arange(len(y)), test_size=0.25, random_state=0,
                                  stratify=y if counts.min() >= 2 else None)
        print("Dataset split not usable for all words -> random split "
              "(the same signer may appear in train and test, so accuracy is optimistic).")

    X_tr, y_tr = X[tr], y[tr]
    if USE_MIRROR_AUGMENTATION:
        mv = [(record_to_vector(records[i], mirror=True), y[i]) for i in tr]
        mv = [(v, l) for v, l in mv if v is not None]
        X_tr = np.vstack([X_tr] + [np.stack([v for v, _ in mv])])
        y_tr = np.concatenate([y_tr, [l for _, l in mv]])

    candidates = {
        "RandomForest": RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=0),
        "SVM (rbf)": make_pipeline(StandardScaler(),
                                   SVC(kernel="rbf", C=10, gamma="scale", probability=True)),
    }
    best_name, best_model, best_acc = None, None, -1
    print(f"\nChance level (random guessing) = {1 / len(words):.2f}")
    for name, model in candidates.items():
        t0 = time.time()
        model.fit(X_tr, y_tr)
        train_s = time.time() - t0
        t0 = time.time()
        pred = model.predict(X[te])
        ms = 1000 * (time.time() - t0) / len(te)
        acc = accuracy_score(y[te], pred)
        print(f"{name:14s} test accuracy = {acc:.3f}   train {train_s:.1f}s   {ms:.2f} ms/sign")
        if acc > best_acc:
            best_name, best_model, best_acc = name, model, acc

    pred = best_model.predict(X[te])
    print(f"\nBest: {best_name} ({best_acc:.3f})")
    print(classification_report(y[te], pred, zero_division=0))
    cm = confusion_matrix(y[te], pred, labels=list(words))
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(words)), words, rotation=90)
    ax.set_yticks(range(len(words)), words)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix - {best_name} (accuracy {best_acc:.2f})")
    fig.tight_layout(); fig.savefig(confusion_png, dpi=150); plt.show()

    final = best_model
    if REFIT_ON_ALL_DATA:
        X_all, y_all = X, y
        if USE_MIRROR_AUGMENTATION:
            mv = [(record_to_vector(r, mirror=True), r["word"]) for r in records]
            mv = [(v, l) for v, l in mv if v is not None]
            X_all = np.vstack([X, np.stack([v for v, _ in mv])])
            y_all = np.concatenate([y, [l for _, l in mv]])
        final = candidates[best_name].fit(X_all, y_all)
        print("Refitted on all data for use in the game (test accuracy above is from the first fit).")
    joblib.dump({"model": final, "name": best_name, "accuracy": best_acc,
                 "words": list(words), "n_steps": N_STEPS}, model_path)
    print(f"Saved {model_path}")
    return final


# ---------- the object the rest of the team uses ----------
_loaded = {}

def _model():
    if "m" not in _loaded:
        _loaded["m"] = joblib.load(CLASSIFIER_FILE)["model"]
    return _loaded["m"]


class SignLock:
    """A word locks when the same guess (conf >= min_conf) holds for hold_ms."""
    def __init__(self, min_conf=MIN_CONFIDENCE, hold_ms=HOLD_MS):
        self.min_conf, self.hold_ms = min_conf, hold_ms
        self.label, self.since = None, None

    def update(self, label, conf, now_ms):
        if label is None or conf < self.min_conf:
            self.label, self.since = None, None
            return None, False
        if label != self.label:
            self.label, self.since = label, now_ms
        return self.label, (now_ms - self.since) >= self.hold_ms


class SignWordRecognizer:
    """ONE INSTANCE PER PLAYER. Call update() once per camera frame with that
    player's hands. It keeps a short history, finds the sign inside it, and
    returns (word, confidence, locked).

        rec = SignWordRecognizer()
        word, conf, locked = rec.update(hands_px, frame_w, frame_h, t_ms)

    hands_px : list of 0, 1 or 2 arrays of shape (21,3) in PIXEL units
               (x*width, y*height, z*width), from the NOT-mirrored camera frame.
    t_ms     : time of the frame in milliseconds (increasing).
    """
    def __init__(self):
        self.buf = deque()
        self.lock = SignLock()
        self.last_pred_t = -10 ** 9
        self.label, self.conf = None, 0.0

    def reset(self):
        self.buf.clear()
        self.lock = SignLock()
        self.label, self.conf = None, 0.0

    def _active_sign(self, t_ms):
        items = list(self.buf)
        pres = [it[2] for it in items]
        times = [it[0] for it in items]
        runs = find_runs(times, pres, GAP_MS)
        if not runs:
            return None
        s, e = runs[-1]                                    # most recent sign
        if t_ms - times[e] > END_GRACE_MS or times[e] - times[s] < MIN_SIGN_MS:
            return None
        return np.stack([items[i][1] for i in range(s, e + 1)])

    def update(self, hands_px, width, height, t_ms):
        raw, present = place_in_slots(hands_px)
        self.buf.append((t_ms, frame_features(raw, present, width, height), bool(present.any())))
        while self.buf and t_ms - self.buf[0][0] > BUFFER_MS:
            self.buf.popleft()
        if t_ms - self.last_pred_t >= PREDICT_EVERY_MS:
            self.last_pred_t = t_ms
            seg = self._active_sign(t_ms)
            if seg is None:
                self.label, self.conf = None, 0.0
            else:
                model = _model()
                probs = model.predict_proba(resample_sequence(seg).reshape(1, -1))[0]
                k = int(np.argmax(probs))
                self.label, self.conf = str(model.classes_[k]), float(probs[k])
        held, locked = self.lock.update(self.label, self.conf, t_ms)
        return self.label, self.conf, locked


# %% Cell 2: LOOK AT THE DATASET (fast) - choose your words from this list
if __name__ == "__main__":
    index = load_dataset_index(DATASET_DIR)
    show_available_words(index, top=60)


# %% Cell 3: EXTRACT HAND POINTS FROM THE VIDEOS (slow, run once)
if __name__ == "__main__":
    selection = select_words(index, WORDS, N_WORDS, MIN_VIDEOS_PER_WORD)
    records = extract_sequences(selection, SEQUENCES_FILE)


# %% Cell 4: TRAIN + EVALUATE
if __name__ == "__main__":
    model = train_and_evaluate(SEQUENCES_FILE, CLASSIFIER_FILE, CONFUSION_PNG)


# %% Cell 5: LIVE WEBCAM TEST - one person, up to two hands (q = quit, r = reset)
def run_live(camera_index=0):
    import cv2
    import mediapipe as mp

    detector = make_detector(video_mode=True, num_hands=2)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try camera_index=1.")
    recognizer = SignWordRecognizer()
    t_start, last_t, fps = time.time(), time.time(), 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            t_ms = int((time.time() - t_start) * 1000)
            # IMPORTANT: detect on the NOT-mirrored frame (same as the training videos).
            res = detector.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), t_ms)
            hands = [landmarks_to_array(lm, w, h) for lm in res.hand_landmarks]
            word, conf, locked = recognizer.update(hands, w, h, t_ms)

            view = cv2.flip(frame, 1)                       # mirror only for display
            for p in hands:
                for x, y, _ in p:
                    cv2.circle(view, (int(w - x), int(y)), 3, (0, 255, 0), -1)
            color = (0, 200, 0) if locked else (0, 165, 255)
            text = "show a sign..." if word is None else \
                f"{word}  {conf:.2f}" + ("  LOCKED" if locked else "")
            cv2.putText(view, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            fps = 0.9 * fps + 0.1 * (1.0 / max(time.time() - last_t, 1e-6))
            last_t = time.time()
            cv2.putText(view, f"FPS {fps:.0f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)
            cv2.imshow("Sign word classifier (q quit, r reset)", view)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("r"):
                recognizer.reset()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


if __name__ == "__main__":
    run_live()