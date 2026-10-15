"""Convert downloaded dataset videos (data/external/videos/<Label>/*.mp4)
into the same clip-feature CSV format that 02_data_collection.ipynb writes,
so 03_train_classifier.ipynb can train on webcam recordings and dataset
videos together.

Per video:
  1. Read frames and mirror them horizontally (the live webcam pipeline
     flips frames before detection, so dataset frames must match).
  2. Detect up to 2 hands per frame (many ASL signs are two-handed).
  3. Link detections across frames into tracks by nearest wrist position.
  4. Keep the track whose wrist moves the most (the dominant signing hand).
  5. Build the same shape + trajectory feature vector as live recognition
     (src.features.extract_frame / build_clip_feature).

By default (--window 0) each video becomes ONE sample covering the whole sign.
Optional --window 1.2 cuts each video into overlapping 1.2 s windows instead
(stride --stride), mimicking the live rolling buffer. We tried that: in a
real-time replay of 24 held-out videos it did NOT beat whole-video samples
(11/24 vs 14/24 signs confirmed), so whole-video is the default.

Run from the project root:
    python scripts/extract_features.py
    python scripts/extract_features.py --limit 5     # quick test, 5 per class
"""
import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.features import build_clip_feature, extract_frame, mirror_raw63  # noqa: E402
from src.hand_model import corrected_handedness  # noqa: E402

VIDEO_DIR = os.path.join(ROOT, "data", "external", "videos")
MODEL_PATH = os.path.join(ROOT, "models", "hand_landmarker.task")
TARGET_FRAMES = 20          # must match TARGET_FRAMES in 02_data_collection.ipynb
MIN_TRACK_FRAMES = 5        # need at least this many frames with the hand visible
MAX_LINK_DIST = 0.25        # max wrist jump (fraction of frame) to stay on one track

_landmarker = None


def _get_landmarker():
    global _landmarker
    if _landmarker is None:
        from src.hand_model import load_hand_landmarker
        _landmarker = load_hand_landmarker(model_path=MODEL_PATH, num_hands=2)
    return _landmarker


def build_tracks(per_frame_hands):
    """per_frame_hands: list (one entry per frame) of lists of
    (frame_idx, landmarks). Greedy nearest-wrist linking -> list of tracks,
    each a list of (frame_idx, landmarks)."""
    tracks = []  # each: {"last_wrist": np.array(2), "last_frame": int, "items": [...]}
    for hands in per_frame_hands:
        unused = list(range(len(tracks)))
        for frame_idx, lms, label in hands:
            wrist = np.array([lms[0].x, lms[0].y])
            best, best_d = None, MAX_LINK_DIST
            for ti in unused:
                d = np.linalg.norm(tracks[ti]["last_wrist"] - wrist)
                if d < best_d:
                    best, best_d = ti, d
            if best is None:
                tracks.append({"last_wrist": wrist, "items": [(frame_idx, lms, label)]})
            else:
                unused.remove(best)
                tracks[best]["last_wrist"] = wrist
                tracks[best]["items"].append((frame_idx, lms, label))
    return [t["items"] for t in tracks]


def track_motion(items):
    """Total wrist path length, normalized by median hand size."""
    wrists = np.array([[lms[0].x, lms[0].y] for _, lms, _ in items])
    scales = [np.hypot(lms[9].x - lms[0].x, lms[9].y - lms[0].y) for _, lms, _ in items]
    path = float(np.sum(np.linalg.norm(np.diff(wrists, axis=0), axis=1))) if len(wrists) > 1 else 0.0
    return path / max(float(np.median(scales)), 1e-6)


def process_video(args):
    label, path, window, stride = args
    import mediapipe as mp

    landmarker = _get_landmarker()
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1:
        fps = 30.0

    aspect = cap.get(cv2.CAP_PROP_FRAME_WIDTH) / max(cap.get(cv2.CAP_PROP_FRAME_HEIGHT), 1.0)
    per_frame_hands = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # match the mirrored live webcam feed
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        per_frame_hands.append([(idx, lms, corrected_handedness(hd[0].category_name))
                                for lms, hd in zip(result.hand_landmarks, result.handedness)])
        idx += 1
    cap.release()

    tracks = [t for t in build_tracks(per_frame_hands) if len(t) >= MIN_TRACK_FRAMES]
    if not tracks:
        return label, path, [], "no_hand_track"

    # dominant signing hand = most wrist movement (ties -> most frames)
    best = max(tracks, key=lambda t: (track_motion(t), len(t)))

    t0 = best[0][0]
    raw_all, scale_all, time_all = [], [], []
    for frame_idx, lms, _ in best:
        raw63, scale = extract_frame(lms, aspect)
        raw_all.append(raw63)
        scale_all.append(scale)
        time_all.append((frame_idx - t0) / fps)

    # left-handed signer (majority of this track's frames) -> mirror into right-hand form
    if sum(1 for b in best if b[2] == "Left") * 2 > len(best):
        raw_all = [mirror_raw63(r) for r in raw_all]

    def make(idx):
        base = time_all[idx[0]]
        return build_clip_feature(
            [raw_all[k] for k in idx], [scale_all[k] for k in idx],
            [time_all[k] - base for k in idx], TARGET_FRAMES,
        )

    span = time_all[-1] - time_all[0]
    samples = []  # (feature_vector, window_start_seconds)
    if window <= 0 or span < 0.7 * window:
        feat = make(list(range(len(best))))          # whole video as one sample
        if feat is not None:
            samples.append((feat, 0.0))
    else:
        s = time_all[0]
        while s <= time_all[-1] - 0.7 * window + 1e-9:
            idx = [k for k, t in enumerate(time_all) if s <= t < s + window]
            if len(idx) >= 5 and time_all[idx[-1]] - time_all[idx[0]] >= 0.7 * window:
                feat = make(idx)
                if feat is not None:
                    samples.append((feat, round(s, 3)))
            s += stride

    if not samples:
        return label, path, [], "feature_failed"
    return label, path, samples, f"ok(track_frames={len(best)}/{len(per_frame_hands)})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="max videos per class (0 = all)")
    ap.add_argument("--window", type=float, default=0.0, help="window length in s (0 = one sample per whole video)")
    ap.add_argument("--stride", type=float, default=0.3, help="window stride in s")
    ap.add_argument("--words-file", default="", help="JSON list of dataset words: process only these (label = Capitalised word)")
    ap.add_argument("--out-dir", default="", help="where to write the CSV (default: data/raw)")
    args = ap.parse_args()

    only = {w.capitalize() for w in json.load(open(args.words_file))} if args.words_file else None
    jobs = []
    for label_dir in sorted(glob.glob(os.path.join(VIDEO_DIR, "*"))):
        label = os.path.basename(label_dir)
        if only and label not in only:
            continue
        vids = sorted(glob.glob(os.path.join(label_dir, "*.mp4")))
        if args.limit:
            vids = vids[: args.limit]
        jobs += [(label, v, args.window, args.stride) for v in vids]
    print(f"Processing {len(jobs)} videos with {args.workers} workers...", flush=True)

    t0 = time.time()
    records, status = [], {}
    with Pool(args.workers) as pool:
        for i, (label, path, samples, msg) in enumerate(pool.imap_unordered(process_video, jobs, chunksize=4), 1):
            ok = bool(samples)
            st = status.setdefault(label, {"ok": 0, "fail": 0, "clips": 0})
            st["ok" if ok else "fail"] += 1
            st["clips"] += len(samples)
            for feat, wstart in samples:
                rec = {f"feature_{j}": v for j, v in enumerate(feat)}
                rec["label"] = label
                rec["collector"] = "hf_dataset"
                rec["timestamp"] = datetime.now().isoformat()
                rec["source"] = os.path.basename(path)
                rec["window_start"] = wstart
                records.append(rec)
            if i % 50 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  ({time.time() - t0:.0f}s)", flush=True)

    print("\nPer-class yield (usable videos / total videos -> training samples):")
    for label in sorted(status):
        st = status[label]
        print(f"  {label:12} {st['ok']:3d} / {st['ok'] + st['fail']:3d} -> {st['clips']:4d} samples")

    if records:
        out_dir = args.out_dir or os.path.join(ROOT, "data", "raw")
        os.makedirs(out_dir, exist_ok=True)
        tag = "candidates" if args.words_file else "dataset"
        out = os.path.join(out_dir, f"hf_{tag}_{'win_' if args.window > 0 else ''}{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        pd.DataFrame(records).to_csv(out, index=False)
        print(f"\nSaved {len(records)} clips to {out}")
    else:
        print("No clips extracted.")


if __name__ == "__main__":
    main()
