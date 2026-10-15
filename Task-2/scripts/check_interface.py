"""Check that src.recognizer.SignRecognizer emits EXACTLY the agreed
Task 2 -> Task 3 format, and exercise two-hand tracking, using two dataset
videos placed side by side as a stand-in for two players.

    python scripts/check_interface.py                     # Yes (left) + Please (right)
    python scripts/check_interface.py --left Hello --right Milk

Needs the dataset videos (data/external/videos, see README) and the trained
models in models/. Exits non-zero if the format is violated.
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.recognizer import SignRecognizer  # noqa: E402


def validate(out, frame_id):
    assert set(out) == {"frame_id", "hands"}, f"top-level keys: {sorted(out)}"
    assert out["frame_id"] == frame_id
    assert isinstance(out["hands"], list)
    for h in out["hands"]:
        assert set(h) == {"points", "confidence", "handedness", "box"}, f"hand keys: {sorted(h)}"
        assert len(h["points"]) == 21 and all(len(p) == 3 for p in h["points"]), "points must be 21 x [x,y,z]"
        assert all(isinstance(v, float) for p in h["points"] for v in p)
        assert isinstance(h["confidence"], float) and 0.0 <= h["confidence"] <= 1.0
        assert h["handedness"] in ("Left", "Right")
        x, y, w, bh = h["box"]
        assert len(h["box"]) == 4 and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and w >= 0 and bh >= 0
        assert x + w <= 1.0 + 1e-6 and y + bh <= 1.0 + 1e-6
    xs = [h["box"][0] + h["box"][2] / 2 for h in out["hands"]]
    assert xs == sorted(xs), "hands must be listed left-to-right"


def read_video(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames, fps


def pick(label, k):
    vids = sorted(glob.glob(os.path.join(ROOT, "data", "external", "videos", label, "*.mp4")))
    return vids[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", default="Yes")
    ap.add_argument("--right", default="Please")
    args = ap.parse_args()

    (fl, fps), (fr, _) = read_video(pick(args.left, 0)), read_video(pick(args.right, 0))
    n = max(len(fl), len(fr)) + 20                       # pad: let the hands rest at the end
    pad = lambda fs, k: fs + [fs[-1]] * (k - len(fs))
    fl, fr = pad(fl, n), pad(fr, n)
    frames = [np.hstack([a, b]) for a, b in zip(fl, fr)]
    print(f"two-player test: left='{args.left}' right='{args.right}', {n} frames of "
          f"{frames[0].shape[1]}x{frames[0].shape[0]} at {fps:.0f} fps")

    rec = SignRecognizer()
    counts, events, timings, ids_by_side = {}, [], [], {"L": set(), "R": set()}
    for i, f in enumerate(frames):
        out = rec.update(f, frame_id=i, timestamp=i / fps)   # video time -> deterministic
        validate(out, i)
        counts[len(out["hands"])] = counts.get(len(out["hands"]), 0) + 1
        timings.append(rec.timing["total_ms"])
        for h, s in zip(out["hands"], rec.signs):
            side = "L" if (h["box"][0] + h["box"][2] / 2) < 0.5 else "R"
            ids_by_side[side].add(s["track_id"])
            if s["confirmed"]:
                events.append((i, side, s["label"], s["guess_confidence"]))
    rec.close()

    print("\nFORMAT: every frame matched the agreed schema exactly.")
    print(f"frames by number of hands detected: {dict(sorted(counts.items()))}")
    print(f"track ids seen on the left half: {sorted(ids_by_side['L'])} | right half: {sorted(ids_by_side['R'])}")
    print("confirmed signs (frame, side, word, confidence):")
    for e in events:
        print("   ", e)
    t = np.array(timings)
    print(f"\nper-frame processing time: mean {t.mean():.1f} ms | p95 {np.percentile(t, 95):.1f} ms | max {t.max():.1f} ms "
          f"(=> up to ~{1000 / t.mean():.0f} fps)")


if __name__ == "__main__":
    main()
