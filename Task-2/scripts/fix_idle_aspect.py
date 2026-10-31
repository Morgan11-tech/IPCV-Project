"""One-off: convert webcam clip CSVs recorded BEFORE aspect-ratio correction
(extract_frame without `aspect`) to the corrected feature space, so they stay
consistent with the corrected dataset features and live recognition.

Why this is (approximately) possible: each stored shape vector is (landmark - wrist) / |wrist->MCP9|,
so landmark 9's stored vector is (nearly) a unit vector. Stretching x and z by
k and re-normalising by that landmark's new length recovers the corrected
shape; the trajectory is stretched the same way and divided by the median
change in hand scale. This is an APPROXIMATION (the stored vectors were
interpolated between frames): in a known-answer simulation it removes 97% of the
average distortion (worst case ~80%). Fine for the catch-all Idle class; do not use it for signs
(re-extract those instead).

    python scripts/fix_idle_aspect.py data/raw/student1_*.csv [--width 1920 --height 1080]

Originals are copied to data/archive_old_format/. Files already converted
(they have an `aspect_corrected` column) are skipped.
"""
import argparse
import glob
import os
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.features import REFERENCE_ASPECT  # noqa: E402


def convert(features, k):
    """features: (n_clips, T*63) uncorrected -> corrected, k = aspect / REFERENCE_ASPECT."""
    n = features.shape[0]
    X = features.reshape(n, -1, 63).copy()          # (clips, T, 63)
    T = X.shape[1]
    traj = X[:, :, 0:3]                              # wrist path, relative to clip start
    shape = X[:, :, 3:].reshape(n, T, 20, 3)         # landmarks 1..20 relative to wrist
    stretch = np.array([k, 1.0, k])
    shape = shape * stretch
    n9 = np.linalg.norm(shape[:, :, 8, :], axis=2)   # landmark 9 -> index 8; was unit length
    shape = shape / n9[:, :, None, None]
    ratio = np.median(n9, axis=1)                    # ~ new hand scale / old hand scale, per clip
    traj = traj * stretch / ratio[:, None, None]
    return np.concatenate([traj, shape.reshape(n, T, 60)], axis=2).reshape(n, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--width", type=float, default=1920)
    ap.add_argument("--height", type=float, default=1080)
    args = ap.parse_args()
    k = (args.width / args.height) / REFERENCE_ASPECT
    os.makedirs(os.path.join(ROOT, "data", "archive_old_format"), exist_ok=True)
    for pattern in args.files:
        for f in sorted(glob.glob(pattern)):
            df = pd.read_csv(f)
            if "aspect_corrected" in df.columns:
                print("skip (already converted):", f)
                continue
            fc = [c for c in df.columns if c.startswith("feature_")]
            shutil.copyfile(f, os.path.join(ROOT, "data", "archive_old_format", os.path.basename(f)))
            df[fc] = convert(df[fc].values, k)
            df["aspect_corrected"] = True
            df.to_csv(f, index=False)
            print(f"converted {len(df)} clips in {os.path.basename(f)} (k = {k:.3f})")


if __name__ == "__main__":
    main()
