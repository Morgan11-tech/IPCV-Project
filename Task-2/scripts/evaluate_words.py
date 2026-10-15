"""Per-word reliability of the REAL recogniser pipeline, on videos the model
has not seen.

5-fold split over the dataset videos. For each fold a classifier is trained on
the other folds (plus all Idle clips), then every held-out video is replayed
through src.recognizer.SignRecognizer (same smoothing, tracking, motion gate,
confidence threshold and hold time as live) using video time as the clock.
Per word we report:
    hit    = the right word was confirmed at least once
    wrong  = a different word was confirmed (a false confirmation)
    miss   = nothing confirmed
and which wrong word appears most.

    python scripts/evaluate_words.py                       # all words in data/raw
    python scripts/evaluate_words.py --words Yes,No,Please # only these (+ Idle)

Caveat: folds are random over videos, so clips from the same signer/session
can sit on both sides of the split. Treat the numbers as optimistic; the real
test is your own webcam.
"""
import argparse
import glob
import os
import sys
import tempfile
from collections import Counter
from multiprocessing import Pool

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.recognizer import SignRecognizer  # noqa: E402

N_FOLDS = 5
EXTRA = False   # include data/external/candidates/*.csv (set by --with-candidates)


def style(name):
    import re
    if not isinstance(name, str):
        return "own"
    if re.match(r"^\d+-.*\.mp4$", name):
        return "dict_id"
    if re.search(r"_2024\d{4}_\d{6}", name):
        return "session"
    if "_video_" in name:
        return "word_video_N"
    return "word_N"


def load(words):
    files = sorted(glob.glob(os.path.join(ROOT, "data", "raw", "*.csv")))
    if EXTRA:
        files += sorted(glob.glob(os.path.join(ROOT, "data", "external", "candidates", "*.csv")))
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    has_src = df["source"].notna()                       # Idle clips have no source: keep all of them
    df = pd.concat([df[~has_src], df[has_src].drop_duplicates(subset=["label", "source"])], ignore_index=True)
    if words:
        df = df[df["label"].isin(list(words) + ["Idle"])]
    return df.reset_index(drop=True)


def run_fold(args):
    fold, train_idx, test_idx, words, conf, hold, extra, flip, mirror_mode = args
    global EXTRA
    EXTRA = extra
    df = load(words)
    fc = [c for c in df.columns if c.startswith("feature_")]
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=2, class_weight="balanced_subsample")
    clf.fit(df.loc[train_idx, fc].values, df.loc[train_idx, "label"].values)
    path = os.path.join(tempfile.gettempdir(), f"eval_fold{fold}.joblib")
    joblib.dump({"model": clf, "classes": list(clf.classes_), "target_frames": len(fc) // 63}, path)
    rec = SignRecognizer(classifier_path=path, confidence_threshold=conf, confirm_seconds=hold, mirror_mode=mirror_mode)
    rows = []
    for i in test_idx:
        label, src = df.loc[i, "label"], df.loc[i, "source"]
        cap = cv2.VideoCapture(os.path.join(ROOT, "data", "external", "videos", label, src))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        if flip:                                          # simulate a left-handed signer
            frames = [cv2.flip(f, 1) for f in frames]
        frames += [frames[-1]] * 10                      # let the last hand pose settle
        rec.reset()
        confirmed = []
        for k, f in enumerate(frames):
            rec.update(f, k, timestamp=k / fps)
            confirmed += [s["label"] for s in rec.signs if s["confirmed"]]
        rows.append((label, src, confirmed))
    rec.close()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", default="", help="comma-separated subset (Idle is always kept)")
    ap.add_argument("--out", default="")
    ap.add_argument("--with-candidates", action="store_true", help="also load data/external/candidates/*.csv")
    ap.add_argument("--split", choices=["video", "source"], default="video",
                    help="video: random 5-fold over videos; source: hold out one whole recording source at a time (stricter)")
    ap.add_argument("--mirror-mode", choices=["auto", "never", "always", "best"], default="auto", help="left/right handling in the recogniser")
    ap.add_argument("--flip", action="store_true", help="mirror every TEST video (simulates left-handed signing; training is untouched)")
    ap.add_argument("--conf", type=float, default=0.35, help="confidence threshold")
    ap.add_argument("--hold", type=float, default=0.3, help="seconds the same word must stay on top")
    args = ap.parse_args()
    words = [w for w in args.words.split(",") if w]

    global EXTRA
    EXTRA = args.with_candidates
    df = load(words)
    is_vid = (df["collector"] == "hf_dataset").values
    vid_idx = np.where(is_vid)[0]
    idle_idx = np.where(~is_vid)[0]
    jobs = []
    if args.split == "source":
        df["style"] = df["source"].map(style)
        for fold, st in enumerate(sorted(set(df.loc[vid_idx, "style"]))):
            train = vid_idx[df.loc[vid_idx, "style"].values != st]
            trained = set(df.loc[train, "label"]) | {"Idle"}
            test = [i for i in vid_idx if df.at[i, "style"] == st and df.at[i, "label"] in trained]
            if test:
                jobs.append((fold, np.concatenate([train, idle_idx]), np.array(test), words, args.conf, args.hold, EXTRA, args.flip, args.mirror_mode))
        evaluable = {df.at[i, "label"] for j in jobs for i in j[2]}
        skipped = sorted(set(df.loc[vid_idx, "label"]) - evaluable)
        if skipped:
            print("not evaluable (only one recording source, nothing to hold out):", ", ".join(skipped), flush=True)
    else:
        skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=0)
        for fold, (tr, te) in enumerate(skf.split(vid_idx, df.loc[vid_idx, "label"].values)):
            jobs.append((fold, np.concatenate([vid_idx[tr], idle_idx]), vid_idx[te], words, args.conf, args.hold, EXTRA, args.flip, args.mirror_mode))
    print(f"{len(vid_idx)} videos, {len(set(df.loc[vid_idx, 'label']))} words, {len(idle_idx)} Idle clips, "
          f"{len(jobs)} folds in parallel ({args.split} split)...", flush=True)

    with Pool(len(jobs)) as pool:
        results = [r for fold_rows in pool.map(run_fold, jobs) for r in fold_rows]

    stats = {}
    for label, src, confirmed in results:
        s = stats.setdefault(label, {"n": 0, "hit": 0, "wrong": 0, "miss": 0, "wrong_words": Counter()})
        s["n"] += 1
        right = label in confirmed
        wrongs = [c for c in confirmed if c != label]
        s["hit"] += right
        s["wrong"] += bool(wrongs)
        s["miss"] += (not confirmed)
        s["wrong_words"].update(wrongs)
    table = []
    for label, s in stats.items():
        ww = s["wrong_words"].most_common(1)
        table.append((label, s["n"], s["hit"] / s["n"], s["wrong"] / s["n"], s["miss"] / s["n"],
                      f"{ww[0][0]} x{ww[0][1]}" if ww else "-"))
    table.sort(key=lambda r: (-(r[2] - r[3]), -r[2]))
    print(f"\n{'word':12} {'videos':>6} {'hit':>6} {'wrong':>6} {'nothing':>8}   most common wrong confirmation")
    for t in table:
        print(f"{t[0]:12} {t[1]:>6} {t[2]:>6.0%} {t[3]:>6.0%} {t[4]:>8.0%}   {t[5]}")
    tot = sum(t[1] for t in table)
    print(f"\nOVERALL: hit {sum(t[1]*t[2] for t in table)/tot:.0%} | wrong {sum(t[1]*t[3] for t in table)/tot:.0%} "
          f"| nothing {sum(t[1]*t[4] for t in table)/tot:.0%}   ({tot} held-out videos)")
    out = args.out or os.path.join(ROOT, "data", "word_reliability.csv")
    pd.DataFrame(table, columns=["word", "videos", "hit", "wrong", "nothing", "top_wrong"]).to_csv(out, index=False)
    conf = Counter()
    for label, src, confirmed in results:
        for c in confirmed:
            if c != label:
                conf[(label, c)] += 1
    pd.DataFrame([(a, b, n) for (a, b), n in conf.items()], columns=["true", "confirmed_as", "count"]).to_csv(
        out.replace(".csv", "_confusions.csv"), index=False)
    print("saved", out)


if __name__ == "__main__":
    main()
