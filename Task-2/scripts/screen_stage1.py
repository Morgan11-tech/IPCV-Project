"""Stage-1 screen of candidate words (cheap, clip-level, cross-source).

For each candidate word w, a classifier is trained on
    the KEPT words (all sources) + Idle + w's clips from the other recording
    sources,
then tested on w's clips from ONE held-out source. That repeats for every source
w has, so a word only scores well if it generalises to a source it never saw
(the opposite of "memorised one signer's near-duplicates").

Reported per candidate:
    recall   mean / min held-out-source recall for w
    stolen   share of the kept words' held-out clips that get labelled w
             (a good candidate must not take over words we already rely on)

    python scripts/screen_stage1.py
"""
import glob
import os
import re
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEPT = ["Please", "Bathroom", "No", "Yes", "Hello", "More", "Apple", "Color", "Boy"]
MIN_CLIPS_PER_SOURCE = 6

_G = {}


def style(name):
    if re.match(r"^\d+-.*\.mp4$", name):
        return "dict_id"
    if re.search(r"_2024\d{4}_\d{6}", name):
        return "session"
    if "_video_" in name:
        return "word_video_N"
    return "word_N"


def init():
    base = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(ROOT, "data", "raw", "*.csv")))], ignore_index=True)
    base = base[base["label"].isin(KEPT + ["Idle"])].reset_index(drop=True)
    cand = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(ROOT, "data", "external", "candidates", "*.csv")))], ignore_index=True)
    fc = [c for c in base.columns if c.startswith("feature_")]
    for d in (base, cand):
        d["style"] = d["source"].map(lambda n: style(n) if isinstance(n, str) else "own")
    _G.update(base=base, cand=cand, fc=fc)


def score(word):
    base, cand, fc = _G["base"], _G["cand"], _G["fc"]
    w = cand[cand["label"] == word]
    srcs = [s for s, n in w["style"].value_counts().items() if n >= MIN_CLIPS_PER_SOURCE]
    if len(srcs) < 2:
        return None
    recalls, stolen_n, stolen_d = [], 0, 0
    for s in srcs:
        train = pd.concat([base[base["style"] != s], w[w["style"] != s]])
        clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=1, class_weight="balanced_subsample")
        clf.fit(train[fc].values, train["label"].values)
        test_w = w[w["style"] == s]
        recalls.append(float(np.mean(clf.predict(test_w[fc].values) == word)))
        kept_test = base[(base["style"] == s) & base["label"].isin(KEPT) & base["label"].isin(set(train["label"]))]
        if len(kept_test):
            stolen_n += int(np.sum(clf.predict(kept_test[fc].values) == word))
            stolen_d += len(kept_test)
    return {"word": word, "clips": len(w), "sources": "+".join(f"{s}:{int((w['style']==s).sum())}" for s in srcs),
            "recall_mean": float(np.mean(recalls)), "recall_min": float(np.min(recalls)),
            "stolen": stolen_n / stolen_d if stolen_d else float("nan")}


def main():
    init()
    words = sorted(w for w in _G["cand"]["label"].unique() if w not in KEPT)
    print(f"screening {len(words)} candidate words (clip-level, leave-one-source-out)...", flush=True)
    with Pool(6, initializer=init) as pool:
        res = [r for r in pool.imap_unordered(score, words, chunksize=2) if r]
    df = pd.DataFrame(res)
    df["score"] = df["recall_min"] * 0.5 + df["recall_mean"] * 0.5 - 2 * df["stolen"].fillna(0)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    out = os.path.join(ROOT, "data", "external", "stage1_ranking.csv")
    df.to_csv(out, index=False)
    print(f"{len(df)} candidates scored (needed >=2 sources with >={MIN_CLIPS_PER_SOURCE} clips)")
    print(df.head(40).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("saved", out)


if __name__ == "__main__":
    main()
