"""Automatic round-by-round word selection (backward elimination).

Start from the protected words (the current vocabulary) plus a pool of
candidate words. Each round replays all held-out videos (recording sources
held out one at a time) through the real recogniser, then removes the worst
candidate(s), until the target number of added words is left. A final round
checks the survivors against the bar.

A candidate's "badness" counts its own wrong confirmations, its misses, and how
many of the PROTECTED words' videos get mistaken for it (stealing).

    python scripts/select_words.py --candidates Pig,Fish,Water,... --target 6
"""
import argparse
import os
import subprocess
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROTECTED = ["Apple", "Please", "Color", "Yes", "No", "More", "Boy", "Hello", "Bathroom"]
BAR_HIT, BAR_WRONG = 0.60, 0.10


def evaluate(words, out, conf, hold):
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "evaluate_words.py"), "--words", ",".join(words),
           "--with-candidates", "--split", "source", "--conf", str(conf), "--hold", str(hold), "--out", out]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   env={**os.environ, "PYTHONWARNINGS": "ignore"})
    return pd.read_csv(out), pd.read_csv(out.replace(".csv", "_confusions.csv"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--target", type=int, default=6)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--hold", type=float, default=0.2)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "data", "external", "selection"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cands = [c for c in args.candidates.split(",") if c]
    rnd = 0
    while True:
        rnd += 1
        out = os.path.join(args.out_dir, f"round_{rnd}.csv")
        res, conf = evaluate(PROTECTED + cands, out, args.conf, args.hold)
        res = res.set_index("word")
        prot_videos = int(res.loc[[w for w in PROTECTED if w in res.index], "videos"].sum())
        prot_hit = float((res.loc[PROTECTED, "hit"] * res.loc[PROTECTED, "videos"]).sum() / prot_videos)
        prot_wrong = float((res.loc[PROTECTED, "wrong"] * res.loc[PROTECTED, "videos"]).sum() / prot_videos)
        rows = []
        for c in cands:
            if c not in res.index:
                rows.append((c, 0, 0, 9, 9, 99))
                continue
            stolen = int(conf[(conf["confirmed_as"] == c) & conf["true"].isin(PROTECTED)]["count"].sum()) / prot_videos
            bad = 2 * res.loc[c, "wrong"] + (1 - res.loc[c, "hit"]) + 6 * stolen
            rows.append((c, res.loc[c, "hit"], res.loc[c, "wrong"], stolen, bad, int(res.loc[c, "videos"])))
        tab = pd.DataFrame(rows, columns=["word", "hit", "wrong", "stolen_from_protected", "badness", "videos"]).sort_values("badness")
        print(f"\n=== round {rnd}: {len(PROTECTED)} protected + {len(cands)} candidates ===")
        print(f"protected words together: right {prot_hit:.0%} / wrong {prot_wrong:.0%}")
        print(tab.to_string(index=False, float_format=lambda x: f"{x:.2f}"), flush=True)
        if len(cands) <= args.target:
            passed = tab[(tab["hit"] >= BAR_HIT) & (tab["wrong"] <= BAR_WRONG)]["word"].tolist()
            print(f"\nFINAL candidates left: {cands}\npassing the bar (right >= {BAR_HIT:.0%}, wrong <= {BAR_WRONG:.0%}): {passed}")
            tab.to_csv(os.path.join(args.out_dir, "final.csv"), index=False)
            break
        k = 2 if len(cands) > args.target + 4 else 1
        drop = tab.tail(k)["word"].tolist()
        print("removing:", drop, flush=True)
        cands = [c for c in cands if c not in drop]


if __name__ == "__main__":
    main()
