"""Try adding ONE candidate word at a time to a base vocabulary.

For each candidate: replay held-out recording sources through the real
recogniser with BASE + candidate, and report the candidate's own right/wrong
rates, how many base-word videos get mistaken for it ("stolen"), and the worst
drop in any base word versus the base-only run.

    python scripts/try_adding.py --candidates Government,Late,World
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from select_words import evaluate  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = ["Apple", "Please", "Color", "Yes", "No", "More", "Boy", "Hello", "Bathroom", "Yesterday", "Brown", "Football", "Water", "Red"]


def agg(res, words):
    r = res.set_index("word").loc[words]
    n = r["videos"].sum()
    return float((r["hit"] * r["videos"]).sum() / n), float((r["wrong"] * r["videos"]).sum() / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--flip", action="store_true")
    args = ap.parse_args()
    out_dir = os.path.join(ROOT, "data", "external", "adding")
    os.makedirs(out_dir, exist_ok=True)
    base_res, _ = evaluate(BASE, os.path.join(out_dir, "base.csv"), 0.30, 0.2)
    b_hit, b_wrong = agg(base_res, BASE)
    print(f"BASE ({len(BASE)} words): right {b_hit:.0%} / wrong {b_wrong:.0%}", flush=True)
    rows = []
    for c in args.candidates.split(","):
        res, conf = evaluate(BASE + [c], os.path.join(out_dir, f"{c}.csv"), 0.30, 0.2)
        r = res.set_index("word")
        hit, wrong = agg(res, BASE)
        n_base = int(r.loc[BASE, "videos"].sum())
        stolen = int(conf[(conf["confirmed_as"] == c) & conf["true"].isin(BASE)]["count"].sum()) / n_base
        drop = max(float(base_res.set_index("word").loc[w, "hit"] - r.loc[w, "hit"]) for w in BASE)
        rows.append((c, r.loc[c, "hit"], r.loc[c, "wrong"], stolen, hit - b_hit, wrong - b_wrong, drop, r.loc[c, "top_wrong"]))
        print(f"  {c:11} own right {r.loc[c, 'hit']:.0%} / wrong {r.loc[c, 'wrong']:.0%} | stolen {stolen:.1%} | "
              f"base words: right {hit - b_hit:+.0%}, wrong {wrong - b_wrong:+.0%}, worst single drop {drop:.0%} | top wrong: {r.loc[c, 'top_wrong']}", flush=True)
    t = pd.DataFrame(rows, columns=["word", "hit", "wrong", "stolen", "base_hit_change", "base_wrong_change", "worst_drop", "top_wrong"])
    t.to_csv(os.path.join(out_dir, "summary.csv"), index=False)


if __name__ == "__main__":
    main()
