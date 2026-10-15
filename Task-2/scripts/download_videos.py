"""Download a capped, seeded random subset of videos per sign from the
HuggingFace ASL dataset (akasheroor/American-Sign-Language-Dataset, MIT
license; duplicate of ZahidYasinMittha/American-Sign-Language-Dataset).

Only the words in LABEL_TO_WORD are fetched (not the full 54 GB). Run from
the project root:

    python scripts/download_videos.py            # default cap: 40 per word
    python scripts/download_videos.py --cap 25
"""
import argparse
import csv
import json
import os
import random
import re
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import hf_hub_download

REPO_ID = "akasheroor/American-Sign-Language-Dataset"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTERNAL = os.path.join(ROOT, "data", "external")
INDEX_PATH = os.path.join(EXTERNAL, "hf_file_index.json")
VIDEO_DIR = os.path.join(EXTERNAL, "videos")

# Our class label -> word used in the dataset's "Aslense Dataset.csv".
# Idle is not in the dataset and must be recorded by hand.
LABEL_TO_WORD = {
    "Hello": "hello", "SeeYouLater": "later", "IMe": "i", "Father": "father",
    "Mother": "mother", "Yes": "yes", "No": "no", "Help": "help",
    "Please": "please", "ThankYou": "thankyou", "Want": "want", "What": "what",
    "Dog": "dog", "Cat": "cat", "Again": "again", "EatFood": "eat",
    "Milk": "milk", "More": "more", "GoTo": "go", "Bathroom": "bathroom",
    "Fine": "fine", "Like": "like", "Learn": "learn", "Sign": "sign",
    "Finish": "finish",
}


def style(name):
    """Rough recording-source guess from the filename pattern."""
    if re.match(r"^\d+-.*\.mp4$", name):
        return "dict_id"
    if re.search(r"_2024\d{4}_\d{6}", name):
        return "session"
    if "_video_" in name:
        return "word_video_N"
    return "word_N"


def pick_balanced(names, cap, rng):
    """Take up to `cap` videos, round-robin across recording sources so every
    source is represented (instead of one signer's near-duplicates dominating)."""
    groups = defaultdict(list)
    for n in names:
        groups[style(n)].append(n)
    for g in groups.values():
        rng.shuffle(g)
    out = []
    while len(out) < cap and any(groups.values()):
        for g in groups.values():
            if g and len(out) < cap:
                out.append(g.pop())
    return sorted(out)


def load_word_videos():
    csv_path = hf_hub_download(REPO_ID, "Aslense Dataset.csv", repo_type="dataset")
    by_word = defaultdict(list)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            by_word[row["word"].lower()].append(row["videos"])
    return by_word


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=40, help="max videos per sign")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--words-file", default="", help="JSON list of dataset words to fetch instead of the default 25 (label = Capitalised word); sampling is balanced across recording sources")
    args = ap.parse_args()

    index = json.load(open(INDEX_PATH))
    by_word = load_word_videos()
    rng = random.Random(args.seed)

    label_to_word = LABEL_TO_WORD
    if args.words_file:
        label_to_word = {w.capitalize(): w for w in json.load(open(args.words_file))}

    jobs = []  # (label, filename, repo_path)
    for label, word in label_to_word.items():
        names = [n for n in by_word.get(word, []) if n in index]
        if args.words_file:
            chosen = pick_balanced(names, args.cap, rng)
        else:
            rng.shuffle(names)
            chosen = sorted(names[: args.cap])
        print(f"{label:12} word={word:10} available={len(names):4} -> taking {len(chosen)}")
        for n in chosen:
            jobs.append((label, n, index[n][0]))

    print(f"\nDownloading {len(jobs)} videos with {args.workers} workers...")

    def fetch(job):
        label, name, repo_path = job
        out_dir = os.path.join(VIDEO_DIR, label)
        out_path = os.path.join(out_dir, name)
        if os.path.exists(out_path):
            return job, "cached"
        os.makedirs(out_dir, exist_ok=True)
        try:
            tmp_dir = os.path.join(EXTERNAL, "_hf_tmp")
            fetched = hf_hub_download(
                REPO_ID, repo_path, repo_type="dataset", local_dir=tmp_dir
            )
            shutil.move(fetched, out_path)
            return job, "ok"
        except Exception as e:  # keep going; report at the end
            return job, f"FAILED: {e}"

    done, failed = 0, []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(fetch, j) for j in jobs]):
            job, status = fut.result()
            done += 1
            if status.startswith("FAILED"):
                failed.append((job, status))
            if done % 50 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} done ({len(failed)} failed)", flush=True)

    print(f"\nFinished. {len(jobs) - len(failed)} ok, {len(failed)} failed.")
    for job, status in failed[:20]:
        print("  ", job[1], status)


if __name__ == "__main__":
    main()
