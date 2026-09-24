"""
Simple measurements for the Task 1 technical evaluation.

1. Whole-run statistics (always on): average FPS and frame processing time,
   printed when the program exits.
2. Evaluation trials (press E in live mode to start/stop): every frame of the
   trial is logged to eval_logs/trial_XX.csv (+ annotated video), and a
   summary is printed and saved:
       - two-face detection rate
       - jitter of raw vs. smoothed face centre (stationary: std,
         dynamic: mean second difference)
       - effect alignment offset (smoothed effect centre vs. raw detection)
       - occlusion / missing episodes and how they were recovered
       - possible identity switches (automatic flags, to be checked in video)
       - FPS, latency, mean image brightness (for lighting comparison)

Re-analyse a saved trial with:
    python evaluation.py eval_logs/trial_01.csv
"""
import csv
import os
import sys
import time

import cv2
import numpy as np

import config

FIELDS = ["frame", "time_s", "n_faces", "brightness", "frame_ms", "player", "status",
          "source", "similarity", "raw_cx", "raw_cy", "raw_w", "raw_h",
          "smooth_cx", "smooth_cy", "smooth_w", "smooth_h", "roll"]


# ---------------------------------------------------------------------------
# 1. Whole-run FPS / latency
# ---------------------------------------------------------------------------
def new_run_stats():
    return {"start": time.perf_counter(), "frames": 0,
            "stage_ms": {"detect": [], "recognize": [], "track_draw": [], "total": []}}


def record_run_frame(stats, timings):
    stats["frames"] += 1
    for stage, values in stats["stage_ms"].items():
        values.append(timings.get(stage, 0.0))


def print_run_summary(stats):
    elapsed = time.perf_counter() - stats["start"]
    if stats["frames"] == 0:
        return
    print("\n=== Run summary ===")
    print(f"Frames:                        {stats['frames']}")
    print(f"Average FPS (incl. camera):    {stats['frames'] / elapsed:.1f}")
    for stage, values in stats["stage_ms"].items():
        print(f"Average {stage + ' time:':22s} {np.mean(values):.1f} ms")


# ---------------------------------------------------------------------------
# 2. Evaluation trials
# ---------------------------------------------------------------------------
def start_trial(frame_size):
    os.makedirs(config.EVAL_DIR, exist_ok=True)
    number = 1
    while os.path.exists(os.path.join(config.EVAL_DIR, f"trial_{number:02d}.csv")):
        number += 1
    base = os.path.join(config.EVAL_DIR, f"trial_{number:02d}")
    video = None
    if config.EVAL_SAVE_VIDEO:
        video = cv2.VideoWriter(base + ".mp4", cv2.VideoWriter_fourcc(*"mp4v"), 20, frame_size)
    print(f"\n[eval] started {base}")
    return {"base": base, "rows": [], "video": video, "start": time.perf_counter()}


def record_trial_frame(trial, frame_idx, frame, display, faces, players, timings):
    """Log one frame: one row per player with raw and smoothed geometry."""
    # Mean grey level of the camera image: an objective number for the
    # lighting condition of the trial (0 = black, 255 = white).
    brightness = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
    t = time.perf_counter() - trial["start"]
    for pid, p in players.items():
        row = {"frame": frame_idx, "time_s": round(t, 4), "n_faces": len(faces),
               "brightness": round(brightness, 1), "frame_ms": round(timings.get("total", 0), 2),
               "player": pid, "status": p.status, "source": p.source if p.status == "TRACKED" else "",
               "similarity": round(p.similarity, 4) if p.status == "TRACKED" else "",
               "roll": round(p.roll, 2) if p.visible else ""}
        for prefix, box, ok in (("raw", p.raw_box, p.status == "TRACKED"),
                                ("smooth", p.smooth_box, p.visible)):
            if ok:
                x, y, w, h = box
                row.update({f"{prefix}_cx": round(x + w / 2, 2), f"{prefix}_cy": round(y + h / 2, 2),
                            f"{prefix}_w": round(w, 2), f"{prefix}_h": round(h, 2)})
        trial["rows"].append(row)
    if trial["video"] is not None:
        trial["video"].write(display)


def finish_trial(trial):
    if trial["video"] is not None:
        trial["video"].release()
    with open(trial["base"] + ".csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(trial["rows"])
    lines = analyze_rows(trial["rows"])
    with open(trial["base"] + "_summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[eval] saved {trial['base']}.csv / _summary.txt" +
          (" / .mp4" if trial["video"] is not None else ""))


def load_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def jitter_metrics(centres):
    """
    centres: (k, 2) array of face centres from CONSECUTIVE frames.

    std:         spread of the centre around its mean. For a person keeping
                 their head still this is pure jitter (stationary test).
    second_diff: mean |c[t] - 2c[t-1] + c[t-2]|. This is the frame-to-frame
                 change in velocity. Smooth real motion (even fast) has a small
                 second difference, while jitter makes it large, so this
                 measures jitter during movement (dynamic test).
    """
    std = centres.std(axis=0)
    second = centres[2:] - 2 * centres[1:-1] + centres[:-2]
    return std, float(np.linalg.norm(second, axis=1).mean()) if len(second) else float("nan")


def longest_consecutive_run(frames):
    """Indices of the longest block of consecutive frame numbers."""
    best, start = (0, 0), 0
    for i in range(1, len(frames) + 1):
        if i == len(frames) or frames[i] != frames[i - 1] + 1:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = i
    return best


def analyze_rows(rows):
    frames = sorted({int(r["frame"]) for r in rows})
    if not frames:
        return ["(empty trial)"]
    per_frame = {int(r["frame"]): r for r in rows}   # frame-level fields
    times = [float(per_frame[f]["time_s"]) for f in frames]
    frame_ms = np.array([float(per_frame[f]["frame_ms"]) for f in frames])
    n_faces = np.array([int(per_frame[f]["n_faces"]) for f in frames])
    duration = times[-1] - times[0]

    out = ["=== Task 1 evaluation trial ===",
           f"Frames: {len(frames)}   duration: {duration:.1f} s   "
           f"FPS: {(len(frames) - 1) / duration if duration > 0 else 0:.1f}",
           f"Processing time per frame (detection -> effects drawn): mean {frame_ms.mean():.1f} ms, "
           f"median {np.median(frame_ms):.1f} ms, 95th pct {np.percentile(frame_ms, 95):.1f} ms",
           f"Mean image brightness: {np.mean([float(per_frame[f]['brightness']) for f in frames]):.0f} / 255",
           f"Two-face detection rate (frames with >= 2 faces): {100 * np.mean(n_faces >= 2):.1f} %   "
           f"(>= 1 face: {100 * np.mean(n_faces >= 1):.1f} %)"]

    centres_by_player = {}
    for pid in (1, 2):
        pr = sorted((r for r in rows if int(r["player"]) == pid), key=lambda r: int(r["frame"]))
        status = [r["status"] for r in pr]
        tracked = [r for r in pr if r["status"] == "TRACKED"]
        recognized = [r for r in tracked if r["source"] == "recognized"]
        out.append(f"\n--- Player {pid} ---")
        out.append(f"Detected+assigned: {100 * len(tracked) / len(pr):.1f} %   "
                   f"by embedding: {100 * len(recognized) / len(pr):.1f} %   "
                   f"held: {status.count('HELD')} frames   lost: {status.count('LOST')} frames   "
                   f"never seen: {status.count('SEARCHING')} frames")
        if recognized:
            sims = [float(r["similarity"]) for r in recognized]
            out.append(f"Similarity when recognised: mean {np.mean(sims):.3f}, min {np.min(sims):.3f}")
        if len(tracked) < 3:
            out.append("Not enough tracked frames for jitter/alignment.")
            continue

        # Jitter on the longest run of consecutive tracked frames.
        tf = [int(r["frame"]) for r in tracked]
        a, b = longest_consecutive_run(tf)
        run = tracked[a:b]
        raw = np.array([[float(r["raw_cx"]), float(r["raw_cy"])] for r in run])
        smooth = np.array([[float(r["smooth_cx"]), float(r["smooth_cy"])] for r in run])
        face_w = np.mean([float(r["raw_w"]) for r in run])
        raw_std, raw_d2 = jitter_metrics(raw)
        sm_std, sm_d2 = jitter_metrics(smooth)
        out.append(f"Jitter over {len(run)} consecutive tracked frames (mean face width {face_w:.0f} px)."
                   f" std = stationary trials only; 2nd-diff = any trial:")
        out.append(f"  raw      std x {raw_std[0]:6.2f} px   std y {raw_std[1]:6.2f} px   "
                   f"2nd-diff {raw_d2:6.2f} px")
        out.append(f"  smoothed std x {sm_std[0]:6.2f} px   std y {sm_std[1]:6.2f} px   "
                   f"2nd-diff {sm_d2:6.2f} px")
        if raw_d2 > 0:
            out.append(f"  second-difference jitter reduced by {100 * (1 - sm_d2 / raw_d2):.0f} %")

        # Alignment: how far the drawn effect centre is from the detector's
        # centre on the same frame. This is the lag cost of smoothing.
        offset = np.linalg.norm(raw - smooth, axis=1)
        out.append(f"Effect alignment offset (smoothed vs. raw centre): mean {offset.mean():.1f} px "
                   f"({100 * offset.mean() / face_w:.1f} % of face width), "
                   f"95th pct {np.percentile(offset, 95):.1f} px")

        # Missing / occlusion episodes.
        episodes, gap_start = [], None
        for i, s in enumerate(status):
            if s != "TRACKED" and gap_start is None and i > 0 and status[i - 1] == "TRACKED":
                gap_start = i
            elif s == "TRACKED" and gap_start is not None:
                gap = i - gap_start
                kind = "bridged (effect held)" if gap <= config.MAX_MISSING_FRAMES else "re-acquired after LOST"
                secs = float(pr[i]["time_s"]) - float(pr[gap_start]["time_s"])
                episodes.append(f"  frame {pr[gap_start]['frame']}: missing {gap} frames ({secs:.2f} s), "
                                f"{kind}, via {pr[i]['source']}")
                gap_start = None
        if gap_start is not None:
            episodes.append(f"  frame {pr[gap_start]['frame']}: missing until end of trial (not recovered)")
        out.append(f"Missing episodes: {len(episodes)}")
        out.extend(episodes)

        # Possible identity switches: big jumps of the raw centre within the grace window.
        flags = []
        for prev, cur in zip(tracked, tracked[1:]):
            gap = int(cur["frame"]) - int(prev["frame"])
            if gap > config.MAX_MISSING_FRAMES + 1:
                continue    # player was LOST; re-appearing elsewhere is legitimate
            jump = np.hypot(float(cur["raw_cx"]) - float(prev["raw_cx"]),
                            float(cur["raw_cy"]) - float(prev["raw_cy"]))
            if jump > config.SWITCH_JUMP_FACE_WIDTHS * float(prev["raw_w"]):
                flags.append(f"  frame {cur['frame']}: centre jumped {jump:.0f} px")
        out.append(f"Possible identity switches (jump > {config.SWITCH_JUMP_FACE_WIDTHS} face widths): "
                   f"{len(flags)}  -> verify in the trial video")
        out.extend(flags)
        centres_by_player[pid] = {int(r["frame"]): float(r["raw_cx"]) for r in tracked}

    # Left/right order changes = crossings actually performed in the trial.
    if len(centres_by_player) == 2:
        both = sorted(set(centres_by_player[1]) & set(centres_by_player[2]))
        signs = [np.sign(centres_by_player[1][f] - centres_by_player[2][f]) for f in both]
        crossings = sum(1 for s0, s1 in zip(signs, signs[1:]) if s0 != s1 and s0 != 0 and s1 != 0)
        out.append(f"\nLeft/right order changes between Player 1 and Player 2: {crossings} "
                   f"(frames with both tracked: {len(both)})")
    out.append("Note: switch flags and crossings are automatic indicators, not ground truth. "
               "Confirm identities by watching the saved trial video.")
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python evaluation.py eval_logs/trial_XX.csv")
        sys.exit(1)
    print("\n".join(analyze_rows(load_rows(sys.argv[1]))))
