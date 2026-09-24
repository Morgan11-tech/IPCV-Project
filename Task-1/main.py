"""
Task 1 - Face tracking and augmented effects (standalone prototype).

    REGISTER PLAYER 1 -> REGISTER PLAYER 2 -> PREPARE REFERENCES -> LIVE MODE

Live mode, per frame:
    detect faces -> embeddings -> compare with Player 1 / Player 2
    -> one-to-one assignment (PLAYER 1 / PLAYER 2 / UNKNOWN)
    -> update tracks -> EMA smoothing -> grace period for missing faces
    -> draw player effects -> display

Run:  python main.py          (register two players)
      python main.py --load   (reuse the last saved registration)
"""
import argparse
import time

import cv2
import numpy as np

import config
import effects
import evaluation
import registration
import tracking
from recognition import build_reference_embedding, cosine_similarity, detect_faces, \
    load_models, recognize_faces

WINDOW = "IPCV Task 1 - Face Tracking"


def prepare_references(embeddings):
    """
    PREPARE / "TRAIN" phase.

    embeddings: {1: (N,128) array, 2: (N,128) array}
    Returns the two reference embeddings plus a short report of the
    similarities we observe, which is our evidence for whether
    SIMILARITY_THRESHOLD separates these two players.
    """
    references = {pid: build_reference_embedding(e) for pid, e in embeddings.items()}
    report = []
    for pid in (1, 2):
        other = 2 if pid == 1 else 1
        own = [cosine_similarity(e, references[pid]) for e in embeddings[pid]]
        cross = [cosine_similarity(e, references[other]) for e in embeddings[pid]]
        report.append(f"Player {pid} samples vs own reference: min {min(own):.2f} mean {np.mean(own):.2f}"
                      f"  | vs Player {other}: max {max(cross):.2f}")
    report.append(f"Player 1 reference vs Player 2 reference: "
                  f"{cosine_similarity(references[1], references[2]):.2f}"
                  f"   (threshold {config.SIMILARITY_THRESHOLD})")
    return references, report


def process_frame(frame, detector, recognizer, references, players, frame_idx, debug):
    """
    Full Task 1 pipeline for one frame. This is the function the integrated
    game (Task 5) can call every frame; afterwards `players[1]` and
    `players[2]` hold the tracked face state.

    Returns (display_frame, faces, timings_ms).
    """
    timings = {}
    t0 = time.perf_counter()
    faces = detect_faces(detector, frame)
    t1 = time.perf_counter()

    assignments, similarities = {}, None
    if faces and frame_idx % config.RECOGNITION_INTERVAL == 0:
        assignments, similarities = recognize_faces(recognizer, frame, faces, references)
    t2 = time.perf_counter()

    unknown = tracking.update_all_tracks(players, faces, assignments, similarities, frame_idx)

    display = frame.copy()
    for f in unknown:
        effects.draw_unknown_face(display, faces[f],
                                  similarities[f] if similarities is not None else None, debug)
    for player in players.values():
        effects.draw_face_effect(display, player)
        effects.draw_player_tag(display, player)
        effects.draw_feedback(display, player)
    effects.draw_hud(display, players)
    t3 = time.perf_counter()

    timings["detect"] = (t1 - t0) * 1000
    timings["recognize"] = (t2 - t1) * 1000
    timings["track_draw"] = (t3 - t2) * 1000
    timings["total"] = (t3 - t0) * 1000
    return display, faces, timings


def run_live(cap, detector, recognizer, references):
    """Live loop. Returns "reregister" if R was pressed, otherwise None."""
    players = tracking.create_players()
    debug = config.DEBUG_MODE
    run_stats = evaluation.new_run_stats()
    trial = None
    fps, last_time = 0.0, time.perf_counter()
    frame_idx = 0
    result = None

    while True:
        frame = registration.read_frame(cap)
        if frame is None:
            print("Camera stopped delivering frames.")
            break

        display, faces, timings = process_frame(frame, detector, recognizer, references,
                                                players, frame_idx, debug)

        # FPS of the whole loop (camera + processing + display), itself
        # EMA-smoothed so the number on screen is readable.
        now = time.perf_counter()
        instant_fps = 1.0 / max(now - last_time, 1e-6)
        fps = instant_fps if fps == 0 else tracking.smooth_value(fps, instant_fps, 0.1)
        last_time = now

        if debug:
            effects.draw_debug(display, players, fps, timings, recording=trial is not None)
        evaluation.record_run_frame(run_stats, timings)
        if trial is not None:
            evaluation.record_trial_frame(trial, frame_idx, frame, display, faces, players, timings)

        cv2.imshow(WINDOW, display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("d"):
            debug = not debug
        elif key in (ord("1"), ord("2")):
            # Simulated scoring; the real game module will call award_point().
            tracking.award_point(players[int(chr(key))])
        elif key == ord("e"):
            if trial is None:
                trial = evaluation.start_trial((display.shape[1], display.shape[0]))
            else:
                evaluation.finish_trial(trial)
                trial = None
        elif key == ord("r"):
            result = "reregister"
            break
        frame_idx += 1

    if trial is not None:
        evaluation.finish_trial(trial)
    evaluation.print_run_summary(run_stats)
    return result


def register_both(cap, detector, recognizer):
    """Phases 1 and 2. Returns {1: embeddings, 2: embeddings} or None on quit."""
    result = registration.register_player(cap, WINDOW, detector, recognizer, 1)
    if result is None:
        return None
    emb1, crops1 = result
    result = registration.register_player(cap, WINDOW, detector, recognizer, 2,
                                          other_reference=build_reference_embedding(emb1))
    if result is None:
        return None
    emb2, crops2 = result
    registration.save_player_data(1, emb1, crops1)
    registration.save_player_data(2, emb2, crops2)
    return {1: emb1, 2: emb2}


def main():
    parser = argparse.ArgumentParser(description="IPCV Task 1: face tracking and augmented effects")
    parser.add_argument("--load", action="store_true", help="reuse the saved player registration")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX, help="webcam index")
    args = parser.parse_args()

    detector, recognizer = load_models()
    cap = registration.open_camera(args.camera)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    embeddings = None
    if args.load:
        embeddings = {pid: registration.load_player_data(pid) for pid in (1, 2)}
        if any(e is None for e in embeddings.values()):
            print("No saved registration found, starting registration.")
            embeddings = None

    while True:
        if embeddings is None:
            embeddings = register_both(cap, detector, recognizer)
            if embeddings is None:
                break

        # Phase 3: PREPARE / "TRAIN"
        references, report = prepare_references(embeddings)
        print("\nPreparing player recognition...")
        print("\n".join(report))
        registration.show_message(cap, WINDOW, ["Preparing player recognition..."] + report, 3.0,
                                  color=(255, 255, 0))

        # Phase 4: LIVE
        if run_live(cap, detector, recognizer, references) != "reregister":
            break
        embeddings = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
