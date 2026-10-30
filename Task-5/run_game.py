"""
Task 5 - the integrated sign race (one Python program, OpenCV window).

Start-up:    title + loading screen (models load in the background)
             -> face registration if needed -> game
Per frame:   camera -> perception (Tasks 1-3) -> game flow (Task 4 rules)
             -> renderer (everything on screen) -> window

    pipeline.py   Tasks 1-3 + adapters -> one PlayerInput per player
    flow.py       Task 4's GameController + round scheduling
    render/       all drawing (theme.py holds every colour)
    measure.py    stage timings and FPS

Run from the project root:
    python Task-5/run_game.py --register          # register faces, then play
    python Task-5/run_game.py                     # reuse saved faces
    python Task-5/run_game.py --fake-p2           # play alone (fake Player 2)

Keys: Q quit   D debug   F fullscreen   R restart game   M mute   B arena on/off
"""
import argparse
import time

import bootstrap                               # must come first: sets up imports and paths
import cv2

import config
import registration
from task3_player_association import draw_debug

from audio import Audio
from flow import GameFlow
from loading import LoadingScreen
from measure import FpsMeter, FrameTimer
from pipeline import Models, PerceptionPipeline, get_references, loading_steps
from render import Renderer
from render.overlays import draw_fake_player, draw_stats

WINDOW = "IPCV Sign Race"


def main():
    parser = argparse.ArgumentParser(description="IPCV sign race (Task 5 integration)")
    parser.add_argument("--register", action="store_true", help="register faces before playing")
    parser.add_argument("--fake-p2", action="store_true", help="play alone against a fake Player 2")
    parser.add_argument("--no-pose", action="store_true", help="skip pose (faster, Task 3 uses circles)")
    parser.add_argument("--no-arena", action="store_true", help="plain camera background, no segmentation")
    parser.add_argument("--mute", action="store_true", help="no music or sound effects")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX, help="webcam index")
    args = parser.parse_args()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    audio = Audio(enabled=not args.mute)                 # lobby music starts on the title screen
    loading = LoadingScreen(WINDOW, config.FRAME_WIDTH, config.FRAME_HEIGHT)
    loading.show("Starting camera...", 0.0)
    cap = registration.open_camera(args.camera)          # main thread only (macOS)
    models = Models()
    steps = loading_steps(models, use_pose=not args.no_pose)
    if not loading.run(steps, start_frac=1 / (len(steps) + 1), audio=audio):
        audio.close()
        cap.release()
        return

    references = get_references(cap, WINDOW, models.detector, models.recognizer,
                                args.register, args.fake_p2)
    if references is None:
        return

    perception = PerceptionPipeline(models, references, fake_p2=args.fake_p2,
                                    use_segmentation=not args.no_arena)
    flow, renderer = GameFlow(), Renderer()
    timer, fps_meter = FrameTimer(), FpsMeter()

    debug, fullscreen = config.DEBUG_MODE, False
    start = time.perf_counter()
    frame_id = 0

    while True:
        frame = registration.read_frame(cap)
        if frame is None:
            print("Camera stopped delivering frames.")
            break
        t0 = timer.start()
        t_ms = int((t0 - start) * 1000)           # one clock for every module

        seen = perception.step(frame, frame_id, t0, timer)

        events = flow.update(t_ms, seen.inputs, frame_id)
        snap = flow.snapshot(t_ms)
        for p in snap["players"]:
            seen.players[p["player_id"]].score = p["score"]   # the face name tag shows the game score
        audio.on_events(events)
        audio.update(snap)
        timer.lap("T4 game")

        display = renderer.draw(frame, seen, snap, events, t_ms)
        if args.fake_p2:
            draw_fake_player(display, seen.inputs[2])
        if debug:
            draw_debug(display, perception.assoc, seen.hands)
        timer.lap("draw")
        if debug:
            draw_stats(display, fps_meter.tick(), timer)
        else:
            fps_meter.tick()

        cv2.imshow(WINDOW, display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("d"):
            debug = not debug
        elif key == ord("f"):
            fullscreen = not fullscreen
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
        elif key == ord("b"):
            perception.use_segmentation = not perception.use_segmentation
        elif key == ord("m"):
            audio.toggle_mute()
        elif key == ord("r"):
            flow.reset()
            renderer.reset()
            audio.reset()
        frame_id += 1

    audio.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
