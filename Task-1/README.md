# Task 1 — Face Tracking and Augmented Effects

Standalone prototype for IPCV 2026-1A (University of Twente), to be integrated into
our two-player sign-language reaction game.

**Responsible student:** _[full name, s-number]_

## 1. Purpose

Track the faces of two registered players with one webcam and attach a
player-specific augmented effect to each face. The effect has to stay smooth,
aligned and on the correct player while the players move, turn, change distance,
cross positions or are briefly occluded (IPCV project description §3.2.1).

## 2. What Task 1 covers (and what it does not)

| In Task 1 | Not in Task 1 |
|---|---|
| Two-face detection, face tracking, EMA smoothing | Sign / hand / pose recognition (Task 2) |
| Minimal face-based Player 1 / Player 2 / UNKNOWN recognition, so effects cannot switch players | Full identity tracking of bodies, poses and controls (Task 3) |
| Face effect + game-state effect (name, score, "+1 CORRECT!") | Scoring rules, sign timing (Task 4) |
| Task 1 evaluation measurements | Scene rendering and integration (Task 5) |

## 3. Architecture

```
REGISTER P1 ─► REGISTER P2 ─► PREPARE REFERENCES ─► LIVE LOOP (per frame)
 7 samples      7 samples      mean embedding          detect faces (YuNet)
 → embeddings   → embeddings   per player              → align + embed (SFace, 128-D)
                                                       → cosine similarity vs P1 / P2
                                                       → one-to-one assignment + threshold
                                                         → PLAYER 1 / PLAYER 2 / UNKNOWN
                                                       → IoU continuation for unrecognised faces
                                                       → EMA smoothing of box and roll
                                                       → grace period for missing faces
                                                       → draw effects → display
```

## 4. Installation

```bash
cd Task-1
python3.12 -m venv .venv            # or: uv venv --python 3.12 .venv
source .venv/bin/activate
pip install -r requirements.txt

# Pretrained models from the official OpenCV model zoo (not trained by us)
mkdir -p models
curl -L -o models/face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
curl -L -o models/face_recognition_sface_2021dec.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
```

SHA-256 of the files we tested with:
`8f2383e4…2fa4` (YuNet 2023mar, 233 KB), `0ba9fbfa…4e79` (SFace 2021dec, 38.7 MB).

On macOS, allow camera access for your terminal / IDE
(System Settings → Privacy & Security → Camera).

## 5. How to run

```bash
python main.py          # register Player 1, Player 2, then live mode
python main.py --load   # reuse the last registration from player_data/
python main.py --camera 1
```

## 6. Keyboard controls

| Key | Registration | Live mode |
|---|---|---|
| SPACE | capture a sample | – |
| R | restart this player's samples | go back and register both players again |
| 1 / 2 | – | simulate Player 1 / Player 2 scoring ("+1 CORRECT!") |
| D | – | toggle debug overlay |
| E | – | start / stop an evaluation trial |
| Q / ESC | quit | quit (prints FPS / latency summary) |

## 7. File structure

```
Task-1/
├── main.py           application flow; process_frame() = full per-frame pipeline
├── registration.py   camera helpers, sample validation + capture, save/load
├── recognition.py    detection, embeddings, cosine similarity, 1-to-1 assignment
├── tracking.py       PlayerState, EMA smoothing, IoU, roll angle, grace period, award_point()
├── effects.py        face halo, name/score tag, feedback popup, HUD, debug overlay
├── evaluation.py     FPS/latency stats, trial logging, jitter/alignment/occlusion/switch analysis
├── config.py         all parameters, each with its reasoning
├── requirements.txt
├── models/           the two ONNX files (downloaded)
├── player_data/      player_1/, player_2/: embeddings.npy + aligned sample crops (created at runtime)
└── eval_logs/        trial_XX.csv, trial_XX_summary.txt, trial_XX.mp4 (created at runtime)
```

## 8. Models and libraries

| Component | What it is | Used for |
|---|---|---|
| **YuNet** (`cv2.FaceDetectorYN`, 2023mar) | Small CNN face detector (OpenCV zoo) | Face boxes, 5 landmarks (eyes, nose, mouth corners), confidence |
| **SFace** (`cv2.FaceRecognizerSF`, 2021dec) | CNN face-recognition model (MobileFaceNet-style, trained with a margin-based softmax loss) | Aligned 112×112 crop → 128-D identity embedding |
| OpenCV 4.11, NumPy 2.2 | – | Image I/O, drawing, maths |

**Our own code:** L2 normalisation, cosine similarity, reference averaging,
threshold / UNKNOWN decision, one-to-one assignment, tracking state, IoU
continuation, EMA smoothing, grace period, roll angle, all effects, all evaluation
maths. We do not use OpenCV's `match()` helper, so every step of the similarity
decision is in our code. No network is trained or fine-tuned.

## 9. Important parameters (`config.py`)

| Parameter | Value | Meaning / reason |
|---|---|---|
| `NUM_REGISTRATION_SAMPLES` | 7 | Samples per player with varied pose, expression and distance |
| `REGISTRATION_MIN_SCORE` | 0.9 | Only confident detections become samples |
| `DETECTION_SCORE_THRESHOLD` | 0.7 | Lower in live mode so turned/occluded faces are still found |
| `DETECTION_WIDTH` | 640 | Detect on a downscaled frame: 6.0 ms vs 15.5 ms at 1280 px (M1 Pro) |
| `SIMILARITY_THRESHOLD` | 0.45 | Published SFace value is 0.363; raised after our offline impostor test (see §10.3). **Must be re-validated with our webcam.** |
| `RECOGNITION_INTERVAL` | 1 | Embeddings on every frame (2 faces ≈ 22 ms of model time) |
| `EMA_ALPHA` | 0.3 | Weight of the new measurement; lower = smoother but more lag |
| `MAX_MISSING_FRAMES` | 10 | ≈0.4 s at 25 FPS during which the effect is held |
| `TRACK_IOU_THRESHOLD` | 0.3 | Overlap needed to keep an identity without an embedding match |
| `SMOOTHING_RESET_DISTANCE` | 1.0 face width | Bigger jumps are re-acquisitions, so the EMA restarts |

## 10. Evaluation

### 10.1 Built-in measurements

- **Every run:** on exit, prints average FPS (including camera and display) and the
  average time of each stage (detect, recognize, track+draw, total).
- **Trials (key E):** each trial writes `eval_logs/trial_XX.csv` (one row per player
  per frame: status, similarity, raw and smoothed centre and size, roll, frame time,
  image brightness), an annotated `trial_XX.mp4`, and `trial_XX_summary.txt` containing:
  - two-face detection rate (% of frames with ≥ 2 faces)
  - jitter: std of raw vs smoothed centre (stationary trials) and mean second
    difference |c[t] − 2c[t−1] + c[t−2]| (all trials, including movement)
  - effect alignment offset: distance between the drawn (smoothed) centre and the
    detector's centre, in px and % of face width
  - missing episodes: length, and whether each one was bridged (held) or re-acquired after LOST
  - possible identity switches: centre jumps > 1.5 face widths inside the grace window
  - left/right order changes between the two players (number of crossings)
  - FPS, mean / median / 95th-percentile processing time, mean brightness

  Re-analyse a saved trial with `python evaluation.py eval_logs/trial_01.csv`.

Switch flags are **indicators, not ground truth.** For identity accuracy, watch
`trial_XX.mp4` and count frames or crossings where a label is on the wrong person.

### 10.2 Test protocol (to run with the real webcam)

Register once, then use `python main.py --load` and press **E** at the start and
end of each trial. Repeat each trial 3× and note the distance and lighting.

| # | Trial | Expected | Read from summary / video |
|---|---|---|---|
| 1 | P1 alone | recognised as P1 | similarity, % by embedding |
| 2 | P2 alone | recognised as P2 | same |
| 3 | Both in frame | both correct | two-face rate, video |
| 4 | Swap left/right 3× | labels stay on people | order changes = 3, switch flags, video |
| 5 | Walk closer/further | halo follows position and size | video, alignment offset |
| 6 | Both still, 10 s | stable effect | raw vs smoothed **std** |
| 7 | Fast head movement | follows without large lag | 2nd-diff, alignment offset 95th pct |
| 8 | Hand over face ~0.3 s | effect held, same identity | missing episodes "bridged" |
| 9 | Leave frame, return | correct identity re-acquired | episode "re-acquired", video |
| 10 | Third person | UNKNOWN | video; their similarities in debug overlay |
| 11 | Normal / dim / backlit | record differences | brightness, two-face rate, % by embedding, FPS |
| 12 | Cross with partial occlusion | record any switch | switch flags + video |

**Lighting (trial 11):** run trials 3 and 6 under (a) normal room light, (b) dim
light (main light off), (c) backlit (window or lamp behind players). The mean
brightness value in each summary documents the condition objectively. Report
detection rate, % recognised by embedding, mean similarity and FPS for each.
We do not claim lighting robustness until these numbers exist.

### 10.3 Offline verification done so far (not a substitute for webcam trials)

The development machine could not access a camera, so the pipeline was checked
offline on an M1 Pro:

- **Recognition:** two people from a public group photo (OpenCV zoo example image)
  were "registered" with 7 augmented samples each (rotation, scale, brightness).
  Held-out samples of the same person: similarity 0.89–0.96. Other person: 0.23–0.28.
  Across 13 different people, impostor similarity reached 0.38–0.40, and 11.5 % of
  impostor pairs passed 0.363. That result is why the threshold was raised to 0.45.
- **Synthetic video** (340 frames: stationary → crossing with occlusion →
  scale/roll change → 6-frame and 30-frame occlusion → third person → P1 leaves):
  0 wrong identities in 587 checked player-frames. The 6-frame occlusion was
  bridged, the 30-frame occlusion was re-acquired correctly, and the third person
  and background faces stayed UNKNOWN. Second-difference jitter was reduced by 77 %
  for both players.
- **Timing:** YuNet 6.0 ms (640 px input), SFace 8.1 ms per face, drawing ≈1.5 ms,
  so ≈24 ms of processing per frame with two faces.

Augmented copies of one photo are much easier than real pose changes, so the
similarity values above are optimistic. Real webcam results must be measured
with §10.2.

## 11. Known limitations

- **Threshold is not yet validated on the real webcam.** 0.45 is based on offline
  data. Check the prepare-phase report (own-sample vs other-player similarity) and
  the trial similarities.
- Large head turns (profile) lower both the detector score and the similarity.
  Identity is then carried only by IoU continuation. If the face is also lost for
  more than 10 frames, recovery waits for a more frontal view.
- IoU continuation can, in principle, hand a track to a different face that appears
  at the same place within the grace window (for example one player's face exactly
  covering the other's while the covered one is unrecognised). Embedding assignment
  runs first on every frame, which limits this, but it is the main residual switch risk.
- A HELD effect stays where the face was last seen (no motion prediction), so it
  can lag during fast movement combined with occlusion.
- Orientation: only in-plane **roll** (from the eyes) is estimated and used. Yaw
  and pitch are not estimated.
- EMA with a fixed alpha trades jitter against lag. An adaptive filter
  (e.g. One-Euro) would reduce lag during fast motion.
- Two players are hard-coded (assignment enumerates P1 × P2). More players would
  need Hungarian assignment.
- Embeddings are biometric data. `player_data/` stays local and is in `.gitignore`.

## 12. Integration interface (for Tasks 3–5)

```python
from recognition import load_models, build_reference_embedding
from registration import load_player_data
import tracking
from main import process_frame

detector, recognizer = load_models()
references = {pid: build_reference_embedding(load_player_data(pid)) for pid in (1, 2)}
players = tracking.create_players()          # {1: PlayerState, 2: PlayerState}

# every frame (BGR, already mirrored):
display, faces, timings = process_frame(frame, detector, recognizer, references,
                                        players, frame_idx, debug=False)

p1 = players[1]
p1.visible, p1.status             # TRACKED / HELD / LOST / SEARCHING
p1.smooth_box, p1.center          # smoothed face geometry (pixels)
p1.face_width, p1.roll, p1.similarity
tracking.award_point(p1)          # score += 1 and "+1 CORRECT!" above P1's face
tracking.show_feedback(p1, "TOO SLOW")
```

`process_frame` draws on a copy of the frame and returns it, so Task 5 can
either use that image or draw its own scene using `players[...]`.
The Task 3 identity module can replace `recognize_faces()` as long as it returns
the same `{player_id: (face_index, similarity)}` assignment.
