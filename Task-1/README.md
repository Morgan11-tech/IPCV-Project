# Task 1 — Face Tracking and Augmented Effects

**Responsible student:** _[full name, s-number]_

## Purpose

Task 1 tracks the faces of two registered players with one webcam and attaches a
player-specific visual effect to each face. The effect has to stay smooth,
aligned and on the correct player while players move, tilt their heads, change
distance, swap positions or are briefly occluded. Task 1 also shows each
player's game state (name, score, "+1 CORRECT!" feedback) attached to their
face, so the sign-language game can show who scored.

## Architecture

The application runs in four phases:

```
REGISTER PLAYER 1 → REGISTER PLAYER 2 → PREPARE REFERENCES → LIVE TRACKING
```

1. **Registration (per player).** The player captures 7 webcam samples with
   slightly varied head angle, expression and distance. A sample is rejected if
   no face or more than one face is visible, if detection confidence is below
   0.9, or if the face is smaller than 80 px. For Player 2, a sample is also
   rejected if it matches Player 1. Each accepted sample becomes a 128-D face
   embedding.
2. **Preparation ("training").** Each player's 7 embeddings are averaged and
   re-normalised into one reference embedding. No neural network is trained;
   the pretrained model stays frozen and we only store a template per player.
3. **Live loop (every frame):**
   - **Detect** all faces (box, 5 landmarks, confidence) on a 640 px
     downscaled frame.
   - **Embed** each face: align it to 112×112 using the landmarks, then compute
     a 128-D embedding.
   - **Compare** each embedding with both references using cosine similarity.
   - **Assign** faces to players one-to-one: try every valid combination and
     keep the one with the highest total similarity. A pair only counts if its
     similarity is ≥ 0.45. Faces that are left over are labelled **UNKNOWN**.
   - **Track:** if a known player isn't recognised in a frame (e.g. head
     turned), they keep a face that overlaps their last box (IoU ≥ 0.3).
   - **Smooth** the box (x, y, w, h) and head roll with an Exponential Moving
     Average (α = 0.3) to remove detector jitter.
   - **Handle missing faces:** a player who isn't detected keeps their last
     smoothed position for up to 10 frames (HELD). After that they are marked
     LOST and their effect is hidden until they are recognised again.
   - **Augment:** draw a coloured halo (blue = P1, red = P2) that scales with
     face size and rotates with head roll (computed from the eye landmarks),
     plus a name/score tag, score panels and a "+1 CORRECT!" popup.

Identity always comes from the face embedding, never from screen position, so
effects stay with the right person when players cross.

## Code structure

| File | Responsibility |
|---|---|
| `main.py` | Application flow; `process_frame()` runs the full per-frame pipeline |
| `registration.py` | Camera access, sample validation and capture, saving/loading player data |
| `recognition.py` | Face detection, embeddings, cosine similarity, one-to-one assignment |
| `tracking.py` | Player state, EMA smoothing, IoU matching, roll angle, missing-frame handling, `award_point()` |
| `effects.py` | Face halo, name/score tag, feedback popup, score panels, debug overlay |
| `evaluation.py` | FPS/latency statistics, trial logging, jitter / alignment / occlusion / identity-switch analysis |
| `config.py` | All tunable parameters, each with its reasoning |
| `requirements.txt` | Python dependencies |
| `models/` | The two pretrained ONNX models (downloaded, see Setup) |
| `player_data/` | Saved registration per player (created at runtime, not committed) |
| `eval_logs/` | Evaluation trial CSV, summary and video (created at runtime, not committed) |

## Libraries and models

| Tool | Version | Role |
|---|---|---|
| **OpenCV** (`opencv-python`) | 4.11.0.86 | Webcam capture, image processing, drawing; also loads and runs both models |
| **NumPy** | 2.2.6 | Embedding maths (normalisation, averaging, cosine similarity), evaluation statistics |
| **YuNet** face detector (`cv2.FaceDetectorYN`) | 2023mar ONNX, OpenCV model zoo | Pretrained CNN that outputs face boxes, 5 facial landmarks and a confidence score |
| **SFace** face recogniser (`cv2.FaceRecognizerSF`) | 2021dec ONNX, OpenCV model zoo | Pretrained CNN that turns an aligned face into a 128-D identity embedding |

Both models run on the CPU through OpenCV. No GPU, dlib or extra deep-learning
framework is needed.

**Implemented by us:** embedding normalisation, cosine similarity, reference
averaging, the threshold/UNKNOWN decision, the one-to-one assignment, tracking
state and IoU matching, EMA smoothing, missing-frame handling, roll estimation,
all visual effects, and all evaluation measurements. We do not use OpenCV's
built-in matching helper, so every step of the identity decision is in our own
code.

## Setup

```bash
cd Task-1

# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the pretrained models
mkdir -p models
curl -L -o models/face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
curl -L -o models/face_recognition_sface_2021dec.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
```

On macOS, allow camera access for your terminal/IDE in
System Settings → Privacy & Security → Camera.

## Run

```bash
source .venv/bin/activate
python main.py            # register Player 1 and Player 2, then live mode
python main.py --load     # skip registration, reuse saved players
```

## Controls

| Key | Action |
|---|---|
| SPACE | capture a registration sample |
| R | restart registration |
| 1 / 2 | simulate Player 1 / Player 2 scoring |
| D | toggle debug info |
| E | start / stop an evaluation trial |
| Q | quit |

## Integration with other tasks

Each frame, the game calls `process_frame()`. After the call, `players[1]` and
`players[2]` hold each player's tracked state: visibility, smoothed face box and
centre, face size, head roll and similarity. Calling
`tracking.award_point(players[n])` adds a point and shows "+1 CORRECT!" above
that player's face. The Task 3 identity module can replace `recognize_faces()`
if it returns the same `{player_id: (face_index, similarity)}` format.
