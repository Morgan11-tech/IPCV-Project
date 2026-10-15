# IPCV Task 2 — Hand Pose Estimation & Dynamic Sign Classification

Sign-language speed game — Task 2 module. Detects a hand, classifies **dynamic**
(motion-based) signs, and emits confirmed sign events for Task 3 (player
identity) and Task 4 (scoring).

## Setup

```bash
cd ipcv-task2
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name=ipcv-task2 --display-name "Python (ipcv-task2)"

# Download the hand landmark model (not committed to git, see models/ below)
mkdir -p models
curl -L -o models/hand_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
```

In VS Code: open a notebook under `notebooks/`, select the
**Python (ipcv-task2)** kernel (top right), and run the cells top to bottom.

**Note on MediaPipe version:** this project pins `mediapipe==0.10.35`, not the
newer 1.0.x. MediaPipe 1.0.1 currently crashes on Apple Silicon Macs when
loading `HandLandmarker` (`Check failed: service_ Service is unavailable`,
[known bug](https://github.com/google-ai-edge/mediapipe/issues/6356)). Also,
MediaPipe removed the old `mp.solutions.hands` API starting from 1.0 — this
project uses the newer `mediapipe.tasks.python.vision.HandLandmarker` API
instead. Don't upgrade mediapipe without checking that bug is fixed.

## Final vocabulary (14 words) — chosen from data

**Words: Apple, Please, Color, Yes, No, More, Boy, Hello, Bathroom, Yesterday, Brown, Football, Water, Red**
(+ the required Idle class). Nothing was recorded by hand except Idle.

**Glasses was removed after live testing** (it passed the dataset test at 80% / 8% wrong, but with a real left hand
it was mistaken for Apple; Yesterday had the same problem at first). The model went from 15 to 14 words.
**A 15th word was searched for and not found:** six more candidates (Government, Late, Delicious, World, Year,
Family) were each added alone to the 14 words and tested on held-out recording sources; none passed the bar
(confirmed >= 60%, <= 10% wrong, no harm to existing words). Closest was Late (57% / 5% wrong with the right hand)
but with simulated left-hand signing it stole from Yes (13x), Color (7x), Football and No, so it was not added.
Results are in `data/external/adding/` (`scripts/try_adding.py`). Backup of the 15-word model:
`models/sign_classifier_15words_v4.joblib`.

**How the words were chosen** (all from the public dataset; the same procedure was used twice)
1. Screened **300 dataset words** in two batches of 150 (only words recorded by several different sources,
   40 videos each, balanced across recording sources): `scripts/download_videos.py --words-file`,
   `scripts/extract_features.py --words-file`.
2. Stage 1, `scripts/screen_stage1.py` (cheap): each candidate trained alongside the current words and tested on
   videos from a **recording source the model never saw**. Best of the second batch: Yesterday 98%, Football 98%,
   Water 89%, Pink 90%, Brown 92%, Red 85%, Glasses 87%.
3. Stage 2, `scripts/select_words.py` (strict, the real recogniser pipeline): the top 14 candidates plus the 9
   existing words were replayed together with one whole recording source held out at a time; the worst
   candidate(s) were removed each round (7 rounds). Removed: Dance (stole Yes/No), Single, Accident, Awake, Alone,
   Awkward, Pink, Chair. A candidate stays only if it is confirmed >= 60% with <= 10% wrong and does not steal
   from the existing words. All six survivors passed.

**Result for the final 15 words** (strict test: held-out recording sources; the final model trains on all sources,
so live numbers should be a bit better; still not a guarantee):

| test | right word confirmed | wrong word confirmed |
|---|---|---|
| 15 words, right-hand-like videos | **68%** | **10%** |
| 15 words, simulated LEFT hand (videos mirrored) | **67%** | **10%** |
| the 6 new words alone (inside the 15) | 75% | 8% |
| (earlier, for comparison) 24 words | 46% | 19% |

Per word, right-hand-like / left-hand (right word confirmed % / wrong %): Please 88/4 and 88/4, Apple 82/5 and 72/10,
Color 72/12 and 70/12, Yes 71/9 and 66/14, No 66/7 and 64/7, Boy 60/12 and 60/12, More 59/4 and 63/6,
Bathroom 49/24 and 44/22, Hello 39/15 and 33/11, Glasses 80/8 and 70/10, Yesterday 78/10 and 80/8,
Brown 75/8 and 75/5, Football 75/10 and 82/15, Water 72/8 and 80/2, Red 68/5 and 68/5.

**Known weak words:** Bathroom (often confirmed as Apple) and Hello (often missed) are below the bar on dataset
videos. Both were kept because they work well in live testing by the team. Boy is sometimes mistaken for Please.

**Key findings (good report material)**
- Wider search + more careful selection beat my first attempt: the first 150 screened words only supported ~8-9
  words; the next 150 plus the left-hand fix supported 15 at the same error rate.
- **Single-source words are traps.** Cat and Milk (each from ONE recording session) scored well in random splits
  but, in a multi-source test, Cat became the top wrong answer for other words. They were removed.
- **Left vs right hand (found from live testing, then measured).** The training signers are mostly right-handed,
  so a left hand (a mirror image) looked different to the model. Replaying the test videos mirrored to simulate
  a left-handed signer dropped accuracy from 65% to 50% (Apple 80 -> 32%, Please 77 -> 42%, Yes 71 -> 45%).
  Fix: the signing hand's Left/Right label (majority vote over recent frames) decides whether its features are
  mirrored into right-hand form, in both training (`extract_features.py`) and live (`recognizer.py`). Left-hand
  accuracy is now equal to right-hand. Caveat: it relies on MediaPipe's left/right label.
- **Please is an attractor** (many words are mistaken for it).
- Father/Mother (earlier 25-word test): same handshape, differ only by forehead vs chin -> confused.
- Caveats: dataset videos (not the team's webcam); no 100% guarantee; recording-source labels are guessed from
  filename patterns.

**Files.** Training data: `data/raw/hf_final_15words_mirrored.csv` (the 15 words, left-hand mirrored) plus the Idle
clips. Backups in `models/`: `sign_classifier_9words_v3.joblib` (9 words), `sign_classifier_8words_v2.joblib`,
`sign_classifier_8words_v1.joblib`, `sign_classifier_all25.joblib`. The screening pool (200 words, mirrored features)
is `data/external/candidates/`; the selection rounds and rankings are in `data/external/selection/` and
`data/external/stage1_ranking.csv`. To train on other words edit `KEEP_WORDS` in notebook 03. To rebuild the final
file: download with `scripts/download_videos.py --words-file ...`, extract with `scripts/extract_features.py
--words-file data/external/final_words.json --out-dir data/raw`.

## Sign vocabulary (all 25 candidate words)

25 ASL words, chosen by the team, plus a required **Idle** class
(natural non-signing movement — needed so the live system can tell a real
sign apart from normal hand motion between signs). Since this pipeline
tracks **one hand only** (no face/body reference), only the dominant
signing hand is used, even for signs that are two-handed in real ASL.

**Important:** the training clips come from real ASL videos (see "Training
data" below), so your own webcam recordings should imitate the *real ASL
sign as shown in those videos* — watch a few in `data/external/videos/<Label>/`
before recording. The table below is a rough description of each gesture
(some entries are my one-handed simplifications of two-handed signs), not
the ground truth; the dataset videos are.

| Sign (label used in code) | Gesture performed |
|---|---|
| Hello | Flat hand near forehead/temple, short outward motion |
| SeeYouLater | "L" handshape (thumb+index out), rotates forward from vertical to pointing away |
| IMe | Index finger, repeated poke toward own chest |
| Father | Open "5" hand, thumb taps forehead a couple times |
| Mother | Open "5" hand, thumb taps chin a couple times |
| Yes | Closed fist nodding up and down |
| No | Thumb + index + middle fingers snapping open/closed |
| Help | One-handed approx: closed fist, thumb up, moving straight upward |
| Please | Flat hand, circular motion on chest |
| ThankYou | Flat hand starts touching chin, moves outward and down |
| Want | One-handed approx: open claw-shaped hand pulling inward toward the body |
| What | One-handed approx: open hand palm-up, short shake/shrug side to side |
| Dog | One-handed approx: snap fingers a couple times |
| Cat | Thumb+index pinch near cheek, pulling outward twice (whiskers) |
| Again | Bent/claw hand tapping forward into the air, repeated |
| EatFood | Fingertips pinched together, hand moves repeatedly toward the mouth |
| Milk | Loose fist squeezing open/closed repeatedly |
| More | One-handed approx: fingertips tapping against the thumb repeatedly |
| GoTo | Index finger pointing forward, hand moves away from the body |
| Bathroom | "T" handshape (fist, thumb between index/middle), shaking side to side |
| Fine | Flat hand (thumb extended), starts at chest, moves outward |
| Like | Open "5" hand on chest, pulls into a pinch while moving outward |
| Learn | One-handed approx: fingers close into a pinch, then move up toward the forehead |
| Sign | Index finger extended, small forward circles in the air |
| Finish | Hand starts palm-in near chest, quickly flips so palm faces out/down |
| Idle | *Not a sign* — natural random hand movement (required negative class) |

**Known limitation, by design:** `Father` and `Mother` use the identical
handshape and motion in real ASL — they differ only by touching the
forehead vs. the chin. This pipeline has no face/body position reference, so
these two are expected to be confused with each other. This is intentionally
kept as a documented failure case for the report (required per the brief:
"discuss significant failure cases") rather than worked around, since
explaining *why* it fails is itself good evidence of understanding the
system's limits.

**Other pairs worth watching in the confusion matrix** (moderate risk, likely
separable via handshape, but keep an eye on them): Please vs. Sorry-style
circular motions aren't both present here, but ThankYou vs. Fine vs. Like
all share a "starts near the body, moves outward" trajectory shape —
expect these to be the next most-confused group after Father/Mother.

Edit the `SIGNS` list in `02_data_collection.ipynb` to change the vocabulary.

## Interface to Task 3 (agreed team format)

`src/recognizer.py` is the importable Task 2 module. For each camera frame:

```python
from src.recognizer import SignRecognizer
rec = SignRecognizer()                       # loads models/ (mirror_input=True if your pipeline already mirrors the frame)
out = rec.update(frame_bgr, frame_id)        # optional: timestamp=seconds
```

`out` is **exactly** the agreed format (checked on every frame by `scripts/check_interface.py`):

```
{ "frame_id": 812,
  "hands": [ { "points": [[x, y, z] x 21], "confidence": 0.94, "handedness": "Right", "box": [x, y, w, h] },
             { ... } ] }
```

- `points`: 21 **smoothed** landmarks (One-Euro filter), x/y normalised 0..1 across the frame you passed
  in (top-left origin, y down), z = MediaPipe relative depth (wrist = 0). Points can fall slightly outside
  0..1 when a hand is partly out of frame (y = 1.01 happens); `box` is clipped to 0..1.
- `box`: `[x, y, w, h]`, normalised, top-left origin, from the smoothed points.
- `handedness`: the person's real hand (MediaPipe's label is inverted for mirrored views; corrected). Left hands are
  also mirrored into right-hand form internally for classification (the output points are NOT mirrored).
- `confidence`: MediaPipe's handedness score for that detection.
- Hands are listed left-to-right (by box centre x) in the frame you passed.
- Coordinates refer to the frame as passed in. The recogniser flips internally only because the classifier
  was trained on mirrored frames; if every module receives an already-mirrored frame, use `mirror_input=True`.
- No hand detected -> `"hands": []`. A hand missing for < 0.3 s keeps its smoothing/recognition state.

**The agreed format has no field for the recognised word**, so it is exposed separately and the format is
untouched: `rec.signs` is a list aligned 1:1 with `out["hands"]`, each entry
`{"track_id", "guess", "guess_confidence", "confirmed", "label"}` (`confirmed` is True only on the frame a sign is
confirmed; `track_id` stays the same while that hand stays visible). `rec.timing` has the last frame's
`detect_ms / recognize_ms / total_ms`. If the team wants the sign inside the hand object, it is a one-line change.

**Smoothing / responsiveness trade-off** (One-Euro, `min_cutoff=0.5, beta=20`; measured on synthetic motion
with known ground truth: 30 fps, 2.5 px of simulated jitter, a 2 Hz hand wave):

| setting | jitter when still | lag in fast motion |
|---|---|---|
| raw MediaPipe | 1.0x | 0 ms |
| min_cutoff 1.0, beta 50 | 1.9x lower | 9 ms |
| **min_cutoff 0.5, beta 20 (used)** | **2.6x lower** | **18 ms** |
| min_cutoff 0.5, beta 5 | 3.7x lower | 39 ms |
| min_cutoff 0.3, beta 5 | 4.2x lower | 41 ms |

Smoothing is applied to the *output* points only; the classifier is fed the raw landmarks it was trained on.

**Measured speed** (Apple M5, 1280x480 two-player test frame, 2 hands, classifier running on both):
mean 9.9 ms, p95 25 ms per frame (~100 fps capacity). Not yet measured on the integrated game.

**Aspect-ratio correction (important finding).** Landmarks are fractions of image width/height, so the same
hand looks horizontally squashed on a 16:9 webcam compared with the 4:3 training videos. In a 4:3-vs-16:9
test, the right word was confirmed in 52% vs 44% of videos before the fix. `extract_frame(..., aspect=W/H)`
now corrects this (4:3 input is unchanged); the recogniser, the notebooks and the dataset extractor all pass
the real aspect. After the fix 16:9 reaches 48%; the remaining gap is the detector seeing smaller hands in a
wider image. Two-player side-by-side frames (2.67:1) are the hard case: one of two signs confirmed in the
simulated test. If hands are lost in wide frames, tiling the detection per player is the next step.
`scripts/fix_idle_aspect.py` converted the Idle clips recorded before the fix (approximate: removes ~97% of the
distortion on average).

## Training data

Two sources, merged by `03_train_classifier.ipynb`:

1. **Public dataset (bulk of training data):** `akasheroor/American-Sign-Language-Dataset`
   on Hugging Face (MIT license; a duplicate of `ZahidYasinMittha/American-Sign-Language-Dataset`,
   ~108k short ASL word videos). We use a seeded random subset of up to 60
   videos for each of our 25 words (1,360 videos, ~560 MB; 1,348 usable after
   hand detection). Mapping: `SeeYouLater` uses the dataset's *later*,
   `EatFood` uses *eat*, `GoTo` uses *go*, `IMe` uses *i*, `ThankYou` uses *thankyou*.
   *Caveat for the disclosure statement:* the repo is MIT-licensed, but the
   clips were scraped from several upstream sources whose individual terms
   we haven't verified.
2. **Own webcam recordings (Idle only, by default):** `02_data_collection.ipynb`
   records ~40 `Idle` clips (natural non-signing hand movement), because the
   dataset has no such class. Optionally, a few own clips per sign can be
   recorded too, purely to measure the dataset-to-webcam accuracy gap — the
   25 signs themselves are trained from the dataset.

Reproduce the dataset part (run from the project root, venv active):

```bash
python - <<'EOF'   # one-time: index every file in the repo -> data/external/hf_file_index.json
import json, os
from huggingface_hub import HfApi
files = HfApi().list_repo_files("akasheroor/American-Sign-Language-Dataset", repo_type="dataset")
idx = {}
for f in files:
    if f.endswith(".mp4"): idx.setdefault(os.path.basename(f), []).append(f)
os.makedirs("data/external", exist_ok=True)
json.dump(idx, open("data/external/hf_file_index.json", "w"))
EOF
python scripts/download_videos.py --cap 60      # selective download (~560 MB)
python scripts/extract_features.py --words-file data/external/final_words.json --out-dir data/raw   # videos -> feature CSV (~2 min)
```

`extract_features.py` mirrors each frame (to match the mirrored live
webcam), detects up to 2 hands, tracks them across frames, keeps the hand
that moves most (the dominant signing hand), and builds the same
shape + trajectory feature vector used live.

**Measured baseline (Random Forest, dataset clips only, 25 classes, chance = 4%):**
~70% with a random 5-fold split (optimistic — near-duplicate clips from one
signer/session can land on both sides); 44–73% when a whole recording
"style" is held out. Weakest words: What (25% recall), Finish, Again, GoTo.
Most confused pairs: What→Want, **Father↔Mother (19 mix-ups, the expected
failure case)**, Help↔Milk. Not yet measured: accuracy on our own webcam
(see the domain-gap cell in notebook 03).

**Live-recognition replay test** — *measured before the aspect-ratio fix and before Idle existed* (24 held-out dataset videos — 8 words × 3,
not in the training set — played through the real `04_live_recognition.ipynb`
loop at real-time speed, frames delivered by wall-clock like a webcam):

| Setting | Signs confirmed correctly | Wrong confirmations |
|---|---|---|
| Original (threshold 0.6, needs 0.84 s of motion) | 10–11 / 24 (42–46%) | 0 |
| + threshold 0.35 with 0.3 s hold | 11 / 24 | 0 |
| + needs only 0.6 s of motion (`MIN_SPAN_FRACTION=0.5`) — **shipped** | **14 / 24 (58%)** | 1 |

Why it was confirming so little: (1) a 25-way Random Forest spreads its
probability thin — correct predictions have a median top probability of only
~0.5, so a 0.6 cut-off confirmed ~26% of clips; (2) most ASL words last ~1 s,
but the loop refused to classify until 0.84 s of motion had accumulated, so
short signs were over first (7 of 24 videos were never classified at all).
Remaining misses are mostly per-signer variation (e.g. some *Help*/*Dog*
videos get P(true word) ≈ 0.07). Training on overlapping 1.2 s windows
instead of whole-sign clips was tried and did **not** help (11/24), so
whole-sign clips remain the default (`extract_features.py --window 0`).
Small sample (24 videos): treat differences of 1–3 signs as noise.

## Notebooks (run in order)

1. **`01_hand_tracking_setup.ipynb`** — verifies webcam + MediaPipe hand
   tracking work, with corrected left/right handedness labeling.
2. **`02_data_collection.ipynb`** — records labeled gesture *clips* (not
   single frames) per sign, resampled to a fixed length, saved to
   `data/raw/*.csv`.
3. **`03_train_classifier.ipynb`** — merges all collected CSVs, trains a
   Random Forest on the resampled sequences, evaluates (accuracy, confusion
   matrix, feature importance), saves the model to `models/sign_classifier.joblib`.
4. **`04_live_recognition.ipynb`** — real-time recognition (single hand): rolling buffer,
   motion gating (to ignore idle movement), confidence threshold, and a
   cooldown so one gesture isn't double-counted. Prints confirmed events —
   this is the interface Task 3/Task 4 would consume.

5. **`05_multi_hand_demo.ipynb`** — live demo of `src/recognizer.py`: multi-hand
   tracking, smoothed points, per-hand signs, in the agreed output format.

## Project structure

```
ipcv-task2/
├── notebooks/    # 01 tracking setup, 02 data collection, 03 training, 04 live recognition
├── src/          # shared modules: features.py (shape+trajectory features, resample, motion), hand_model.py (landmarker + handedness fix)
├── scripts/      # download_videos.py, extract_features.py (dataset -> CSV), check_interface.py (format test), fix_idle_aspect.py
├── data/
│   ├── raw/      # training CSVs: hf_dataset_*.csv + own webcam recordings (gitignored)
│   └── external/ # downloaded dataset videos + file index (gitignored)
├── models/       # hand_landmarker.task + sign_classifier.joblib (gitignored, see Setup)
├── requirements.txt
└── README.md
```

## Status
- [x] Environment set up, MediaPipe + OpenCV verified
- [x] Sign vocabulary chosen (25 words + Idle, one-handed interpretations defined)
- [x] Data collection notebook (clip-based, with recording progress bar)
- [x] Shared `src/` modules (shape+trajectory feature extraction, resampling, motion energy, handedness fix)
- [x] Fixed a feature-design bug where whole-hand movement (not just finger shape)
      wasn't being captured — needed for signs like Hello/ThankYou/Finish/GoTo
      that are defined by hand *movement*, not just handshape
- [x] Training notebook (Random Forest + evaluation)
- [x] Live recognition notebook (rolling buffer, motion gate, confirmation + cooldown)
- [x] Public dataset downloaded and converted to features (1,348 clips, 25 words)
- [x] Baseline classifier measured on dataset clips (~70%, see "Training data")
- [x] Idle clips recorded (54, converted to the aspect-corrected feature space)
- [x] Classifier trained on the final 14 words + Idle (15 classes), left-hand mirroring in place; see "Final vocabulary"
- [ ] (Optional) own-webcam clips per sign + domain-gap check in notebook 03
- [ ] Motion threshold / confidence threshold tuned from real calibration
- [x] Multi-hand recogniser in the agreed Task 2 -> Task 3 format (`src/recognizer.py`, notebook 05)
- [x] Landmark smoothing (One-Euro) with measured jitter/lag trade-off
- [ ] Test the recogniser live with two real people at a real webcam
- [ ] Confirm with the team: where the recognised sign travels (see "Interface to Task 3"), mirrored-or-not frame
- [ ] Report evidence gathered: accuracy, latency, false-positive rate, failure cases
      (Father/Mother confusion expected and should be discussed explicitly)
