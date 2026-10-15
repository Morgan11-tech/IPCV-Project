# Start here — IPCV Task 2 (hand tracking + sign recognition)

Recognises 14 ASL words from a webcam (Apple, Please, Color, Yes, No, More, Boy, Hello, Bathroom, Yesterday,
Brown, Football, Water, Red) plus an "Idle" class, for up to 4 hands at once, and outputs hands in the team's
agreed Task 2 -> Task 3 format. Full details, results and design decisions: **README.md**.

## Set up (tested on macOS Apple Silicon, Python 3.14; other platforms untested)
```bash
cd ipcv-task2
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name=ipcv-task2 --display-name "Python (ipcv-task2)"
```
`mediapipe` is pinned to 0.10.35 on purpose (1.0.x crashes on Apple Silicon; see README).

## Run
- Open `notebooks/05_multi_hand_demo.ipynb` in VS Code / Jupyter, pick the **Python (ipcv-task2)** kernel, run top to
  bottom. Press `q` in the video window to quit. (Needs a webcam and camera permission.)
- Or from Python:
  ```python
  from src.recognizer import SignRecognizer
  rec = SignRecognizer()
  out = rec.update(frame_bgr, frame_id)   # {"frame_id": ..., "hands": [{"points","confidence","handedness","box"}]}
  rec.signs                               # recognised word per hand (side channel), see README "Interface to Task 3"
  ```

## What is in this folder
- `src/` — recogniser, features, smoothing, hand model loader. `notebooks/` — 01 tracking check, 02 record Idle clips,
  03 train classifier, 04 legacy single-hand test, 05 multi-hand demo. `scripts/` — dataset download/extract, word
  screening and evaluation, interface check.
- `models/` — `hand_landmarker.task` (MediaPipe) and `sign_classifier.joblib` (the trained 14-word model).
- `data/raw/` — training features (dataset clips + Idle clips). Retrain with notebook 03.

## Not included (too large; see README to rebuild)
`venv/` (recreate with the steps above), the downloaded dataset videos `data/external/` (~5.5 GB), and older model
backups. The model and training CSVs above are enough to run and retrain.
