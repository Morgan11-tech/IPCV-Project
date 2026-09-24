# Task 1 — Face Tracking and Augmented Effects

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
