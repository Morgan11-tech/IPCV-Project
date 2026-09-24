"""
All tunable parameters for Task 1 in one place.

Every value below is a STARTING POINT. The recognition threshold and the
smoothing/grace values in particular must be validated empirically with our
own webcam, room and players (see README -> Evaluation procedure).
"""

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0            # 0 = built-in webcam
FRAME_WIDTH = 1280          # requested capture size (camera may pick nearest)
FRAME_HEIGHT = 720
MIRROR_VIEW = True          # flip horizontally so players see a "mirror"

# ---------------------------------------------------------------------------
# Models (downloaded once, see README -> Installation)
# ---------------------------------------------------------------------------
DETECTOR_MODEL = "models/face_detection_yunet_2023mar.onnx"
RECOGNIZER_MODEL = "models/face_recognition_sface_2021dec.onnx"

# Face detection runs on a downscaled copy of the frame. At 640 px width a face
# 1.5-2 m from a laptop webcam is still ~60+ px wide, which YuNet handles well,
# while detection cost drops with the number of pixels (measured on M1 Pro:
# 6.0 ms at 640 px vs 15.5 ms at 1280 px). Boxes are scaled back afterwards.
# Embeddings are still computed from the full-resolution frame.
DETECTION_WIDTH = 640

# YuNet confidence (0..1) needed to accept a detection in live mode.
# Lower than the registration value so that partially occluded or turned faces
# (which get lower scores) are still found; false positives are filtered later
# because they will not match either player's embedding.
DETECTION_SCORE_THRESHOLD = 0.7
DETECTION_NMS_THRESHOLD = 0.3   # overlap above which duplicate boxes are merged

# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
NUM_REGISTRATION_SAMPLES = 7
REGISTRATION_MIN_SCORE = 0.9    # only clear, confident, frontal-ish samples
REGISTRATION_MIN_FACE_WIDTH = 80  # px; tiny faces give poor embeddings
PLAYER_DATA_DIR = "player_data"

# ---------------------------------------------------------------------------
# Recognition
# ---------------------------------------------------------------------------
# Cosine-similarity threshold for SFace (face vs. averaged player reference).
# Starting point: 0.363, the value published by the OpenCV model authors for
# comparing two single images (tuned on the LFW benchmark).
# Our offline test (13 different people from a group photo, see README) found
# impostors scoring 0.38-0.40 against averaged references, which 0.363 would
# wrongly accept, while genuine matches scored ~0.87-0.95. We therefore use
# 0.45. A genuine player whose similarity dips below this (head turned, hand
# over face) keeps their identity through the IoU track continuation below;
# the threshold mainly controls who can be ACQUIRED as a player.
# This is NOT universally optimal: re-check it with our own webcam using the
# similarities printed in the prepare phase and logged in evaluation trials.
SIMILARITY_THRESHOLD = 0.45

# Run the embedding network every N frames (1 = every frame). On frames in
# between, faces keep their identity through box overlap with their track.
# Measured on an M1 Pro (CPU): detection 6 ms + 8 ms per face embedding, so
# two players cost ~22 ms/frame -> N = 1 still keeps up with a 30 FPS webcam.
RECOGNITION_INTERVAL = 1

# ---------------------------------------------------------------------------
# Tracking and smoothing
# ---------------------------------------------------------------------------
# EMA weight of the NEW measurement: smoothed = a*current + (1-a)*previous.
# Small alpha -> smoother but more lag. Large alpha -> responsive but jittery.
EMA_ALPHA = 0.3

# Grace period: a player not detected for up to this many frames keeps their
# last smoothed position (effect is "held"). After that the player is LOST
# and the effect is hidden. 10 frames is ~0.4 s at 25 FPS: long enough to
# bridge detector drop-outs or a hand passing the face, short enough that an
# effect does not float in empty space after the player has really left.
MAX_MISSING_FRAMES = 10

# A face that was NOT matched by embedding (e.g. head turned so similarity
# dropped) still continues a player's track if its box overlaps that player's
# last box by at least this Intersection-over-Union.
TRACK_IOU_THRESHOLD = 0.3

# If a new detection is further than this many face-widths from the smoothed
# position, it is a re-acquisition (not jitter) and the EMA is reset, so the
# effect jumps to the face instead of sliding across the screen.
SMOOTHING_RESET_DISTANCE = 1.0

# ---------------------------------------------------------------------------
# Effects / game state display
# ---------------------------------------------------------------------------
PLAYER_NAMES = {1: "PLAYER 1", 2: "PLAYER 2"}
PLAYER_COLORS = {1: (255, 140, 0), 2: (40, 40, 230)}   # BGR: blue, red
UNKNOWN_COLOR = (160, 160, 160)
FEEDBACK_DURATION = 1.5     # seconds the "+1 CORRECT!" popup is shown

DEBUG_MODE = True           # toggle at runtime with the D key

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
EVAL_DIR = "eval_logs"
EVAL_SAVE_VIDEO = True      # also save the annotated video of every trial
# A player's raw face centre jumping by more than this many face widths
# between two observations inside the grace window is flagged as a POSSIBLE
# identity switch (a real head cannot move that far in a few frames).
# Flags must be confirmed by watching the saved trial video.
SWITCH_JUMP_FACE_WIDTHS = 1.5
