"""Feature extraction shared across data collection, training, and live
recognition notebooks, so all three use identical math.
"""
import numpy as np

# MediaPipe landmarks are fractions of the image width and height, so the same
# hand looks horizontally squashed on a wider image (16:9 webcam vs the 4:3
# 640x480 training videos). extract_frame() undoes this by stretching x and z
# by aspect / REFERENCE_ASPECT; 4:3 input is left exactly unchanged.
REFERENCE_ASPECT = 4.0 / 3.0


def normalize_landmarks(hand_landmarks):
    """Convert a HandLandmarker hand's 21 landmarks to a flat, translation-
    and scale-invariant 63-length feature vector (21 landmarks x 3 coords).

    `hand_landmarks` is one entry from `result.hand_landmarks` (a list of
    21 landmark objects with .x, .y, .z).
    """
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])

    # Translate: wrist (landmark 0) becomes the origin
    coords -= coords[0]

    # Scale: normalize by the distance from wrist to middle-finger MCP (landmark 9)
    scale = np.linalg.norm(coords[9])
    if scale > 1e-6:
        coords /= scale

    return coords.flatten()  # shape (63,)


def resample_sequence(frames, timestamps, target_len=20):
    """Resample a variable-length sequence of per-frame feature vectors to a
    fixed number of time steps via linear interpolation, so a gesture
    performed quickly or slowly produces a same-size feature vector.

    frames: list/array of shape (N, F) per-frame feature vectors (e.g. the
        63-dim output of `normalize_landmarks`, one per frame in a clip).
    timestamps: list/array of N timestamps in seconds, increasing, aligned
        with `frames` (e.g. seconds since the clip/recording started).
    target_len: number of evenly spaced output time steps.

    Returns a flat (target_len * F,) array, or None if there isn't enough
    valid data to resample (e.g. the hand was lost for nearly the whole clip).
    """
    frames = np.asarray(frames, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)

    if len(frames) < 2:
        return None
    if timestamps[-1] - timestamps[0] < 1e-6:
        return None

    target_times = np.linspace(timestamps[0], timestamps[-1], target_len)
    resampled = np.zeros((target_len, frames.shape[1]))
    for i in range(frames.shape[1]):
        resampled[:, i] = np.interp(target_times, timestamps, frames[:, i])

    return resampled.flatten()


def extract_frame(hand_landmarks, aspect=REFERENCE_ASPECT):
    """Extract one frame's combined shape + raw-position features for
    CLIP-based (trajectory + shape) gesture recognition.

    Unlike `normalize_landmarks`, this keeps the wrist's RAW (x, y, z)
    position instead of zeroing it out — trajectory normalization (relative
    to where the clip started) happens later, once the whole clip is known,
    in `build_clip_feature`. Without this, whole-hand-movement signs (a wave,
    a hand moving outward, etc.) would be indistinguishable from a hand held
    still, since per-frame-only normalization discards exactly that motion.

    Returns:
        raw63: (63,) array = [wrist_x, wrist_y, wrist_z, shape_60...], where
            shape_60 is translation+scale invariant (relative to this frame's
            own wrist and hand size) — same idea as `normalize_landmarks`,
            just without landmark 0 (always zero after centering, so it's
            dropped to make room for the raw wrist position instead).
        hand_scale: float, this frame's wrist-to-middle-MCP distance. Used
            later to normalize the wrist trajectory by hand size / distance
            from the camera.

    `aspect` is width / height of the image the landmarks came from. Pass the
    real value (e.g. 1920 / 1080) so hand geometry is comparable across
    cameras; the default 4:3 matches the training videos.
    """
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
    k = aspect / REFERENCE_ASPECT
    coords[:, 0] *= k       # x and z are both normalised by image width
    coords[:, 2] *= k
    wrist = coords[0].copy()
    hand_scale = float(np.linalg.norm(coords[9] - coords[0]))

    shape = coords[1:] - coords[0]
    if hand_scale > 1e-6:
        shape = shape / hand_scale

    raw63 = np.concatenate([wrist, shape.flatten()])
    return raw63, hand_scale


def mirror_raw63(raw63):
    """Mirror a per-frame feature vector from `extract_frame` left <-> right
    (negate every x component: the wrist x and each landmark's x).

    A left hand performing a sign is the mirror image of a right hand
    performing it. The training videos are mostly right-handed signers, so
    left hands are mirrored into right-hand form before classification
    (and left-handed training signers likewise)."""
    out = np.array(raw63, dtype=float, copy=True)
    out[0::3] *= -1.0
    return out


def build_clip_feature(raw63_list, hand_scales, timestamps, target_len=20):
    """Combine a variable-length clip of `extract_frame` outputs into one
    fixed-length feature vector capturing BOTH hand shape over time AND
    hand trajectory (wrist movement) over time — relative to where the clip
    started (so a gesture works regardless of where in the frame it's
    performed) and scaled by hand size (so it works regardless of distance
    from the camera).

    Returns a flat (target_len * 63,) array, or None if there isn't enough
    valid data to build it.
    """
    resampled_flat = resample_sequence(raw63_list, timestamps, target_len)
    if resampled_flat is None:
        return None
    resampled = resampled_flat.reshape(target_len, 63)

    scale_estimate = float(np.median(hand_scales)) if len(hand_scales) else 1.0
    scale_estimate = max(scale_estimate, 1e-6)

    trajectory = resampled[:, 0:3]
    trajectory = (trajectory - trajectory[0]) / scale_estimate

    shape = resampled[:, 3:63]
    return np.concatenate([trajectory, shape], axis=1).flatten()


def motion_energy(frames):
    """Rough measure of how much a sequence of per-frame feature vectors is
    moving: the mean frame-to-frame L2 distance. Used to gate when to even
    attempt gesture classification during live recognition (so a resting
    hand doesn't get fed into the classifier as if it were mid-gesture).
    """
    frames = np.asarray(frames, dtype=float)
    if len(frames) < 2:
        return 0.0
    diffs = np.diff(frames, axis=0)
    return float(np.mean(np.linalg.norm(diffs, axis=1)))
