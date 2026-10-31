"""
Task 5 - person segmentation: which pixels are people, which are the room.

MediaPipe Selfie Segmenter (landscape model, 144x256 input, ~2 ms) gives a
person probability per pixel. Before it is used as a blend mask it is
cleaned up, at a quarter of the frame size for speed:

  1. temporal smoothing  m = a * new + (1 - a) * previous
                         removes flicker at the outline between frames
  2. contrast curve      smoothstep between LOW and HIGH: confident pixels
                         become fully 0 or 1, only the edge stays soft
  3. feathering          Gaussian blur, so the outline blends into the arena
                         instead of looking cut out with scissors
  4. upscale             bilinear back to the frame size

Returns a float32 HxW mask in 0..1 (1 = person).
"""
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import bootstrap

WORK_SCALE = 0.25          # clean the mask at 1/4 size (320x180 for 720p)
TEMPORAL_ALPHA = 0.6       # weight of the new mask (higher = less lag, more flicker)
LOW, HIGH = 0.45, 0.85     # below LOW -> room, above HIGH -> person (biased inwards: no halo)
FEATHER_PX = 1.2           # Gaussian sigma at work size (~5 px at full size)


class PersonSegmenter:
    def __init__(self, model_path=bootstrap.SEGMENTER_MODEL):
        options = vision.ImageSegmenterOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            output_confidence_masks=True,
            output_category_mask=False,
        )
        self._seg = vision.ImageSegmenter.create_from_options(options)
        self._prev = None
        self._last_ms = -1

    def mask(self, frame, t_s):
        """frame: BGR (already mirrored), t_s: seconds. Returns HxW float32 0..1."""
        H, W = frame.shape[:2]
        ts = max(int(t_s * 1000), self._last_ms + 1)      # VIDEO mode needs rising timestamps
        self._last_ms = ts
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._seg.segment_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        raw = result.confidence_masks[0].numpy_view().reshape(H, W)

        w, h = max(1, int(W * WORK_SCALE)), max(1, int(H * WORK_SCALE))
        m = cv2.resize(raw, (w, h), interpolation=cv2.INTER_AREA)
        if self._prev is not None and self._prev.shape == m.shape:
            m = TEMPORAL_ALPHA * m + (1 - TEMPORAL_ALPHA) * self._prev
        self._prev = m
        u = np.clip((m - LOW) / (HIGH - LOW), 0, 1)
        u = u * u * (3 - 2 * u)                                  # smoothstep
        u = cv2.GaussianBlur(u, (0, 0), FEATHER_PX)
        return cv2.resize(u, (W, H), interpolation=cv2.INTER_LINEAR)

    def close(self):
        self._seg.close()
