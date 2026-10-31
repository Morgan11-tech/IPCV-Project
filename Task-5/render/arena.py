"""
The rendered game environment behind the players: the pixel-art landscape
from pixel_scene.py, one look per scene:

    lobby      sunset in the logo's colours (continues the title screen)
    level 1    calm daytime
    level 2    golden dusk
    level 3    red night
    game over  night sky fading into the winner's colour

Higher levels move faster (clouds, shimmer). A scene change crossfades over
SCENE_FADE_MS. The canvas is 1/PIXEL of the window and scaled up with
nearest-neighbour interpolation, so the arena keeps hard pixels.
"""
import cv2

from render.pixel_scene import PIXEL, SPEED, PixelScene, palette_for

SCENE_FADE_MS = 700


def scene_of(snap):
    if snap["phase"] == "playing":
        return snap["level"]
    if snap["phase"] == "game_over":
        return "game_over"
    return "lobby"


class Arena:
    def __init__(self):
        self._scenes = {}               # (scene, winner) -> PixelScene
        self._key, self._prev, self._t_change = None, None, 0

    def _scene(self, key):
        if key not in self._scenes:
            scene, winner = key
            self._scenes[key] = PixelScene(palette_for(scene, winner), SPEED.get(scene, 1.0))
        return self._scenes[key]

    def render(self, snap, t, W, H):
        """Full-size uint8 arena image for this frame."""
        scene = scene_of(snap)
        key = (scene, snap.get("winner") if scene == "game_over" else None)
        if key != self._key:
            self._prev, self._key, self._t_change = self._key, key, t
        w, h = W // PIXEL, H // PIXEL
        img = self._scene(key).render(w, h, t)
        u = (t - self._t_change) / SCENE_FADE_MS
        if self._prev is not None and u < 1:                  # crossfade from the last scene
            img = cv2.addWeighted(self._scene(self._prev).render(w, h, t), 1 - u, img, u, 0)
        return cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
