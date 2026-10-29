"""
Task 5 renderer: everything that is drawn on screen goes through here.

Draw order (back to front):
    camera frame -> player halves -> face rings + name tags (Task 1's effect,
    in the player's colour) -> event effects -> HUD -> winner celebration
"""
import effects as face_effects          # Task 1

from render.fx import EffectsLayer, anchors_from_inputs
from render.hud import draw_hud
from render.winner import WinnerDancer


class Renderer:
    def __init__(self):
        self.fx = EffectsLayer()
        self.winner = WinnerDancer()

    def on_events(self, events, t):
        self.fx.on_events(events, t)
        for e in events:
            if e["type"] == "game_start":
                self.winner.stop()
            elif e["type"] == "game_over" and e["winner"] is not None:
                self.winner.start(t, e["winner"], score=e["scores"][e["winner"]])

    def reset(self):
        self.fx.clear()
        self.winner.stop()

    def draw(self, frame, perception, snap, t):
        """Return a new image: the camera frame with the whole game drawn on it."""
        display = frame.copy()
        for player in perception.players.values():
            face_effects.draw_face_effect(display, player)
            face_effects.draw_player_tag(display, player)
        self.fx.draw(display, t, anchors_from_inputs(perception.inputs))
        draw_hud(display, snap, perception.inputs, t)
        if snap["phase"] == "game_over":
            self.winner.draw(display, t)
        return display
