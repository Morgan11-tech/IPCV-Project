"""
Task 5 renderer: everything that is drawn on screen goes through here.

Draw order (back to front):
    camera frame, colour-graded for the current scene (backdrop.py)
    -> face rings + name tags in the player's colour (players.py)
    -> particles (particles.py) -> event effects (fx.py) -> HUD (hud.py)
    -> winner celebration (winner.py)
    -> whole-screen flash / shake (screen_fx.py)

Game feel per event:
    correct sign   sparks + ring + "+10" in the scorer's colour, faint flash
                   in their colour, score number bumps
    wrong sign     "WRONG" wobbles, red flash, small shake
    time's up      banner, red flash, bigger shake, lost hearts swell and fade
    level up       colour band slides across, flash in the level's colour
    game over      confetti rain + "<NAME> WINS!" in the winner's colour
Each of these also gets a feedback line ("Lightning hands!", "So close!") from
feedback.py, shown under the effect.
"""
import theme
from feedback import Feedback

from render.backdrop import Backdrop
from render.fx import EffectsLayer, anchors_from_inputs
from render.hud import HudAnim, draw_hud
from render.particles import Particles, confetti_colors, spark_colors
from render.players import draw_players
from render.screen_fx import ScreenFx
from render.winner import WinnerScreen


class Renderer:
    def __init__(self):
        self.backdrop = Backdrop()
        self.fx = EffectsLayer()
        self.hud_anim = HudAnim()
        self.particles = Particles()
        self.screen = ScreenFx()
        self.winner = WinnerScreen()
        self.feedback = Feedback()

    def reset(self):
        self.fx.clear()
        self.hud_anim.clear()
        self.particles.clear()
        self.screen.clear()
        self.winner.stop()

    def _on_events(self, events, snap, anchors, t):
        lines = self.feedback.for_events(events, snap)
        self.fx.on_events(events, t, lines)
        for e, line in zip(events, lines):
            k = e["type"]
            if k == "score_event":
                pid = e["player_id"]
                if pid in anchors:
                    self.particles.burst(anchors[pid]["center"], spark_colors(pid), n=50)
                self.screen.flash(theme.player_color(pid), t, 300, 0.18)
            elif k == "wrong_sign":
                self.screen.flash(theme.BAD, t, 250, 0.22)
                self.screen.shake(t, 250, 6)
            elif k == "round_timeout":
                self.screen.flash(theme.BAD, t, 400, 0.3)
                self.screen.shake(t, 400, 12)
            elif k == "level_up":
                self.screen.flash(theme.LEVEL_COLORS.get(e["level"], theme.ACCENT), t, 400, 0.25)
            elif k == "game_start":
                self.winner.stop()
            elif k == "game_over" and e["winner"] is not None:
                self.winner.start(t, e["winner"], score=e["scores"][e["winner"]], line=line)

    def draw(self, frame, perception, snap, events, t):
        """Return a new image: the camera frame with the whole game drawn on it."""
        anchors = anchors_from_inputs(perception.inputs)
        self._on_events(events, snap, anchors, t)
        self.hud_anim.observe(snap, events, t)
        if snap["phase"] == "game_over":
            self.particles.confetti(frame.shape[1], confetti_colors(snap["winner"]), n=3)
        self.particles.update(t)

        display = self.backdrop.apply(frame, snap, t)
        draw_players(display, perception.players)
        self.particles.draw(display)
        self.fx.draw(display, t, anchors)
        draw_hud(display, snap, perception.inputs, t, self.hud_anim)
        if snap["phase"] == "game_over":
            self.winner.draw(display, t)
        return self.screen.apply(display, t)
