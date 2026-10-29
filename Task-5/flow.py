"""
Task 5 - game flow around Task 4's rules.

Task 4's GameController decides who scored, lost a heart or won. This class
adds what the controller leaves to the integrator: starting the first word,
pausing between rounds, and naming the current screen.

Screens (from Task 4's phase):  "lobby" -> "playing" -> "game_over" -> "lobby" ...
"""
from task4_game import GameController

NEXT_WORD_DELAY_MS = 1500     # pause between rounds so players see the result


class GameFlow:
    def __init__(self):
        self.game = GameController()
        self.next_word_at = None

    @property
    def screen(self):
        return {"waiting": "lobby"}.get(self.game.phase, self.game.phase)

    def update(self, t_ms, inputs, frame_id):
        """Advance the game one frame. Returns Task 4's events."""
        events = self.game.update(t_ms, inputs, frame_id)
        for e in events:
            if e["type"] == "game_start":
                self.game.next_word(t_ms)
                self.next_word_at = None
            elif e["type"] in ("score_event", "clash", "round_timeout"):
                self.next_word_at = t_ms + NEXT_WORD_DELAY_MS
            elif e["type"] == "game_over":
                self.next_word_at = None
        if (self.next_word_at is not None and t_ms >= self.next_word_at
                and self.game.phase == "playing"):
            self.game.next_word(t_ms)
            self.next_word_at = None
        return events

    def snapshot(self, t_ms):
        return self.game.snapshot(t_ms)

    def reset(self):
        self.game.reset()
        self.next_word_at = None
