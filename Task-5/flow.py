"""
Task 5 - game flow around Task 4's rules.

Task 4's GameController decides who scored, lost a heart or won. This class
adds what the controller leaves to the integrator: a 3-2-1 countdown before
the first word, the pause between rounds, a few seconds of celebration before
a rematch can start, and naming the current screen.

Screens (from Task 4's phase):  "lobby" -> "playing" -> "game_over" -> "lobby" ...

Besides Task 4's events, update() emits one of its own:
    {"type": "round_start", "word", "first"}   a new target word appeared
"""
from task4_game import GameController, PlayerInput

COUNTDOWN_MS = 3000           # 3-2-1 before the first word of a game
NEXT_WORD_DELAY_MS = 1500     # pause between rounds so players see the result
GAME_OVER_HOLD_MS = 4000      # celebration time before an open palm can start a rematch


class GameFlow:
    def __init__(self):
        self.game = GameController()
        self.next_word_at = None
        self.countdown_end = None
        self.game_over_at = None
        self._first_round = False

    @property
    def screen(self):
        return {"waiting": "lobby"}.get(self.game.phase, self.game.phase)

    def update(self, t_ms, inputs, frame_id):
        """Advance the game one frame. Returns Task 4's events (+ round_start)."""
        if self._rematch_blocked(t_ms):
            # nobody counts as present, so held palms cannot start a rematch yet
            inputs = {pid: PlayerInput(present=False) for pid in inputs}
        events = self.game.update(t_ms, inputs, frame_id)
        for e in events:
            if e["type"] == "game_start":
                self.countdown_end = self.next_word_at = t_ms + COUNTDOWN_MS
                self._first_round = True
            elif e["type"] in ("score_event", "clash", "round_timeout"):
                self.next_word_at = t_ms + NEXT_WORD_DELAY_MS
            elif e["type"] == "game_over":
                self.next_word_at = self.countdown_end = None
                self.game_over_at = t_ms
        if (self.next_word_at is not None and t_ms >= self.next_word_at
                and self.game.phase == "playing"):
            self.game.next_word(t_ms)
            events.append({"type": "round_start", "frame_id": frame_id, "timestamp_ms": t_ms,
                           "word": self.game.target["word"], "first": self._first_round})
            self.next_word_at = self.countdown_end = None
            self._first_round = False
        return events

    def _rematch_blocked(self, t_ms):
        return (self.game.phase == "game_over" and self.game_over_at is not None
                and t_ms < self.game_over_at + GAME_OVER_HOLD_MS)

    def snapshot(self, t_ms):
        """Task 4's snapshot plus "countdown_ms" (time left before the first word)
        and "rematch_open" (an open palm can start the next game)."""
        snap = self.game.snapshot(t_ms)
        snap["countdown_ms"] = (max(0, self.countdown_end - t_ms)
                                if self.countdown_end is not None else 0)
        snap["rematch_open"] = not self._rematch_blocked(t_ms)
        return snap

    def reset(self):
        self.game.reset()
        self.next_word_at = self.countdown_end = self.game_over_at = None
        self._first_round = False
