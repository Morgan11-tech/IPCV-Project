"""
Task 5 - match statistics for the game-over screen.

Counted from Task 4's events, per player and per game:
    correct      signs that scored
    wrong        wrong signs
    fastest_ms   quickest correct answer (time since the word appeared)
    avg_ms       average time of the correct answers
    best_streak  most correct answers in a row
Plus games won by each player since the program started (or since R).
"""
from dataclasses import dataclass, field


@dataclass
class PlayerStats:
    correct: int = 0
    wrong: int = 0
    best_streak: int = 0
    times_ms: list = field(default_factory=list)

    @property
    def fastest_ms(self):
        return min(self.times_ms) if self.times_ms else None

    @property
    def avg_ms(self):
        return sum(self.times_ms) / len(self.times_ms) if self.times_ms else None


class MatchStats:
    def __init__(self, player_ids=(1, 2)):
        self.ids = player_ids
        self.games_won = {pid: 0 for pid in player_ids}
        self.draws = 0
        self.new_game()

    def new_game(self):
        self.players = {pid: PlayerStats() for pid in self.ids}

    def observe(self, events, snap):
        target = snap.get("target") or {}
        for e in events:
            k = e["type"]
            if k == "game_start":
                self.new_game()
            elif k == "score_event":
                s = self.players[e["player_id"]]
                s.correct += 1
                s.best_streak = max(s.best_streak, e.get("streak", 0))
                if "spawned_ms" in target:
                    s.times_ms.append(max(0, e["timestamp_ms"] - target["spawned_ms"]))
            elif k == "wrong_sign":
                self.players[e["player_id"]].wrong += 1
            elif k == "game_over":
                if e["winner"] is None:
                    self.draws += 1
                else:
                    self.games_won[e["winner"]] += 1

    def reset(self):
        self.__init__(self.ids)
