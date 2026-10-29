"""
Game-over celebration: a dancing stick figure in the winner's colour with a
ring of confetti (ported from Task 4's WinnerDancer).
"""
import cv2
import numpy as np

import theme

from render.text import draw as text

CONFETTI = (theme.ACCENT, theme.GOOD, theme.WHITE)


class WinnerDancer:
    def __init__(self):
        self.start_t, self.winner, self.score = None, None, 0

    def start(self, t, winner_id, score=0):
        self.start_t, self.winner, self.score = t, winner_id, score

    def stop(self):
        self.start_t, self.winner = None, None

    def draw(self, frame, t):
        if self.start_t is None:
            return
        H, W = frame.shape[:2]
        c = theme.player_color(self.winner)
        u = (t - self.start_t) / 1000.0
        cx = W // 2 + int(10 * np.sin(u * 3.0))
        cy = int(H * 0.55) + int(12 * np.sin(u * 6.0))

        cv2.line(frame, (cx, cy), (cx, cy + 70), c, 6, cv2.LINE_AA)            # body
        for side, phase in ((-1, 0.0), (1, np.pi)):                            # legs
            hip = np.array([cx, cy + 70])
            knee = hip + np.array([side * (28 + 10 * np.sin(u * 6 + phase)),
                                   45 - 15 * np.sin(u * 6 + phase)])
            foot = knee + np.array([side * 20, 35])
            cv2.line(frame, tuple(hip.astype(int)), tuple(knee.astype(int)), c, 5, cv2.LINE_AA)
            cv2.line(frame, tuple(knee.astype(int)), tuple(foot.astype(int)), c, 5, cv2.LINE_AA)
        for side, phase in ((-1, 0.0), (1, np.pi)):                            # arms
            shoulder = np.array([cx, cy + 5])
            elbow = shoulder + np.array([side * (35 + 15 * np.sin(u * 6 + phase)),
                                         -20 + 20 * np.cos(u * 6 + phase)])
            hand = elbow + np.array([side * (25 + 10 * np.sin(u * 6 + phase)),
                                     -25 + 15 * np.cos(u * 6 + phase)])
            cv2.line(frame, tuple(shoulder.astype(int)), tuple(elbow.astype(int)), c, 5, cv2.LINE_AA)
            cv2.line(frame, tuple(elbow.astype(int)), tuple(hand.astype(int)), c, 5, cv2.LINE_AA)
        tilt = int(8 * np.sin(u * 4.0))                                        # head
        cv2.circle(frame, (cx + tilt, cy - 20), 22, c, -1, cv2.LINE_AA)
        for dx in (-7, 7):
            cv2.circle(frame, (cx + tilt + dx, cy - 25), 3, theme.OUTLINE, -1)
        cv2.ellipse(frame, (cx + tilt, cy - 14), (9, 5), 0, 0, 180, theme.OUTLINE, 2, cv2.LINE_AA)

        for i in range(14):                                                    # confetti
            ang = u * 2.0 + i * (2 * np.pi / 14)
            rad = 90 + 20 * np.sin(u * 5 + i)
            px, py = int(cx + rad * np.cos(ang)), int(cy + rad * np.sin(ang))
            col = c if i % 2 == 0 else CONFETTI[(i // 2) % len(CONFETTI)]
            cv2.circle(frame, (px, py), 5, col, -1, cv2.LINE_AA)

        text(frame, f"{theme.PLAYER_NAMES[self.winner]} WINS!", (W // 2, cy - 110), 56, c,
             align="center")
        text(frame, f"Final score: {self.score}", (W // 2, cy - 72), 30, theme.WHITE,
             font="body", align="center")
