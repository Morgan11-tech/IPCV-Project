"""
Task 5 - one place for every colour, name and size the game draws with.

Each player owns ONE colour, used everywhere: their half of the screen, face
ring, name tag, score panel, hearts and effects. Blue and orange stay easy to
tell apart for colour-blind players.

Layout rule: the view is mirrored, so Player 1 plays on the LEFT half and
Player 2 on the RIGHT half; each player's panel sits in their own half.
All colours are BGR (OpenCV order).
"""

PLAYER_IDS = (1, 2)
PLAYER_NAMES = {1: "PLAYER 1", 2: "PLAYER 2"}
PLAYER_COLORS = {1: (225, 140, 30),     # blue
                 2: (30, 140, 255)}     # orange
NEUTRAL = (170, 170, 170)               # unknown face / unassigned hand

# UI palette
WHITE = (255, 255, 255)
OUTLINE = (20, 20, 20)                  # text outline for readability on video
PANEL = (45, 32, 28)                    # dark panel fill
TRACK = (80, 80, 80)                    # empty part of bars
EMPTY_HEART = (95, 95, 95)
GOOD = (90, 200, 80)                    # correct / ready
BAD = (70, 70, 235)                     # wrong / locked out / time running out
ACCENT = (40, 210, 255)                 # highlights (target word, streaks)
LEVEL_COLORS = {1: GOOD, 2: ACCENT, 3: BAD}

HALF_STRIP = 8                          # px, colour strip along each player's half
PANEL_ALPHA = 0.78


def player_color(pid):
    return PLAYER_COLORS.get(pid, NEUTRAL)


def half_bounds(pid, W):
    """(x0, x1) of a player's half of the screen."""
    return (0, W // 2) if pid == 1 else (W // 2, W)


def scaled(color, factor):
    """Darker (factor < 1) or lighter (factor > 1) version of a colour."""
    return tuple(int(min(255, c * factor)) for c in color)
