"""
Task 5 - game sound: background music, effects and the announcer voice.

Event -> sound:
    game_start            voice "Are you ready?"   (3-2-1 countdown starts)
    round_start (first)   voice "Go!"
    score_event           coin
    wrong_sign            wrong-answer buzzer
    clash                 "fah" (both signed correctly at the same moment)
    level_up              voice "Level up!"
    game_over (winner)    victory sound, then voice "Winner!"

Music: one looping track per level (level 1 also plays in the lobby and on the
game-over screen); a level change crossfades to the next track. The music is
mixed quietly and dips further while the announcer channel (voices + victory
sound) is playing, so those stay clear.
Files live in assets/music, assets/sfx and assets/voice (see SOUNDS.md).
A missing file, or no audio device, just means that sound is silent.
"""
import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from bootstrap import ASSETS

MUSIC = {1: "music/game.mp3", 2: "music/level2.mp3", 3: "music/level3.mp3"}   # by level
SOUNDS = {
    "correct": "sfx/correct.wav",
    "wrong": "sfx/wrong.wav",
    "clash": "sfx/clash.wav",
    "victory": "sfx/victory.wav",
    "ready": "voice/voice_ready.wav",
    "go": "voice/voice_go.wav",
    "level_up": "voice/voice_level_up.wav",
    "winner": "voice/voice_winner.wav",
}

MUSIC_VOLUME = 0.30        # the track is loud and busy: keep it in the background
MUSIC_DUCKED = 0.10        # while the announcer speaks or the victory sound plays
DUCK_SPEED = 0.15          # fraction of the gap closed per frame (smooth fade)
FADE_STEP = 0.08           # crossfade between tracks: ~0.4 s out, ~0.4 s in at 30 fps
SFX_VOLUME = 0.85
VOICE_VOLUME = 1.0


class Audio:
    def __init__(self, enabled=True):
        self.ok = False
        self.muted = False
        self.sounds = {}
        self._music_vol = MUSIC_VOLUME
        self._track = None          # level whose track is loaded
        self._fade = 1.0            # crossfade multiplier, 0..1
        if not enabled:
            return
        try:
            import pygame
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        except Exception:
            return                                   # no audio device: play silently
        self.mixer = pygame.mixer
        self.mixer.set_num_channels(16)
        self.mixer.set_reserved(1)                   # channel 0 = announcer only
        self.announcer = self.mixer.Channel(0)
        for name, rel in SOUNDS.items():
            path = os.path.join(ASSETS, rel)
            if os.path.exists(path):
                self.sounds[name] = self.mixer.Sound(path)
        for name in ("correct", "wrong", "clash"):
            if name in self.sounds:
                self.sounds[name].set_volume(SFX_VOLUME)
        for name in ("ready", "go", "level_up", "winner", "victory"):
            if name in self.sounds:
                self.sounds[name].set_volume(VOICE_VOLUME)
        self.music = {lvl: os.path.join(ASSETS, rel) for lvl, rel in MUSIC.items()
                      if os.path.exists(os.path.join(ASSETS, rel))}
        self.ok = True
        self._play_track(1)

    def _effect(self, name):
        if name in self.sounds:
            self.sounds[name].play()                 # any free channel, may overlap

    def _announce(self, *names):
        """Play names one after another on the announcer channel, cutting off
        whatever it was saying (the newest news matters most)."""
        names = [n for n in names if n in self.sounds]
        if not names:
            return
        self.announcer.play(self.sounds[names[0]])
        for n in names[1:]:
            self.announcer.queue(self.sounds[n])

    def on_events(self, events):
        if not self.ok or self.muted:
            return
        for e in events:
            k = e["type"]
            if k == "game_start":
                self._announce("ready")
            elif k == "round_start" and e["first"]:
                self._announce("go")
            elif k == "score_event":
                self._effect("correct")
            elif k == "wrong_sign":
                self._effect("wrong")
            elif k == "clash":
                self._effect("clash")
            elif k == "level_up":
                self._announce("level_up")
            elif k == "game_over" and e["winner"] is not None:
                self._announce("victory", "winner")

    def _play_track(self, level):
        level = level if level in self.music else 1
        self._track = level
        if level in self.music:
            self.mixer.music.load(self.music[level])
            self.mixer.music.play(loops=-1)

    def update(self, snap):
        """Call once per frame: picks the track for the current level,
        crossfades when it changes and dips the music under the announcer."""
        if not self.ok or self.muted:
            return
        wanted = snap["level"] if snap["phase"] == "playing" else 1
        wanted = wanted if wanted in self.music else 1
        if wanted != self._track:
            self._fade = max(0.0, self._fade - FADE_STEP)
            if self._fade == 0.0:
                self._play_track(wanted)
        else:
            self._fade = min(1.0, self._fade + FADE_STEP)
        target = MUSIC_DUCKED if self.announcer.get_busy() else MUSIC_VOLUME
        self._music_vol += (target - self._music_vol) * DUCK_SPEED
        self.mixer.music.set_volume(self._music_vol * self._fade)

    def toggle_mute(self):
        if not self.ok:
            return
        self.muted = not self.muted
        if self.muted:
            self.mixer.stop()
            self.mixer.music.pause()
        else:
            self.mixer.music.unpause()

    def reset(self):
        if self.ok:
            self.announcer.stop()

    def close(self):
        if self.ok:
            self.mixer.quit()
