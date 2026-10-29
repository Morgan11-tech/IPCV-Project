"""
Task 5 - game sound: background music, effects and the announcer voice.

Event -> sound:
    game_start            voice "Are you ready?"   (3-2-1 countdown starts)
    round_start (first)   voice "Go!"
    score_event           coin
    wrong_sign            wrong-answer buzzer
    level_up              voice "Level up!"
    game_over (winner)    victory sound, then voice "Winner!"

The music loops the whole time. It is mixed quietly and dips further while the
announcer channel (voices + victory sound) is playing, so those stay clear.
Files live in assets/music, assets/sfx and assets/voice (see SOUNDS.md).
A missing file, or no audio device, just means that sound is silent.
"""
import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from bootstrap import ASSETS

MUSIC = os.path.join(ASSETS, "music", "game.mp3")
SOUNDS = {
    "correct": "sfx/correct.wav",
    "wrong": "sfx/wrong.wav",
    "victory": "sfx/victory.wav",
    "ready": "voice/voice_ready.wav",
    "go": "voice/voice_go.wav",
    "level_up": "voice/voice_level_up.wav",
    "winner": "voice/voice_winner.wav",
}

MUSIC_VOLUME = 0.30        # the track is loud and busy: keep it in the background
MUSIC_DUCKED = 0.10        # while the announcer speaks or the victory sound plays
DUCK_SPEED = 0.15          # fraction of the gap closed per frame (smooth fade)
SFX_VOLUME = 0.85
VOICE_VOLUME = 1.0


class Audio:
    def __init__(self, enabled=True):
        self.ok = False
        self.muted = False
        self.sounds = {}
        self._music_vol = MUSIC_VOLUME
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
        for name in ("correct", "wrong"):
            if name in self.sounds:
                self.sounds[name].set_volume(SFX_VOLUME)
        for name in ("ready", "go", "level_up", "winner", "victory"):
            if name in self.sounds:
                self.sounds[name].set_volume(VOICE_VOLUME)
        if os.path.exists(MUSIC):
            self.mixer.music.load(MUSIC)
            self.mixer.music.set_volume(MUSIC_VOLUME)
            self.mixer.music.play(loops=-1)
        self.ok = True

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
            elif k == "level_up":
                self._announce("level_up")
            elif k == "game_over" and e["winner"] is not None:
                self._announce("victory", "winner")

    def update(self):
        """Call once per frame: fades the music down under the announcer."""
        if not self.ok:
            return
        target = MUSIC_DUCKED if self.announcer.get_busy() else MUSIC_VOLUME
        self._music_vol += (target - self._music_vol) * DUCK_SPEED
        self.mixer.music.set_volume(self._music_vol)

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
