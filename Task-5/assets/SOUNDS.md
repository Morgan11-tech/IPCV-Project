# Game sounds: what we need and where it goes

## In the game now (`audio.py`)

| File | Plays on |
|---|---|
| `music/game.mp3` | the whole time, looped, quiet; dips under voices and the victory sound |
| `voice/voice_ready.wav` | game starts ("Are you ready?", during 3-2-1) |
| `voice/voice_go.wav` | first word of a game |
| `sfx/correct.wav` | correct sign |
| `sfx/wrong.wav` | wrong sign |
| `voice/voice_level_up.wav` | level up |
| `sfx/victory.wav` then `voice/voice_winner.wav` | someone wins |

Everything else in the tables below is still optional and silent until a file is added.

**How to add a sound:** drop the downloaded file (any format: `.mp3`, `.wav`, `.ogg`) into
`assets/incoming/` and rename it to the name in the tables below, keeping its own extension.
For example `incoming/correct.mp3`. Task 5 converts it to the final format, trims silence,
balances the volume and moves it into `music/`, `sfx/` or `voice/`.

Then add one line per file to `CREDITS.md`: site, tool or model, licence or plan, and the
prompt you used. Every AI-generated file also goes in the AI statement.

Final format (done for you during conversion):
- `sfx/` and `voice/`: WAV, 16-bit, 44.1 kHz, mono, short (small files, load instantly).
- `music/`: MP3, 44.1 kHz, stereo, loops seamlessly.

(The local ffmpeg has no good OGG encoder, so WAV + MP3 are used; pygame plays both.)

Style for everything: **arcade, bright, playful, not neon/synthwave.** Think party game or game
show, not cyberpunk.

## Music (4 files, looping in the background)

| # | File name | When it plays | Length | Prompt idea |
|---|---|---|---|---|
| M1 | `lobby` | lobby + waiting for players, game-over screen after the fanfare | 60–120 s loop | "upbeat playful arcade menu music, light percussion, marimba and synth bass, friendly, loopable, no vocals" |
| M2 | `round` | levels 1–2 | 60–120 s loop | "energetic arcade game music, bouncy drums, catchy lead, medium tempo 120 bpm, loopable, no vocals" |
| M3 | `round_fast` | level 3 (5-second rounds) | 60–120 s loop | "intense fast arcade game music, 150 bpm, driving drums, urgent but fun, loopable, no vocals" |
| M4 | `victory` | once when someone wins | 4–8 s, not looped | "short triumphant game victory fanfare, brass and drums, cheerful" |

## Sound effects (12 files, short)

| # | File name | When it plays | Length | Prompt idea |
|---|---|---|---|---|
| S1 | `ready` | a player holds the palm and becomes READY | 0.3–0.6 s | "bright positive UI confirm chime, two rising notes" |
| S2 | `countdown` | each of 3 · 2 · 1 | 0.2–0.4 s | "short arcade countdown beep" |
| S3 | `go` | GO! (first word appears) | 0.4–0.8 s | "arcade start horn / 'go' sting, punchy" |
| S4 | `new_word` | each new target word | 0.2–0.5 s | "quick swoosh with a soft pop, cartoon" |
| S5 | `correct` | correct sign (+10) | 0.4–0.8 s | "cheerful coin collect / correct answer ding, game show" |
| S6 | `streak` | correct with a streak of 3+ (replaces S5) | 0.6–1.0 s | "rising combo arpeggio, sparkly, excited" |
| S7 | `wrong` | wrong sign | 0.3–0.6 s | "cartoon wrong answer buzzer, playful not harsh" |
| S8 | `tick` | every second in the last 3 seconds | 0.1–0.2 s | "clock tick, crisp, single" |
| S9 | `timeout` | TIME'S UP | 0.5–1.0 s | "game show time's up buzzer, descending" |
| S10 | `heart_lost` | a heart is lost | 0.3–0.6 s | "cartoon deflate / sad descending blip" |
| S11 | `level_up` | LEVEL 2 / LEVEL 3 band | 0.8–1.5 s | "level up jingle, rising, energetic arcade" |
| S12 | `clash` | both correct at the same moment | 0.4–0.8 s | "two swords clash / cymbal hit, cartoon" |

## Voice lines (optional, 5 files)

| # | File name | Text to speak | Notes |
|---|---|---|---|
| V1 | `voice_ready` | "Ready?" | energetic game-show announcer |
| V2 | `voice_go` | "Go!" | |
| V3 | `voice_level_up` | "Level up!" | |
| V4 | `voice_winner` | "Winner!" | |
| V5 | `voice_draw` | "It's a draw!" | |

**Total: 16 required (4 music + 12 effects) + 5 optional voice lines.**
If something is missing, the game still runs and that sound is just silent.

## Where to get free AI-generated sounds

Free plans change their terms often. Before downloading, check on each site that the free plan
allows use in a university project, and whether it requires a credit. Write the answer in
`CREDITS.md`.

| Need | Site | AI? | Notes |
|---|---|---|---|
| Music (M1–M4) | **Suno** (suno.com) | yes | free daily credits; best quality for full loops; free plan is non-commercial, keep a credit |
| Music (M1–M4) | **Udio** (udio.com) | yes | free tier; similar to Suno |
| Music + effects | **Stable Audio** (stableaudio.com) | yes | free tier; good for instrumental loops and short effects |
| Music + effects | **MusicGen / AudioGen** (Meta, free demos on huggingface.co/spaces) | yes | runs in the browser, no account needed; model licence is non-commercial |
| Effects (S1–S12) | **ElevenLabs Sound Effects** (elevenlabs.io/sound-effects) | yes | type the prompt, get 4 variations; free plan asks for attribution |
| Voice (V1–V5) | **ElevenLabs Text to Speech** (elevenlabs.io) | yes | announcer-style voices; free plan asks for attribution |
| Voice (fallback) | macOS `say` command | built-in | free and offline, sounds robotic; Task 5 can generate these for you |
| Effects (fallback, not AI) | **sfxr / jsfxr** (sfxr.me) | generator | classic arcade bleeps; free to use |
| Effects (fallback, not AI) | **Kenney** (kenney.nl/assets, "audio") | no | CC0 packs (interface, digital, impact sounds); no credit needed |
| Music/effects (fallback, not AI) | **Pixabay** (pixabay.com/music, /sound-effects) | mixed | free licence, no credit needed |

Tips:
- Generate 2–4 variations per sound and keep the best.
- For music, ask for "loopable" and "no vocals". Vocals clash with the feedback text and the
  voice lines.
- Effects should start immediately. Trim any silence at the start, or Task 5 will.
