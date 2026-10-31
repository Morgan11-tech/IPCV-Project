# Task 5 – Scene Processing and Real-Time Integration

**Responsible student:** _[full name, s-number]_

## Goal

Tasks 1–4 each work on their own: faces, hands and signs, hand-to-player
association, and the game rules. Task 5 puts them together into **one program
that you can actually play**. It has to keep everything in sync on one clock,
show a game world that reacts to what the players do, stay real-time on a
laptop, and keep running when one of the modules gives bad or missing output.
We also tried not to change the other tasks' code. Where their outputs didn't
fit together, Task 5 adds small adapters in between.

## Pipeline

```
start-up: title + loading screen (models load in a background thread)
          → face registration if needed → game

every frame (one clock, t_ms, shared by all modules):
camera frame (mirrored)
   ├─ T1 faces ....... YuNet + SFace, Player 1 / Player 2   ┐
   ├─ T2 hands ....... MediaPipe hands + sign classifier    │ pipeline.py
   ├─ T3 association . hands → players (pose regions)       │ + stabilize.py
   ├─ adapters ....... one PlayerInput per player           │ + adapters.py
   ├─ T5 segment ..... person mask (MediaPipe Selfie)       ┘
   ├─ T4 game ........ GameController + countdown / pauses   flow.py
   ├─ audio .......... music, sound effects, announcer       audio.py
   └─ render ......... arena + players + effects + HUD       render/
   ▼
window (OpenCV), stage timings in debug mode               measure.py
```

`run_game.py` is the whole main loop. Perception, game flow and rendering are
separate modules, so for example the renderer never touches the game rules.

## Integration work

**Adapters (`adapters.py`).** The modules were written separately, so their
data didn't fit together directly:

- Task 2 gives hand landmarks normalised to 0–1, Task 3 expects pixels.
- Task 2's labels are capitalised ("Hello"), Task 4's word lists are lower case.
  Without converting, Task 4's `word_probs.get(target)` was always 0, so no
  sign could ever score. This was one of the first bugs we hit when joining
  the modules.
- Task 4 needs one `PlayerInput` per player (present, hands, word, confidence,
  locked, probabilities, face box). This is built from Task 1's face state,
  Task 3's hand owners and Task 2's signs.
- The sign model knows more words than the levels use. A small wrapper sets
  the probability of all other words to 0 and re-normalises, so the game can
  only ever receive words that are actually in play.

**Stabiliser (`stabilize.py`).** When everything ran together, faces and hand
owners sometimes flipped for a single frame. On top of Task 1 and Task 3, without
changing their code:

- the face box is smoothed with a **One Euro filter** instead of a fixed EMA:
  strong smoothing when the face is still, little lag when it moves. On a
  simulated face (4 px detector noise, 30 fps) it gave 0.8–1.2 px jitter
  versus 1.5 px for the EMA, and 9 px instead of 23 px lag at 300 px/s;
- a player only "jumps" to a face far away (> 1.5 face widths) after
  recognition agrees for 5 frames in a row, so one bad frame can't swap
  identities;
- a hand only changes owner after Task 3 has picked the new owner for 5 frames
  in a row. This is keyed by Task 2's stable track id instead of the position
  in the hand list.

**Game flow (`flow.py`).** Task 4 decides who scores. Task 5 adds the parts the
rules leave open: a 3-2-1 countdown, a 1.5 s pause between rounds so players
can see the result, and 4 s of celebration before a rematch can start.

**Start-up.** Loading all models takes a few seconds, so it runs in a
background thread while an animated title screen with a real progress bar is
shown. The camera itself is opened on the main thread, because macOS requires
that.

## Scene processing

**Person segmentation (`segment.py`).** MediaPipe's Selfie Segmenter gives a
person probability per pixel. Used directly, the outline flickers and has a
hard, cut-out edge, so we clean the mask at a quarter of the frame size:

1. temporal smoothing `m = 0.6·new + 0.4·previous` against flicker,
2. a smoothstep contrast curve between 0.45 and 0.85, so confident pixels
   become fully person or fully background and only the edge stays soft,
3. Gaussian feathering, so the player blends into the scene,
4. bilinear upscaling back to full size.

The mask is used to place the players from the camera into a rendered
pixel-art arena.

**The scene reacts to the game.** The arena changes with the game state:
sunset in the lobby, daytime at level 1, dusk at level 2, red night at level
3, and a night sky in the winner's colour at game over. Scene changes crossfade,
and higher levels move faster. Game events trigger effects:

| Event (Task 4) | What happens on screen and in audio |
|---|---|
| correct sign | sparks, ring and "+10" in the player's colour, coin sound |
| wrong sign | "WRONG" wobbles, red flash, small screen shake, buzzer |
| time's up | banner, red flash, bigger shake, lost hearts fade |
| level up | colour band, new scene, new music track, voice "Level up!" |
| game over | confetti, "PLAYER X WINS!" or "DRAW!", stats table |

The stats table shows correct and wrong signs, fastest and average answer time,
and best streak per player.

**Retro look (`pixelate.py`).** Everything drawn on top of the camera is
turned into 2×2 pixel blocks in one pass: we compare the frame before and
after drawing, keep only blocks where enough pixels changed, and scale them up
with nearest-neighbour. The players themselves stay sharp.

## Real-time performance

`measure.py` times every stage of every frame (T1 faces, T2 hands, T3 assoc,
T5 segment, T4 game, draw), and debug mode (D) shows these together with the
FPS. Things we did to keep it real-time:

- face detection on a 640 px copy of the frame (Task 1),
- segmentation mask cleaned at ¼ resolution (about 2 ms for the model),
- `--no-pose` switches off the pose model if a laptop is too slow (Task 3
  then falls back to circles), and `--no-arena` switches off segmentation,
- models load once, in the background, before the game starts.

_[Measured results to add: mean and low-percentile FPS during a normal 2-player
game, time per stage (which stage is the bottleneck), CPU/memory and laptop
specs, start-up time until the game is playable, a long run (e.g. 15 min)
with crashes/freezes, and behaviour under fast movement and dim lighting.]_

## Controlled behaviour when things go wrong

- A player who isn't seen keeps their state for a short grace period (Task 1)
  and isn't counted as present, so they can't score or lose hearts by
  accident.
- Identity and hand-owner switches need several frames of agreement
  (stabiliser).
- Missing sound files or no audio device: the game simply stays silent.
- `--fake-p2` adds a fake second player for testing alone.

## Course topics used

- **Segmentation:** per-pixel person mask, thresholding with a soft
  (smoothstep) curve, alpha blending of player and background.
- **Image filtering:** Gaussian feathering of the mask, temporal smoothing,
  the One Euro filter as an adaptive low-pass filter.
- **Basic image processing:** resizing with area and bilinear interpolation,
  nearest-neighbour upscaling for the pixel look, colour grading per scene,
  frame differencing to find the drawn overlay.
- **Tracking:** hysteresis on identity and hand ownership.

## Limitations

- **Segmentation edges** are still visible with fast movement, low light or a
  background close to skin or clothing colour.
- **Everything runs in one thread**, one stage after the other. The frame rate
  is limited by the sum of all models (faces, hands, pose, segmentation), not
  by the slowest one.
- The stabiliser makes switches safer but also slower: a real swap of hand
  owner takes about 5 frames.
- Tested mainly on our own laptops and room. Other cameras and lighting still
  need checking.
