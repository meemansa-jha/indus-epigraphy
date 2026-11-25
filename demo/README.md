# Demo video (≤ 3 minutes)

The video is a **live run** of the project; the explanations are in the report.
`demo.py` loads every model once (about 5 s) and then runs 7 steps.
Each step prints its result in the terminal, writes the engine's visual report to `reports/demo/<step>/`, and waits for Enter.

```bash
python demo/demo.py --open        # --open shows each step's visual report in the browser
python demo/demo.py --no-pause    # dry run without pauses
python demo/demo.py --seed 7      # a different shuffle, random text and generated inscription
```

The default seed (5) is the run checked below. Do a dry run first: step 7 generates a new inscription each time the seed changes.

## Recording setup
- **Screen layout:** terminal on the left half (font size 16 or larger, dark theme), browser on the right half.
- **Recording:** QuickTime → File → New Screen Recording (macOS), or OBS. Record at 1080p.
- **Narration:** none needed. Put the caption text below on screen in editing, or speak one line per step.
- **Timing:** press Enter once the browser panel has been on screen for about 5 seconds.

## Shot list

| Time | Step | What appears | On-screen caption |
|---|---|---|---|
| 0:00–0:10 | title | repository README, then run the command | *Indus Epigraphy AI: one engine through four milestones* |
| 0:10–0:30 | 1 typed real inscription | M2 roles (opener … ender), M4 **consistent**; browser: M2/M3/M4 panels | *A real inscription: M2 finds positional roles, M4 says it looks real* |
| 0:30–0:55 | 2 damage each sign | table: M3 recovers 4 of 7 hidden signs, with confidence | *M3 restores damaged signs: 4/7 here, 33% overall, calibrated confidence* |
| 0:55–1:10 | 3 shuffled | score jumps from 11 to 83, still under the threshold | *Shuffled: much less likely, but the guardrail is tuned to catch noise* |
| 1:10–1:20 | 4 random signs | M4 **flagged: noise-like** | *Random signs: rejected* |
| 1:20–1:45 | 5 synthetic image | browser: boxes, crops vs templates; every sign read correctly | *From an image: M1 reads all 7 signs and the pipeline continues* |
| 1:45–2:10 | 6 real photo | browser: real strip with boxes; M4 **flagged: noise-like** (3,715 vs 20 for real text) | *A real photograph: signs found, but misread, and M4 catches it* |
| 2:10–2:45 | 7 generate → draw → read back | new 4-sign text (not in training), drawn, read back correctly, passes M4 | *M3 writes a new inscription; we draw it and the engine reads it back* |
| 2:45–3:00 | close | `reports/engine/generated_gallery.png` or `notebooks/`, then the GitHub URL | *Code, notebooks and results: github.com/meemansa-jha/indus-epigraphy* |

## What not to claim in the video
- Step 5 uses a synthetic image. Real photos are step 6, and the difference between the two is the M1 result.
- Passing M4 means "statistically like real inscriptions", not "meaningful" or "deciphered".
- Step 2 is one inscription. Quote the overall numbers (33% top-1 / 56% top-5) if anyone asks.
