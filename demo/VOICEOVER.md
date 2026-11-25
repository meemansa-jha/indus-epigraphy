# Voice-over script for the demo video

Matches `python demo/demo.py --open` with the default seed (5) and the timings in `README.md`.
It is about 400 words, which takes roughly 3 minutes at a calm pace (about 140 words a minute).
Each section starts as its step appears on screen. A plain-text copy for text-to-speech is in `VOICEOVER.txt`.

---

**[0:00–0:10] Title: the command is typed and the models load**

This is the Indus Epigraphy engine. One program runs all four milestones: reading signs from an image, positional statistics, a masked-sign model, and a guardrail that checks every output.

**[0:10–0:30] Step 1: a real inscription, typed in**

We start with a real inscription the models never saw in training. Milestone 2 labels each sign's role: sign 817 typically opens a text, and the jar sign, 740, typically ends one. Milestone 4 compares the whole text with real inscriptions, and its verdict is: consistent with real inscriptions.

**[0:30–0:55] Step 2: each sign hidden in turn**

Now we damage the inscription one sign at a time and ask Milestone 3 to restore it. It recovers four of the seven signs, and it is most confident exactly where it is right: the jar sign at 94%. Across all held-out data it gets the first guess right a third of the time, and its confidence is calibrated.

**[0:55–1:10] Step 3: the same signs, shuffled**

Same signs, wrong order. The score gets about eight times worse. It still stays under the guardrail's threshold, because the guardrail is built to catch noise, not every reordering.

**[1:10–1:20] Step 4: random signs**

Random signs from the sign list are rejected immediately as noise-like.

**[1:20–1:45] Step 5: the inscription as an image**

Now the same inscription as an image: a synthetic carved strip. Milestone 1 finds each sign, matches it to Mahadevan's sign list, and orders the signs right to left. All seven are read correctly. The numbering is converted automatically, and the rest of the pipeline gives the same verdict as the typed text.

**[1:45–2:10] Step 6: a real photograph**

This is a real photograph. Detection works: the signs are boxed. Recognition does not: on real photos only twelve to seventeen percent of signs are identified correctly. The guardrail catches it. The reading is flagged as noise-like, with a score nearly two hundred times worse than typical real text.

**[2:10–2:45] Step 7: generate, draw and read back**

Finally, generation. Milestone 3 writes a new inscription. We check that it is not a copy of any training text, or even one sign away from one. The guardrail accepts it as statistically realistic. We draw it as a carved strip, and the engine reads it back correctly. Passing the guardrail means the text is statistically like real inscriptions. It does not mean it has a meaning.

**[2:45–3:00] Close: the gallery and the repository**

All the code, notebooks and results are in our public repository. The report explains what works, what doesn't, and why.
