# Sign embeddings and gap filling: word2vec vs LLM2Vec vs our encoder

All models trained on the same Indus inscriptions (Milestone 3 v2 train split, 2,792 inscriptions, Wells numbering) and tested on the same 350 unseen test inscriptions. Values are the mean over 3 training runs, with the range in brackets (count-based and frequency baselines are deterministic, so they have one value).

| Method | Gap filling: first guess right | Gap filling: right sign in top 5 | Gap filling: perplexity (lower = better) | Knows a sign's position (random ≈ 0.32) | Recognises variants of the same sign (max 1.0) |
|---|---|---|---|---|---|
| Random vectors (control) | — | — | — | 0.32 (0.29–0.37) | 0.04 (0.02–0.05) |
| Unigram: always guess by frequency | 11.0% | 24.4% | 112 | — | — |
| Count-based embedding (PPMI-SVD) | — | — | — | 0.36 | 0.30 |
| **Word2vec** (skip-gram, negative sampling) | 22.0% (21.3%–22.5%) | 44.5% (44.0%–44.8%) | 186 (182–190) | 0.46 (0.42–0.52) | 0.52 (0.49–0.55) |
| LLM2Vec step 0: left-to-right model only | 24.6% (23.9%–24.9%) | 43.1% (42.7%–43.6%) | 53 (53–54) | 0.55 (0.52–0.61) | 0.60 (0.52–0.70) |
| LLM2Vec steps 1–2: + bidirectional + MNTP | 29.2% (27.8%–30.1%) | 50.2% (49.4%–51.5%) | 38 (36–39) | 0.60 (0.54–0.68) | 0.67 (0.59–0.74) |
| **LLM2Vec, full** (+ SimCSE) | 27.6% (26.3%–29.1%) | 47.2% (46.4%–48.5%) | 45 (42–47) | 0.59 (0.54–0.67) | 0.67 (0.59–0.74) |
| **M3 masked encoder** (single model; ± = std over 3 runs) | 32.7% (±0.5) | 54.9% (±0.6) | 32 (±0.2) | 0.62 (0.58–0.65) | 0.68 (0.62–0.75) |
| Witten–Bell bigram (Milestone 2) | 33.5% | 56.3% | 31 | — | — |
| M3 ensemble of 3 + calibration (deliverable) | 33.4% | 57.0% | 26 | — | — |

Contextual vectors (each sign encoded on its own between start and end markers) instead of the embedding table:

| Method | Knows a sign's position | Recognises variants |
|---|---|---|
| LLM2Vec, full: contextual vectors | 0.45 (0.43–0.47) | 0.59 (0.48–0.65) |
| M3 encoder: contextual vectors | 0.45 (0.44–0.46) | 0.52 (0.44–0.59) |

**How to read the columns**

- *Gap filling*: every sign of the 350 test inscriptions is hidden once (1,304 guesses) and the model ranks all 426 signs. Random guessing = 0.2%.
- *Perplexity*: 26 means the model is as unsure as if choosing evenly among 26 signs.
- *Knows a sign's position*: for 109 frequent signs, a 7-nearest-neighbour classifier guesses from the vector alone whether the sign usually starts, ends, or sits inside inscriptions; average success over the three groups.
- *Recognises variants*: 34 signs that the published sign concordance lists as variants of one sign; score = average of 1/rank of the twin among frequent signs (1.0 = always ranked first).

**Assumptions**

1. The starting decoder is trained from scratch on Indus data, not a pretrained LLM (the email says only 'try LLM2vec').
2. Model size equals Milestone 3's selected configuration (2 layers, 64 dimensions, 4 heads, dropout 0.2), so the comparison is about the training recipe, not size.
3. Training data = Milestone 3 v2 train split in reading order; validation split used for early stopping only; test split used only for evaluation.
4. Step 0: next-sign prediction, AdamW lr 2e-3, up to 60 epochs, early stopping (patience 8) on validation loss.
5. Step 2 (MNTP): 20% of signs masked per sequence (at least 1), BERT-style 80% [MASK] / 10% random / 10% unchanged; masked sign predicted from the previous position; lr 1e-3, up to 40 epochs, early stopping on validation MNTP loss.
6. Step 3 (SimCSE): mean pooling over non-padding positions, dropout (0.2) as the only augmentation, in-batch negatives, temperature 0.05, 300 steps, lr 1e-4, batch 64.
7. Sign vectors for T2/T3 = the model's input embedding table, the same choice as for word2vec and the M3 encoder; a contextual variant (each sign encoded alone between edge markers, mean-pooled) is reported separately.
8. Gap filling compares single models with uncalibrated softmax (the M3 ensemble + calibration figure, 33.4%, is listed separately).
9. T2 and T3 are tests designed by us; T2 uses ~110 signs and T3 34 variant signs, so differences of a few hundredths are within noise.
10. All LLM2Vec hyper-parameters (learning rates, epochs, mask rate, SimCSE steps and temperature) are our choices for a model this small; the paper tuned them for 7B-parameter models, and they were not tuned here.
11. Word2vec gap filling uses its best variant (context score plus a sign-frequency prior); without the prior it reaches 8% first-guess accuracy.
12. The bigram row is Milestone 3's Witten-Bell baseline on the same test, not a re-run of the Milestone 2 notebook (which uses a different corpus).
13. The M3 encoder's gap-filling figures come from its own training run (mean and standard deviation over 3 seeds, as reported in results.json); its embedding-test figures are the mean and range over the same 3 checkpoints.
14. Signs outside the 426-sign vocabulary are excluded from every test, as in Milestone 3.
