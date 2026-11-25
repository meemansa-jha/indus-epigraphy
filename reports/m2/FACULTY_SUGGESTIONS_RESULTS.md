# Faculty suggestions: what we tried and what difference it made

Three suggestions were received:

1. **Use Mahadevan's book** (*The Indus Script: Texts, Concordance and Tables*, 1977). From Sandesh Kamath.
2. **Indus word2vec**: skip-gram with negative sampling, from Tanmay Verlekar's note.
3. **Try LLM2Vec as an alternative to word2vec.** From Sandesh Kamath.

All three were tested on the same data as the rest of the project. That data is the Milestone 3 inscription corpus in Wells numbering, put into reading order, with the Milestone 3 train/test split.

---

## 1. Mahadevan's book as an external check (`src/m2_markov/mahadevan_table1.py`)

**What we did.** Table I of the book gives, for every sign, how often it appears alone, first, in the middle, and last, in reading order. We only have the archive.org OCR of the book, and in it the sign column is garbled. So we kept only rows whose four position counts add up exactly to the printed total. That gave:

- 307 of 417 rows
- 9,280 of 13,372 sign occurrences (69%)

| | Mahadevan, Table I | Our corpus (complete texts, reading order) |
|---|---|---|
| Jar sign (IM77 342 = W740 = P324): total | 1,395 | 1,358 |
| Jar sign: share in final position | **69.6%** | **68.4%** |
| Jar sign: share in medial position | 30.1% | 30.3% |
| Jar sign: share in initial position | 0.1% | 0.4% |
| All signs: solo / initial / medial / final | 1.4 / 21.7 / 52.3 / 24.6% | 2.7 / 24.2 / 48.9 / 24.2% |

**Difference it made.**

- It gives our corpus and our direction finding an **independent, published confirmation**. The most frequent sign ends inscriptions about 69% of the time in both sources.
- Overall positional shares agree within about 3 points.

**Limit.** The OCR cannot be matched to sign numbers row by row, so we cannot use the book for sign-level statistics. Doing that properly means typing Table I in by hand. About 67 frequent signs would take roughly 1–2 hours, and would also unblock Milestone 1's experiment of restricting recognition to the frequent signs.

---

## 2. Indus word2vec (`src/m2_markov/word2vec.py`)

*Numbers in this section are from a single training run. The final comparison, averaged over 3 runs and including LLM2Vec, is in `embedding_comparison.md`; word2vec there averages 22.0% / 0.46 / 0.52.*

**What we did.** We implemented skip-gram with negative sampling exactly as in the note. The training objective was:

L = log σ(u_ctx · v_tgt) + Σᵢ log σ(−u_negᵢ · v_tgt)

Settings:

- window of 2 signs on each side, because inscriptions are short
- 5 negative samples per pair, drawn from unigram^0.75
- 64-dimensional embeddings, trained for 60 epochs
- 43,660 training pairs from the training split only
- the start and end markers kept as tokens

Training loss fell from 3.11 to 2.10. We then measured three things.

### E1. Can word2vec fill a missing sign? (Milestone 3's exact test, 1,304 hidden signs)

| Model | Top-1 | Top-5 | Perplexity |
|---|---|---|---|
| Unigram (frequency only) | 11.0% | 24.4% | 112 |
| word2vec, context only | 8.0% | 27.0% | 468 |
| word2vec + frequency prior | 22.1% | 44.8% | 190 |
| Witten–Bell bigram (Milestone 2) | **33.5%** | 56.3% | 31 |
| Milestone 3 Transformer | 33.4% | **57.0%** | **26** |

### E2. Do the vectors encode a sign's positional role?

For each of the frequent signs, we predicted from its vector alone whether it is a reading-initial, medial or reading-final sign. We used a 7-nearest-neighbour classifier with leave-one-out. We report macro recall because 72% of these signs are medial, which makes plain accuracy misleading.

| Representation | Macro recall |
|---|---|
| Random vectors (control) | 0.37 |
| PPMI-SVD (classic count-based embedding) | 0.36 |
| **word2vec** | **0.43** (0.41–0.48 across seeds and splits) |
| Milestone 3 encoder embeddings | **0.65** |

### E4. Do the vectors recognise variants of the same sign?

The crosswalk says some different Wells numbers are variants of one Parpola sign. For example, W817 and W861 are both P385. For each such variant (34 in total), we asked how highly its partner ranks among the other frequent signs. The score is mean reciprocal rank, where 1.0 means the partner always comes first.

| Representation | Mean reciprocal rank |
|---|---|
| Random (control) | 0.02 |
| PPMI-SVD | 0.30 |
| **word2vec** | **0.53** (0.42–0.59 across seeds and splits) |
| Milestone 3 encoder | **0.62** |

**Difference it made.**

- **word2vec learns real structure.** It is far above random for both positional role and sign variants.
- **Its advantage over a simple count-based embedding is modest and unstable**: clearly better at recognising variants on the current split (0.53 vs 0.30), only slightly better at positional role (0.43 vs 0.36), and its results swing between random seeds. It is a **weak gap-filler**, well short of M2's bigram and M3.
- That fits how word2vec works. It ignores order inside its window, and with about 45,000 training pairs it has little data to learn from.
- **Conclusion:** word2vec gives a useful *map* of the signs, but it does not improve any milestone deliverable.

---

## 3. LLM2Vec (`src/m2_markov/llm2vec.py`)

**What we did.** LLM2Vec turns a left-to-right language model into an embedding model in three steps: switch on bidirectional attention, train with masked next-token prediction (MNTP), then train with unsupervised SimCSE. The email did not say which starting model to use. No pretrained LLM has seen Indus signs, so we trained the starting left-to-right model **from scratch on the Indus corpus**, using the same size as our Milestone 3 model, and then applied all three steps. We measured the model after each step, over 3 training runs.

**Result (mean of 3 runs).**

| | Gap filling, first guess right | Knows a sign's position | Recognises variants |
|---|---|---|---|
| word2vec | 22.0% | 0.46 | 0.52 |
| LLM2Vec, full | 27.6% | 0.59 | 0.67 |
| M3 encoder | 32.7% | 0.62 | 0.68 |

- LLM2Vec clearly beats word2vec on all three tests.
- Its sign embeddings are about as good as our M3 encoder's. The differences on the two embedding tests (0.59 vs 0.62, 0.67 vs 0.68) are within run-to-run noise.
- It is a weaker gap-filler than M3 (27.6% vs 32.7%).
- Its biggest gain comes from steps 1–2 (bidirectional attention + MNTP). The final SimCSE step does not improve the sign embeddings and makes gap filling slightly worse (29.2% → 27.6%).

The full table, with every step and all assumptions, is in `embedding_comparison.md`.

---

## Overall

| Suggestion | Effect on the project |
|---|---|
| Mahadevan's book | **Validates** our corpus and direction finding against the standard published source (jar sign final 69.6% vs our 68.4%). Sign-level use needs Table I typed in by hand. |
| word2vec | Learns real structure, only modestly and unstably better than count-based embeddings, and worse than M2/M3 for prediction. Useful as a visual map of signs, not as an upgrade. |
| LLM2Vec | Applied faithfully to a model trained from scratch on Indus data, it **clearly beats word2vec** and matches our M3 encoder on sign embeddings, but fills gaps less well than M3. |

**Caveats.** Small samples (about 100 signs for the role test, 34 variants for the variant test), and word2vec varies with the random seed. All numbers above are on the leak-free Milestone 3 split (v2).
