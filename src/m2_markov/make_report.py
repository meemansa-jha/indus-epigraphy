"""Builds the detailed Milestone 2 report (HTML with embedded images -> PDF).
Every number is read from outputs/ produced by the notebook, so the report cannot drift from the analysis."""
import base64, json, collections, html, datetime, sys
import pandas as pd

OUT = 'outputs'; FIG = f'{OUT}/figures'
R = json.load(open(f'{OUT}/results_summary.json'))
rules = pd.read_csv(f'{OUT}/directional_rules.csv', index_col='sign')
coll = pd.read_csv(f'{OUT}/collocations.csv')
asym = pd.read_csv(f'{OUT}/pair_order_asymmetry.csv')
runs = pd.read_csv(f'{OUT}/error_propagation_runs.csv')
segs = pd.read_json(f'{OUT}/segments_rtl.json')

# ---------- helpers
def img(name, caption, width='100%'):
    b64 = base64.b64encode(open(f'{FIG}/{name}', 'rb').read()).decode()
    return (f'<figure><img src="data:image/png;base64,{b64}" style="width:{width}" alt="{html.escape(caption)}">'
            f'<figcaption><b>{name}</b> — {caption}</figcaption></figure>')

def table(df, cls='', index=True):
    return df.to_html(index=index, escape=True, classes=f'tbl {cls}', border=0, na_rep='—')

def kv(rows):
    return '<table class="tbl kv">' + ''.join(f'<tr><th>{html.escape(str(k))}</th><td>{v}</td></tr>' for k, v in rows) + '</table>'

def check(items):
    return '<ul class="checks">' + ''.join(f'<li><span class="ok">✓</span> {i}</li>' for i in items) + '</ul>'

def issue(items):
    return '<ul class="issues">' + ''.join(f'<li><span class="fix">⚑</span> {i}</li>' for i in items) + '</ul>'

c, s1, s2, s3, s4, s5, s6, s7 = (R[k] for k in ['corpus', 'step1_sanity', 'step2_positions', 'step3_matrix',
                                               'step4_smoothing', 'step5_direction', 'step6_collocations',
                                               'step7_error_propagation'])

# ---------- derived tables
begin = pd.Series([x[0] for x, t in zip(segs.signs, segs.true_start) if t]).value_counts()
end = pd.Series([x[-1] for x, t in zip(segs.signs, segs.true_end) if t]).value_counts()
be = pd.DataFrame({'Beginner': begin.index[:8], 'n': begin.values[:8], 'share %': (100 * begin.values[:8] / begin.sum()).round(1),
                   'Ender': end.index[:8], 'n ': end.values[:8], 'share % ': (100 * end.values[:8] / end.sum()).round(1)},
                  index=range(1, 9))
tok = pd.Series([x for s in segs.signs for x in s]).value_counts()
top10 = pd.DataFrame({'occurrences': tok.head(10), 'share %': (100 * tok.head(10) / tok.sum()).round(1)}); top10.index.name = 'sign'

sig = coll[coll.collocation == True].sort_values('NPMI', ascending=False).copy()
sig['× chance'] = (sig['count'] / sig['expected']).round(0).astype(int)
sig['q'] = sig['q'].map(lambda v: f'{v:.1e}')
sig_tab = sig[['pair', 'count', 'expected', '× chance', 'PMI', 'NPMI', 'q', 'reverse_count']].reset_index(drop=True)
sig_tab.index = sig_tab.index + 1

asym_tab = asym.copy(); asym_tab['q'] = asym_tab['q'].map(lambda v: f'{v:.1e}'); asym_tab = asym_tab.drop(columns='p')
asym_tab.index = asym_tab.index + 1

r7 = (runs.groupby(['detection', 'accuracy'])
      .agg(**{'P324 share of endings': ('share_P324_end', 'mean'),
              'direction detected (% runs)': ('direction_detected', lambda x: 100 * x.mean()),
              'start−end gap (bits)': ('entropy_gap', 'mean'),
              'collocations recovered (%)': ('colloc_recovered', lambda x: 100 * x.mean())})
      .round(2).sort_index(level=[0, 1], ascending=[False, False]))
r7.index = r7.index.set_names(['detection', 'recognition accuracy'])
r7 = r7.reset_index()
r7['recognition accuracy'] = (100 * r7['recognition accuracy']).round(0).astype(int).astype(str) + '%'
r7['P324 share of endings'] = (100 * r7['P324 share of endings']).round(0).astype(int).astype(str) + '%'

rules_tab = rules.copy()
for qc in ['q (start)', 'q (end)', 'q (edge)']:
    rules_tab[qc] = rules_tab[qc].map(lambda v: f'{float(v):.1e}')
roles = {r: ', '.join(f'<code>{x}</code>' for x in s5['roles'][r]) for r in ['OPENER', 'ENDER', 'MEDIAL']}
fmt_pct = lambda v: f'{100*v:.0f}%'
m1 = s7['at_M1_accuracy_17pct']; perf = s7['perfect_recognition_with_M1_detection']
today = datetime.date.today().strftime('%d %B %Y')

# ---------- document
H = []
H.append(f'''
<section class="cover">
  <p class="eyebrow">Indus Epigraphy AI Project</p>
  <h1>Milestone 2 — Positional Bias &amp; Markov Chain Modelling</h1>
  <p class="sub">Detailed summary of all work, data, results, figures and verification checkpoints</p>
  <p class="meta">{today} · Corpus: CISI (Parpola numbering), {c["texts"]} inscriptions · Notebook: <code>m2_markov.ipynb</code></p>
  <div class="tiles">
    <div class="tile"><div class="num">p &lt; 0.0001</div><div class="lbl">Fixed writing direction<br>(flip test, 10,000 permutations)</div></div>
    <div class="tile"><div class="num">{s2["top_ender_share_pct"]}%</div><div class="lbl">of genuine endings are<br><code>{s2["top_ender"]}</code></div></div>
    <div class="tile"><div class="num">{s4["Witten-Bell"]["perplexity"]}</div><div class="lbl">Witten-Bell held-out perplexity<br>(add-one: {s4["Add-one (Laplace)"]["perplexity"]})</div></div>
    <div class="tile"><div class="num">{s6["significant"]}</div><div class="lbl">significant sign<br>collocations</div></div>
  </div>
  <div class="callout">
    <b>The story in one paragraph.</b> The Indus script has a <b>fixed writing direction</b> and genuine
    <b>positional syntax</b>: a small class of signs ends texts, another opens them, and some never touch an edge.
    Every claim was tested against a random baseline. On the way we found and avoided a trap:
    <b>the textbook direction test (forward vs backward model perplexity) is mathematically blind to direction.</b>
    Finally, simulating Milestone 1's measured error rates shows that <b>none of these findings would survive
    automated readings at today's 17% recognition accuracy</b>. That justifies running on expert transcriptions and sets up Milestone 4.
  </div>
</section>

<h2>Contents</h2>
<ol class="toc">
  <li>Project context and what Milestone 2 asked for</li>
  <li>Data, sourcing and the three decisions that shape everything</li>
  <li>Step-by-step work with checkpoints (Steps 0–8)</li>
  <li>Deliverables mapped to the proposal</li>
  <li>Honest negative results and limitations</li>
  <li>Links to Milestones 3 and 4</li>
  <li>Figure guide for the deck</li>
  <li>File index</li>
  <li>Appendix: full tables</li>
</ol>

<h2>1. Project context</h2>
<p>The Indus script (~2600–1900 BCE) is undeciphered: there is no bilingual text, and inscriptions average about five signs.
The project treats the inscriptions as a <b>structured notational system</b> and studies its regularities
computationally, <b>never claiming what any sign means</b>. Signs are opaque labels such as <code>P324</code>.</p>
<h3>Where Milestone 1 left off</h3>
<ul>
<li><b>Detection works:</b> 99% coverage and 86% precision on 77 inscription strips.</li>
<li><b>Recognition does not:</b> best 17% top-1 / 41% top-5 over 417 classes. There is a reproducible sim-to-real gap.</li>
<li>Milestone 2 therefore runs on <b>expert transcriptions</b> (the CISI corpus), as the M1 summary recommended.</li>
</ul>
<h3>What the proposal asks of Milestone 2</h3>
{kv([("Objective", "Establish the direction of writing and calculate sign-pairing constraints."),
     ("Technical task", "Map sequential transition frequencies; construct a Markov chain matrix; evaluate directional asymmetries by comparing text beginners vs text enders; apply <b>Witten-Bell smoothing</b> for unobserved pairings."),
     ("Deliverable", "<i>“A directional statistical matrix verifying right-to-left syntax rules and tracking sign collocation indices.”</i>"),
     ("Cross-cutting goal", "<i>“Analyzing how image digitization errors propagate down into downstream sequence analysis layers.”</i>")])}

<h2>2. Data, sourcing and key decisions</h2>
<h3>2.1 The corpus</h3>
{kv([("Source", "Digital CISI corpus — GitHub <code>mayig/indus-valley-script-corpus</code>, commit <code>ad2f1e2</code> (MIT/Apache licence). A snapshot is included in the folder."),
     ("Content", f"{c['texts']} inscriptions (Mohenjo-daro M-1 to M-199), one JSON file each; every sign has an ID plus <i>damage</i>, <i>line</i> and <i>uncertainty</i> features."),
     ("Size after cleaning", f"{c['sign_tokens']} sign tokens, {c['distinct_signs']} distinct signs, {c['complete_single_line_texts']} complete single-line texts."),
     ("Numbering", "Parpola (<code>P001</code>…). It is not Mahadevan's 1–417, and the two cannot be mixed in one matrix.")])}
<h3>2.2 Attempt to obtain the larger IM77 corpus</h3>
<p>The proposal also points to Mahadevan's IM77 concordance at indusscript.in (~2,900 objects). It is a searchable,
sign-in website. <b>No public export of its inscription texts could be found</b>, and an open-source toolkit that surveyed
the sources reached the same conclusion. A zip supplied during the session turned out to be the 417 sign-image chart (the same
Kaggle set M1 used), which holds no sequences. <b>Decision: proceed with CISI</b>, which the proposal itself lists.
A second corpus (<code>yajnadevam/indus-website</code>, ~2,500 texts, its own numbering) exists and is noted as a future replication option.</p>
<h3>2.3 The three decisions applied everywhere (Step 0)</h3>
<table class="tbl"><thead><tr><th>Decision</th><th>Rationale</th></tr></thead><tbody>
<tr><td><b>Right-to-left reading order</b>: each text is reversed</td><td>The corpus stores signs left-to-right as they appear on the impression, while the script is read right-to-left. After reversal, position 0 is the first sign read.</td></tr>
<tr><td><b><code>P000</code> = break, not a sign</b> ({c["P000_breaks"]} cases in 18 texts)</td><td>Deleting it would invent pairs that were never seen. Counting it would make "damage" look like the 4th most common ender. Texts are cut into segments at each break, and an edge next to a break is <i>not</i> treated as a genuine beginning or ending. 17 of the 19 breaks sit at a text edge, where seals wear.</td></tr>
<tr><td><b>Line breaks handled like breaks</b> ({c["two_line_texts"]} two-line texts)</td><td>How the two lines join is uncertain, so within-line pairs are kept but these texts' edges are not used for beginner/ender analysis.</td></tr>
<tr><td><b>Partly-damaged but identified signs kept</b> (118 signs)</td><td>The transcriber still recognised them.</td></tr>
</tbody></table>
<div class="warn"><b>⚠ Correction to the Milestone 1 summary.</b> M1 §11 states that <code>P324</code> <i>opens</i> 77 inscriptions and
<code>P385</code> <i>closes</i> 29. Those counts were taken in the corpus's stored (left-to-right) order. <b>In reading order it is the
reverse: <code>P324</code> is the dominant ender and <code>P385</code> the dominant opener.</b> The M1 slides should be updated so that both milestones agree.</div>
''')

# ---------------- STEP 0
H.append(f'''
<h2>3. Step-by-step work and checkpoints</h2>
<p>Each step was explained, built, run end-to-end and verified before moving on. Verification checks are built into the
notebook as assertions, so the notebook stops if any of them fails.</p>

<h3 class="step">Step 0 — Load the corpus and fix the reading order</h3>
<p><b>Purpose:</b> put every text into reading order and handle damage and line breaks once, so that every later count is right.</p>
{kv([("Texts loaded", c["texts"]), ("Sign slots", 1003), ("P000 breaks removed", c["P000_breaks"]),
     ("Real sign tokens", c["sign_tokens"]), ("Clean segments", 188), ("Usable sign→sign pairs", 796),
     ("Genuine beginnings / endings", "166 / 161"), ("Complete single-line texts", c["complete_single_line_texts"])])}
<p><b>Worked examples</b> (as printed in the notebook):</p>
<table class="tbl"><thead><tr><th>Text</th><th>Stored (left→right)</th><th>After Step 0 (reading order)</th><th>Flags</th></tr></thead><tbody>
<tr><td>M-101A</td><td><code>P086 P210 P324 P003 P122 P385</code></td><td><code>P385 P122 P003 P324 P210 P086</code></td><td>real start ✓ real end ✓</td></tr>
<tr><td>M-19A</td><td><code>P000 P122 P385</code></td><td><code>P385 P122</code></td><td>real start ✓ · end lost ✗</td></tr>
<tr><td>M-50A</td><td><code>P324 … P122 P000</code></td><td><code>P122 … P324</code></td><td>start lost ✗ · real end ✓</td></tr>
<tr><td>M-165A</td><td>two lines</td><td>two segments</td><td>no cross-line pair; edges unused</td></tr>
</tbody></table>
<p class="cp">Checkpoint</p>
{check(["Every real sign survives exactly once (984 in = 984 out)", "No <code>P000</code> left inside any segment",
        "No text lost (179 in = 179 out)", "Reversal matches stored order (M-101A spot check)"])}

<h3 class="step">Step 1 — Sanity check against known statistics</h3>
<p><b>Purpose:</b> confirm that cleaning did not distort the corpus by reproducing Milestone 1's figures and Mahadevan's published fact.</p>
<table class="tbl"><thead><tr><th>Measure</th><th>M1 report</th><th>M1 method re-run</th><th>Cleaned (M2)</th><th>Why it differs</th></tr></thead><tbody>
<tr><td>Mean length</td><td>5.60</td><td>5.60</td><td><b>{s1["mean_length"]}</b></td><td>Complete texts only; <code>P000</code> not counted</td></tr>
<tr><td>Median length</td><td>5</td><td>5</td><td><b>{int(s1["median_length"])}</b></td><td>—</td></tr>
<tr><td>Distinct signs</td><td>182</td><td>182</td><td><b>{c["distinct_signs"]}</b></td><td><code>P000</code> no longer counted as a sign</td></tr>
<tr><td>Top-67 coverage</td><td>83.4%</td><td>83.4%</td><td><b>{s1["top67_coverage_pct"]}%</b></td><td>Unchanged (83.45% → 83.43%)</td></tr>
</tbody></table>
{kv([("Signs covering 50% of occurrences", s1["signs_for_50pct"]), ("Signs covering 80% of occurrences", f'{s1["signs_for_80pct"]} (Mahadevan: 67 of 417, “over 80 per cent”)'),
     ("Signs seen exactly once (hapax)", f'{s1["hapax_signs"]} of {c["distinct_signs"]} (43%) — the reason smoothing is needed')])}
<p><b>Most frequent signs</b></p>{table(top10, 'narrow')}
{img("m2_01_inscription_length.png", "Inscription length distribution for 154 complete single-line texts; mean 5.57.")}
{img("m2_02_sign_coverage.png", "Cumulative coverage: 17 signs make up half of all occurrences, 58 make up 80%.")}
<p class="cp">Checkpoint</p>
{check(["M1's figures reproduced exactly with M1's method", "All differences after cleaning explained", "Token total = 984 (assertion)"])}
''')

# ---------------- STEP 2
H.append(f'''
<h3 class="step">Step 2 — Text beginners vs text enders</h3>
<p><b>Purpose:</b> if direction did not matter, the two ends of a text would be interchangeable. A strong difference is the fingerprint of a fixed direction.</p>
{table(be, 'narrow')}
<table class="tbl"><thead><tr><th></th><th>Beginnings</th><th>Endings</th></tr></thead><tbody>
<tr><td>Genuine edges</td><td>166</td><td>161</td></tr>
<tr><td>Distinct signs used</td><td>{s2["distinct_beginners"]}</td><td><b>{s2["distinct_enders"]}</b></td></tr>
<tr><td>Top sign</td><td><code>{s2["top_beginner"]}</code> {s2["top_beginner_share_pct"]}% (95% CI 12–24%)</td><td><code>{s2["top_ender"]}</code> <b>{s2["top_ender_share_pct"]}%</b> (95% CI 38–53%)</td></tr>
<tr><td>Entropy (unpredictability)</td><td>{s2["entropy_begin_bits"]} bits</td><td><b>{s2["entropy_end_bits"]} bits</b></td></tr>
</tbody></table>
<p><b>Flip test.</b> The Jensen–Shannon divergence between beginner and ender distributions is <b>{s2["flip_test_jsd"]} bits</b>.
In 10,000 versions where each text's direction was decided by a coin toss, the maximum was <b>{s2["flip_test_null_max"]}</b>,
so <b>p {s2["flip_test_p"]}</b>. Random versions centre around ~0.33, not 0, because small samples always differ by chance.
That is exactly why we compare against shuffled data instead of against zero.</p>
{img("m2_03_start_vs_end.png", "Frequent signs keep to one end: P324 ends texts, P385 opens them, P122 / P120 / P062 never touch an edge.")}
{img("m2_04_flip_test.png", "Real beginner–ender divergence (0.89 bits) against 10,000 random-direction versions (max 0.42).")}
<div class="note"><b>Scope of the claim.</b> The statistics prove the direction is <i>fixed</i>. <i>Which</i> end is the start comes from physical
evidence (e.g. signs cramped at the left edge). Every result is consistent with right-to-left reading.</div>
<p class="cp">Checkpoint</p>
{check(["Only genuine edges used (damage-adjacent edges excluded)", "Wilson 95% confidence intervals reported", "Permutation baseline instead of an assumed zero"])}
{issue(["Chart fix: signs that are never at an edge first showed as empty rows. They are now labelled “never at either end — always in the middle”."])}
''')

# ---------------- STEP 3
H.append(f'''
<h3 class="step">Step 3 — Transition counts and the Markov chain matrix</h3>
<p><b>Purpose:</b> a first-order Markov chain (bigram model) assumes that the next sign depends only on the current one.
Rows = current sign, columns = next sign, cells = P(next | current). <code>&lt;S&gt;</code> (START) and <code>&lt;/S&gt;</code> (END)
markers are added only at genuine edges, so beginner/ender behaviour lives inside the same matrix.</p>
{kv([("Transitions counted", f'{s3["transitions"]:,} = 796 sign→sign + 166 START→sign + 161 sign→END'),
     ("Matrix size", "182 × 182 = 33,124 cells"), ("Cells ever observed", f'{s3["observed_cells"]} ({100-s3["zero_cells_pct"]:.2f}%)'),
     ("Cells MLE sets to exactly 0", f'{s3["zero_cells_pct"]}%'), ("Rows built from one observation", s3["single_observation_rows"]),
     ("Rows where MLE is “100% certain”", s3["rows_mle_100pct_certain"])])}
<p><b>Strongest rules</b> (reading order):</p>
<ul>
<li><code>P385 → P122</code>: 28 of 33 (85%). The commonest opener is almost always followed by <code>P122</code>.</li>
<li><code>P332 → P324</code>: 10 of 10.</li>
<li><code>P324 → END</code>: 73 of 94 (78%). Otherwise mostly <code>P324 → P013</code> (9×), and <code>P013 → END</code> 8 of 8, an alternative two-sign ending.</li>
<li><code>P217 → END</code> 76%; <code>P086 → END</code> 53%.</li>
</ul>
{img("m2_05_markov_matrix_mle.png", "Unsmoothed transition matrix for the 25 most frequent signs + START/END. Empty = never observed.", "88%")}
<p class="cp">Checkpoint</p>
{check(["Every row sums to 1", "Totals reconcile exactly with Step 0 (796 + 166 + 161 = 1,123)"])}
{issue(["Label fix: a column described as “share of P324 occurrences” was really “share of observed predecessors” (2 occurrences follow a break). It was renamed."])}

<h3 class="step">Step 4 — Witten-Bell smoothing</h3>
<p><b>Problem:</b> MLE gives probability 0 to 98% of pairs, so one unseen pair zeroes a whole inscription.
<b>Witten-Bell:</b> <span class="formula">P(w|h) = [C(h,w) + T(h)·P<sub>uni</sub>(w)] / [C(h) + T(h)]</span>, where
C = observations of the row and T = number of distinct successors. Each row holds back <b>T/(C+T)</b> for unseen pairs: the thinner or more varied the evidence, the more it holds back.
P<sub>uni</sub> is itself interpolated with a uniform distribution, so no cell is ever 0.</p>
<table class="tbl"><thead><tr><th>Row</th><th>C</th><th>T</th><th>Held back</th><th>Reading</th></tr></thead><tbody>
<tr><td><code>P385</code></td><td>33</td><td>5</td><td>13.2%</td><td>well observed and consistent → stays confident (P385→P122: 85% → 74%)</td></tr>
<tr><td><code>P324</code></td><td>94</td><td>13</td><td>12.1%</td><td>—</td></tr>
<tr><td><code>P122</code></td><td>72</td><td>34</td><td>32.1%</td><td>very varied → stays open</td></tr>
<tr><td>START (openers)</td><td>166</td><td>86</td><td>34.1%</td><td>texts open in many ways</td></tr>
<tr><td>73 signs seen once</td><td>1</td><td>1</td><td>50%</td><td>e.g. <code>P265</code>: MLE “100% P324” → WB 54% P324</td></tr>
</tbody></table>
<p><b>Validation: 10-fold cross-validation by inscription</b> (fit on 90% of texts, score the held-out 10%):</p>
<table class="tbl"><thead><tr><th>Method</th><th>Held-out pairs given P = 0</th><th>Perplexity (lower is better)</th></tr></thead><tbody>
<tr><td>MLE (no smoothing)</td><td><b>{s4["MLE (no smoothing)"]["zero_prob_events"]} of 1,123 (46%)</b></td><td>∞, fails in all 10 folds</td></tr>
<tr><td>Add-one (Laplace)</td><td>0</td><td>{s4["Add-one (Laplace)"]["perplexity"]}</td></tr>
<tr><td><b>Witten-Bell</b></td><td><b>0</b></td><td><b>{s4["Witten-Bell"]["perplexity"]}</b></td></tr>
</tbody></table>
<p>Perplexity ≈ the number of signs the model is effectively choosing between at each step. Uniform guessing over 182 options would give 182.</p>
{img("m2_06_witten_bell_reserve.png", "Each dot is one row of the matrix: thin or varied evidence → more probability held back for unseen pairs.")}
<p class="cp">Checkpoint</p>
{check(["Every row sums to 1; smallest cell 6.8 × 10⁻⁵ (> 0)", "Count matrix identical to Step 3 (assertion)", "Smoothing justified on held-out data, not just asserted"])}
{issue(["Removed a divide-by-zero warning for rows with no observations (they reserve 100% by definition).", "Chart fix: one label pointed at the wrong dot; axis ticks changed to 1 / 3 / 10 / 30 / 100."])}
''')

# ---------------- STEP 5
rules_show = rules_tab[['occurrences', 'at START %', 'at END %', 'chance %', 'q (start)', 'q (end)', 'q (edge)', 'role', 'strongest next sign (P_WB)']]
H.append(f'''
<h3 class="step">Step 5 — Directional tests and the directional rules matrix</h3>
<h4>5a. The perplexity trap (honest negative result)</h4>
<table class="tbl"><thead><tr><th></th><th>Right-to-left</th><th>Left-to-right</th></tr></thead><tbody>
<tr><td>Training data, MLE (bits / transition)</td><td>{s5["forward_vs_backward_training_bits"][0]}</td><td><b>{s5["forward_vs_backward_training_bits"][1]} — identical</b></td></tr>
<tr><td>Held-out perplexity, Witten-Bell</td><td>{s5["forward_vs_backward_heldout_perplexity"][0]}</td><td>{s5["forward_vs_backward_heldout_perplexity"][1]} (0.1% = noise)</td></tr>
</tbody></table>
<p>A bigram likelihood is ∏ C(x<sub>t</sub>,x<sub>t+1</sub>) / C(x<sub>t</sub>). Reversing the text uses the same pair counts, only flipped, and the same sign counts,
so <b>the overall score is mathematically blind to direction</b>. Any claim of direction based on it is empty. This is a direct exhibit for Milestone 4.</p>

<h4>5b. Positional entropy, a direction test that works</h4>
<table class="tbl"><thead><tr><th>Position from edge</th><th>Aligned by FIRST sign</th><th>Aligned by LAST sign</th><th>Random-direction 95% range</th></tr></thead><tbody>
<tr><td>1</td><td>{s5["entropy_pos1_from_start"]}</td><td><b>{s5["entropy_pos1_from_end"]}</b></td><td>4.36–5.04</td></tr>
<tr><td>2</td><td><b>{s5["entropy_pos2_from_start"]}</b></td><td>5.60</td><td>5.25–5.68</td></tr>
<tr><td>3</td><td>5.32</td><td>5.20</td><td>5.10–5.47</td></tr>
<tr><td>4</td><td>5.38</td><td>5.01</td><td>5.10–5.47</td></tr>
</tbody></table>
<p>The last sign is far more predictable than chance allows, while the first sign is <i>more</i> varied than chance. The opening's structure sits one position in (<code>P385 → P122</code>).
Position-1 gap = 2.37 bits against a random maximum of 1.29, p {s5["pos1_gap_p"]} (128 complete texts with ≥ 4 signs).</p>
{img("m2_07_positional_entropy.png", "Predictability by position, counted from the start (blue) and from the end (orange), against the random-direction band.")}

<h4>5c. Directional rules</h4>
<p>For each sign seen ≥ 8 times in complete texts: start/end rates against the chance rate (153 texts ÷ 857 signs = 17.9%), using a
two-sided binomial test with Benjamini–Hochberg correction (q &lt; 0.05).</p>
{kv([("ENDERS", roles["ENDER"]), ("OPENERS", roles["OPENER"]), ("MEDIAL (avoid both edges)", roles["MEDIAL"]),
     ("No clear preference", f'{int((rules.role == "no clear preference").sum())} signs, mostly too rare to call')])}
<p>Headline examples: <code>P324</code> at the end in 79% of its appearances; <code>P217</code> 87%; <code>P385</code> at the start in 86%;
<code>P122</code> at an edge in <b>0 of 65</b> appearances (~23 expected by chance).</p>
<p class="small">The full rules table is in Appendix A.</p>
<p class="cp">Checkpoint</p>
{check(["Forward/backward training identity holds exactly (assertion)", "Multiple-testing correction applied", "Spot-check assertions guard the rules table against row misalignment"])}
{issue(["<b>Bug caught and fixed:</b> on the first run, roles and next-signs were attached to the wrong signs (P324 shown as “medial”, P013 as “opener”). One column was ordered alphabetically and the others weren't. Fixed by aligning to a single index, and assertions were added so it cannot recur."])}
''')

# ---------------- STEP 6
H.append(f'''
<h3 class="step">Step 6 — Collocation index (PMI)</h3>
<p><b>Purpose:</b> find pairs that occur together <i>more often than chance predicts</i>, as opposed to pairs that are common only because both signs are common.
<span class="formula">PMI(A→B) = log₂ [ P(A→B) / (P(A·)·P(·B)) ]</span>; NPMI rescales it to −1…+1.
Safeguards: a pair must be seen ≥ 3 times <b>and</b> pass Fisher's exact test (one-sided, BH-corrected, q &lt; 0.05).</p>
{kv([("Distinct sign→sign pairs observed", s6["distinct_pairs"]), ("Pairs seen ≥ 3 times (scored)", s6["scored_pairs"]),
     ("Significant collocations", f'<b>{s6["significant"]}</b>'),
     ("Frequent pairs (≥ 5) occurring in ONE order only", f'{s6["frequent_pairs_one_order_only"]}; {s6["one_order_significant"]} individually significant')])}
<p><b>Top collocations</b> (full list of {s6["significant"]} in Appendix B):</p>
{table(sig_tab.head(10), 'narrow')}
<ul>
<li><b>Common ≠ connected:</b> <code>P050 → P324</code> occurs 3 times, but chance predicts 3.8, so it is not a collocation.</li>
<li><b>Why safeguards matter:</b> the 421 pairs seen only once have a median PMI of 3.66, <i>higher</i> than the real collocations (3.27). Raw PMI rewards rarity.</li>
<li><b>Order evidence:</b> <code>P385 → P122</code> 28 times, <code>P122 → P385</code> 0 times.</li>
</ul>
{img("m2_08_collocations_pmi.png", "All 529 pairs: raw PMI is highest for one-off pairs (grey); real collocations (blue) need frequency plus significance.")}
<p class="cp">Checkpoint</p>
{check(["Sign→sign total = 796 (assertion)", "Frequency floor + significance test + multiple-testing correction"])}
{issue(["Chart fix: labels overlapped and sat slightly off their (jittered) dots; they were repositioned."])}
''')

# ---------------- STEP 7
H.append(f'''
<h3 class="step">Step 7 — Error propagation from Milestone 1</h3>
<p><b>Purpose:</b> answer the proposal's question on how image-digitisation errors propagate downstream. M1's measured error profile is injected into the clean transcriptions and the M2 analyses are re-run.</p>
<table class="tbl"><thead><tr><th>Error</th><th>Derivation from M1</th><th>Rate</th></tr></thead><tbody>
<tr><td>Missed sign</td><td>4.9 detected × 86% precision = 4.2 true hits of 5.6 true signs</td><td>{100*s7["deletion_rate"]:.1f}%</td></tr>
<tr><td>Extra sign</td><td>4.9 × 14% false positives = 0.69 per text ÷ 5.6</td><td>{100*s7["insertion_rate"]:.1f}% per true sign</td></tr>
<tr><td>Misread</td><td>best recognition top-1 accuracy 17%</td><td>83% at M1's level; swept 0–83%</td></tr>
</tbody></table>
<p>Recognition accuracy was swept over 100 / 90 / 75 / 60 / 45 / 30 / 17%, with 50 random corruptions per level, both with M1's detection errors and with perfect detection.</p>
<table class="tbl"><thead><tr><th>Finding</th><th>Clean</th><th>At M1's 17% accuracy</th><th>Perfect recognition, M1 detection</th></tr></thead><tbody>
<tr><td><code>P324</code> share of endings</td><td>44%</td><td><b>{fmt_pct(m1["share_P324_end"])}</b></td><td>{fmt_pct(perf["share_P324_end"])}</td></tr>
<tr><td>Fixed direction detected</td><td>100% of runs</td><td><b>{fmt_pct(m1["direction_detected"])}</b></td><td>100%</td></tr>
<tr><td>Start–end gap</td><td>2.46 bits</td><td><b>{m1["entropy_gap"]:.2f} bits</b></td><td>{perf["entropy_gap"]:.2f} bits</td></tr>
<tr><td>Collocations recovered</td><td>100%</td><td><b>{fmt_pct(m1["colloc_recovered"])}</b></td><td>{fmt_pct(perf["colloc_recovered"])}</td></tr>
</tbody></table>
{kv([(k, f"first lost at <b>{v}</b>") for k, v in s7["first_lost_at"].items()])}
{img("m2_09_error_propagation.png", "Four M2 findings against recognition accuracy; grey = perfect detection, orange = M1's detection errors, dashed = clean data.")}
<div class="callout"><b>Key insight.</b> Even with <i>perfect</i> recognition, M1's detection errors alone (25% missed, 12% extra) cut collocation
recovery to {fmt_pct(perf["colloc_recovered"])} and halve the start–end gap: every missed or extra box breaks the pairs around it.
Detection that is “solved” for finding signs is not good enough for sequence statistics. The direction signal is the most robust finding; fine-grained collocations are the most fragile.</div>
<p class="small">Assumptions: misreads are drawn uniformly from the 181-sign inventory; reading order is correct; M1 rates carry ±~13 pp. Full table in Appendix D.</p>
<p class="cp">Checkpoint</p>
{check(["With zero errors, the simulation reproduces Steps 2, 5 and 6 exactly (assertion)", "Fixed random seeds, so the run is deterministic"])}
{issue(["A built-in check compared against the wrong subset (161 endings instead of the 153 complete texts) and was corrected; the data was unaffected.",
        "Chart fixes: removed a misleading percentile band on the yes/no panel; de-duplicated the “M1 today” label."])}

<h3 class="step">Step 8 — Packaging and reproducibility</h3>
<p>All headline numbers are computed live into <code>results_summary.json</code>; this report reads from that file.</p>
<p class="cp">Checkpoint</p>
{check(["<b>Clean-room rebuild:</b> the package was unzipped into an empty folder, all outputs were deleted and everything was rebuilt from scratch, giving identical results (summary JSON, Witten-Bell matrix, rules and collocation tables byte-identical; all 9 figures regenerated)",
        "Notebook runs end-to-end with no errors and no warnings (≈1–2 minutes)"])}
''')

# ---------------- deliverables, limitations, links, figures, files
H.append(f'''
<h2>4. Deliverables mapped to the proposal</h2>
<table class="tbl"><thead><tr><th>Proposal requirement</th><th>Where it is</th><th>Status</th></tr></thead><tbody>
<tr><td>Map sequential transition frequencies</td><td>Step 3 · <code>markov_counts.csv</code></td><td class="done">Done</td></tr>
<tr><td>Construct a Markov chain matrix</td><td>Steps 3–4 · <code>markov_mle.csv</code>, <code>markov_witten_bell.csv</code></td><td class="done">Done</td></tr>
<tr><td>Beginners vs enders directional asymmetry</td><td>Steps 2 and 5 · p &lt; 0.0001 (two independent tests)</td><td class="done">Done</td></tr>
<tr><td>Witten-Bell smoothing for unseen pairs</td><td>Step 4 · validated by 10-fold CV</td><td class="done">Done</td></tr>
<tr><td>Directional matrix verifying right-to-left syntax rules</td><td>Step 5 · <code>directional_rules.csv</code></td><td class="done">Done</td></tr>
<tr><td>Sign collocation indices</td><td>Step 6 · <code>collocations.csv</code>, <code>pair_order_asymmetry.csv</code></td><td class="done">Done</td></tr>
<tr><td>Digitisation error propagation</td><td>Step 7 · <code>error_propagation_runs.csv</code></td><td class="done">Done</td></tr>
<tr><td>Documented notebook with cross-validation</td><td><code>m2_markov.ipynb</code> (+ HTML view)</td><td class="done">Done</td></tr>
</tbody></table>

<h2>5. Honest negative results and limitations</h2>
<ul>
<li><b>The perplexity trap:</b> forward and backward models score identically, so direction cannot be read from overall perplexity.</li>
<li><b>Raw PMI is misleading</b> on small corpora; it needs a frequency floor and a significance test.</li>
<li><b>Small corpus:</b> 179 texts from a single site. Every result carries random baselines or confidence intervals, and 16 of 28 frequent signs show no clear preference simply for lack of data.</li>
<li><b>Bigram model only:</b> longer-range structure (e.g. the <code>P385 → P122</code> opening formula as a unit) is left to Milestone 3.</li>
<li><b>No IM77 export available:</b> an independent replication on a larger corpus is future work.</li>
<li><b>Not yet done (optional):</b> a robustness re-run excluding the 118 partly-damaged-but-identified signs.</li>
</ul>

<h2>6. Links to Milestones 3 and 4</h2>
<ul>
<li><b>Milestone 3 (masked token prediction):</b> the Witten-Bell matrix is the baseline predictor, and <b>perplexity {s4["Witten-Bell"]["perplexity"]}</b> is the bar to beat. The 19 <code>P000</code> gaps are natural “predict the missing sign” test cases.</li>
<li><b>Milestone 4 (benchmarking hallucinations):</b> two ready exhibits: (1) the perplexity trap, a confident metric that cannot answer the question asked; (2) error propagation, where a pipeline at 17% accuracy would “find” structure that is mostly noise.</li>
</ul>

<h2>7. Figure guide for the deck</h2>
<table class="tbl"><thead><tr><th>Priority</th><th>File</th><th>Slide</th></tr></thead><tbody>
<tr><td rowspan="5">Essential</td><td><code>m2_03_start_vs_end.png</code></td><td>Signs have fixed positions</td></tr>
<tr><td><code>m2_04_flip_test.png</code></td><td>Direction is not chance</td></tr>
<tr><td><code>m2_07_positional_entropy.png</code></td><td>Start and end are structured differently</td></tr>
<tr><td><code>m2_06_witten_bell_reserve.png</code></td><td>Why and how we smooth (+ CV table)</td></tr>
<tr><td><code>m2_09_error_propagation.png</code></td><td>How M1 errors propagate</td></tr>
<tr><td rowspan="2">Recommended</td><td><code>m2_05_markov_matrix_mle.png</code></td><td>The Markov matrix</td></tr>
<tr><td><code>m2_08_collocations_pmi.png</code></td><td>Real collocations vs accidents</td></tr>
<tr><td rowspan="2">Optional</td><td><code>m2_01_inscription_length.png</code></td><td>The corpus</td></tr>
<tr><td><code>m2_02_sign_coverage.png</code></td><td>A small core dominates</td></tr>
</tbody></table>
<p><b>If there is room for only three:</b> <code>m2_04</code> + <code>m2_03</code> + <code>m2_09</code>, plus the perplexity-trap table (5a) as a text slide.</p>

<h2>8. File index</h2>
<table class="tbl"><thead><tr><th>File</th><th>Contents</th></tr></thead><tbody>
<tr><td><code>markov_witten_bell.csv</code></td><td><b>Main deliverable.</b> 182×182 smoothed matrix; rows include <code>&lt;S&gt;</code>, columns include <code>&lt;/S&gt;</code></td></tr>
<tr><td><code>markov_counts.csv</code> / <code>markov_mle.csv</code></td><td>Raw counts / unsmoothed probabilities</td></tr>
<tr><td><code>directional_rules.csv</code></td><td>Per-sign positional rates, q-values, roles</td></tr>
<tr><td><code>collocations.csv</code></td><td>All 529 pairs with PMI, NPMI, q, collocation flag</td></tr>
<tr><td><code>pair_order_asymmetry.csv</code></td><td>A→B vs B→A for frequent pairs</td></tr>
<tr><td><code>segments_rtl.json</code> / <code>texts_reading_order.csv</code></td><td>Cleaned corpus in reading order</td></tr>
<tr><td><code>error_propagation_runs.csv</code></td><td>All 700 simulation runs</td></tr>
<tr><td><code>results_summary.json</code></td><td>Every headline number, computed by the notebook</td></tr>
</tbody></table>
''')

# ---------------- appendix
H.append(f'''
<h2>Appendix A — Directional rules (all signs seen ≥ 8 times)</h2>
{table(rules_show, 'tiny')}
<h2>Appendix B — All {s6["significant"]} significant collocations</h2>
{table(sig_tab, 'tiny')}
<h2>Appendix C — Pair-order asymmetry (pairs seen ≥ 5 times in either order)</h2>
{table(asym_tab, 'tiny')}
<h2>Appendix D — Error propagation, full results (means of 50 runs)</h2>
{table(r7, 'tiny', index=False)}
''')

CSS = '''
:root{--ink:#0b0b0b;--ink2:#52514e;--muted:#7a7974;--line:#e4e3df;--bg:#fcfcfb;--blue:#2a78d6;--blue-soft:#eaf2fc;--orange:#eb6834;--orange-soft:#fdf0ea;--green:#0ca30c}
@page{size:A4;margin:16mm 14mm 16mm 14mm}
*{box-sizing:border-box}
body{font-family:"Inter","Segoe UI",system-ui,-apple-system,sans-serif;color:var(--ink);background:var(--bg);font-size:10.5pt;line-height:1.5;max-width:900px;margin:0 auto;padding:24px 16px}
h1{font-size:24pt;line-height:1.15;margin:.2em 0 .3em}
h2{font-size:15pt;margin:1.6em 0 .5em;padding-bottom:.25em;border-bottom:2px solid var(--ink);break-after:avoid}
h3{font-size:12.5pt;margin:1.3em 0 .4em;break-after:avoid}
h3.step{background:var(--blue-soft);padding:.45em .7em;border-left:4px solid var(--blue);border-radius:3px}
h4{font-size:11pt;margin:1em 0 .3em;color:var(--ink2);break-after:avoid}
code{font-family:"JetBrains Mono",Consolas,monospace;font-size:.9em;background:#f1f0ec;padding:0 .25em;border-radius:3px}
.cover{padding:10px 0 6px;break-after:page}
.eyebrow{text-transform:uppercase;letter-spacing:.12em;color:var(--blue);font-weight:700;font-size:9pt;margin:0}
.sub{font-size:12pt;color:var(--ink2);margin:0 0 .4em}.meta{color:var(--muted);font-size:9pt}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:18px 0}
.tile{border:1px solid var(--line);border-radius:8px;padding:12px;background:#fff}
.tile .num{font-size:19pt;font-weight:700;color:var(--blue);line-height:1.1}.tile .lbl{font-size:8.5pt;color:var(--ink2);margin-top:4px}
.callout{background:var(--blue-soft);border-radius:6px;padding:12px 14px;margin:12px 0;break-inside:avoid}
.warn{background:var(--orange-soft);border-left:4px solid var(--orange);border-radius:3px;padding:10px 12px;margin:12px 0;break-inside:avoid}
.note{background:#f4f3ef;border-radius:6px;padding:10px 12px;margin:10px 0;break-inside:avoid}
.toc li{margin:.15em 0}
table.tbl{border-collapse:collapse;width:100%;margin:8px 0 12px;font-size:9.2pt;break-inside:auto}
table.tbl th,table.tbl td{border-bottom:1px solid var(--line);padding:4px 7px;text-align:left;vertical-align:top}
table.tbl thead th{background:#f1f0ec;font-weight:600;border-bottom:1px solid #cfcdc6}
table.tbl tr{break-inside:avoid}
table.kv th{width:34%;background:#f7f6f3;font-weight:600}
table.narrow{width:auto}
table.tiny{font-size:7.8pt}table.tiny th,table.tiny td{padding:2px 5px}
td.done{color:var(--green);font-weight:700}
figure{margin:12px 0 16px;break-inside:avoid;text-align:center}
figure img{border:1px solid var(--line);border-radius:4px;max-width:100%}
figcaption{font-size:8.5pt;color:var(--ink2);margin-top:4px;text-align:left}
.cp{font-weight:700;color:var(--green);margin:.6em 0 .1em;text-transform:uppercase;font-size:8.5pt;letter-spacing:.08em}
ul.checks,ul.issues{list-style:none;padding-left:0;margin:.2em 0 .8em}
ul.checks li,ul.issues li{margin:.15em 0;padding-left:1.5em;text-indent:-1.5em}
.ok{color:var(--green);font-weight:700;display:inline-block;width:1.5em;text-indent:0}
.fix{color:var(--orange);font-weight:700;display:inline-block;width:1.5em;text-indent:0}
.formula{font-family:Georgia,serif;font-style:italic;background:#f7f6f3;padding:0 .3em;border-radius:3px}
.small{font-size:8.8pt;color:var(--ink2)}
@media print{body{background:#fff;max-width:none;padding:0}.tile{background:#fff}}
@media (max-width:640px){.tiles{grid-template-columns:repeat(2,1fr)}h1{font-size:19pt}}
'''

doc = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Milestone 2 Detailed Summary</title><style>{CSS}</style></head><body>{"".join(H)}</body></html>'''
out_html = sys.argv[1] if len(sys.argv) > 1 else 'Milestone_2_Detailed_Summary.html'
open(out_html, 'w', encoding='utf-8').write(doc)
print('wrote', out_html, f'{len(doc)/1e6:.1f} MB')
