#!/usr/bin/env bash
# Reproduce the project's results. Run from the repository root after following DATA_SOURCES.md.
#   bash run_all.sh            evaluation only (needs trained weights: M1 runs/*/best.pt, M3 artifacts/ckpt_seed*.pt)
#   bash run_all.sh --train    also (re)trains the M3 models, cross-validation and LLM2Vec runs (~30 min on a laptop CPU)
# M1 training is not scripted here; see src/m1_vision/README.md.
set -euo pipefail
TRAIN=${1:-}
step() { echo; echo "=== $* ==="; }

step "Shared: sign-numbering crosswalk (+ validation)"
python src/common/signs.py

step "M2: notebook on CISI, replication on the Wells corpus, Mahadevan Table I check"
( cd src/m2_markov && jupyter nbconvert --to notebook --execute --inplace m2_markov.ipynb --ExecutePreprocessor.timeout=1800 )
python src/m2_markov/replicate_wells.py
python src/m2_markov/mahadevan_table1.py

step "M3: masked-sign model"
if [[ "$TRAIN" == "--train" || ! -f src/m3_masking/artifacts/ckpt_seed0.pt ]]; then
  echo "training M3 (model search + 3 seeds)"; fi
( cd src/m3_masking && python run_all.py )          # trains only what is missing, else re-evaluates
if [[ "$TRAIN" == "--train" ]]; then python src/m3_masking/cv.py; fi

step "M2/M3 extras: word2vec and LLM2Vec comparison"
python src/m2_markov/word2vec.py
if [[ "$TRAIN" == "--train" ]]; then for s in 0 1 2; do python src/m2_markov/llm2vec.py --seed $s; done; fi
python src/m2_markov/llm2vec.py --table

step "M4: guardrails and experiments"
python src/m4_guardrails/controls.py
python src/m4_guardrails/perplexity_bench.py
python src/m4_guardrails/nb_guardrail.py
python src/m4_guardrails/synthetic_check.py
python src/m4_guardrails/f5_reading_test.py
python src/m4_guardrails/m1_outputs.py
python src/m4_guardrails/f6_error_propagation.py
python src/m4_guardrails/f7_guardrail_on_m1.py
python src/m4_guardrails/f8_generate.py

step "Engine"
python src/engine/make_synthetic_inputs.py --n 100
python src/engine/eval_synthetic.py
python src/engine/eval_real.py
python src/engine/generate_and_read.py

step "Notebooks (load saved results only)"
python notebooks/build_notebooks.py
for nb in notebooks/0*.ipynb; do jupyter nbconvert --to notebook --execute --inplace "$nb" --ExecutePreprocessor.timeout=600; done
echo; echo "done"
