# Hiver Support Agent — AppleSupport (Work in Progress)

**Status: In progress.** This README reflects what is actually built and tested
so far, not the final assignment deliverable. See "Not Yet Implemented" below
for what remains.

## What this is

An AI customer-support agent for **AppleSupport**, built from the
[Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset. Goal: classify customer message intent, draft a reply grounded in
Apple's real historical resolutions, and decide whether to auto-handle or
escalate to a human — plus prove, with real evaluation, that the system can
be trusted.

## What's built and working

### 1. Data pipeline
- `src/data_prep/download.py` — downloads the Kaggle dataset, saves a
  500,000-row subsample to `data/raw/`.
- `src/data_prep/clean.py` — filters to AppleSupport conversations, reconstructs
  (customer message → Apple reply) pairs via self-join on tweet reply-chain IDs.
  Produces `data/processed/apple_pairs.csv` (12,616 pairs from 15,705 customer
  messages — ~80% pair-completeness; see Known Limitations).

### 2. Intent taxonomy
- `src/intents/taxonomy.py` — 10 categories, derived from manually reading and
  labeling a 60-row random sample of real customer messages:
  `update_installation`, `battery_performance`, `bug_report`, `account_security`,
  `icloud_sync`, `feature_howto`, `hardware_issue`, `support_escalation`,
  `billing_payment`, `other_unclear`.

### 3. Intent classifier
- `src/intents/classifier.py` — few-shot LLM classification via Groq
  (`openai/gpt-oss-120b`), using one real labeled example per category plus
  taxonomy descriptions.
- **Result on a 50-row held-out set** (excluding few-shot examples):
  - Trivial baseline (always predict most common label): **34.00%**
  - Nearest-neighbor baseline (embedding similarity, same 10 examples): **24.00%**
  - LLM classifier (main system): **80.00%**
- A follow-up experiment showed the NN baseline improves to 33.33% with more
  fitting examples (45 vs 10) — directionally supporting a "data-starved
  anchor bias" explanation for its earlier weak result, though still not
  clearly ahead of the trivial baseline at this sample size.

### 4. Retrieval index
- `src/retrieval/index_builder.py` — embeds all 12,616 historical customer
  messages (`sentence-transformers`, `all-MiniLM-L6-v2`), saves to
  `data/processed/retrieval_index.npz`.
- `src/retrieval/retriever.py` — given a new message, returns the top-k most
  similar historical (customer message, Apple reply) pairs via cosine
  similarity (typical top-match similarity: 0.6–0.7 for genuinely related
  messages, once an indexing bug that was silently deflating scores was found
  and fixed — see `decision_log.md` entry 11).

## Not yet implemented

- Reply generation (grounded in retrieved historical replies)
- Escalation policy (auto-handle vs. escalate + reason)
- Golden evaluation set (150–250 hand-labeled examples)
- Evaluation harness (automated metrics + LLM-as-judge + judge-human agreement)
- Final report and full decision log writeup
- End-to-end pipeline wiring (`src/pipeline.py`)

## Setup (current state)

```bash
git clone <this-repo>
cd hiver-support-agent
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash; use .venv/bin/activate on Mac/Linux

pip install -r requirements.txt
```

Create `.env` in project root:
GROQ_API_KEY=<you_api_key_here>


Kaggle API credentials required at `~/.kaggle/kaggle.json` — see
[Kaggle API docs](https://www.kaggle.com/docs/api).

Run the pipeline built so far, in order:
```bash
python src/data_prep/download.py
python -m src.data_prep.clean
python -m src.retrieval.index_builder
python -m src.retrieval.retriever            # demo: retrieve similar historical cases
python -m eval.run_classifier_sanity_check   # ~2 min, calls Groq API
python -m eval.run_baseline_comparison
```

## Known limitations (so far)

- ~20% of customer messages have no matched Apple reply in the data —
  appears to be a structural property of the dataset (unanswered messages,
  non-brand reply chains), not a subsampling artifact (confirmed by testing
  at 2.5x subsample size with no meaningful change in ratio).
- Historical Apple replies are dominated by public-triage behavior
  ("let's go to DM") rather than full resolutions, since actual fixes often
  happen in private DMs not present in this dataset. "Grounded reply" is
  scoped accordingly — see `decision_log.md` entry 3.
- Taxonomy validated against only 60 examples; two categories
  (`icloud_sync`, `billing_payment`) have just 1 example each in that sample.

See `decision_log.md` for full reasoning behind every non-obvious decision made.