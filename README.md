# Hiver Support Agent — AppleSupport (Work in Progress)

**Status: In progress.** This README reflects what is actually built and tested
so far, not the final assignment deliverable. See "Not Yet Implemented" below
for what remains.

## What this is

An AI customer-support agent for **AppleSupport**, built from the
[Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset. Classifies customer message intent, drafts a reply grounded in
Apple's real historical resolutions, and decides whether to auto-handle or
escalate to a human — plus proves, with real evaluation, that the system
can be trusted.

## What's built and working

### 1. Data pipeline
- `src/data_prep/download.py` — downloads the Kaggle dataset, saves a
  500,000-row subsample to `data/raw/`.
- `src/data_prep/clean.py` — filters to AppleSupport conversations, reconstructs
  (customer message → Apple reply) pairs via self-join. Produces
  `data/processed/apple_pairs.csv` (12,616 pairs; ~80% pair-completeness —
  see Known Limitations).

### 2. Intent taxonomy
- `src/intents/taxonomy.py` — 10 categories, derived from manually labeling a
  60-row sample and validated against a 175-row golden set: `update_installation`,
  `battery_performance`, `bug_report`, `account_security`, `icloud_sync`,
  `feature_howto`, `hardware_issue`, `support_escalation`, `billing_payment`,
  `other_unclear`.

### 3. Intent classifier
- `src/intents/classifier.py` — few-shot LLM classification via Groq
  (`openai/gpt-oss-120b`).
- **Final authoritative result** (full 175-row golden set, independently
  hand-labeled ground truth, real production `classify_intent` function):
  - Trivial baseline: **17.14%**
  - Nearest-neighbor baseline: **21.71%**
  - LLM classifier (main system): **71.43%** — a 3.3–4.2x improvement over
    both baselines.

### 4. Retrieval index
- `src/retrieval/index_builder.py` / `retriever.py` — embeds all 12,616
  historical customer messages (`sentence-transformers`, `all-MiniLM-L6-v2`),
  retrieves top-k similar historical (customer, reply) pairs via cosine
  similarity. Typical top-match similarity for genuine matches: 0.6–0.76.

### 5. Reply generator
- `src/reply/generator.py` — drafts a reply grounded in retrieved historical
  examples (similarity ≥ 0.5 threshold), explicitly instructed not to
  fabricate links (see Known Limitations for the bug this fixes).
- **Automated metrics (full 175-row golden set):** 0% empty replies, 0%
  containing a fabricated/real link, 0% over 280 characters, avg length
  117 characters, avg 2.98 grounded examples per reply.
- **LLM-as-judge results:** strict PASS rate (reply must meet *all* elements
  of a human-written quality checklist) **8.00%**; average partial credit
  (fraction of checklist elements met) **37.76%**. The gap between these two
  numbers is itself a key finding — see Known Limitations.

### 6. Escalation policy
- `src/routing/escalation.py` — deterministic, priority-ordered rule set
  (not an LLM call, by design — see `decision_log.md` entry 14): always-escalate
  intents (sensitive/unclear categories), non-English detection, short/low-context
  messages, weak or absent retrieval grounding. Returns a decision plus a
  specific, traceable reason.

### 7. Golden evaluation set
- `data/golden_eval_with_judge.csv` — 175 rows (150 stratified across all 10
  intents + 25 random), independently hand-labeled for true intent, escalation
  judgment, and reply-quality criteria.

## Not yet implemented

- Judge-human agreement validation (manually score a subset, compare to LLM
  judge — the assignment's explicit "how well does your judge agree with a
  human" requirement)
- Escalation policy evaluation against golden-set ground truth
- End-to-end pipeline wiring (`src/pipeline.py`)
- Final written report (problem framing, full failure analysis, "what's
  misleading about my headline number" section)
- Full decision log compilation (25 entries currently tracked across
  development; needs final consolidation into `decision_log.md`)

## Setup

```bash
git clone <this-repo>
cd hiver-support-agent
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash; use .venv/bin/activate on Mac/Linux

pip install -r requirements.txt
```

Create `.env` in project root:
GROQ_API_KEY=<your_api_key_here>


Kaggle API credentials required at `~/.kaggle/kaggle.json` — see
[Kaggle API docs](https://www.kaggle.com/docs/api).

Run the pipeline, in order:
```bash
python src/data_prep/download.py
python -m src.data_prep.clean
python -m src.retrieval.index_builder
python -m eval.build_golden_sample          # builds the 175-row golden set
python -m eval.reclassify_golden_set        # ~8 min, calls Groq API
python -m eval.generate_golden_replies      # ~10 min, calls Groq API
python -m eval.compute_automated_metrics
python -m eval.run_judge                    # ~10 min, calls Groq API
```

**Note on API usage:** the full evaluation pipeline makes several hundred
Groq API calls. Groq's free tier has a 200,000 tokens/day cap; a full run
of all steps above may require spanning more than one day, or reducing
sample sizes. See `decision_log.md` for the retry/checkpoint patterns used
to handle this gracefully.

## Known limitations

- ~20% of customer messages have no matched Apple reply — a structural
  property of the dataset (unanswered messages, non-brand reply chains),
  not a subsampling artifact.
- Historical Apple replies favor public-triage behavior ("let's go to DM")
  over full resolutions, since actual fixes often happen in private DMs not
  present in this dataset. "Grounded reply" is scoped accordingly.
- The reply generator initially hallucinated a placeholder URL by
  pattern-matching historical examples' link structure; fixed via URL
  stripping and explicit prompt instruction — confirmed fixed at full scale
  (0% of 175 golden replies contain a link).
- LLM judge results show a large gap between strict PASS rate (8.00%) and
  partial credit (37.76%), driven by three identified causes: replies
  addressing only one part of multi-part criteria, lost conversational
  context from single-message-only input (no prior thread turns visible to
  the system), and some criteria expecting specifics (named internal teams,
  full instructions) beyond what a single public triage-style reply was
  scoped to provide.
- Both baselines and the LLM judge's reliability depend on evaluation design
  choices documented and, in several cases, corrected mid-project after
  discovering bugs (see `decision_log.md` entries 8, 11, 13) — full
  transparency on these corrections is part of this project's approach to
  the assignment's "prove it's good enough to trust" requirement.

See `decision_log.md` for full reasoning behind every non-obvious decision made.