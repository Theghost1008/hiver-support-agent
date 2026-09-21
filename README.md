# AI Support Agent for AppleSupport

## What this is

An AI customer-support agent for **AppleSupport**, built from the
[Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset. Classifies customer message intent, drafts a reply grounded in
Apple's real historical resolutions, and decides whether to auto-handle or
escalate to a human — plus proves, with real evaluation, that the system
can be trusted.

See `report.md` for problem framing, results vs. baselines, failure analysis,
and the "what's misleading about my headline number" section. See
`decision_log.md` for the reasoning behind every non-obvious decision made
across development, including several real bugs found and fixed along the way.

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
- **Final result** (full 175-row golden set, independently hand-labeled
  ground truth, real production `classify_intent` function):
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
  fabricate links or unverified specifics.
- **Automated metrics (full 175-row golden set):** 0% empty replies, 0%
  containing a fabricated/real link, 0% over 280 characters, avg length
  117 characters, avg 2.98 grounded examples per reply.
- **LLM-as-judge results, after evidence-driven prompt improvements:**
  strict PASS rate **14.29%** (up from 8.00%), average partial credit
  (fraction of checklist elements met) **47.02%** (up from 37.76%).
- **Judge validated against human judgment:** 30-row blind sample, 93.33%
  raw agreement, Cohen's kappa 0.474 (moderate agreement, correcting for
  class imbalance).

### 6. Escalation policy
- `src/routing/escalation.py` — deterministic, priority-ordered rule set
  (not an LLM call, by design — see `decision_log.md`): severity/safety
  signals (checked first), always-escalate intents, non-English detection,
  short/low-context messages, weak or absent retrieval grounding. Returns a
  decision plus a specific, traceable reason.
- **Result on golden-set ground truth:** accuracy 66.29%, recall on true
  escalations 64.94% (up from an initial 53.25% before adding
  severity/tone-based rules found through direct failure analysis),
  precision 60.98%.

### 7. Golden evaluation set
- `data/golden_eval_with_escalation.csv` — 175 rows (150 stratified across
  all 10 intents + 25 random), independently hand-labeled for true intent,
  escalation judgment, and reply-quality criteria.

### 8. End-to-end pipeline
- `src/pipeline.py` — `process_message()`, the single function wiring
  classification, retrieval, reply generation, and escalation into the real
  production flow. Verified end-to-end, including a safety-critical example
  correctly triggering escalation.

## Setup

```bash
git clone <this-repo>
cd hiver-support-agent
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash; use .venv/bin/activate on Mac/Linux

pip install -r requirements.txt
```

Create `.env` in project root:

```bash
GROQ_API_KEY=<your_api_key_here>
```

Kaggle API credentials required at `~/.kaggle/kaggle.json` — see
[Kaggle API docs](https://www.kaggle.com/docs/api).

**Quick demo:**
```bash
python -m src.pipeline
```

**Full reproduction, in order:**
```bash
python src/data_prep/download.py
python -m src.data_prep.clean
python -m src.retrieval.index_builder
python -m eval.build_golden_sample
python -m eval.reclassify_golden_set        # ~8 min, calls Groq API
python -m eval.generate_golden_replies      # ~10 min, calls Groq API
python -m eval.compute_automated_metrics
python -m eval.run_judge                    # ~10 min, calls Groq API
python -m eval.evaluate_escalation
```

**Note on API usage:** the full evaluation pipeline makes several hundred
Groq API calls. Groq's free tier has a 200,000 tokens/day cap; a full run
of all steps above may require spanning more than one day. All long-running
scripts checkpoint and resume safely if interrupted — see `decision_log.md`
for the retry/checkpoint patterns used.

## Known limitations

- ~20% of customer messages have no matched Apple reply — a structural
  property of the dataset (unanswered messages, non-brand reply chains),
  not a subsampling artifact (confirmed by testing at 2.5x subsample size
  with no meaningful change in ratio).
- Historical Apple replies favor public-triage behavior ("let's go to DM")
  over full resolutions, since actual fixes often happen in private DMs not
  present in this dataset. "Grounded reply" is scoped accordingly.
- The system only ever sees one isolated customer message, never prior
  thread context — the single largest identified driver of remaining
  failure modes in both classification and reply generation (see `report.md`).
- Escalation rules were evaluated on the same golden set used to design them
  (failure analysis on false negatives directly informed the added rules);
  reported recall is likely somewhat optimistic relative to true
  generalization on unseen data.
- Several bugs were found and fixed mid-project through direct testing
  rather than assumed to work correctly — including a retrieval-indexing
  bug that silently deflated similarity scores, and a reply-generation
  hallucination of a placeholder URL. Full transparency on these
  corrections is part of this project's approach to the assignment's
  "prove it's good enough to trust" requirement.

See `decision_log.md` for full reasoning behind every non-obvious decision made.