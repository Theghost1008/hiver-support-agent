# Decision Log

A running list of non-obvious decisions made during this project, and the reasoning behind them.

---

### 1. Technology stack selection
**Decision:** Python 3.11+, pandas for data handling, Groq API with `openai/gpt-oss-120b` for LLM calls, `sentence-transformers` (local) for embeddings, plain numpy cosine similarity instead of a vector DB, pytest for testing. No GitHub Actions/CI.

**Reasoning:**
- Groq + `gpt-oss-120b` chosen over paid APIs (Claude/OpenAI) for zero cost and fast inference. Tradeoff accepted: free-tier rate limits and slightly less mature structured-output reliability than top-tier paid models.
- Embeddings run locally via `sentence-transformers` rather than through an LLM API — free, fast, and keeps API calls reserved for generation/judging, where free-tier rate limits matter most.
- Plain numpy cosine similarity instead of FAISS/Chroma — a full vector DB is unnecessary infrastructure at our subsample scale (hundreds–low thousands of messages); avoiding overengineering.
- Skipped GitHub Actions/CI entirely — not required by the assignment; would trade time on the eval harness/report (what's actually graded) for infrastructure polish nobody asked for.

---

### 2. Brand selection
**Decision:** Built the agent for **AppleSupport** (5,511 brand-reply messages in a 200k-row subsample of the full dataset).

**Reasoning:**
- Considered `AmazonHelp` (15,323 messages, highest volume) but rejected it — Amazon's catalog spans electronics, groceries, subscriptions, and delivery, making a clean, tight intent taxonomy harder to define and defend.
- Considered airline accounts (Delta, AmericanAir, etc., ~2,000 messages each) — naturally clean intents (delays, baggage, refunds) but lower volume.
- Chose AppleSupport as the balance: solid volume (enough for a meaningful retrieval index and golden set), a coherent tech/device support domain, and personal familiarity that helps with manual labeling quality.

---

### 3. Grounding scope decision
**Decision:** Defined "grounded reply" as mimicking AppleSupport's real *public-facing* response pattern for a given intent (including appropriately redirecting to DM when that matches historical behavior) — not fabricating a deep technical resolution we have no evidence for.

**Reasoning:**
- Manual inspection of (customer message → Apple reply) pairs showed that most real Apple replies redirect to DM ("let's go to DM", "we've received your DM") rather than publicly resolving the issue — actual fixes happen in private DMs, which aren't in this dataset.
- Treating public triage replies as if they were full resolutions would misrepresent what the data actually supports.
- This is a deliberate, stated scope boundary — documented explicitly in the report's problem-framing section — not an oversight discovered later.
- Flagged as a likely source of a misleading headline metric: high textual similarity to retrieved historical replies could look strong while mostly reflecting a generic "let's DM" template, not genuine reply quality.

---

### 4. Intent taxonomy definition
**Decision:** Defined 9 intent categories for AppleSupport (see src/intents/taxonomy.py)
by manually reading 60 randomly sampled customer messages and grouping recurring
patterns, rather than starting from a predefined taxonomy like Banking77's 77 labels.

**Reasoning:**
- Banking77's labels are domain-specific to banking and too fine-grained for a
  general tech-support brand; not directly applicable.
- Initial pass produced an overloaded "software issue" bucket covering ~55% of
  sampled messages — split into update_installation, battery_performance,
  bug_report, account_security, and icloud_sync once sub-patterns became clear.
- support_escalation kept as its own category rather than folded elsewhere,
  since "customer already contacted before and still unresolved" is a strong,
  distinct signal for the later escalation policy.
- other_unclear retained as a deliberate catch-all rather than forcing edge
  cases into an ill-fitting bucket.

---

### 5. Taxonomy validation and refinement against full 60-row sample
**Decision:** Validated the 9-category taxonomy against the full 60-row labeled
sample (not just the initial 20). Added a 10th category, `billing_payment`, after
finding a payment-info request that didn't fit any existing bucket. Expanded
`account_security`'s description to include phishing/fraud reports.

**Reasoning:**
- `bug_report` emerged as the largest bucket (18/60, 30%) but was not split
  further — with only 60 total examples, subdividing risks creating sub-categories
  with too few examples to be meaningful. Revisit only if this imbalance persists
  at larger scale.
- `icloud_sync` and `billing_payment` each have exactly 1 example in this sample —
  noted as a real limitation: cannot distinguish "genuinely rare for this brand"
  from "under-sampled" with only 60 rows. Kept as distinct categories rather than
  merged, since they are clearly real and distinct intents, not noise.
- Correction to earlier finding: initial 20-row sample suggested DM-redirect was
  near-universal; the full 60-row sample shows a more balanced 55%/45% split.
  Documented as a caution against over-generalizing from small samples — directly
  relevant to sizing the golden eval set at 150-250, not fewer.

---

### 6. Reasoning-model token budget and SDK version fix
**Decision:** Set max_tokens=300 (up from an initial 20) and reasoning_effort="low"
on all classification calls to gpt-oss-120b. Upgraded the groq SDK from 0.11.0 to
support the reasoning_effort parameter.

**Reasoning:**
- gpt-oss-120b is a reasoning model that spends completion tokens on internal
  chain-of-thought before emitting a final answer. With max_tokens=20, all tokens
  were consumed by reasoning, causing content to silently return as an empty
  string ("") rather than an error — masked by our own defensive fallback logic.
- reasoning_effort="low" chosen deliberately, not just to fix the bug: intent
  classification into a small known label set doesn't benefit from deep multi-step
  reasoning, so low effort is both correct for the task and reduces cost/latency.
- The reasoning_effort parameter required upgrading the groq SDK, since 0.11.0
  predated support for gpt-oss reasoning controls.

---

### 7. Classifier evaluation reveals context-dependency limitation
**Decision:** Ran classify_intent against 50 held-out labeled examples (80% accuracy,
before baseline comparison). Manual review of the 10 mismatches identified 4 distinct
failure patterns rather than uniform "model error":
1. Mid-thread reply fragments lacking standalone context (6/10 mismatches) — customer
   replies answering Apple's clarifying question, uninterpretable in isolation without
   the preceding thread.
2. Genuinely overlapping taxonomy boundaries (e.g. bug_report vs support_escalation
   for "still got activation issues" — both are defensible readings).
3. Non-English text (one French message) — outside the system's intended scope.
4. Possible ground-truth labeling error in our own manual label (one case), not
   necessarily a model failure.

**Implication:** Raw accuracy alone (80%) understates real performance, since a
meaningful fraction of "errors" reflect data/scope limitations rather than
classifier weakness. This is directly relevant to the report's mandatory
"misleading headline number" section.

---

### 8. Baseline design flaw and NN-baseline data-starvation finding
**Decision:** Corrected trivial baseline to fit on the full 60-row labeled sample
(real class distribution) instead of the 10-example few-shot set, after noticing
it degenerated to a single always-predicted label due to an artificial 10-way tie.

**Reasoning:**
- Fitting the trivial baseline on a deliberately balanced 10-example set defeats
  its purpose, since it relies on exploiting real class imbalance.
- Post-fix result (34%) exactly matches the true label distribution of the
  held-out set (17/50 bug_report), confirming the fix.

**Finding:** The NN (simple) baseline, fit on the same 10 examples as the LLM
classifier, scored 24% — lower than the corrected trivial baseline (34%).
Hypothesis: with only one example per category, nearest-neighbor matching
suffers from "anchor bias" — a single example's incidental wording can absorb
predictions for unrelated messages (observed: nn=battery_performance predicted
far more often than that intent's real frequency). This suggests semantic
similarity methods need meaningfully more than one example per class to be
reliable, a genuinely relevant caveat for the report given time/scope
constraints prevented building a larger labeled set before this milestone.

---

### 9. NN baseline data-scaling experiment
**Decision:** Ran a side experiment refitting the NN baseline on 45 examples
(vs. the original 10) using a fresh, leakage-free 15-row eval split, to test
whether the earlier NN underperformance (24%, below trivial's 34%) was due to
data starvation.

**Finding:** NN accuracy improved to 33.33% with more fitting data — directionally
supporting the anchor-bias hypothesis — but still did not clearly exceed the
trivial baseline. Given the small eval size (15 rows), this difference is within
plausible sampling noise and should not be overstated as "NN never beats trivial."
Honest conclusion: more data measurably helps NN, but at this scale neither
baseline clearly outperforms simple majority-class guessing on this task —
reinforcing that the LLM classifier's 80% is a substantial, non-trivial
improvement over both, not merely edging past a weak floor.

---

### 10. Pair completeness is a structural data property, not a subsampling artifact
**Initial hypothesis:** ~18-20% of customer messages lack a matched Apple reply
due to row-order truncation from sequential subsampling (first N rows only).

**Test:** Increased subsample from 200k to 500k rows (2.5x). If truncation were
the cause, pair-completeness should measurably improve.

**Result:** Completeness stayed essentially flat — 81.6% (4,386/5,372) at 200k
rows vs. 80.3% (12,616/15,705) at 500k rows. Hypothesis rejected.

**Revised conclusion:** The ~20% of customer messages without a paired Apple
reply likely reflects real structural properties of the dataset — messages that
genuinely went unanswered publicly, or reply-chain links pointing to other
customer tweets rather than a brand reply — not a subsampling boundary effect.
Accepted 12,616 pairs as the retrieval base going forward; further increasing
subsample size is not expected to meaningfully change this ratio, so not pursued
further.

---

