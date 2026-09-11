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

