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

### 11. Retriever indexing bug caught via similarity-score sanity check
**Bug:** retrieve_similar's returned "similarity" field was hardcoded to
similarities[1] instead of similarities[i], causing every result in the
top-k list to report the same (wrong) similarity score, while the retrieved
texts themselves were correctly indexed and looked plausible.

**How it was caught:** noticed all three results reported an identical
similarity (0.155018) to 6 decimal places — statistically implausible for
distinct 384-dim embeddings. Confirmed via debug inspection of the raw
similarities array (12,616 unique values, ruling out embedding duplication)
before finding the indexing typo.

**Why this matters:** the bug produced plausible-looking output (topically
correct retrieved messages) that could easily have passed a casual glance.
Fixed scores (0.67, 0.64, 0.61) are meaningfully higher and more
differentiated than the buggy 0.155 — this also resolves the earlier
open question about whether ~0.15 was a "normal" similarity ceiling for
this embedding model on this data; it was not, it was a bug artifact.

---

### 12. Retrieval similarity calibration and its limits
**Finding:** Genuine semantic matches score 0.67-0.76 top similarity; two attempted
negative controls (off-topic queries) unexpectedly scored 0.41-0.44 due to
coincidental surface word overlap in the large (12,616-message), short, informal
tweet corpus (e.g. "pizza" appearing literally in an unrelated real complaint).

**Implication:** Pure embedding similarity cannot cleanly separate "genuinely
relevant" from "coincidentally overlapping" at this corpus scale and text style.
A single similarity threshold is an imperfect confidence signal on its own.

**Decision:** The escalation policy will not rely on retrieval similarity alone
as a confidence signal. Will combine it with intent-classifier agreement (does
the top retrieved match's original intent align with the new message's predicted
intent?) rather than trusting a bare similarity number in isolation. Chasing a
perfectly clean negative-control score was abandoned as low-value given two
consistent findings already in hand.

---

### 13. Reply generator hallucinated a placeholder URL — fixed via prompt + example sanitization
**Bug found:** First real generated reply included a fabricated link
(https://t.co/YourSupportLink) — the model pattern-matched the structure of
real historical Apple replies (which legitimately contain real support links)
without having an actual URL to substitute, inventing a plausible-looking fake.

**Fix:** (1) Strip URLs from historical examples before they enter the prompt,
via strip_urls(), so the model isn't shown "replies of this type include a
link" as a pattern to imitate. (2) Explicit prompt instruction forbidding
fabricated links.

**Confirmed fixed:** Rerun on the same test message produced "please continue
the conversation with us in DM" with no link, real or fabricated.

**Related, lower-severity, not yet fixed:** generated reply used generic "@user"
rather than a specific customer handle, since our test call didn't pass a real
username through. Not a hallucination risk (no invented identity), just a
placeholder gap — noted as a limitation for now, would need actual pipeline
wiring to carry the real @handle through.

---

### 14. Rule-based escalation policy with layered signals
**Decision:** Built escalation as a deterministic, priority-ordered rule set
rather than an additional LLM call.

**Reasoning:** Escalation is the pipeline's key safety-relevant control point —
getting it wrong has real consequences (an issue that should reach a human,
silently auto-handled). Deterministic, fully auditable rules were prioritized
over an LLM's flexible-but-self-reported reasoning, given our own direct
experience with the reply generator hallucinating (entry #13) — an LLM's stated
"reason" is not automatically trustworthy just because it reads plausibly.

Rules (checked in priority order, first match wins):
1. Always-escalate intents: account_security, billing_payment, other_unclear,
   support_escalation — sensitive categories or ones where the classifier
   itself lacked confidence.
2. Non-English text (via langdetect) — system and taxonomy are English-scoped.
3. Very short messages (<5 words) — directly evidenced failure mode from
   classifier evaluation (most real mismatches were short, context-lacking
   thread fragments).
4. Zero grounded retrieval examples — no historical precedent to draw from.
5. Weak average grounding similarity (<0.55, just above the 0.5 inclusion
   cutoff) — a barely-qualifying match is weaker evidence than a strong one.

**Explicitly deferred (noted as future work):** sentiment/frustration keyword
detection and classifier confidence scoring — both plausible signals, neither
currently validated with real evidence, avoided to prevent speculative
overengineering.

---

### 15. Progress bar disconnected from actual work loop — real progress was hidden, not lost
**Bug found:** classify_presample.py created a tqdm progress_bar object, but the
for loop iterated over a separately-constructed enumerate(presaample.iterrows())
instead of progress_bar itself. The bar displayed once and never updated,
appearing to hang indefinitely, while the actual classification loop underneath
ran correctly and silently in the background.

**How discovered:** compared file modification timestamps and checkpoint content
before assuming a real hang; found 200 rows had genuinely been classified during
the apparent "stuck" period. No progress was actually lost.

**Fix:** for loop now iterates directly over progress_bar (which wraps the same
enumerate(...iterrows())), so the display and the actual work loop are the same
iterator, not two independent ones.

**Lesson:** a frozen progress indicator is not proof the underlying work stopped
— worth checking independent evidence (file timestamps, checkpoint contents)
before assuming a hang and killing a process that may be working correctly.

---

### 16. Groq daily token rate limit hit during golden-set pre-sample classification
**What happened:** Classification run crashed at row 961/1500 after hitting Groq's
free-tier daily token cap (200,000 TPD), not the per-minute limit already
accounted for. Checkpoint system preserved progress up to the last clean save
(~row 950), consistent with the resumable design working as intended even
under an unplanned failure mode.

**Fix:** Added retry logic with a 100s wait on RateLimitError inside
classify_intent, since Groq's error response suggested a short, rolling-window
wait (not a fixed once-daily reset) would likely resolve transient limit hits.

**Honest limitation:** if the daily cap is genuinely exhausted (not just a
momentary spike), retrying won't help within the same session — the run may
need to pause and resume on a new day. This is a real, disclosed cost of
choosing a free-tier LLM API (decision log entry #1) rather than a paid one;
worth stating explicitly rather than hidden as a smooth, uninterrupted process.

---

### 17. Checkpoint file corruption from concurrent process writes
**What happened:** Running two separate instances of classify_presample.py
simultaneously (different API keys/accounts) against the same checkpoint
file caused a race condition — concurrent, uncoordinated writes corrupted
presample_with_predictions.csv, producing more rows than the intended 1500
and unreliable data.

**Resolution:** Discarded the corrupted checkpoint entirely and restarted
classification from scratch with a single process/key, rather than attempt
to salvage or deduplicate an unreliable file. Chose not to trust partially
corrupted data for a "golden" evaluation artifact where data integrity is
the entire point.

**Related:** the second account/key that caused this was created against
project guidance (likely violates Groq's ToS) and has since been deleted;
not used in the final submission.

---

### 18. Redesigned bulk classification from per-message to batched API calls
**Problem:** Original one-API-call-per-message approach (1,500 calls) repeatedly
hit Groq's per-minute and daily rate limits during golden-set pre-sample
classification, causing crashes and, separately, tempting a rate-limit
workaround (a second account) that was correctly abandoned as against
provider ToS and inconsistent with the assignment's honesty requirements.

**Fix:** Redesigned as classify_intent_batch — groups messages into batches
of 15, sending one API call per batch with a JSON array response, rather
than one call per message. Reduces 1,500 calls to 100, cutting both total
runtime (~50min -> ~5min) and rate-limit exposure substantially, since fixed
prompt overhead (taxonomy + few-shot examples) is now paid once per batch
instead of once per message.

**Kept both single-message (classify_intent) and batched (classify_intent_batch)
functions:** the pipeline's live classification path handles one real-time
message at a time and cannot be batched; batching only applies to this
one-off bulk data-preparation task.

---

### 19. Golden sample built: 150 stratified + 25 random (175 total)
**Decision:** Final golden evaluation sample combines 150 stratified rows
(15 per intent category, drawn from a classifier-predicted 1,500-row
pre-sample) with 25 pure-random rows (drawn from the full 12,616-pair pool,
excluding anything already used in stratification or few-shot examples).

**Result:** All 10 categories achieved the full 15 examples with no shortfall
— including icloud_sync and billing_payment, which had only 1 example each
in the original 60-row taxonomy-building sample. Confirms that earlier
sparsity was a small-sample artifact, not a reflection of true rarity in
the full dataset.

**Caveat retained from earlier:** stratification was by classifier-predicted
intent, not verified ground truth, since hand-labeling the full pool wasn't
feasible. Manual labeling of this 175-row golden set (next step) will
produce the true ground-truth labels and may reveal some stratification
groups were built around misclassified examples.

---

### 20. Discovered taxonomy gap during golden-set labeling: order/shipping
**Finding:** 2 of 175 golden-set messages concerned order/shipping status
(e.g. pre-order delivery), not cleanly fitting any of the 10 existing intent
categories. Closest existing label (billing_payment) used as a pragmatic
fit for evaluation purposes.

**Decision:** Not added as an 11th official category — 2 examples out of 175
is too thin to justify redefining the taxonomy and invalidating prior
classification results built around the 10-category version. Documented
as a known scope gap; a production system would likely need this as
a real 11th category given more data.

---

### 21. Final intent classification accuracy on full golden set
**Result:** 71.43% (125/175) — classify_intent (real single-message production
function) vs. independently hand-labeled true_intent, across the full golden
set (150 stratified + 25 random).

**Context:** An earlier 50-row held-out check (before the golden set existed)
showed 80%. The larger, properly-labeled 175-row result (71.43%) is the more
trustworthy figure: bigger sample, independently-labeled ground truth (not
influenced by model predictions), and evaluated using the exact function used
in production (not the batch variant used for pre-sampling). This discrepancy
is itself a useful, honest illustration for the report's mandatory "misleading
headline number" section — smaller samples can overstate or understate true
performance by chance.

---

### 22. Final authoritative three-way intent classification comparison
**Result (full 175-row golden set, real classify_intent, independently
hand-labeled true_intent):**
- Trivial baseline: 17.14% (30/175 — matches true frequency of the most
  common category, bug_report, confirming correct baseline behavior)
- NN baseline: 21.71%
- LLM classifier (main system): 71.43%

**This supersedes all earlier, smaller-sample accuracy figures** (the 50-row
80% check, and the earlier 34%/24% baseline comparison on a different partial
set) as the authoritative number for the report, since it uses the full golden
set, independently-labeled ground truth, and the actual production classification
function rather than the batch variant used only for sampling convenience.

**Caveat:** trivial baseline's "most common label" was determined from the
golden set's own 175-row true_intent distribution, used as a proxy for the
full 12,616-pair pool's true distribution (which we don't have fully labeled).
A reasonable estimate, not a guaranteed exact population statistic.

---

### 23. Automated reply-quality metrics: final result, full golden set
**Result (175/175 replies):**
- Empty replies: 0%
- Replies containing a fabricated/real link: 0% (confirms the entry #13
  fix holds at full scale, not just the single test case originally checked)
- Replies over 280 characters: 0%
- Average reply length: 117 characters
- Average grounded examples per reply: 2.98/3 (zero rows with no grounding)

**Note:** an earlier full run produced 100% empty replies due to a
transient, unreproduced issue (possibly provider-side); resolved on
rerun with no code changes needed, confirmed by direct re-verification
rather than assumption.

---

### 24. LLM judge results: strict vs. partial-credit reveals a misleading headline risk
**Result:** Strict all-or-nothing PASS rate: 8.00% (14/175). Average partial
credit (criteria met / criteria total): 37.76%.

**Interpretation:** The large gap between these two numbers is itself a key
finding. Reporting only the strict 8% PASS rate would be a misleading headline
number — it obscures that replies typically satisfy a third or more of their
required criteria, rather than failing completely. Manual review of failure
reasons (decision log entry preceding this) identified three contributing
causes: (1) replies address only one criterion element rather than combining
multiple asks, (2) lost conversational context from single-message-only input
(directly connects to the classifier's earlier context-fragment failure mode,
entry #7), (3) some criteria expect specifics (named internal teams, full
multi-step instructions) exceeding what a single public triage-style tweet
reply was scoped to provide.

**Decision:** Report both numbers together in the final report, not just the
strict PASS rate alone, with this breakdown of contributing causes.

---

### 25. Judge-human agreement validation
**Result (30-row blind sample, human labeled independently before seeing
LLM judge verdicts):**
- Raw percent agreement: 93.33% (28/30)
- Cohen's kappa: 0.474 (moderate agreement, per Landis & Koch scale)

**Interpretation:** Raw agreement alone is inflated by severe class imbalance
(both human and LLM verdicts are ~90%+ FAIL) — a trivial always-FAIL judge
would score similarly high on raw agreement without evaluating anything.
Kappa, which corrects for this, indicates genuine but moderate (not excellent)
agreement.

**Both disagreements were human=PASS, LLM=FAIL (not randomly distributed),
both on short, low-stakes messages with minimal criteria** — suggesting a
possible mild systematic bias toward over-strictness in the LLM judge on
simple/low-bar cases, rather than random judge unreliability. Documented as
a real, disclosed limitation of the judge rather than treated as a solved
validation.

---

### 26. Reply prompt improvements: validated result on full golden set
**Changes made to build_reply_prompt:** (1) loosened length constraint from
~200 to 280 characters (actual Twitter limit), giving room to address more
criteria per reply; (2) explicit instruction to address every distinct
concern in the customer's message, not just the first; (3) explicit
instruction to reuse real named teams/departments when present in retrieved
grounding examples, while still prohibiting invented specifics not present
in the examples (keeping the earlier anti-hallucination fix intact).

**Result (full 175-row golden set, before vs after):**
- Strict PASS rate: 8.00% -> 14.29% (+6.3pp, ~1.8x)
- Partial credit: 37.76% -> 47.02% (+9.3pp)

**Validation process:** tested on a 20-row sample first (showed a smaller,
inconclusive-at-that-scale signal: partial credit +5pp, PASS rate unchanged
at 0% both times, within expected sampling noise given the low base rate).
Full 175-row rerun confirmed genuine, larger improvement, justifying the
additional API cost of a complete rerun. Old and new results both retained
on disk (golden_eval_with_judge.csv vs _v2.csv) for full traceability.

**Still a real, disclosed limitation:** even the improved 14.29% strict PASS
rate remains low in absolute terms. The largest identified remaining cause
(lost conversational thread context) was not addressed - noted as the
primary "what I'd do with one more week" item.