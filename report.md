# Report: AI Support Agent for AppleSupport

## 1. Problem framing

**What "good" means for this brand.** AppleSupport's public Twitter replies are
overwhelmingly *triage*, not resolution: across the historical data, most real
replies acknowledge the issue, ask a clarifying question, and redirect to DM —
actual fixes happen privately, off-platform. Given that, "good" for this system
means: correctly classify what the customer needs, draft a reply consistent
with Apple's real public triage style (not a fabricated deep resolution the
brand itself doesn't post publicly), and route anything sensitive, severe, or
poorly-grounded to a human rather than guess.

**What I chose *not* to build.** No attempt to reconstruct or predict actual
resolution content (refund amounts, specific fixes) since that data isn't
observable in public tweets — doing so would mean the system inventing
specifics it has no basis for, which is a hallucination risk, not a feature.
No fine-tuning (data and time insufficient to do it well). No vector database
infrastructure (FAISS/Chroma) — at ~12,600 messages, plain numpy cosine
similarity is sufficient and avoids unnecessary infrastructure. No CI/CD — not
required by the assignment, and would trade report/evaluation time for
polish nobody asked for.

## 2. Results vs. baselines

### Intent classification (full 175-row golden set, independently hand-labeled)
| Approach | Accuracy |
|---|---|
| Trivial (always predict most common label) | 17.14% |
| Nearest-neighbor (embedding similarity) | 21.71% |
| **LLM classifier (main system)** | **71.43%** |

The main system is a 3.3–4.2x improvement over both required baselines.

### Reply quality (full 175-row golden set, LLM-as-judge)
| | Strict PASS rate | Partial credit |
|---|---|---|
| Initial prompt | 8.00% | 37.76% |
| Improved prompt (after failure analysis) | **14.29%** | **47.02%** |

Judge validated against a 30-row blind human-labeled sample: raw agreement
93.33%, Cohen's kappa 0.474 (moderate agreement, correcting for class
imbalance — see §4).

### Escalation policy (should_escalate ground truth)
| | Before severity rules | After severity rules |
|---|---|---|
| Accuracy | 62.86% | 66.29% |
| Recall (catching true escalations) | 53.25% | **64.94%** |
| Precision | 58.57% | 60.98% |

## 3. Top failure modes (with real examples)

**1. Lost conversational context.** The system sees one isolated tweet, never
prior thread turns. Real example: a customer replies *"Not on beta, downloaded
the GM when it went live overnight"* — direct context from a prior exchange —
and the generated reply says only *"Please DM us and we'll take a look,"*
ignoring what was just said. This same root cause degrades both intent
classification (short reply-fragments like *"The newest one"* are
uninterpretable alone) and reply quality. **This is the single largest lever
for improvement** (see §5).

**2. Overlapping taxonomy boundaries.** E.g. *"Thanks for the lead, but still
got activation issues"* is defensibly either `bug_report` or
`support_escalation` — both readings are reasonable, and the taxonomy doesn't
fully disambiguate.

**3. Partial-criteria replies.** Generated replies often address one part of
a multi-part request and stop (e.g. asks for device model, never gets to
troubleshooting). Improved by 40% relative (partial credit 37.76%→47.02%) with
prompt changes, but not eliminated.

**4. Non-English / out-of-scope text.** A French message was misclassified;
the system and taxonomy are English-scoped. Now caught by an escalation rule,
but the classifier/generator themselves have no multilingual capability.

**5. Requested specifics the system has no reliable source for.** Some
criteria expect named internal teams or exact policy details that appear in
real historical replies only inconsistently, and the system is explicitly
instructed never to invent them (to avoid the fabricated-link failure mode
found and fixed earlier). This is a direct, deliberate consequence of the
grounding-scope decision in §1, not an oversight.

## 4. What is misleading about my headline number?

Several real numbers in this project would be misleading if reported alone,
without the context that follows:

- **71.43% classifier accuracy looks worse than an earlier 80% figure** —
  but the 80% came from just 50 rows and a slightly different classification
  mechanism (batch vs. the real single-message production function). The
  71.43% figure, from the full 175-row independently-labeled golden set using
  the actual production function, is the trustworthy one. Smaller or
  mismatched samples can mislead in *either* direction.
- **8.00% strict reply-quality PASS rate sounds close to total failure** —
  but replies satisfy 37.76% of their criteria elements *on average*, not
  zero. An all-or-nothing rubric on multi-part criteria makes "partially
  good" replies look identical to "completely bad" ones unless a partial-
  credit metric is reported alongside.
- **A "smarter" retrieval baseline scored *worse* than a dumb one early on**
  (24% NN vs. 34% trivial) — not because similarity retrieval is a bad idea,
  but because it was fit on only 10 examples (one per class), causing
  unreliable "anchor bias." With more fitting data it improved (33.33%). A
  single baseline comparison, under-resourced, can make a legitimately useful
  technique look worse than it is.
- **Retrieval similarity scores can't be trusted as "relevance" in isolation**
  — a deliberately off-topic calibration query still scored 0.41–0.44
  similarity purely from incidental word overlap in a large, short-text
  corpus, close to genuine matches (0.6–0.76). Similarity alone is not a
  clean confidence signal; the escalation policy deliberately does not rely
  on it exclusively.
- **The escalation rules were tuned on the same golden set used to evaluate
  them** (failure analysis on false negatives directly informed the new
  severity rules, then re-evaluated on the same 175 rows). The reported
  64.94% recall is real, but likely somewhat optimistic relative to true
  generalization — a held-out validation set would be needed to know by how
  much.
- **~20% of customer messages have no matched Apple reply in the data** —
  initially assumed to be a subsampling artifact; testing at 2.5x more data
  showed the same ~20% gap, revealing it's a structural property of the
  dataset (unanswered messages, non-brand reply chains), not something more
  data would fix.

## 5. What I'd do with one more week

1. **Wire in prior conversation thread context** (highest priority — the
   dataset's reply-chain IDs support reconstructing multi-turn threads; this
   is the common root cause behind failure modes #1 and #3).
2. **Validate escalation rules on a held-out set** distinct from the one used
   to design them, to get an honest generalization estimate rather than the
   likely-optimistic current number.
3. **A second human labeler** for a subset of the golden set, to measure
   inter-annotator agreement and quantify how much of the reported numbers
   reflect one person's subjective judgment.
4. Formalize `order_shipping` as an 11th taxonomy category (found during
   golden-set labeling, currently folded into `billing_payment` for lack of
   sufficient evidence to justify a full taxonomy revision mid-project).
5. Replace regex-based severity/tone detection with a lightweight trained
   classifier, reducing false positives/negatives from keyword matching.
6. Remove the duplicate `retrieve_similar` call in `pipeline.py` (currently
   called once for escalation grounding, once again inside `generate_reply`).

---

*Full reasoning behind every decision, including several corrected mistakes
found through direct testing rather than assumed to work, is in
`decision_log.md`.*