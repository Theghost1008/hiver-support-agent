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
