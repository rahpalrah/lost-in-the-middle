# Contract-review benchmark

An end-to-end test of the whole system on a real business workflow.

## Scenario

A buyer's **contract-review desk** uses a contract-management system that calls the
longdoc HTTP API. A SaaS vendor has sent its contract pack:

- **Master Services Agreement**, 22 articles, with **Exhibit A (Data Processing
  Agreement)** and **Exhibit B (Order Form)** attached, as one 53-page PDF.
- An unrelated **NDA** with another counterparty, to check that documents never leak
  into each other's answers.

An analyst asks 16 questions a reviewer would ask before signing or renewing:

| Type | What it tests |
|---|---|
| Lookups early, mid and late in the document | Basic accuracy and page citations |
| The termination-for-convenience clause, buried mid-document | The "lost in the middle" failure |
| A near-duplicate clause (vendor vs customer termination) | Picking the right clause, not the first match |
| Conflicts between the MSA and its exhibits | Applying the order-of-precedence clause (DPA: 48 h breach notice overrides the MSA's 72 h; Order Form: net 45 overrides the MSA's net 30) |
| Questions answered only by the exhibits | Coverage of attachments |
| A question the contracts don't answer | Saying "not covered" instead of inventing an answer |
| A question to the NDA | Document isolation |

The documents are **synthetic but realistic**: generated deterministically by
`generate_pack.py`, with every fact planted exactly once, so the correct answer and
its page are known.

## What it measures

Each question runs three ways:

| Strategy | Path |
|---|---|
| `retrieval` | longdoc parent-child retrieval, via the HTTP API |
| `cached_document` | longdoc whole-document prompt caching, via the HTTP API |
| `naive` | Baseline anti-pattern: the full extracted PDF text pasted into every prompt, no caching |

For each answer: **correct** (all required facts present, nothing forbidden), **citation
correct** (cites the section that holds the fact), **retrieval recall** (the right
section was sent to the model), **latency**, **tokens**, **cache hit ratio** and **cost**.

## Run it

```bash
pip install -e ".[dev]"

# Offline plumbing check: fake model, costs nothing. Answers are placeholders.
python benchmarks/contract_review/run.py --offline

# Live: real Claude calls. Stops before spending more than --max-cost (US$).
export ANTHROPIC_API_KEY=...
python benchmarks/contract_review/run.py --live --max-cost 5
```

Results are written to `benchmarks/contract_review/results/<timestamp>-<mode>/`
(`report.md` and `results.json`).

**Expected live cost:** about $4–5 with `claude-opus-5-5` at `medium` effort, most of
it the naive baseline (16 × ~53k tokens). Run `--strategies retrieval,cached_document`
to skip the baseline (about $1).

## Results

### Local model, no API key (Phi-3.5-mini, 4-core CPU, 2026-11-23)

| Retrieval | Correct | Right section retrieved | Median time per answer |
|---|---|---|---|
| BM25 keyword | 14/16 (88%) | 13/15 | 46 s |
| Hybrid: BM25 + bge-small embeddings | **15/16 (94%)** | 14/15 | 54 s |

- Hybrid search fixed Q10 ("how much are we paying per year" -> "Annual Subscription
  Fee"). Q13 ("the vendor is acquired" -> "Change of Control") still fails: neither
  keyword nor embedding search ranks that section in the top 100 for this wording. The
  model correctly said the sections it received did not answer the question.
- Citations were correct in only 67% of hybrid-run answers (80% keyword-only). Reading every answer found problems the scoring missed: wrong or missing citations
  (Q05, Q06, Q08, Q11, Q13), echoed prompt markup (Q03, Q09, Q11), a misquote (Q12).
  The SDK now strips echoed markup and accepts bare section IDs (0.5.0), but small
  models still need human review of citations and quotes.
- **Whole-document strategies (`cached_document`, `naive`) did not finish.** The
  contract is 45k tokens for this model. With a 48k context and a compressed KV cache it
  loads (12 GB of 15 GB) but the first question had not produced an answer after 90
  minutes on 4 cores, so the run was stopped. On a CPU this size, only parent-child
  retrieval is practical; whole-document prompting needs a GPU or a hosted model.
- Not measured: Claude (needs an API key), and the whole-document accuracy comparison.

Reproduce: `scripts/download_models.sh`, then
`python benchmarks/contract_review/run.py --local models/Phi-3.5-mini-instruct-Q4_K_M.gguf --strategies retrieval --embed-model models/bge-small-en-v1.5-q8_0.gguf`.

### Defects the benchmark found in the SDK (fixed in 0.3.0 - 0.5.0)

1. No section headings detected in PDFs; running headers and page numbers repeated in
   every chunk.
2. Keyword search missed paraphrases: stemming and heading indexing took retrieval
   recall from 67% to 87%; adding embeddings reached 93% (14/15).
3. Models echo `<section ...>` tags and write bare IDs instead of `[p0007]`.
