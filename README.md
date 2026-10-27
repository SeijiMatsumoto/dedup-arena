# Retrieval Arena

Is RAG dead? Depends on the task, so I'm measuring it on a real one.

This repo compares retrieval strategies (BM25, ten embedding setups, hybrid fusion, rerankers,
"just put everything in the context window", and an agent with a search tool) on **470 real items
of dedup traffic from [enable-ai](#where-the-data-comes-from)**, my AI-engineering news product.
Each one is scored on retrieval quality, end-to-end accuracy, latency, and cost.

The first finding came from scoring production itself: **a third of the items that belonged to an
existing story became duplicate stories**, and about half of those never reached the matching model
because a cosine-distance cutoff was too tight. Any of the off-the-shelf embedding setups here
retrieves the right story far more often.

The task: a new post, article, or GitHub release arrives. Does it cover a story we already have
(attach it), or is it a new development (start a story)? Get it wrong one way and the feed shows
duplicates. Get it wrong the other way and unrelated news gets merged.

It's the retrieval half of [decision-arena](../decision-arena), which compares the models that
make the final call. Here, the question is whether the right story even reaches that model.

<!-- snapshot:start -->
## Latest results

**[2026-10-27 snapshot](snapshots/2026-10-27_2053/)**: Use **hybrid-voyage** (recall@5 96%, right story first 82%, 245ms median, $0.0118 per 1k). Production today (its logged decisions): 90.9% correct end to end; 31% of items that belong to an existing story became a duplicate story, and 2% of all items were merged into the wrong story. Production's logged candidates contained the right story for 84.5% of matching items, so 15% never reached the judge. openai-3-large reaches 97.3% (paired McNemar p=0.0005).

Full tables, scenario breakdowns and the distance-cutoff sweep are in the snapshot; all runs are listed in [`snapshots/`](snapshots/).
<!-- snapshot:end -->

## Status

| Part | State |
|---|---|
| Dataset + labels (470 items) | Done |
| Retrieval metrics: production (logged + replay), BM25, Gemini, OpenAI-large, Voyage, Qwen, hybrid-voyage | Done (in the snapshot) |
| Production's own end-to-end accuracy, from its logs | Done |
| Distance-cutoff sweep | Done |
| Remaining embedding setups, rerankers, end-to-end through Jev | Next run (~$1.50 of API credit) |
| Long context, agentic search, corpus-size sweep: the "is RAG dead?" part | Next run (~$2.50) |

## Setup

```bash
uv sync
cp .env.example .env   # add OPENROUTER_API_KEY
```

Every model (embeddings, rerankers, LLMs, the Jev decision model) goes through OpenRouter, so one
key covers everything.

```bash
uv run arena list                          # retrievers + datasets
uv run arena run                           # every enabled retriever, full end-to-end
uv run arena run --retrievers bm25,voyage-4,hybrid-gemini --limit 50
uv run arena run --no-e2e                  # retrieval metrics only, no judge calls
uv run arena run --misses                  # print items most retrievers got wrong
uv run arena scale --retrievers lc-haiku-5.5,hybrid+voyage-rerank-2.5 --sizes 50,150,all
uv run arena show                          # re-print the latest report
uv run arena snapshot                      # save latest run (+ scale run) under snapshots/
uv run arena run --retrievers mock-good,mock-weak,bm25 --no-e2e   # offline, no key needed
```

Each run writes `reports/run-<timestamp>.json` plus an `.html` page with the scatter plots, the
cutoff sweep and a per-item grid. Every API response is cached in `.cache/`, keyed on the exact
request, so rerunning or adding one retriever only pays for what's new.

## What gets compared (`retrievers.yaml`)

| Family | Retrievers | What it tests |
|---|---|---|
| Production | `production-logged`, `production-replay` | What enable-ai does today. *Logged* is the exact candidate list production retrieved, read from its logs. *Replay* re-implements the query: subject identity **or** Gemini cosine distance ≤ 0.10, identity first, version guard, top 5. |
| Lexical | `bm25` | The 1990s baseline everyone should beat. |
| Dense | `gemini-emb-2` (+ `-full`, `-meta`), `openai-3-small/large`, `voyage-4`, `voyage-4-lite`, `qwen3-embed-8b`, `bge-m3` | Embedding model choice, and *what text to embed* (raw item vs. extracted subject/version vs. both). |
| Hybrid | `hybrid-gemini`, `hybrid-voyage` | BM25 + dense fused with reciprocal rank fusion. |
| Rerank | `hybrid+cohere-4-fast/pro`, `hybrid+voyage-rerank-2.5` | Hybrid top 25 reordered by a cross-encoder. |
| Long context | `lc-haiku-5.5`, `lc-gpt-6-luna`, `lc-gemini-3.5-flash-lite` | **No retrieval.** Every eligible story goes into one prompt and the LLM picks. The "RAG is dead" position. |
| Agentic | `agent-haiku-5.5` | An LLM that calls a hybrid search tool up to 3 times, then answers. |

Adding a contender is one YAML entry. Composites refer to other entries by name.

## How it's scored

**Time travel.** Each item only searches the stories that existed when it arrived (median ~290
of 491), and never the story it created itself. No retriever gets to see the future.

**Retrieval metrics**, on items that belong to an existing story:
- **Recall@k**: is at least one right story in the top k? Recall@5 matters most, because
  production passes 5 candidates on to the judge.
- **MRR, nDCG@10**: how high the right story ranks.

**End to end**, on every item: the top 5 go to a decision model (`typesafe/jev-1.14`, the same
one enable-ai uses) with the grouping policy. It picks a story, or none. Long-context and
agentic retrievers make that call themselves.
- **Accuracy**, split into **attach** and **new**.
- **Dupes**: items that belong to a story but became a new one.
- **Wrong merge**: items attached to a story that doesn't cover them.
- **Latency** (p50/p95, retrieval + judge, client round trip) and **cost per 1k items**.

**`production-actual`** scores the decisions enable-ai actually made, from its logs, against the
same labels. That's the bar to beat.

**Cutoff sweep.** For each dense retriever: recall@5 and the number of candidates passed on, as a
function of the cosine-distance cutoff. This answers whether production's 0.10 is too tight.

**Corpus size** (`arena scale`). The same queries rerun with each item's eligible stories cut to N
(the right ones always kept). Shows how long-context accuracy, latency, and cost move as the
corpus grows, compared with retrieval.

## Takeaways

`recommend.py` turns a report into picks with fixed rules and no LLM:

1. **Contenders:** within 2 points of the best end-to-end accuracy, and not significantly worse by a
   paired exact McNemar test (p < 0.05).
2. **Use:** the contender with the best combined cost + latency rank.
3. **Avoid:** significantly worse than the best, or had failed calls.
4. **Notes:** production today vs. the best, what the 0.10 cutoff costs, the smallest cutoff that
   keeps full recall, long context vs. retrieval (accuracy, speed, and cost ratios with a
   significance test), the best embedding setup, reranker gains, and weak scenario tags.

`production-logged` is never a pick: it's a log replay, not something you can deploy.

## Where the data comes from

The dataset was exported read-only from enable-ai's production database (the export script is kept
private because it encodes the product's schema):

- **Corpus (`corpus.jsonl`)**: every Story, represented by its *discovery* item as it was when the
  Story was created. The Story's later AI summary is deliberately not used: it was written from
  evidence that includes the queries themselves, which would leak the answer.
- **Queries (`queries.raw.jsonl`)**: every item that went through story matching and was applied
  (attached, or started a new story), with the candidates production retrieved and what it decided.

Only public material is exported: titles, extracted subjects/versions, URLs, and a ≤700-char excerpt
of each public post, article, or release. Ids are short opaque handles, not database keys. Nothing
user-specific.

## Labels

Production's own decisions can't be the ground truth, because the point is to find where
production is wrong. Labels are built TREC-style:

1. **Pool:** for each item, the union of the top 8 from five diverse retrievers (BM25, Gemini,
   OpenAI, Voyage, Qwen embeddings) plus every candidate production retrieved and the story it
   chose. The pool is listed chronologically, so the judge can't see any retriever's ranking.
2. **Labeler:** a model picks the matching stories from the pool by one written grouping policy
   ([`task.yaml`](datasets/story_dedup/task.yaml)). The policy restates the rules enable-ai's matcher
   follows in plain terms. The same policy goes to the decision model and the LLM retrievers, so
   everything is graded against one rule. The labeler never sees production's decision.
   - 274 items: Claude Sonnet 5.5 through the API (`arena label`; raw verdicts in `judgments.jsonl`).
   - 196 items: Gemini CLI working from the same pools (`arena label --sheet` writes them as Markdown;
     answers in `agent/agent_labels.yaml`). Same policy, no API spend.
3. **Second opinion:** what production actually decided. Where the labeler and production agree,
   that's the label (`judge+production`, 398 items). Where they disagree, which is exactly where the
   interesting errors are, the case goes to [`review.md`](datasets/story_dedup/review.md) and a model
   from another vendor decides it with a one-line note in
   [`adjudications.yaml`](datasets/story_dedup/adjudications.yaml) (`reviewed`, 72 items: Gemini CLI
   reviewed Sonnet's 43, Codex reviewed Gemini's 29). Decisions were spot-checked by hand.

Reviewers didn't just pick a side by default: Gemini sided with Sonnet 31 times and with production 11 times. Codex
leaned harder toward its labeler (25 vs. 2), so treat the second batch's review as lighter.

Commands: `uv run arena label` (API judge, ≈$0.012 per item, cached), `uv run arena label --sheet`
(pools for a coding agent, free), `uv run arena label --merge-only` (rebuild `queries.jsonl` from
saved verdicts and adjudications, free).

## Caveats

- **Labels are model judgments,** checked against production's decisions, with every disagreement
  reviewed by a second model. They are not independent human annotation, and two different
  labelers were used (see [Labels](#labels)). Where the labeler and production make the *same*
  mistake, the label is wrong too, and that error favors production, so production's
  numbers here are, if anything, flattering. Spot-check `queries.jsonl`.
- **Pooling bias:** a right story that no pooled retriever ranked in its top 8 was never judged.
  Production's candidates are always pooled, so this can't hide a story production *found*. It
  could hide one everybody missed.
- **Replay is approximate.** Production embeds a story as the centroid of all its items so far.
  Here a story is its discovery item. `production-logged` is the exact record; `production-replay`
  shows the query logic with today's embedding API.
- **One task, one corpus.** ~470 items and ~490 stories from one product's AI-news feed. The
  conclusions are about this workload, which is the point. Treat gaps of a couple of points as noise
  (the McNemar test is there for that).
- **Small corpus for long context.** A few hundred stories fit easily in a 1M-token window. The
  `scale` run shows the trend, but this repo can't test a million-document corpus.
- Latency is the client-side round trip from one machine through OpenRouter, so it includes network
  time and OpenRouter's routing.

## Project layout

```
datasets/story_dedup/   task.yaml (policy, pool, judges), corpus.jsonl, queries.jsonl (labeled),
                        judgments.jsonl, agent/ (agent labels), adjudications.yaml
retrievers.yaml         every contender
src/retrieval_arena/
  retrievers.py         all retriever kinds behind search(query, eligible) -> ranked ids
  label.py              pooling, API judge, coding-agent sheets, adjudication merge
  e2e.py                top 5 -> decision model -> attach / new
  metrics.py            recall@k, MRR, nDCG, cutoff sweep, end-to-end, McNemar
  recommend.py          deterministic picks and notes
  runner.py, cli.py, snapshot.py, html_report.py
snapshots/              dated, committed results
```
