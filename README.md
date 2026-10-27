# Retrieval Arena

**Does my news feed find the story a new article belongs to? I tested 8 retrieval setups on 470
real items from my own product to find out.**

[enable-ai](#where-the-data-comes-from) is my AI-engineering news product. Every new post, article,
or GitHub release has to be matched to an existing story or start a new one. Miss the match and
the feed shows the same news twice. This repo measures how well different retrieval methods
find the right story, how fast, and at what cost, so the choice is based on data, not vibes.

<!-- snapshot:start -->
**→ Latest results: [2026-10-27 snapshot](snapshots/2026-10-27_2053/)** (full tables, per-scenario breakdown, and the distance-cutoff sweep). All runs: [`snapshots/`](snapshots/).
<!-- snapshot:end -->

## What I found

1. **Production was creating duplicates.** Of the items that belonged to an existing story,
   **31% became a duplicate story** anyway.
2. **About half of that was retrieval.** The right story wasn't even among the candidates for
   **15%** of those items, so the model making the final call never had a chance.
3. **The cause was a cutoff that was too strict.** Production only considered stories within
   cosine distance 0.10. Loosening it to **0.20** finds the right story **97%** of the time,
   up from 79%.
4. **Any modern embedding model fixes it.** Every embedding model tested gets the right story into
   the top 5 about 97% of the time. OpenAI's `text-embedding-3-large` puts it *first* most often (93%).

| Retriever | Right story in top 5 | Right story first | Median latency | Cost per 1k items |
|---|---:|---:|---:|---:|
| OpenAI text-embedding-3-large | 97.3% | 93% | 396 ms | $0.025 |
| Qwen3 Embedding 8B | 97.3% | 90% | 688 ms | $0.002 |
| Gemini Embedding 2 | 97.3% | 83% | 610 ms | $0.037 |
| BM25 + Voyage 4 (hybrid) | 96.4% | 82% | 245 ms | $0.012 |
| Voyage 4 | 95.5% | 89% | 245 ms | $0.012 |
| BM25 (keyword search, no AI) | 94.6% | 78% | 1 ms | $0 |
| **Production, as logged** | **84.5%** | 81% | n/a | n/a |
| **Production's query, replayed** | **78.6%** | 79% | 610 ms | $0.037 |

*Measured on the 112 items that belong to an existing story. The gap between production and the
others is statistically significant (paired McNemar test, p < 0.001).*

## Is RAG dead?

That's the next run. The arena also includes **long context** (skip retrieval, give an LLM every
story at once) and an **agent** that searches on its own. It also includes rerankers and a
corpus-size sweep that shows when long context stops being worth it. They're configured and tested
but not yet run on the full data (about $4 of API credit).

| Part | Status |
|---|---|
| Dataset + labels (470 items) | ✅ Done |
| Retrieval quality: production, BM25, 4 embedding models, hybrid | ✅ Done |
| Production's real end-to-end accuracy (from its logs) | ✅ Done |
| Distance-cutoff sweep | ✅ Done |
| More embedding setups, rerankers, end to end through the decision model | ⏳ Next run |
| Long context vs. retrieval, agentic search, corpus-size sweep | ⏳ Next run |

## How it works

1. **Real data, replayed in order.** Each item searches only the stories that existed when it
   arrived (median 293 of 491), exactly as production saw them. No peeking at the future.
2. **Every method gets the same question:** given this item, which existing stories might it
   belong to? Most return a ranked top 5. Long-context and agentic methods give a final answer.
3. **End to end** (next run): the top 5 go to a decision model (`typesafe/jev-1.14`, the one
   enable-ai uses), which picks a story or says "new". That measures what actually ends up
   in the feed.
4. **Scored on quality, speed, and cost together,** with fixed rules that pick a recommendation
   ([`recommend.py`](src/retrieval_arena/recommend.py)). The same report always gives the same answer.

| Family | What's compared |
|---|---|
| Production | What enable-ai does today: the exact logged candidates, plus a re-implementation of its query (subject match **or** Gemini cosine distance ≤ 0.10, top 5) |
| Keyword | BM25 |
| Embeddings | Gemini Embedding 2 (three ways of choosing what text to embed), OpenAI 3 small/large, Voyage 4 / 4 lite, Qwen3 8B, BGE-M3 |
| Hybrid | BM25 + embeddings, fused with reciprocal rank fusion |
| Rerank | Hybrid top 25 re-ordered by Cohere Rerank 4 or Voyage Rerank 2.5 |
| Long context | No retrieval: every story in one prompt (Claude Haiku 5.5, GPT-6 Luna, Gemini 3.5 Flash-Lite) |
| Agentic | An LLM that runs up to 3 searches, then answers |

Every contender is one entry in [`retrievers.yaml`](retrievers.yaml).

## Run it yourself

```bash
uv sync
cp .env.example .env        # add OPENROUTER_API_KEY (one key covers every model)

uv run arena list           # contenders and dataset
uv run arena run            # everything enabled, end to end
uv run arena run --retrievers bm25,voyage-4 --limit 50   # a quick slice
uv run arena run --no-e2e   # retrieval metrics only
uv run arena scale --retrievers lc-haiku-5.5,hybrid-voyage --sizes 50,150,all
uv run arena snapshot       # save the latest run under snapshots/
uv run pytest               # offline tests, no key needed
```

Every API response is cached on disk, so rerunning, or adding one new retriever, only pays for
what's new. Each run also writes an HTML report with charts and a per-item grid.

## Where the data comes from

470 items and 491 stories were exported read-only from enable-ai's production database. The
export script stays private because it encodes the product's schema.

- **Stories** ([`corpus.jsonl`](datasets/story_dedup/corpus.jsonl)) as they looked when first
  created. Their later AI-written summaries are left out because they were built from the very items
  being tested, which would leak the answer.
- **Items** ([`queries.raw.jsonl`](datasets/story_dedup/queries.raw.jsonl)): every item that went
  through story matching, with what production retrieved and decided.
- Only public material: titles, subjects, URLs, and short excerpts of public posts, articles,
  and releases. Ids are opaque handles, and nothing is user-specific.

## How the labels were made

Production's own decisions can't be the answer key, because the point is to find where
production is wrong. So:

1. **Pool candidates.** For each item, collect the top 8 stories from five different retrievers,
   plus everything production retrieved.
2. **Label blind.** A model picks the matching stories using one written
   [grouping policy](datasets/story_dedup/task.yaml), without seeing production's decision.
   Claude Sonnet 5.5 labeled 274 items through the API. Gemini CLI labeled the other 196 from the
   same candidate lists.
3. **Cross-check with production.** Where the labeler and production agree, that's the label (398
   items). Where they disagree (72 items), a model from a different vendor reads the case and
   decides, with a one-line reason in [`adjudications.yaml`](datasets/story_dedup/adjudications.yaml).
   Decisions were spot-checked by hand.

The same policy is given to every LLM in the arena, so everything is graded against one rule.

## Caveats

- **Labels come from models, not humans.** They were cross-checked and disputes were reviewed, but
  where the labeler and production made the *same* mistake, the label is wrong too. That bias
  favors production, so its numbers here are, if anything, generous.
- **Pooling blind spot.** A correct story that none of the pooled retrievers found was never
  labeled. Production's own candidates were always included.
- **Replay is approximate.** Production represents a story by the average of all its items so
  far. Here it's the first item. The *logged* row is the exact production record.
- **One workload.** These conclusions are about this feed, which is the point. With 112 matching
  items, treat gaps of a couple of points as noise. The significance tests are there for that.
- Latency is the round trip from one laptop through OpenRouter, so it includes network time.

## Related

[decision-arena](https://github.com/SeijiMatsumoto/decision-arena) compares the models that make
the final "same story or not" call. This repo is about whether the right story reaches them.
