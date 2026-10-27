# Results snapshot: 2026-10-27 20:53 UTC

## story_dedup

470 real items from enable-ai (112 that belong to an existing story, 358 that start a new one), searched against the stories that existed when each item arrived (median 293 of 491). Labels: 398 judge+production, 72 reviewed (see [Labels](../../README.md#how-the-labels-were-made)). This run is retrieval-only (no judge calls); `production-actual`'s end-to-end numbers come from production's own logs. Latency is the client round trip from one machine. Raw data: [`report.json`](report.json).

| Retriever | Family | Recall@1 | Recall@5 | MRR | End-to-end | Attach | New | Dupes | Wrong merge | p50 | $ per 1k |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| openai-3-large | dense | 93% | 97.3% | 0.95 | — | — | — | — | — | 396ms | $0.0247 |
| qwen3-embed-8b | dense | 90% | 97.3% | 0.93 | — | — | — | — | — | 688ms | $0.0022 |
| gemini-emb-2 | dense | 83% | 97.3% | 0.89 | — | — | — | — | — | 610ms | $0.0368 |
| hybrid-voyage | hybrid | 82% | 96.4% | 0.89 | — | — | — | — | — | 245ms | $0.0118 |
| voyage-4 | dense | 89% | 95.5% | 0.92 | — | — | — | — | — | 245ms | $0.0118 |
| bm25 | lexical | 78% | 94.6% | 0.85 | — | — | — | — | — | 1ms | $0.0000 |
| production-logged | production | 81% | 84.5% | 0.83 | — | — | — | — | — | — | — |
| production-replay | production | 79% | 78.6% | 0.79 | — | — | — | — | — | 610ms | $0.0368 |
| **production-actual** | production | — | — | — | 90.9% | 67% | 98% | 31% | 2% | — | — |

Recall is measured on the items that belong to an existing story. *Dupes*: items that belong to a story but became a new one. *Wrong merge*: items attached to a story that doesn't cover them. `production-actual` is what enable-ai actually decided (from its logs), scored against the same labels. End-to-end columns are empty for retrievers this run did not send to the judge.

### Takeaways ([how these are derived](../../README.md#how-it-works))

- **Use: hybrid-voyage.** recall@5 96.4%, right story first 82% of the time (MRR 0.89), 245ms median, $0.012 per 1k items.
- **Most accurate: openai-3-large.** recall@5 97.3%, right story first 93% of the time (MRR 0.95), 396ms median, $0.025 per 1k items.
- **Cheapest: qwen3-embed-8b.** recall@5 97.3%, right story first 90% of the time (MRR 0.93), 688ms median, $0.002 per 1k items.
- **Avoid: production-replay.** recall@5 78.6% vs 97.3% for openai-3-large (paired McNemar p=0.000).
- Production today (its logged decisions): 90.9% correct end to end; 31% of items that belong to an existing story became a duplicate story, and 2% of all items were merged into the wrong story.
- Production's logged candidates contained the right story for 84.5% of matching items, so 15% never reached the judge. openai-3-large reaches 97.3% (paired McNemar p=0.0005).
- Production retrieval is weakest on: cross-source (70%, n=20), subject-drift (58%, n=38).
- The 0.10 cosine cutoff: production's candidate query (re-implemented) reaches recall@5 79%; the same embeddings without the cutoff reach 97%.
- Smallest cutoff that keeps gemini-emb-2's full recall@5: distance ≤ 0.20, passing 4.2 candidates per item on average (vs 0.6 at 0.10).
- Best embedding setup: openai-3-large (recall@5 97%, recall@1 93%, MRR 0.95).

### Distance cutoff sweep (gemini-emb-2, production's embedding text)

Production uses 0.10. Candidates are capped at 5, like production.

| Max cosine distance | Recall@5 | Avg candidates passed on | New items with no candidate |
|---:|---:|---:|---:|
| 0.05 | 16% | 0.1 | 97% |
| 0.10 | 52% | 0.6 | 88% |
| 0.15 | 85% | 1.6 | 52% |
| 0.20 | 97% | 4.2 | 6% |
| 0.25 | 97% | 4.9 | 1% |
| 0.30 | 97% | 5.0 | 0% |
| 0.35 | 97% | 5.0 | 0% |
| 0.40 | 97% | 5.0 | 0% |
| 0.45 | 97% | 5.0 | 0% |
| 0.50 | 97% | 5.0 | 0% |
| 0.55 | 97% | 5.0 | 0% |
| 0.60 | 97% | 5.0 | 0% |

### Recall@5 by scenario (items that belong to an existing story)

| Tag | n | openai-3-large | qwen3-embed-8b | gemini-emb-2 | hybrid-voyage | voyage-4 | bm25 | production-logged | production-replay |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cross-source | 22 | 91% | 95% | 91% | 91% | 91% | 82% | 70% | 68% |
| duplicate-stories | 28 | 100% | 100% | 100% | 100% | 96% | 100% | 86% | 86% |
| prod-missed-match | 33 | 94% | 97% | 94% | 94% | 91% | 88% | 45% | 55% |
| prod-wrong-attach | 4 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| subject-drift | 40 | 95% | 98% | 95% | 95% | 92% | 90% | 58% | 40% |
| article | 69 | 99% | 97% | 99% | 97% | 97% | 94% | 85% | 77% |
| github_release | 16 | 94% | 94% | 94% | 94% | 94% | 94% | 88% | 94% |
| x_post | 27 | 96% | 100% | 96% | 96% | 93% | 96% | 81% | 74% |
| version-sibling | 4 | 75% | 75% | 75% | 75% | 75% | 75% | 75% | 75% |
