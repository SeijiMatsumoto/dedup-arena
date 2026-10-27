"""Scoring. Retrieval metrics use only queries whose item matches an existing story ("attach");
queries that start a new story ("new") are where thresholds and deciding retrievers are judged.

recall@k here is the hit rate: at least one relevant story in the top k. For dedup any
duplicate is enough, because the decision step only needs one correct candidate to attach to.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any

from .datasets import Query

KS = (1, 3, 5, 10, 25)
SWEEP_CUTOFFS = [round(0.05 * i, 2) for i in range(1, 13)]  # cosine distance 0.05 .. 0.60


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[round(q * 100) - 1]


def first_hit(ids: list[str], relevant: list[str]) -> int | None:
    """1-based rank of the first relevant id."""
    for rank, doc_id in enumerate(ids, start=1):
        if doc_id in relevant:
            return rank
    return None


def ndcg(ids: list[str], relevant: list[str], k: int = 10) -> float:
    dcg = sum(1 / math.log2(r + 1) for r, d in enumerate(ids[:k], start=1) if d in relevant)
    ideal = sum(1 / math.log2(r + 1) for r in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def cost_per_1k(costs: list[float | None]) -> float | None:
    known = [c for c in costs if c is not None]
    return 1000 * sum(known) / len(known) if known else None


def retrieval_metrics(queries: list[Query], results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """results: query id -> Result dict. Errors count as misses; skipped queries are excluded."""
    pos = [q for q in queries if q.relevant and not results[q.id].get("skipped")]
    neg = [q for q in queries if q.relevant == [] and not results[q.id].get("skipped")]
    used = pos + neg
    hits: dict[int, list[bool]] = {k: [] for k in KS}
    rr, gains, hit5 = [], [], {}
    tag_hits: dict[str, list[bool]] = defaultdict(list)
    for q in pos:
        r = results[q.id]
        ids = [] if r.get("error") else r["ids"]
        rank = first_hit(ids, q.relevant)
        for k in KS:
            hits[k].append(rank is not None and rank <= k)
        rr.append(1 / rank if rank else 0.0)
        gains.append(ndcg(ids, q.relevant))
        hit5[q.id] = rank is not None and rank <= 5
        for tag in q.tags:
            tag_hits[tag].append(hit5[q.id])
    neg_empty = [not results[q.id].get("ids") and not results[q.id].get("error") for q in neg]
    live = [results[q.id] for q in used if not results[q.id].get("error")]
    latencies = [r["latency_ms"] for r in live if r.get("latency_ms") is not None]
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    return {
        "n_attach": len(pos),
        "n_new": len(neg),
        **{f"recall@{k}": mean(hits[k]) for k in KS},
        "mrr": mean(rr),
        "ndcg@10": mean(gains),
        # Of queries that should start a new story, how often nothing came back. Only
        # meaningful for retrievers with a cutoff or that decide (top-k always returns k).
        "new_empty_rate": mean(neg_empty),
        "candidates_mean": mean([min(len(r.get("ids", [])), 5) for r in live]),
        "latency_p50_ms": percentile(latencies, 0.5),
        "latency_p95_ms": percentile(latencies, 0.95),
        "cost_per_1k": cost_per_1k([r.get("cost") for r in live]),
        "errors": sum(1 for q in used if results[q.id].get("error")),
        "skipped": len(queries) - len(used),
        # Recall@5 per scenario tag, over items that belong to an existing story.
        "by_tag": {t: {"n": len(v), "rate": mean(v)} for t, v in sorted(tag_hits.items())},
        "hit5": hit5,
    }


def cutoff_sweep(queries: list[Query], results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """For a dense retriever: recall@5 and candidate load if only stories within a cosine-distance
    cutoff were passed on. Uses the top-25 similarities each result stored."""
    rows = []
    pos = [q for q in queries if q.relevant and not results[q.id].get("error")]
    allq = [q for q in queries if q.relevant is not None and not results[q.id].get("error")]
    for cutoff in SWEEP_CUTOFFS:
        def kept(q: Query, cutoff: float = cutoff) -> list[str]:
            r = results[q.id]
            return [i for i, s in zip(r.get("ids", []), r.get("scores", []), strict=False)
                    if 1 - s <= cutoff + 1e-9][:5]
        found = [first_hit(kept(q), q.relevant) is not None for q in pos]
        load = [len(kept(q)) for q in allq]
        new_empty = [not kept(q) for q in allq if q.relevant == []]
        rows.append(
            {
                "max_distance": cutoff,
                "recall@5": sum(found) / len(found) if found else None,
                "candidates_mean": sum(load) / len(load) if load else None,
                "new_empty_rate": sum(new_empty) / len(new_empty) if new_empty else None,
            }
        )
    return rows


def e2e_metrics(queries: list[Query], preds: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """preds: query id -> {pred, latency_ms, cost, error?}. pred is a story id or None (new story)."""
    rows = [(q, preds[q.id]) for q in queries if q.relevant is not None and q.id in preds]
    correct: dict[str, bool] = {}
    attach_ok, new_ok, missed, wrong = [], [], [], []
    tag_ok: dict[str, list[bool]] = defaultdict(list)
    for q, p in rows:
        if p.get("error"):
            ok = False
        else:
            ok = p["pred"] in q.relevant if q.relevant else p["pred"] is None
        correct[q.id] = ok
        if q.relevant:
            attach_ok.append(ok)
            missed.append(not p.get("error") and p["pred"] is None)
            wrong.append(not p.get("error") and p["pred"] is not None and p["pred"] not in q.relevant)
        else:
            new_ok.append(ok)
            wrong.append(not p.get("error") and p["pred"] is not None)
        for tag in q.tags:
            tag_ok[tag].append(ok)
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    live = [p for _, p in rows if not p.get("error")]
    latencies = [p["latency_ms"] for p in live if p.get("latency_ms") is not None]
    return {
        "n": len(rows),
        "accuracy": mean(list(correct.values())),
        "accuracy_attach": mean(attach_ok),
        "accuracy_new": mean(new_ok),
        # Attach queries answered "new story": each one is a duplicate story in the feed.
        "duplicate_rate": mean(missed),
        # Any query attached to a story that does not cover it: a wrongly merged story.
        "wrong_merge_rate": mean(wrong),
        "latency_p50_ms": percentile(latencies, 0.5),
        "latency_p95_ms": percentile(latencies, 0.95),
        "cost_per_1k": cost_per_1k([p.get("cost") for p in live]),
        "errors": sum(1 for _, p in rows if p.get("error")),
        "by_tag": {t: {"n": len(v), "rate": mean(v)} for t, v in sorted(tag_ok.items())},
        "correct": correct,
    }


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value from the two discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_p(a: dict[str, bool], b: dict[str, bool]) -> float:
    shared = a.keys() & b.keys()
    only_a = sum(1 for k in shared if a[k] and not b[k])
    only_b = sum(1 for k in shared if b[k] and not a[k])
    return mcnemar_p(only_a, only_b)
