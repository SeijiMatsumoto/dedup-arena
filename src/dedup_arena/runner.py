from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from .client import Client
from .datasets import Dataset, Query
from .e2e import JUDGE_MODEL, decide, production_actual
from .metrics import cutoff_sweep, e2e_metrics, retrieval_metrics
from .retrievers import Arena, Result, RetrieverSpec

STORED_IDS = 10  # ids kept per query in the report


def subsample(dataset: Dataset, query: Query, size: int | None) -> list[str]:
    """Eligible stories, optionally cut to `size`: the relevant ones plus a fixed random sample."""
    eligible = dataset.eligible(query)
    if not size or len(eligible) <= size:
        return eligible
    keep = [i for i in eligible if i in (query.relevant or [])]
    rest = [i for i in eligible if i not in keep]
    random.Random(f"{query.id}:{size}").shuffle(rest)
    chosen = set(keep + rest[: max(size - len(keep), 0)])
    return [i for i in eligible if i in chosen]


async def run_dataset(
    dataset: Dataset,
    specs: dict[str, RetrieverSpec],
    selected: list[RetrieverSpec],
    client: Client,
    *,
    corpus_size: int | None = None,
    e2e: bool = True,
    judge_model: str = JUDGE_MODEL,
    on_progress: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    arena = Arena(dataset, specs, client)
    for spec in selected:
        await arena.prepare(spec.name)
    queries = [q for q in dataset.queries if q.relevant is not None]
    eligible = {q.id: subsample(dataset, q, corpus_size) for q in queries}
    results: dict[str, dict[str, Result]] = {s.name: {} for s in selected}
    preds: dict[str, dict[str, dict[str, Any]]] = {s.name: {} for s in selected}

    async def one(spec: RetrieverSpec, query: Query) -> None:
        result = await arena.search(spec.name, query, eligible[query.id])
        results[spec.name][query.id] = result
        if e2e and not result.skipped:
            preds[spec.name][query.id] = await decide(client, dataset, query, result, judge_model)
        if on_progress:
            on_progress(1)

    await asyncio.gather(*(one(s, q) for s in selected for q in queries))

    out: dict[str, Any] = {
        "description": dataset.task.description,
        "use_case": dataset.task.use_case,
        "n_corpus": len(dataset.corpus),
        "eligible_median": sorted(len(v) for v in eligible.values())[len(eligible) // 2],
        "queries": [
            {
                "id": q.id, "subject": q.subject, "version": q.version, "item_type": q.item_type,
                "headline": (q.title or q.text)[:160], "url": q.url, "relevant": q.relevant,
                "tags": q.tags, "label_source": q.label_source,
                "production_action": q.production.get("action"),
            }
            for q in queries
        ],
        "docs": {
            d.id: {"subject": d.subject, "version": d.version, "title": d.title, "url": d.url}
            for d in dataset.corpus
        },
        "index_cost": {s.name: arena.retrievers[s.name].index_cost for s in selected},
        "results": {},
        "metrics": {},
        "sweeps": {},
        "e2e": {},
    }
    for spec in selected:
        raw = {qid: r.model_dump(exclude_none=True) for qid, r in results[spec.name].items()}
        out["metrics"][spec.name] = retrieval_metrics(queries, raw)
        if spec.kind == "dense":
            out["sweeps"][spec.name] = cutoff_sweep(queries, raw)
        out["results"][spec.name] = {
            qid: {**r, "ids": r.get("ids", [])[:STORED_IDS], "scores": r.get("scores", [])[:STORED_IDS]}
            for qid, r in raw.items()
        }
        if e2e:
            out["e2e"][spec.name] = {
                "metrics": e2e_metrics(queries, preds[spec.name]),
                "preds": {qid: p.get("pred") for qid, p in preds[spec.name].items()},
            }
    # What production actually decided costs nothing to score, so it is always included.
    actual = {q.id: p for q in queries if (p := production_actual(q)) is not None}
    if actual:
        out["e2e"]["production-actual"] = {
            "metrics": e2e_metrics(queries, actual),
            "preds": {qid: p["pred"] for qid, p in actual.items()},
        }
    return out


async def run_eval(
    datasets: list[Dataset],
    specs: dict[str, RetrieverSpec],
    selected: list[RetrieverSpec],
    cache_dir: Path,
    *,
    corpus_size: int | None = None,
    e2e: bool = True,
    judge_model: str = JUDGE_MODEL,
    on_progress: Callable[[int, int], None] | None = None,
    cache_only: bool = False,
) -> dict[str, Any]:
    total = sum(len([q for q in d.queries if q.relevant is not None]) for d in datasets) * len(selected)
    done = 0

    def tick(n: int) -> None:
        nonlocal done
        done += n
        if on_progress:
            on_progress(done, total)

    report: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "corpus_size": corpus_size,
        "judge_model": judge_model if e2e else None,
        "retrievers": [
            {**s.model_dump(exclude_defaults=True), "name": s.name, "kind": s.kind,
             "family": s.family, "decides": s.decides}
            for s in selected
        ],
        "datasets": {},
    }
    async with httpx.AsyncClient(timeout=180) as http:
        client = Client(cache_dir, http, cache_only=cache_only)
        for dataset in datasets:
            report["datasets"][dataset.name] = await run_dataset(
                dataset, specs, selected, client, corpus_size=corpus_size, e2e=e2e,
                judge_model=judge_model, on_progress=tick,
            )
    return report


def save_report(report: dict[str, Any], reports_dir: Path, prefix: str = "run") -> Path:
    stamp = report["created_at"].replace(":", "").replace("-", "")[:15]
    path = reports_dir / f"{prefix}-{stamp}.json"
    path.write_text(json.dumps(report, indent=1))
    return path
