"""End to end: a retriever's top 5 -> one decision-model call -> attach to a story, or new story.

This is enable-ai's real pipeline shape. Retrieval sets the ceiling: if the right story is not
in the top 5, the judge cannot pick it. Retrievers that decide on their own (long context,
agentic) skip the judge; their first match, or none, is the answer.
"""

from __future__ import annotations

from typing import Any

from .client import Client, usage_cost
from .datasets import Dataset, Query
from .retrievers import Result

TOP_K = 5
JUDGE_MODEL = "typesafe/jev-1.14"


def judge_state(dataset: Dataset, query: Query, candidates: list[str]) -> dict[str, Any]:
    def doc(i: str) -> dict[str, Any]:
        d = dataset.doc(i)
        return {
            "id": d.id, "subject": d.subject, "version": d.version, "title": d.title,
            "item_type": d.item_type, "published_at": d.published_at, "excerpt": d.excerpt[:500],
        }

    return {
        "incoming": {
            "subject": query.subject, "version": query.version, "title": query.title,
            "item_type": query.item_type, "published_at": query.published_at,
            "content": query.text,
        },
        "candidates": [doc(i) for i in candidates],
    }


def judge_question(dataset: Dataset, candidates: list[str]) -> dict[str, Any]:
    criteria = {"none": "No candidate covers the same development."}
    criteria.update({i: f"Same development as candidate {i}." for i in candidates})
    return {
        "type": "choice",
        "instructions": (
            f"<grouping>\n{dataset.task.definition}</grouping>\n<question>\nWhich candidate covers "
            "the same development as incoming? Compare the underlying event, not completeness or "
            "article emphasis. Choose none for a genuinely different event.\n</question>"
        ),
        "criteria": criteria,
    }


async def decide(client: Client, dataset: Dataset, query: Query, result: Result,
                 model: str = JUDGE_MODEL) -> dict[str, Any]:
    """Returns {pred, latency_ms, cost[, error]}; latency and cost include retrieval."""
    if result.error:
        return {"pred": None, "error": result.error}
    retrieval_ms = result.latency_ms or 0.0
    if result.decided:
        return {"pred": result.ids[0] if result.ids else None, "latency_ms": result.latency_ms,
                "cost": result.cost}
    candidates = result.ids[:TOP_K]
    if not candidates:
        return {"pred": None, "latency_ms": result.latency_ms, "cost": result.cost}
    body = {
        "model": model,
        "state": judge_state(dataset, query, candidates),
        "questions": {"match": judge_question(dataset, candidates)},
    }
    try:
        async with client.gate(f"decide:{model}", 8):
            payload, latency, _ = await client.post("/alpha/decisions", body, limit_key=model)
        choice = str(payload["answers"]["match"]["choice"])
    except Exception as exc:  # noqa: BLE001 - one failed call is one wrong answer, not a crashed run
        return {"pred": None, "error": f"{type(exc).__name__}: {exc}"[:300]}
    pred = None if choice == "none" or choice not in candidates else choice
    if result.latency_ms is None:
        # Replayed from a log: retrieval latency/cost unknown, so the total is too.
        return {"pred": pred}
    cost = usage_cost(payload)
    total_cost = None if result.cost is None and cost is None else (result.cost or 0) + (cost or 0)
    return {"pred": pred, "latency_ms": retrieval_ms + latency, "cost": total_cost}


def production_actual(query: Query) -> dict[str, Any] | None:
    """What enable-ai actually did with this item (no new calls)."""
    action = query.production.get("action")
    if action is None:
        return None
    return {"pred": query.production.get("story") if action != "new" else None}
