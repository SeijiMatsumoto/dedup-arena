"""Relevance labels by pooling + an LLM judge + review of disagreements.

1. Pool: for each query, the union of the top `depth` stories from several diverse retrievers,
   plus every candidate production retrieved and the story production attached it to.
2. Judge: each model in task.yaml `judges` picks, from the pool, the stories covering the same
   development (task.yaml `definition`), or none.
3. Merge: the label needs two agreeing verdicts. With several judges, they must agree. With one
   judge (the cheap default), its second opinion is what production actually decided. Any
   disagreement makes the query `disputed` until it has an entry in adjudications.yaml,
   written after reading the case.

Pooling is how TREC builds labels for large corpora. Its known blind spot: a relevant story
that no pooled retriever ranked in its top `depth` is never judged. Production's own
candidates are always in the pool, so a miss that production found cannot be missed here.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from .client import CacheMiss, Client, usage_cost
from .datasets import Dataset, Query, doc_card, query_card, read_jsonl, write_jsonl
from .retrievers import Arena, RetrieverSpec, same_subject

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string", "description": "One or two sentences."},
        "matches": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["reason", "matches"],
    "additionalProperties": False,
}


def judge_prompt(dataset: Dataset, query: Query, pool: list[str]) -> str:
    cards = "\n".join(doc_card(dataset.doc(i), 350) for i in pool)
    return (
        "You are labeling a retrieval benchmark for a news-dedup system. Decide which existing "
        "stories cover the same development as the incoming item, using exactly this policy:\n"
        f"<grouping>\n{dataset.task.definition}</grouping>\n\n"
        f"<stories>\n{cards}\n</stories>\n\n<incoming>\n{query_card(query, 900)}\n</incoming>\n\n"
        "List every story id that covers the same development (several stories can be duplicates "
        "of one event). Return an empty list if none does. Be strict: same company or product is "
        "not enough. Answer only with the JSON object."
    )


async def judge(client: Client, model: str, prompt: str, rpm: float | None = None) -> dict[str, Any]:
    body = {
        "model": model,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "judgment", "strict": True, "schema": JUDGE_SCHEMA},
        },
    }
    async with client.gate(f"judge:{model}", 6):
        payload, _, _ = await client.post("/v1/chat/completions", body, limit_key=model, rpm=rpm)
    parsed = json.loads(payload["choices"][0]["message"]["content"])
    return {"matches": parsed["matches"], "reason": parsed["reason"], "cost": usage_cost(payload)}


async def build_pool(arena: Arena, dataset: Dataset, query: Query, pool_names: list[str],
                     depth: int) -> list[str]:
    eligible = dataset.eligible(query)
    allowed = set(eligible)
    results = await asyncio.gather(*(arena.search(n, query, eligible) for n in pool_names))
    errors = [f"{n}: {r.error}" for n, r in zip(pool_names, results, strict=True) if r.error]
    if errors:
        raise RuntimeError(f"{query.id} pool retriever failed: {errors[0]}")
    pooled = {i for r in results for i in r.ids[:depth]}
    prod = query.production
    pooled |= {i for i in prod.get("candidates", []) if i in allowed}
    if prod.get("story") in allowed:
        pooled.add(prod["story"])
    # Chronological, so the judge sees no retriever's ranking.
    return sorted(pooled, key=lambda i: (dataset.doc(i).available_at, i))


async def collect_judgments(
    dataset: Dataset,
    specs: dict[str, RetrieverSpec],
    client: Client,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    pool_names = dataset.task.pool.get("retrievers", [])
    depth = dataset.task.pool.get("depth", 8)
    arena = Arena(dataset, specs, client)
    for name in pool_names:
        await arena.prepare(name)
    done = 0
    total = len(dataset.queries)

    rpm = dataset.task.pool.get("judge_rpm", {})

    async def one(query: Query) -> dict[str, Any] | None:
        try:
            return await judge_one(query)
        except CacheMiss:
            return None
        except Exception as exc:  # noqa: BLE001 - leave it unlabeled; a rerun retries (cached work is free)
            print(f"{query.id}: {type(exc).__name__}: {str(exc)[:160]}")
            return None

    async def judge_one(query: Query) -> dict[str, Any]:
        nonlocal done
        pool = await build_pool(arena, dataset, query, pool_names, depth)
        prompt = judge_prompt(dataset, query, pool)
        verdicts = await asyncio.gather(*(judge(client, m, prompt, rpm.get(m)) for m in dataset.task.judges))
        done += 1
        if on_progress:
            on_progress(done, total)
        return {
            "query": query.id,
            "pool": pool,
            "judges": {
                m: {**v, "matches": [i for i in v["matches"] if i in set(pool)]}
                for m, v in zip(dataset.task.judges, verdicts, strict=True)
            },
        }

    results = await asyncio.gather(*(one(q) for q in dataset.queries))
    return [r for r in results if r is not None]


def tags_for(dataset: Dataset, query: Query, relevant: list[str]) -> list[str]:
    tags = [f"type:{query.item_type or 'unknown'}"]
    docs = [dataset.doc(i) for i in relevant]
    if not relevant:
        tags.append("new-event")
    else:
        tags.append("attach")
        if not any(d.item_type == query.item_type for d in docs):
            tags.append("cross-source")
        if not any(same_subject(query.subject, d.subject) for d in docs):
            tags.append("subject-drift")
        if len(relevant) > 1:
            tags.append("duplicate-stories")
    siblings = [
        i for i in dataset.eligible(query)
        if i not in relevant
        and query.version and dataset.doc(i).version
        and same_subject(query.subject, dataset.doc(i).subject)
        and dataset.doc(i).version != query.version
    ]
    if siblings:
        tags.append("version-sibling")
    action = query.production.get("action")
    if action == "new" and relevant:
        tags.append("prod-missed-match")
    if action in ("update", "quiet") and query.production.get("story") not in relevant:
        tags.append("prod-wrong-attach")
    return tags


def merge_labels(dataset: Dataset, judgments: list[dict[str, Any]],
                 adjudications: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Returns (labeled query rows, still-disputed judgment rows)."""
    by_query = {j["query"]: j for j in judgments}
    labeled, disputed = [], []
    for query in dataset.queries:
        j = by_query.get(query.id)
        if j is None:
            continue
        verdicts = [set(v["matches"]) for v in j["judges"].values()]
        agreed = "judges-agree"
        if len(verdicts) == 1:
            action = query.production.get("action")
            if action is None:
                verdicts.append(None)  # nothing to agree with: review it
            else:
                verdicts.append({query.production["story"]} if action != "new" else set())
            agreed = "judge+production"
        review = adjudications.get(query.id)
        if review is not None:
            relevant, source = sorted(review["relevant"]), "reviewed"
        elif all(v == verdicts[0] for v in verdicts):
            relevant, source = sorted(verdicts[0]), agreed
        else:
            disputed.append(j)
            continue
        row = query.model_dump()
        row.update(relevant=relevant, label_source=source, tags=tags_for(dataset, query, relevant))
        labeled.append(row)
    return labeled, disputed


def load_adjudications(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def write_review_sheet(dataset: Dataset, disputed: list[dict[str, Any]], path: Path) -> None:
    """Human-readable file for adjudicating disagreements: the item, each verdict, the stories
    they name in full, and the rest of the pool as one-liners (the answer may be neither)."""
    queries = {q.id: q for q in dataset.queries}
    lines = [
        "# Disputed labels",
        "",
        "The judge and its second opinion (production's decision, or a second judge) disagree.",
        "Decide each in adjudications.yaml using the policy in task.yaml (`definition`).",
        "",
    ]
    for j in disputed:
        q = queries[j["query"]]
        prod = q.production
        decided = prod.get("story") if prod.get("action") not in (None, "new") else "new story"
        lines.append(f"## {q.id}")
        lines.append(f"\nProduction decided: **{decided}** ({prod.get('action')})\n")
        lines.append("```\n" + query_card(q, 900) + "\n```\n")
        for model, v in j["judges"].items():
            lines.append(f"- **{model}**: {', '.join(v['matches']) or 'new story'} — {v['reason']}")
        named = sorted({i for v in j["judges"].values() for i in v["matches"]}
                       | ({prod["story"]} if prod.get("story") else set()))
        for i in named:
            lines.append("\n```\n" + doc_card(dataset.doc(i), 600) + "\n```")
        rest = [i for i in j["pool"] if i not in named]
        if rest:
            lines.append("\nRest of the candidate pool:\n")
            for i in rest:
                d = dataset.doc(i)
                version = f" v{d.version}" if d.version else ""
                lines.append(f"- `{i}` {d.subject}{version} | {d.title} ({(d.published_at or '')[:10]})")
        lines.append("")
    path.write_text("\n".join(lines))


# --- Labeling by a coding agent (no API spend) -------------------------------------------------
# For items the API judge hasn't covered, `arena label --sheet` writes the same pools as
# Markdown for a coding agent (Codex, Gemini CLI, ...). Its answers in agent_labels.yaml are
# treated as one more judge, so they go through the same merge: agreement with production ->
# label, disagreement -> review.md. The sheets omit production's decision so the agent judges blind.

AGENT_DIR = "agent"


async def collect_pools(dataset: Dataset, specs: dict[str, RetrieverSpec], client: Client,
                        queries: list[Query]) -> dict[str, list[str]]:
    pool_names = dataset.task.pool.get("retrievers", [])
    depth = dataset.task.pool.get("depth", 8)
    arena = Arena(dataset, specs, client)
    for name in pool_names:
        await arena.prepare(name)
    pools = await asyncio.gather(*(build_pool(arena, dataset, q, pool_names, depth) for q in queries))
    return {q.id: p for q, p in zip(queries, pools, strict=True)}


def write_agent_sheets(dataset: Dataset, pools: dict[str, list[str]], folder: Path,
                       per_part: int = 50) -> list[Path]:
    out = folder / AGENT_DIR
    out.mkdir(exist_ok=True)
    for old in out.glob("part-*.md"):
        old.unlink()
    write_jsonl(out / "pools.jsonl", [{"query": q, "pool": p} for q, p in pools.items()])
    queries = {q.id: q for q in dataset.queries}
    ids = list(pools)
    paths = []
    for n, start in enumerate(range(0, len(ids), per_part), start=1):
        chunk = ids[start : start + per_part]
        lines = [f"# Items to label, part {n} ({len(chunk)} items)", ""]
        for qid in chunk:
            lines += [f"## {qid}", "", "```", query_card(queries[qid], 900), "```", "",
                      "Candidate stories:", ""]
            lines += [doc_card(dataset.doc(i), 250).replace("\n  ", "\n  > ") for i in pools[qid]]
            lines.append("")
        path = out / f"part-{n}.md"
        path.write_text("\n".join(lines))
        paths.append(path)
    return paths


def agent_judgments(folder: Path) -> list[dict[str, Any]]:
    """agent_labels.yaml answers as judgment rows (one judge named by the file's `labeler`)."""
    pools_path, labels_path = folder / AGENT_DIR / "pools.jsonl", folder / AGENT_DIR / "agent_labels.yaml"
    if not (pools_path.exists() and labels_path.exists()):
        return []
    labels = yaml.safe_load(labels_path.read_text()) or {}
    name = f"agent:{labels.pop('labeler', 'unknown')}"
    rows = []
    for p in read_jsonl(pools_path):
        entry = labels.get(p["query"])
        if entry is None:
            continue
        allowed = set(p["pool"])
        rows.append({"query": p["query"], "pool": p["pool"], "judges": {name: {
            "matches": [i for i in entry.get("relevant") or [] if i in allowed],
            "reason": entry.get("note", ""), "cost": 0.0}}})
    return rows


def save(folder: Path, judgments: list[dict[str, Any]]) -> None:
    write_jsonl(folder / "judgments.jsonl", judgments)


def load_judgments(folder: Path) -> list[dict[str, Any]]:
    path = folder / "judgments.jsonl"
    return read_jsonl(path) if path.exists() else []
