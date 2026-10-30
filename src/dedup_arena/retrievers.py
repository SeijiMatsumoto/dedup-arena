"""Every retrieval strategy behind one interface: search(query, eligible ids) -> ranked ids.

Kinds (set in retrievers.yaml):
  logged        What production actually retrieved, read from production logs.
  production    enable-ai's candidate query re-implemented: subject identity OR cosine
                distance <= max_distance, identity first, version guard, top 5.
  bm25          Lexical baseline.
  dense         One embedding model, brute-force cosine (the corpus is small enough).
  hybrid        Reciprocal rank fusion of other retrievers.
  rerank        A base retriever's top_n reordered by a reranker model.
  long_context  No retrieval: every eligible story goes into one LLM prompt.
  agentic       An LLM that calls a search tool (another retriever) for a few rounds.
  mock          Noisy oracle for offline runs and tests.

long_context and agentic retrievers also *decide*: an empty list means "no existing story".
The others always return their nearest neighbors and leave the decision to a judge model.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, Field

from .client import Client, usage_cost
from .datasets import (
    Dataset,
    Query,
    View,
    doc_card,
    doc_text,
    normalize_subject,
    query_card,
    query_text,
)

Kind = Literal[
    "logged", "production", "bm25", "dense", "hybrid", "rerank", "long_context", "agentic", "mock"
]
FAMILY = {
    "logged": "production",
    "production": "production",
    "bm25": "lexical",
    "dense": "dense",
    "hybrid": "hybrid",
    "rerank": "rerank",
    "long_context": "long context",
    "agentic": "agentic",
    "mock": "mock",
}
EMBED_BATCH = 64
MAX_EMBED_CHARS = 2000
RRF_K = 60


class RetrieverSpec(BaseModel):
    name: str
    kind: Kind
    enabled: bool = True
    note: str = ""
    model: str | None = None
    dims: int | None = None  # embedding output dimensions, when the model supports it
    prefix: str = ""  # prepended to every embedded text (e.g. Gemini task prompts)
    doc_view: View = "full"
    query_view: View = "full"
    max_distance: float | None = None  # cosine distance cutoff (dense / production)
    top_k: int = 25  # how many ids to return
    components: list[str] = Field(default_factory=list)  # hybrid
    base: str | None = None  # rerank
    top_n: int = 25  # rerank: how many of the base results to reorder
    tool: str | None = None  # agentic: retriever behind the search tool
    tool_k: int = 8
    max_rounds: int = 3
    excerpt_chars: int = 160  # long_context / agentic story cards
    price_in: float | None = None  # USD per 1M tokens, if OpenRouter reports no cost
    price_out: float | None = None
    rpm: float | None = None
    concurrency: int = 8
    extra: dict[str, Any] = Field(default_factory=dict)
    accuracy: float = 0.8  # mock only

    @property
    def family(self) -> str:
        return FAMILY[self.kind]

    @property
    def decides(self) -> bool:
        return self.kind in ("long_context", "agentic")


class Result(BaseModel):
    ids: list[str] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)  # cosine similarity for dense kinds
    latency_ms: float | None = None
    cost: float | None = None
    cached: bool = False
    decided: bool = False
    skipped: bool = False  # no data for this query (e.g. it predates logging); not a miss
    error: str | None = None
    trace: list[dict[str, Any]] | None = None


def load_specs(path: Path) -> dict[str, RetrieverSpec]:
    specs = [RetrieverSpec.model_validate(r) for r in yaml.safe_load(path.read_text())["retrievers"]]
    by_name = {s.name: s for s in specs}
    if len(by_name) != len(specs):
        raise SystemExit("Duplicate retriever names in retrievers.yaml")
    return by_name


def select_specs(specs: dict[str, RetrieverSpec], names: list[str] | None) -> list[RetrieverSpec]:
    if not names:
        return [s for s in specs.values() if s.enabled]
    unknown = [n for n in names if n not in specs]
    if unknown:
        raise SystemExit(f"Unknown retriever(s): {', '.join(unknown)}")
    return [specs[n] for n in names]


def _sum(values: list[float | None]) -> float | None:
    known = [v for v in values if v is not None]
    return sum(known) if known else None


class Retriever:
    def __init__(self, spec: RetrieverSpec, arena: Arena):
        self.spec = spec
        self.arena = arena
        self.index_cost: float | None = 0.0

    @property
    def dataset(self) -> Dataset:
        return self.arena.dataset

    async def prepare(self) -> None:
        """Build any index. Called once per dataset before searching."""

    async def search(self, query: Query, eligible: list[str]) -> Result:
        raise NotImplementedError


# --- Production -------------------------------------------------------------------------------


class LoggedRetriever(Retriever):
    async def search(self, query: Query, eligible: list[str]) -> Result:
        if not query.production.get("logged"):
            return Result(skipped=True)
        allowed = set(eligible)
        return Result(ids=[c for c in query.production.get("candidates", []) if c in allowed])


def product_identity(value: str) -> str | None:
    """enable-ai's conservative versioned product suffix (e.g. 'gemini 3'); never generic names."""
    tokens = normalize_subject(value).split()
    for index in range(len(tokens) - 1, -1, -1):
        token = tokens[index]
        if token.isalpha():
            tail = tokens[index + 1 :]
            if (
                len(token) >= 3
                and token not in {"model", "version", "agent", "release", "preview"}
                and tail
                and all(t.isdigit() for t in tail)
            ):
                return " ".join(tokens[index:])
            break
    return None


def same_subject(left: str, right: str) -> bool:
    a, b = normalize_subject(left), normalize_subject(right)
    return bool(a and a == b) or bool(
        product_identity(left) and product_identity(left) == product_identity(right)
    )


def compatible(query: Query, doc) -> bool:
    """enable-ai's version guard: same subject + explicitly different versions -> not a candidate."""
    left, right = query.version, doc.version
    if not left or not right or not same_subject(query.subject, doc.subject):
        return True
    left, right = left.casefold().removeprefix("v"), right.casefold().removeprefix("v")
    if left == right:
        return True
    if query.item_type == "github_release":
        a = re.fullmatch(r"(\d+)\.(\d+)\.\d+", left)
        b = re.fullmatch(r"(\d+)\.(\d+)\.\d+", right)
        return bool(a and b and a.groups() == b.groups())
    return False


# --- Dense ------------------------------------------------------------------------------------


class DenseRetriever(Retriever):
    async def prepare(self) -> None:
        texts = [self.spec.prefix + doc_text(d, self.spec.doc_view) for d in self.dataset.corpus]
        vectors, costs = [], []
        batches = [texts[i : i + EMBED_BATCH] for i in range(0, len(texts), EMBED_BATCH)]
        results = await asyncio.gather(*(self._embed(b) for b in batches))
        for batch_vectors, _, cost, _ in results:
            vectors.extend(batch_vectors)
            costs.append(cost)
        self.matrix = self._normalize(np.array(vectors, dtype=np.float32))
        self.row = {d.id: i for i, d in enumerate(self.dataset.corpus)}
        self.index_cost = _sum(costs)

    @staticmethod
    def _normalize(m: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(m, axis=-1, keepdims=True)
        return m / np.where(norms == 0, 1, norms)

    async def _embed(self, texts: list[str]) -> tuple[list[list[float]], float, float | None, bool]:
        body: dict[str, Any] = {
            "model": self.spec.model,
            "input": [t[:MAX_EMBED_CHARS] for t in texts],
            **self.spec.extra,
        }
        if self.spec.dims:
            body["dimensions"] = self.spec.dims
        async with self.arena.client.gate(f"embed:{self.spec.model}", self.spec.concurrency):
            payload, latency, cached = await self.arena.client.post(
                "/v1/embeddings", body, limit_key=self.spec.model, rpm=self.spec.rpm
            )
        data = sorted(payload["data"], key=lambda d: d["index"])
        if len(data) != len(texts):
            raise RuntimeError(f"expected {len(texts)} embeddings, got {len(data)}")
        return [d["embedding"] for d in data], latency, usage_cost(payload, self.spec.price_in), cached

    async def embed_query(self, query: Query) -> tuple[np.ndarray, float, float | None, bool]:
        text = self.spec.prefix + query_text(query, self.spec.query_view)
        vectors, latency, cost, cached = await self._embed([text])
        return self._normalize(np.array(vectors[0], dtype=np.float32)), latency, cost, cached

    def similarities(self, vector: np.ndarray, eligible: list[str]) -> np.ndarray:
        rows = [self.row[i] for i in eligible]
        return self.matrix[rows] @ vector

    async def search(self, query: Query, eligible: list[str]) -> Result:
        vector, latency, cost, cached = await self.embed_query(query)
        started = perf_counter()
        sims = self.similarities(vector, eligible)
        order = np.argsort(-sims, kind="stable")
        if self.spec.max_distance is not None:
            order = [i for i in order if 1 - sims[i] <= self.spec.max_distance]
        order = order[: self.spec.top_k]
        local_ms = (perf_counter() - started) * 1000
        return Result(
            ids=[eligible[i] for i in order],
            scores=[round(float(sims[i]), 5) for i in order],
            latency_ms=latency + local_ms,
            cost=cost,
            cached=cached,
        )


class ProductionRetriever(DenseRetriever):
    """enable-ai's candidate query: subject identity OR distance cutoff, identity first, top 5.

    Approximation: production embeds each Story as the centroid of all its items so far; here a
    Story is its discovery item only. The `logged` retriever is the exact production record.
    """

    async def search(self, query: Query, eligible: list[str]) -> Result:
        vector, latency, cost, cached = await self.embed_query(query)
        started = perf_counter()
        sims = self.similarities(vector, eligible)
        subject = normalize_subject(query.subject)
        product = product_identity(query.subject)
        cutoff = self.spec.max_distance if self.spec.max_distance is not None else 0.10
        rows = []
        for i, doc_id in enumerate(eligible):
            doc = self.dataset.doc(doc_id)
            norm = normalize_subject(doc.subject)
            identity = bool(subject) and norm == subject
            if product:
                identity = identity or norm == product or norm.endswith(" " + product)
            distance = 1 - float(sims[i])
            if identity or distance <= cutoff:
                rows.append((not identity, distance, doc_id, float(sims[i])))
        rows.sort()
        rows = [r for r in rows[: 5 * 5] if compatible(query, self.dataset.doc(r[2]))][:5]
        local_ms = (perf_counter() - started) * 1000
        return Result(
            ids=[r[2] for r in rows],
            scores=[round(r[3], 5) for r in rows],
            latency_ms=latency + local_ms,
            cost=cost,
            cached=cached,
        )


# --- Lexical ----------------------------------------------------------------------------------


def tokenize(text: str) -> list[str]:
    return normalize_subject(text).split()


class BM25Retriever(Retriever):
    K1 = 1.2
    B = 0.75

    async def prepare(self) -> None:
        docs = [tokenize(doc_text(d, self.spec.doc_view)) for d in self.dataset.corpus]
        self.tf = {d.id: Counter(toks) for d, toks in zip(self.dataset.corpus, docs, strict=True)}
        self.length = {d.id: len(toks) for d, toks in zip(self.dataset.corpus, docs, strict=True)}
        self.avg = sum(self.length.values()) / max(len(docs), 1)
        df = Counter(t for toks in docs for t in set(toks))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    async def search(self, query: Query, eligible: list[str]) -> Result:
        started = perf_counter()
        terms = Counter(tokenize(query_text(query, self.spec.query_view)))
        scores = []
        for doc_id in eligible:
            tf, length = self.tf[doc_id], self.length[doc_id]
            score = 0.0
            for term, qtf in terms.items():
                f = tf.get(term)
                if f:
                    norm = f * (self.K1 + 1) / (f + self.K1 * (1 - self.B + self.B * length / self.avg))
                    score += self.idf[term] * norm * min(qtf, 3)
            scores.append((score, doc_id))
        scores.sort(key=lambda s: -s[0])
        top = [s for s in scores if s[0] > 0][: self.spec.top_k]
        return Result(
            ids=[d for _, d in top],
            scores=[round(s, 4) for s, _ in top],
            latency_ms=(perf_counter() - started) * 1000,
            cost=0.0,
        )


# --- Composites -------------------------------------------------------------------------------


class HybridRetriever(Retriever):
    async def search(self, query: Query, eligible: list[str]) -> Result:
        parts = await asyncio.gather(
            *(self.arena.search(name, query, eligible) for name in self.spec.components)
        )
        failed = next((p.error for p in parts if p.error), None)
        if failed:
            return Result(error=failed)
        fused: dict[str, float] = {}
        for part in parts:
            for rank, doc_id in enumerate(part.ids):
                fused[doc_id] = fused.get(doc_id, 0.0) + 1 / (RRF_K + rank + 1)
        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[: self.spec.top_k]
        return Result(
            ids=[d for d, _ in ranked],
            scores=[round(s, 5) for _, s in ranked],
            # Components run in parallel, so the slowest one sets the latency.
            latency_ms=max((p.latency_ms or 0) for p in parts),
            cost=_sum([p.cost for p in parts]),
            cached=all(p.cached for p in parts),
        )


class RerankRetriever(Retriever):
    async def search(self, query: Query, eligible: list[str]) -> Result:
        base = await self.arena.search(self.spec.base, query, eligible)
        if base.error or not base.ids:
            return base.model_copy(update={"scores": []})
        candidates = base.ids[: self.spec.top_n]
        body = {
            "model": self.spec.model,
            "query": query_text(query, self.spec.query_view)[:MAX_EMBED_CHARS],
            "documents": [
                doc_text(self.dataset.doc(i), self.spec.doc_view)[:MAX_EMBED_CHARS] for i in candidates
            ],
            **self.spec.extra,
        }
        async with self.arena.client.gate(f"rerank:{self.spec.model}", self.spec.concurrency):
            payload, latency, cached = await self.arena.client.post(
                "/v1/rerank", body, limit_key=self.spec.model, rpm=self.spec.rpm
            )
        ranked = sorted(payload["results"], key=lambda r: -r["relevance_score"])
        return Result(
            ids=[candidates[r["index"]] for r in ranked][: self.spec.top_k],
            scores=[round(r["relevance_score"], 5) for r in ranked][: self.spec.top_k],
            latency_ms=(base.latency_ms or 0) + latency,
            cost=_sum([base.cost, usage_cost(payload)]),
            cached=base.cached and cached,
        )


# --- LLM-based --------------------------------------------------------------------------------

MATCH_INSTRUCTIONS = (
    "Decide which existing stories cover the same development as the incoming item, using this "
    "policy:\n<grouping>\n{definition}</grouping>\n"
)
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "matches": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Ids of stories covering the same development, best first. Empty if none.",
        }
    },
    "required": ["matches"],
    "additionalProperties": False,
}


class LLMRetriever(Retriever):
    async def chat(self, body: dict[str, Any]) -> tuple[dict[str, Any], float, float | None, bool]:
        body = {"model": self.spec.model, "temperature": 0, **body, **self.spec.extra}
        async with self.arena.client.gate(f"chat:{self.spec.name}", self.spec.concurrency):
            payload, latency, cached = await self.arena.client.post(
                "/v1/chat/completions", body, limit_key=self.spec.model, rpm=self.spec.rpm
            )
        if not payload.get("choices"):
            raise RuntimeError(f"no choices in response: {json.dumps(payload)[:200]}")
        return payload, latency, usage_cost(payload, self.spec.price_in, self.spec.price_out), cached

    def clean(self, ids: list[Any], eligible: list[str]) -> list[str]:
        allowed = set(eligible)
        seen: list[str] = []
        for raw in ids:
            doc_id = str(raw).strip().strip("[]")
            if doc_id in allowed and doc_id not in seen:
                seen.append(doc_id)
        return seen


class LongContextRetriever(LLMRetriever):
    async def search(self, query: Query, eligible: list[str]) -> Result:
        cards = "\n".join(doc_card(self.dataset.doc(i), self.spec.excerpt_chars) for i in eligible)
        prompt = (
            MATCH_INSTRUCTIONS.format(definition=self.dataset.task.definition)
            + f"\n<stories count=\"{len(eligible)}\">\n{cards}\n</stories>\n\n"
            + f"<incoming>\n{query_card(query)}\n</incoming>\n\n"
            "Return the ids of the stories that cover the same development, best first, or an "
            "empty list if none does. Answer only with the JSON object."
        )
        payload, latency, cost, cached = await self.chat(
            {
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "matches", "strict": True, "schema": ANSWER_SCHEMA},
                },
            }
        )
        parsed = json.loads(payload["choices"][0]["message"]["content"])
        return Result(
            ids=self.clean(parsed.get("matches", []), eligible),
            latency_ms=latency,
            cost=cost,
            cached=cached,
            decided=True,
        )


AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_stories",
            "description": "Search existing stories. Returns the closest stories with ids, subjects and excerpts.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Free-text search query"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_matches",
            "description": "Final answer. Ids of stories covering the same development, best first; empty if none.",
            "parameters": {
                "type": "object",
                "properties": {"story_ids": {"type": "array", "items": {"type": "string"}}},
                "required": ["story_ids"],
            },
        },
    },
]


class AgenticRetriever(LLMRetriever):
    async def search(self, query: Query, eligible: list[str]) -> Result:
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": MATCH_INSTRUCTIONS.format(definition=self.dataset.task.definition)
                + f"\nYou can call search_stories up to {self.spec.max_rounds} times, then you must "
                "call submit_matches. Search for the specific development (product, version, event), "
                "not just the company.",
            },
            {"role": "user", "content": f"<incoming>\n{query_card(query)}\n</incoming>"},
        ]
        latency = 0.0
        costs: list[float | None] = []
        cached = True
        trace: list[dict[str, Any]] = []
        for round_ in range(self.spec.max_rounds + 1):
            final = round_ == self.spec.max_rounds
            choice: Any = (
                {"type": "function", "function": {"name": "submit_matches"}} if final else "required"
            )
            payload, call_ms, cost, hit = await self.chat(
                {"messages": messages, "tools": AGENT_TOOLS, "tool_choice": choice}
            )
            latency += call_ms
            costs.append(cost)
            cached = cached and hit
            message = payload["choices"][0]["message"]
            calls = message.get("tool_calls") or []
            if not calls:
                raise RuntimeError("agent answered without calling a tool")
            messages.append(
                {"role": "assistant", "content": message.get("content") or "", "tool_calls": calls}
            )
            for call in calls:
                name = call["function"]["name"]
                try:
                    args = json.loads(call["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                if name == "submit_matches":
                    ids = self.clean(args.get("story_ids", []), eligible)
                    trace.append({"submit": ids})
                    return Result(
                        ids=ids, latency_ms=latency, cost=_sum(costs), cached=cached,
                        decided=True, trace=trace,
                    )
                text = str(args.get("query", ""))[:500]
                probe = query.model_copy(
                    update={"id": f"{query.id}#search:{text}", "text": text, "title": None,
                            "subject": "", "version": None}
                )
                found = await self.arena.search(self.spec.tool, probe, eligible)
                latency += found.latency_ms or 0
                costs.append(found.cost)
                cached = cached and found.cached
                hits = found.ids[: self.spec.tool_k]
                trace.append({"search": text, "results": hits})
                content = "\n".join(
                    doc_card(self.dataset.doc(i), self.spec.excerpt_chars) for i in hits
                ) or "No results."
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})
        raise RuntimeError("agent never submitted")


# --- Mock -------------------------------------------------------------------------------------


class MockRetriever(Retriever):
    """Puts a relevant story first with probability `accuracy`; otherwise random order."""

    async def search(self, query: Query, eligible: list[str]) -> Result:
        rng = random.Random(f"{self.spec.name}:{query.id}")
        ids = list(eligible)
        rng.shuffle(ids)
        relevant = [i for i in (query.relevant or []) if i in eligible]
        if relevant and rng.random() < self.spec.accuracy:
            ids.remove(relevant[0])
            ids.insert(0, relevant[0])
        ids = ids[: self.spec.top_k]
        return Result(
            ids=ids,
            scores=[round(1 - r * 0.02, 3) for r in range(len(ids))],
            latency_ms=rng.uniform(20, 80),
            cost=0.0,
        )


KINDS: dict[str, type[Retriever]] = {
    "logged": LoggedRetriever,
    "production": ProductionRetriever,
    "bm25": BM25Retriever,
    "dense": DenseRetriever,
    "hybrid": HybridRetriever,
    "rerank": RerankRetriever,
    "long_context": LongContextRetriever,
    "agentic": AgenticRetriever,
    "mock": MockRetriever,
}


class Arena:
    """Holds one dataset's retrievers. Composites look up their parts by name, built on demand,
    and searches are memoized so a hybrid and its components never repeat the same call."""

    def __init__(self, dataset: Dataset, specs: dict[str, RetrieverSpec], client: Client):
        self.dataset = dataset
        self.specs = specs
        self.client = client
        self.retrievers: dict[str, Retriever] = {}
        self.ready: dict[str, asyncio.Task] = {}
        self.memo: dict[tuple[str, str], asyncio.Task] = {}

    def dependencies(self, name: str) -> list[str]:
        spec = self.specs[name]
        direct = [*spec.components, *([spec.base] if spec.base else []), *([spec.tool] if spec.tool else [])]
        out: list[str] = []
        for dep in direct:
            if dep not in self.specs:
                raise SystemExit(f"{name} references unknown retriever {dep}")
            out += [*self.dependencies(dep), dep]
        return out

    async def prepare(self, name: str) -> None:
        if name not in self.ready:
            retriever = KINDS[self.specs[name].kind](self.specs[name], self)
            self.retrievers[name] = retriever
            self.ready[name] = asyncio.ensure_future(retriever.prepare())
        await self.ready[name]
        for dep in self.dependencies(name):
            await self.prepare(dep)

    async def search(self, name: str, query: Query, eligible: list[str]) -> Result:
        key = (name, query.id)
        if key not in self.memo:
            self.memo[key] = asyncio.ensure_future(self._search(name, query, eligible))
        return await self.memo[key]

    async def _search(self, name: str, query: Query, eligible: list[str]) -> Result:
        await self.prepare(name)
        try:
            return await self.retrievers[name].search(query, eligible)
        except Exception as exc:  # noqa: BLE001 - recorded per query so one bad retriever can't sink a run
            return Result(error=f"{type(exc).__name__}: {exc}"[:400])
