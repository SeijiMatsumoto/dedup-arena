import asyncio
import json

import pytest
import yaml

from dedup_arena import runner
from dedup_arena.datasets import load_dataset
from dedup_arena.html_report import write_html
from dedup_arena.label import merge_labels
from dedup_arena.metrics import cutoff_sweep, e2e_metrics, mcnemar_p, ndcg, retrieval_metrics
from dedup_arena.recommend import recommend
from dedup_arena.retrievers import (
    Arena,
    RetrieverSpec,
    compatible,
    product_identity,
)
from dedup_arena.snapshot import render

DOCS = [
    {"id": "s_sdk3", "title": "OpenAI Python SDK 3.0 released", "subject": "OpenAI Python SDK",
     "version": "3.0", "item_type": "github_release", "excerpt": "typed tool calls",
     "available_at": "2026-10-01T00:00:00"},
    {"id": "s_gem", "title": "Gemini 3.5 Flash launches", "subject": "Gemini 3.5 Flash",
     "item_type": "article", "excerpt": "Google launches a faster Gemini model",
     "available_at": "2026-10-02T00:00:00"},
    {"id": "s_late", "title": "Something from the future", "subject": "Future thing",
     "excerpt": "should never be eligible", "available_at": "2026-12-01T00:00:00"},
]
QUERIES = [
    {"id": "q_sdk", "text": "Python client v3 for OpenAI adds typed tool-call responses.",
     "subject": "OpenAI Python SDK", "version": "3.0", "asked_at": "2026-10-05T00:00:00",
     "relevant": ["s_sdk3"], "label_source": "judges-agree", "tags": ["attach"],
     "production": {"action": "quiet", "story": "s_sdk3", "candidates": ["s_sdk3"], "logged": True}},
    {"id": "q_new", "text": "Mistral Large 4 preview tops a coding benchmark.", "subject": "Mistral Large",
     "asked_at": "2026-10-05T00:00:00", "relevant": [], "label_source": "judges-agree",
     "tags": ["new-event"], "exclude": ["s_gem"],
     "production": {"action": "new", "candidates": [], "logged": True}},
]


@pytest.fixture
def dataset(tmp_path):
    folder = tmp_path / "toy"
    folder.mkdir()
    (folder / "task.yaml").write_text(yaml.safe_dump(
        {"name": "toy", "description": "d", "use_case": "u", "definition": "Same event.\n"}))
    (folder / "corpus.jsonl").write_text("".join(json.dumps(d) + "\n" for d in DOCS))
    (folder / "queries.jsonl").write_text("".join(json.dumps(q) + "\n" for q in QUERIES))
    (folder / "queries.raw.jsonl").write_text("".join(json.dumps(q) + "\n" for q in QUERIES))
    return load_dataset(folder)


def test_eligible_respects_time_and_exclude(dataset):
    q_sdk, q_new = dataset.queries
    assert dataset.eligible(q_sdk) == ["s_sdk3", "s_gem"]
    assert dataset.eligible(q_new) == ["s_sdk3"]


def test_production_helpers(dataset):
    assert product_identity("Gemini 3") == "gemini 3"
    assert product_identity("model 3") is None
    q = dataset.queries[0].model_copy(update={"version": "4.0"})
    assert not compatible(q, dataset.doc("s_sdk3"))  # same subject, different version
    assert compatible(dataset.queries[0], dataset.doc("s_sdk3"))


def test_bm25_and_logged(dataset):
    specs = {s.name: s for s in [RetrieverSpec(name="bm25", kind="bm25"),
                                 RetrieverSpec(name="log", kind="logged")]}

    async def go():
        arena = Arena(dataset, specs, client=None)
        q = dataset.queries[0]
        return (await arena.search("bm25", q, dataset.eligible(q)),
                await arena.search("log", q, dataset.eligible(q)))

    bm25, logged = asyncio.run(go())
    assert bm25.ids[0] == "s_sdk3"
    assert logged.ids == ["s_sdk3"]


def test_metrics():
    assert ndcg(["a", "b"], ["b"]) == pytest.approx(1 / 1.585, rel=1e-3)
    assert mcnemar_p(0, 0) == 1.0
    assert mcnemar_p(10, 0) < 0.01


def test_retrieval_and_e2e_metrics(dataset):
    q_sdk, q_new = dataset.queries
    results = {q_sdk.id: {"ids": ["s_gem", "s_sdk3"], "scores": [0.95, 0.85], "latency_ms": 10},
               q_new.id: {"ids": [], "latency_ms": 5}}
    m = retrieval_metrics(dataset.queries, results)
    assert m["recall@1"] == 0 and m["recall@3"] == 1 and m["mrr"] == 0.5
    assert m["new_empty_rate"] == 1
    sweep = {r["max_distance"]: r for r in cutoff_sweep(dataset.queries, results)}
    assert sweep[0.1]["recall@5"] == 0 and sweep[0.15]["recall@5"] == 1

    e = e2e_metrics(dataset.queries, {q_sdk.id: {"pred": None}, q_new.id: {"pred": "s_sdk3"}})
    assert e["accuracy"] == 0 and e["duplicate_rate"] == 1 and e["wrong_merge_rate"] == 0.5


def test_merge_labels(dataset):
    judgments = [
        {"query": "q_sdk", "pool": ["s_sdk3"], "judges": {"a": {"matches": ["s_sdk3"], "reason": ""},
                                                          "b": {"matches": ["s_sdk3"], "reason": ""}}},
        {"query": "q_new", "pool": ["s_sdk3"], "judges": {"a": {"matches": [], "reason": ""},
                                                          "b": {"matches": ["s_sdk3"], "reason": ""}}},
    ]
    labeled, disputed = merge_labels(dataset, judgments, {})
    assert [r["id"] for r in labeled] == ["q_sdk"] and [d["query"] for d in disputed] == ["q_new"]
    labeled, disputed = merge_labels(dataset, judgments, {"q_new": {"relevant": []}})
    assert {r["id"]: r["label_source"] for r in labeled} == {"q_sdk": "judges-agree", "q_new": "reviewed"}
    assert "new-event" in labeled[1]["tags"]

    # One judge: production's decision is the second opinion.
    single = [{"query": "q_sdk", "pool": [], "judges": {"a": {"matches": ["s_sdk3"], "reason": ""}}},
              {"query": "q_new", "pool": [], "judges": {"a": {"matches": ["s_sdk3"], "reason": ""}}}]
    labeled, disputed = merge_labels(dataset, single, {})
    assert [(r["id"], r["label_source"]) for r in labeled] == [("q_sdk", "judge+production")]
    assert [d["query"] for d in disputed] == ["q_new"]


def test_offline_run_report_and_snapshot(dataset, tmp_path, monkeypatch):
    async def fake_decide(client, dataset, query, result, model):
        return {"pred": result.ids[0] if result.ids else None, "latency_ms": 1.0, "cost": 0.0}

    monkeypatch.setattr(runner, "decide", fake_decide)
    specs = {s.name: s for s in [RetrieverSpec(name="mock-good", kind="mock", accuracy=1.0),
                                 RetrieverSpec(name="bm25", kind="bm25")]}
    report = asyncio.run(runner.run_eval([dataset], specs, list(specs.values()), tmp_path / "cache"))
    data = report["datasets"]["toy"]
    assert data["metrics"]["mock-good"]["recall@1"] == 1
    assert "production-actual" in data["e2e"]
    assert data["e2e"]["production-actual"]["metrics"]["accuracy"] == 1
    rec = recommend(report)["toy"]
    assert rec["picks"]
    assert "# Results snapshot" in render(report)
    page = write_html(report, tmp_path / "r.html")
    assert "<svg" in page.read_text()
