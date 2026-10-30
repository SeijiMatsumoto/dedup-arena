"""Turn a report into picks with fixed rules and no LLM, so the same report always reads the same.

1. Contenders: retrievers within MARGIN of the best end-to-end accuracy (or recall@5, for a
   retrieval-only run) that a paired exact McNemar test does not show to be significantly
   worse (p < ALPHA).
2. Use: the contender with the best combined cost + latency rank (accuracy breaks ties).
3. Avoid: significantly worse than the best, or with failed calls.
4. Notes: the production baseline, what the distance cutoff costs, long context vs retrieval,
   best embedding model, reranker gain, weak scenario tags.
"""

from __future__ import annotations

from typing import Any

from .metrics import paired_p

MARGIN = 0.02
ALPHA = 0.05
WEAK_TAG = 0.8
MIN_TAG_N = 5


def _ms(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v / 1000:.1f}s" if v >= 1000 else f"{v:.0f}ms"


def _usd(v: float | None) -> str:
    return "n/a" if v is None else f"${v:.3f}"


def _ratio(a: float | None, b: float | None) -> str | None:
    if not a or not b:
        return None
    r = a / b
    return f"{r:.0f}×" if r >= 10 else f"{r:.1f}×"


def _scores(data: dict[str, Any], kinds: dict[str, dict[str, Any]]) -> tuple[str, dict[str, dict]]:
    """The metric picks are made on: end-to-end accuracy when the run has it, else recall@5.

    Logged production candidates are a baseline to beat, not something you can deploy.
    """
    deployable = {k for k in kinds if kinds[k]["kind"] != "logged"}
    e2e = {k: v["metrics"] for k, v in data["e2e"].items() if k in deployable}
    if e2e:
        return "e2e", {k: {"score": m["accuracy"], "correct": m["correct"], "p50": m["latency_p50_ms"],
                           "cost": m["cost_per_1k"], "errors": m["errors"], "by_tag": m["by_tag"],
                           "mrr": data["metrics"].get(k, {}).get("mrr")}
                       for k, m in e2e.items()}
    return "retrieval", {
        k: {"score": m["recall@5"], "correct": m["hit5"], "p50": m["latency_p50_ms"],
            "cost": m["cost_per_1k"], "errors": m["errors"], "by_tag": m["by_tag"], "mrr": m["mrr"]}
        for k, m in data["metrics"].items() if k in deployable and m["recall@5"] is not None
    }


def recommend_dataset(data: dict[str, Any], kinds: dict[str, dict[str, Any]]) -> dict[str, Any]:
    mode, sc = _scores(data, kinds)
    retrieval = data["metrics"]
    e2e = {k: v["metrics"] for k, v in data["e2e"].items()}
    rec: dict[str, Any] = {"mode": mode, "picks": [], "avoid": [], "notes": []}
    if not sc:
        return rec
    # MRR breaks ties: recall@5 saturates when several retrievers find nearly everything.
    ranked = sorted(sc, key=lambda k: (-(sc[k]["score"] or 0), -(sc[k]["mrr"] or 0)))
    best = ranked[0]
    top = sc[best]["score"] or 0
    contenders, worse = [], []
    for name in ranked:
        m = sc[name]
        p = paired_p(m["correct"], sc[best]["correct"]) if name != best else 1.0
        if name != best and p < ALPHA:
            worse.append((name, p))
        elif (m["score"] or 0) >= top - MARGIN and not m["errors"]:
            contenders.append(name)
    if not contenders:
        contenders = [best]

    def rank_of(key: str) -> dict[str, int]:
        values = sorted(contenders, key=lambda n: (sc[n][key] is None, sc[n][key] or 0))
        return {n: i for i, n in enumerate(values)}

    cost_rank, lat_rank = rank_of("cost"), rank_of("p50")
    use = min(contenders, key=lambda n: (cost_rank[n] + lat_rank[n], -(sc[n]["score"] or 0),
                                         -(sc[n]["mrr"] or 0)))
    label = "end-to-end" if mode == "e2e" else "recall@5"

    def describe(name: str, role: str) -> dict[str, Any]:
        m, r = sc[name], retrieval.get(name, {})
        if mode == "e2e":
            e = e2e[name]
            detail = (f"{e['accuracy']:.1%} end-to-end ({e['accuracy_attach']:.0%} on attach, "
                      f"{e['accuracy_new']:.0%} on new), recall@5 {r['recall@5']:.0%}")
        else:
            detail = (f"recall@5 {r['recall@5']:.1%}, right story first {r['recall@1']:.0%} of the "
                      f"time (MRR {r['mrr']:.2f})")
        return {
            "role": role, "retriever": name, "family": kinds[name]["family"], "metric": label,
            "accuracy": m["score"], "recall@5": r.get("recall@5"), "recall@1": r.get("recall@1"),
            "p50_ms": m["p50"], "cost_per_1k": m["cost"],
            "why": f"{detail}, {_ms(m['p50'])} median, {_usd(m['cost'])} per 1k items.",
        }

    rec["picks"].append(describe(use, "recommended"))
    if best != use:
        rec["picks"].append(describe(best, "most_accurate"))
    for role, key in (("fastest", "p50"), ("cheapest", "cost")):
        known = [n for n in contenders if sc[n][key] is not None]
        if known:
            pick = min(known, key=lambda n: sc[n][key])
            if pick not in {p["retriever"] for p in rec["picks"]}:
                rec["picks"].append(describe(pick, role))
    for name, p in worse:
        rec["avoid"].append({"retriever": name, "why": (
            f"{label} {sc[name]['score']:.1%} vs {top:.1%} for {best} (paired McNemar p={p:.3f})."
        )})
    for name in ranked:
        if sc[name]["errors"] and name not in {a["retriever"] for a in rec["avoid"]}:
            rec["avoid"].append({"retriever": name, "why": f"{sc[name]['errors']} failed calls."})

    notes = rec["notes"]
    actual = e2e.get("production-actual")
    if actual and actual["accuracy"] is not None:
        notes.append(
            f"Production today (its logged decisions): {actual['accuracy']:.1%} correct end to end; "
            f"{actual['duplicate_rate']:.0%} of items that belong to an existing story became a "
            f"duplicate story, and {actual['wrong_merge_rate']:.0%} of all items were merged into the "
            "wrong story."
        )
    logged = retrieval.get("production-logged")
    if logged and logged["recall@5"] is not None:
        p = paired_p(logged["hit5"], retrieval[best]["hit5"]) if best in retrieval else None
        notes.append(
            f"Production's logged candidates contained the right story for {logged['recall@5']:.1%} "
            f"of matching items, so {1 - logged['recall@5']:.0%} never reached the judge. "
            f"{best} reaches {retrieval[best]['recall@5']:.1%}"
            + (f" (paired McNemar p={p:.4f})." if p is not None else ".")
        )
        weak = [f"{t.removeprefix('type:')} ({v['rate']:.0%}, n={v['n']})"
                for t, v in logged["by_tag"].items()
                if t not in ("attach", "new-event") and not t.startswith("prod-")
                and v["n"] >= MIN_TAG_N and v["rate"] is not None and v["rate"] < WEAK_TAG]
        if weak:
            notes.append("Production retrieval is weakest on: " + ", ".join(weak) + ".")
    if "production-replay" in retrieval and "gemini-emb-2" in retrieval:
        a, b = retrieval["production-replay"]["recall@5"], retrieval["gemini-emb-2"]["recall@5"]
        if a is not None and b is not None:
            notes.append(
                f"The 0.10 cosine cutoff: production's candidate query (re-implemented) reaches "
                f"recall@5 {a:.0%}; the same embeddings without the cutoff reach {b:.0%}."
            )
        sweep = data["sweeps"].get("gemini-emb-2")
        if sweep and b is not None:
            at = next((s for s in sweep if abs(s["max_distance"] - 0.10) < 1e-9), None)
            good = [s for s in sweep if s["recall@5"] is not None and s["recall@5"] >= b - 0.005]
            if good:
                s = good[0]
                notes.append(
                    f"Smallest cutoff that keeps gemini-emb-2's full recall@5: distance ≤ "
                    f"{s['max_distance']:.2f}, passing {s['candidates_mean']:.1f} candidates per item "
                    "on average" + (f" (vs {at['candidates_mean']:.1f} at 0.10)." if at else ".")
                )
    lc = [n for n in e2e if n in kinds and kinds[n]["family"] == "long context"]
    rag = [n for n in e2e if n in kinds
           and kinds[n]["family"] not in ("long context", "agentic", "production")]
    if lc and rag:
        best_lc = max(lc, key=lambda n: e2e[n]["accuracy"] or 0)
        best_rag = max(rag, key=lambda n: e2e[n]["accuracy"] or 0)
        a, b = e2e[best_lc], e2e[best_rag]
        p = paired_p(a["correct"], b["correct"])
        verdict = "no significant difference" if p >= ALPHA else (
            "long context is significantly better" if a["accuracy"] > b["accuracy"]
            else "retrieval is significantly better"
        )
        slower = (a["latency_p50_ms"] or 0) >= (b["latency_p50_ms"] or 0)
        speed = (_ratio(a["latency_p50_ms"], b["latency_p50_ms"]) if slower
                 else _ratio(b["latency_p50_ms"], a["latency_p50_ms"]))
        cost = _ratio(a["cost_per_1k"], b["cost_per_1k"])
        notes.append(
            f"Long context vs retrieval: {best_lc} {a['accuracy']:.1%} vs {best_rag} "
            f"{b['accuracy']:.1%} ({verdict}, p={p:.3f})"
            + (f"; long context is {speed} {'slower' if slower else 'faster'}" if speed else "")
            + (f" and {cost} the cost" if cost else "") + "."
        )
    dense = [n for n in retrieval if kinds.get(n, {}).get("family") == "dense"]
    if dense:
        top_dense = max(dense, key=lambda n: (retrieval[n]["recall@5"] or 0, retrieval[n]["mrr"] or 0))
        m = retrieval[top_dense]
        notes.append(
            f"Best embedding setup: {top_dense} (recall@5 {m['recall@5']:.0%}, recall@1 "
            f"{m['recall@1']:.0%}, MRR {m['mrr']:.2f})."
        )
    for name, spec in kinds.items():
        if spec["kind"] == "rerank" and spec.get("base") in retrieval and name in retrieval:
            gain = (retrieval[name]["mrr"] or 0) - (retrieval[spec["base"]]["mrr"] or 0)
            if abs(gain) >= 0.02:
                notes.append(f"{name} changes MRR by {gain:+.2f} over {spec['base']}.")
    if mode == "e2e":
        weak = [f"{t} ({v['rate']:.0%}, n={v['n']})" for t, v in sc[use]["by_tag"].items()
                if v["n"] >= MIN_TAG_N and v["rate"] is not None and v["rate"] < WEAK_TAG]
        if weak:
            notes.append(f"{use} is weakest on: " + ", ".join(weak) + ".")
    return rec


def recommend(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    kinds = {r["name"]: r for r in report["retrievers"]}
    return {name: recommend_dataset(data, kinds) for name, data in report["datasets"].items()}
