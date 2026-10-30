"""arena-summary/1 for the arenas hub (https://github.com/SeijiMatsumoto/arenas).

Maps a report onto the shared contract with the same scores and picks as the snapshot README.
Production rows are baselines to beat, not deployable contenders. Hand-written fields come from
arena.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from arena_kit import contract

from .metrics import paired_p
from .recommend import ALPHA, MARGIN, _scores, recommend
from .snapshot import headline, snapshot_name

RULES = (
    f"Tied = within {MARGIN:.0%} of the best score and not significantly worse "
    f"(exact McNemar, p < {ALPHA}); pick = best combined cost and latency rank among tied. "
    "Score is end-to-end accuracy when the run has it, else recall@5."
)


def build_summary(report: dict[str, Any], root: Path, folder: Path) -> dict[str, Any]:
    name, data = next(iter(report["datasets"].items()))
    kinds = {r["name"]: r for r in report["retrievers"]}
    mode, sc = _scores(data, kinds)
    rec = recommend(report)[name]
    pick = rec["picks"][0] if rec["picks"] else None
    avoid = {a["retriever"] for a in rec["avoid"]}
    best = max(sc, key=lambda k: (sc[k]["score"] or 0, sc[k]["mrr"] or 0)) if sc else None

    def verdict(k: str) -> str:
        if pick and k == pick["retriever"]:
            return "pick"
        if sc[k]["errors"]:
            return "excluded"
        if k in avoid:
            return "worse"
        tied = (sc[k]["score"] or 0) >= (sc[best]["score"] or 0) - MARGIN and (
            k == best or paired_p(sc[k]["correct"], sc[best]["correct"]) >= ALPHA)
        return "tied" if tied else "other"

    contenders = [{"name": k, "kind": kinds[k]["family"], "quality": s["score"],
                   "latency_ms": s["p50"], "cost_per_1k_usd": s["cost"], "errors": s["errors"],
                   "verdict": verdict(k),
                   **({"note": "Production's query, replayed."}
                      if kinds[k]["kind"] == "production" else {})} for k, s in sc.items()]
    for k, r in kinds.items():  # logged production candidates: the baseline, never deployable
        m = data["metrics"].get(k)
        if k not in sc and m and m.get("recall@5") is not None:
            contenders.append({"name": k, "kind": r["family"], "quality": m["recall@5"],
                               "latency_ms": None, "cost_per_1k_usd": None,
                               "errors": m.get("errors", 0), "verdict": "baseline",
                               "note": "Production's logged candidates, as recorded."})
    contenders.sort(key=lambda c: (-(c["quality"] or 0), c["name"]))
    label = "End-to-end accuracy" if mode == "e2e" else "Right story in top 5"
    return contract.build(
        contract.load_manifest(root), snapshot_id=snapshot_name(report),
        created_at=report["created_at"], snapshot_path=folder.relative_to(root).as_posix(),
        headline=headline(report),
        pick={"name": pick["retriever"], "reason": pick["why"]} if pick else None,
        metric={"name": "accuracy" if mode == "e2e" else "recall@5", "label": label,
                "higher_is_better": True, "format": "percent"},
        contenders=contenders, items=len(sc[best]["correct"]) if best else 0, rules=RULES,
        mock=all(r.get("kind") == "mock" for r in report["retrievers"]),
    )


def write_summary(report: dict[str, Any], root: Path, folder: Path) -> Path:
    return contract.write(root, folder, build_summary(report, root, folder))
