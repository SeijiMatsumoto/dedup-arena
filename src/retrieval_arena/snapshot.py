"""Dated, committed snapshots of eval runs.

Each snapshot is a folder `snapshots/<YYYY-MM-DD_HHMM>/` holding the full report (`report.json`),
the corpus-size sweep if one was run (`scale.json`), and a rendered summary (`README.md`).
`snapshots/README.md` indexes every snapshot, and the project README points at the latest one.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .recommend import recommend

START = "<!-- snapshot:start -->"
END = "<!-- snapshot:end -->"
LABELS = {"recommended": "Use", "most_accurate": "Most accurate", "fastest": "Fastest",
          "cheapest": "Cheapest"}


def _pct(v: float | None, digits: int = 0) -> str:
    return "—" if v is None else f"{v:.{digits}%}"


def _ms(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v / 1000:.1f}s" if v >= 1000 else f"{v:.0f}ms"


def _usd(v: float | None) -> str:
    if v is None:
        return "—"
    return f"${v:.2f}" if v >= 1 else f"${v:.4f}"


def _num(v: float | None) -> str:
    return "—" if v is None else f"{v:.2f}"


def snapshot_name(report: dict[str, Any]) -> str:
    return datetime.fromisoformat(report["created_at"]).strftime("%Y-%m-%d_%H%M")


def headline(report: dict[str, Any]) -> str:
    name, _data = next(iter(report["datasets"].items()))
    rec = recommend(report)[name]
    if not rec["picks"]:
        return "No successful answers in this run."
    pick = rec["picks"][0]
    if rec["mode"] == "e2e":
        quality = f"{pick['accuracy']:.0%} end-to-end, recall@5 {pick['recall@5']:.0%}"
    else:
        quality = f"recall@5 {pick['recall@5']:.0%}, right story first {pick['recall@1']:.0%}"
    text = (
        f"Use **{pick['retriever']}** ({quality}, {_ms(pick['p50_ms'])} median, "
        f"{_usd(pick['cost_per_1k'])} per 1k)."
    )
    extras = [n for n in rec["notes"]
              if n.startswith(("Production today", "Production's logged candidates", "Long context vs"))]
    return " ".join([text, *extras])


def scale_table(scale: dict[str, Any]) -> list[str]:
    sizes = scale["sizes"]
    lines = [
        "| Retriever | " + " | ".join(f"{s} stories" for s in sizes) + " |",
        "|---|" + "---:|" * len(sizes),
    ]
    for name, rows in scale["retrievers"].items():
        cells = [
            f"{_pct(r['accuracy'], 1)} · {_ms(r['latency_p50_ms'])} · {_usd(r['cost_per_1k'])}"
            for r in rows
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


def scenario_table(data: dict[str, Any], shown: list[str]) -> list[str]:
    """Recall@5 per scenario tag, over the items that belong to an existing story."""
    attach = [x for x in data["queries"] if x["relevant"]]
    tags = sorted({t for x in attach for t in x["tags"] if t != "attach"})
    lines = ["", "### Recall@5 by scenario (items that belong to an existing story)", "",
             "| Tag | n | " + " | ".join(shown) + " |", "|---|---:|" + "---:|" * len(shown)]
    for tag in tags:
        items = [x for x in attach if tag in x["tags"]]
        cells = []
        for ret in shown:
            res = data["results"].get(ret, {})
            hits = [any(i in x["relevant"] for i in res[x["id"]].get("ids", [])[:5])
                    for x in items if x["id"] in res and not res[x["id"]].get("skipped")]
            cells.append(_pct(sum(hits) / len(hits)) if hits else "—")
        lines.append(f"| {tag.removeprefix('type:')} | {len(items)} | " + " | ".join(cells) + " |")
    return lines


def render(report: dict[str, Any], scale: dict[str, Any] | None = None,
           project_readme: str = "../../README.md") -> str:
    date = datetime.fromisoformat(report["created_at"]).strftime("%Y-%m-%d %H:%M UTC")
    recs = recommend(report)
    kinds = {r["name"]: r for r in report["retrievers"]}
    lines = [f"# Results snapshot: {date}", ""]
    for name, data in report["datasets"].items():
        q = data["queries"]
        attach = sum(bool(x["relevant"]) for x in q)
        sources: dict[str, int] = {}
        for x in q:
            sources[x["label_source"]] = sources.get(x["label_source"], 0) + 1
        label_text = ", ".join(f"{v} {k}" for k, v in sorted(sources.items()))
        lines += [
            f"## {name}",
            "",
            (
                f"{len(q)} real items from enable-ai ({attach} that belong to an existing story, "
                f"{len(q) - attach} that start a new one), searched against the stories that existed "
                f"when each item arrived (median {data['eligible_median']} of {data['n_corpus']}). "
                f"Labels: {label_text} (see [Labels]({project_readme}#labels)). "
                + (f"End-to-end = top 5 → `{report['judge_model']}` picks a story or none; "
                   "long-context and agentic retrievers decide on their own. "
                   if report.get("judge_model") else
                   "This run is retrieval-only (no judge calls); `production-actual`'s end-to-end "
                   "numbers come from production's own logs. ")
                + "Latency is the client round trip from one machine. "
                "Raw data: [`report.json`](report.json)."
            ),
            "",
            ("| Retriever | Family | Recall@1 | Recall@5 | MRR | End-to-end | Attach | New | Dupes "
             "| Wrong merge | p50 | $ per 1k |"),
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        names = list(dict.fromkeys([*data["metrics"], *data["e2e"]]))

        def sort_key(n: str, data: dict[str, Any] = data) -> tuple:
            e = (data["e2e"].get(n) or {}).get("metrics", {})
            r = data["metrics"].get(n, {})
            return (-(e.get("accuracy") or 0) if n in kinds and e else 0,
                    -(r.get("recall@5") or 0), -(r.get("mrr") or 0))

        order = sorted(names, key=sort_key)
        for ret in order:
            e = (data["e2e"].get(ret) or {}).get("metrics", {})
            r = data["metrics"].get(ret, {})
            family = kinds[ret]["family"] if ret in kinds else "production"
            label = f"**{ret}**" if ret == "production-actual" else ret
            lines.append(
                f"| {label} | {family} | {_pct(r.get('recall@1'))} | {_pct(r.get('recall@5'), 1)} "
                f"| {_num(r.get('mrr'))} | {_pct(e.get('accuracy'), 1)} | {_pct(e.get('accuracy_attach'))} "
                f"| {_pct(e.get('accuracy_new'))} | {_pct(e.get('duplicate_rate'))} "
                f"| {_pct(e.get('wrong_merge_rate'))} "
                f"| {_ms(e.get('latency_p50_ms', r.get('latency_p50_ms')))} "
                f"| {_usd(e.get('cost_per_1k', r.get('cost_per_1k')))} |"
            )
        no_e2e = [n for n in data["metrics"] if n not in data["e2e"]]
        lines += [
            "",
            ("Recall is measured on the items that belong to an existing story. *Dupes*: items that "
             "belong to a story but became a new one. *Wrong merge*: items attached to a story that "
             "doesn't cover them. `production-actual` is what enable-ai actually decided (from its "
             "logs), scored against the same labels."
             + (" End-to-end columns are empty for retrievers this run did not send to the judge."
                if no_e2e else "")),
            "",
            f"### Takeaways ([how these are derived]({project_readme}#takeaways))",
            "",
        ]
        rec = recs[name]
        for pick in rec["picks"]:
            lines.append(f"- **{LABELS[pick['role']]}: {pick['retriever']}.** {pick['why']}")
        for item in rec["avoid"]:
            lines.append(f"- **Avoid: {item['retriever']}.** {item['why']}")
        for note in rec["notes"]:
            lines.append(f"- {note}")

        sweep = data["sweeps"].get("gemini-emb-2")
        if sweep:
            lines += [
                "",
                "### Distance cutoff sweep (gemini-emb-2, production's embedding text)",
                "",
                "Production uses 0.10. Candidates are capped at 5, like production.",
                "",
                "| Max cosine distance | Recall@5 | Avg candidates passed on | New items with no candidate |",
                "|---:|---:|---:|---:|",
            ]
            for sw in sweep:
                lines.append(
                    f"| {sw['max_distance']:.2f} | {_pct(sw['recall@5'])} | "
                    f"{sw['candidates_mean']:.1f} | {_pct(sw['new_empty_rate'])} |"
                )
        lines += scenario_table(data, [n for n in order if n in data["metrics"]][:8])
    if scale:
        lines += [
            "",
            "## Corpus size (does long context hold up as the corpus grows?)",
            "",
            ("Same queries, with each one's eligible stories cut to N (relevant stories always kept). "
            "Cells: end-to-end accuracy · median latency · $ per 1k items. Raw: [`scale.json`](scale.json)."),
            "",
            *scale_table(scale),
        ]
    return "\n".join(lines) + "\n"


def save_snapshot(report: dict[str, Any], snapshots_dir: Path, scale: dict[str, Any] | None) -> Path:
    folder = snapshots_dir / snapshot_name(report)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "report.json").write_text(json.dumps(report, separators=(",", ":")))
    if scale:
        (folder / "scale.json").write_text(json.dumps(scale, indent=1))
    (folder / "README.md").write_text(render(report, scale))
    return folder


def load_snapshots(snapshots_dir: Path) -> list[tuple[str, dict[str, Any]]]:
    return [
        (path.parent.name, json.loads(path.read_text()))
        for path in sorted(snapshots_dir.glob("*/report.json"), reverse=True)
    ]


def write_index(snapshots_dir: Path) -> None:
    lines = [
        "# Snapshots",
        "",
        "Dated eval runs, newest first. Each folder has a readable summary and the full `report.json`.",
        "Create one from your latest run with `uv run arena snapshot`.",
        "",
        "| Date | Retrievers | Queries | Headline |",
        "|---|---:|---:|---|",
    ]
    for name, report in load_snapshots(snapshots_dir):
        n = sum(len(d["queries"]) for d in report["datasets"].values())
        lines.append(f"| [{name}]({name}/) | {len(report['retrievers'])} | {n} | {headline(report)} |")
    (snapshots_dir / "README.md").write_text("\n".join(lines) + "\n")


def write_readme_pointer(readme: Path, snapshots_dir: Path) -> None:
    snapshots = load_snapshots(snapshots_dir)
    if not snapshots:
        return
    name, report = snapshots[0]
    rel = snapshots_dir.relative_to(readme.parent).as_posix()
    date = datetime.fromisoformat(report["created_at"]).strftime("%Y-%m-%d")
    block = "\n".join([
        START,
        "## Latest results",
        "",
        f"**[{date} snapshot]({rel}/{name}/)**: {headline(report)}",
        "",
        (f"Full tables, scenario breakdowns and the distance-cutoff sweep are in the snapshot; all "
        f"runs are listed in [`{rel}/`]({rel}/)."),
        END,
    ])
    text = readme.read_text()
    if START in text and END in text:
        text = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: block, text, flags=re.DOTALL)
    else:
        text = f"{text}\n{block}\n"
    readme.write_text(text)
