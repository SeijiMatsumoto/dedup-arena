"""Dedup Arena: compare retrieval strategies on real, labeled dedup queries."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from . import label as labeling
from .client import Client
from .datasets import load_datasets, write_jsonl
from .html_report import write_html
from .recommend import recommend
from .retrievers import load_specs, select_specs
from .runner import run_eval, save_report
from .snapshot import render, save_snapshot, scale_table, write_index, write_readme_pointer
from .summary import write_summary

PROJECT_ROOT = Path(__file__).resolve().parents[2]
console = Console()


def client_for(http: httpx.AsyncClient, cache_only: bool = False) -> Client:
    return Client(PROJECT_ROOT / ".cache", http, cache_only=cache_only)


def progress_run(label: str, coro_factory):
    with Progress(console=console) as progress:
        bar = progress.add_task(label, total=None)

        def on_progress(done: int, total: int) -> None:
            progress.update(bar, completed=done, total=total)

        return asyncio.run(coro_factory(on_progress))


def cmd_label(args: argparse.Namespace) -> None:
    specs = load_specs(args.retrievers_file)
    for dataset in load_datasets(args.datasets_dir, args.datasets, labeled=False):
        folder = dataset.path
        if args.limit:
            dataset.queries = dataset.queries[: args.limit]
        if args.sheet:
            done = {j["query"] for j in labeling.load_judgments(folder)}
            todo = [q for q in dataset.queries if q.id not in done]

            async def pools(dataset=dataset, todo=todo):
                async with httpx.AsyncClient(timeout=180) as http:
                    return await labeling.collect_pools(dataset, specs, client_for(http), todo)

            paths = labeling.write_agent_sheets(dataset, asyncio.run(pools()), folder)
            console.print(f"{len(todo)} unlabeled items -> " + ", ".join(
                str(p.relative_to(PROJECT_ROOT)) for p in paths))
            continue
        if not args.merge_only:
            async def go(on_progress, dataset=dataset):
                async with httpx.AsyncClient(timeout=180) as http:
                    return await labeling.collect_judgments(
                        dataset, specs, client_for(http, args.cached_only), on_progress)

            judgments = progress_run(f"Judging {dataset.name}", go)
            labeling.save(folder, judgments)
        judgments = labeling.load_judgments(folder)
        judged = {j["query"] for j in judgments}
        judgments += [j for j in labeling.agent_judgments(folder) if j["query"] not in judged]
        adjudications = labeling.load_adjudications(folder / "adjudications.yaml")
        labeled, disputed = labeling.merge_labels(dataset, judgments, adjudications)
        write_jsonl(folder / "queries.jsonl", labeled)
        labeling.write_review_sheet(dataset, disputed, folder / "review.md")
        cost = sum((v.get("cost") or 0) for j in judgments for v in j["judges"].values())
        missing = len(dataset.queries) - len(labeled) - len(disputed)
        attach = sum(bool(r["relevant"]) for r in labeled)
        console.print(
            f"[bold]{dataset.name}[/bold]: {len(labeled)} labeled ({attach} attach, "
            f"{len(labeled) - attach} new), {len(disputed)} disputed -> review.md, "
            f"{missing} not judged yet. "
            f"Judge cost ${cost:.2f}"
        )


def fmt(value: Any, kind: str = "pct") -> str:
    if value is None:
        return "-"
    if kind == "pct":
        return f"{value:.0%}"
    if kind == "ms":
        return f"{value:.0f}"
    if kind == "usd":
        return f"${value:.4f}"
    return f"{value:.2f}"


ROLE_LABELS = {"recommended": "Use", "most_accurate": "Most accurate", "fastest": "Fastest",
               "cheapest": "Cheapest"}


def print_report(report: dict[str, Any], show_misses: bool) -> None:
    recs = recommend(report)
    for name, data in report["datasets"].items():
        table = Table(
            title=f"{name}  [dim](n={len(data['queries'])}, judge {report.get('judge_model')})[/dim]",
            box=box.ROUNDED, title_justify="left", header_style="bold",
        )
        table.add_column("Retriever", no_wrap=True)
        for col in ("E2E", "Attach", "New", "Dupes", "Wrong", "R@1", "R@5", "MRR", "p50 ms", "$/1k",
                    "Err"):
            table.add_column(col, justify="right")
        order = sorted(data["e2e"] or data["metrics"],
                       key=lambda n: -((data["e2e"].get(n) or {}).get("metrics", {}).get("accuracy")
                                       or data["metrics"].get(n, {}).get("recall@5") or 0))
        for ret in order:
            e = (data["e2e"].get(ret) or {}).get("metrics", {})
            r = data["metrics"].get(ret, {})
            table.add_row(
                ret, fmt(e.get("accuracy")), fmt(e.get("accuracy_attach")), fmt(e.get("accuracy_new")),
                fmt(e.get("duplicate_rate")), fmt(e.get("wrong_merge_rate")), fmt(r.get("recall@1")),
                fmt(r.get("recall@5")), fmt(r.get("mrr"), "num"),
                fmt(e.get("latency_p50_ms", r.get("latency_p50_ms")), "ms"),
                fmt(e.get("cost_per_1k", r.get("cost_per_1k")), "usd"),
                str((e.get("errors") or r.get("errors")) or ""),
            )
        console.print(table)
        for ret in order:
            first = next((x["error"] for x in data["results"].get(ret, {}).values() if x.get("error")), None)
            if first:
                console.print(f"  [red]{ret}[/red] first error: {first[:200]}")
        if show_misses and data["e2e"]:
            shown = order[:6]
            for q in data["queries"]:
                wrong = [s for s in shown if not data["e2e"][s]["metrics"]["correct"].get(q["id"], True)]
                if len(wrong) >= max(2, len(shown) // 2):
                    console.print(f"  [yellow]{q['id']}[/yellow] {q['headline'][:70]!r} expected "
                                  f"{q['relevant'] or 'new'} -> wrong: {', '.join(wrong)}")
        rec = recs[name]
        console.rule("Takeaways")
        for pick in rec["picks"]:
            console.print(f"  [green]{ROLE_LABELS[pick['role']]:>13}[/green]  {pick['retriever']}: {pick['why']}")
        for item in rec["avoid"]:
            console.print(f"  [red]{'Avoid':>13}[/red]  {item['retriever']}: {item['why']}")
        for note in rec["notes"]:
            console.print(f"  [dim]{'Note':>13}  {note}[/dim]")


def latest(prefix: str) -> Path | None:
    return max((PROJECT_ROOT / "reports").glob(f"{prefix}-*.json"), default=None)


def cmd_run(args: argparse.Namespace) -> None:
    datasets = load_datasets(args.datasets_dir, args.datasets)
    specs = load_specs(args.retrievers_file)
    selected = select_specs(specs, args.retrievers)
    for d in datasets:
        if args.limit:
            d.queries = d.queries[: args.limit]
    console.print(f"Running [bold]{len(selected)}[/bold] retrievers over "
                  f"[bold]{sum(len(d.queries) for d in datasets)}[/bold] queries"
                  + (f", corpus capped at {args.corpus_size}" if args.corpus_size else ""))
    report = progress_run("Searching", lambda cb: run_eval(
        datasets, specs, selected, PROJECT_ROOT / ".cache", corpus_size=args.corpus_size,
        e2e=not args.no_e2e, judge_model=args.judge, on_progress=cb, cache_only=args.cached_only))
    out = args.out or save_report(report, PROJECT_ROOT / "reports")
    if args.out:
        out.write_text(json.dumps(report, indent=1))
    page = write_html(report, out.with_suffix(".html"))
    print_report(report, args.misses)
    console.print(f"\nReport: {out}\nHTML:   {page}")


def cmd_scale(args: argparse.Namespace) -> None:
    """Same queries at several corpus sizes: how accuracy, latency and cost move with N."""
    datasets = load_datasets(args.datasets_dir, args.datasets)[:1]
    specs = load_specs(args.retrievers_file)
    selected = select_specs(specs, args.retrievers)
    if args.limit:
        datasets[0].queries = datasets[0].queries[: args.limit]
    sizes: list[int | None] = [None if s == "all" else int(s) for s in args.sizes.split(",")]
    scale: dict[str, Any] = {"dataset": datasets[0].name, "sizes": [], "retrievers": {s.name: [] for s in selected}}
    for size in sizes:
        report = progress_run(f"Corpus {size or 'all'}", lambda cb, size=size: run_eval(
            datasets, specs, selected, PROJECT_ROOT / ".cache", corpus_size=size,
            judge_model=args.judge, on_progress=cb))
        data = report["datasets"][datasets[0].name]
        scale["sizes"].append(size or data["eligible_median"])
        scale["created_at"] = report["created_at"]
        for spec in selected:
            m = data["e2e"][spec.name]["metrics"]
            scale["retrievers"][spec.name].append(
                {k: m[k] for k in ("accuracy", "accuracy_attach", "accuracy_new", "latency_p50_ms", "cost_per_1k")}
                | {"recall@5": data["metrics"][spec.name]["recall@5"]})
    if sizes[-1] is None:
        scale["note"] = "The last size is uncapped; its label is the median eligible corpus."
    path = save_report(scale, PROJECT_ROOT / "reports", prefix="scale")
    console.print("\n".join(scale_table(scale)))
    console.print(f"\nScale report: {path}")


def cmd_show(args: argparse.Namespace) -> None:
    path = args.report or latest("run")
    if path is None:
        raise SystemExit("No reports yet. Run `arena run` first.")
    print_report(json.loads(Path(path).read_text()), args.misses)


def cmd_snapshot(args: argparse.Namespace) -> None:
    path = args.report or latest("run")
    if path is None:
        raise SystemExit("No reports yet. Run `arena run` first.")
    report = json.loads(Path(path).read_text())
    scale_path = args.scale or latest("scale")
    scale = json.loads(Path(scale_path).read_text()) if scale_path and not args.no_scale else None
    if args.print:
        print(render(report, scale))
        return
    snapshots = PROJECT_ROOT / "snapshots"
    folder = save_snapshot(report, snapshots, scale)
    write_html(report, folder / "report.html")
    write_summary(report, PROJECT_ROOT, folder)
    write_index(snapshots)
    write_readme_pointer(PROJECT_ROOT / "README.md", snapshots)
    console.print(f"Saved {folder.relative_to(PROJECT_ROOT)}/ and pointed README.md at it")


def cmd_list(args: argparse.Namespace) -> None:
    table = Table(title="Retrievers", box=box.ROUNDED, header_style="bold", title_justify="left")
    for col in ("Name", "Kind", "Model", "Enabled", "Note"):
        table.add_column(col)
    for spec in load_specs(args.retrievers_file).values():
        table.add_row(spec.name, spec.kind, spec.model or "", "yes" if spec.enabled else "[dim]no[/dim]",
                      spec.note)
    console.print(table)
    for d in load_datasets(args.datasets_dir, labeled=False):
        console.print(f"[bold]{d.name}[/bold]: {len(d.corpus)} docs, {len(d.queries)} queries. {d.task.use_case}")


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(prog="arena", description=__doc__)
    parser.add_argument("--datasets-dir", type=Path, default=PROJECT_ROOT / "datasets")
    parser.add_argument("--retrievers-file", type=Path, default=PROJECT_ROOT / "retrievers.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="Show retrievers and datasets").set_defaults(func=cmd_list)

    lab = sub.add_parser("label", help="Pool candidates, ask two judges, write queries.jsonl")
    lab.add_argument("--datasets", type=lambda s: s.split(","))
    lab.add_argument("--limit", type=int)
    lab.add_argument("--merge-only", action="store_true",
                     help="Skip judging; re-merge saved judgments with adjudications.yaml")
    lab.add_argument("--sheet", action="store_true",
                     help="Write candidate pools of unjudged items for a coding agent to label (no judge calls)")
    lab.add_argument("--cached-only", action="store_true",
                     help="Never call the API; queries needing a new call are skipped")
    lab.set_defaults(func=cmd_label)

    run = sub.add_parser("run", help="Run retrievers (+ end-to-end judge) and write a report")
    run.add_argument("--retrievers", type=lambda s: s.split(","), help="Comma-separated names")
    run.add_argument("--datasets", type=lambda s: s.split(","))
    run.add_argument("--limit", type=int, help="Only the first N queries per dataset")
    run.add_argument("--corpus-size", type=int, help="Cap each query's eligible stories at N")
    run.add_argument("--no-e2e", action="store_true", help="Retrieval metrics only, no judge calls")
    run.add_argument("--judge", default="typesafe/jev-1.14", help="Decision model for the end-to-end step")
    run.add_argument("--misses", action="store_true", help="List items most retrievers got wrong")
    run.add_argument("--out", type=Path)
    run.add_argument("--cached-only", action="store_true",
                     help="Never call the API; uncached calls count as errors")
    run.set_defaults(func=cmd_run)

    sc = sub.add_parser("scale", help="Re-run at several corpus sizes (long context vs retrieval)")
    sc.add_argument("--retrievers", type=lambda s: s.split(","), required=True)
    sc.add_argument("--sizes", default="50,150,all")
    sc.add_argument("--datasets", type=lambda s: s.split(","))
    sc.add_argument("--limit", type=int)
    sc.add_argument("--judge", default="typesafe/jev-1.14")
    sc.set_defaults(func=cmd_scale)

    show = sub.add_parser("show", help="Re-print a saved report (latest by default)")
    show.add_argument("report", nargs="?", type=Path)
    show.add_argument("--misses", action="store_true")
    show.set_defaults(func=cmd_show)

    snap = sub.add_parser("snapshot", help="Save a report (+ latest scale run) under snapshots/")
    snap.add_argument("report", nargs="?", type=Path)
    snap.add_argument("--scale", type=Path, help="Scale report (latest by default)")
    snap.add_argument("--no-scale", action="store_true")
    snap.add_argument("--print", action="store_true", help="Print the Markdown instead of saving")
    snap.set_defaults(func=cmd_snapshot)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

