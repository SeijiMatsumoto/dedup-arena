"""Self-contained HTML report: takeaways, results table, accuracy-vs-latency/cost scatters,
the distance-cutoff sweep, and a per-query grid of what each retriever answered."""

from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Any

from .recommend import recommend

# Scatter color groups. Three categorical slots is the most a scatter can carry and stay
# distinguishable for color-blind readers; production baselines are neutral and labeled.
GROUP = {
    "lexical": "retrieval", "dense": "retrieval", "hybrid": "retrieval", "rerank": "retrieval",
    "long context": "long context", "agentic": "agentic", "production": "production", "mock": "retrieval",
}
GROUP_VAR = {"retrieval": "--series-1", "long context": "--series-2", "agentic": "--series-3",
             "production": "--neutral"}

CSS = """
:root { color-scheme: light;
  --surface: #fcfcfb; --surface-2: #f3f2ef; --text: #0b0b0b; --text-2: #52514e; --muted: #8a8984;
  --grid: #e4e3df; --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a; --neutral: #8a8984;
  --good: #008300; --bad: #e34948; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark;
  --surface: #1a1a19; --surface-2: #242423; --text: #f0efec; --text-2: #c3c2b7; --muted: #8f8e88;
  --grid: #34332f; --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --neutral: #8f8e88;
  --good: #3fae3f; --bad: #e66767; } }
:root[data-theme="dark"] { color-scheme: dark;
  --surface: #1a1a19; --surface-2: #242423; --text: #f0efec; --text-2: #c3c2b7; --muted: #8f8e88;
  --grid: #34332f; --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --neutral: #8f8e88;
  --good: #3fae3f; --bad: #e66767; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface); color: var(--text);
  font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 22px; margin: 0 0 4px; } h2 { font-size: 17px; margin: 32px 0 8px; }
p, li { color: var(--text-2); } .muted { color: var(--muted); }
.table-wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { padding: 6px 8px; border-bottom: 1px solid var(--grid); text-align: right; white-space: nowrap; }
th:first-child, td:first-child, th.l, td.l { text-align: left; }
th { font-weight: 600; color: var(--text-2); position: sticky; top: 0; background: var(--surface); }
tr.prod td { background: var(--surface-2); }
.charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }
figure { margin: 0; } figcaption { font-weight: 600; margin-bottom: 4px; }
svg text { fill: var(--text-2); font-size: 11px; }
.legend { display: flex; gap: 16px; flex-wrap: wrap; margin: 8px 0; color: var(--text-2); }
.legend span::before { content: ""; display: inline-block; width: 10px; height: 10px;
  border-radius: 50%; margin-right: 6px; background: var(--c); vertical-align: -1px; }
.ok { color: var(--good); } .no { color: var(--bad); }
.grid td { font-size: 12px; } .grid td.l { max-width: 420px; overflow: hidden; text-overflow: ellipsis; }
#tip { position: fixed; pointer-events: none; background: var(--surface-2); color: var(--text);
  border: 1px solid var(--grid); border-radius: 6px; padding: 6px 8px; font-size: 12px;
  display: none; max-width: 320px; z-index: 10; }
.filter { margin: 8px 0; color: var(--text-2); }
"""

JS = """
const tip = document.getElementById('tip');
document.querySelectorAll('[data-tip]').forEach(el => {
  el.addEventListener('mousemove', e => { tip.innerHTML = el.dataset.tip; tip.style.display = 'block';
    tip.style.left = Math.min(e.clientX + 12, innerWidth - 330) + 'px'; tip.style.top = (e.clientY + 12) + 'px'; });
  el.addEventListener('mouseleave', () => tip.style.display = 'none');
});
const box = document.getElementById('misses-only');
if (box) box.addEventListener('change', () => document.querySelectorAll('tr[data-allok]').forEach(
  tr => tr.style.display = box.checked && tr.dataset.allok === '1' ? 'none' : ''));
"""


def esc(v: Any) -> str:
    return html.escape(str(v), quote=True)


def pct(v: float | None, d: int = 0) -> str:
    return "—" if v is None else f"{v:.{d}%}"


def ms(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v / 1000:.1f}s" if v >= 1000 else f"{v:.0f}ms"


def usd(v: float | None) -> str:
    if v is None:
        return "—"
    return f"${v:.2f}" if v >= 1 else f"${v:.4f}"


def scatter(points: list[dict[str, Any]], xkey: str, xlabel: str, fmt) -> str:
    """Accuracy (y) vs a log-scaled x. One dot per retriever, direct labels, hover tooltips."""
    pts = [p for p in points if p[xkey] and p["accuracy"] is not None]
    if not pts:
        return "<p class='muted'>No data.</p>"
    w, h, left, right, top, bottom = 520, 320, 44, 16, 12, 36
    xs = [math.log10(p[xkey]) for p in pts]
    x0, x1 = math.floor(min(xs)), math.ceil(max(xs))
    x1 = x1 if x1 > x0 else x0 + 1
    ys = [p["accuracy"] for p in pts]
    y0 = max(0.0, math.floor((min(ys) - 0.02) * 20) / 20)
    y1 = min(1.0, math.ceil((max(ys) + 0.01) * 20) / 20)
    y1 = y1 if y1 > y0 else y0 + 0.05
    sx = lambda v: left + (math.log10(v) - x0) / (x1 - x0) * (w - left - right)
    sy = lambda v: top + (1 - (v - y0) / (y1 - y0)) * (h - top - bottom)
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="Accuracy vs {xlabel}">']
    for e in range(x0, x1 + 1):
        x = sx(10**e)
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{h - bottom}" stroke="var(--grid)"/>')
        out.append(f'<text x="{x:.1f}" y="{h - bottom + 14}" text-anchor="middle">{esc(fmt(10**e))}</text>')
    steps = max(1, round((y1 - y0) / 0.05))
    for i in range(steps + 1):
        v = y0 + (y1 - y0) * i / steps
        y = sy(v)
        out.append(f'<line x1="{left}" x2="{w - right}" y1="{y:.1f}" y2="{y:.1f}" stroke="var(--grid)"/>')
        out.append(f'<text x="{left - 6}" y="{y + 4:.1f}" text-anchor="end">{v:.0%}</text>')
    out.append(f'<text x="{(w + left) / 2}" y="{h - 4}" text-anchor="middle">{esc(xlabel)} (log scale) →</text>')
    best = sorted(pts, key=lambda p: -p["accuracy"])[:4]
    labeled = {p["name"] for p in best} | {p["name"] for p in pts if p["group"] != "retrieval"}
    for p in sorted(pts, key=lambda p: p["group"] == "production"):
        x, y = sx(p[xkey]), sy(p["accuracy"])
        tip = (f"<b>{esc(p['name'])}</b><br>{esc(p['family'])}<br>end-to-end {pct(p['accuracy'], 1)}"
               f"<br>recall@5 {pct(p['recall5'])}<br>{ms(p['latency'])} · {usd(p['cost'])} per 1k")
        out.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="var({GROUP_VAR[p["group"]]})" '
            f'stroke="var(--surface)" stroke-width="2" data-tip="{esc(tip)}"/>'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="12" fill="transparent" data-tip="{esc(tip)}"/>'
        )
        if p["name"] in labeled:
            anchor = "end" if x > w * 0.7 else "start"
            dx = -9 if anchor == "end" else 9
            out.append(f'<text x="{x + dx:.1f}" y="{y + 4:.1f}" text-anchor="{anchor}">{esc(p["name"])}</text>')
    out.append("</svg>")
    return "".join(out)


def line_chart(rows: list[dict[str, Any]], key: str, label: str, fmt, mark: float | None) -> str:
    rows = [r for r in rows if r[key] is not None]
    if not rows:
        return ""
    w, h, left, right, top, bottom = 420, 220, 44, 16, 12, 36
    xs = [r["max_distance"] for r in rows]
    vals = [r[key] for r in rows]
    x0, x1 = 0.0, max(xs)
    y0, y1 = 0.0, max(max(vals) * 1.1, 1e-9) if key != "recall@5" else 1.0
    sx = lambda v: left + (v - x0) / (x1 - x0) * (w - left - right)
    sy = lambda v: top + (1 - (v - y0) / (y1 - y0)) * (h - top - bottom)
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="{esc(label)}">']
    for i in range(5):
        v = y0 + (y1 - y0) * i / 4
        out.append(f'<line x1="{left}" x2="{w - right}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="var(--grid)"/>'
                   f'<text x="{left - 6}" y="{sy(v) + 4:.1f}" text-anchor="end">{esc(fmt(v))}</text>')
    for v in xs[1::2]:
        out.append(f'<text x="{sx(v):.1f}" y="{h - bottom + 14}" text-anchor="middle">{v:.2f}</text>')
    out.append(f'<text x="{(w + left) / 2}" y="{h - 4}" text-anchor="middle">max cosine distance →</text>')
    if mark is not None:
        out.append(f'<line x1="{sx(mark):.1f}" x2="{sx(mark):.1f}" y1="{top}" y2="{h - bottom}" '
                   f'stroke="var(--muted)" stroke-dasharray="4 3"/>'
                   f'<text x="{sx(mark) + 4:.1f}" y="{top + 10}">production 0.10</text>')
    path = " ".join(f"{'M' if i == 0 else 'L'}{sx(x):.1f},{sy(v):.1f}" for i, (x, v) in enumerate(zip(xs, vals, strict=True)))
    out.append(f'<path d="{path}" fill="none" stroke="var(--series-1)" stroke-width="2"/>')
    for x, v in zip(xs, vals, strict=True):
        out.append(f'<circle cx="{sx(x):.1f}" cy="{sy(v):.1f}" r="4" fill="var(--series-1)" stroke="var(--surface)" '
                   f'stroke-width="2" data-tip="distance ≤ {x:.2f}: {esc(fmt(v))}"/>'
                   f'<circle cx="{sx(x):.1f}" cy="{sy(v):.1f}" r="10" fill="transparent" '
                   f'data-tip="distance ≤ {x:.2f}: {esc(fmt(v))}"/>')
    out.append("</svg>")
    return "".join(out)


def dataset_section(name: str, data: dict[str, Any], kinds: dict[str, Any], rec: dict[str, Any]) -> str:
    e2e_mode = rec.get("mode") == "e2e"

    def e2e_of(n: str) -> dict[str, Any]:
        return (data["e2e"].get(n) or {}).get("metrics", {})

    names = list(dict.fromkeys([*data["metrics"], *data["e2e"]]))
    order = sorted(names, key=lambda n: (
        -(e2e_of(n).get("accuracy") or 0) if e2e_mode else 0,
        -(data["metrics"].get(n, {}).get("recall@5") or 0), -(data["metrics"].get(n, {}).get("mrr") or 0)))
    parts = [f"<h2>{esc(name)}</h2><p>{esc(data['description'])}</p><ul>"]
    for p in rec["picks"]:
        parts.append(f"<li><b>{esc(p['role'].replace('_', ' ').capitalize())}: {esc(p['retriever'])}.</b> {esc(p['why'])}</li>")
    for a in rec["avoid"]:
        parts.append(f"<li><b>Avoid {esc(a['retriever'])}.</b> {esc(a['why'])}</li>")
    for n in rec["notes"]:
        parts.append(f"<li>{esc(n)}</li>")
    parts.append("</ul>")

    points = []
    for ret in order:
        e, r = e2e_of(ret), data["metrics"].get(ret, {})
        family = kinds[ret]["family"] if ret in kinds else "production"
        if e2e_mode and not e:
            continue
        points.append({"name": ret, "family": family, "group": GROUP[family],
                       "accuracy": e.get("accuracy") if e2e_mode else r.get("recall@5"),
                       "latency": e.get("latency_p50_ms") if e2e_mode else r.get("latency_p50_ms"),
                       "cost": e.get("cost_per_1k") if e2e_mode else r.get("cost_per_1k"),
                       "recall5": r.get("recall@5")})
    metric = "End-to-end accuracy" if e2e_mode else "Recall@5"
    legend = "".join(f'<span style="--c: var({GROUP_VAR[g]})">{esc(g)}</span>'
                     for g in ("retrieval", "long context", "agentic", "production"))
    parts.append(f'<div class="legend">{legend}</div><div class="charts">'
                 f'<figure><figcaption>{metric} vs median latency</figcaption>'
                 f'{scatter(points, "latency", "median latency", ms)}</figure>'
                 f'<figure><figcaption>{metric} vs cost per 1k items</figcaption>'
                 f'{scatter(points, "cost", "$ per 1k items", usd)}</figure></div>')

    head = ["Retriever", "Family", "End-to-end", "Attach", "New", "Dupes", "Wrong merge", "R@1", "R@5",
            "R@10", "MRR", "nDCG@10", "p50", "p95", "$ / 1k", "Index $", "Err"]
    rows = []
    for ret in order:
        e, r = e2e_of(ret), data["metrics"].get(ret, {})
        family = kinds[ret]["family"] if ret in kinds else "production"
        idx = data["index_cost"].get(ret)
        cells = [ret, family, pct(e.get("accuracy"), 1), pct(e.get("accuracy_attach")),
                 pct(e.get("accuracy_new")), pct(e.get("duplicate_rate")), pct(e.get("wrong_merge_rate")),
                 pct(r.get("recall@1")), pct(r.get("recall@5"), 1), pct(r.get("recall@10")),
                 "—" if r.get("mrr") is None else f"{r['mrr']:.2f}",
                 "—" if r.get("ndcg@10") is None else f"{r['ndcg@10']:.2f}",
                 ms(e.get("latency_p50_ms", r.get("latency_p50_ms"))),
                 ms(e.get("latency_p95_ms", r.get("latency_p95_ms"))),
                 usd(e.get("cost_per_1k", r.get("cost_per_1k"))),
                 usd(idx) if idx else "—", str((e.get("errors") or r.get("errors")) or "")]
        cls = ' class="prod"' if family == "production" else ""
        rows.append(f"<tr{cls}>" + "".join(
            f'<td class="l">{esc(c)}</td>' if i < 2 else f"<td>{esc(c)}</td>" for i, c in enumerate(cells)) + "</tr>")
    parts.append('<h2>All retrievers</h2><div class="table-wrap"><table><thead><tr>'
                 + "".join(f'<th class="l">{h}</th>' if i < 2 else f"<th>{h}</th>" for i, h in enumerate(head))
                 + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")

    sweep = data["sweeps"].get("gemini-emb-2")
    if sweep:
        parts.append('<h2>Distance cutoff sweep (gemini-emb-2)</h2><p>What production would retrieve '
                     'with a different cosine-distance cutoff, using the same embeddings.</p><div class="charts">'
                     f'<figure><figcaption>Recall@5</figcaption>{line_chart(sweep, "recall@5", "Recall@5", lambda v: f"{v:.0%}", 0.10)}</figure>'
                     f'<figure><figcaption>Candidates passed to the judge (avg, max 5)</figcaption>'
                     f'{line_chart(sweep, "candidates_mean", "Candidates", lambda v: f"{v:.1f}", 0.10)}</figure></div>')

    shown = [n for n in order if (n in data["e2e"] if e2e_mode else n in data["results"])][:8]
    docs = data["docs"]

    def title(i: str | None) -> str:
        if i is None:
            return "new story"
        d = docs.get(i, {})
        return f"{i}: {d.get('subject', '')} {d.get('version') or ''} | {d.get('title', '')}"

    grid = []
    for q in data["queries"]:
        cells, all_ok = [], True
        for ret in shown:
            if e2e_mode:
                preds = data["e2e"][ret]["preds"]
                if q["id"] not in preds:
                    cells.append('<td class="muted">·</td>')
                    continue
                pred = preds[q["id"]]
                ok = data["e2e"][ret]["metrics"]["correct"].get(q["id"])
            else:
                # Retrieval only: did the top 5 contain a right story (or, for new items, skip).
                res = data["results"].get(ret, {}).get(q["id"])
                if not q["relevant"] or not res or res.get("skipped"):
                    cells.append('<td class="muted">·</td>')
                    continue
                top = res.get("ids", [])[:5]
                ok = any(i in q["relevant"] for i in top)
                pred = top[0] if top else None
            all_ok &= bool(ok)
            mark = "✓" if ok else "✗"
            cells.append(f'<td class="{"ok" if ok else "no"}" data-tip="{esc(esc(title(pred)))}">{mark}</td>')
        expected = ", ".join(q["relevant"]) or "new story"
        tip = esc(esc("<br>".join(title(i) for i in q["relevant"]) or "new story"))
        headline = f'<a href="{esc(q["url"])}">{esc(q["headline"])}</a>' if q.get("url") else esc(q["headline"])
        grid.append(f'<tr data-allok="{int(all_ok)}"><td class="l">{esc(q["id"])}</td><td class="l">{headline}</td>'
                    f'<td class="l" data-tip="{tip}">{esc(expected)}</td><td class="l">{esc(", ".join(q["tags"]))}</td>'
                    + "".join(cells) + "</tr>")
    parts.append('<h2>Per item</h2><label class="filter"><input type="checkbox" id="misses-only" checked> '
                 'Only items some retriever got wrong</label><div class="table-wrap"><table class="grid"><thead><tr>'
                 '<th class="l">Id</th><th class="l">Item</th><th class="l">Expected</th><th class="l">Tags</th>'
                 + "".join(f"<th>{esc(s)}</th>" for s in shown) + "</tr></thead><tbody>" + "".join(grid)
                 + "</tbody></table></div>")
    return "".join(parts)


def write_html(report: dict[str, Any], path: Path) -> Path:
    kinds = {r["name"]: r for r in report["retrievers"]}
    recs = recommend(report)
    body = "".join(dataset_section(n, d, kinds, recs[n]) for n, d in report["datasets"].items())
    doc = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>Retrieval Arena report</title><style>{CSS}</style></head><body><main>"
        f"<h1>Retrieval Arena</h1><p class='muted'>Run {esc(report['created_at'])} · "
        f"judge {esc(report.get('judge_model'))}"
        + (f" · corpus capped at {report['corpus_size']}" if report.get("corpus_size") else "")
        + f"</p>{body}</main><div id='tip'></div><script>{JS}"
        "document.getElementById('misses-only')?.dispatchEvent(new Event('change'));</script></body></html>"
    )
    path.write_text(doc)
    return path

