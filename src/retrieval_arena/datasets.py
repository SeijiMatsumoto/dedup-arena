from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

View = Literal["item", "meta", "full"]


class Doc(BaseModel):
    """One existing Story, as it looked when it was created."""

    id: str
    title: str
    item_title: str | None = None
    subject: str = ""
    event: str | None = None
    version: str | None = None
    item_type: str | None = None
    url: str | None = None
    excerpt: str = ""
    published_at: str | None = None
    available_at: str


class Query(BaseModel):
    """An incoming item. `relevant` is empty when it starts a new Story, None when unlabeled."""

    id: str
    text: str
    title: str | None = None
    subject: str = ""
    event: str | None = None
    version: str | None = None
    item_type: str | None = None
    url: str | None = None
    published_at: str | None = None
    asked_at: str
    exclude: list[str] = Field(default_factory=list)
    production: dict[str, Any] = Field(default_factory=dict)
    relevant: list[str] | None = None
    label_source: str | None = None
    tags: list[str] = Field(default_factory=list)


class Task(BaseModel):
    name: str
    description: str
    use_case: str
    source: str = ""
    definition: str
    pool: dict[str, Any] = Field(default_factory=dict)
    judges: list[str] = Field(default_factory=list)


class Dataset(BaseModel):
    task: Task
    corpus: list[Doc]
    queries: list[Query]
    path: Path

    @property
    def name(self) -> str:
        return self.task.name

    def doc(self, doc_id: str) -> Doc:
        return self._by_id[doc_id]

    def model_post_init(self, _context: Any) -> None:
        self._by_id = {d.id: d for d in self.corpus}

    def eligible(self, query: Query) -> list[str]:
        """Stories that existed when the item arrived. Retrievers only ever search these."""
        skip = set(query.exclude)
        return [d.id for d in self.corpus if d.available_at < query.asked_at and d.id not in skip]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def load_dataset(folder: Path, labeled: bool = True) -> Dataset:
    """Load corpus + queries. With labeled=True, only queries that have labels are returned."""
    task = Task.model_validate(yaml.safe_load((folder / "task.yaml").read_text()))
    corpus = [Doc.model_validate(d) for d in read_jsonl(folder / "corpus.jsonl")]
    labeled_path = folder / "queries.jsonl"
    if labeled and labeled_path.exists():
        queries = [Query.model_validate(q) for q in read_jsonl(labeled_path)]
    elif labeled:
        raise SystemExit(f"{labeled_path} does not exist yet. Run `arena label` first.")
    else:
        queries = [Query.model_validate(q) for q in read_jsonl(folder / "queries.raw.jsonl")]
    ids = {d.id for d in corpus}
    for q in queries:
        unknown = set(q.relevant or []) - ids
        if unknown:
            raise ValueError(f"{q.id}: relevant ids not in corpus: {sorted(unknown)}")
    return Dataset(task=task, corpus=corpus, queries=queries, path=folder)


def load_datasets(root: Path, names: list[str] | None = None, labeled: bool = True) -> list[Dataset]:
    folders = sorted(p.parent for p in root.glob("*/task.yaml"))
    if names:
        missing = set(names) - {f.name for f in folders}
        if missing:
            raise SystemExit(f"Unknown dataset(s): {', '.join(sorted(missing))}")
        folders = [f for f in folders if f.name in names]
    return [load_dataset(f, labeled) for f in folders]


# --- Text views -------------------------------------------------------------------------------
# "item" mirrors what enable-ai embeds today: the source item's own title and text.
# "meta" is the extracted subject/version plus a headline. "full" is both.


def _meta(subject: str, version: str | None, headline: str) -> str:
    return " ".join(p for p in (subject, version or "", "|", headline) if p).strip()


def doc_text(doc: Doc, view: View) -> str:
    item = f"{doc.item_title}. {doc.excerpt}" if doc.item_title else doc.excerpt
    if view == "item":
        return item
    meta = _meta(doc.subject, doc.version, doc.title)
    return meta if view == "meta" else f"{meta}\n{item}"


def query_text(query: Query, view: View) -> str:
    item = f"{query.title}. {query.text}" if query.title else query.text
    if view == "item":
        return item
    meta = _meta(query.subject, query.version, query.title or query.text[:200])
    return meta if view == "meta" else f"{meta}\n{item}"


def _date(value: str | None) -> str:
    return (value or "")[:10] or "undated"


def doc_card(doc: Doc, excerpt_chars: int = 200) -> str:
    """Compact one-story listing for LLM prompts."""
    version = f" v{doc.version}" if doc.version else ""
    excerpt = doc.excerpt[:excerpt_chars]
    return (
        f"[{doc.id}] {doc.subject}{version} | {doc.title} "
        f"({doc.item_type or 'item'}, {_date(doc.published_at)})\n  {excerpt}"
    )


def query_card(query: Query, excerpt_chars: int = 600) -> str:
    version = f" v{query.version}" if query.version else ""
    title = f"{query.title}\n" if query.title else ""
    return (
        f"Subject: {query.subject}{version}\nType: {query.item_type or 'item'}, "
        f"published {_date(query.published_at)}\n{title}{query.text[:excerpt_chars]}"
    )


def normalize_subject(value: str) -> str:
    """Same normalization enable-ai uses for subject identity."""
    value = re.sub(r"([a-z])([0-9])", r"\1 \2", (value or "").casefold())
    return " ".join(re.findall(r"[a-z0-9]+", value))
