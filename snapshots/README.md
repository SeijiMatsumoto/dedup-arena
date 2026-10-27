# Snapshots

Dated eval runs, newest first. Each folder has a readable summary and the full `report.json`.
Create one from your latest run with `uv run arena snapshot`.

| Date | Retrievers | Queries | Headline |
|---|---:|---:|---|
| [2026-10-27_2053](2026-10-27_2053/) | 8 | 470 | Use **hybrid-voyage** (recall@5 96%, right story first 82%, 245ms median, $0.0118 per 1k). Production today (its logged decisions): 90.9% correct end to end; 31% of items that belong to an existing story became a duplicate story, and 2% of all items were merged into the wrong story. Production's logged candidates contained the right story for 84.5% of matching items, so 15% never reached the judge. openai-3-large reaches 97.3% (paired McNemar p=0.0005). |
