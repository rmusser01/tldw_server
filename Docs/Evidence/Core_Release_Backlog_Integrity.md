# Core Release Backlog Integrity

This check validates the public release-candidate graph owned by `TASK-13013`,
its external frontend-safety dependency `TASK-12116`, downstream handoff
`TASK-12983`, and the four completed records moved out of colliding identities
by `TASK-13013.10`.

Run from the repository root with the existing Python/PyYAML environment:

```bash
python3 - <<'PY'
from collections import defaultdict
from pathlib import Path
import re
import sys

import yaml

records = []
for directory in ("backlog/tasks", "backlog/completed", "backlog/archive"):
    base = Path(directory)
    if not base.exists():
        continue
    for path in base.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        if not text.startswith("---\n"):
            continue
        try:
            frontmatter = yaml.safe_load(text.split("---\n", 2)[1]) or {}
        except yaml.YAMLError:
            continue
        task_id = str(frontmatter.get("id", "")).upper()
        if task_id.startswith("TASK-"):
            records.append((task_id, path, frontmatter, text))

by_id = defaultdict(list)
for record in records:
    by_id[record[0]].append(record)

release_ids = [
    "TASK-13013",
    *[f"TASK-13013.{number}" for number in range(1, 11)],
    "TASK-12116",
    "TASK-12983",
]
historical_ids = ["TASK-13140", "TASK-13141", "TASK-13142", "TASK-13143"]
errors = []

for task_id in release_ids + historical_ids:
    if len(by_id[task_id]) != 1:
        errors.append(f"{task_id}: expected 1 record, found {len(by_id[task_id])}")

for task_id in release_ids:
    if len(by_id[task_id]) != 1:
        continue
    frontmatter = by_id[task_id][0][2]
    dependencies = frontmatter.get("dependencies") or []
    if not isinstance(dependencies, list):
        errors.append(f"{task_id}: dependencies is not a list")
        continue
    for dependency in map(str, dependencies):
        dependency = dependency.upper()
        if len(by_id[dependency]) != 1:
            errors.append(
                f"{task_id}: dependency {dependency} resolves "
                f"{len(by_id[dependency])} times"
            )
    parent = frontmatter.get("parent_task_id")
    if parent and len(by_id[str(parent).upper()]) != 1:
        errors.append(
            f"{task_id}: parent {parent} resolves "
            f"{len(by_id[str(parent).upper()])} times"
        )

for task_id in historical_ids:
    if len(by_id[task_id]) != 1:
        continue
    frontmatter = by_id[task_id][0][2]
    text = by_id[task_id][0][3]
    if str(frontmatter.get("status")) != "Done":
        errors.append(f"{task_id}: historical record is not Done")
    for section in ("AC", "DOD"):
        match = re.search(
            rf"<!-- {section}:BEGIN -->(.*?)<!-- {section}:END -->", text, re.S
        )
        if match and re.search(r"^\s*- \[ \]", match.group(1), re.M):
            errors.append(f"{task_id}: Done record has unchecked {section}")

graph = {}
for task_id in release_ids:
    if len(by_id[task_id]) == 1:
        frontmatter = by_id[task_id][0][2]
        graph[task_id] = [
            str(dependency).upper()
            for dependency in (frontmatter.get("dependencies") or [])
        ]

visiting = set()
visited = set()


def visit(node):
    if node in visiting:
        errors.append(f"cycle reaches {node}")
        return
    if node in visited:
        return
    visiting.add(node)
    for dependency in graph.get(node, []):
        visit(dependency)
    visiting.remove(node)
    visited.add(node)


for node in graph:
    visit(node)

if errors:
    print("release_backlog_integrity=FAIL")
    print("\n".join(f"- {error}" for error in errors))
    sys.exit(1)

print(
    "release_backlog_integrity=PASS "
    f"release_nodes={len(release_ids)} "
    f"historical_records={len(historical_ids)}"
)
PY
```

## Recorded result

- Baseline at `origin/dev` `9fd2246157ce8a32ae6a6691a75efab788229f77`:
  `FAIL` because `TASK-13013` resolved four times and `TASK-12116`
  resolved twice.
- After the `TASK-13013.10` identity migration:
  `release_backlog_integrity=PASS release_nodes=13 historical_records=4`.
- The migrated completed records contain no unchecked acceptance criteria or
  Definition of Done items. The stale PR #2790 completion summary was reconciled
  against merge commit `e4a1cc3afe0608d26e8e5bfb474da4387c1e7d6b`.

This is a scoped release-graph check, not a claim that every unrelated historical
Backlog collision elsewhere in the repository has been normalized.
