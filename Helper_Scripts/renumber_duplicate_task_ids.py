"""Decide, and with --apply carry out, the renumbering of duplicate backlog task ids (TASK-N1).

Run from the repository root, on a branch based on current origin/dev:

    PYTHONPATH=tools/backlog-py/src python Helper_Scripts/renumber_duplicate_task_ids.py --task TASK-N1
    PYTHONPATH=tools/backlog-py/src python Helper_Scripts/renumber_duplicate_task_ids.py --task TASK-N1 --apply

Both write the record, Docs/Development/Backlog_Task_Id_Renumbering.md. --apply then renumbers through
backlog-py, retargets dependencies through backlog-py, and rewrites file-name links in files that are
not task files. While a decision is missing it changes nothing else, lists the gap, and exits 2.

Keeper rule, first that applies: (a) referenced by file name from code or tests outside backlog/ and
Docs/; (b) owns subtasks; (c) touched by an open PR; (d) first added to dev: the copy whose add commit
reached dev's first-parent chain first, so the one dev has held longest (not the earliest created_date,
nor the earliest commit date of a branch that merged later). When rules a-c name different copies, or a
group has subtasks no copy owns, OWNER_DECISIONS decides.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess  # nosec B404 - fixed git/gh argv, no shell
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from backlog_py.core.repository import MutableRepository
from backlog_py.markdown.task_parser import parse_task_markdown

TASK_DIRS = ("backlog/tasks/", "backlog/completed/", "backlog/archive/tasks/")
DIR_RANK = {"backlog/completed": 0, "backlog/archive/tasks": 1, "backlog/tasks": 2}  # rule (d) tie-break
# tasks/ first, archive/ last: the block's highest ids then sort first in this PR's file list.
ALLOC_RANK = {"backlog/tasks": 0, "backlog/completed": 1, "backlog/archive/tasks": 2}
BLOCK_BASE = 15000  # renumbered top-level ids start here; the highest id in use on 2026-10-30 was 13958
HISTORY = "origin/dev"  # rule (d) and dependency ages come from dev's history, never this branch's
RECORD = "Docs/Development/Backlog_Task_Id_Renumbering.md"
SELF = "Helper_Scripts/renumber_duplicate_task_ids.py"
REPO = "rmusser01/tldw_server"
CODE_SUFFIXES = (".py", ".ts", ".tsx", ".js", ".mjs", ".sh", ".yml", ".yaml", ".toml", ".cfg", ".ini")

# Rule (b) by topic, for subtasks not first added in their parent copy's first commit:
# group id -> (file name of the copy they plainly continue, or None when no copy owns them; evidence).
_JULY = "whose parent became {} when 3a660d4f1d (2026-07-05) renumbered it without its subtasks"
TOPIC_OWNERS: dict[str, tuple[str | None, str]] = {
    "TASK-394": ("task-394 - Implement-Quick-Ingest-UX-remediation.md",
                 "the backlog/tasks copies of 394.1-.7 are stale re-adds (8afc86be86) of its completed subtasks"),
    "TASK-407": (None, "407.1-.6 are chat role-play work " + _JULY.format("TASK-12295")),
    "TASK-410": (None, "410.1 is Persona Buddy work " + _JULY.format("TASK-12303")),
    "TASK-418": ("task-418 - Plan-WebUI-UX-remediation-implementation-slices.md",
                 "418.11-.13 and the backlog/tasks 418.14 continue the WebUI plan"),
    "TASK-463": (None, "463.1-.5 are Research Workspace work " + _JULY.format("TASK-12418")),
    "TASK-481": (None, "481.1-.19 are the notes UX PR slices " + _JULY.format("TASK-12446")),
    "TASK-528": (None, "528.1-.9 are /knowledge QA work " + _JULY.format("TASK-12561")),
    "TASK-9935": ("task-9935 - Design-Claims-Jobs-operational-control-plane-refactor.md",
                  "9935.1-.4 are Claims Jobs Stage 2B and its PR #3093"),
    "TASK-12098": ("task-12098 - Create-Chatbooks-backup-import-remediation-PRD-and-task-set.md",
                   "12098.4-.5 cite the Chatbooks post-merge UAT review and remediation plans"),
    "TASK-12948": ("task-12948 - Address-PR-2713-authentication-persistence-review-findings-and-run-UAT.md",
                   "12948.1 addresses PR 2713's remaining review threads"),
    "TASK-12963": ("task-12963 - Consolidate-provider-credential-runtime-for-Chat-and-Knowledge-QA.md",
                   "12963.1-.2 cite PR #2727, the provider credential runtime consolidation"),
    "TASK-12968": ("task-12968 - Expand-research-source-coverage-through-a-shared-discovery-pipeline.md",
                   "12968.7-.10 are its source inventory and PRs #2741 and #2800 (shared discovery routes)"),
    "TASK-12989": ("task-12989 - Plan-Web-Scraping-Phase-4-extraction-and-article-orchestration.md",
                   "12989.1-.6 are Web_Scraping Phase 4A-4D"),
    "TASK-13000": ("task-13000 - Fix-audit-original-file-storage-cleanup-on-DB-registration-failure.md",
                   "13000.1 bounds the same retained media originals and their storage quota"),
    "TASK-13001": ("task-13001 - Plan-UserProfiles-Stage-2-single-update-pipeline-implementation.md",
                   "13001.1 is Work Package 1 of the UserProfiles Stage 2 plan"),
    "TASK-13243": ("task-13243 - Validate-real-VZ-Linux-execution-under-launchd.md",
                   "13243.1-.7 and 13243.9-.14 are VZ launchd drills"),
    "TASK-13263": ("task-13263 - Prepare-the-0.1.43-release-with-all-changes-since-v0.1.42.md",
                   "13263.1-.3 resolve the 0.1.43 release PRs' review findings"),
    "TASK-13264": (
        "task-13264 - Prepare-0.1.44-release-with-all-changes-since-v0.1.43-and-land-prior-closure-records.md",
        "13264.8-.12 are 0.1.44 release review, fixture and dev-synchronization work",
    ),
}
# Loose subtasks that belong to no copy although TOPIC_OWNERS names one for the rest of the group:
# group id -> (their paths, evidence) (Review v2 M2).
TOPIC_EXCEPTIONS: dict[str, tuple[tuple[str, ...], str]] = {
    "TASK-418": (
        ("backlog/completed/task-418.14 - Lock-llama.cpp-runtime-API-compatibility.md",
         "backlog/completed/task-418.15 - Finalize-llama.cpp-managed-runtime-rollout.md",
         "backlog/completed/task-418.16 - Address-PR-1854-llama.cpp-rollout-review-comments.md"),
        "the completed 418.14 and 418.15-.16 are llama.cpp work " + _JULY.format("TASK-12318"),
    ),
}
# Owner table: group id -> (file name of the keeper, reason). Pre-filled with the plan's recommendations.
OWNER_DECISIONS: dict[str, tuple[str, str]] = {
    "TASK-2234": ("task-2234 - Split-full-suite-CI-checks-to-avoid-PR-timeouts.md",
                  "recommended: 2234.1-.12 are CI and test-harness slices of PR #2727, not Workspace Core work"),
}
# Title calls: (dependent id, duplicated id) -> (file name of the copy the entry means, reason).
_AMBIENT = "task-12123 - Plan-Stage-1-Persona-ambient-companion-implementation.md"
_SECURITY = "task-2247 - Confirm-Security-module-decisions-for-ADR-backfill.md"
_UPLOAD = "task-12092 - Plan-chat-document-upload-processing-choices-implementation.md"
_JOURNAL = "task-13160 - Journal-canonical-Personal-Context-publications-atomically.md"
_JOBS = "task-13215 - Fix-Jobs-completion-missing-row-concurrent-insert-atomicity.md"
DEPENDENCY_DECISIONS: dict[tuple[str, str], tuple[str, str]] = {
    ("TASK-12106", "TASK-12108"): (
        "task-12108 - Add-persistent-HttpOnly-sessions-for-same-origin-single-user-WebUI-auth.md",
        "both are single-user WebUI auth persistence",
    ),
    ("TASK-12124", "TASK-12123"): (_AMBIENT, "Persona Stage 2 follows the Stage 1 plan"),
    ("TASK-12125", "TASK-12123"): (_AMBIENT, "implements that plan"),
    ("TASK-12705", "TASK-2247"): (_SECURITY, "Security ADR backfill"),
    ("TASK-2248", "TASK-2247"): (_SECURITY, "Security ADR backfill"),
    ("TASK-2311", "TASK-2247"): (_SECURITY, "Security ADR backfill"),
    ("TASK-12934", "TASK-12092"): (_UPLOAD, "chat document upload"),
    ("TASK-12935", "TASK-12092"): (_UPLOAD, "chat document upload"),
    ("TASK-12936", "TASK-12092"): (_UPLOAD, "chat document upload"),
    ("TASK-12968.2", "TASK-12971"): ("task-12971 - Add-a-connected-peer-verified-streaming-HTTP-hop-primitive.md",
                                     "its description requires the peer-enforcing one-physical-hop gateway"),
    ("TASK-13011", "TASK-13010"): ("task-13010 - Design-Moderation-shared-models-extraction.md",
                                   "implements that design"),
    ("TASK-13111", "TASK-13014"): ("task-13014 - Implement-canonical-admin-webhook-control-plane-and-migration.md",
                                   "canonical admin webhooks"),
    ("TASK-13145", "TASK-13144"): ("task-13144 - Add-encrypted-server-Personal-Context-repository.md",
                                   "Personal Context"),
    ("TASK-13145", "TASK-13111"): ("task-13111 - Implement-canonical-admin-webhook-delivery-substrate-and-recovery.md",
                                   "canonical admin webhooks"),
    ("TASK-13146", "TASK-13145"): ("task-13145 - Expose-authenticated-Personal-Context-API.md", "Personal Context"),
    ("TASK-13160", "TASK-13159"): ("task-13159 - Version-Personal-Context-ongoing-sync-wire-contract.md",
                                   "Personal Context sync"),
    ("TASK-13161", "TASK-13160"): (_JOURNAL, "Personal Context publications"),
    ("TASK-13166", "TASK-13160"): (_JOURNAL, "Personal Context authority"),
    ("TASK-13167", "TASK-13160"): (_JOURNAL, "Personal Context receipts"),
    ("TASK-13216", "TASK-13215"): (_JOBS, "Jobs completion"),
    ("TASK-13217", "TASK-13215"): (_JOBS, "Jobs completion"),
    ("TASK-2281", "TASK-2387"): ("task-2387 - Design-MCP-smoke-client-transport-harness.md",
                                 "its description names the MCP smoke client harness from TASK-2387"),
    ("TASK-2294.3.2", "TASK-12963"): (
        "task-12963 - Consolidate-provider-credential-runtime-for-Chat-and-Knowledge-QA.md",
        "a model completion adapter needs the provider credential runtime",
    ),
}


@dataclass
class Task:
    path: str
    id: str
    title: str
    parent: str
    deps: list[str]
    size: int
    origin: str = ""
    origin_index: int = -1
    origin_date: str = ""
    arrival: int = -1  # index on dev's first-parent chain of the commit that brought the add commit in
    code_refs: list[str] = field(default_factory=list)
    owns: list[str] = field(default_factory=list)
    open_prs: list[int] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.path.rsplit("/", 1)[1]

    @property
    def directory(self) -> str:
        return self.path.rsplit("/", 1)[0]


def run(*argv: str, check: bool = True) -> str:
    return subprocess.run(argv, capture_output=True, text=True, check=check).stdout  # nosec B603


def git(*args: str) -> str:
    return run("git", *args)


def id_from_name(name: str) -> str:
    return name.rsplit("/", 1)[-1].split(" - ", 1)[0].removesuffix(".md").upper()


def tail_of(path: str) -> str:
    name = path.rsplit("/", 1)[-1]
    return name.split(" - ", 1)[1] if " - " in name else name


def id_key(task_id: str) -> tuple[int, ...]:
    return tuple(int(part) for part in task_id.removeprefix("TASK-").split("."))


def load_tasks(root: Path) -> list[Task]:
    tasks = []
    for directory in TASK_DIRS:
        for path in sorted((root / directory).glob("task-*.md")):
            text = path.read_text(encoding="utf-8")
            meta = parse_task_markdown(text).frontmatter
            rel = path.relative_to(root).as_posix()
            task_id = str(meta.get("id") or id_from_name(rel)).upper()
            tasks.append(Task(
                path=rel,
                id=task_id,
                title=str(meta.get("title") or ""),
                # Some subtasks have no parent_task_id; their id prefix names the parent.
                parent=str(meta.get("parent_task_id") or task_id.rpartition(".")[0]).upper(),
                deps=[str(dep).upper() for dep in meta.get("dependencies") or []],
                size=len(text),
            ))
    return tasks


def first_parent_arrival() -> tuple[dict[str, int], int]:
    """Commit -> index of the first commit on HISTORY's first-parent chain that contains it, and the chain length.

    Each commit on the chain, oldest first, claims every ancestor no earlier chain commit claimed: the
    commits a merge brought in arrive at that merge. One pass over the graph.
    """
    parents = {}
    for line in git("rev-list", "--parents", HISTORY).splitlines():
        sha, *ups = line.split()
        parents[sha] = ups
    chain = git("rev-list", "--first-parent", "--reverse", HISTORY).split()
    arrival: dict[str, int] = {}
    for index, tip in enumerate(chain):
        stack = [tip]
        while stack:
            sha = stack.pop()
            if sha not in arrival:
                arrival[sha] = index
                stack.extend(parents.get(sha, ()))
    return arrival, len(chain)


def walk_history(tasks: list[Task]) -> dict[str, list[str]]:
    """Set each task's first add commit in dev's history, following only renames that keep its id.

    A rename is an R entry, or a delete and an add of the same file name in one commit: a move between
    directories that edited too much for -M90%. Returns every path each file has had, through all
    renames, for reading its earlier contents. Walking HISTORY, not HEAD, keeps a re-run on this branch
    (Task 4 Step 7) from reading its own renumber and revert commits as renames. Also sets when each add
    commit reached dev (``arrival``), which rule (d) compares first.
    """
    arrival, not_on_dev = first_parent_arrival()
    order = {sha: i for i, sha in enumerate(git("rev-list", "--date-order", "--reverse", HISTORY).split())}
    origin: dict[str, tuple[str, int, str]] = {}
    lineage: dict[str, list[str]] = {}
    commit, adds, deletes = ("", -1, ""), [], {}

    def follow(old: str | None, new: str) -> None:
        lineage[new] = lineage.get(old, [old] if old else []) + [new]
        same_id = old in origin and id_from_name(old) == id_from_name(new)
        origin[new] = origin[old] if same_id else commit

    log = git("log", "--date-order", "--reverse", "--diff-filter=ADR", "-M90%", "--name-status",
              "--format=@@%H %cs", HISTORY, "--", "backlog/")
    for line in [*log.splitlines(), "@@end -"]:
        if line.startswith("@@"):
            for path in adds:
                follow(deletes.get(path.rsplit("/", 1)[1]), path)
            sha, day = line[2:].split()
            commit, adds, deletes = (sha, order.get(sha, -1), day), [], {}
        elif line.startswith("A\t"):
            adds.append(line[2:])
        elif line.startswith("D\t"):
            deletes[line.rsplit("/", 1)[1]] = line[2:]
        elif line.startswith("R"):
            _, old, new = line.split("\t")
            follow(old, new)
    for task in tasks:
        if task.path not in origin:  # added by a merge commit, whose diff git log does not show, or not on dev
            found = git("log", "--format=%H %cs", HISTORY, "--", task.path) or git(
                "log", "-1", "--format=%H %cs", "HEAD", "--", task.path)
            sha, day = found.splitlines()[-1].split()
            origin[task.path], lineage[task.path] = (sha, order.get(sha, len(order)), day), [task.path]
        task.origin, task.origin_index, task.origin_date = origin[task.path]
        task.arrival = arrival.get(task.origin, not_on_dev)
    return lineage


def references(names: set[str]) -> dict[str, list[str]]:
    """File name -> tracked text files that mention it, raw or %20-encoded."""
    variants = {v: name for name in names for v in (name, name.replace(" ", "%20"))}
    if not variants:
        return {}
    found = run("git", "grep", "-l", "-z", "-I", "-F", *[a for v in sorted(variants) for a in ("-e", v)], check=False)
    refs: dict[str, list[str]] = defaultdict(list)
    for path in filter(None, found.split("\0")):
        if path in (SELF, RECORD):  # this script's own tables, and the record it writes
            continue
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        for name in {name for v, name in variants.items() if v in text}:
            refs[name].append(path)
    return refs


def open_pr_files(cache: Path | None, refresh: bool = False) -> list[dict]:
    """Every other open PR with its full file list: the REST API pages past gh pr list's 100-file cap.

    The cache records the origin/dev commit it was fetched at. A run on a newer origin/dev, or with
    ``refresh``, fetches again, so a re-run sees the PRs opened since the first run (Review v2 I1).
    """
    dev = git("rev-parse", "origin/dev").strip()
    saved = json.loads(cache.read_text()) if cache and cache.exists() and not refresh else {}
    if saved.get("dev") == dev:
        prs = saved["prs"]
    else:
        prs = json.loads(run("gh", "pr", "list", "--state", "open", "--limit", "500", "--json",
                             "number,title,author,headRefName"))
        for pr in prs:
            out = run("gh", "api", "--paginate", f"repos/{REPO}/pulls/{pr['number']}/files?per_page=100",
                      "--jq", '.[] | [.status, .filename, (.previous_filename // "")] | @tsv')
            pr["files"] = [dict(zip(("status", "path", "previous"), (line.split("\t") + ["", ""])[:3]))
                           for line in out.splitlines()]
            pr["author"] = (pr.get("author") or {}).get("login")
        if cache:
            cache.write_text(json.dumps({"dev": dev, "prs": prs}, indent=1))
    branch = git("rev-parse", "--abbrev-ref", "HEAD").strip()
    return [pr for pr in prs if pr.get("headRefName") != branch]  # a re-run must not see its own PR


def choose_keeper(group_id: str, copies: list[Task], loose: list[str]) -> tuple[Task | None, str, str]:
    """(keeper, rule, why the owner decides). keeper is None while OWNER_DECISIONS lacks a needed entry."""
    named = {rule: [c for c in copies if getattr(c, attr)]
             for rule, attr in (("a", "code_refs"), ("b", "owns"), ("c", "open_prs"))}
    reasons = []
    if len({c.path for hits in named.values() for c in hits}) > 1:
        reasons.append("; ".join(f"({rule}) names {', '.join(c.name for c in hits)}"
                                 for rule, hits in named.items() if hits))
    if loose:
        reasons.append(f"subtasks no copy owns: {', '.join(loose)}")
    if reasons:
        decision = OWNER_DECISIONS.get(group_id)
        keeper = next((c for c in copies if decision and c.name == decision[0]), None)
        return keeper, "owner", "; ".join(reasons)
    for rule, hits in named.items():
        if hits:
            return hits[0], rule, ""
    return min(copies, key=lambda c: (c.arrival, c.origin_index, DIR_RANK[c.directory], -c.size)), "d", ""


def pr_duplicates(prs: list[dict], tasks: list[Task]) -> list[tuple[dict, str, list[str], list[str]]]:
    """Open-PR task files whose id a dev task file holds: (pr, path, those dev files, same-task dev files).

    A dev file with the same file-name tail, other than one the PR renames or removes, means dev already
    has that task, so the PR should drop its file rather than renumber it (Review v2 M1).
    """
    on_dev = {t.path for t in tasks}
    by_id: dict[str, list[str]] = defaultdict(list)
    by_tail: dict[str, list[str]] = defaultdict(list)
    for t in tasks:
        by_id[t.id].append(t.path)
        by_tail[tail_of(t.path)].append(t.path)
    found = []
    for pr in prs:
        gone = {f["previous"] or f["path"] for f in pr["files"] if f["status"] in ("removed", "renamed")}
        for f in pr["files"]:
            if f["path"].startswith(TASK_DIRS) and f["path"] not in on_dev and f["status"] != "removed":
                holders = sorted(p for p in by_id.get(id_from_name(f["path"]), []) if p not in gone)
                if holders:
                    same = sorted(p for p in by_tail.get(tail_of(f["path"]), []) if p not in gone)
                    found.append((pr, f["path"], holders, same))
    return found


def is_ancestor(older: str, newer: str) -> bool:
    return subprocess.run(["git", "merge-base", "--is-ancestor", older, newer]).returncode == 0  # nosec B603 B607


def entry_written_in(dependency: str, lineage: list[str]) -> str:
    """The first commit whose copy of the dependent lists ``dependency``: git log -S, exact on ids."""
    for sha in git("log", "--format=%H", "--reverse", HISTORY, "--", *lineage).split():
        for path in lineage:
            shown = subprocess.run(["git", "show", f"{sha}:{path}"], capture_output=True, text=True)  # nosec
            if shown.returncode == 0:
                listed = parse_task_markdown(shown.stdout).frontmatter.get("dependencies") or []
                if dependency in {str(dep).upper() for dep in listed}:
                    return sha
    raise SystemExit(f"{lineage[-1]}: no commit lists {dependency}")


def plan(root: Path, pr_cache: Path | None, refresh: bool = False) -> dict:
    tasks = load_tasks(root)
    by_id: dict[str, list[Task]] = defaultdict(list)
    for task in tasks:
        by_id[task.id].append(task)
    groups = {i: c for i, c in sorted(by_id.items(), key=lambda kv: id_key(kv[0])) if len(c) > 1}
    lineage = walk_history(tasks)
    copies = [c for group in groups.values() for c in group]
    by_path = {c.path: c for c in copies}
    unsettled: list[str] = []

    # Rule (a).
    refs = references({c.name for c in copies})
    for c in copies:
        c.code_refs = [f for f in refs.get(c.name, [])
                       if not f.startswith(("backlog/", "Docs/")) and f.endswith(CODE_SUFFIXES)]

    # Rule (b): a child first added in a copy's own first commit is that copy's; TOPIC_OWNERS covers the rest.
    children: dict[str, list[Task]] = defaultdict(list)
    for task in tasks:
        if task.parent in groups:
            children[task.parent].append(task)
    loose: dict[str, list[str]] = {}
    by_topic = []
    for group_id, group in groups.items():
        kids = []
        for child in sorted(children[group_id], key=lambda t: id_key(t.id)):
            owners = [c for c in group if c.origin == child.origin]
            if len(owners) == 1:
                owners[0].owns.append(child.id)
            else:
                kids.append(child)
        loose[group_id] = [k.id for k in kids]
        if not kids or group_id not in TOPIC_OWNERS:
            continue
        name, why = TOPIC_OWNERS[group_id]
        exceptions, why_not = TOPIC_EXCEPTIONS.get(group_id, ((), ""))
        claimed = [k.id for k in kids if k.path not in exceptions]
        if name and claimed:
            owner = next((c for c in group if c.name == name), None)
            if owner is None:
                unsettled.append(f"{group_id}: TOPIC_OWNERS names {name}, which is not a copy (retitled?)")
                continue
            owner.owns.extend(claimed)
        if claimed:
            by_topic.append((group_id, claimed, name or "no copy", why))
        if len(claimed) < len(kids):
            by_topic.append((group_id, [k.id for k in kids if k.path in exceptions], "no copy", why_not))
        loose[group_id] = []

    # Rule (c), any status: a PR that adds a path dev already has also writes that copy.
    prs = open_pr_files(pr_cache, refresh)
    pr_by_path: dict[str, list[dict]] = defaultdict(list)
    for pr in prs:
        for f in pr["files"]:
            for path in {f["path"], f["previous"]} - {""}:
                pr_by_path[path].append(pr)
    for c in copies:
        c.open_prs = sorted({pr["number"] for pr in pr_by_path.get(c.path, [])})

    rows, owner_rows = [], []
    for group_id, group in groups.items():
        keeper, rule, why = choose_keeper(group_id, group, loose[group_id])
        if why:
            reason = OWNER_DECISIONS.get(group_id, ("", "no decision recorded"))[1]
            owner_rows.append({"id": group_id, "why": why, "copies": group, "keeper": keeper, "reason": reason})
        if keeper is None:
            unsettled.append(f"{group_id}: needs an OWNER_DECISIONS entry ({why})")
            continue
        rows += [{"old": group_id, "keeper": keeper, "copy": c, "rule": rule} for c in group if c is not keeper]

    # New ids: a reserved block above every id on dev and in open PRs; a subtask stays under its parent.
    used = {t.id for t in tasks} | {id_from_name(f["path"]) for pr in prs for f in pr["files"]
                                    if re.search(r"/task-\d[\d.]* - ", f["path"])}
    next_top = max(BLOCK_BASE, max(id_key(i)[0] for i in used) + 1)
    for row in sorted(rows, key=lambda r: (ALLOC_RANK[r["copy"].directory], id_key(r["old"]), r["copy"].name)):
        parent = row["old"].rpartition(".")[0]
        if parent:
            row["new"] = f"{parent}.{max(id_key(i)[-1] for i in used if i.rpartition('.')[0] == parent) + 1}"
        else:
            row["new"], next_top = f"TASK-{next_top}", next_top + 1
        used.add(row["new"])
        c = row["copy"]
        row["new_path"] = f"{c.directory}/{row['new'].lower()} - {c.name.split(' - ', 1)[1]}"
    rows.sort(key=lambda r: id_key(r["new"]))
    new_id = {r["copy"].path: r["new"] for r in rows}
    final_path = {r["copy"].path: r["new_path"] for r in rows}
    keeper_of = {r["old"]: r["keeper"] for r in rows}

    # Dependencies that name a duplicated id. Applied after every renumber, by the dependent's final id.
    deps, edits, left = [], defaultdict(dict), []
    renumbered_groups = set(keeper_of)
    for task in tasks:
        for dep in task.deps:
            if dep not in renumbered_groups:
                continue
            written = entry_written_in(dep, lineage[task.path])
            existed = [c for c in groups[dep] if is_ancestor(c.origin, written)]
            if len(existed) == 1:
                target, reason = existed[0], f"only this copy existed when the entry was written ({written[:10]})"
            elif (task.id, dep) in DEPENDENCY_DECISIONS:
                name, why = DEPENDENCY_DECISIONS[(task.id, dep)]
                target, reason = next((c for c in groups[dep] if c.name == name), None), f"title: {why}"
                if target is None:
                    unsettled.append(f"dependency {task.path} -> {dep}: DEPENDENCY_DECISIONS names {name}, "
                                     "which is not a copy (retitled?)")
                    continue
            else:
                unsettled.append(f"dependency {task.path} -> {dep}: {len(existed)} copies existed at {written[:10]}")
                continue
            entry = {"dependent": task, "dep": dep, "target": target, "reason": reason,
                     "new": new_id.get(target.path, dep), "keeper": keeper_of[dep]}
            deps.append(entry)
            if entry["new"] == dep:
                continue
            if task.path in pr_by_path:  # an open PR changes this file: it makes the change (Review v2 I2)
                left += [(pr, entry) for pr in pr_by_path[task.path]]
            else:
                edits[task.path][dep] = entry["new"]
    for path, change in edits.items():  # task edit finds only backlog/tasks files
        if not final_path.get(path, path).startswith("backlog/tasks/"):
            unsettled.append(f"dependency edit {change} on {path}: task edit cannot reach it")
    unreachable = sorted({e["dependent"].path for e in deps if not e["dependent"].path.startswith("backlog/tasks/")})

    # File-name links to renamed files. A renamed file whose name another copy keeps leaves its links alone.
    kept_names = {c.name for c in copies if c.path not in new_id}
    renamed = {r["copy"].name: r["new_path"].rsplit("/", 1)[1] for r in rows if r["copy"].name not in kept_names}
    rewrites, dead, self_links = defaultdict(list), [], []
    for old_name, files in sorted(references(set(renamed)).items()):
        for f in files:
            if not f.startswith(TASK_DIRS):
                rewrites[f].append(old_name)
            elif by_path.get(f) is None or by_path[f].name != old_name:
                dead.append((f, old_name))
            else:  # the renamed file cites its own old name (review v3 M2)
                self_links.append((final_path[f], old_name))

    return {"head": git("rev-parse", "HEAD").strip(), "tasks": tasks, "groups": groups, "rows": rows,
            "owner_rows": owner_rows, "by_topic": by_topic, "deps": deps, "edits": edits, "left": left,
            "unreachable": unreachable, "renamed": renamed, "rewrites": rewrites, "dead": dead,
            "self_links": self_links,
            "same_name": sorted(r["copy"].path for r in rows if r["copy"].name in kept_names),
            "pr_dups": pr_duplicates(prs, tasks), "unsettled": unsettled}


def note(row: dict, task_ref: str, today: str) -> str:
    old, keeper = row["old"], row["keeper"]
    return (f"Renumbered from {old} to {row['new']} on {today} ({task_ref}): {old} now names only "
            f"\"{keeper.title}\" ({keeper.path}). Before {today} both tasks used {old}; "
            f"{RECORD} lists every renumbered id.")


def cell(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def record(p: dict, task_ref: str, today: str) -> str:
    rule_names = {"a": "(a) referenced by code or tests", "b": "(b) owns subtasks", "c": "(c) touched by an open PR",
                  "d": "(d) first added to dev", "owner": "owner decision"}
    out = [f"# Backlog task id renumbering ({task_ref})", "",
           f"Generated by `{SELF}` on {today} from `{p['head'][:10]}`. {len(p['groups'])} ids were held by more "
           f"than one file; {len(p['rows'])} files were renumbered. A renumbered file keeps its directory and its "
           "file-name tail, and its Implementation Notes name its old id.", "",
           "## Owner decisions", "",
           "| Id | Why the rules could not settle it | Copies | Kept | Reason |", "|---|---|---|---|---|"]
    for o in p["owner_rows"]:
        copies = "<br>".join(f"`{c.path}` (added {c.origin_date}, {c.origin[:10]})" for c in o["copies"])
        kept = o["keeper"].name if o["keeper"] else "UNDECIDED"
        out.append(f"| {o['id']} | {cell(o['why'])} | {copies} | {cell(kept)} | {cell(o['reason'])} |")
    out += ["", "## Subtasks attributed by topic", "",
            "Rule (b) for subtasks first added in a commit other than their parent copy's.", "",
            "| Id | Subtasks | Belong to | Evidence |", "|---|---|---|---|"]
    out += [f"| {gid} | {', '.join(kids)} | {cell(name)} | {cell(why)} |" for gid, kids, name, why in p["by_topic"]]
    out += ["", "## Renumbered files", "",
            "| Old id | Kept by | Renumbered file (old path) | New id | Rule |", "|---|---|---|---|---|"]
    out += [f"| {r['old']} | `{cell(r['keeper'].path)}` | `{cell(r['copy'].path)}` | {r['new']} | "
            f"{rule_names[r['rule']]} |" for r in p["rows"]]
    out += ["", "## Dependency decisions", "", "Applied after every renumber, by the dependent's final id.", "",
            "| Dependent | Entry | Means | Becomes | Reason |", "|---|---|---|---|---|"]
    out += [f"| `{cell(e['dependent'].path)}` | {e['dep']} | `{cell(e['target'].path)}` | {e['new']} | "
            f"{cell(e['reason'])} |" for e in p["deps"]]
    out += ["", "Dependents outside `backlog/tasks/`, which `task edit` cannot reach (each needs no change):", ""]
    out += [f"- `{path}`" for path in p["unreachable"]] or ["- none"]
    out += ["", "## Dependency retargets left to open PRs", "",
            "An open PR changes each of these files, so this renumbering leaves the entry as it was; the PR "
            "changes it.", ""]
    out += [f"- #{pr['number']} ({pr['author']}): in `{e['dependent'].path}`, change the dependency `{e['dep']}` "
            f"to `{e['new']}` (`{e['dep']}` now names only `{e['keeper'].path}`)" for pr, e in p["left"]] or ["- none"]
    out += ["", "## Links", "", "File-name links rewritten in files that are not task files:", ""]
    out += [f"- `{f}`: {len(names)}" for f, names in sorted(p["rewrites"].items())] or ["- none"]
    out += ["", "Links inside other task files, left as they are (backlog-py cannot rewrite notes in place):", ""]
    out += [f"- `{f}` -> `{name}`" for f, name in p["dead"]] or ["- none"]
    out += ["", "Renumbered files that cite their own old file name, left as they are:", ""]
    out += [f"- `{f}` -> `{name}`" for f, name in p["self_links"]] or ["- none"]
    out += ["", "Renumbered files whose name another copy keeps (links to that name still resolve to it):", ""]
    out += [f"- `{path}`" for path in p["same_name"]] or ["- none"]
    out += ["", "## Open PRs that add a duplicate id", ""]
    for pr, path, holders, same in p["pr_dups"]:
        if same:
            out.append(f"- #{pr['number']} ({pr['author']}): remove `{path}`; dev already has this task as "
                       + ", ".join(f"`{s}`" for s in same))
        else:
            out.append(f"- #{pr['number']} ({pr['author']}): renumber `{path}` (dev: "
                       + ", ".join(f"`{h}`" for h in holders) + ")")
    if not p["pr_dups"]:
        out.append("- none")
    return "\n".join(out) + "\n"


def apply(p: dict, root: Path, task_ref: str, today: str) -> None:
    repo = MutableRepository.from_path(root)
    for r in p["rows"]:
        repo.renumber_task(Path(r["copy"].path), r["new"], note=note(r, task_ref, today))
    new_id = {r["copy"].path: r["new"] for r in p["rows"]}
    by_path = {t.path: t for t in p["tasks"]}
    for path, change in p["edits"].items():
        task = by_path[path]
        repo.edit_task(new_id.get(path, task.id), dependencies=[change.get(d, d) for d in task.deps])
    for f, names in p["rewrites"].items():  # not task files, so a direct edit is allowed
        with open(f, encoding="utf-8", newline="") as handle:
            text = handle.read()
        for old in names:
            new = p["renamed"][old]
            text = text.replace(old, new).replace(old.replace(" ", "%20"), new.replace(" ", "%20"))
        with open(f, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", required=True, help="the tracking task id, for the notes")
    parser.add_argument("--apply", action="store_true", help="renumber, retarget dependencies, rewrite links")
    parser.add_argument("--pr-cache", type=Path, help="read the open PR file lists from here, or save them here")
    parser.add_argument("--refresh", action="store_true", help="fetch the open PR file lists again")
    args = parser.parse_args(argv)
    root, today = Path.cwd(), date.today().isoformat()
    p = plan(root, args.pr_cache, args.refresh)
    (root / RECORD).write_text(record(p, args.task, today), encoding="utf-8")
    print(f"{len(p['groups'])} duplicated ids, {len(p['rows'])} files to renumber, {len(p['owner_rows'])} owner "
          f"rows, {len(p['deps'])} dependency entries ({sum(map(len, p['edits'].values()))} retargeted, "
          f"{len({id(e) for _, e in p['left']})} left to open PRs), "
          f"{sum(map(len, p['rewrites'].values()))} links in {len(p['rewrites'])} files, "
          f"{len(p['dead'])} dead links and {len(p['self_links'])} dead self-links in task files, "
          f"{len(p['pr_dups'])} duplicate files in open PRs "
          f"({sum(1 for *_, same in p['pr_dups'] if same)} already on dev)")
    for item in p["unsettled"]:
        print(f"UNSETTLED {item}", file=sys.stderr)
    if p["unsettled"]:
        return 2
    if args.apply:
        apply(p, root, args.task, today)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
