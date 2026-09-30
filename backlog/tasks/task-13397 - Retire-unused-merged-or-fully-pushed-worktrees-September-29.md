---
id: TASK-13397
title: Retire unused merged or fully pushed worktrees September 29
status: Done
assignee: []
created_date: '2026-09-30 05:41'
updated_date: '2026-09-30 05:52'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User requested disk cleanup of merged worktrees and worktrees fully represented on the remote. Remove only clean, inactive regular Git worktrees with remote or merged-PR evidence. Preserve local changes, unpushed commits, recent use, runtime data, and Codex-managed checkouts requiring the archive API.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Eligible worktrees removed with Git worktree removal
- [x] #2 Skipped worktrees and reasons recorded
- [x] #3 Remaining registrations and disk space verified
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Refreshed origin and checked the latest 1000 merged PR heads, plus all remote branches for the remaining 47 unproven heads. Assessed 158 registered worktrees. Removed 31 clean inactive regular Git worktrees whose commits were merged or fully present on the remote. Kept primary checkout, 8 recently used trees, 1 open/in-use tree, 43 dirty trees, 47 trees without remote evidence, and 27 Codex-managed trees requiring their owning-chat archive workflow. Preserved and byte-verified 26 database archives before removal. Recovery folder: /Users/macbook-dev/Documents/worktree-cleanup-recovery-20260929. Audit lists every removed and retained worktree. No source implementation changes; verification checked absent paths/registrations, preserved commits, and archives. Final free disk space about 11 GiB; snapshots may retain deleted blocks.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Retired 31 eligible worktrees. Preserved 118 MiB of local database backups and all Git branch refs. Verified 127 worktrees remain registered. Active work and unique local changes were retained.
<!-- SECTION:FINAL_SUMMARY:END -->
