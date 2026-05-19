## Change summary

This PR creates the durable tracker for the remaining tldw WebUI and extension design-system migration. It adds the reviewed design spec, execution plan, local issue-body artifacts, public issue map, and Backlog mirror tasks so the remaining migration can be tracked by product area and governance track instead of by individual baseline entries.

The tracker uses GitHub as the public source of truth for mutable status, current counts, and PR links, while Backlog.md keeps local execution notes and verification evidence. That split keeps the public roadmap readable without turning Backlog into a second baseline file.

Public tracker created:

- Epic: https://github.com/rmusser01/tldw_server/issues/1655
- Product-state migration issues: #1658 through #1670
- Governance issues: #1671 through #1676
- Backlog parent: TASK-45.44
- Backlog children: TASK-45.44.1 through TASK-45.44.19

## Verification

- `git diff --check`
- `git diff --cached --check`
- `gh issue list --repo rmusser01/tldw_server --state open --label design-system --limit 100 --json number,title,url,labels`
- `gh issue view 1655 --repo rmusser01/tldw_server --json number,title,body,url,labels`
- `gh issue view 1658 --repo rmusser01/tldw_server --json number,title,body,url,labels`
- `gh issue view 1671 --repo rmusser01/tldw_server --json number,title,body,url,labels`
- `backlog task TASK-45.44 --plain`
- Bandit skipped: Markdown, Backlog metadata, and GitHub issue state only

## Merge note

This PR is AI-authored. Before merge, the required human-owned `Change summary` should be reviewed or rewritten by the human requester according to the repository AI-generated PR policy.
