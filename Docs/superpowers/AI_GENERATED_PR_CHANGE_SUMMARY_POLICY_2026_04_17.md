# AI-Generated PR Change Summary Policy (2026-04-17)

## Purpose

Require human ownership of AI-assisted pull requests.

If AI materially authored the change, the human requester must be able to explain what changed and why that implementation path was chosen. If they cannot do that, the pull request is not ready to merge.

## Scope

This policy applies to pull requests where AI materially generated or substantially reworked the code, tests, docs, configuration, or review-fix changes.

It does not depend on which model or tool was used. The question is whether AI materially shaped the merged result.

## Hard Merge Gate

Every AI-generated pull request must include a human-written `Change summary`.

Merge is blocked when that section is missing, shallow, or clearly not owned by the human requester.

## Required Content

The `Change summary` must explain, in the human requester's own words:

1. What changed at a meaningful level.
2. Why the AI chose those specific implementation decisions.
3. Which constraints, tradeoffs, or repo-local patterns drove the chosen path when they materially influenced the result.

The standard is comprehension, not polish. A concise summary is fine if it proves the human requester actually understands the change.

## Not Sufficient

The following do not satisfy this policy:

- a file-by-file diff recap with no explanation of decision-making
- text copied from AI output without clear human ownership
- generic claims like `the AI said this was cleaner` without repo-specific reasoning
- a summary the human requester cannot defend in review discussion

## Reviewer Enforcement

Reviewers should block merge when:

- the `Change summary` is missing
- the summary explains only what changed, not why
- the reasoning is generic, contradictory, or disconnected from the actual diff
- the human requester cannot explain the implementation choices in their own words

The operating rule is simple: if the person who prompted the AI cannot explain why the change looks the way it does, the PR does not merge.
