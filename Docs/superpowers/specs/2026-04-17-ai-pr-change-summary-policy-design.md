# AI-Generated PR Change Summary Policy Design

- Date: 2026-04-17
- Project: tldw_server
- Topic: Add a hard merge gate requiring human-authored change summaries on AI-generated pull requests
- Mode: Design for implementation

## 1. Objective

Add repo-local superpowers guidance that blocks merge of AI-generated pull requests unless the human requester provides a change summary in their own words.

The summary must explain both:

- what changed
- why the AI made those specific implementation choices

The policy exists to enforce human ownership of AI-assisted changes. If the person who prompted the AI cannot explain the reasoning behind the change, the pull request is not merge-ready.

## 2. Scope

### In Scope

- repo-local governance for AI-generated pull requests
- reviewer guidance for when to block merge
- a canonical policy document under `Docs/superpowers/`
- a short enforcement hook in `AGENTS.md`

### Out Of Scope

- changes to the global installed superpowers skill set under `~/.codex/superpowers`
- automation that validates pull request bodies
- retroactive edits to historical plans, specs, or review artifacts

## 3. Policy Requirements

The new guidance must state that:

1. AI-generated pull requests require a human-written `Change summary`.
2. The summary must explain both the change set and the reasoning behind the chosen implementation path.
3. A summary that only restates the diff is insufficient.
4. Reviewers should block merge when the summary is missing, shallow, or clearly not understood by the human requester.
5. If the human requester cannot explain the rationale in their own words, the PR does not merge.

## 4. Approaches Considered

### Recommended: Canonical Policy Doc Plus AGENTS Hook

Create one standing policy document under `Docs/superpowers/` and add a concise enforcement reference in `AGENTS.md`.

Pros:

- gives the repo a canonical superpowers policy artifact
- places enforcement in the repo instruction file agents are most likely to read first
- avoids scattering policy text across historical planning artifacts

Cons:

- policy lives in two places, so wording must stay aligned

### Alternative: AGENTS-Only

Put the entire rule only in `AGENTS.md`.

Pros:

- strongest immediate enforcement point for coding agents

Cons:

- no standalone superpowers policy artifact in the repo docs

### Rejected: Docs-Only

Put the policy only in `Docs/superpowers/`.

Reason rejected:

- too easy to miss during active implementation and review work
- weaker as a real merge gate

## 5. Approved Design

Implement the recommended approach:

- add `Docs/superpowers/AI_GENERATED_PR_CHANGE_SUMMARY_POLICY_2026_04_17.md`
- update `AGENTS.md` with a short hard-gate section in `Quality Gates`

The policy document should define purpose, scope, required summary content, unacceptable summaries, and reviewer enforcement expectations.

The `AGENTS.md` update should stay short and directive, with `AGENTS.md` serving as the enforcement surface and the policy doc serving as the canonical explanation.
