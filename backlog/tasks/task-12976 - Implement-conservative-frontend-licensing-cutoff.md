---
id: TASK-12976
title: Implement conservative frontend licensing cutoff
status: Done
labels:
- licensing
- frontend
- implementation
priority: high
documentation:
- Docs/superpowers/specs/2026-07-19-frontend-source-available-licensing-design.md
- Docs/superpowers/plans/2026-07-20-conservative-frontend-licensing-cutoff-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the approved pre-counsel licensing cutoff. Establish the prospective Perimeter path boundary, preserve public history and third-party notices, declare the OpenAPI contract Apache-2.0, pause unlicensed contribution paths, isolate the GPL API image, and suspend protected artifact publishing. Do not add custom post-counsel grants or publish protected binaries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The root scope map and verbatim legal corpus establish the approved prospective licensing boundary without altering prior public grants.
- [x] #2 Protected package, repository, UI, contribution, and third-party notices consistently describe the frontend as source-available.
- [x] #3 The generated OpenAPI contract declares Apache-2.0 while the server implementation remains GPL-3.0-only.
- [x] #4 The required base-controlled workflow blocks third-party protected, legal-governance, and conservative API declaration changes until later grants exist.
- [x] #5 The GPL API image excludes protected frontend material and rolling protected image publishing is suspended.
- [x] #6 All verification gates pass and the result is submitted as a license-only PR into dev before PR #2727.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

- Task 4's original PR-controlled/newline design was rejected in review and replaced by TASK-12977. Bootstrap PR #2753 placed the trusted `pull_request_target` workflow and NUL-safe classifier on `main`; source-bound `/main` and `/dev` statuses were proven before active rulesets `5653432` and `19362594` required their matching contexts from GitHub Actions App `15368`, with no bypass actors.
- The licensing branch now carries the reviewed trusted files byte-for-byte from merged `main`, restores `frontend-required.yml` byte-for-byte from `origin/dev`, and adds a negative regression contract forbidding license enforcement or status publication in that PR-controlled workflow. Changed paths use bounded NUL transport, `surrogateescape`, no trimming, and `--no-renames` so rename old/new paths are both examined.
- Task 4 RED failed on the rejected checkout/gate behavior; GREEN passed 2/2 after reconciliation. Final local verification passed 40/40 focused tests with six pre-existing warnings, pinned actionlint 1.7.12, Ruff, Black, Bandit with zero findings/errors across 74 classifier LOC, deterministic owner/external allow/deny cases, public ruleset evidence assertions, marker integrity, and `git diff --check`. Independent code/security review and the corrected-plan re-review were CLEAN.
- Bootstrap PR #2753's required human-written `Change summary` remained empty when it merged. That repository-policy requirement was not satisfied and remains explicitly recorded as known noncompliance.
- Task 5 removed all protected roots from the production API image, added the root legal corpus and third-party notices to the runtime image, and reduced `publish-ghcr-main` to the backend `app` image without changing its tags, cache, push, or attestation controls. WebUI and Admin UI remain build-checked but are not published during the licensing freeze.
- Task 5 TDD reproduced both intended failures before implementation. Final verification passed 20/20 Docker and release-workflow contract tests, Ruff, Black, pinned actionlint 1.7.12, documentation consistency review, and `git diff --check`. Independent final review was CLEAN and separately reproduced 20/20 passing tests.
- Full-branch review found that the first nested notices' broad "repository-authored material" wording could override the root Markdown GPL carve-out. TDD reproduced the ambiguity and stale PR evidence as 2/2 failures. All four notices now mirror the root's exact protected categories and explicitly preserve Markdown as GPL-3.0-only unless a more-specific notice classifies it as protected release material. The historical snapshot was refreshed on 2026-07-21 to public PR #2727 head `e8bcc4c8b705df50a5f7e6299335ba8001ff4811`; focused policy tests passed 10/10 and the fix-only re-review was CLEAN.
- Final verification passed 62/62 targeted Python tests and 2/2 protected About Vitest tests. Ruff and Black passed on every touched Python test used in the final correction. Bandit reported zero findings and zero errors across the classifier and `app/main.py` (2,554 LOC). Pinned actionlint 1.7.12 passed all four changed/related workflows; stale-language and whitespace scans were clean.
- `Dockerfiles/Dockerfile.prod` built successfully as `tldw-server:license-cutoff` (`sha256:1e44c831aef0790cf7b6a392df1991efaac27be7c1abba24fc011221b9a2b1ed`). A one-shot runtime assertion confirmed that all four protected roots are absent and the root license, GPL, AGPL, PolyForm Perimeter 1.0.1, and third-party notices are present.
- Scope review found no protected binary, active Community/Dedicated Customer grant, frontend CLA, completed Countdown grant, or PR #2727 feature implementation. PR #2727 has 649 changed files; its only filename overlap is `app/main.py`, where its TTS exception patch is line-disjoint from this branch's OpenAPI metadata change.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

Implemented the conservative pre-counsel frontend licensing cutoff. The server
and unlisted repository material remain GPL-3.0-only; the four approved
frontend roots use PolyForm Perimeter 1.0.1 for repository-authored code,
tests, build definitions, and original assets; Markdown in those roots remains
GPL-3.0-only; and the generated OpenAPI contract is Apache-2.0 without changing
the server implementation license. Existing public grants and upstream
third-party terms are expressly preserved.

The base-controlled trusted gate now pauses unlicensed protected and API
contract-boundary contributions. The production API image excludes all four
protected roots and includes the legal corpus. Rolling and release workflows
publish no protected frontend images during the freeze; WebUI and Admin UI
remain build-checked.

The tracked change inventory consists of root licensing/contribution/public
notices; `LICENSES/**`; the four protected package notices, manifests, and
minimal licensing UI copy/tests; OpenAPI metadata/tests; the trusted workflow,
classifier, ruleset evidence, and CI tests; backend Docker/publish controls and
container/release documentation; and the approved design, plans, Backlog
records, and append-only progress record. No feature code from PR #2727 or
protected binary artifact is included.

PR #2755 is the draft license-only PR into `dev`. PR #2727 remains open, draft,
and blocked behind it. Counsel-reviewed custom grants, a frontend CLA,
commercial-license templates, completed per-release Countdown grants, and
protected artifact publishing are intentionally deferred. PR #2755 is not
merge-ready while its required human-written `Change summary` remains empty;
bootstrap PR #2753's earlier empty summary remains recorded policy
noncompliance.

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
