# PR #2727 Landing and Private Customer Pilot Design

- **Date:** 2026-07-21
- **Landing task:** TASK-12982
- **Dependent pilot task:** TASK-12983
- **Parent feature task:** TASK-12963
- **Pull request:** [#2727](https://github.com/rmusser01/tldw_server/pull/2727)
- **Status:** Approved design; implementation planning and execution remain separate gates
- **Operator:** Robert Benjamin Jake Musser

## Purpose

Land the provider credential runtime in PR #2727 on the current `dev` branch,
revalidate the exact integrated revision, and operate an access-controlled
customer pilot from infrastructure controlled by Robert Benjamin Jake Musser.
The pilot must not reopen protected artifact publishing or bypass the repository's
frontend licensing, CI, review, or human-authorship gates.

This document defines engineering and release-process boundaries. It is not a
software license, customer agreement, privacy policy, or legal advice. The root
`LICENSE`, files under `LICENSES/`, third-party notices, and executed customer
agreements govern those subjects.

## Current State

At the start of this design review, PR #2727 had 103 commits changing 649 files
at head `e8bcc4c8b705df50a5f7e6299335ba8001ff4811`, based on
`29acaca8c781213e27b12066372df13855e2e7a6`. Its dedicated worktree contained
pre-existing user-owned modifications. A read-only `git merge-tree`
calculation predicted that current `dev` would integrate without conflicts and
that `tldw_Server_API/app/main.py` would merge automatically.

During documentation, a separate owner process advanced and pushed the PR
without using this design-writing action's staging area:

- `7d76bdfcc0c467c779596cd6b92d2f078aa8529e` explicitly committed the
  previously dirty CI/runtime/test follow-up as a separate reviewed change.
- `0e8eadc55f48ff50f55525b8996140cbad43630c` merged current protected `dev`
  `8ed612c7e0335ab922b6abd5f5c11ba1407d552d`. Its first parent is the
  follow-up commit and its second parent is that exact `dev` tip; the merge
  completed without conflicts.
- `6065c64ab4a06687cc10e938eb0bd1cc5b6fd031` recorded post-merge validation
  and became the remote PR head.

At spec finalization, PR #2727 therefore has 106 commits changing 653 files,
targets and contains the licensing-cutoff `dev` tip, remains a draft, and has
fresh exact-head CI in progress. Existing non-green, canceled, and pending
contexts remain evidence to diagnose or await, not checks to bypass. The only
remaining untracked worktree files are unrelated artifacts that must not be
staged by this task.

## Decision

The selected integration path is a merge of verified current `dev` into the
existing PR history, with user-owned follow-up changes handled through their own
explicit commit rather than implicit integration staging. Preserve the PR's
history; do not rebase or squash the series merely for cosmetic cleanup. That
merge has now been executed and pushed consistently with the design. The
remaining landing work is to finish fresh exact-head gates, fix only reproduced
failures, recheck base freshness, satisfy the human-authorship/review gates, and
merge PR #2727 into `dev`.

After merge, deploy a private pilot from one exact `dev` commit to
operator-controlled infrastructure. Customers access the official service;
they do not receive reusable WebUI or Admin UI container images, browser
extension packages, or a formal protected release bundle during this phase.

## Considered Approaches

### Merge `dev` into the PR branch (selected)

This preserves reviewed history, avoids replaying 103 commits, incorporates the
licensing cutoff in the PR head, and triggers the trusted licensing gate and
ordinary CI against one exact integrated revision. The read-only merge
calculation found no conflicts.

### Merge the outdated PR directly

This is mechanically possible but rejected. Its current checks are stale and
include failures, and its head does not contain the merged licensing cutoff.
Stabilizing only after merge would move avoidable integration risk onto the
shared `dev` branch.

### Rebase or squash onto current `dev`

This would produce a cleaner-looking history but is rejected. Replaying or
collapsing a large reviewed series increases force-push, review-mapping,
regression, and provenance risk without improving the deployed result.

## Landing Sequence

At spec finalization, the separate owner process had completed steps 1 through
4 consistently with this sequence. Steps 5 through 12 remain required.

1. Record a read-only inventory of the dirty PR worktree, including the output
   of `git status`, unstaged/staged path lists, and diff identities. The landing
   integration must not absorb them incidentally; intended follow-up changes may
   enter only through their own explicit reviewed commit. No reset, stash,
   checkout, clean, or incidental staging is permitted.
2. Fetch and record verified remote SHAs for the then-current PR head and
   `origin/dev`. Perform the integration in a separate clean worktree. The
   resulting merge commit must have the PR head as first parent and the verified
   `origin/dev` tip as second parent; it must not contain the dirty worktree's
   uncommitted content.
3. Review the integrated diff for licensing-scope changes, unintended feature
   loss, conflict-resolution artifacts, and changes outside PR #2727 plus the
   expected `dev` merge.
4. Push the integration commit without force to
   `codex/provider-credential-runtime-dev`.
5. Require the base-controlled `frontend-license-policy/trusted/dev` context
   and all ordinary exact-head required checks to complete.
6. Triage underlying failed jobs before parent/summary failures. Reproduce each
   underlying failure, distinguish product regressions from infrastructure
   timeouts or established baseline failures, and fix only failures attributable
   to the integrated PR. A required failure that reproduces on the protected
   `dev` baseline still blocks landing: correct it separately on `dev`, then
   reintegrate that fix and rerun the PR's exact-head gates.
7. Rerun focused tests, static checks, Bandit on touched production scope, and
   the repository's proportionate final verification gate.
8. Resolve actionable review findings without accepting stale or technically
   incorrect suggestions blindly.
9. Require the requester to write the repository-mandated human **Change
   summary** explaining what changed and why those implementation choices were
   made. AI-generated substitute text does not satisfy this gate.
10. Before marking ready, re-read the remote PR and `dev` SHAs. If `dev` has
    advanced since the validated integration, merge the new tip and rerun all
    exact-head required gates. Do not rely on checks from an older head.
11. Mark the PR ready only after the human summary, current exact-head CI,
    trusted licensing context, and review conditions are satisfied. Merge into
    `dev`, not directly into `main`.
12. Record the actual merge SHA and verify that it contains both the validated
    PR head and the protected `dev` tip used for the final gate. Confirm the
    merged licensing metadata and trusted-policy files are present before
    allowing TASK-12983 to begin. A merge queue or equivalent up-to-date gate is
    preferred when available.

## Private Pilot Boundary

TASK-12983 begins only after TASK-12982 verifies the actual merge commit. The
first customer test is an access-controlled, pre-release service operated by
Robert Benjamin Jake Musser from that exact merged `dev` revision.

- Record the deployed source commit, backend image digest, frontend build
  identity, configuration schema version, deployment time, and rollback target.
- Keep the WebUI/Admin UI About or Legal surface and all required copyright,
  source-available, warranty, and third-party notices intact in browser-served
  assets.
- Describe the frontend as source-available during the restricted period, not
  as open source. Describe the backend separately as GPL-3.0-only and the
  canonical OpenAPI contract as Apache-2.0.
- The permitted client-artifact boundary is browser delivery of JavaScript,
  CSS, and assets needed to use the access-controlled official service. Do not
  publish deployable protected frontend images to GHCR or Docker Hub, an
  extension-store package, a source or downloadable deployment bundle, or a
  formal tagged protected release from this pilot.
- Do not create an incomplete or improvised Countdown grant. A later public
  protected release must use the repository's release-specific record and
  completed Countdown process.
- Customer browsers necessarily receive copyable JavaScript, CSS, and assets.
  The pilot therefore preserves the public notices and artifact provenance even
  though customers do not receive deployable images, extension packages,
  source bundles, or the service's control plane.
- Assess database and configuration migrations for forward and rollback
  compatibility. Create a recoverable data backup and verify restoration before
  admitting customer data. Record configuration and secret-reference rollback
  procedures without copying secret values into evidence.
- Customer authentication, tenant isolation, incident response,
  deletion/export behavior, customer-data logging, secrets handling, and
  exact-artifact smoke/security checks must pass in the deployed environment
  before admitting customer data.
- Pilot access, support, privacy, service levels, warranties, and customer data
  terms are separate contractual subjects. This engineering design does not
  invent those terms.

## Licensing Reality and Product Boundary

The merged cutoff cannot revoke permissions already granted for code that was
public before the cutoff, including code previously exposed in PR #2727.
Genuinely new repository-authored frontend changes first published after the
cutoff may be offered under the new terms, while prior grants remain unaffected.

PolyForm Perimeter 1.0.1 permits licensed uses other than providing to others a
product that competes with the protected software. Its definition includes
goods and services and can treat even free substitutes as competing products.
The selected private pilot uses the official service operated by the copyright
owner and does not publish deployable images, extension packages, source
bundles, or a tagged protected release to customers.

A future completed PolyForm Countdown grant applies only to the named release
and starts the specified new license terms at noon UTC on its start date. No
pilot deployment, branch merge, or incomplete template silently creates that
grant.

Authoritative references:

- <https://polyformproject.org/licenses/perimeter/1.0.1/>
- <https://polyformproject.org/licenses/countdown/1.0.0/>
- <https://www.gnu.org/licenses/agpl-3.0.en.html>
- `LICENSE`
- `LICENSES/releases/README.md`
- `Docs/superpowers/specs/2026-07-19-frontend-source-available-licensing-design.md`

## Failure and Rollback Handling

- If integration unexpectedly changes or drops a reviewed PR patch, stop and
  repair the integration before pushing it.
- If the trusted licensing context fails, do not bypass it. Determine whether
  the failure is an identity, path-classification, base-policy, or workflow
  problem and correct the source condition.
- If CI fails after integration, preserve the exact failing SHA and logs. Fix
  reproducible PR regressions; rerun transient infrastructure failures only
  after recording evidence. A required baseline failure is not a bypass: land
  its correction separately on `dev`, reintegrate, and revalidate.
- If the pilot fails its readiness checks, keep it private or unavailable. Do
  not widen access to compensate for missing authentication, tenant isolation,
  backup, rollback, or observability controls.
- If a deployed pilot regression is material, roll back to the recorded known
  good commit and image digest together with compatible configuration, secret
  references, and database migration state. Use the verified backup/restore
  path when rollback is not data-compatible. Never replace a versioned artifact
  silently.

## Verification and Evidence

Landing evidence must include:

- pre-integration and post-integration commit identities and merge parents;
- verified remote refs, non-force push evidence, final base-freshness check, and
  the actual merge commit's ancestry;
- confirmation that the user-owned dirty worktree files were not overwritten or
  included unintentionally;
- exact-head required-check and trusted-license results;
- reproduced failure diagnoses and fixes, if any;
- focused tests, static checks, Bandit results, and review disposition;
- the requester-authored Change summary and final PR merge commit; and
- pilot deployment and rollback identities; database/configuration migration
  compatibility; backup/restore verification; exact-artifact smoke/security
  results; and a record of access and artifact publication boundaries.

## Non-Goals

- Publishing a protected WebUI, Admin UI, or extension artifact.
- Creating the first formal release-specific Countdown record.
- Reopening protected-path or OpenAPI contract contributions.
- Drafting a frontend CLA, customer-deployment grant, trademark policy, privacy
  agreement, or commercial license.
- Rewriting PR #2727 history solely to make it shorter.
- Claiming that the new cutoff retroactively protects code already published
  under prior terms.

## Success Criteria

TASK-12982 succeeds when PR #2727 contains the current protected `dev` tip, all
required exact-head gates pass, the requester-authored Change summary is
present, the PR merges into `dev`, and the actual merge SHA is verified. The
pre-existing dirty worktree changes must be preserved; at spec finalization they
had entered through explicit follow-up commit `7d76bdfcc0`, before the separate
`dev` merge, rather than through incidental staging or a force-push.

TASK-12983 then succeeds when a private customer pilot runs from that recorded
merge lineage on operator-controlled infrastructure; migration, backup/restore,
rollback, authentication, isolation, secrets, and exact-artifact checks pass;
and no deployable protected frontend image, extension package, source bundle,
or tagged protected release is published. Browser delivery remains limited to
assets needed to use the access-controlled official service.
