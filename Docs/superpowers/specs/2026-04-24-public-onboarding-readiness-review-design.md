# Public Onboarding Readiness Review Design

Date: 2026-04-24
Status: Approved for planning
Owner: Codex brainstorming session

## Summary

Run a staged hybrid review of the public self-hosting onboarding experience for `tldw_server`. The review combines a static documentation audit with fresh-user walkthroughs so the project can be judged from the perspective of a non-super-technical person who has just cloned the repo and is following the README.

The review covers the three canonical public setup profiles:

- Docker single-user + WebUI
- Docker multi-user + Postgres
- Local single-user

Each profile is evaluated against the same first-run success bar: a new user should be able to install and start the system, understand the next step, complete a first chat or API action, ingest content and retrieve it, and follow audio guidance far enough to verify STT and TTS expectations.

## Problem

The current onboarding surface is broad and powerful, but it is also distributed across multiple layers:

- the root `README.md`
- `Docs/Getting_Started/README.md`
- the profile-specific setup guides
- `/setup`
- audio setup guides
- WebUI onboarding behavior

For an experienced contributor this may be manageable. For a new user who is not highly technical, the risk is different:

- they may choose the wrong setup path
- they may hit hidden prerequisites after investing time
- they may not know what is required versus optional
- they may reach a healthy server but still not know how to get first value
- they may encounter conflicting guidance between docs, `/setup`, and audio-specific paths

That makes it hard to know whether the project is ready to share publicly as a straightforward first-run experience.

## Goals

- Evaluate the public onboarding path from the perspective of a careful but non-expert self-hoster.
- Cover all three canonical public setup profiles, not just the recommended default path.
- Hold macOS, Windows/WSL, and Linux documentation clarity to the same standard, even when hands-on execution depth differs by environment.
- Judge onboarding by real first-value outcomes, not only by install or health-check success.
- Produce a prioritized findings report with only high-signal public-readiness issues.
- Produce walkthrough logs that show exactly where user confusion begins and what hidden knowledge was required.
- End with a clear readiness verdict and a focused top-issues list.

## Non-Goals

- Exhaustively test every feature or endpoint in the product.
- Perform a deep product QA pass beyond first-run onboarding.
- Rewrite onboarding docs or implement fixes as part of this review.
- Treat minor polish issues as equivalent to public-sharing blockers.
- Evaluate hosted SaaS, browser extension onboarding, or admin UX beyond what directly affects the three public self-hosting profiles.

## Review Standard

The target user is not highly technical. The standard is not "a motivated developer can eventually figure it out." The standard is:

> A careful first-time self-hoster can choose the right path, follow the documented steps in order, recover from ordinary mistakes, and reach a real "this works" moment without relying on insider knowledge.

A profile is only considered successful if the user can reach all of the following with docs-supported steps:

- start the system successfully
- understand what to do next
- complete one meaningful chat or API action
- ingest at least one piece of content and retrieve or search it
- understand and complete enough audio setup to verify STT and TTS behavior
- know what is optional versus required

The final synthesis should still produce one overall public-readiness verdict, but it should report `core onboarding readiness` and `audio onboarding readiness` separately inside that verdict. This keeps the audio requirement in scope without letting audio complexity erase whether the non-audio onboarding path is otherwise understandable.

## Current State

Initial review of the current onboarding surface shows several characteristics that make a strict public-readiness pass necessary:

- the root README routes users into multiple profile guides
- the Docker single-user + WebUI path is the recommended default
- the multi-user guide expects secret generation, database configuration, and later admin-user creation
- the local single-user guide reads more like a contributor workflow
- audio onboarding currently spans both `/setup` bundle provisioning guidance and separate first-time audio docs with different recommended defaults
- several first-run handoffs depend on the user inferring what comes next after the server becomes healthy

The review therefore needs to validate both the written onboarding story and the lived onboarding journey.

## Proposed Design

### 1. Use a staged hybrid review

The review should run in two linked passes:

- `Pass 1: Static onboarding audit`
  - read the onboarding docs exactly as a new user would
  - map decision points, prerequisites, promises, hidden assumptions, and handoff gaps
- `Pass 2: Fresh-user walkthroughs`
  - run the three canonical public profiles against the same first-run success bar
  - record where real runtime friction confirms or disproves the static audit

This hybrid structure prevents two failure modes:

- a docs-only review that misses runtime pain
- a walkthrough-only review that misses structural documentation contradictions

### 2. Define the onboarding surface explicitly

The review should treat the following as the canonical public onboarding surface:

- [README.md](/Users/appledev/Documents/GitHub/tldw_server/README.md)
- [Docs/Getting_Started/README.md](/Users/appledev/Documents/GitHub/tldw_server/Docs/Getting_Started/README.md)
- [Docs/Getting_Started/Profile_Docker_Single_User.md](/Users/appledev/Documents/GitHub/tldw_server/Docs/Getting_Started/Profile_Docker_Single_User.md)
- [Docs/Getting_Started/Profile_Docker_Multi_User_Postgres.md](/Users/appledev/Documents/GitHub/tldw_server/Docs/Getting_Started/Profile_Docker_Multi_User_Postgres.md)
- [Docs/Getting_Started/Profile_Local_Single_User.md](/Users/appledev/Documents/GitHub/tldw_server/Docs/Getting_Started/Profile_Local_Single_User.md)
- [Docs/Deployment/setup-wizard-guide.md](/Users/appledev/Documents/GitHub/tldw_server/Docs/Deployment/setup-wizard-guide.md)
- [Docs/Getting_Started/First_Time_Audio_Setup_CPU.md](/Users/appledev/Documents/GitHub/tldw_server/Docs/Getting_Started/First_Time_Audio_Setup_CPU.md)
- [Docs/Getting_Started/First_Time_Audio_Setup_GPU_Accelerated.md](/Users/appledev/Documents/GitHub/tldw_server/Docs/Getting_Started/First_Time_Audio_Setup_GPU_Accelerated.md)

If other docs are needed to complete a path, that itself is a review signal and should be captured as part of the walkthrough.

### 3. Use one success rubric across all profiles

All three public profiles should be measured with the same rubric:

- `Path selection clarity`
  - can the user tell which profile is for them
- `Prerequisite clarity`
  - are OS, tooling, hardware, and account assumptions stated before failure
- `Command realism`
  - are commands runnable as written and is shell context clear
- `Handoff clarity`
  - after each success point, does the user know the single next step
- `First-value clarity`
  - does the user reach a real success moment quickly enough
- `Recovery quality`
  - do the docs explain symptoms, likely cause, and exact recovery
- `Optional vs required separation`
  - are setup, provider config, ingestion, WebUI, and audio boundaries clear
- `Cross-doc consistency`
  - do README, profile guides, `/setup`, and audio docs tell one coherent story

This keeps the review comparable and prevents one profile from getting a softer standard than another.

### 4. Define a platform evidence model

The review should make platform coverage explicit instead of implying equal execution depth everywhere.

For each profile and milestone, evidence should be tagged as one of:

- `Executed`
  - directly run and observed in the current environment
- `Docs-validated`
  - not run end-to-end on that platform, but the written path was checked for completeness and internal consistency
- `Probable risk`
  - not fully run, but the documented path shows a likely failure or confusion pattern
- `Unverified`
  - not meaningfully validated and should not support a strong claim

This is especially important because the review is holding macOS, Windows/WSL, and Linux documentation to the same clarity standard while hands-on runtime depth may differ by platform.

### 5. Define a golden path for each public profile

To avoid inconsistent walkthrough depth, each profile should have one primary first-value path:

- `Docker single-user + WebUI`
  - setup
  - open WebUI
  - configure provider
  - complete first chat
  - ingest one item
  - search or retrieve it
  - complete first audio verification through the documented path
- `Docker multi-user + Postgres`
  - setup
  - create the first admin user
  - authenticate successfully
  - configure provider
  - complete one authenticated first-value action through the shortest documented path
  - ingest one item
  - search or retrieve it
  - complete first audio verification through the documented path
- `Local single-user`
  - setup
  - start the API cleanly
  - configure provider
  - complete first chat or API success
  - ingest one item
  - search or retrieve it
  - complete first audio verification through the documented path

If a profile offers multiple plausible first-value branches, the review should follow the shortest documented branch first and treat any need to invent a better path as a finding.

### 6. Use a strict but limited severity model

Findings should be limited to issues that matter for public-sharing readiness:

- `Blocker`
  - setup or first-use is likely to fail outright, or the documented path routes users into failure
- `Major confusion trap`
  - a non-technical user is likely to choose the wrong path, miss a hidden prerequisite, or get stuck even though recovery is possible

Minor friction should still be recorded in walkthrough notes, but it should not drive the public-readiness verdict unless it forms a broader pattern.

### 7. Make findings traceable to a concrete journey

Every reported issue must connect to a real onboarding consequence:

- the exact doc or step where the misunderstanding begins
- what a first-time user would likely infer
- what actually happens next
- what hidden knowledge, if any, is required to recover

This review should avoid abstract criticism of documentation style unless it creates a demonstrated onboarding failure or trap.

## Review Architecture

### Stage order

1. Baseline source capture
2. Static onboarding audit
3. Profile walkthrough setup matrix
4. Docker single-user + WebUI walkthrough
5. Docker multi-user + Postgres walkthrough
6. Local single-user walkthrough
7. Cross-profile audio onboarding synthesis
8. Final readiness synthesis

### Data flow

1. Public onboarding docs establish the promised path.
2. The static audit extracts decisions, assumptions, and expected user actions.
3. Walkthroughs test those expectations in the order a new user would encounter them.
4. Findings are written only when the walkthrough confirms a blocker or major confusion trap.
5. Final synthesis groups profile-specific issues and cross-cutting onboarding failures.

## Artifacts

The review should produce two linked output types:

- `Prioritized findings report`
  - grouped by profile and cross-cutting onboarding issues
  - findings ordered by severity and public-readiness impact
  - each finding tagged as `docs`, `runtime`, or `cross-layer`
  - each finding includes source step, user inference, outcome, and recommended priority
- `Walkthrough notes`
  - step-by-step logs for each profile
  - what the docs say
  - what the user would likely infer
  - what happened in practice
  - what unstated knowledge was needed

The static audit should also produce one lightweight comparison artifact:

- `Onboarding contract matrix`
  - profile-by-profile view of prerequisites, auth setup, first-value path, ingest path, audio path, and explicit verification step
  - used to compare what the README, profile guides, `/setup`, and audio docs promise versus what the walkthroughs actually support

Recommended artifact layout:

- `Docs/superpowers/reviews/public-onboarding-readiness/2026-04-24-contract-matrix.md`
- `Docs/superpowers/reviews/public-onboarding-readiness/2026-04-24-docker-single-user-walkthrough.md`
- `Docs/superpowers/reviews/public-onboarding-readiness/2026-04-24-docker-multi-user-walkthrough.md`
- `Docs/superpowers/reviews/public-onboarding-readiness/2026-04-24-local-single-user-walkthrough.md`
- `Docs/superpowers/reviews/public-onboarding-readiness/2026-04-24-findings.md`
- `Docs/superpowers/reviews/public-onboarding-readiness/2026-04-24-synthesis.md`

## Error Handling In The Review

The review should be disciplined about uncertainty.

If a step cannot be fully verified because of environment limits or platform-specific constraints:

- do not overstate it as a confirmed defect
- label it as `Probable risk` or `Open question` in working notes
- only promote it to a formal finding if the evidence supports a likely real-user trap

If docs and runtime disagree:

- record both the documented expectation and the runtime result
- treat the contradiction itself as part of the onboarding defect, not as reviewer confusion

If a profile requires knowledge from outside the declared onboarding surface:

- count that as an onboarding signal
- record the point where the user had to leave the canonical path

## Validation

This review design is successful if it produces a final synthesis that can answer four concrete questions:

- Is the project ready to share publicly with non-technical self-hosters?
- Which of the three public profiles is safest to recommend today?
- Which blockers or confusion traps must be fixed first?
- Which profiles or sub-flows should be deprioritized or hidden until improved?

The final synthesis should include:

- one overall public-readiness verdict
- one per-profile verdict
- a separate cross-profile `audio onboarding readiness` summary
- a platform evidence table showing where conclusions come from executed validation versus doc-only validation

The final recommendation should use one of three verdicts:

- `Ready to share`
  - no blockers and only limited confusion
- `Conditionally ready`
  - core setup can succeed, but major confusion traps remain
- `Not ready`
  - one or more public profiles cannot reliably reach basic first value

## Risks And Constraints

- The review spans multiple OS expectations even if the hands-on execution environment is narrower.
- Multi-user onboarding includes auth and database setup complexity that may dominate findings.
- Audio onboarding is likely to expose cross-doc inconsistencies because `/setup` and the standalone audio guides currently describe different first-run defaults in some places.
- The review must stay focused on onboarding and not expand into a full product audit.

## Recommendation

Proceed with this staged hybrid review design. It matches the actual shape of the onboarding surface, keeps the public-readiness bar appropriately high, and produces artifacts that are useful for both diagnosis and follow-up planning.
