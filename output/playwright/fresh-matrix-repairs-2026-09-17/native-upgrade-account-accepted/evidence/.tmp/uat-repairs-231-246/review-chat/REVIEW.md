# Independent review — UAT231 / 232 / 236 / 243

**Verdict: CLEAR for this bounded four-repair unit. Native acceptance remains pending.**

Reviewed frozen source manifest `43713006f45a461a49d7c6a5e202c98c94838c2da60c6e14ca9bdf3df4a97c8c`: six production files and ten tests. Every current/snapshot hash matched before and after verification. All six production baseline snapshots also match commit `84f1d4e2511e7c52a7bb4128999192fbb3342aa7`.

## Source assessment

- **231:** unconfigured-server guidance now names Settings correctly for the current web/extension surface; configuration and request behavior are unchanged.
- **232:** the existing unavailable-model recovery UI recognizes the actual HTTP error envelope and sanitizes diagnostic detail. Unrelated failures retain their existing classification.
- **236:** llama.cpp aliases use the existing resolver/readiness rules without weakening configured-provider/catalog checks. Settings uses the existing selected-model hook, updating the active owner and persisted selection together; full model identifiers are preserved.
- **243:** the first saved ordinary completion avoids reselecting an already-published conversation ID, preserving resolved metadata. The existing authority check remains before publication. Character preference, subsequent routing, and switching controls remain covered.

No actionable defect was found in these changes. Scope is limited; this is not an audit of all Chat authority or transport behavior.

## Independent verification

| Check | Result |
|---|---|
| Focused actual client/hook/coordinator/components, ten files | **327 passed**, zero skipped, 38.23s |
| Relevant adjacent contracts, eight files | **103 passed, 2 failed**, zero skipped, 5.73s |
| Same speech suite against baseline source | **2 passed, same 2 failed**, 0.83s |
| Actual-boundary tests against baseline source | **5 expected failures**, 206 filtered, 5.15s |
| ESLint baseline/current logical paths | 623 / 623 warnings; zero errors; no added/removed diagnostics |
| TypeScript baseline/current | 90 / 90 pre-existing diagnostics; no added/removed diagnostics |
| Bandit on six production TS/TSX files | Six parse errors; **no meaningful TypeScript security coverage** |
| Current/source snapshot consistency | All 16 hashes stable |

The two adjacent failures are existing `synthesizeSpeech` fixtures: they populate a private `client.config` while the mocked authoritative storage returns null. Both fail configuration lookup before exercising their timeout assertions. The corrected surface wording changes that error text but not the failing boundary. They are preserved, not suppressed. The five baseline actual-boundary failures independently confirm corrected model guidance/ordinary metadata behavior; the final source passes those tests in the full focused run.

Compiler and lint are differential checks, not clean-build claims. Tests use deterministic local collaborators; there was no native browser run, ASGI/proxy timing measurement, provider inference, or DB mutation in this review.

## Separate authority finding and UAT246 limits

`AUTHORITY-FINDING.md` records an independently confirmed **pre-existing, separately scoped** late Character persistence dispatch after actual WebUI logout invalidates the real lease. The four repairs do not cause or fix it. It is sent to the controller for a new tracked repair; targeted native launch should remain held until that disposition is resolved. No cross-account backend write is claimed.

The retained UAT246 fake-clock/fetch controls establish configured idle timing, progress activity, and caller-abort behavior only. A probe that aborts both a fake invalidation signal and the caller cannot establish that real principal changes always abort the caller. The new actual-lease/logout probe addresses that gap and fails. None of these controls proves the cause of the original native 45-second failure or closes UAT246.

## Evidence

`reviewer-manifest.json` binds this report, the separate finding, exact commands, independent logs, static comparisons, source verification, private probe and its initial setup error. No production/test source, runtime, browser, task, tracker or Git mutation was made.
