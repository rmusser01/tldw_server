# UAT048 fresh multi-user verification — 2026-09-15

- Isolated UI18181/API18101 only. Product repairs at integrated revision7dc8db2efb; no source edits by this verifier.
- Multi frontend PID89213 remains running for the parent-authorized cross-tab fix and retest. Browser round2-multi-recovery is online; tab1 Alice Notes, tab0 Sign in. Databases/config/profile retained.

## Results before remaining cross-tab repair

| Check | Result | Evidence |
|---|---|---|
| Alice new offline draft stored locally | PASS | uat048-final-alice-offline-draft.txt: exact title Alice UAT048 offline retest 20260915-0725, body,1 pending sync. Explicit Save produced the expected POST/notes connection notice; dismissed normally. |
| Offline Settings Logout completes local sign-out | PASS | uat048-final-offline-logout.txt and .png: Settings Login Required and Login form, no Next runtime overlay. Screenshot visually inspected. |
| Failed remote logout remains honestly reported | PASS | console-2026-09-15T07-25-25-356Z.log lines64-66: failed POST/auth/logout and bounded warning continuing local sign-out without confirmed remote revocation. No successful remote revocation claimed. |
| Local tokens cleared | PASS | uat048-final-local-clear-booleans.txt: configContainsAuthTokens,legacyAccessPresent,legacyRefreshPresent allfalse. No credential values exposed. |
| Other active Notes tab remains usable offline | FAIL — UAT048 remains open | Newly active Notes tab0 from this run automatically changed from /notes to chrome-error://chromewebdata/ upon Logout. uat048-final-notes-tab-logout-error.txt explicitly shows ERR_INTERNET_DISCONNECTED. This was the tab where the fresh draft was created, not a stale preexisting tab. |
| Reconnected Bob sees no Alice draft or metadata | PASS | uat048-final-bob-notes.txt:1 total, Bob Garden Note only, own Recent/pin, empty editor, no Alice title/body or queued indication. |
| Alice draft sync remains with Alice | PASS | uat048-final-alice-recovered.txt:3 own notes, exact fresh draft/body,Version1 and saved time. uat048-final-alice-persisted.txt: authenticated server GET200 returns id2e78af30-3594-455d-a472-dbd3f25b4305,title,version1 after reload. No new-note foreign GET is claimed in this bounded pass. |

## Exact fresh action sequence

Reopened the existing persistent profile at07:24:46 UTC with Notes as new tab0; Alice retained prior own Notes. Opened Settings as new tab1 at07:25:25 while online. Confirmed Logout, set browser context offline, selected tab0, created and saved the synthetic draft. Dismissed the explicit-save connection notice. Selected tab1 and clicked Logout at07:26:59. Settings changed to Login Required while tab0 became the browser offline error page. Reconnected, signed in normally as Bob, verified Notes isolation, logged out online, signed in normally as Alice, and verified draft recovery/server persistence.

The other tab later automatically recovered to Sign in after browser network was restored. The later screenshot is correctly named uat048-final-notes-tab-reconnected-login.png; it is not evidence of the offline error screen. The text snapshot preserves that failure.

## Tooling notes

An earlier CLI session closed before reporting the offline draft step; only the same persistent multi profile was reopened. No current-run draft existed after recovery. A response observer used URL, which is unavailable in this CLI VM; therefore no POST201 capture is claimed. The corrected response observer matched request URL strings and verified the persisted note through Alice GET200. Every browser CLI invocation was elevated to preserve its Unix sockets. Credentials were read by an outer Node helper and credential-bearing CLI stdout/stderr withheld.

Parent/chat reviewer confirmed the remaining redirect cause: _app storage-auth refresh invokes router.push('/login'); offline uncached Next dev Pages manifest fails and falls back to hard navigation. No verifier code edits were made. Browser is paused online for that narrow repair.
