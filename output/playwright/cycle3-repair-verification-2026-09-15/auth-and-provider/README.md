# Cycle 3 targeted auth and Provider Keys checks

These checks use the existing isolated multi-user runtime on API port 18201 and WebUI port 18281. They are repair verification, not a new fresh full UAT. The original frozen run and its hashes remain unchanged.

## Results

| Check | Observed result | Evidence |
| --- | --- | --- |
| Natural access-token expiry | Bob logged in normally at 14:55:46 UTC; access expired at 15:25:46. Refresh request 399 returned 200 at 15:25:50 and private requests recovered. | `uat080-settings-requests-after-expiry-metadata.json`, `uat087-runtime-credential-projection.json` |
| Notifications after that refresh | Both tabs retained sign-in-required state despite successful private reads. This is UAT087; its later repair is not validated by this failed capture. | `uat080-settings-after-expiry.txt`, `uat080-notes-after-expiry.txt` |
| Actual rotated-session revocation | Only Bob's own current test session was revoked, returning 200 at 15:34:38. The subsequent refresh returned 401, but private polling/UI initially continued. The raw original JWT was incorrectly treated as a newer login. | `uat080-current-session-revocation.json`, `uat080-terminal-refresh-token-metadata.json`, `uat080-notes-requests-after-revocation-metadata.json`, `uat080-notes-after-revocation.txt` |
| Revocation after helper correction | On `bfcb393a44` source, Notes redirected to sign-in and the other Settings tab displayed Login Required with reset checks. | `uat080-after-helper-fix.txt`, `uat080-settings-after-helper-fix.txt` |
| Polling after invalidation | Settings resource timing totals stayed at 135 resources / 7 private fetches between 15:50:20 and 15:51:13 UTC; the latest private-fetch timestamp did not change. | `uat080-settings-poll-metadata-start.txt`, `uat080-settings-poll-metadata-end.txt` |
| Ordinary replacement login | Bob's UI login at 15:52:26 issued a new session expiring 16:22:26. Notes restored all three saved rows and active notifications. | `uat080-relogin-credential-metadata.txt`, `uat080-notes-after-normal-relogin.txt` |
| Other-tab Settings recovery | Settings still displayed Login Required after that login. This separate stale presentation is UAT088. The initial task note claiming recovery was corrected. | `uat080-settings-after-normal-relogin.txt`, `uat080-settings-relogin-settled.txt` |
| Provider Keys | Normal admin login in a separate browser profile, then `/settings/provider-keys`, rendered precise BYOK-disabled guidance for the actual 403. No prior ICU/object rendering crash. Repair committed as `7e48f29cb1`. The empty title remains UAT058. | `uat077-admin-provider-keys-live.txt`, `uat077-provider-keys-console.txt` |

## Boundaries and retention

- The API contained the reviewed auth backend repair `3751292380`; the development WebUI loaded the incremental frontend repairs. Hot reload occurred, so this is not a frozen-build acceptance run.
- No clock changes, token-expiry changes, mock model answers or synthetic refresh responses were used for the live expiry/revocation checks.
- The next natural expiry must verify notifications on reviewed repair `fb79cc565a`. Settings recovery needs its own completed repair and targeted check.
- Request evidence is reduced to request number, HTTP method, endpoint category, port and status. Raw headers, bodies, query strings and credential values are not retained here.
- All retained captures passed exact known-runtime-credential, JWT and private-key scans. `retention-summary.json` records the result; `SHA256SUMS` indexes capture bytes. Read-only browser timing metadata was used after automatic approval review rejected an unfiltered request-log capture.
