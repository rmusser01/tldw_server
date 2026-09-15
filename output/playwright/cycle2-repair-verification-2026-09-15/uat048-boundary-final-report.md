# UAT048 final live retest — PASS

Date:2026-09-15. Isolated UI18181/API18101 only. Frozen cross-tab signed-out gate plus title follow-up; no verifier source edits or commits.

## Exact workflow and results

1. Reloaded Alice Notes online to load the updated app; refreshed a separate Settings tab online and confirmed Logout.
2. Set the shared browser context offline. Created Alice UAT048 signed-out boundary 20260915 with a distinctive private body. Normal Save stored1 draft pending sync. Dismissed the expected explicit POST/notes connection notice. Evidence: uat048-boundary-alice-offline-draft.txt.
3. Clicked Settings Logout offline. The other active Notes tab stayed at http://127.0.0.1:18181/notes and rendered only Signed out / Reconnect to sign in. Its private draft, list, editor, and metadata were unmounted. No Chrome error page, optional Help chunk failure, or Next runtime overlay appeared. Evidence: uat048-boundary-notes-signed-out.txt/.png and uat048-boundary-offline-booleans.txt. Screenshot visually inspected.
4. Restored browser network. That same tab automatically navigated to /login with Sign in | tldw. Evidence: uat048-boundary-auto-login.txt.
5. Signed Bob in normally through the UI using the private fixture credentials. Notes showed1 own Bob Garden Note, own Recent/pin, empty editor, no Alice title/body or queued-draft indicator. Evidence: uat048-boundary-bob-notes.txt.
6. Logged Bob out online and signed Alice in normally. Opening Notes synced the exact queued draft with observed POST201. New server note f6d2d7cd-a29a-4014-be48-b338d71633e9,version1. Alice showed4 own notes, exact title/body and savedVersion1, no queue. Evidence: uat048-boundary-alice-sync.txt and uat048-boundary-alice-synced.txt.
7. Independent server ownership control used Bob normal password authentication through the same auth/login endpoint: login200, GET new Alice note404, temporary verifier logout200. Evidence: uat048-boundary-bob-foreign-note.json. No credential/token/header values were printed.
8. After the title-only follow-up, reloaded online and repeated a short two-tab offline Logout without creating another draft. Confirmed URL /notes, document title Signed out | tldw and only native signed-out text;0 console errors and3 bounded auth warnings at this checkpoint. Evidence: uat048-boundary-final-title.txt, uat048-boundary-final-signed-out.txt/.png. Screenshot visually inspected.
9. Restored network again; automatic /login, title Sign in | tldw, navigator.onLine true. Evidence: uat048-boundary-final-online.txt.

## Scope and limitations

The remote logout call predictably fails while offline and logs the bounded warning that local sign-out completed without confirmed remote revocation. This result does not claim successful remote token revocation while offline. The earlier UAT048 browser-error failure is retained separately in uat048-final-multi-report.md; the current report records its successful repair retest.

Only ordinary UI operations, the explicit bounded ownership control, and required runtime retirement were performed. The new Alice test note is deliberately retained. All browser CLI calls used escalation to preserve session sockets. Browser profile/database/config/evidence remain available, with browser online at Sign in. Multi frontend retirement is recorded below once verified.

## Runtime retirement

Verified frontend PIDs89213 and89214 stopped; removed only the untracked generated multi frontend cache .next-live-tier-full-multi-20260915. The multi backend18101, databases, private config, browser profile, and evidence were retained. Browser was last verified online at /login before frontend retirement.
