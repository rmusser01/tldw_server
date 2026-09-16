# Native UAT108 fresh recheck — FAILED

TASK13260 / task49. Committed repair 7dcee3d72c. Root intentionally restarted API18301 (PID95509; health200); Next18381 remained running. A normal reload loaded the final source before a new saved General chat. No profile reset, configuration change, or cleanup of prior duplicate records.

| Check | Outcome | Evidence |
|---|---|---|
| New ordinary saved conversation | PASS | New ID 0dff2eee-7bf3-4e92-9d1c-688318065fc1; settled-new/general-menu snapshots |
| Actual unavailable Ollama request | PASS, expected502 | negative-request.txt: exactly one user message |
| Before Retry UI | FAIL | negative.txt: two identical user bubbles after initial502 |
| Exact configured Gemma, actual Retry | Response200, final answer observed | requests.txt, retry-after.txt |
| Retry original request exactly once | FAIL | requests.txt: two identical user messages, explicit retry metadata true |
| No display error in model context | PASS | requests.txt: user content only |
| Canonical original user exactly once | FAIL | canonical-before-reload.txt and canonical-after-reload.txt: two user IDs plus system and assistant |
| Ordinary reload | Confirms failure | Same canonical URL and same four server rows; UI retains local error as fifth row |
| Optional Note/backlink | NOT RUN | Stopped this optional check after canonical duplication failed |

Exclusive Gemma lease was released to root after final answer. No more generation. Separate evidence unit; prior failures preserved. See final-report.md for limits and exact IDs.
