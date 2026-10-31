# Bound native roleplay feedback — TASK-13670

The already-bound/private shortcut delegates to sendSavedRoleplay without the existing guarded error wrapper used after fresh binding. Popup consumers therefore see raw provider transport text and cannot fence the error against its original H1 source.

Reuse NativeHistoryRoleplaySendError only around the delegated send, with captured original owner/view/lease plus draft/source currentness. Preserve typed branch/context/settings refusals, preparation/record-read errors, existing journal writes and original IDs. No new retry, automatic replay, admission claim or error-class hierarchy. Existing13664 recovery policy and ADR-049 govern; no new durable architecture rule.

Scope: native helper, real saved-roleplay transport/journal regressions, connected popup/full/compact consumers and their unchanged recovery assertions. Positive401 must retain original input/append and settings guidance; held failures after H1/source/account/draft retirement must not qualify current feedback.
