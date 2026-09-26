# Saved Chat retry and accessible failure presentation

Tasks TASK13260.64 (UAT124) and TASK13260.65 (UAT125). Native discovery on53ef4bf4d5 is retained separately before edits. The user requested all identified issues addressed before another full UAT. Both isolated Next frontends are paused; APIs and synthetic records remain intact.

## UAT124: restore the existing failed-variant relationship

The live UI groups retries under their user, but the general exception path passes the raw optional assistantParentMessageId to saveMessageOnError. The resolved parent is already used in the live UI and other persistence paths. Persisted errors therefore have no parent, and the existing formatter cannot group them on reload. Use the resolved parent in this path, then exercise actual pipeline, persistence and restoration. Preserve the latest generated variant, exact local user/image, ambiguous server Retry semantics and owner cancellation. No backend change, retrospective guessing of damaged records, or new persistence for arbitrary manually selected historical variants.

## UAT125: announce failures truthfully

PlaygroundMessage briefly renders the polite Response complete announcement whenever processing/streaming stops, including a decoded error. Keep the existing error alert and its recovery actions; suppress this completion announcement for error payloads. Successful completed responses retain their announcement. Immediate active Retry also suppresses the old completion state, as a reused message slot can retain it. Production scope is Message.tsx; regression cases belong in the existing error-recovery integration suite. No broad ActionInfo lifecycle redesign.

## Verification

Add failing behavior tests first, make the smallest corrections, and run related persistence/error/accessibility tests. Independent review checks the shared parent relationship and successful/error announcements. Compare scoped lint and full compiler against the retained baseline. Retain native repeated Retry/reload and immediate accessibility-tree evidence on frozen commits before full fresh cycle5.
