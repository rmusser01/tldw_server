# Buddy v1 live qualification

Task: TASK-13227. Artwork incident: TASK-13211. Voice acceptance: TASK-13202.

ADR required: no. ADR path: backlog/decisions/005-independent-buddy-bindings-and-work-ownership.md.
Reason: verify the approved existing boundaries; routine defect repairs require a reproduced mechanism first.

1. Pin merged dev and launch the real authenticated server and WebUI with disposable configuration, storage and browser state. Record source provenance and resolved database locations.
2. Drive fresh setup through Buddy & Persona, select independent artwork, attach a conversation/workspace, save defaults and change expression motion. Verify Apply/Cancel and replies without page navigation.
3. Exercise repeated page/state transitions and inspect artwork/session request counts and failures. Separate the legacy Persona Live host from the independent Buddy host; compare actual behavior with TASK-13211's recorded incident.
4. Check disposable prior-schema upgrade behavior and the packaged extension. Record source-bound screenshots, request receipts and limitations. If a defect is reproduced, add a focused regression before repairing and reviewing it.
5. Repair the reproduced first-use model-catalog blocker: shipped blank optional `*_max_tokens` values raise during catalog construction. Preserve unset limits, verify valid configured limits and the original configuration with a focused regression, then resume real Buddy replies.
6. Repair the reproduced Buddy transcript error-envelope leak using existing chat error decoding. A controlled Chat timeout is stored as an encoded error; the Buddy should show its readable summary/hint and preserve ordinary messages, with a focused regression.
7. Apply the same assistant-only presentation to queued read-aloud after its authorized transcript read. Preserve conversation-name prefixes and ordinary/user content; add an activity-to-speech regression, review the correction and rebuild the Chrome artifact.

Use only synthetic test content and isolated profiles. Do not capture a physical microphone automatically. Do not infer a real provider/audio pass from fixtures. No full local test sweep; root owns UI operation and integration while independent Chatbook checks use their own worktree/data.

## Qualification status

Fresh WebUI setup, model-catalog repair, Static/Dynamic options, workspace default inheritance, accepted reply continuation across navigation, exact workspace reply targeting and individual result acknowledgement are verified. Both reproduced production defects have focused regressions. Backend SQLite upgrade/rollback and independent Buddy/turn contracts passed; the PostgreSQL case skipped because no database was reachable. The Chrome production artifact includes the final assistant-only error presentation fix and passes its build gates.

The historical legacy artwork loop remains unreproduced with no speculative repair. Native Terminal is explicitly disallowed by Computer Use, and native Chrome lacks Computer Use permissions. Packaged extension installation, native Chatbook, upgraded-profile WebUI and real voice/audio therefore remain open. See `Docs/Reviews/2026-09-09-buddy-v1-qualification.md` for receipts, exact checks and follow-up usability findings. Keep the qualification task In Progress.


## Usability follow-ups (TASK-13228, TASK-13229, TASK-13230)

ADR required: no. ADR path: backlog/decisions/005-independent-buddy-bindings-and-work-ownership.md.
Reason: repair the three recorded usability gaps within the existing attachment, settings, and work-ownership boundaries.

1. Move fresh/reset Buddy placement below the top navigation, preserving saved user placements and viewport clamping. Verify composer hit targets in the live WebUI and focused layout tests.
2. Derive consistent labels for same-titled loaded conversations using localized creation times and stable identifiers when needed. Keep saved titles and reply/acknowledgement IDs unchanged; test long titles, timestamp collisions, and reordered lists.
3. Preserve explicit provider/model selections from authorized neutral workspace Chat. Read effective reply settings through the current Buddy attachment, sharing acceptance resolution. Show missing setup before Send, retain drafts, and preserve per-reply overrides. Verify with owned SQLite/HTTP journeys and live UI recovery.
4. Review, run focused checks and update the existing PR with source-bound follow-up evidence. Native extension/terminal and actual voice remain separate qualification gaps.
