# Single-user bounded native verification — 2026-09-15

## Result

PASS for the assigned existing Aster backlink, two normal reloads, native browser mirror, and saved Robot/Aster sidebar switching. No remaining finding in this bounded check. This is not a full UAT acceptance claim.

Session: `cycle3-single-20260915`, tab0, `http://127.0.0.1:18280`. Final browser state is the original Aster conversation with the Server sidebar expanded and scrolled to its row. Browser work is complete and paused.

The final reviewed 20-file runtime manifest matched all files at the start (23:14:24 UTC) and completion. See `uat-cycle3-single-final-resumed-source-check.json` and `uat-cycle3-single-final-completion-source-check.json`. This resumed run used `/private/tmp/uat-cycle3-reviewed-runtime-source-manifest.json`; initial/paused artifacts before 23:13 are historical only.

## Actual UI checks

1. Opened the existing original Note via its Notes-list button. The recorded Playwright action resolved `notes-open-button-080ce690-1d8f-4920-86b9-bc90bcc4f7ba`. Loaded Note showed Saved, version1, and its original garden answer. Used actual **More actions → Open conversation**.
2. Backlink settled at `/chat` with browser/header title `Cycle3 Aster Guide Chat (20260915_084036)`, character4 (`Cycle3 Aster Guide`), and three original messages: greeting, raised-bed question, and exact answer **Project Aster has seven raised beds.**
3. Performed two normal CLI `reload` operations. Both settled with the same title, character4, three messages, and exact answer. No visible build overlay or unexpected-error screen appeared.
4. Opened the actual Chat sidebar and expanded Recent conversations. Server view showed nine saved chats. Mouse-clicked the discovered `Cycle3 Beep Robot Chat (20260915_084745)` row; settled title and character5 (`Cycle3 Beep Robot`) matched, with two original messages: `Hello, who are you?` and `BEEP BOOP`.
5. Mouse-clicked the original `Cycle3 Aster Guide Chat (20260915_084036)` saved row. It restored the original Aster ID, character4, title, three original messages, and exact answer. At 23:19:41 UTC its state had `serverChatMetaLoaded=true`, `isLoading=false`, and the same owned history/message identities as after reload2.

One initial sidebar click used an obsolete snapshot ref and was rejected by the CLI before any action. A fresh snapshot supplied the valid Expand sidebar control; the normal click then succeeded. This was an automation reference issue, not an observed product failure.

## Read-only native IndexedDB check

Read the already-existing `PageAssistDatabase` through native IndexedDB, aborting any unexpected database upgrade and using readonly transactions only. At 23:17:04 UTC:

- Active Aster server chat: `e004e361-604c-4a6a-85ce-019bc75fdeea`.
- Active history: `pa_8f95-08a1-778-c8af`, with an owner marker and matching saved title.
- All three mirrored messages use local IDs qualified by that history and retain canonical server message IDs.
- The answer retains canonical ID `pa_ec2a-b676-39f-15e0` and contains the exact garden answer.
- Visible state and owned mirror agree; metadata loaded and chat idle.
- Existing legacy unowned histories were also present; they are not the active history. No records were changed or deleted by the probe.

Robot's actual selected server chat ID, discovered after its UI activation, is `ff85f37d-2c47-4467-af8b-db3fc305729b`; its state had character5, two messages, loaded metadata, and idle loading. Return to Aster restored the same active history and all three qualified IDs.

## Evidence (all under `/private/tmp`)

- `uat-cycle3-single-final-aster-note-open.txt`, `...-aster-note-loaded.txt`, `...-aster-note-menu.txt`, `...-aster-backlink-action.txt`: actual original Note and backlink controls.
- `uat-cycle3-single-final-aster-backlink-settled.txt` / `...-aster-backlink.png`: destination.
- `uat-cycle3-single-final-aster-reload1-action.txt` / `...-aster-reload1-settled.txt`: first normal reload.
- `uat-cycle3-single-final-aster-reload2-action.txt` / `...-aster-reload2-settled.txt` / `...-aster-reload2.png`: second normal reload.
- `uat-cycle3-single-final-aster-reload2-dexie.txt`: native readonly mirror plus current store identity.
- `uat-cycle3-single-final-sidebar-recent.txt`, `...-robot-row-click.txt`, `...-robot-settled.txt`, `...-robot-state.txt`, `...-robot.png`: actual discovered Server row and Robot result.
- `uat-cycle3-single-final-aster-return-row-click.txt`, `...-aster-return-settled.txt`, `...-aster-return-state.txt`, `...-aster-return.png`: actual saved-row return and final state.

All four resumed-run PNGs (backlink, reload2, Robot, Aster return) were visually inspected. Screenshots show the exact answer or BEEP BOOP unobscured. Greeting/user rows can be above the viewport; snapshots and state capture verify their presence. Sidebar mouse activation succeeded with the actual scrolling container.

## Boundaries

No inference, message send, identity edit, scene drawer interaction, title rename, deletion, or content mutation was performed. Navigation/reloads naturally exercised application persistence; the diagnostic reads did not mutate storage or application state. No source, test, runtime, global documentation, or commit changes were made. Other browser tabs were left untouched; the preexisting error tab was outside this task.

Root owns multi-user/auth testing, definitive layout geometry, title rename, scene retry verification, durable native evidence integration, and final whole-cycle acceptance. The compiler result (90 known baseline diagnostics, zero new) was supplied by root and was not rerun during native verification. No offline or extension continuation claim is made here.
