# Chat usability-study script

TASK-13458; approved scope: Docs/Design/2026-10-04-chat-conversation-ux-remediation-design.md. This is a collection script, not participant results. No recruitment, telemetry or recordings are enabled by this document. ADR assessment: existing client UX scope; no new architecture decision.

## Preparation

Use a disposable local account and synthetic conversations/cards. Record commit, surface (WebUI, full extension or compact), browser, machine, viewport, theme, zoom, input/assistive technology and participant experience. Give novice and experienced participants equal time. Start with a working connection and usable provider; time connection/provider setup separately. Use neutral task wording and do not point out controls. Obtain consent before any recording; record no credentials or private conversation content.

## Tasks

| Task | Neutral instruction | Completion evidence |
|---|---|---|
| First conversation | “Ask a question, give this conversation a name you will recognize, then leave it.” | First reply received and intended conversation named; distinguish temporary from saved mode. |
| Find and continue | “Find the conversation you just named and ask a follow-up in it.” | Correct account/origin/conversation reopened, prior context visible and follow-up in intended conversation. Include duplicate titles among distractors. |
| Character setup | “Import this supplied character card, inspect what will guide its responses, and begin a scene with it.” | Intended card selected; greeting or own opening understood; two successful turns. Unsupported tracked persona is explained rather than offered as a working path. |
| Character return | “Leave this scene, return to it, change one supported scene setting, and continue.” | Correct saved session resumed; scope/commit acknowledged; next turn uses supported change or clearly explained frozen limitation. |
| Power workflow | “Reuse a saved preset, inspect next-turn context, find a message and use an available keyboard command.” | Preset scope and context provenance understood; message found; focus returns correctly and input preserved. |
| Recovery | “The connection/provider is unavailable. Keep your input safe and work out how to continue.” | Input retained; useful recovery action found; saved-history boundary explicitly reviewed where required; no automatic resend. |
| Surface transfer | “Continue your draft in the other supported chat surface.” | Destination and carried text understood; omitted settings/history/resources explained; no wrong-account history mutation. Compact-only limitations recorded. |

Counterbalance first-conversation and character tasks across participants. Experienced participants also complete the same baseline tasks; do not replace them with expert-only work. Stop a task when the participant declares completion or asks to stop. If assistance is requested, record the request and elapsed time before giving one neutral hint; mark assisted completion separately.

## Record for each task

Record start/end time, independent/assisted/incomplete completion, wrong-chat selections, assistance requests, error/recovery events, scope misunderstanding and a post-task ease rating (1 very difficult–7 very easy). Ask “What did you expect to happen?” and “What made this easier or harder?” Avoid explaining the implementation or suggesting the answer. Record observations against all ten Nielsen heuristics: status, real-world language, control, consistency, prevention, recognition, efficiency, minimal presentation, recovery and help.

Proposed targets are two minutes to first standard message after usable connection/provider setup and thirty seconds to reopen the correct conversation. They are product hypotheses, not measured outcomes or automatic release criteria. Keep setup time, provider latency and task time distinct. Report participant counts, experience, surface and assistance alongside any timing; do not infer participant success from automated checks or expert inspection.

## Reporting

Report actionable observations with trigger, observed behavior, expected behavior, affected audience/surface, severity and smallest proposed remedy. Separate actual measurements, expert judgments and unrun cells. Retain only consented, minimal synthetic evidence; do not publish credentials or transcripts. No participant measurements have been collected for this script.
