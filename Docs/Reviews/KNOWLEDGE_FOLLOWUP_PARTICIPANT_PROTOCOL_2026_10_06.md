# Knowledge follow-up participant protocol — TASK13512

This is a ready-to-run formative study protocol. No human participant sessions have been conducted. Agent browser, accessibility-tree and API checks are implementation evidence, not observed human usability.

Use people who would actually ingest and research their own content. Ask them to think aloud, observe their actions, and avoid teaching the interface during the task. Distinguish independent success from success after assistance. These practices follow [NN/g's usability testing guide](https://www.nngroup.com/articles/usability-testing-101/) and [thinking-aloud guidance](https://www.nngroup.com/articles/thinking-aloud-the-1-usability-tool/). Think-aloud timings are descriptive observations, not performance benchmarks.

## Participants and setup

Run an initial pass with first-time users and people already comfortable with the extension, Notes and Research Workspace. Keep those groups separate in the findings. Include a participant who normally uses a screen reader and a participant using a real iPhone/iPad when available. Record browser, OS, input method, assistive technology, experience and display size. Viewport emulation does not exercise an onscreen keyboard, device browser chrome or human screen-reader strategies.

Use a disposable account containing two related greenhouse documents, one unrelated orchard document and a research note. The baseline has three calibration days and 120 litres of water per plot per week; the refined phase has five days and 90 litres. The note's appendix gives the next review date. Operating cost was not recorded. Use a separate staged ingestion failure for retry tasks; keep real participant files outside logs/screenshots unless they consent.

## Moderator tasks

| Persona | Task given to participant | Evidence to observe |
| --- | --- | --- |
| First use | Add one document, then find out how long baseline calibration took. | Discovery of Add, understanding of readiness, use of Ask, identification of supporting source. |
| First use | Add three files; one fails. Finish adding the remaining content. | Whether the failure is noticed, which action is retried, whether successful items are duplicated. |
| First use | Explain what the answer's citation/status label tells you. | Distinction between available references, exact matching and verified claim support. |
| First use | Save this answer, revise one sentence, leave, and find the revised content. | Save acknowledgment, canonical Notes discovery, loss of edits, retained source references. |
| Power user | Select only the two greenhouse documents and compare their water use. | Source scope recognition, excluded orchard content, multiselection persistence, batch review. |
| Power user | Review both selected documents and open their evidence in turn. | Source-to-claim mapping, keyboard traversal, list/detail orientation, return to selection. |
| Power user | Continue the note into Research and locate its appendix. | Recognition of full versioned snapshot versus original retrieved excerpts; understanding of refresh. |
| Power user | Ask for the operating cost and decide what to do next. | Recognition of missing evidence, abstention, next research action, avoidance of invented certainty. |
| Extension | Capture this article, save a note, and ask a question about that saved item. | Native entry point discovery, captured content review, save feedback, scoped Ask handoff. |
| Mobile / assistive technology | Repeat Add, Ask, evidence navigation and Save using your normal input method. | Spoken labels/live changes, focus return, keyboard obstruction, scrolling and dialogs. |

Ask after each task: “What happened?”, “What would you do next?”, and “Which source supports that conclusion?” Use neutral prompts if the participant falls silent. Record assistance separately; do not count a demonstrated route as independent completion.

## Observation record

For each session/task record: participant pseudonym, experience group, device/browser/input, attempted path, independent/assisted/failed outcome, exact error or hesitation, recovery, source interpretation, confidence explanation and proposed severity. Preserve short consented quotes only when needed to explain a finding. Leave missing observations empty.

Review recurring breakdowns against visibility of system status, user control, error prevention/recovery, recognition and flexibility. Use the observed consequence to rank severity; task completion alone cannot show that a cited answer was understood or trusted appropriately.

## Coverage still required

Real novice/power-user sessions, actual VoiceOver spoken navigation, Safari/iOS interaction, and real mobile keyboard behavior remain pending. Desktop Safari rendering and automated keyboard/AX checks cannot replace these sessions.
