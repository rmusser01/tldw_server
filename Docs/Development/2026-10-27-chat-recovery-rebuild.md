# Chat recovery rebuild on dev's retry design (2026-10-27)

Branch `fix/chat-recovery-on-retry-design`, off `dev` `35e711080a`.

This rebuilds the Wave 3 chat-recovery lane (CC-04, CC-02, CM-02, CM-06,
CC-N2; #3101, groups #3106, #3107, #3112) on `dev`'s interrupted/failed-turn
retry design. The original lane is kept on `ux3/chat-recovery`; it could not
merge (see `2026-10-22-ux-remediation-wave3-integration-log.md` §3). It also
fixes #3283 (sampling settings never sent).

## Owner decisions (2026-10-27)

- **CC-04:** move "Use no prior messages" into the history review. A
  "Start fresh" option there moves focus to "Restore all", which is the undo.
  An "AI sees N of M messages · Restore all" line shows a partial selection,
  and hidden messages appear dimmed and read-only in the review. The review
  button appears only with 2+ turns or alternative replies. This supersedes
  TASK-13550 acceptance criterion 2. The transcript still shows only the
  selected messages (ADR-049).
- **#3283:** always send the frequency and presence penalties. Send Top K
  and Min P only when the provider supports them (the server's capability
  registry); otherwise the control is disabled with "Not supported by
  <provider>".

## Re-triage against dev 35e711080a

Checked in code and with a probe on the Playground integration harness.

| Item | Status on dev | What remains |
|---|---|---|
| CC-04 | Partly fixed: both controls hide while the captured history is empty (TASK-13550). | Everything in the owner decision above, plus plain-language review copy and chronological order. |
| CC-02 | Open. After a provider 502 with no output, fresh and saved chats show "Turn needs review / User input accepted; response not saved / Inspect original input and result / Review retry / Copy recovered text / Dismiss recovery". No cause, no alert. Only the model chip says Unreachable (Wave 3). | Plain-language failure card with the cause, `role=alert`, Retry and Switch model. |
| CM-02 | Mostly fixed. A partial reply on Stop is kept and marked Stopped with Retry (CS-04). The in-bubble Stop has a name. Stop before dispatch no longer shows `request_config_scope_changed`. | Stop before the first token: the question stays and the draft is also put back in the composer, with no "Stopped" note. Busy Send still reads "Queue" (the side panel says "Send after this reply", XS-10). |
| CC-N2 | Partly fixed: the model chip turns Unreachable after a provider failure (Wave 3 CC-01). | Error copy is the generic "Something went wrong while talking to your tldw server". The same error shows twice, in the reply bubble and in the composer banner. |
| CM-06 | Open. Screen readers hear "Still generating response (checkpoint N)" every 5 s; no elapsed time. A startup timeout has friendly copy but doesn't name the model. | Plain announcements at about 10 s and 30 s naming the model, a visible elapsed timer, and a timeout message that names the model. |

Not changed: the quiet empty-draft Send is the owner's redesign (#3309).

## Stages

### Stage 1: provider failure copy (CC-N2)
**Goal:** a provider failure says which provider and model didn't answer.
**Success criteria:** a 502/503 or `provider_unavailable`/`provider_error`
gives "Couldn't reach <provider> (<model>)" with a hint that the tldw server
is fine, and "Switch model" as the recovery action.
**Tests:** `chat-error-message` unit tests; a temporary-chat integration case.
**Status:** Complete

### Stage 2: failed turns in saved chats (CC-02, CM-02)
**Goal:** a saved-chat send that gets no reply explains why, in plain words,
and offers Switch model next to dev's Retry.
**Success criteria:** the recovery card says "No reply to this message"
("Reply not saved" when a reply arrived but wasn't saved), then the cause
(`role=alert`), then what is kept. A failure about the model (CC-N2) adds
"Switch model", which opens the picker on the full page and in the side
panel. The other states, "Show what was sent and received" and "Earlier
messages without a reply" use plain words. Busy Send keeps its "Send" label
and is named "Send after this reply", as in the side panel (XS-10).
**Tests:** `HistorySelectionReview` unit tests; a saved-chat 502 integration
case that opens the picker from the card; `useModelPickerOpenRequests`;
`PlaygroundSendControl` label test. Existing copy assertions updated, with
negative checks pointed at the new titles.
**Status:** Complete

**Not done: Stop before the first token (CM-02).** The question stays in the
transcript and the draft also returns to the composer, so sending again
saves the question twice. Keeping an empty "Stopped" reply with Retry was
tried and backed out: dev rebuilds the transcript from the selected history
plus kept turns, and a kept turn without reply text is not shown
(`isRetainedPartialReply` needs `result_text`). Without a Retry there, the
restored draft is the only quick way to ask again, so it stays. A proper fix
belongs with dev's retry design: let a stopped turn with no text use the
saved-question retry, then stop restoring the draft (a `questionKept` flag on
the skipped result is enough for the second part).

### Stage 3: long waits (CM-06) and one error surface (CC-N2)
**Goal:** waits show elapsed time and announce plainly; an error shows once.
**Success criteria:** the status pill shows the elapsed time after 5 s
("0:12"). Screen readers hear "Still working on the reply from <model>." at
10 s and "<model> is taking a while. You can stop the reply and try again."
at 30 s, instead of "checkpoint N" every 5 s. A reply that never starts says
"No response from <model> after 30 seconds." The composer error banner stays
hidden while the transcript shows its latest message (the error bubble), and
appears when the user has scrolled away from it.
**Tests:** `ActionInfo` tests with fake timers; timeout copy unit tests; the
temporary-chat provider failure shows its message once.
**Status:** Complete

### Stage 4: history review (CC-04)
**Goal:** the owner decision above.
**Success criteria:**
- No top-level "Use no prior messages". "Start fresh" inside the review
  hides every earlier message from the AI and moves focus to "Restore all".
- While the AI sees only part of the conversation: "AI sees N of M
  messages · Restore all". When it sees none, the hidden messages are listed
  read-only and dimmed (the latest 3, then "and N earlier messages").
  Hidden while a reply is being sent, unless the AI sees nothing; "Restore
  all" is disabled during a send.
- The review is offered with two exchanges or alternative replies, or when
  it must explain something (legacy review, stale selection, partial choice,
  retained outcomes); not for a single exchange.
- Plain review copy: "You" and "Assistant", plain-text previews, a reply
  never listed before its question, "4 of 4 messages included", "Hidden from
  the AI", "AI sees this as message 2"; rows no longer overlap.
- H1 design §112 and the 2026-10-04 remediation design record the change.
**Tests:** `history-review-presentation` unit tests (ported from
`8ee0bc3208`); 8 `HistorySelectionReview` cases; 2 real-harness cases
(one exchange shows nothing; Start fresh → AI sees 0 of 4 → Restore all).
The retry-race test now starts fresh from the review. Extension e2e helper
updated (not run locally: disk space).
**Status:** Complete

### Stage 5: sampling settings (#3283)
**Goal:** the owner decision above.
**Tests:** `pageAssistModel` and `TldwChat` request-body tests; control gating
tests.
**Status:** Not Started
