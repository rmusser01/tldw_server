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

### Stage 2: failed and stopped turns in saved chats (CC-02, CM-02)
**Goal:** a saved-chat send that gets no reply explains why, in plain words,
and offers Retry and Switch model. A Stop before the first token says so and
doesn't duplicate the question.
**Success criteria:** the recovery card names the cause from the error
(provider, model, status), has `role=alert`, and keeps dev's retry rules.
Stop before the first token shows "Stopped before reply" under the kept
question, without also restoring it to the composer. Busy Send reads
"Send after this reply".
**Tests:** harness integration cases for fresh and saved chats (502, Stop
before token); recovery-card unit tests; label tests.
**Status:** Not Started

### Stage 3: long waits (CM-06) and one error surface (CC-N2)
**Goal:** waits show elapsed time and announce plainly; an error shows once.
**Success criteria:** "Waiting for <model>… 0:12"; announcements at ~10 s and
~30 s, no "checkpoint"; a startup timeout names the model; the composer
error banner doesn't repeat an error bubble the user can see.
**Tests:** `ActionInfo` tests with fake timers; banner tests.
**Status:** Not Started

### Stage 4: history review (CC-04)
**Goal:** the owner decision above.
**Tests:** `history-review-presentation` unit tests; `HistorySelectionReview`
tests; integration and side-panel parity cases; e2e helper updates.
**Status:** Not Started

### Stage 5: sampling settings (#3283)
**Goal:** the owner decision above.
**Tests:** `pageAssistModel` and `TldwChat` request-body tests; control gating
tests.
**Status:** Not Started
