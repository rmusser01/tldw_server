# Frontend Audit — Round-2 Follow-up & Validation

Tracking doc for the work that follows the merged audit remediation
(**PR #2575**, merged into `dev` as `e101f19e81`). It covers two things:

1. **Validations** — changes that shipped in #2575 but that I (Claude) could **not**
   verify without a running app/backend. These must be smoke-tested before the
   affected features are trusted in `dev`.
2. **Continued work** — the follow-up tickets that were deliberately deferred, with
   the reason each was deferred.

Nothing in this PR changes runtime behavior on its own — it is the checklist +
plan for the remaining work. Full audit context is in
[`FRONTEND_AUDIT.md`](./FRONTEND_AUDIT.md).

---

## 1. Validations required before trusting (highest priority)

### ⚠️ V1 — WebSocket auth moved out of the URL (R3 / `task-12113`)
The token was removed from the connection URL and is now sent via the WebSocket
subprotocol (persona) or a `{type:"auth"}` first message (audio/STT). This matches
the backend (`persona.py:3705-3712`, `streaming_service.py`), is unit-tested, and
has a charset fallback — but the handshake/timing were **not** validated against a
running server. If it's wrong, these features fail to connect.

- [ ] **Persona live session** connects and streams (multi-user JWT).
- [ ] **Voice chat** (`useVoiceChatStream`) connects + streams audio.
- [ ] **Streaming transcription** (audio STT) authenticates and transcribes.
- [ ] **Extension STT** (`background.ts`) connects (not unit-tested).
- [ ] In devtools, confirm the WS URL contains **no** `token=`/`api_key=`.
- [ ] **Single-user with a non-token-safe API key** (contains `/`, `=`, space): confirm the query-string fallback kicks in and doesn't throw at `new WebSocket`.
- [ ] Auth arrives **before** config/audio (server 5s auth timeout is met).

Files if a revert/patch is needed: `apps/packages/ui/src/services/persona-stream.ts`,
`apps/packages/ui/src/services/tldw/voice-conversation.ts`,
`apps/packages/ui/src/hooks/useVoiceChatStream.tsx`,
`apps/packages/ui/src/routes/hooks/usePersonaLiveSession.tsx`,
`apps/packages/ui/src/hooks/usePersonaLiveControl.tsx`,
`apps/packages/ui/src/entries/background.ts` (STT implementation; extension
entrypoint re-exported through `apps/extension/entrypoints/background.ts`).

### V2 — CSP dropped `'unsafe-inline'` (H1 / `task-12093`)
`script-src` no longer allows `'unsafe-inline'`; the one trusted inline script
(theme bootstrap) is SHA-256-hash-allowlisted.

- [ ] Console-check `/`, `/chat`, `/research`, `/media`, `/audio-studio`, `/settings`, `/login` — **no** "Refused to execute inline script" / "Refused to load" violations.
- [ ] Theme applies without a flash (proves the hashed bootstrap runs).
- [ ] Exercise the **dev error overlay** / error boundary (dev-only inline scripts).
- [ ] Mic features work under `Permissions-Policy: microphone=(self)`.

---

## 2. Continued work (open tickets, deferred with reason)

| Ticket | What | Why deferred |
|---|---|---|
| `task-12113` | Finalize R3 (WS token-out-of-URL) | Gated on **V1** above — mark Done once smoke-tested live. |
| `task-12116` | Re-enable TS `strict` / remove `ignoreBuildErrors` | Blocked on **~66 pre-existing `tsc` errors** at `strict:false` (16 frontend + ~50 packages/ui) in unrelated code. A real migration: clear the baseline (a `typecheck` script was added to measure it), then enable incrementally. Not a flag flip. (Renumbered from `task-12102`, which a teammate independently used on `dev`.) |
| `task-12108` | Consolidate the 3 `characterChatMode` copies | The high-value **stream-inactivity watchdog** already shipped on both live paths; the full 3-copy merge is a large refactor and the character-chat files are actively churning on `dev` — best done once that settles. |
| H1 follow-up (in `task-12093`) | Drop CSP `'unsafe-eval'` | Needs per-feature (WASM/OCR/tokenizer) browser verification. Also consider a build-time-computed theme-script hash + CSP `report-uri` + HSTS (HSTS is deployment-sensitive for self-hosted HTTP/LAN). |
| `task-12103` | Delete the `extension/routes` mirror | Runtime-unused but kept in sync by **~22 parity tests**; removal must migrate/retire those tests first — negative-value churn during dev's active route work. Quarantined via `_RUNTIME_UNUSED.md`. |

Also open from earlier: the R9/R10 note that two *other* favorite selectors
(`Sidepanel/Chat/CharacterSelect`, `Common/AssistantSelect`) still use the
localStorage favorites store and would need a follow-up to fully unify with the
server flag.

---

## 3. Known pre-existing (not introduced by this work)
- `usePersonaLiveControl` test "loads sessions and chooses backend-focused session" fails identically on the pre-audit base (a session-loading mock dropping args). Unrelated.
- `workspace.ts` quota-warning test and a couple of others were confirmed pre-existing during the audit.

---

## 4. Notes for whoever picks this up
- Do the **V1 voice/STT smoke test first** — it's the one change most likely to break a user-facing feature, and it's the cheapest to validate or revert.
- The audit's per-finding detail, verification log, and reviewer notes are summarized in `FRONTEND_AUDIT.md` (§0.6 and the Round-2 section).
