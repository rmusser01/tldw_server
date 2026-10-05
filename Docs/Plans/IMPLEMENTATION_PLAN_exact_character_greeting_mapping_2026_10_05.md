# Exact character greeting mapping — TASK-13454.7

Baseline cc68a1c04e; origin/dev025627214c integrated. Existing character-greetings utility defines compatibility display ordering and IDs. Backend character_conversation_factory._select_creation_greeting consumes raw first_message or a raw alternate_greetings list index; it does not use the filtered/deduplicated display index. Existing chat-display-name helper and both real consumers handle rendering.

## Stage 1: Reproduce mapping and identity mismatches
**Goal**: show raw-index loss, unsupported seed aliases and conflated placeholder names.
**Success Criteria**: meaningful failing tests before changes.
**Tests**: existing utility suite plus shared name helper and actual hook/picker consumer assertions.
**Status**: Complete

## Stage 2: Preserve canonical seed descriptors
**Goal**: retain raw default/alternate position and text separately from display text without changing compatibility IDs/order/checksums.
**Success Criteria**: blank/nonstring entries and duplicate text cannot shift the seed; unsupported aliases/encodings have no descriptor.
**Tests**: canonical first_message, raw list/single JSON list, double encoding limitation, duplicate alias/canonical/default/alternate cases.
**Status**: Complete

## Stage 3: Render distinct identities
**Goal**: user placeholders use the user name; character placeholders use the card name; missing names preserve unresolved tokens.
**Success Criteria**: both current callers pass the card name; raw seed templates stay untouched.
**Tests**: shared helper and actual hook/picker with user/character and literal dollar text.
**Status**: Complete

## Stage 4: Verify and commit
**Goal**: reviewed bounded prerequisite.
**Success Criteria**: affected suites/types/extension entrypoint/no added lint/diff/two reviews. Actual preview isolation, native create/load/owned-row/zero-provider greeting startup, Sync and live acceptance remain required.
**Tests**: existing backend selection checks where available; Bandit N/A for TypeScript-only production scope.
**Status**: Complete


Verification: six mapping/name helper failures in `/private/tmp/chat-greeting-exact-mapping-red.log`, two actual-consumer failures in `/private/tmp/chat-greeting-identity-consumer-red.log`, and one reviewer-found missing-card-name failure in `/private/tmp/chat-greeting-missing-name-red.log`. Final4suites38pass, existing greeting API7pass, actual existing backend raw-selector parity passes; WebUI types/extension entrypoint compile/no added touched lint/diff checks pass. Logs `/private/tmp/chat-greeting-exact-mapping-final-{green,types,extension,lint}.log`, `/private/tmp/chat-greeting-exact-mapping-backend.log`, `/private/tmp/chat-greeting-backend-parity.log`. Both authorized source reviewers clear. Bandit N/A TypeScript-only production scope.

Descriptors retain raw first_message/default or raw alternate_greetings position/text even when compatibility aliases occur earlier or blank/nonstring/duplicate entries disappear from display. Only one JSON decode matches the backend seed contract; nested encodings and alias-only content remain display-compatible without a fabricated seed. Speaker label fallback is separate from character identity; missing names preserve placeholders, and dollar-containing display names are literal. Original seed text is never rendered or substituted in-place.

Display checksums intentionally remain compatible and do not attest raw seed positions. Actual startup must reread the current descriptor and qualify stored owner/history rows; seed headers and IDs alone remain insufficient. AntD is mocked in current consumer tests. Preview isolation, actual native startup/zero-provider behavior, raw stored greeting validation, packaged/native/VO/Sync and broader stage acceptance remain open. Fresh dev advanced27ce976387 (PR3195 native ordinary first-send) during verification; integrate immediately after this local checkpoint. Completed .6 plan archivedcc68a1c04e and removed per repository policy.
