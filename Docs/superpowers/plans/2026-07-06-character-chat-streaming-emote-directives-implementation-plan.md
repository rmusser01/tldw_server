# Character Chat Streaming Emote Directives Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement explicit `Emote: <state>` directives for character chat so portraits can change during streaming while visible and persisted chat text stays clean.

**Architecture:** Add a small shared directive parser contract, then wire it into the existing character chat stream and persist paths. Frontend owns live streaming portrait changes; backend owns API validation and server-side non-streaming sanitization. Keep classifier mood detection as fallback only.

**Tech Stack:** TypeScript, React, Vitest, FastAPI, Pydantic, pytest, existing `metadata_extra` message storage.

---

## Scope And Sequencing

Implement `TASK-12163` only. `TASK-12164` is a follow-up and must not be pulled into this work.

Use TDD. Each task should leave the repo in a working state and commit before moving on.

## File Structure

Create:

- `apps/packages/ui/src/utils/character-emotes.ts`
  - Frontend directive grammar, final parser, streaming parser, event cap, state normalization, metadata guards.
- `apps/packages/ui/src/utils/__tests__/character-emotes.test.ts`
  - TypeScript parser and shared-fixture coverage.
- `apps/packages/ui/src/utils/__fixtures__/character-emote-directives.json`
  - Shared parser vectors read by both TypeScript and Python tests.
- `tldw_Server_API/app/core/Character_Chat/emote_directives.py`
  - Backend final parser and event validation helpers.
- `tldw_Server_API/tests/Character_Chat_NEW/unit/test_character_emote_directives.py`
  - Python parser and shared-fixture coverage.

Modify:

- `apps/packages/ui/src/utils/character-mood.ts`
  - Broaden mood image parsing/resolution to safe custom emote slugs while keeping `detectCharacterMood()` on built-in labels.
- `apps/packages/ui/src/utils/__tests__/character-mood.test.ts`
  - Add custom emote image cases.
- `apps/packages/ui/src/store/option/types.ts`
  - Add typed `emote_events` metadata shape.
- `tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py`
  - Add `CharacterEmoteEvent` model and optional `emote_events` to stream persist request/response path as needed.
- `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py`
  - Add prompt instruction, backend non-streaming sanitization, stream persist metadata storage, and idempotent side-effect handling.
- `tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_chat_stream_and_persist.py`
  - Add persist validation and metadata integration cases.
- `apps/packages/ui/src/hooks/chat/useChatActions.ts`
  - Apply streaming parser and explicit emote metadata in the primary WebUI character-chat path.
- `apps/packages/ui/src/hooks/chat/useCharacterChatMode.ts`
  - Apply the same helper logic to the older/parallel character-chat hook.
- `apps/packages/ui/src/hooks/useMessage.tsx`
  - Apply the same helper logic to the legacy character-chat stream/persist path so sibling callers do not leak directives.
- `apps/packages/ui/src/hooks/chat/__tests__/useChatActions.character.integration.test.tsx`
  - Add streaming/persist tests for stripped directives and explicit mood override.
- `apps/packages/ui/src/hooks/chat/__tests__/useCharacterChatMode.contract.test.ts`
  - Add parity coverage for the older hook.
- `apps/packages/ui/src/components/Common/Playground/Message.tsx`
  - Stop narrowing explicit `moodLabel` to built-in classifier labels before portrait lookup; preserve custom explicit emote slugs.
- `apps/packages/ui/src/components/Common/Playground/__tests__/Message.routing-fallback.integration.test.tsx`
  - Add a minimal portrait resolution test for custom explicit emote slugs.

Do not modify character editor mood-image management in this PR.

## Task 1: Shared Directive Parser Contract

**Files:**
- Create: `apps/packages/ui/src/utils/__fixtures__/character-emote-directives.json`
- Create: `apps/packages/ui/src/utils/character-emotes.ts`
- Create: `apps/packages/ui/src/utils/__tests__/character-emotes.test.ts`
- Create: `tldw_Server_API/app/core/Character_Chat/emote_directives.py`
- Create: `tldw_Server_API/tests/Character_Chat_NEW/unit/test_character_emote_directives.py`

- [ ] **Step 1: Write the shared fixture**

Create `apps/packages/ui/src/utils/__fixtures__/character-emote-directives.json`:

```json
[
  {
    "name": "strips directives and records offsets",
    "input": "Emote: annoyed\nWhat now?\n\nEmote: smug\nFine.",
    "clean_text": "What now?\n\nFine.",
    "events": [
      { "state": "annoyed", "at_char": 0 },
      { "state": "smug", "at_char": 11 }
    ]
  },
  {
    "name": "keeps inline prose visible",
    "input": "She says Emote: smug and smiles.",
    "clean_text": "She says Emote: smug and smiles.",
    "events": []
  },
  {
    "name": "keeps directives inside code fences visible",
    "input": "```text\nEmote: smug\n```\nDone.",
    "clean_text": "```text\nEmote: smug\n```\nDone.",
    "events": []
  },
  {
    "name": "strips invalid standalone directive",
    "input": "Emote: ../../bad\nVisible.",
    "clean_text": "Visible.",
    "events": []
  },
  {
    "name": "normalizes whitespace state",
    "input": "  Emote: Thinking Hard  \nVisible.",
    "clean_text": "Visible.",
    "events": [{ "state": "thinking-hard", "at_char": 0 }]
  },
  {
    "name": "drops duplicate consecutive state",
    "input": "Emote: smug\nEmote: smug\nVisible.",
    "clean_text": "Visible.",
    "events": [{ "state": "smug", "at_char": 0 }]
  }
]
```

- [ ] **Step 2: Write failing TypeScript parser tests**

Create `apps/packages/ui/src/utils/__tests__/character-emotes.test.ts`:

```ts
import { describe, expect, it } from "vitest"
import fixtures from "../__fixtures__/character-emote-directives.json"
import {
  EMOTE_EVENT_LIMIT,
  createCharacterEmoteStreamParser,
  isValidCharacterEmoteEvent,
  parseCharacterEmoteDirectives,
  normalizeCharacterEmoteState
} from "../character-emotes"

describe("character emote directives", () => {
  it.each(fixtures)("$name", (fixture) => {
    expect(parseCharacterEmoteDirectives(fixture.input)).toEqual({
      cleanText: fixture.clean_text,
      events: fixture.events
    })
  })

  it("uses exact slug normalization rules", () => {
    expect(normalizeCharacterEmoteState(" Thinking Hard ")).toBe("thinking-hard")
    expect(normalizeCharacterEmoteState("../../bad")).toBeNull()
    expect(normalizeCharacterEmoteState("a".repeat(40))).toBe("a".repeat(40))
    expect(normalizeCharacterEmoteState("a".repeat(41))).toBeNull()
  })

  it("caps accepted events but strips later directives", () => {
    const input = Array.from({ length: EMOTE_EVENT_LIMIT + 2 }, (_, index) => `Emote: mood-${index}\n`).join("") + "Done."
    const result = parseCharacterEmoteDirectives(input)
    expect(result.cleanText).toBe("Done.")
    expect(result.events).toHaveLength(EMOTE_EVENT_LIMIT)
  })

  it("does not leak split directives during streaming", () => {
    const parser = createCharacterEmoteStreamParser()
    expect(parser.push("Em")).toEqual({ visibleText: "", events: [] })
    expect(parser.push("ote: smug\nHello")).toEqual({
      visibleText: "Hello",
      events: [{ state: "smug", at_char: 0 }]
    })
    expect(parser.flush()).toEqual({ visibleText: "", events: [] })
  })

  it("streams long non-directive text before newline", () => {
    const parser = createCharacterEmoteStreamParser()
    const result = parser.push("This is just normal text without a newline")
    expect(result.visibleText).toContain("This is just normal text")
  })

  it("validates metadata event shape", () => {
    expect(isValidCharacterEmoteEvent({ state: "smug", at_char: 0 })).toBe(true)
    expect(isValidCharacterEmoteEvent({ state: "../../bad", at_char: 0 })).toBe(false)
    expect(isValidCharacterEmoteEvent({ state: "smug", at_char: -1 })).toBe(false)
  })
})
```

- [ ] **Step 3: Run TypeScript parser tests to verify failure**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/utils/__tests__/character-emotes.test.ts
```

Expected: FAIL because `character-emotes.ts` does not exist.

- [ ] **Step 4: Implement frontend parser**

Create `apps/packages/ui/src/utils/character-emotes.ts` with these public exports:

```ts
export const EMOTE_EVENT_LIMIT = 5
export const CHARACTER_EMOTE_STATE_PATTERN = /^[a-z0-9][a-z0-9_-]{0,39}$/

export type CharacterEmoteEvent = {
  state: string
  at_char: number
}

export type CharacterEmoteParseResult = {
  cleanText: string
  events: CharacterEmoteEvent[]
}

export const normalizeCharacterEmoteState = (value: unknown): string | null => {
  if (typeof value !== "string") return null
  const normalized = value.trim().toLowerCase().replace(/\s+/g, "-")
  return CHARACTER_EMOTE_STATE_PATTERN.test(normalized) ? normalized : null
}
```

Implementation rules:

- Parse only standalone directive lines after trimming leading/trailing whitespace.
- Strip invalid standalone directive lines.
- Ignore directives inside fenced code blocks and keep them visible.
- Track `at_char` as clean-text length before the directive.
- Drop duplicate consecutive accepted states.
- Cap accepted events at `EMOTE_EVENT_LIMIT`, but keep stripping later directives.
- `createCharacterEmoteStreamParser()` should withhold only possible directive/fence prefixes and flush final unterminated text.

- [ ] **Step 5: Run TypeScript parser tests to verify pass**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/utils/__tests__/character-emotes.test.ts
```

Expected: PASS.

- [ ] **Step 6: Write failing Python parser tests**

Create `tldw_Server_API/tests/Character_Chat_NEW/unit/test_character_emote_directives.py`:

```python
import json
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Character_Chat.emote_directives import (
    EMOTE_EVENT_LIMIT,
    parse_character_emote_directives,
    resolve_character_emote_completion,
    validate_emote_events,
)


FIXTURE_PATH = (
    Path(__file__).resolve().parents[4]
    / "apps/packages/ui/src/utils/__fixtures__/character-emote-directives.json"
)


@pytest.mark.unit
def test_shared_character_emote_fixtures() -> None:
    fixtures = json.loads(FIXTURE_PATH.read_text())
    for fixture in fixtures:
        result = parse_character_emote_directives(fixture["input"])
        assert result.clean_text == fixture["clean_text"], fixture["name"]
        assert [event.model_dump() for event in result.events] == fixture["events"], fixture["name"]


@pytest.mark.unit
def test_validate_emote_events_rejects_malformed_values() -> None:
    assert validate_emote_events([{"state": "smug", "at_char": 0}])[0].state == "smug"
    with pytest.raises(ValueError):
        validate_emote_events([{"state": "../../bad", "at_char": 0}])
    with pytest.raises(ValueError):
        validate_emote_events([{"state": "smug", "at_char": -1}])
    with pytest.raises(ValueError):
        validate_emote_events(
            [{"state": f"mood-{index}", "at_char": index} for index in range(EMOTE_EVENT_LIMIT + 1)]
        )
```

- [ ] **Step 7: Run Python parser tests to verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Character_Chat_NEW/unit/test_character_emote_directives.py -q
```

Expected: FAIL because `emote_directives.py` does not exist.

- [ ] **Step 8: Implement backend parser**

Create `tldw_Server_API/app/core/Character_Chat/emote_directives.py`:

```python
from __future__ import annotations

import re

from pydantic import BaseModel, Field

EMOTE_EVENT_LIMIT = 5
CHARACTER_EMOTE_STATE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")


class CharacterEmoteEvent(BaseModel):
    state: str = Field(..., min_length=1, max_length=40)
    at_char: int = Field(..., ge=0)


class CharacterEmoteParseResult(BaseModel):
    clean_text: str
    events: list[CharacterEmoteEvent]


class CharacterEmoteCompletionResult(BaseModel):
    clean_text: str
    mood_label: str | None
    mood_confidence: float | None
    mood_topic: str | None
    emote_events: list[CharacterEmoteEvent]
```

Implementation rules mirror TypeScript exactly. Keep this parser final-text only; streaming remains frontend-only.

- [ ] **Step 9: Run parser tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/utils/__tests__/character-emotes.test.ts
cd ../../..
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Character_Chat_NEW/unit/test_character_emote_directives.py -q
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add apps/packages/ui/src/utils/__fixtures__/character-emote-directives.json \
  apps/packages/ui/src/utils/character-emotes.ts \
  apps/packages/ui/src/utils/__tests__/character-emotes.test.ts \
  tldw_Server_API/app/core/Character_Chat/emote_directives.py \
  tldw_Server_API/tests/Character_Chat_NEW/unit/test_character_emote_directives.py
git commit -m "feat: add character emote directive parser"
```

## Task 2: Custom Emote Image Resolution

**Files:**
- Modify: `apps/packages/ui/src/utils/character-mood.ts`
- Modify: `apps/packages/ui/src/utils/__tests__/character-mood.test.ts`

- [ ] **Step 1: Add failing image map tests**

Extend `apps/packages/ui/src/utils/__tests__/character-mood.test.ts`:

```ts
it("resolves custom emote image states without expanding classifier labels", () => {
  const extensions = {
    tldw: {
      mood_images: {
        smug: TINY_PNG_BASE64,
        "thinking-hard": TINY_PNG_BASE64,
        "../../bad": TINY_PNG_BASE64
      }
    }
  }

  expect(resolveCharacterMoodImageUrl({ extensions }, "smug")).toMatch(/^data:image\/png;base64,/)
  expect(resolveCharacterMoodImageUrl({ extensions }, "thinking hard")).toMatch(/^data:image\/png;base64,/)
  expect(resolveCharacterMoodImageUrl({ extensions }, "../../bad")).toBe("")
  expect(normalizeCharacterMoodLabel("smug")).toBeNull()
})
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/utils/__tests__/character-mood.test.ts
```

Expected: FAIL because custom `smug` images are ignored.

- [ ] **Step 3: Broaden image resolver only**

Modify `apps/packages/ui/src/utils/character-mood.ts`:

- Import or duplicate only the safe slug normalizer from `character-emotes.ts`.
- Change the mood image map type to `Record<string, string>` for image storage.
- In `getCharacterMoodImagesFromExtensions()`, normalize keys with `normalizeCharacterEmoteState(rawMood)` instead of `normalizeCharacterMoodLabel(rawMood)`.
- In `mergeCharacterMoodImagesIntoExtensions()`, normalize keys the same way.
- In `resolveCharacterMoodImageUrl()`, normalize `moodLabel` with `normalizeCharacterEmoteState(moodLabel)`.
- Keep `normalizeCharacterMoodLabel()` and `detectCharacterMood()` unchanged for built-in classifier labels.

- [ ] **Step 4: Run tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/utils/__tests__/character-mood.test.ts src/utils/__tests__/character-emotes.test.ts
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/utils/character-mood.ts apps/packages/ui/src/utils/__tests__/character-mood.test.ts
git commit -m "feat: resolve custom character emote images"
```

## Task 3: Backend Persist And Non-Streaming Sanitization

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py`
- Modify: `tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_chat_stream_and_persist.py`

- [ ] **Step 1: Add failing stream persist validation test**

Extend `test_character_chat_stream_and_persist.py`:

```python
def test_persist_streamed_message_stores_emote_events(test_client: TestClient, auth_headers) -> None:
    _, chat_id = _create_character_and_chat(test_client, auth_headers)

    response = test_client.post(
        f"/api/v1/chats/{chat_id}/completions/persist",
        json={
            "assistant_content": "What now?\nFine.",
            "mood_label": "smug",
            "emote_events": [
                {"state": "annoyed", "at_char": 0},
                {"state": "smug", "at_char": 10},
            ],
        },
        headers=auth_headers,
    )

    assert response.status_code == 200
    assistant_message_id = response.json()["assistant_message_id"]
    message_resp = test_client.get(
        f"/api/v1/messages/{assistant_message_id}",
        params={"include_metadata": "true"},
        headers=auth_headers,
    )
    assert message_resp.status_code == 200
    metadata_extra = message_resp.json()["metadata_extra"]
    assert metadata_extra["mood_label"] == "smug"
    assert metadata_extra["emote_events"] == [
        {"state": "annoyed", "at_char": 0},
        {"state": "smug", "at_char": 10},
    ]
```

Add a rejection case:

```python
def test_persist_streamed_message_rejects_invalid_emote_events(test_client: TestClient, auth_headers) -> None:
    _, chat_id = _create_character_and_chat(test_client, auth_headers)

    response = test_client.post(
        f"/api/v1/chats/{chat_id}/completions/persist",
        json={
            "assistant_content": "bad metadata",
            "emote_events": [{"state": "../../bad", "at_char": 0}],
        },
        headers=auth_headers,
    )

    assert response.status_code == 422
```

- [ ] **Step 2: Add failing non-streaming sanitization helper test**

Extend `tldw_Server_API/tests/Character_Chat_NEW/unit/test_character_emote_directives.py`:

```python
@pytest.mark.unit
def test_resolve_completion_emotes_prefers_last_explicit_event() -> None:
    result = resolve_character_emote_completion(
        "Emote: annoyed\nWhat now?\nEmote: smug\nFine.",
        fallback_mood_label="happy",
        fallback_mood_confidence=0.9,
        fallback_mood_topic="fallback",
    )

    assert result.clean_text == "What now?\nFine."
    assert result.mood_label == "smug"
    assert result.mood_confidence is None
    assert result.mood_topic is None
    assert [event.model_dump() for event in result.emote_events] == [
        {"state": "annoyed", "at_char": 0},
        {"state": "smug", "at_char": 10},
    ]
```

This helper is the non-streaming endpoint contract: `/complete-v2` must call it
before returning `assistant_content` or storing assistant metadata.

- [ ] **Step 3: Run backend tests to verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Character_Chat_NEW/unit/test_character_emote_directives.py \
  tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_chat_stream_and_persist.py \
  -q
```

Expected: FAIL because schema/endpoint do not support `emote_events`.

- [ ] **Step 4: Add schema model**

Modify `tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py`:

```python
from tldw_Server_API.app.core.Character_Chat.emote_directives import CharacterEmoteEvent
```

Add to `CharacterChatStreamPersistRequest`:

```python
emote_events: Optional[list[CharacterEmoteEvent]] = Field(
    None,
    max_length=5,
    description="Optional sanitized character emote events for this assistant message.",
)
```

If Pydantic does not enforce list `max_length` as expected in this project version, add a validator that calls `validate_emote_events()`.

- [ ] **Step 5: Store emote metadata in stream persist endpoint**

Modify `_build_stream_persist_metadata_extra()` in `character_chat_sessions.py`:

- Add parameter `emote_events: list[CharacterEmoteEvent] | None`.
- When present, store `[event.model_dump() for event in emote_events]` under `metadata_extra["emote_events"]`.
- Pass `body.emote_events` in both new persist and existing idempotent side-effect paths.

- [ ] **Step 6: Sanitize backend non-streaming completion**

In `complete_character_chat_v2()` near the existing `assistant_text` and `resolved_mood_label` block:

```python
resolved_emotes = resolve_character_emote_completion(
    assistant_text,
    fallback_mood_label=resolved_mood_label,
    fallback_mood_confidence=resolved_mood_confidence,
    fallback_mood_topic=resolved_mood_topic,
)
assistant_text = resolved_emotes.clean_text
resolved_mood_label = resolved_emotes.mood_label
resolved_mood_confidence = resolved_emotes.mood_confidence
resolved_mood_topic = resolved_emotes.mood_topic
```

When building `metadata_extra`, include `resolved_emotes.emote_events` if any exist.

When returning `CharacterChatCompletionV2Response`, return sanitized `assistant_content` and `mood_label` equal to the last accepted event.

- [ ] **Step 7: Add prompt instruction**

In the same endpoint, after `_build_system_prompt_for_preset()` returns `sys_text`, append a short instruction only for character context prompts:

```text
When the character expression should change, emit a standalone line exactly like `Emote: <state>`. Prefer these available states: <states>. Do not emit an emote after every sentence.
```

Derive `<states>` from the active character's mood image extension keys using the backend parser/normalizer. If none exist, omit the "Prefer..." sentence.

- [ ] **Step 8: Run backend tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Character_Chat_NEW/unit/test_character_emote_directives.py \
  tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_chat_stream_and_persist.py \
  -q
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add tldw_Server_API/app/core/Character_Chat/emote_directives.py \
  tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py \
  tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py \
  tldw_Server_API/tests/Character_Chat_NEW/unit/test_character_emote_directives.py \
  tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_chat_stream_and_persist.py
git commit -m "feat: persist character emote metadata"
```

## Task 4: Frontend Character Stream Integration

**Files:**
- Modify: `apps/packages/ui/src/store/option/types.ts`
- Modify: `apps/packages/ui/src/hooks/chat/useChatActions.ts`
- Modify: `apps/packages/ui/src/hooks/chat/useCharacterChatMode.ts`
- Modify: `apps/packages/ui/src/hooks/useMessage.tsx`
- Modify: `apps/packages/ui/src/hooks/chat/__tests__/useChatActions.character.integration.test.tsx`
- Modify: `apps/packages/ui/src/hooks/chat/__tests__/useCharacterChatMode.contract.test.ts`

- [ ] **Step 1: Add metadata type**

Modify `MessageMetadataExtra` in `apps/packages/ui/src/store/option/types.ts`:

```ts
import type { CharacterEmoteEvent } from "@/utils/character-emotes"

export type MessageMetadataExtra = Record<string, unknown> & {
  dynamic_ui?: DynamicUIEnvelope
  dynamic_ui_action?: DynamicUIActionUserMetadata
  emote_events?: CharacterEmoteEvent[]
}
```

- [ ] **Step 2: Add failing primary hook integration test**

In `useChatActions.character.integration.test.tsx`, configure `streamCharacterChatCompletionMock` to emit split directive chunks:

```ts
streamCharacterChatCompletionMock.mockImplementationOnce(async function* () {
  yield "Em"
  yield "ote: smug\n"
  yield "Hello "
  yield "there.\n"
  yield "Emote: annoyed\n"
  yield "Fine."
})
```

Assert:

```ts
expect(persistCharacterCompletionMock).toHaveBeenCalledWith(
  expect.anything(),
  expect.objectContaining({
    assistant_content: "Hello there.\nFine.",
    mood_label: "annoyed",
    emote_events: [
      { state: "smug", at_char: 0 },
      { state: "annoyed", at_char: 13 }
    ]
  }),
  expect.anything()
)
expect(JSON.stringify(options.setMessages.mock.calls)).not.toContain("Emote:")

const initialAssistantMessage = {
  id: "generated-assistant-id",
  isBot: true,
  name: "Ashley",
  message: "",
  sources: []
}
const messagesAfterUpdates = options.setMessages.mock.calls.reduce(
  (messages, [updater]) =>
    typeof updater === "function" ? updater(messages) : updater,
  [initialAssistantMessage]
)
const moodLabelUpdates = messagesAfterUpdates.map((message) => message?.moodLabel)
expect(moodLabelUpdates).toContain("smug")
expect(moodLabelUpdates).toContain("annoyed")
```

Use the generated assistant message id from the existing test harness instead of
the sample id above. The assertion must prove `moodLabel` changes during
chunk processing, before final persistence.

- [ ] **Step 3: Run primary hook test to verify failure**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/hooks/chat/__tests__/useChatActions.character.integration.test.tsx
```

Expected: FAIL because raw directives are persisted.

- [ ] **Step 4: Implement stream parser wiring in `useChatActions.ts`**

In the character stream branch:

- Create `const emoteParser = createCharacterEmoteStreamParser()`.
- Keep `fullText` and `contentToSave` sanitized.
- Feed only text token deltas through the parser.
- For each accepted event:
  - append to local `emoteEvents`
  - update the assistant message `moodLabel` to `event.state`
  - add/merge `metadataExtra.emote_events`
- Schedule streaming updates with sanitized text only.
- On stream completion, call `emoteParser.flush()` before final persistence.
- If `emoteEvents.length > 0`, set `resolvedMoodLabel = emoteEvents.at(-1)?.state` and skip `detectCharacterMood()`.
- Include `emote_events` in `persistPayload` and `metadataExtra`.

Do not log full assistant content on parser errors.

- [ ] **Step 5: Add failing parity test for `useCharacterChatMode.ts`**

Extend `apps/packages/ui/src/hooks/chat/__tests__/useCharacterChatMode.contract.test.ts` with the same split directive stream and persistence assertions.

- [ ] **Step 6: Implement parity in `useCharacterChatMode.ts`**

Apply the same parser flow as `useChatActions.ts`. If duplication becomes too large, extract a tiny helper into:

- `apps/packages/ui/src/hooks/chat/character-emote-stream.ts`

Keep helper scope limited to:

```ts
export const applyCharacterEmoteEventsToMessage = (...)
export const resolveExplicitOrDetectedMood = (...)
```

Do not create a general event bus or runtime.

- [ ] **Step 7: Update `useMessage.tsx` legacy path**

Run:

```bash
rg -n "streamCharacterChatCompletion|persistCharacterCompletion|detectCharacterMood" apps/packages/ui/src/hooks/useMessage.tsx
```

Apply the same parser/persist helper in each character-stream branch that persists character completions. If no existing test harness can drive `useMessage.tsx`, rely on the shared helper tests plus the two hook integration tests, and record the legacy-path coverage decision in `TASK-12163` implementation notes.

- [ ] **Step 8: Run frontend hook tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run \
  src/utils/__tests__/character-emotes.test.ts \
  src/hooks/chat/__tests__/useChatActions.character.integration.test.tsx \
  src/hooks/chat/__tests__/useCharacterChatMode.contract.test.ts
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add apps/packages/ui/src/store/option/types.ts \
  apps/packages/ui/src/hooks/chat/useChatActions.ts \
  apps/packages/ui/src/hooks/chat/useCharacterChatMode.ts \
  apps/packages/ui/src/hooks/useMessage.tsx \
  apps/packages/ui/src/hooks/chat/__tests__/useChatActions.character.integration.test.tsx \
  apps/packages/ui/src/hooks/chat/__tests__/useCharacterChatMode.contract.test.ts
git commit -m "feat: stream character emote directives in webui"
```

If `useMessage.tsx` is not touched, omit it from `git add`.

## Task 5: UI Portrait Behavior And History Reload

**Files:**
- Modify: `apps/packages/ui/src/hooks/chat/useServerChatLoader.ts`
- Modify: `apps/packages/ui/src/db/dexie/helpers.ts`
- Modify: `apps/packages/ui/src/components/Common/Playground/Message.tsx`
- Modify: `apps/packages/ui/src/hooks/__tests__/useServerChatLoader.test.ts`
- Modify: `apps/packages/ui/src/utils/__tests__/character-mood.test.ts`
- Modify: `apps/packages/ui/src/components/Common/Playground/__tests__/Message.routing-fallback.integration.test.tsx`

- [ ] **Step 1: Add failing history reload test**

Extend `apps/packages/ui/src/hooks/__tests__/useServerChatLoader.test.ts` with a message containing:

```ts
metadata_extra: {
  mood_label: "smug",
  emote_events: [{ state: "smug", at_char: 0 }]
}
```

Assert loaded message has:

```ts
expect(message.moodLabel).toBe("smug")
expect(message.metadataExtra?.emote_events).toEqual([{ state: "smug", at_char: 0 }])
```

- [ ] **Step 2: Add Dexie local replay test if local helper drops mood fields**

If `formatToMessage()` in `apps/packages/ui/src/db/dexie/helpers.ts` still does not promote `metadataExtra.mood_label`, add a small test near existing Dexie helper tests or create:

- `apps/packages/ui/src/db/dexie/__tests__/helpers.character-emotes.test.ts`

Assert local messages with `metadataExtra.mood_label` reload with `moodLabel`.

- [ ] **Step 3: Implement history/local metadata normalization**

Modify:

- `useServerChatLoader.ts`: preserve `emote_events` in `metadataExtra` and set `moodLabel` from string `metadataExtra.mood_label`, including custom slugs.
- `db/dexie/helpers.ts`: when converting local message variants, set `moodLabel` from `metadataExtra.mood_label` if the top-level field is absent.

- [ ] **Step 4: Fix portrait resolver behavior**

Modify `Message.tsx` so explicit `props.moodLabel` uses the custom emote slug normalizer for `resolvedMoodLabel`, while `detectCharacterMood()` remains the fallback and still returns built-in labels. A custom explicit `smug` label must be able to reach `resolveCharacterMoodImageUrl()`.

- [ ] **Step 5: Add UI behavior test**

Extend `Message.routing-fallback.integration.test.tsx` to render a bot message with:

- `moodLabel="smug"`
- `characterIdentity.extensions.tldw.mood_images.smug`

Assert the rendered portrait image uses the `smug` data URL.

- [ ] **Step 6: Run UI/history tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run \
  src/hooks/__tests__/useServerChatLoader.test.ts \
  src/utils/__tests__/character-mood.test.ts \
  src/components/Common/Playground/__tests__/Message.routing-fallback.integration.test.tsx
```

The Message test is required for this task.

- [ ] **Step 7: Commit**

```bash
git add apps/packages/ui/src/hooks/chat/useServerChatLoader.ts \
  apps/packages/ui/src/db/dexie/helpers.ts \
  apps/packages/ui/src/components/Common/Playground/Message.tsx \
  apps/packages/ui/src/hooks/__tests__/useServerChatLoader.test.ts \
  apps/packages/ui/src/utils/__tests__/character-mood.test.ts \
  apps/packages/ui/src/components/Common/Playground/__tests__/Message.routing-fallback.integration.test.tsx
git commit -m "feat: restore explicit character emotes from history"
```

Omit unchanged optional files from `git add`.

## Task 6: Final Verification And Backlog Update

**Files:**
- Modify: `backlog/tasks/task-12163 - Add-explicit-streaming-emote-directives-for-character-chat-portraits.md`

- [ ] **Step 1: Run targeted frontend tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run \
  src/utils/__tests__/character-emotes.test.ts \
  src/utils/__tests__/character-mood.test.ts \
  src/hooks/chat/__tests__/useChatActions.character.integration.test.tsx \
  src/hooks/chat/__tests__/useCharacterChatMode.contract.test.ts \
  src/hooks/__tests__/useServerChatLoader.test.ts
```

Expected: PASS.

- [ ] **Step 2: Run targeted backend tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Character_Chat_NEW/unit/test_character_emote_directives.py \
  tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_chat_stream_and_persist.py \
  -q
```

Expected: PASS.

- [ ] **Step 3: Run typecheck for touched frontend package if time allows**

Run:

```bash
cd apps/tldw-frontend
bun run typecheck
```

Expected: PASS. If repo-wide baseline fails outside touched files, record the failure and evidence in `TASK-12163`.

- [ ] **Step 4: Run Bandit on touched backend scope**

Run:

```bash
source .venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/core/Character_Chat/emote_directives.py \
  tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py \
  tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py \
  -f json -o /tmp/bandit_character_emotes.json
```

Expected: no new findings in touched code. If Bandit reports unrelated existing findings in the huge endpoint file, record the exact finding IDs and why they are unrelated.

- [ ] **Step 5: Update Backlog task**

Use Backlog MCP or CLI:

```bash
backlog task edit TASK-12163 \
  --notes "Implemented streaming character emote directives. Verification: <commands/results>. Bandit: /tmp/bandit_character_emotes.json." \
  --plain
```

Check acceptance criteria as completed only after the matching tests pass.

- [ ] **Step 6: Final commit**

```bash
git add "backlog/tasks/task-12163 - Add-explicit-streaming-emote-directives-for-character-chat-portraits.md"
git commit -m "chore: record character emote verification"
```

## Out Of Scope

- No Persona Visual runtime unification.
- No `set_emote` tool.
- No character editor mood-image manager.
- No historic beat replay.
- No fuzzy asset matching.

## Notes For Implementers

- Do not let raw `Emote:` lines reach rendered or persisted assistant content.
- Do not let explicit directives be overwritten by `detectCharacterMood()`.
- Do not expand the classifier taxonomy; custom slugs are for explicit emotes and image lookup only.
- Keep `emote_events` compact and metadata-only.
- The current worktree may have unrelated dirty files. Stage only files touched by this plan.
