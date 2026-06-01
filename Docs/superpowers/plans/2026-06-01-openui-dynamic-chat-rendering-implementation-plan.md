# OpenUI Dynamic Chat Rendering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add literal OpenUI support as the first renderer in a shared dynamic UI chat layer, initially enabled for completed `/chat` responses with safe source fallback elsewhere.

**Architecture:** Add renderer-neutral dynamic UI metadata and validation utilities in `apps/packages/ui`, preserve that metadata through local/server chat persistence, render metadata-tagged assistant messages through a lazy registry, and wire `/chat` OpenUI request mode into the existing chat pipeline. Keep OpenUI runtime enablement behind a feasibility gate and feature/capability checks.

**Tech Stack:** React 18, TypeScript, Vitest, Testing Library, Ant Design, lucide-react, Dexie, tldw frontend monorepo under `apps/`, OpenUI runtime package selected during Stage 0.

**Spec:** `Docs/superpowers/specs/2026-06-01-openui-dynamic-chat-rendering-design.md`
**Backlog:** TASK-493

---

## File Structure

Create:

- `Docs/superpowers/reviews/openui-runtime-feasibility-2026-06-01.md` - records package/runtime/CSP/license/bundle findings before enabling OpenUI.
- `apps/packages/ui/src/types/dynamic-ui.ts` - shared dynamic UI envelope, action, surface, and request types.
- `apps/packages/ui/src/utils/dynamic-ui.ts` - envelope normalization, source preflight, feature/capability helpers, and action serialization.
- `apps/packages/ui/src/utils/__tests__/dynamic-ui.test.ts` - unit coverage for dynamic UI validation and action formatting.
- `apps/packages/ui/src/components/Common/DynamicUI/DynamicMessageRenderer.tsx` - shared dynamic message renderer with lazy registry lookup and fallback.
- `apps/packages/ui/src/components/Common/DynamicUI/DynamicUIErrorBoundary.tsx` - catches renderer component crashes and shows source fallback.
- `apps/packages/ui/src/components/Common/DynamicUI/DynamicUISourceFallback.tsx` - readable source/error fallback.
- `apps/packages/ui/src/components/Common/DynamicUI/registry.ts` - renderer registry and surface capability checks.
- `apps/packages/ui/src/components/Common/DynamicUI/renderers/OpenUIRenderer.tsx` - placeholder in Task 3, then real OpenUI adapter in Task 4 after Stage 0 findings.
- `apps/packages/ui/src/components/Common/DynamicUI/__tests__/DynamicMessageRenderer.test.tsx` - renderer/fallback behavior.
- `apps/packages/ui/src/hooks/chat/useDynamicUIActionBridge.ts` - validates renderer actions and sends normal follow-up chat turns.
- `apps/packages/ui/src/hooks/chat/__tests__/useDynamicUIActionBridge.test.tsx` - action bridge coverage.
- `apps/packages/ui/src/hooks/chat-modes/__tests__/chatModePipeline.dynamic-ui.test.ts` - OpenUI request metadata and preflight behavior.
- `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundForm.openui-mode.test.tsx` - `/chat` control coverage.
- `apps/tldw-frontend/e2e/smoke/chat-openui-dynamic-ui.spec.ts` - browser smoke for `/chat` rendering/fallback after implementation.

Modify:

- `apps/bun.lock` - workspace dependency lock update after Stage 0 passes.
- `apps/tldw-frontend/package.json` - add selected OpenUI runtime dependency if Stage 0 passes.
- `apps/extension/package.json` - add selected OpenUI runtime dependency only if the shared lazy import requires extension build-time resolution.
- `apps/packages/ui/package.json` - add selected OpenUI runtime as peer dependency if shared UI imports it.
- `apps/packages/ui/src/store/option/types.ts` - add typed dynamic UI fields to `Message`, `MessageVariant`, and save payloads where appropriate.
- `apps/packages/ui/src/db/dexie/types.ts` - persist `metadataExtra` on local messages.
- `apps/packages/ui/src/db/dexie/helpers.ts` - save/load `metadataExtra` and variant metadata.
- `apps/packages/ui/src/hooks/chat-helper/index.ts` - pass user/assistant metadata into local `saveMessage`.
- `apps/packages/ui/src/types/chat-modes.ts` - add `dynamicUIRequest`, `userMetadataExtra`, and `assistantMetadataExtra`.
- `apps/packages/ui/src/hooks/chat-modes/chatModePipeline.ts` - inject dynamic UI prompt, preflight completed output, update message metadata, and save metadata.
- `apps/packages/ui/src/hooks/chat-modes/normalChatMode.ts` - accept dynamic request params if needed by prompt assembly.
- `apps/packages/ui/src/hooks/chat/useChatActions.ts` - thread request/action metadata through send and server mirroring.
- `apps/packages/ui/src/hooks/chat/useServerChatLoader.ts` - validate hydrated `metadata_extra.dynamic_ui`.
- `apps/packages/ui/src/components/Common/Playground/Message.tsx` - pass dynamic UI metadata and action callback into message content.
- `apps/packages/ui/src/components/Common/Playground/MessageContent.tsx` - render dynamic UI before Markdown when metadata is valid.
- `apps/packages/ui/src/components/Option/Playground/PlaygroundChat.tsx` - provide `/chat` dynamic action bridge and surface ID.
- `apps/packages/ui/src/components/Sidepanel/Chat/body.tsx` - pass sidepanel surface ID for source fallback.
- `apps/packages/ui/src/components/Option/ChatWorkspace/WorkspaceChatPanel.tsx` - pass workspace surface ID for source fallback.
- `apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx` - add transient OpenUI request-mode control.
- `apps/packages/ui/src/components/Option/Playground/hooks/usePlaygroundSubmit.ts` - dispatch OpenUI request mode for the next send.
- `apps/packages/ui/src/public/_locales/en/playground.json` - labels/tooltips/errors for OpenUI mode.

---

## Task 0: Runtime Feasibility Gate

**Files:**
- Create: `Docs/superpowers/reviews/openui-runtime-feasibility-2026-06-01.md`
- Modify only after pass: `apps/tldw-frontend/package.json`, `apps/extension/package.json`, `apps/packages/ui/package.json`, `apps/bun.lock`

- [ ] **Step 1: Inspect OpenUI package metadata without changing repo files**

Run from repo root:

```bash
npm view @openuidev/react-lang version license peerDependencies dependencies dist.unpackedSize --json
npm view @openuidev/react-ui version license peerDependencies dependencies dist.unpackedSize --json
npm view @openuidev/react-headless version license peerDependencies dependencies dist.unpackedSize --json
```

Expected: JSON output for each package. If any package is missing, license-incompatible, or requires unsupported React versions, stop and update the feasibility review with the blocker.

- [ ] **Step 2: Pack the candidate runtime into `/tmp` and inspect for dynamic evaluation**

Run:

```bash
mkdir -p /tmp/tldw-openui-audit
cd /tmp/tldw-openui-audit
npm pack @openuidev/react-lang @openuidev/react-ui @openuidev/react-headless
tar -xf openuidev-react-lang-*.tgz
tar -xf openuidev-react-ui-*.tgz
tar -xf openuidev-react-headless-*.tgz
rg -n "eval\\(|new Function|Function\\(|dangerouslySetInnerHTML|innerHTML|script" .
```

Expected: no unsafe dynamic evaluation required by the render path. If matches exist, inspect whether they are dead/test/build-only code. Record findings.

- [ ] **Step 3: Record feasibility findings**

Create `Docs/superpowers/reviews/openui-runtime-feasibility-2026-06-01.md`:

```markdown
# OpenUI Runtime Feasibility Review

Date: 2026-06-01
Backlog: TASK-493

## Packages Checked

- @openuidev/react-lang: <version>, <license>, <unpacked size>
- @openuidev/react-ui: <version>, <license>, <unpacked size>
- @openuidev/react-headless: <version>, <license>, <unpacked size>

## Findings

- React peer compatibility:
- Dynamic evaluation / CSP:
- Bundle impact:
- Component allowlist:
- Extension build risk:

## Decision

PASS | FAIL

## Notes
```

Expected: explicit `PASS` before continuing to OpenUI dependency work. If `FAIL`, stop implementation and leave subsequent tasks unchecked.

- [ ] **Step 4: Add dependencies only if feasibility passes**

Edit package manifests based on the selected packages:

```json
"@openuidev/react-lang": "<selected-version>",
"@openuidev/react-ui": "<selected-version>",
"@openuidev/react-headless": "<selected-version>"
```

Rules:

- Add runtime dependencies to `apps/tldw-frontend/package.json`.
- Add the same dependencies to `apps/extension/package.json` only if the shared lazy import must resolve during extension build.
- Add peer dependency ranges to `apps/packages/ui/package.json` if shared UI imports the packages.
- Do not commit generated vendor bundles.

- [ ] **Step 5: Update the workspace lockfile**

Run:

```bash
cd apps
bun install
```

Expected: `apps/bun.lock` updates, no install errors.

- [ ] **Step 6: Verify dependency/build baseline**

Run:

```bash
cd apps
bunx vitest run packages/ui/src/utils/__tests__/dynamic-ui.test.ts
```

Expected before Task 1: FAIL because the test file does not exist. This confirms the next task starts red.

- [ ] **Step 7: Commit**

```bash
git add Docs/superpowers/reviews/openui-runtime-feasibility-2026-06-01.md apps/bun.lock apps/tldw-frontend/package.json apps/extension/package.json apps/packages/ui/package.json
git commit -m "chore: verify OpenUI runtime feasibility"
```

Only add package files that actually changed.

---

## Task 1: Dynamic UI Types And Validation Utilities

**Files:**
- Create: `apps/packages/ui/src/types/dynamic-ui.ts`
- Create: `apps/packages/ui/src/utils/dynamic-ui.ts`
- Test: `apps/packages/ui/src/utils/__tests__/dynamic-ui.test.ts`

- [ ] **Step 1: Write failing utility tests**

Create `apps/packages/ui/src/utils/__tests__/dynamic-ui.test.ts`:

```ts
import { describe, expect, it } from "vitest"
import {
  buildDynamicUIEnvelope,
  formatDynamicUIActionUserMessage,
  normalizeDynamicUIActionPayload,
  normalizeDynamicUIEnvelope,
  preflightOpenUISource,
  shouldBlockDynamicUIActionValues
} from "../dynamic-ui"

describe("dynamic UI utilities", () => {
  it("normalizes valid OpenUI envelopes", () => {
    const envelope = normalizeDynamicUIEnvelope({
      renderer: "openui",
      version: "v1",
      source: "root = <Card />",
      state: { count: 1 },
      capabilities: ["forms"]
    })

    expect(envelope).toEqual({
      renderer: "openui",
      version: "v1",
      source: "root = <Card />",
      state: { count: 1 },
      capabilities: ["forms"]
    })
  })

  it("rejects unknown renderers and empty source", () => {
    expect(normalizeDynamicUIEnvelope({ renderer: "html", source: "<script />" })).toBeNull()
    expect(normalizeDynamicUIEnvelope({ renderer: "openui", source: "" })).toBeNull()
  })

  it("rejects unsupported dynamic UI contract versions", () => {
    expect(
      normalizeDynamicUIEnvelope({
        renderer: "openui",
        version: "v2",
        source: "root = <Card />"
      })
    ).toBeNull()
  })

  it("preflights only plausible completed OpenUI source", () => {
    expect(preflightOpenUISource("root = <Card><Text>Hello</Text></Card>").ok).toBe(true)
    expect(preflightOpenUISource("I cannot produce that UI.").ok).toBe(false)
  })

  it("builds envelopes only after source preflight", () => {
    expect(buildDynamicUIEnvelope("openui", "root = <Card />")).toMatchObject({
      renderer: "openui",
      source: "root = <Card />"
    })
    expect(buildDynamicUIEnvelope("openui", "plain refusal")).toBeNull()
  })

  it("normalizes action payloads and blocks sensitive-looking values", () => {
    const payload = normalizeDynamicUIActionPayload({
      renderer: "openui",
      sourceMessageId: "assistant-1",
      actionId: "profile-submit",
      actionType: "submit",
      values: { name: "Ada", password: "secret" }
    }, { currentMessageIds: new Set(["assistant-1"]) })

    expect(payload?.actionId).toBe("profile-submit")
    expect(shouldBlockDynamicUIActionValues(payload?.values)).toBe(true)
  })

  it("blocks nested sensitive-looking action values", () => {
    expect(
      shouldBlockDynamicUIActionValues({
        profile: { password: "secret" }
      })
    ).toBe(true)
    expect(
      shouldBlockDynamicUIActionValues([
        { settings: { authToken: "abc123" } }
      ])
    ).toBe(true)
  })

  it("rejects non-serializable action values without throwing", () => {
    const circular: Record<string, unknown> = {}
    circular.self = circular
    expect(
      normalizeDynamicUIActionPayload(
        {
          renderer: "openui",
          sourceMessageId: "assistant-1",
          actionId: "bad",
          actionType: "submit",
          values: circular
        },
        { currentMessageIds: new Set(["assistant-1"]) }
      )
    ).toBeNull()
  })

  it("rejects action values that are not strict JSON values", () => {
    expect(
      normalizeDynamicUIActionPayload(
        {
          renderer: "openui",
          sourceMessageId: "assistant-1",
          actionId: "bad",
          actionType: "submit",
          values: { callback: () => undefined }
        },
        { currentMessageIds: new Set(["assistant-1"]) }
      )
    ).toBeNull()

    expect(
      normalizeDynamicUIActionPayload(
        {
          renderer: "openui",
          sourceMessageId: "assistant-1",
          actionId: "bad",
          actionType: "submit",
          values: { nested: { missing: undefined, token: Symbol("secret") } }
        },
        { currentMessageIds: new Set(["assistant-1"]) }
      )
    ).toBeNull()
  })

  it("rejects Blob-like action values", () => {
    const blob =
      typeof Blob === "function"
        ? new Blob(["file contents"])
        : { [Symbol.toStringTag]: "Blob", size: 13 }

    expect(
      normalizeDynamicUIActionPayload(
        {
          renderer: "openui",
          sourceMessageId: "assistant-1",
          actionId: "bad",
          actionType: "submit",
          values: { upload: blob }
        },
        { currentMessageIds: new Set(["assistant-1"]) }
      )
    ).toBeNull()
  })

  it("rejects non-plain action value objects", () => {
    expect(
      normalizeDynamicUIActionPayload(
        {
          renderer: "openui",
          sourceMessageId: "assistant-1",
          actionId: "bad",
          actionType: "submit",
          values: { selected: new Map([["a", "b"]]) }
        },
        { currentMessageIds: new Set(["assistant-1"]) }
      )
    ).toBeNull()

    class CustomValue {
      answer = "yes"
    }

    expect(
      normalizeDynamicUIActionPayload(
        {
          renderer: "openui",
          sourceMessageId: "assistant-1",
          actionId: "bad",
          actionType: "submit",
          values: { custom: new CustomValue() }
        },
        { currentMessageIds: new Set(["assistant-1"]) }
      )
    ).toBeNull()
  })

  it("formats action payloads as visible user messages", () => {
    const text = formatDynamicUIActionUserMessage({
      renderer: "openui",
      sourceMessageId: "assistant-1",
      actionId: "survey",
      actionType: "submit",
      values: { answer: "yes" },
      submittedAt: "2026-06-01T00:00:00.000Z"
    })

    expect(text).toContain("OpenUI action: submit survey")
    expect(text).toContain("- answer: yes")
  })
})
```

- [ ] **Step 2: Run the test and verify it fails**

```bash
cd apps
bunx vitest run packages/ui/src/utils/__tests__/dynamic-ui.test.ts
```

Expected: FAIL because `../dynamic-ui` does not exist.

- [ ] **Step 3: Add shared types**

Create `apps/packages/ui/src/types/dynamic-ui.ts`:

```ts
export type DynamicUIRendererId = "openui"
export type DynamicUISurface = "web-chat" | "extension-sidepanel" | "workspace" | "artifact"
export type DynamicUIActionType = "submit"

export type DynamicUIEnvelope = {
  renderer: DynamicUIRendererId
  version: "v1"
  source: string
  state?: Record<string, unknown>
  capabilities?: string[]
}

export type DynamicUIRequest = {
  renderer: DynamicUIRendererId
}

export type DynamicUIActionPayload = {
  renderer: DynamicUIRendererId
  sourceMessageId: string
  actionId: string
  actionType: DynamicUIActionType
  values: Record<string, unknown>
}

export type DynamicUIActionUserMetadata = DynamicUIActionPayload & {
  submittedAt: string
}
```

- [ ] **Step 4: Add validation helpers**

Create `apps/packages/ui/src/utils/dynamic-ui.ts` with these exported functions:

```ts
import type {
  DynamicUIActionPayload,
  DynamicUIActionUserMetadata,
  DynamicUIEnvelope,
  DynamicUIRendererId
} from "@/types/dynamic-ui"

const SUPPORTED_RENDERERS = new Set<DynamicUIRendererId>(["openui"])
const SENSITIVE_KEY_PATTERN = /(password|token|secret|credential|api[_-]?key|auth)/i
const MAX_ACTION_STRING_LENGTH = 128
const MAX_ACTION_VALUES_BYTES = 16_384

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value)

const isPlainObject = (value: unknown): value is Record<string, unknown> => {
  if (!isRecord(value)) return false
  const prototype = Object.getPrototypeOf(value)
  return prototype === Object.prototype || prototype === null
}

const isBlobLike = (value: unknown): boolean => {
  if (!isRecord(value)) return false
  const tag = Object.prototype.toString.call(value)
  return tag === "[object Blob]" || tag === "[object File]"
}

const isStrictJSONValue = (value: unknown, depth = 0): boolean => {
  if (depth > 8) return false
  if (value == null) return true
  if (typeof value === "string" || typeof value === "boolean") return true
  if (typeof value === "number") return Number.isFinite(value)
  if (typeof value === "function" || typeof value === "symbol" || typeof value === "undefined") return false
  if (Array.isArray(value)) {
    return value.every((entry) => isStrictJSONValue(entry, depth + 1))
  }
  if (!isPlainObject(value) || isBlobLike(value)) return false
  return Object.values(value).every((entry) => isStrictJSONValue(entry, depth + 1))
}

export const preflightOpenUISource = (source: unknown): { ok: boolean; reason?: string } => {
  if (typeof source !== "string") return { ok: false, reason: "source_not_string" }
  const trimmed = source.trim()
  if (trimmed.length === 0) return { ok: false, reason: "empty_source" }
  if (!/root\s*=/.test(trimmed)) return { ok: false, reason: "missing_root_assignment" }
  if (!/<[A-Z][A-Za-z0-9.]*/.test(trimmed)) return { ok: false, reason: "missing_component_markup" }
  return { ok: true }
}

export const normalizeDynamicUIEnvelope = (value: unknown): DynamicUIEnvelope | null => {
  if (!isRecord(value)) return null
  if (value.renderer !== "openui") return null
  if (!SUPPORTED_RENDERERS.has(value.renderer)) return null
  if (value.version !== "v1") return null
  const preflight = preflightOpenUISource(value.source)
  if (!preflight.ok) return null
  return {
    renderer: "openui",
    version: "v1",
    source: String(value.source).trim(),
    state: isRecord(value.state) ? value.state : undefined,
    capabilities: Array.isArray(value.capabilities)
      ? value.capabilities.filter((entry): entry is string => typeof entry === "string")
      : undefined
  }
}

export const buildDynamicUIEnvelope = (
  renderer: DynamicUIRendererId,
  source: string
): DynamicUIEnvelope | null =>
  normalizeDynamicUIEnvelope({ renderer, version: "v1", source })

export const normalizeDynamicUIActionPayload = (
  value: unknown,
  options: { currentMessageIds: Set<string> }
): DynamicUIActionPayload | null => {
  if (!isRecord(value)) return null
  if (value.renderer !== "openui") return null
  if (value.actionType !== "submit") return null
  const sourceMessageId = typeof value.sourceMessageId === "string" ? value.sourceMessageId.trim() : ""
  const actionId = typeof value.actionId === "string" ? value.actionId.trim() : ""
  if (!sourceMessageId || !options.currentMessageIds.has(sourceMessageId)) return null
  if (!actionId || actionId.length > MAX_ACTION_STRING_LENGTH) return null
  if (!isRecord(value.values)) return null
  if (!isStrictJSONValue(value.values)) return null
  let serialized = ""
  try {
    serialized = JSON.stringify(value.values)
  } catch {
    return null
  }
  if (serialized.length > MAX_ACTION_VALUES_BYTES) return null
  return {
    renderer: "openui",
    sourceMessageId,
    actionId,
    actionType: "submit",
    values: value.values
  }
}

export const shouldBlockDynamicUIActionValues = (value: unknown, depth = 0): boolean => {
  if (depth > 8) return false
  if (Array.isArray(value)) {
    return value.some((entry) => shouldBlockDynamicUIActionValues(entry, depth + 1))
  }
  if (!isRecord(value)) return false
  return Object.entries(value).some(
    ([key, entry]) =>
      SENSITIVE_KEY_PATTERN.test(key) ||
      shouldBlockDynamicUIActionValues(entry, depth + 1)
  )
}

export const formatDynamicUIActionUserMessage = (
  payload: DynamicUIActionUserMetadata
): string => {
  const lines = [`OpenUI action: ${payload.actionType} ${payload.actionId}`, "", "Submitted values:"]
  for (const [key, value] of Object.entries(payload.values)) {
    lines.push(`- ${key}: ${typeof value === "string" ? value : JSON.stringify(value)}`)
  }
  return lines.join("\n")
}
```

- [ ] **Step 5: Run the test**

```bash
cd apps
bunx vitest run packages/ui/src/utils/__tests__/dynamic-ui.test.ts
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add apps/packages/ui/src/types/dynamic-ui.ts apps/packages/ui/src/utils/dynamic-ui.ts apps/packages/ui/src/utils/__tests__/dynamic-ui.test.ts
git commit -m "feat: add dynamic UI metadata utilities"
```

---

## Task 2: Metadata Persistence Plumbing

**Files:**
- Modify: `apps/packages/ui/src/store/option/types.ts`
- Modify: `apps/packages/ui/src/db/dexie/types.ts`
- Modify: `apps/packages/ui/src/db/dexie/helpers.ts`
- Modify: `apps/packages/ui/src/hooks/chat-helper/index.ts`
- Modify: `apps/packages/ui/src/types/chat-modes.ts`
- Modify: `apps/packages/ui/src/hooks/chat/useChatActions.ts`
- Modify: `apps/packages/ui/src/hooks/chat/useServerChatLoader.ts`
- Test: `apps/packages/ui/src/hooks/__tests__/useServerChatLoader.test.ts`
- Test: `apps/packages/ui/src/hooks/chat/__tests__/useChatActions.persist-mirror.guard.test.ts`

- [ ] **Step 1: Add failing loader test**

In `apps/packages/ui/src/hooks/__tests__/useServerChatLoader.test.ts`, add a test near existing `mapServerChatMessagesToPlaygroundMessages` coverage:

```ts
it("preserves valid dynamic UI metadata from server messages", () => {
  const [message] = mapServerChatMessagesToPlaygroundMessages({
    assistantName: "Assistant",
    characterId: null,
    serverMessages: [
      {
        id: "server-1",
        role: "assistant",
        content: "root = <Card />",
        created_at: "2026-06-01T00:00:00.000Z",
        metadata_extra: {
          dynamic_ui: {
            renderer: "openui",
            version: "v1",
            source: "root = <Card />"
          }
        }
      }
    ]
  })

  expect(message.metadataExtra?.dynamic_ui).toMatchObject({
    renderer: "openui",
    source: "root = <Card />"
  })
})
```

- [ ] **Step 2: Add failing save/mirror guard test**

In `apps/packages/ui/src/hooks/chat/__tests__/useChatActions.persist-mirror.guard.test.ts`, add coverage that `saveMessageOnSuccess` passes `assistantMetadataExtra` to `tldwClient.addChatMessage`:

```ts
expect(tldwClient.addChatMessage).toHaveBeenCalledWith(
  "chat-1",
  expect.objectContaining({
    role: "assistant",
    metadata_extra: expect.objectContaining({
      dynamic_ui: expect.objectContaining({ renderer: "openui" })
    })
  })
)
```

Expected initially: FAIL because metadata is not saved or mirrored.

- [ ] **Step 3: Run focused tests and verify failure**

```bash
cd apps
bunx vitest run packages/ui/src/hooks/__tests__/useServerChatLoader.test.ts packages/ui/src/hooks/chat/__tests__/useChatActions.persist-mirror.guard.test.ts
```

Expected: FAIL on new metadata expectations.

- [ ] **Step 4: Add metadata fields to types**

Modify `apps/packages/ui/src/store/option/types.ts`:

```ts
import type { DynamicUIEnvelope, DynamicUIActionUserMetadata } from "@/types/dynamic-ui"

export type MessageMetadataExtra = Record<string, unknown> & {
  dynamic_ui?: DynamicUIEnvelope
  dynamic_ui_action?: DynamicUIActionUserMetadata
}
```

Then use `metadataExtra?: MessageMetadataExtra` on `Message`, and add `metadataExtra?: MessageMetadataExtra` to `MessageVariant`.

Modify `apps/packages/ui/src/types/chat-modes.ts`:

```ts
import type { DynamicUIRequest } from "@/types/dynamic-ui"
import type { MessageMetadataExtra } from "@/store/option"

export interface SaveMessageBase {
  dynamicUIRequest?: DynamicUIRequest
  userMetadataExtra?: MessageMetadataExtra
  assistantMetadataExtra?: MessageMetadataExtra
}
```

- [ ] **Step 5: Persist local metadata**

Modify `apps/packages/ui/src/db/dexie/types.ts`:

```ts
metadataExtra?: Record<string, unknown>;
```

Add `metadataExtra` to `saveMessage` params in `apps/packages/ui/src/db/dexie/helpers.ts`, include it in the `message` object, include it in `buildVariantFromHistory`, and include it in `formatToMessage`.

- [ ] **Step 6: Save metadata through chat helper**

Modify `apps/packages/ui/src/hooks/chat-helper/index.ts`:

- accept `userMetadataExtra` and `assistantMetadataExtra`;
- pass `metadataExtra: userMetadataExtra` on user `saveMessage`;
- pass `metadataExtra: assistantMetadataExtra` on assistant `saveMessage`.

- [ ] **Step 7: Mirror metadata to server chat messages**

Modify `apps/packages/ui/src/hooks/chat/useChatActions.ts` in server mirroring:

```ts
await tldwClient.addChatMessage(cid, {
  role: "assistant",
  content: assistantContent,
  metadata_extra: payload.assistantMetadataExtra
})
```

Also pass `metadata_extra: payload.userMetadataExtra` for the user message only when present.

- [ ] **Step 8: Normalize hydrated server metadata**

Modify `apps/packages/ui/src/hooks/chat/useServerChatLoader.ts` to run valid dynamic metadata through `normalizeDynamicUIEnvelope`:

```ts
const metadataBase = isRecord(metadataExtraCandidate)
  ? metadataExtraCandidate
  : undefined
const dynamicUI = normalizeDynamicUIEnvelope(metadataBase?.dynamic_ui)
const metadataExtra = metadataBase
  ? (() => {
      const { dynamic_ui: _invalidDynamicUI, ...rest } = metadataBase
      return dynamicUI ? { ...rest, dynamic_ui: dynamicUI } : rest
    })()
  : undefined
```

If `dynamic_ui` is present but invalid, omit only `dynamic_ui` and keep unrelated metadata.

- [ ] **Step 9: Run focused tests**

```bash
cd apps
bunx vitest run packages/ui/src/hooks/__tests__/useServerChatLoader.test.ts packages/ui/src/hooks/chat/__tests__/useChatActions.persist-mirror.guard.test.ts packages/ui/src/utils/__tests__/dynamic-ui.test.ts
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add apps/packages/ui/src/store/option/types.ts apps/packages/ui/src/db/dexie/types.ts apps/packages/ui/src/db/dexie/helpers.ts apps/packages/ui/src/hooks/chat-helper/index.ts apps/packages/ui/src/types/chat-modes.ts apps/packages/ui/src/hooks/chat/useChatActions.ts apps/packages/ui/src/hooks/chat/useServerChatLoader.ts apps/packages/ui/src/hooks/__tests__/useServerChatLoader.test.ts apps/packages/ui/src/hooks/chat/__tests__/useChatActions.persist-mirror.guard.test.ts
git commit -m "feat: persist dynamic UI message metadata"
```

---

## Task 3: Renderer Registry And Source Fallback

**Files:**
- Create: `apps/packages/ui/src/components/Common/DynamicUI/DynamicUISourceFallback.tsx`
- Create: `apps/packages/ui/src/components/Common/DynamicUI/DynamicUIErrorBoundary.tsx`
- Create: `apps/packages/ui/src/components/Common/DynamicUI/DynamicMessageRenderer.tsx`
- Create: `apps/packages/ui/src/components/Common/DynamicUI/registry.ts`
- Create: `apps/packages/ui/src/components/Common/DynamicUI/renderers/OpenUIRenderer.tsx`
- Create: `apps/packages/ui/src/components/Common/DynamicUI/__tests__/DynamicMessageRenderer.test.tsx`
- Create: `apps/packages/ui/src/components/Common/Playground/__tests__/Message.dynamic-ui-surface.guard.test.ts`
- Modify: `apps/packages/ui/src/components/Common/Playground/MessageContent.tsx`
- Modify: `apps/packages/ui/src/components/Common/Playground/Message.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundChat.tsx`

- [ ] **Step 1: Write failing renderer tests**

Create `apps/packages/ui/src/components/Common/DynamicUI/__tests__/DynamicMessageRenderer.test.tsx`:

```tsx
// @vitest-environment jsdom
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { DynamicMessageRenderer } from "../DynamicMessageRenderer"

vi.mock("../registry", async () => {
  const actual = await vi.importActual<typeof import("../registry")>("../registry")
  return {
    ...actual,
    loadDynamicUIRenderer: vi.fn(async () => ({
      default: ({
        source,
        onAction
      }: {
        source: string
        onAction?: (payload: unknown) => void
      }) => {
        if (source.includes("throw")) {
          throw new Error("renderer crashed")
        }
        return (
          <button
            type="button"
            data-testid="openui-rendered"
            onClick={() =>
              onAction?.({
                actionId: "survey",
                actionType: "submit",
                values: { answer: "yes" }
              })
            }>
            {source}
          </button>
        )
      }
    }))
  }
})

describe("DynamicMessageRenderer", () => {
  it("renders enabled OpenUI metadata on web chat", async () => {
    render(
      <DynamicMessageRenderer
        envelope={{ renderer: "openui", version: "v1", source: "root = <Card />" }}
        sourceMessageId="assistant-1"
        sourceText="root = <Card />"
        surface="web-chat"
      />
    )

    expect(await screen.findByTestId("openui-rendered")).toHaveTextContent("root = <Card />")
  })

  it("falls back to source when surface is disabled", () => {
    render(
      <DynamicMessageRenderer
        envelope={{ renderer: "openui", version: "v1", source: "root = <Card />" }}
        sourceMessageId="assistant-1"
        sourceText="root = <Card />"
        surface="extension-sidepanel"
      />
    )

    expect(screen.getByText(/OpenUI source/i)).toBeInTheDocument()
  })

  it("falls back to source when the renderer component throws", async () => {
    render(
      <DynamicMessageRenderer
        envelope={{ renderer: "openui", version: "v1", source: "root = <Card /> // throw" }}
        sourceMessageId="assistant-1"
        sourceText="root = <Card /> // throw"
        surface="web-chat"
      />
    )

    expect(await screen.findByRole("alert")).toHaveTextContent("renderer crashed")
    expect(screen.getByText(/root = <Card \/> \/\/ throw/)).toBeInTheDocument()
  })

  it("attaches host-owned source message provenance to renderer actions", async () => {
    const onAction = vi.fn()
    render(
      <DynamicMessageRenderer
        envelope={{ renderer: "openui", version: "v1", source: "root = <Form />" }}
        sourceMessageId="assistant-1"
        sourceText="root = <Form />"
        surface="web-chat"
        onAction={onAction}
      />
    )

    ;(await screen.findByTestId("openui-rendered")).click()

    expect(onAction).toHaveBeenCalledWith({
      renderer: "openui",
      sourceMessageId: "assistant-1",
      actionId: "survey",
      actionType: "submit",
      values: { answer: "yes" }
    })
  })
})
```

- [ ] **Step 2: Run tests and verify failure**

```bash
cd apps
bunx vitest run packages/ui/src/components/Common/DynamicUI/__tests__/DynamicMessageRenderer.test.tsx packages/ui/src/components/Common/Playground/__tests__/Message.dynamic-ui-surface.guard.test.ts
```

Expected: FAIL because components do not exist.

- [ ] **Step 3: Add registry**

Create `registry.ts`:

```ts
import type { ComponentType } from "react"
import type { DynamicUIEnvelope, DynamicUISurface } from "@/types/dynamic-ui"

export type DynamicUIRendererProps = {
  envelope: DynamicUIEnvelope
  sourceMessageId: string
  source: string
  onAction?: (payload: unknown) => void
}

export type DynamicUIRendererComponent = ComponentType<DynamicUIRendererProps>

export const isDynamicUIEnabledForSurface = (surface: DynamicUISurface): boolean =>
  surface === "web-chat"

export const loadDynamicUIRenderer = async (
  renderer: DynamicUIEnvelope["renderer"]
): Promise<{ default: DynamicUIRendererComponent }> => {
  if (renderer === "openui") {
    return import("./renderers/OpenUIRenderer")
  }
  throw new Error(`Unsupported dynamic UI renderer: ${renderer}`)
}
```

- [ ] **Step 4: Add source fallback**

Create `DynamicUISourceFallback.tsx`:

```tsx
import React from "react"

export const DynamicUISourceFallback = ({
  title = "OpenUI source",
  source,
  error
}: {
  title?: string
  source: string
  error?: string
}) => (
  <details className="rounded-md border border-border bg-surface-2 p-3 text-sm" open>
    <summary className="cursor-pointer font-medium text-text">{title}</summary>
    {error ? <p role="alert" className="mt-2 text-danger">{error}</p> : null}
    <pre className="mt-2 max-h-80 overflow-auto whitespace-pre-wrap rounded bg-surface p-2 text-xs text-text-muted">
      {source}
    </pre>
  </details>
)
```

- [ ] **Step 5: Add placeholder OpenUI renderer**

Create `renderers/OpenUIRenderer.tsx` so the Task 3 registry import resolves before Task 4 replaces it with the real adapter:

```tsx
import { DynamicUISourceFallback } from "../DynamicUISourceFallback"
import type { DynamicUIRendererProps } from "../registry"

const OpenUIRenderer = ({ envelope }: DynamicUIRendererProps) => (
  <DynamicUISourceFallback
    source={envelope.source}
    error="OpenUI runtime is not enabled yet."
  />
)

export default OpenUIRenderer
```

- [ ] **Step 6: Add renderer error boundary**

Create `DynamicUIErrorBoundary.tsx`:

```tsx
import React from "react"
import { DynamicUISourceFallback } from "./DynamicUISourceFallback"

export class DynamicUIErrorBoundary extends React.Component<
  { source: string; children: React.ReactNode },
  { error: string | null }
> {
  state = { error: null }

  static getDerivedStateFromError(error: unknown) {
    return {
      error: error instanceof Error ? error.message : "Dynamic UI render failed."
    }
  }

  render() {
    if (this.state.error) {
      return <DynamicUISourceFallback source={this.props.source} error={this.state.error} />
    }
    return this.props.children
  }
}
```

- [ ] **Step 7: Add shared renderer**

Create `DynamicMessageRenderer.tsx`:

```tsx
import React from "react"
import type { DynamicUIEnvelope, DynamicUISurface } from "@/types/dynamic-ui"
import { DynamicUIErrorBoundary } from "./DynamicUIErrorBoundary"
import { DynamicUISourceFallback } from "./DynamicUISourceFallback"
import { isDynamicUIEnabledForSurface, loadDynamicUIRenderer } from "./registry"

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value)

export const DynamicMessageRenderer = ({
  envelope,
  sourceMessageId,
  sourceText,
  surface,
  onAction
}: {
  envelope: DynamicUIEnvelope
  sourceMessageId: string
  sourceText: string
  surface: DynamicUISurface
  onAction?: (payload: unknown) => void
}) => {
  const [Renderer, setRenderer] = React.useState<React.ComponentType<any> | null>(null)
  const [error, setError] = React.useState<string | null>(null)
  const handleAction = React.useCallback(
    (payload: unknown) => {
      if (!onAction) return
      const actionPayload = isRecord(payload) ? payload : { values: payload }
      onAction({
        ...actionPayload,
        renderer: envelope.renderer,
        sourceMessageId
      })
    },
    [envelope.renderer, onAction, sourceMessageId]
  )

  React.useEffect(() => {
    let active = true
    if (!isDynamicUIEnabledForSurface(surface)) return
    loadDynamicUIRenderer(envelope.renderer)
      .then((module) => {
        if (active) setRenderer(() => module.default)
      })
      .catch((err) => {
        if (active) setError(err instanceof Error ? err.message : "Failed to load renderer.")
      })
    return () => {
      active = false
    }
  }, [envelope.renderer, surface])

  if (!isDynamicUIEnabledForSurface(surface)) {
    return <DynamicUISourceFallback source={sourceText} />
  }
  if (error) {
    return <DynamicUISourceFallback source={sourceText} error={error} />
  }
  if (!Renderer) {
    return <DynamicUISourceFallback title="Loading OpenUI source" source={sourceText} />
  }
  return (
    <DynamicUIErrorBoundary source={sourceText}>
      <Renderer
        envelope={envelope}
        sourceMessageId={sourceMessageId}
        source={envelope.source}
        onAction={handleAction}
      />
    </DynamicUIErrorBoundary>
  )
}
```

- [ ] **Step 8: Integrate into message rendering**

Modify `MessageContentProps` to accept:

```ts
metadataExtra?: MessageMetadataExtra
dynamicUISurface?: DynamicUISurface
onDynamicUIAction?: (payload: unknown) => void
```

Before reasoning/Markdown rendering, compute:

```tsx
const dynamicUIEnvelope = normalizeDynamicUIEnvelope(metadataExtra?.dynamic_ui)
if (isBot && !isStreaming && dynamicUIEnvelope) {
  if (!messageId) {
    return (
      <DynamicUISourceFallback
        source={message}
        error="Dynamic UI actions require a saved assistant message id."
      />
    )
  }
  const resolvedSurface = dynamicUISurface ?? "artifact"
  return (
    <DynamicMessageRenderer
      envelope={dynamicUIEnvelope}
      sourceMessageId={messageId}
      sourceText={message}
      surface={resolvedSurface}
      onAction={onDynamicUIAction}
    />
  )
}
```

Import `DynamicUISourceFallback` in `MessageContent.tsx` for the missing-message-id fallback. Thread `messageId`, `dynamicUISurface`, and `onDynamicUIAction` from `Message.tsx` to `MessageContent`. In `Option/Playground/PlaygroundChat.tsx`, pass `dynamicUISurface="web-chat"` only for the main `/chat` transcript. Do not rely on a web-chat default; missing surface props must fall back to source rendering so sidepanel, workspace, tests, and future call sites do not accidentally active-render OpenUI.

Create `Message.dynamic-ui-surface.guard.test.ts` as a source guard, following existing guard-test style:

```ts
import fs from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

const readSource = (relativePath: string) =>
  fs.readFileSync(path.resolve(__dirname, relativePath), "utf8")

describe("dynamic UI surface guard", () => {
  it("does not active-render OpenUI when a message caller omits surface", () => {
    const source = readSource("../MessageContent.tsx")
    expect(source).toContain('dynamicUISurface ?? "artifact"')
    expect(source).not.toContain('dynamicUISurface ?? "web-chat"')
  })

  it("opts the main /chat transcript into active dynamic UI rendering explicitly", () => {
    const source = readSource("../../../Option/Playground/PlaygroundChat.tsx")
    expect(source).toContain('dynamicUISurface="web-chat"')
  })
})
```

- [ ] **Step 9: Run focused tests**

```bash
cd apps
bunx vitest run packages/ui/src/components/Common/DynamicUI/__tests__/DynamicMessageRenderer.test.tsx packages/ui/src/components/Common/Playground/__tests__/Message.dynamic-ui-surface.guard.test.ts packages/ui/src/components/Common/Playground/__tests__/Message.error-recovery.guard.test.ts
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add apps/packages/ui/src/components/Common/DynamicUI apps/packages/ui/src/components/Common/Playground/__tests__/Message.dynamic-ui-surface.guard.test.ts apps/packages/ui/src/components/Common/Playground/Message.tsx apps/packages/ui/src/components/Common/Playground/MessageContent.tsx apps/packages/ui/src/components/Option/Playground/PlaygroundChat.tsx
git commit -m "feat: add dynamic UI message renderer"
```

---

## Task 4: OpenUI Adapter

**Files:**
- Create/modify: `apps/packages/ui/src/components/Common/DynamicUI/renderers/OpenUIRenderer.tsx`
- Test: `apps/packages/ui/src/components/Common/DynamicUI/__tests__/DynamicMessageRenderer.test.tsx`

- [ ] **Step 1: Add adapter smoke test with mocked OpenUI runtime**

Extend `DynamicMessageRenderer.test.tsx` or add `OpenUIRenderer.test.tsx` to mock the selected OpenUI package and assert:

```tsx
expect(screen.getByTestId("openui-runtime")).toHaveTextContent("root = <Card />")
```

Expected initially: FAIL because the adapter is still missing or placeholder-only.

- [ ] **Step 2: Implement the adapter against the selected runtime**

Use the exact imports verified in Task 0. The adapter should look conceptually like:

```tsx
import React from "react"
import type { DynamicUIRendererProps } from "../registry"
// Example only. Replace with Task 0 verified imports.
import { Renderer } from "@openuidev/react-lang"
import { openuiLibrary } from "@openuidev/react-ui"

const toOpenUITheme = () => ({
  // Map app CSS variables/tokens here. Keep this minimal in v1.
})

const OpenUIRenderer = ({ source, onAction }: DynamicUIRendererProps) => {
  return (
    <div className="dynamic-ui-openui rounded-md border border-border bg-surface p-3">
      <Renderer
        source={source}
        library={openuiLibrary}
        theme={toOpenUITheme()}
        onAction={onAction}
      />
    </div>
  )
}

export default OpenUIRenderer
```

If the real OpenUI API differs, adapt the wrapper but keep the same local `DynamicUIRendererProps` contract.

- [ ] **Step 3: Add minimal styling if needed**

Prefer existing tokens. Add local classes only if OpenUI needs a bounded frame. Do not create a one-hue theme or decorative wrapper.

- [ ] **Step 4: Run focused tests**

```bash
cd apps
bunx vitest run packages/ui/src/components/Common/DynamicUI/__tests__/DynamicMessageRenderer.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Verify package builds can resolve the lazy adapter**

Run at least:

```bash
cd apps
bun --cwd tldw-frontend run build:dev
```

If extension dependency resolution was changed, also run:

```bash
cd apps
bun --cwd extension run build:chrome:dev
```

Expected: build passes. If extension fails due CSP/runtime behavior, keep sidepanel fallback enabled and record the blocker in the feasibility review.

- [ ] **Step 6: Commit**

```bash
git add apps/packages/ui/src/components/Common/DynamicUI/renderers/OpenUIRenderer.tsx apps/packages/ui/src/components/Common/DynamicUI/__tests__/DynamicMessageRenderer.test.tsx apps/packages/ui/src/assets/tailwind.css
git commit -m "feat: add OpenUI renderer adapter"
```

Only add `tailwind.css` if styling changed.

---

## Task 5: Request Mode, Prompt Injection, And Metadata Tagging

**Files:**
- Create: `apps/packages/ui/src/utils/dynamic-ui-openui-prompt.ts`
- Modify: `apps/packages/ui/src/hooks/chat-modes/chatModePipeline.ts`
- Modify: `apps/packages/ui/src/hooks/chat-modes/normalChatMode.ts`
- Modify: `apps/packages/ui/src/hooks/chat/useChatActions.ts`
- Test: `apps/packages/ui/src/hooks/chat-modes/__tests__/chatModePipeline.dynamic-ui.test.ts`

- [ ] **Step 1: Write failing pipeline tests**

Create `apps/packages/ui/src/hooks/chat-modes/__tests__/chatModePipeline.dynamic-ui.test.ts`:

```ts
// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest"
import { runChatPipeline, type ChatModeDefinition } from "../chatModePipeline"

const mocks = vi.hoisted(() => ({
  pageAssistModel: vi.fn(),
  saveMessageOnSuccess: vi.fn(async () => "history-1"),
  saveMessageOnError: vi.fn(async () => "history-1"),
  setMessages: vi.fn(),
  setHistory: vi.fn(),
  setIsProcessing: vi.fn(),
  setStreaming: vi.fn(),
  setAbortController: vi.fn(),
  setHistoryId: vi.fn()
}))

vi.mock("@/models", () => ({ pageAssistModel: (...args: unknown[]) => mocks.pageAssistModel(...args) }))
vi.mock("@/db/dexie/helpers", () => ({ generateID: vi.fn(() => "assistant-1") }))
vi.mock("@/db/dexie/nickname", () => ({ getModelNicknameByID: vi.fn(async () => null) }))
vi.mock("@/utils/mcp-disclosure", () => ({ applyMcpModuleDisclosureFromToolCalls: vi.fn() }))
vi.mock("@/store/option", () => ({ useStoreMessageOption: { getState: () => ({ setHistory: vi.fn() }) } }))

const mode: ChatModeDefinition<any> = {
  id: "normal",
  setupMessages: () => ({ targetMessageId: "assistant-1" }),
  preparePrompt: async () => ({
    chatHistory: [{ role: "system", content: "existing system" }],
    humanMessage: { role: "user", content: "Build a dashboard" },
    sources: []
  })
}

describe("runChatPipeline dynamic UI request mode", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("injects OpenUI instructions and saves metadata only after source preflight passes", async () => {
    const stream = vi.fn(async function* () {
      yield "root = <Card />"
    })
    mocks.pageAssistModel.mockResolvedValue({
      stream
    })

    await runChatPipeline(mode, "Build a dashboard", "", false, [], [], new AbortController().signal, {
      selectedModel: "test-model",
      useOCR: false,
      setMessages: mocks.setMessages,
      saveMessageOnSuccess: mocks.saveMessageOnSuccess,
      saveMessageOnError: mocks.saveMessageOnError,
      setHistory: mocks.setHistory,
      setIsProcessing: mocks.setIsProcessing,
      setStreaming: mocks.setStreaming,
      setAbortController: mocks.setAbortController,
      historyId: "history-1",
      setHistoryId: mocks.setHistoryId,
      dynamicUIRequest: { renderer: "openui" }
    })

    const promptMessages = stream.mock.calls[0]?.[0]
    expect(JSON.stringify(promptMessages)).toContain("OpenUI")
    expect(mocks.saveMessageOnSuccess).toHaveBeenCalledWith(
      expect.objectContaining({
        assistantMetadataExtra: expect.objectContaining({
          dynamic_ui: expect.objectContaining({ renderer: "openui" })
        })
      })
    )
  })

  it("does not tag plain text responses even when OpenUI was requested", async () => {
    mocks.pageAssistModel.mockResolvedValue({
      stream: async function* () {
        yield "I cannot do that."
      }
    })

    await runChatPipeline(mode, "Build a dashboard", "", false, [], [], new AbortController().signal, {
      selectedModel: "test-model",
      useOCR: false,
      setMessages: mocks.setMessages,
      saveMessageOnSuccess: mocks.saveMessageOnSuccess,
      saveMessageOnError: mocks.saveMessageOnError,
      setHistory: mocks.setHistory,
      setIsProcessing: mocks.setIsProcessing,
      setStreaming: mocks.setStreaming,
      setAbortController: mocks.setAbortController,
      historyId: "history-1",
      setHistoryId: mocks.setHistoryId,
      dynamicUIRequest: { renderer: "openui" }
    })

    const payload = mocks.saveMessageOnSuccess.mock.calls.at(-1)?.[0]
    expect(payload?.assistantMetadataExtra).toBeUndefined()
  })
})
```

Adjust assertions to inspect the streamed prompt if the existing mock shape requires it.

- [ ] **Step 2: Run the tests and verify failure**

```bash
cd apps
bunx vitest run packages/ui/src/hooks/chat-modes/__tests__/chatModePipeline.dynamic-ui.test.ts
```

Expected: FAIL because `dynamicUIRequest` and prompt injection do not exist.

- [ ] **Step 3: Add OpenUI prompt helper**

Create `apps/packages/ui/src/utils/dynamic-ui-openui-prompt.ts`:

```ts
export const OPENUI_SYSTEM_PROMPT = [
  "You are generating an OpenUI Lang interface for this response.",
  "Return only valid OpenUI Lang source.",
  "Do not wrap the output in Markdown fences.",
  "The root component must be assigned with `root = ...`.",
  "Use forms/buttons only when the user can reasonably act on them.",
  "Do not request passwords, API keys, tokens, credentials, or secrets."
].join("\n")
```

- [ ] **Step 4: Add dynamic request params**

Thread `dynamicUIRequest?: DynamicUIRequest` through:

- `ChatModeParamsBase`
- `NormalChatModeParams`
- `ChatModeOverrides`
- `useChatActions.onSubmit` payload
- `buildChatModeParams`

Also thread `userMetadataExtra?: MessageMetadataExtra` through the same path:

- `usePlaygroundSubmit` dispatch payload;
- `useChatActions.onSubmit` payload type;
- `buildChatModeParams`;
- `runChatPipeline`;
- `saveMessageOnSuccess`;
- server-chat mirroring in `useChatActions`.

This is required so OpenUI action submissions do not lose `metadataExtra.dynamic_ui_action`.

- [ ] **Step 5: Inject prompt in `runChatPipeline`**

After `mode.preparePrompt(context)` and before message steering/tool streaming, add:

```ts
if (params.dynamicUIRequest?.renderer === "openui") {
  promptData.chatHistory = [
    await systemPromptFormatter({ content: OPENUI_SYSTEM_PROMPT }),
    ...promptData.chatHistory
  ]
}
```

Use the existing formatter import path. If `systemPromptFormatter` is not safe to import here, create a small helper that returns the same `{ role: "system", content }` shape expected by the model client.

- [ ] **Step 6: Tag completed assistant output after preflight**

Before the final `setMessagesWithTransition` and `saveMessageOnSuccess`, compute:

```ts
const dynamicUIEnvelope = params.dynamicUIRequest?.renderer === "openui"
  ? buildDynamicUIEnvelope("openui", fullText)
  : null
const assistantMetadataExtra = dynamicUIEnvelope
  ? { dynamic_ui: dynamicUIEnvelope }
  : undefined
```

Update the final message with `metadataExtra: assistantMetadataExtra` only when present, and pass `assistantMetadataExtra` to `saveMessageOnSuccess`.

For non-regenerate user turns, pass `params.userMetadataExtra` through to the user stub and `saveMessageOnSuccess` as `userMetadataExtra`.

- [ ] **Step 7: Keep preflight/image generation path unchanged**

Do not add dynamic UI metadata to `mode.preflight` image generation responses. OpenUI request mode should be ignored for slash-command image generation and other handled preflight paths.

- [ ] **Step 8: Run tests**

```bash
cd apps
bunx vitest run packages/ui/src/hooks/chat-modes/__tests__/chatModePipeline.dynamic-ui.test.ts packages/ui/src/hooks/chat-modes/__tests__/chatModePipeline.conversation-id.test.ts packages/ui/src/utils/__tests__/dynamic-ui.test.ts
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add apps/packages/ui/src/utils/dynamic-ui-openui-prompt.ts apps/packages/ui/src/hooks/chat-modes/chatModePipeline.ts apps/packages/ui/src/hooks/chat-modes/normalChatMode.ts apps/packages/ui/src/hooks/chat/useChatActions.ts apps/packages/ui/src/types/chat-modes.ts apps/packages/ui/src/hooks/chat-modes/__tests__/chatModePipeline.dynamic-ui.test.ts
git commit -m "feat: add OpenUI request mode pipeline support"
```

---

## Task 6: Dynamic UI Action Bridge

**Files:**
- Create: `apps/packages/ui/src/hooks/chat/useDynamicUIActionBridge.ts`
- Test: `apps/packages/ui/src/hooks/chat/__tests__/useDynamicUIActionBridge.test.tsx`
- Test: `apps/packages/ui/src/hooks/chat/__tests__/useChatActions.dynamic-ui-action.integration.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundChat.tsx`
- Modify: `apps/packages/ui/src/components/Common/Playground/Message.tsx`
- Modify: `apps/packages/ui/src/components/Common/Playground/MessageContent.tsx`

- [ ] **Step 1: Write failing hook tests**

Create `apps/packages/ui/src/hooks/chat/__tests__/useDynamicUIActionBridge.test.tsx`:

```tsx
// @vitest-environment jsdom
import { renderHook, act } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { useDynamicUIActionBridge } from "../useDynamicUIActionBridge"

describe("useDynamicUIActionBridge", () => {
  it("submits valid OpenUI actions as normal user messages with metadata", async () => {
    const onSubmit = vi.fn(async () => ({ status: "submitted" }))
    const { result } = renderHook(() =>
      useDynamicUIActionBridge({
        messages: [{ id: "assistant-1", isBot: true, name: "Assistant", message: "", sources: [] }],
        onSubmit,
        confirmSensitiveValues: vi.fn()
      })
    )

    await act(async () => {
      await result.current({
        renderer: "openui",
        sourceMessageId: "assistant-1",
        actionId: "survey",
        actionType: "submit",
        values: { answer: "yes" }
      })
    })

    expect(onSubmit).toHaveBeenCalledWith(
      expect.objectContaining({
        message: expect.stringContaining("OpenUI action: submit survey"),
        userMetadataExtra: expect.objectContaining({
          dynamic_ui_action: expect.objectContaining({ actionId: "survey" })
        })
      })
    )
  })

  it("blocks sensitive-looking values without confirmation", async () => {
    const onSubmit = vi.fn()
    const { result } = renderHook(() =>
      useDynamicUIActionBridge({
        messages: [{ id: "assistant-1", isBot: true, name: "Assistant", message: "", sources: [] }],
        onSubmit,
        confirmSensitiveValues: vi.fn(async () => false)
      })
    )

    await act(async () => {
      await result.current({
        renderer: "openui",
        sourceMessageId: "assistant-1",
        actionId: "login",
        actionType: "submit",
        values: { password: "secret" }
      })
    })

    expect(onSubmit).not.toHaveBeenCalled()
  })
})
```

- [ ] **Step 2: Run tests and verify failure**

```bash
cd apps
bunx vitest run packages/ui/src/hooks/chat/__tests__/useDynamicUIActionBridge.test.tsx
```

Expected: FAIL because the hook does not exist.

- [ ] **Step 3: Implement the bridge**

Create `useDynamicUIActionBridge.ts`:

```ts
import React from "react"
import type { Message } from "@/store/option"
import {
  formatDynamicUIActionUserMessage,
  normalizeDynamicUIActionPayload,
  shouldBlockDynamicUIActionValues
} from "@/utils/dynamic-ui"

export const useDynamicUIActionBridge = ({
  messages,
  onSubmit,
  confirmSensitiveValues
}: {
  messages: Message[]
  onSubmit: (payload: any) => Promise<unknown>
  confirmSensitiveValues: (payload: unknown) => Promise<boolean>
}) =>
  React.useCallback(async (rawPayload: unknown) => {
    const currentMessageIds = new Set(messages.map((message) => message.id).filter(Boolean) as string[])
    const normalized = normalizeDynamicUIActionPayload(rawPayload, { currentMessageIds })
    if (!normalized) return
    if (shouldBlockDynamicUIActionValues(normalized.values)) {
      const confirmed = await confirmSensitiveValues(normalized)
      if (!confirmed) return
    }
    const submittedAt = new Date().toISOString()
    const metadata = { ...normalized, submittedAt }
    await onSubmit({
      message: formatDynamicUIActionUserMessage(metadata),
      image: "",
      userMetadataExtra: {
        dynamic_ui_action: metadata
      }
    })
  }, [confirmSensitiveValues, messages, onSubmit])
```

- [ ] **Step 4: Wire `/chat` message action callback**

In `PlaygroundChat.tsx`, create the bridge using `useMessageOption().onSubmit` and pass it to each `PlaygroundMessage`.

Use a conservative confirmation function:

```ts
const confirmSensitiveValues = React.useCallback(async () => false, [])
```

This implements "blocked" rather than "confirm" for v1, matching the reviewer recommendation to pick one.

- [ ] **Step 5: Verify submit metadata threading**

Confirm these real path signatures accept and forward `userMetadataExtra`:

```ts
// useChatActions.onSubmit payload
userMetadataExtra?: MessageMetadataExtra

// ChatModeParamsBase
userMetadataExtra?: MessageMetadataExtra

// saveMessageOnSuccess payload
userMetadataExtra?: MessageMetadataExtra
```

Add a real-path test, either in a new `useChatActions.dynamic-ui-action.integration.test.tsx` file or beside the existing persist-mirror guard tests, proving `userMetadataExtra.dynamic_ui_action` reaches both the local save helper and server mirroring path:

```tsx
// @vitest-environment jsdom
it("persists dynamic UI action provenance through the normal submit path", async () => {
  const metadata = {
    dynamic_ui_action: {
      renderer: "openui",
      sourceMessageId: "assistant-1",
      actionId: "survey",
      actionType: "submit",
      values: { answer: "yes" },
      submittedAt: "2026-06-01T00:00:00.000Z"
    }
  }

  const { onSubmit, saveMessageOnSuccessMock, addChatMessageMock } =
    renderUseChatActionsHarness({
      serverChatId: "chat-1",
      historyId: "history-1"
    })

  await onSubmit({
    message: "OpenUI action: submit survey\n\nSubmitted values:\n- answer: yes",
    image: "",
    userMetadataExtra: metadata
  })

  expect(saveMessageOnSuccessMock).toHaveBeenCalledWith(
    expect.objectContaining({ userMetadataExtra: metadata })
  )
  expect(addChatMessageMock).toHaveBeenCalledWith(
    "chat-1",
    expect.objectContaining({
      role: "user",
      metadata_extra: metadata
    })
  )
})
```

Use the existing `useChatActions` test harness/mocks if names differ; do not replace this with a hook-only assertion. The test must exercise `onSubmit` because that is where action provenance can be accidentally dropped.

- [ ] **Step 6: Thread callback through message components**

Add `onDynamicUIAction?: (payload: unknown) => void` to `PlaygroundMessage` props and pass it to `MessageContent`, then to `DynamicMessageRenderer`.

- [ ] **Step 7: Run tests**

```bash
cd apps
bunx vitest run packages/ui/src/hooks/chat/__tests__/useDynamicUIActionBridge.test.tsx packages/ui/src/hooks/chat/__tests__/useChatActions.dynamic-ui-action.integration.test.tsx packages/ui/src/components/Common/DynamicUI/__tests__/DynamicMessageRenderer.test.tsx
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add apps/packages/ui/src/hooks/chat/useDynamicUIActionBridge.ts apps/packages/ui/src/hooks/chat/__tests__/useDynamicUIActionBridge.test.tsx apps/packages/ui/src/hooks/chat/__tests__/useChatActions.dynamic-ui-action.integration.test.tsx apps/packages/ui/src/components/Option/Playground/PlaygroundChat.tsx apps/packages/ui/src/components/Common/Playground/Message.tsx apps/packages/ui/src/components/Common/Playground/MessageContent.tsx
git commit -m "feat: bridge OpenUI actions back to chat"
```

---

## Task 7: `/chat` OpenUI Request Mode Control

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/hooks/usePlaygroundSubmit.ts`
- Modify: `apps/packages/ui/src/public/_locales/en/playground.json`
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundForm.openui-mode.test.tsx`

- [ ] **Step 1: Write failing form test**

Create `PlaygroundForm.openui-mode.test.tsx` following existing `PlaygroundForm.*.test.tsx` mock patterns. Assert:

```tsx
expect(screen.getByRole("button", { name: /OpenUI/i })).toBeInTheDocument()
await user.click(screen.getByRole("button", { name: /OpenUI/i }))
await user.type(screen.getByRole("textbox"), "Build a settings form")
await user.click(screen.getByRole("button", { name: /Send/i }))
expect(onSubmit).toHaveBeenCalledWith(
  expect.objectContaining({
    message: "Build a settings form",
    requestOverrides: expect.objectContaining({
      dynamicUIRequest: { renderer: "openui" }
    })
  })
)
```

Expected: FAIL because there is no control or request override.

- [ ] **Step 2: Run the failing test**

```bash
cd apps
bunx vitest run packages/ui/src/components/Option/Playground/__tests__/PlaygroundForm.openui-mode.test.tsx
```

Expected: FAIL.

- [ ] **Step 3: Add transient state to `PlaygroundForm`**

Add:

```ts
const [openUIRequestMode, setOpenUIRequestMode] = React.useState(false)
```

Render a compact toolbar button or switch near other mode controls:

```tsx
<Tooltip title={t("playground:composer.openuiModeTooltip", "Ask the assistant to answer with an OpenUI interface for the next message.")}>
  <Button
    type={openUIRequestMode ? "primary" : "default"}
    aria-pressed={openUIRequestMode}
    onClick={() => setOpenUIRequestMode((value) => !value)}
  >
    OpenUI
  </Button>
</Tooltip>
```

Use an existing icon button style if the surrounding toolbar expects icons. Do not add instructional text in the main chat body.

- [ ] **Step 4: Pass state to `usePlaygroundSubmit`**

Extend `UsePlaygroundSubmitDeps` with:

```ts
openUIRequestMode: boolean
clearOpenUIRequestMode: () => void
```

In dispatch payload:

```ts
requestOverrides: openUIRequestMode
  ? { dynamicUIRequest: { renderer: "openui" } }
  : undefined
```

After successful dispatch, call `clearOpenUIRequestMode()` so the mode is temporary.

- [ ] **Step 5: Add locale strings**

Modify `apps/packages/ui/src/public/_locales/en/playground.json`:

```json
"composer": {
  "openuiMode": "OpenUI",
  "openuiModeTooltip": "Answer with an OpenUI interface for the next message."
}
```

Preserve existing JSON structure and ordering.

- [ ] **Step 6: Run tests**

```bash
cd apps
bunx vitest run packages/ui/src/components/Option/Playground/__tests__/PlaygroundForm.openui-mode.test.tsx packages/ui/src/components/Option/Playground/__tests__/PlaygroundForm.signals.guard.test.ts
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx apps/packages/ui/src/components/Option/Playground/hooks/usePlaygroundSubmit.ts apps/packages/ui/src/public/_locales/en/playground.json apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundForm.openui-mode.test.tsx
git commit -m "feat: add OpenUI request mode control"
```

---

## Task 8: Shared Surface Fallbacks And Build Verification

**Files:**
- Modify: `apps/packages/ui/src/components/Sidepanel/Chat/body.tsx`
- Modify: `apps/packages/ui/src/components/Option/ChatWorkspace/WorkspaceChatPanel.tsx`
- Test: `apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/WorkspaceChatPanel.test.tsx`
- Test: sidepanel test chosen by implementer after inspecting existing coverage
- Create: `apps/tldw-frontend/e2e/smoke/chat-openui-dynamic-ui.spec.ts`

- [ ] **Step 1: Pass explicit surface IDs**

In shared message call sites:

- `/chat` `PlaygroundChat.tsx`: `dynamicUISurface="web-chat"`
- extension sidepanel body: `dynamicUISurface="extension-sidepanel"`
- workspace chat panel: `dynamicUISurface="workspace"`

Expected behavior: non-web-chat surfaces source-fallback unless registry capability changes after CSP checks.

- [ ] **Step 2: Add workspace/sidepanel fallback tests**

Add or update tests to render a message with:

```ts
metadataExtra: {
  dynamic_ui: {
    renderer: "openui",
    version: "v1",
    source: "root = <Card />"
  }
}
```

Assert sidepanel/workspace show source fallback, not the active OpenUI renderer.

- [ ] **Step 3: Add browser smoke**

Create `apps/tldw-frontend/e2e/smoke/chat-openui-dynamic-ui.spec.ts` with a mocked/synthetic persisted message path if available, or a route-level smoke that asserts the OpenUI control exists and the page does not throw.

Minimal first smoke:

```ts
import { test, expect } from "@playwright/test"

test("chat exposes OpenUI mode control", async ({ page }) => {
  await page.goto("/chat")
  await expect(page.getByRole("button", { name: /OpenUI/i })).toBeVisible()
})
```

- [ ] **Step 4: Run focused unit tests**

```bash
cd apps
bunx vitest run packages/ui/src/components/Common/DynamicUI/__tests__/DynamicMessageRenderer.test.tsx packages/ui/src/components/Option/ChatWorkspace/__tests__/WorkspaceChatPanel.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Run frontend build**

```bash
cd apps
bun --cwd tldw-frontend run build:dev
```

Expected: PASS.

- [ ] **Step 6: Run extension build if dependencies changed there**

```bash
cd apps
bun --cwd extension run build:chrome:dev
```

Expected: PASS, or documented source-fallback blocker if OpenUI cannot safely resolve in extension.

- [ ] **Step 7: Commit**

```bash
git add apps/packages/ui/src/components/Sidepanel/Chat/body.tsx apps/packages/ui/src/components/Option/ChatWorkspace/WorkspaceChatPanel.tsx apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/WorkspaceChatPanel.test.tsx apps/tldw-frontend/e2e/smoke/chat-openui-dynamic-ui.spec.ts
git commit -m "test: verify dynamic UI chat surfaces"
```

Add the chosen sidepanel test file if changed.

---

## Task 9: Final Verification And Backlog Closeout

**Files:**
- Modify: `backlog/tasks/task-493 - Plan-OpenUI-dynamic-chat-rendering-implementation.md` only if implementation uses this same task; otherwise close the implementation task created for execution.

- [ ] **Step 1: Run focused frontend test suite**

```bash
cd apps
bunx vitest run \
  packages/ui/src/utils/__tests__/dynamic-ui.test.ts \
  packages/ui/src/components/Common/DynamicUI/__tests__/DynamicMessageRenderer.test.tsx \
  packages/ui/src/hooks/chat-modes/__tests__/chatModePipeline.dynamic-ui.test.ts \
  packages/ui/src/hooks/chat/__tests__/useDynamicUIActionBridge.test.tsx \
  packages/ui/src/hooks/__tests__/useServerChatLoader.test.ts \
  packages/ui/src/components/Option/Playground/__tests__/PlaygroundForm.openui-mode.test.tsx
```

Expected: PASS.

- [ ] **Step 2: Run build verification**

```bash
cd apps
bun --cwd tldw-frontend run build:dev
```

Expected: PASS.

- [ ] **Step 3: Run extension build if OpenUI dependency resolves in extension**

```bash
cd apps
bun --cwd extension run build:chrome:dev
```

Expected: PASS or documented fallback decision.

- [ ] **Step 4: Run Bandit scope check**

This is frontend-only code, but repo policy requires recording the Bandit decision. Run a no-op backend touched-scope check only if Python files changed. If no Python files changed, record: "Bandit skipped: frontend/docs-only implementation, no Python touched."

- [ ] **Step 5: Manual QA**

Start the WebUI:

```bash
cd apps
bun --cwd tldw-frontend run dev
```

Open `/chat` and verify:

- plain chat still renders Markdown normally;
- OpenUI mode control is visible;
- OpenUI mode applies to one send only;
- OpenUI-shaped completed assistant content renders when source preflight passes;
- plain/refusal content from an OpenUI request saves as normal text;
- generated form submit creates a visible user turn;
- sensitive-looking form values are blocked;
- reload preserves a metadata-tagged OpenUI message;
- sidepanel/workspace fallback remains safe if not enabled.

- [ ] **Step 6: Commit closeout updates**

```bash
git status --short
git add <changed-files>
git commit -m "chore: verify OpenUI dynamic chat rendering"
```

Do not include unrelated dirty files.

---

## Execution Notes

- Keep Stage 4a/4b/4c from the design as separate follow-up PRs unless the runtime and v1 path are very small after implementation.
- If Stage 0 fails, stop after committing the feasibility review and do not add OpenUI dependencies.
- If the real OpenUI renderer API requires unsafe dynamic evaluation, preserve the dynamic UI abstraction and leave the OpenUI adapter disabled/source-fallback.
- Prefer blocked sensitive action values in v1; confirmation UI can be a follow-up.
- Do not add a backend OpenUI generation endpoint in this implementation.
- Do not commit generated OpenUI bundles.
