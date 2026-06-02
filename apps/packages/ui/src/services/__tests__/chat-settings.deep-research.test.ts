import { beforeEach, describe, expect, it, vi } from "vitest"

const storageState = vi.hoisted(() => {
  const store = new Map<string, unknown>()
  return {
    store,
    get: vi.fn(async (key: string) => store.get(key)),
    set: vi.fn(async (key: string, value: unknown) => {
      store.set(key, value)
    }),
    remove: vi.fn(async (key: string) => {
      store.delete(key)
    }),
    initialize: vi.fn(async () => undefined),
    getChatSettings: vi.fn(),
    updateChatSettings: vi.fn()
  }
})

vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    get: storageState.get,
    set: storageState.set,
    remove: storageState.remove
  })
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: () => storageState.initialize(),
    getChatSettings: () => storageState.getChatSettings(),
    updateChatSettings: () => storageState.updateChatSettings()
  }
}))

import {
  applyChatSettingsPatch,
  getChatSettingsStorageKey,
  mergeChatSettings,
  normalizeChatSettingsRecord
} from "@/services/chat-settings"

const buildAttachment = (overrides: Record<string, unknown> = {}) => ({
  run_id: "run_123",
  query: "Battery recycling supply chain",
  question: "Battery recycling supply chain",
  outline: [{ title: "Overview" }],
  key_claims: [{ text: "Claim one" }],
  unresolved_questions: ["Open question"],
  verification_summary: { unsupported_claim_count: 0 },
  source_trust_summary: { high_trust_count: 1 },
  research_url: "/research?run=run_123",
  attached_at: "2026-03-08T20:00:00Z",
  updatedAt: "2026-03-08T20:00:00Z",
  ...overrides
})

describe("chat settings deep research attachment", () => {
  beforeEach(() => {
    storageState.store.clear()
    vi.clearAllMocks()
  })

  it("strips malformed persisted deep research attachments during normalization", () => {
    const settings = normalizeChatSettingsRecord({
      schemaVersion: 2,
      updatedAt: "2026-03-08T20:00:00Z",
      deepResearchAttachment: {
        ...buildAttachment(),
        unexpected: true
      }
    })

    expect(settings?.deepResearchAttachment).toBeUndefined()
  })

  it("canonicalizes persisted deep research attachment links from run_id", () => {
    const settings = normalizeChatSettingsRecord({
      schemaVersion: 2,
      updatedAt: "2026-03-08T20:00:00Z",
      deepResearchAttachment: {
        ...buildAttachment(),
        research_url: "https://example.com/not-research"
      }
    })

    expect(settings?.deepResearchAttachment?.research_url).toBe(
      "/research?run=run_123"
    )
  })

  it("prefers the newer attachment timestamp during merge even when top-level settings are older", () => {
    const local = normalizeChatSettingsRecord({
      schemaVersion: 2,
      updatedAt: "2026-03-08T18:00:00Z",
      authorNote: "local note",
      deepResearchAttachment: buildAttachment({
        question: "Newer local attachment",
        updatedAt: "2026-03-08T21:00:00Z"
      })
    })
    const remote = normalizeChatSettingsRecord({
      schemaVersion: 2,
      updatedAt: "2026-03-08T22:00:00Z",
      authorNote: "remote note",
      deepResearchAttachment: buildAttachment({
        question: "Older remote attachment",
        updatedAt: "2026-03-08T19:00:00Z"
      })
    })

    const merged = mergeChatSettings(local, remote)

    expect(merged?.authorNote).toBe("remote note")
    expect(merged?.updatedAt).toBe("2026-03-08T22:00:00Z")
    expect(merged?.deepResearchAttachment?.question).toBe(
      "Newer local attachment"
    )
    expect(merged?.deepResearchAttachment?.updatedAt).toBe(
      "2026-03-08T21:00:00Z"
    )
  })

  it("drops malformed persisted attachments before applying unrelated settings patches", async () => {
    const storageKey = getChatSettingsStorageKey("local:history-1")
    storageState.store.set(storageKey, {
      schemaVersion: 2,
      updatedAt: "2026-03-08T18:00:00Z",
      deepResearchAttachment: {
        ...buildAttachment(),
        updatedAt: "not-a-timestamp"
      }
    })

    const next = await applyChatSettingsPatch({
      historyId: "history-1",
      serverChatId: null,
      patch: { authorNote: "persisted note" }
    })

    expect(next?.authorNote).toBe("persisted note")
    expect(next?.deepResearchAttachment).toBeUndefined()
    expect(storageState.store.get(storageKey)).toMatchObject({
      authorNote: "persisted note"
    })
    expect(
      (storageState.store.get(storageKey) as Record<string, unknown>)
        .deepResearchAttachment
    ).toBeUndefined()
  })
})
