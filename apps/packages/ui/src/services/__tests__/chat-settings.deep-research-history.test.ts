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

const buildAttachment = (overrides: Record<string, unknown> = {}) => {
  const runId = typeof overrides.run_id === "string" ? overrides.run_id : "run_123"
  return {
    run_id: runId,
    query: `Query for ${runId}`,
    question: `Question for ${runId}`,
    outline: [{ title: "Overview" }],
    key_claims: [{ text: `Claim for ${runId}` }],
    unresolved_questions: [`Open question for ${runId}`],
    verification_summary: { unsupported_claim_count: 0 },
    source_trust_summary: { high_trust_count: 1 },
    research_url: `/research?run=${runId}`,
    attached_at: "2026-03-08T20:00:00Z",
    updatedAt: "2026-03-08T20:00:00Z",
    ...overrides
  }
}

describe("chat settings deep research attachment history", () => {
  beforeEach(() => {
    storageState.store.clear()
    vi.clearAllMocks()
  })

  it("keeps valid history entries during normalization", () => {
    const settings = normalizeChatSettingsRecord({
      schemaVersion: 2,
      updatedAt: "2026-03-08T22:00:00Z",
      deepResearchAttachmentHistory: [
        buildAttachment({
          run_id: "run_hist_1",
          updatedAt: "2026-03-08T21:00:00Z"
        }),
        buildAttachment({
          run_id: "run_hist_2",
          updatedAt: "2026-03-08T20:30:00Z"
        })
      ]
    })

    expect(settings?.deepResearchAttachmentHistory).toEqual([
      expect.objectContaining({ run_id: "run_hist_1" }),
      expect.objectContaining({ run_id: "run_hist_2" })
    ])
  })

  it("strips malformed history entries during normalization", () => {
    const settings = normalizeChatSettingsRecord({
      schemaVersion: 2,
      updatedAt: "2026-03-08T22:00:00Z",
      deepResearchAttachmentHistory: [
        buildAttachment({ run_id: "run_hist_valid" }),
        {
          ...buildAttachment({ run_id: "run_hist_bad" }),
          unexpected: true
        },
        {
          ...buildAttachment({ run_id: "run_hist_bad_time" }),
          updatedAt: "not-a-timestamp"
        }
      ]
    })

    expect(settings?.deepResearchAttachmentHistory).toEqual([
      expect.objectContaining({ run_id: "run_hist_valid" })
    ])
  })

  it("dedupes history by run_id, excludes the active run, and caps entries at three", () => {
    const settings = normalizeChatSettingsRecord({
      schemaVersion: 2,
      updatedAt: "2026-03-08T22:00:00Z",
      deepResearchAttachment: buildAttachment({
        run_id: "run_active",
        updatedAt: "2026-03-08T22:00:00Z"
      }),
      deepResearchAttachmentHistory: [
        buildAttachment({
          run_id: "run_hist_old",
          updatedAt: "2026-03-08T19:00:00Z"
        }),
        buildAttachment({
          run_id: "run_active",
          updatedAt: "2026-03-08T21:00:00Z"
        }),
        buildAttachment({
          run_id: "run_hist_dup",
          query: "Older dup",
          updatedAt: "2026-03-08T20:00:00Z"
        }),
        buildAttachment({
          run_id: "run_hist_dup",
          query: "Newer dup",
          updatedAt: "2026-03-08T21:30:00Z"
        }),
        buildAttachment({
          run_id: "run_hist_new",
          updatedAt: "2026-03-08T21:45:00Z"
        }),
        buildAttachment({
          run_id: "run_hist_extra",
          updatedAt: "2026-03-08T20:30:00Z"
        })
      ]
    })

    expect(settings?.deepResearchAttachmentHistory?.map((entry) => entry.run_id)).toEqual([
      "run_hist_new",
      "run_hist_dup",
      "run_hist_extra"
    ])
    expect(settings?.deepResearchAttachmentHistory?.[1]?.query).toBe("Newer dup")
  })

  it("merges history by per-entry updatedAt instead of top-level settings updatedAt", () => {
    const local = normalizeChatSettingsRecord({
      schemaVersion: 2,
      updatedAt: "2026-03-08T18:00:00Z",
      deepResearchAttachment: buildAttachment({
        run_id: "run_active_local",
        updatedAt: "2026-03-08T18:00:00Z"
      }),
      deepResearchAttachmentHistory: [
        buildAttachment({
          run_id: "run_hist_shared",
          query: "Shared older",
          updatedAt: "2026-03-08T19:00:00Z"
        }),
        buildAttachment({
          run_id: "run_hist_local_only",
          updatedAt: "2026-03-08T18:30:00Z"
        })
      ]
    })
    const remote = normalizeChatSettingsRecord({
      schemaVersion: 2,
      updatedAt: "2026-03-08T22:00:00Z",
      deepResearchAttachment: buildAttachment({
        run_id: "run_active_remote",
        updatedAt: "2026-03-08T22:00:00Z"
      }),
      deepResearchAttachmentHistory: [
        buildAttachment({
          run_id: "run_hist_shared",
          query: "Shared newer",
          updatedAt: "2026-03-08T21:00:00Z"
        }),
        buildAttachment({
          run_id: "run_hist_remote_only",
          updatedAt: "2026-03-08T20:00:00Z"
        })
      ]
    })

    const merged = mergeChatSettings(local, remote)

    expect(merged?.deepResearchAttachment?.run_id).toBe("run_active_remote")
    expect(merged?.deepResearchAttachmentHistory?.map((entry) => entry.run_id)).toEqual([
      "run_hist_shared",
      "run_hist_remote_only",
      "run_hist_local_only"
    ])
    expect(merged?.deepResearchAttachmentHistory?.[0]?.query).toBe("Shared newer")
  })

  it("fails cleanly when the combined active-plus-history payload exceeds the byte cap", async () => {
    const hugeClaim = "x".repeat(210_000)

    const next = await applyChatSettingsPatch({
      historyId: "history-too-large",
      serverChatId: null,
      patch: {
        deepResearchAttachment: buildAttachment({
          run_id: "run_active_large",
          key_claims: [{ text: hugeClaim }]
        }),
        deepResearchAttachmentHistory: [
          buildAttachment({ run_id: "run_hist_large_1" }),
          buildAttachment({ run_id: "run_hist_large_2" }),
          buildAttachment({ run_id: "run_hist_large_3" })
        ]
      }
    })

    const storageKey = getChatSettingsStorageKey("local:history-too-large")
    expect(next).toBeNull()
    expect(storageState.store.has(storageKey)).toBe(false)
  })
})
