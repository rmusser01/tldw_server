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

describe("chat settings deep research pinned attachment", () => {
  beforeEach(() => {
    storageState.store.clear()
    vi.clearAllMocks()
  })

  it("keeps a valid pinned attachment during normalization", () => {
    const settings = normalizeChatSettingsRecord({
      schemaVersion: 2,
      updatedAt: "2026-03-08T22:00:00Z",
      deepResearchPinnedAttachment: buildAttachment({
        run_id: "run_pinned",
        updatedAt: "2026-03-08T21:00:00Z"
      })
    })

    expect(settings?.deepResearchPinnedAttachment).toEqual(
      expect.objectContaining({ run_id: "run_pinned" })
    )
  })

  it("strips malformed pinned attachments during normalization", () => {
    const settings = normalizeChatSettingsRecord({
      schemaVersion: 2,
      updatedAt: "2026-03-08T22:00:00Z",
      deepResearchPinnedAttachment: {
        ...buildAttachment({ run_id: "run_pinned" }),
        unexpected: true
      }
    })

    expect(settings?.deepResearchPinnedAttachment).toBeUndefined()
  })

  it("dedupes history against both active and pinned attachments", () => {
    const settings = normalizeChatSettingsRecord({
      schemaVersion: 2,
      updatedAt: "2026-03-08T22:00:00Z",
      deepResearchAttachment: buildAttachment({
        run_id: "run_active",
        updatedAt: "2026-03-08T22:00:00Z"
      }),
      deepResearchPinnedAttachment: buildAttachment({
        run_id: "run_pinned",
        updatedAt: "2026-03-08T21:30:00Z"
      }),
      deepResearchAttachmentHistory: [
        buildAttachment({
          run_id: "run_active",
          updatedAt: "2026-03-08T21:15:00Z"
        }),
        buildAttachment({
          run_id: "run_pinned",
          updatedAt: "2026-03-08T21:20:00Z"
        }),
        buildAttachment({
          run_id: "run_hist_keep",
          updatedAt: "2026-03-08T21:10:00Z"
        })
      ]
    })

    expect(settings?.deepResearchAttachmentHistory?.map((entry) => entry.run_id)).toEqual([
      "run_hist_keep"
    ])
  })

  it("prefers newer pinned updatedAt during merge and keeps history free of pinned duplicates", () => {
    const local = normalizeChatSettingsRecord({
      schemaVersion: 2,
      updatedAt: "2026-03-08T18:00:00Z",
      deepResearchAttachment: buildAttachment({
        run_id: "run_active_local",
        updatedAt: "2026-03-08T18:00:00Z"
      }),
      deepResearchPinnedAttachment: buildAttachment({
        run_id: "run_pinned",
        query: "Pinned local older",
        updatedAt: "2026-03-08T19:00:00Z"
      }),
      deepResearchAttachmentHistory: [
        buildAttachment({
          run_id: "run_hist_local",
          updatedAt: "2026-03-08T18:30:00Z"
        })
      ]
    })
    const remote = normalizeChatSettingsRecord({
      schemaVersion: 2,
      updatedAt: "2026-03-08T22:00:00Z",
      deepResearchPinnedAttachment: buildAttachment({
        run_id: "run_pinned",
        query: "Pinned remote newer",
        updatedAt: "2026-03-08T21:00:00Z"
      }),
      deepResearchAttachmentHistory: [
        buildAttachment({
          run_id: "run_pinned",
          query: "Pinned duplicate in history",
          updatedAt: "2026-03-08T20:30:00Z"
        }),
        buildAttachment({
          run_id: "run_hist_remote",
          updatedAt: "2026-03-08T20:00:00Z"
        })
      ]
    })

    const merged = mergeChatSettings(local, remote)

    expect(merged?.deepResearchPinnedAttachment?.query).toBe("Pinned remote newer")
    expect(merged?.deepResearchAttachmentHistory?.map((entry) => entry.run_id)).toEqual([
      "run_hist_remote",
      "run_hist_local"
    ])
  })
})
