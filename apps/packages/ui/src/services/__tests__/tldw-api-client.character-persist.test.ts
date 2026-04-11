import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn(),
  bgUpload: vi.fn()
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args),
  bgUpload: (...args: unknown[]) => mocks.bgUpload(...args),
  bgStream: vi.fn()
}))

vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    get: vi.fn(async () => null),
    set: vi.fn(async () => undefined),
    remove: vi.fn(async () => undefined)
  }),
  safeStorageSerde: {
    serialize: (value: unknown) => value,
    deserialize: (value: unknown) => value
  }
}))

import { TldwApiClient } from "@/services/tldw/TldwApiClient"

describe("TldwApiClient character persist cache invalidation", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("invalidates cached chat messages when persist reports saved degraded state", async () => {
    const error = Object.assign(new Error("degraded"), {
      status: 503,
      details: {
        detail: {
          code: "persist_validation_degraded",
          saved: true,
          assistant_message_id: "assistant-server-99"
        }
      }
    })
    mocks.bgRequest.mockRejectedValue(error)

    const client = new TldwApiClient()
    const invalidateSpy = vi.spyOn(client, "invalidateChatMessagesCache")

    await expect(
      client.persistCharacterCompletion("chat-1", {
        assistant_content: "saved degraded reply"
      })
    ).rejects.toBe(error)

    expect(invalidateSpy).toHaveBeenCalledWith("chat-1")
  })

  it("invalidates cached chat messages when persist reports saved degraded state via top-level detail", async () => {
    const error = Object.assign(new Error("degraded"), {
      status: 503,
      detail: {
        code: "persist_validation_degraded",
        saved: true,
        assistant_message_id: "assistant-server-99"
      }
    })
    mocks.bgRequest.mockRejectedValue(error)

    const client = new TldwApiClient()
    const invalidateSpy = vi.spyOn(client, "invalidateChatMessagesCache")

    await expect(
      client.persistCharacterCompletion("chat-1", {
        assistant_content: "saved degraded reply"
      })
    ).rejects.toBe(error)

    expect(invalidateSpy).toHaveBeenCalledWith("chat-1")
  })

  it("does not invalidate chat messages for unrelated persist failures", async () => {
    const error = Object.assign(new Error("broken"), {
      status: 500,
      details: { detail: "server error" }
    })
    mocks.bgRequest.mockRejectedValue(error)

    const client = new TldwApiClient()
    const invalidateSpy = vi.spyOn(client, "invalidateChatMessagesCache")

    await expect(
      client.persistCharacterCompletion("chat-1", {
        assistant_content: "normal failure"
      })
    ).rejects.toBe(error)

    expect(invalidateSpy).not.toHaveBeenCalled()
  })
})
