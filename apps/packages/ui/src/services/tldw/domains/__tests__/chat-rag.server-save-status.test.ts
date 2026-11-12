import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn()
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args),
  bgStream: vi.fn(),
  bgUpload: vi.fn()
}))

import {
  getServerChatSaveStatus,
  resetServerChatSaveStatus
} from "@/store/server-chat-save-status"
import type { TldwApiClientCore } from "../../TldwApiClient"
import { chatRagMethods } from "../chat-rag"

const client = {
  invalidateChatMessagesCache: vi.fn(),
  normalizeChatSummary: (value: unknown) => value
} as unknown as TldwApiClientCore

// CS-03 / XS-05 (#3104): every server chat write records whether the server
// acknowledged it, so chat persistence labels can be derived from that outcome.
describe("chat RAG server chat writes record the acknowledged save status", () => {
  beforeEach(() => {
    mocks.bgRequest.mockReset()
    resetServerChatSaveStatus()
  })

  it("marks the chat saving while a message write is in flight and saved once acknowledged", async () => {
    let resolveRequest!: (value: unknown) => void
    mocks.bgRequest.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveRequest = resolve
        })
    )

    const write = chatRagMethods.addChatMessage.call(client, "chat-1", {
      role: "user",
      content: "hello"
    })

    expect(getServerChatSaveStatus("chat-1")).toBe("saving")
    resolveRequest({ id: "m1", version: 1 })
    await expect(write).resolves.toEqual({ id: "m1", version: 1 })
    expect(getServerChatSaveStatus("chat-1")).toBe("saved")
  })

  it("marks the chat failed when the server rejects a message write", async () => {
    const error = new Error("boom")
    mocks.bgRequest.mockRejectedValueOnce(error)

    await expect(
      chatRagMethods.addChatMessage.call(client, "chat-1", {
        role: "user",
        content: "hello"
      })
    ).rejects.toBe(error)

    expect(getServerChatSaveStatus("chat-1")).toBe("failed")
  })

  it("records character completion persistence outcomes", async () => {
    mocks.bgRequest.mockRejectedValueOnce(new Error("persist failed"))
    await expect(
      chatRagMethods.persistCharacterCompletion.call(client, "chat-2", {
        assistant_content: "hi"
      })
    ).rejects.toThrow("persist failed")
    expect(getServerChatSaveStatus("chat-2")).toBe("failed")

    mocks.bgRequest.mockResolvedValueOnce({ assistant_message_id: "a1" })
    await chatRagMethods.persistCharacterCompletion.call(client, "chat-2", {
      assistant_content: "hi"
    })
    expect(getServerChatSaveStatus("chat-2")).toBe("saved")
  })

  it("counts a degraded persist that the server reports as saved as acknowledged", async () => {
    const degraded = Object.assign(new Error("degraded"), {
      status: 503,
      details: {
        detail: { code: "persist_validation_degraded", saved: true }
      }
    })
    mocks.bgRequest.mockRejectedValueOnce(degraded)

    await expect(
      chatRagMethods.persistCharacterCompletion.call(client, "chat-3", {
        assistant_content: "hi"
      })
    ).rejects.toBe(degraded)

    expect(getServerChatSaveStatus("chat-3")).toBe("saved")
  })

  it("records edits and deletes only when the chat is known", async () => {
    mocks.bgRequest.mockRejectedValueOnce(new Error("conflict"))
    await expect(
      chatRagMethods.editMessage.call(client, "m1", "new text", 1, "chat-4")
    ).rejects.toThrow("conflict")
    expect(getServerChatSaveStatus("chat-4")).toBe("failed")

    mocks.bgRequest.mockResolvedValueOnce(undefined)
    await chatRagMethods.deleteMessage.call(client, "m1", 2, "chat-4")
    expect(getServerChatSaveStatus("chat-4")).toBe("saved")

    mocks.bgRequest.mockRejectedValueOnce(new Error("no chat id"))
    await expect(
      chatRagMethods.editMessage.call(client, "m2", "text", 1)
    ).rejects.toThrow("no chat id")
    expect(getServerChatSaveStatus("chat-4")).toBe("saved")
  })
})
