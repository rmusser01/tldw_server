import { beforeEach, describe, expect, it } from "vitest"

import {
  beginServerChatWrite,
  getServerChatSaveStatus,
  resetServerChatSaveStatus,
  trackServerChatWrite
} from "@/store/server-chat-save-status"

// CS-03 / XS-05 (#3104): the "saved on server" label may only be shown once the
// server has acknowledged the latest write for a chat. This store records the
// outcome of every server chat write so labels can be derived from it.
describe("server chat save status", () => {
  beforeEach(() => {
    resetServerChatSaveStatus()
  })

  it("is unknown for a chat with no recorded writes", () => {
    expect(getServerChatSaveStatus("chat-1")).toBe("unknown")
    expect(getServerChatSaveStatus(null)).toBe("unknown")
  })

  it("is saving while a write is in flight and saved once it is acknowledged", async () => {
    let resolveWrite!: (value: { id: string }) => void
    const write = trackServerChatWrite(
      "chat-1",
      () =>
        new Promise<{ id: string }>((resolve) => {
          resolveWrite = resolve
        })
    )

    expect(getServerChatSaveStatus("chat-1")).toBe("saving")

    resolveWrite({ id: "message-1" })
    await expect(write).resolves.toEqual({ id: "message-1" })
    expect(getServerChatSaveStatus("chat-1")).toBe("saved")
  })

  it("is failed when the server rejects the write, and rethrows the error", async () => {
    const error = new Error("server unavailable")

    await expect(
      trackServerChatWrite("chat-1", () => Promise.reject(error))
    ).rejects.toBe(error)

    expect(getServerChatSaveStatus("chat-1")).toBe("failed")
  })

  it("treats an error the caller recognises as already saved as acknowledged", async () => {
    const savedError = Object.assign(new Error("saved, but degraded"), {
      saved: true
    })

    await expect(
      trackServerChatWrite("chat-1", () => Promise.reject(savedError), {
        isAcknowledgedError: (error) =>
          Boolean((error as { saved?: boolean }).saved)
      })
    ).rejects.toBe(savedError)

    expect(getServerChatSaveStatus("chat-1")).toBe("saved")
  })

  it("stays saving until every overlapping write settles", () => {
    const endFirst = beginServerChatWrite("chat-1")
    const endSecond = beginServerChatWrite("chat-1")

    endFirst("saved")
    expect(getServerChatSaveStatus("chat-1")).toBe("saving")

    endSecond("saved")
    expect(getServerChatSaveStatus("chat-1")).toBe("saved")
  })

  it("reports the latest settled outcome so a later success clears a failure", () => {
    beginServerChatWrite("chat-1")("failed")
    expect(getServerChatSaveStatus("chat-1")).toBe("failed")

    beginServerChatWrite("chat-1")("saved")
    expect(getServerChatSaveStatus("chat-1")).toBe("saved")
  })

  it("keeps the previous outcome when a write ends without a known result", () => {
    beginServerChatWrite("chat-1")("failed")
    beginServerChatWrite("chat-1")("unknown")

    expect(getServerChatSaveStatus("chat-1")).toBe("failed")
  })

  it("ignores a second end call for the same write", () => {
    const endFirst = beginServerChatWrite("chat-1")
    const endSecond = beginServerChatWrite("chat-1")

    endFirst("saved")
    endFirst("saved")

    expect(getServerChatSaveStatus("chat-1")).toBe("saving")
    endSecond("failed")
    expect(getServerChatSaveStatus("chat-1")).toBe("failed")
  })

  it("keeps chats independent and normalises ids", () => {
    beginServerChatWrite(42)("failed")
    beginServerChatWrite(" chat-2 ")("saved")

    expect(getServerChatSaveStatus("42")).toBe("failed")
    expect(getServerChatSaveStatus("chat-2")).toBe("saved")
    expect(getServerChatSaveStatus("chat-3")).toBe("unknown")
  })
})
