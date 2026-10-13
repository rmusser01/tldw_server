import { act, renderHook } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import enPlayground from "@/assets/locale/en/playground.json"
import {
  beginServerChatWrite,
  resetServerChatSaveStatus
} from "@/store/server-chat-save-status"
import { resolveChatPersistenceKind } from "@/utils/chat-persistence-status"

import { usePersistenceMode } from "../usePersistenceMode"

// Resolve keys against the shipped English locale, because locale JSON
// overrides inline t() defaults at runtime. This asserts the copy users see.
const lookup = (key: string): string | undefined => {
  const [namespace, path] = key.includes(":") ? key.split(":") : ["", key]
  if (namespace !== "playground") return undefined
  const value = path
    .split(".")
    .reduce<unknown>(
      (node, part) =>
        node && typeof node === "object"
          ? (node as Record<string, unknown>)[part]
          : undefined,
      enPlayground
    )
  return typeof value === "string" ? value : undefined
}

const t = (key: string, defaultValue?: unknown) =>
  lookup(key) ?? (typeof defaultValue === "string" ? defaultValue : key)

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t })
}))

const render = (params: {
  temporaryChat?: boolean
  serverChatId?: string | null
}) =>
  renderHook((props: typeof params) =>
    usePersistenceMode({
      temporaryChat: props.temporaryChat ?? false,
      serverChatId: props.serverChatId ?? null
    }),
    { initialProps: params }
  )

describe("usePersistenceMode labels (CS-03 / XS-05, #3104)", () => {
  beforeEach(() => {
    resetServerChatSaveStatus()
  })

  it("labels a local-only chat as saved on this device, even while the server is connected", () => {
    const { result } = render({ serverChatId: null })

    expect(result.current.persistenceKind).toBe("local")
    expect(result.current.persistencePillLabel).toBe("Saved on this device")
    expect(result.current.persistenceModeLabel).toBe(
      "Saved on this device only. This chat is not on your tldw server."
    )
  })

  it("labels a server chat with an acknowledged write as saved on server", () => {
    beginServerChatWrite("chat-1")("saved")

    const { result } = render({ serverChatId: "chat-1" })

    expect(result.current.persistenceKind).toBe("server")
    expect(result.current.persistencePillLabel).toBe("Saved on server")
    expect(result.current.persistenceModeLabel).toBe(
      "Saved on your tldw server and on this device."
    )
  })

  it("labels a chat opened from the server as saved on server before any new write", () => {
    const { result } = render({ serverChatId: "chat-from-history" })

    expect(result.current.persistenceKind).toBe("server")
    expect(result.current.persistencePillLabel).toBe("Saved on server")
  })

  it("does not claim the server while a server write is pending", () => {
    beginServerChatWrite("chat-1")

    const { result } = render({ serverChatId: "chat-1" })

    expect(result.current.persistenceKind).toBe("serverSaving")
    expect(result.current.persistencePillLabel).toBe("Saving to server…")
    expect(result.current.persistencePillLabel).not.toBe("Saved on server")
    expect(result.current.persistenceModeLabel).toBe(
      "Saved on this device. Waiting for your tldw server to confirm the latest changes."
    )
  })

  it("does not claim the server after a server write failed", () => {
    beginServerChatWrite("chat-1")("failed")

    const { result } = render({ serverChatId: "chat-1" })

    expect(result.current.persistenceKind).toBe("serverFailed")
    expect(result.current.persistencePillLabel).toBe("Couldn't save to server")
    expect(result.current.persistenceModeLabel).toBe(
      "Saved on this device. Your tldw server didn't confirm the latest changes."
    )
  })

  it("updates when the pending server write is acknowledged", () => {
    const endWrite = beginServerChatWrite("chat-1")
    const { result } = render({ serverChatId: "chat-1" })
    expect(result.current.persistenceKind).toBe("serverSaving")

    act(() => endWrite("saved"))

    expect(result.current.persistenceKind).toBe("server")
    expect(result.current.persistencePillLabel).toBe("Saved on server")
  })

  it("labels a temporary chat as temporary even if a server id lingers", () => {
    beginServerChatWrite("chat-1")("saved")

    const { result } = render({ temporaryChat: true, serverChatId: "chat-1" })

    expect(result.current.persistenceKind).toBe("temporary")
    expect(result.current.persistencePillLabel).toBe("Temporary")
  })

  it("never uses the 'Locally + Server' jargon in any state", () => {
    const states = [
      { temporaryChat: true },
      { serverChatId: null },
      { serverChatId: "chat-1" }
    ]
    for (const state of states) {
      const { result } = render(state)
      expect(result.current.persistencePillLabel).not.toMatch(/locally/i)
      expect(result.current.persistenceModeLabel).not.toMatch(/locally/i)
    }
  })
})

describe("resolveChatPersistenceKind", () => {
  it.each([
    [{ temporaryChat: true, serverChatId: "c", serverSaveStatus: "saved" }, "temporary"],
    [{ temporaryChat: false, serverChatId: null, serverSaveStatus: "unknown" }, "local"],
    [{ temporaryChat: false, serverChatId: "  ", serverSaveStatus: "saved" }, "local"],
    [{ temporaryChat: false, serverChatId: "c", serverSaveStatus: "unknown" }, "server"],
    [{ temporaryChat: false, serverChatId: "c", serverSaveStatus: "saved" }, "server"],
    [{ temporaryChat: false, serverChatId: "c", serverSaveStatus: "saving" }, "serverSaving"],
    [{ temporaryChat: false, serverChatId: "c", serverSaveStatus: "failed" }, "serverFailed"]
  ] as const)("%o -> %s", (input, expected) => {
    expect(resolveChatPersistenceKind(input)).toBe(expected)
  })
})
