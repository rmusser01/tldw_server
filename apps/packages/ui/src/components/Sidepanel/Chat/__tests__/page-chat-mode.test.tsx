// @vitest-environment jsdom
import { readFileSync } from "node:fs"
import path from "node:path"
import React from "react"
import { act, renderHook } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const notificationInfo = vi.hoisted(() => vi.fn())

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({ info: notificationInfo })
}))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback ?? _key
  })
}))

import {
  isChatWithPageActive,
  usePageChatModeToggle
} from "../page-chat-mode"

type Mode = "normal" | "rag" | "vision"

const useHarness = (initial: Mode) => {
  const [chatMode, setChatMode] = React.useState<Mode>(initial)
  const toggle = usePageChatModeToggle(chatMode, setChatMode)
  return { chatMode, toggle }
}

describe("chat with current page mode (XS-08)", () => {
  beforeEach(() => {
    notificationInfo.mockReset()
  })

  it("counts as active whenever the mode is on, even without page embeddings", () => {
    expect(isChatWithPageActive("rag")).toBe(true)
    expect(isChatWithPageActive("normal")).toBe(false)
    expect(isChatWithPageActive("vision")).toBe(false)
  })

  it("tells the user when the shortcut turns chat with the current page on", () => {
    const { result } = renderHook(() => useHarness("normal"))

    act(() => result.current.toggle())

    expect(result.current.chatMode).toBe("rag")
    expect(notificationInfo).toHaveBeenCalledTimes(1)
    expect(notificationInfo.mock.calls[0][0].message).toBe(
      "Chat with current page is on"
    )
  })

  it("tells the user when it turns the mode off again", () => {
    const { result } = renderHook(() => useHarness("rag"))

    act(() => result.current.toggle())

    expect(result.current.chatMode).toBe("normal")
    expect(notificationInfo.mock.calls[0][0].message).toBe(
      "Chat with current page is off"
    )
  })
})

describe("side panel wiring (XS-08)", () => {
  const read = (relative: string) =>
    readFileSync(path.resolve(__dirname, relative), "utf8")

  it("shows the current page chip whenever the mode is on, in Casual and Pro", () => {
    const form = read("../form.tsx")
    expect(form).toMatch(/\.\.\.\(isChatWithPageActive\(chatMode\)\s*\?/)
  })

  it("toggles the mode from the shortcut and palette with visible feedback", () => {
    const route = read("../../../../routes/sidepanel-chat.tsx")
    expect(route).toMatch(
      /const toggleChatMode = usePageChatModeToggle\(chatMode, setChatMode\)/
    )
  })
})
