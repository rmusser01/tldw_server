// @vitest-environment jsdom
/**
 * G17 follow-up: every Copy item in the chat's More menu reports a refused
 * clipboard write as an error instead of an uncaught rejection.
 */
import React from "react"
import { act, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { MoreOptions } from "../MoreOptions"

const io = vi.hoisted(() => ({
  copy: vi.fn(),
  success: vi.fn(),
  error: vi.fn()
}))
vi.mock("@/utils/clipboard", () => ({ copyToClipboard: io.copy }))
vi.mock("@/hooks/useAntdMessage", () => ({
  useAntdMessage: () => ({ success: io.success, error: io.error })
}))
vi.mock("@/utils/compare-metrics", () => ({ trackCompareMetric: vi.fn(async () => undefined) }))
vi.mock("../../Common/ShareModal", () => ({ ShareModal: () => null }))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string | { defaultValue?: string }) =>
      typeof fallback === "string" ? fallback : fallback?.defaultValue ?? key
  })
}))
// Render the menu's copy items as plain buttons; the AntD dropdown is not under test.
vi.mock("antd", async (original) => ({
  ...(await original<typeof import("antd")>()),
  Dropdown: ({ menu, children }: { menu: { items: any[] }; children: React.ReactNode }) => (
    <div>
      {children}
      {menu.items
        .flatMap((item) => item?.children ?? [])
        .filter((item) => String(item.key).startsWith("copy"))
        .map((item) => (
          <button key={item.key} type="button" data-key={item.key} onClick={item.onClick}>
            {item.key}
          </button>
        ))}
    </div>
  )
}))

const messages = [
  { isBot: false, name: "You", message: "Question", sources: [] },
  { isBot: true, name: "Assistant", message: "Answer", sources: [] }
] as any[]

afterEach(() => {
  io.copy.mockReset()
  io.success.mockReset()
  io.error.mockReset()
})

describe("MoreOptions copy items", () => {
  it.each(["copy-text", "copy-as-formatted-text", "copy-markdown", "copy-canonical-markdown"])(
    "%s says the copy failed instead of confirming it",
    async (key) => {
      io.copy.mockRejectedValue(new Error("Denied"))
      render(<MoreOptions messages={messages} historyId="history-1" />)
      await act(async () => {
        fireEvent.click(screen.getByRole("button", { name: key }))
      })
      expect(io.error).toHaveBeenCalledWith("Couldn't copy")
      expect(io.success).not.toHaveBeenCalled()
    }
  )

  it("confirms a copy that happened", async () => {
    io.copy.mockResolvedValue(undefined)
    render(<MoreOptions messages={messages} historyId="history-1" />)
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "copy-text" }))
    })
    expect(io.success).toHaveBeenCalledOnce()
    expect(io.error).not.toHaveBeenCalled()
  })
})
