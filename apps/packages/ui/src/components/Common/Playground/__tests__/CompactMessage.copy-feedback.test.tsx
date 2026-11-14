// @vitest-environment jsdom
/**
 * G17 follow-up: the compact row's Copy says when the clipboard refused the
 * write instead of leaving an uncaught rejection and no feedback.
 */
import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { CompactMessage } from "../CompactMessage"

const clipboard = vi.hoisted(() => ({ copy: vi.fn() }))
vi.mock("@/utils/clipboard", () => ({ copyToClipboard: clipboard.copy }))
vi.mock("@/hooks/useTTS", () => ({
  useTTS: () => ({ cancel: vi.fn(), isSpeaking: false, speak: vi.fn() })
}))
vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, defaultValue: unknown) => React.useState(defaultValue)
}))

afterEach(() => clipboard.copy.mockReset())

const renderRow = () =>
  render(
    <CompactMessage message="Saved question" isBot={false} name="You" currentMessageIndex={0} totalMessages={1} />
  )

describe("CompactMessage Copy feedback", () => {
  it("announces Copied once the message is on the clipboard", async () => {
    clipboard.copy.mockResolvedValue(undefined)
    renderRow()
    fireEvent.click(screen.getByRole("button", { name: "Copy" }))
    expect((await screen.findByText("Copied")).closest('[role="status"]')).not.toBeNull()
    expect(clipboard.copy).toHaveBeenCalledWith({ text: "Saved question", formatted: false })
  })

  it("says the copy failed instead of confirming it", async () => {
    clipboard.copy.mockRejectedValue(new Error("Denied"))
    renderRow()
    fireEvent.click(screen.getByRole("button", { name: "Copy" }))
    expect(await screen.findByText("Couldn't copy")).toBeInTheDocument()
    expect(screen.queryByText("Copied")).toBeNull()
    expect(screen.getByRole("button", { name: "Copy" })).toBeInTheDocument()
  })
})
