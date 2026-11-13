// @vitest-environment jsdom
/**
 * G17 follow-up: a refused clipboard write says so instead of leaving an
 * uncaught rejection and no feedback.
 */
import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { CopyButton } from "../CopyButton"

vi.mock("@/services/app", () => ({ isRemoveReasoningTagFromCopy: async () => false }))

const setClipboard = (writeText: unknown) =>
  Object.defineProperty(navigator, "clipboard", { configurable: true, value: { writeText } })

afterEach(() => {
  Reflect.deleteProperty(navigator, "clipboard")
})

describe("Evaluations CopyButton", () => {
  it("announces Copied once the text is on the clipboard", async () => {
    setClipboard(vi.fn().mockResolvedValue(undefined))
    render(<CopyButton text="run-42" showTooltip={false} />)
    fireEvent.click(screen.getByRole("button"))
    expect((await screen.findByText("Copied")).closest('[role="status"]')).not.toBeNull()
  })

  it("says the copy failed instead of confirming it", async () => {
    setClipboard(vi.fn().mockRejectedValue(new Error("Denied")))
    render(<CopyButton text="run-42" showTooltip={false} />)
    fireEvent.click(screen.getByRole("button"))
    expect(await screen.findByText("Couldn't copy")).toBeInTheDocument()
    expect(screen.queryByText("Copied")).toBeNull()
    expect(document.querySelector(".lucide-check")).toBeNull()
  })
})
