// @vitest-environment jsdom
/**
 * G17 review: copy feedback re-announces every successful copy, and a failed
 * copy stays visible long enough to read its hint.
 */
import React from "react"
import { act, fireEvent, render, renderHook, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { CodeCopyButton } from "../CodeCopyButton"
import { COPY_FAILURE_MS, useCopyFeedback } from "../useCopyFeedback"

afterEach(() => {
  vi.useRealTimers()
  Reflect.deleteProperty(navigator, "clipboard")
})

describe("useCopyFeedback", () => {
  it("keeps a failed copy visible for 4 s even where a success clears sooner", async () => {
    vi.useFakeTimers()
    const { result } = renderHook(() => useCopyFeedback(2000))
    await act(async () => {
      await result.current.run(() => Promise.reject(new Error("Denied")))
    })
    act(() => vi.advanceTimersByTime(2500))
    expect(result.current.failed).toBe(true)
    act(() => vi.advanceTimersByTime(1500))
    expect(result.current.status).toBe("idle")
    expect(COPY_FAILURE_MS).toBe(4000)
  })

  it("still clears a successful copy after the caller's window", async () => {
    vi.useFakeTimers()
    const { result } = renderHook(() => useCopyFeedback(2000))
    await act(async () => {
      await result.current.run(() => Promise.resolve())
    })
    act(() => vi.advanceTimersByTime(2000))
    expect(result.current.status).toBe("idle")
  })
})

describe("CopyStatus announcements", () => {
  it("announces a second successful copy within the window again", async () => {
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText: vi.fn().mockResolvedValue(undefined) }
    })
    render(<CodeCopyButton value="print(1)" label="Copy code" copiedLabel="Copied" />)
    fireEvent.click(screen.getByRole("button", { name: "Copy code" }))
    const first = await screen.findByText("Copied")
    fireEvent.click(screen.getByRole("button", { name: "Copy code" }))
    // A new node in the live region is announced; the same unchanged text is not.
    await waitFor(() => expect(screen.getByText("Copied")).not.toBe(first))
    expect(screen.getByText("Copied").closest('[role="status"]')).not.toBeNull()
  })
})
