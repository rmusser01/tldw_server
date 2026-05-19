import React from "react"
import { act, renderHook, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn()
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: mocks.bgRequest
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: <T,>(key: string, initial?: T) =>
    React.useState<T | undefined>(
      (key === "writing:mood-enabled" || key === "writing:echo-enabled")
        ? (true as T)
        : initial
    )
}))

import { useWritingFeedback } from "../useWritingFeedback"

describe("useWritingFeedback", () => {
  beforeEach(() => {
    vi.useRealTimers()
    vi.spyOn(Date, "now").mockReturnValue(60_000)
    mocks.bgRequest.mockReset()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("keeps the accumulated echo characters when no reaction text is returned", async () => {
    mocks.bgRequest.mockResolvedValue({
      choices: [{ message: { content: "" } }]
    })

    const { result, rerender } = renderHook(
      (props: {
        editorText: string
        isOnline: boolean
        isGenerating: boolean
        selectedModel?: string
      }) => useWritingFeedback(props),
      {
        initialProps: {
          editorText: "",
          isOnline: true,
          isGenerating: false,
          selectedModel: "test-model"
        }
      }
    )

    await act(async () => {
      rerender({
        editorText: "a".repeat(500),
        isOnline: true,
        isGenerating: false,
        selectedModel: "test-model"
      })
    })

    await waitFor(() => expect(mocks.bgRequest).toHaveBeenCalledTimes(1))
    await waitFor(() => expect(result.current.echoAnalyzing).toBe(false))

    expect(result.current.charsSinceLastEcho).toBe(500)
    expect(result.current.echoReactions).toEqual([])
  })

  it("ignores late echo responses after the feature is toggled off", async () => {
    let resolveRequest: ((value: unknown) => void) | undefined
    const pendingRequest = new Promise((resolve) => {
      resolveRequest = resolve
    })
    mocks.bgRequest.mockReturnValue(pendingRequest)

    const { result, rerender } = renderHook(
      (props: {
        editorText: string
        isOnline: boolean
        isGenerating: boolean
        selectedModel?: string
      }) => useWritingFeedback(props),
      {
        initialProps: {
          editorText: "",
          isOnline: true,
          isGenerating: false,
          selectedModel: "test-model"
        }
      }
    )

    await act(async () => {
      rerender({
        editorText: "b".repeat(500),
        isOnline: true,
        isGenerating: false,
        selectedModel: "test-model"
      })
    })

    await waitFor(() => expect(result.current.echoAnalyzing).toBe(true))

    await act(async () => {
      result.current.setEchoEnabled(false)
    })

    expect(result.current.echoAnalyzing).toBe(false)

    await act(async () => {
      resolveRequest?.({
        choices: [{ message: { content: "Late reaction" } }]
      })
      await Promise.resolve()
    })

    expect(result.current.echoReactions).toEqual([])
  })
})
