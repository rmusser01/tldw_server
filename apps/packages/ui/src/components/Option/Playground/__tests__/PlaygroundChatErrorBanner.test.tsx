// @vitest-environment jsdom
import { act, renderHook } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import {
  getLatestChatErrorBannerEntry,
  usePlaygroundChatErrorBanner
} from "../PlaygroundChatErrorBanner"

const encodeError = (summary: string, hint = "Try again") =>
  "__tldw_error__:" +
  JSON.stringify({
    summary,
    hint,
    detail: `${summary} detail`
  })

describe("PlaygroundChatErrorBanner", () => {
  it("resolves the newest encoded assistant error", () => {
    const latest = getLatestChatErrorBannerEntry([
      {
        id: "older",
        isBot: true,
        message: encodeError("Older error")
      },
      {
        id: "user",
        isBot: false,
        message: encodeError("User text should not count")
      },
      {
        id: "newer",
        role: "assistant",
        message: encodeError("Newer error", "Open diagnostics")
      }
    ])

    expect(latest?.summary).toBe("Newer error")
    expect(latest?.hint).toBe("Open diagnostics")
  })

  it("dismisses the current error but shows a later chat error", () => {
    const firstMessages = [
      {
        id: "assistant-error-1",
        isBot: true,
        message: encodeError("First error")
      }
    ]
    const { result, rerender } = renderHook(
      ({ messages }) => usePlaygroundChatErrorBanner(messages),
      {
        initialProps: {
          messages: firstMessages
        }
      }
    )

    expect(result.current.visibleError?.summary).toBe("First error")

    act(() => {
      result.current.dismissAfterSuccessfulSubmit()
    })

    expect(result.current.visibleError).toBeNull()

    rerender({
      messages: [
        ...firstMessages,
        {
          id: "assistant-error-2",
          isBot: true,
          message: encodeError("Second error")
        }
      ]
    })

    expect(result.current.visibleError?.summary).toBe("Second error")

    act(() => {
      result.current.dismissError()
    })

    expect(result.current.visibleError).toBeNull()
  })
})
