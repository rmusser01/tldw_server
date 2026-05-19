// @vitest-environment jsdom
import { act, fireEvent, render, renderHook, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { MemoryRouter, useLocation } from "react-router-dom"

import {
  getChatErrorBannerScanSignature,
  getLatestChatErrorBannerEntry,
  PlaygroundChatErrorBanner,
  usePlaygroundChatErrorBanner
} from "../PlaygroundChatErrorBanner"

const encodeError = (
  summary: string,
  hint = "Try again",
  detail = `${summary} detail`
) =>
  "__tldw_error__:" +
  JSON.stringify({
    summary,
    hint,
    detail
  })

describe("PlaygroundChatErrorBanner", () => {
  it("renders chat errors through the shared RecoveryCallout primitive", () => {
    const error = {
      ...JSON.parse(
        JSON.stringify({
          summary: "Generation failed",
          hint: "Open diagnostics for details",
          detail: "server detail"
        })
      ),
      key: "assistant-error-1:abc"
    }

    const LocationProbe = () => {
      const location = useLocation()
      return <div data-testid="current-location">{location.pathname}</div>
    }

    render(
      <MemoryRouter>
        <PlaygroundChatErrorBanner
          error={error}
          diagnosticsLabel="Health & diagnostics"
          dismissLabel="Dismiss error"
          onDismiss={() => undefined}
        />
        <LocationProbe />
      </MemoryRouter>
    )

    const banner = screen.getByTestId("playground-chat-error-banner")

    expect(banner).toHaveAttribute("data-ds-component", "RecoveryCallout")
    expect(banner).toHaveAttribute("role", "alert")
    expect(banner).toHaveTextContent("Generation failed")
    expect(banner).toHaveTextContent("Open diagnostics for details")
    fireEvent.click(screen.getByRole("button", { name: "Health & diagnostics" }))
    expect(screen.getByTestId("current-location")).toHaveTextContent("/settings/health")
    expect(screen.getByRole("button", { name: "Dismiss error" })).toBeInTheDocument()
  })

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

  it("uses compact dismissal keys without embedding the encoded payload", () => {
    const detail = "server detail: " + "x".repeat(2048)
    const latest = getLatestChatErrorBannerEntry([
      {
        id: "assistant-error-1",
        isBot: true,
        message: encodeError("Long detail error", "Open diagnostics", detail)
      }
    ])

    expect(latest?.key).toMatch(/^assistant-error-1:/)
    expect(latest?.key).not.toContain(detail)
    expect(latest?.key.length).toBeLessThan(80)
  })

  it("keeps the scan signature stable for same-message non-error streaming updates", () => {
    const firstSignature = getChatErrorBannerScanSignature([
      {
        id: "assistant-streaming",
        role: "assistant",
        message: "partial"
      }
    ])
    const nextSignature = getChatErrorBannerScanSignature([
      {
        id: "assistant-streaming",
        role: "assistant",
        message: "partial response with more tokens"
      }
    ])
    const errorSignature = getChatErrorBannerScanSignature([
      {
        id: "assistant-streaming",
        role: "assistant",
        message: encodeError("Stream failed")
      }
    ])

    expect(nextSignature).toBe(firstSignature)
    expect(errorSignature).not.toBe(firstSignature)
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

  it("dismisses the captured submit error without hiding a newer error", () => {
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

    const capturedSubmitErrorKey = result.current.visibleError?.key
    expect(capturedSubmitErrorKey).toBeTruthy()

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

    act(() => {
      result.current.dismissAfterSuccessfulSubmit(capturedSubmitErrorKey)
    })

    expect(result.current.visibleError?.summary).toBe("Second error")
  })
})
