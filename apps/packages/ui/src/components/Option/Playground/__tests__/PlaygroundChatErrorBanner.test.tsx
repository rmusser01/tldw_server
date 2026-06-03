// @vitest-environment jsdom
import { act, fireEvent, render, renderHook, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
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

  it("uses model-settings recovery when the encoded error is locally recoverable", () => {
    const modelSettingsListener = vi.fn()
    window.addEventListener("tldw:open-model-settings", modelSettingsListener)

    try {
      render(
        <MemoryRouter>
          <PlaygroundChatErrorBanner
            error={{
              summary: "Character chat model setup needs attention.",
              hint: "Open model settings and configure the provider.",
              detail: "provider_not_configured",
              recoveryAction: "open-model-settings",
              recoveryLabel: "Open model settings",
              key: "assistant-error-1:model"
            } as any}
            diagnosticsLabel="Health & diagnostics"
            dismissLabel="Dismiss error"
            onDismiss={() => undefined}
          />
        </MemoryRouter>
      )

      fireEvent.click(
        screen.getByRole("button", { name: "Open model settings" })
      )

      expect(modelSettingsListener).toHaveBeenCalledTimes(1)
      expect(
        screen.queryByRole("button", { name: "Health & diagnostics" })
      ).toBeNull()
    } finally {
      window.removeEventListener("tldw:open-model-settings", modelSettingsListener)
    }
  })

  it("uses the compact model selector for switch-model recovery", () => {
    const modelSelectorListener = vi.fn()
    const modelSettingsListener = vi.fn()
    window.addEventListener("tldw:open-model-selector", modelSelectorListener)
    window.addEventListener("tldw:open-model-settings", modelSettingsListener)

    try {
      render(
        <MemoryRouter>
          <PlaygroundChatErrorBanner
            error={{
              summary: "The selected model is not available.",
              hint: "Choose a different model and try again.",
              detail: "model_not_found",
              recoveryAction: "open-model-selector",
              recoveryLabel: "Choose another model",
              key: "assistant-error-1:model-selector"
            } as any}
            diagnosticsLabel="Health & diagnostics"
            dismissLabel="Dismiss error"
            onDismiss={() => undefined}
          />
        </MemoryRouter>
      )

      fireEvent.click(
        screen.getByRole("button", { name: "Choose another model" })
      )

      expect(modelSelectorListener).toHaveBeenCalledTimes(1)
      expect(modelSettingsListener).not.toHaveBeenCalled()
      expect(
        screen.queryByRole("button", { name: "Health & diagnostics" })
      ).toBeNull()
    } finally {
      window.removeEventListener("tldw:open-model-selector", modelSelectorListener)
      window.removeEventListener("tldw:open-model-settings", modelSettingsListener)
    }
  })

  it("renders inline first-chat recovery actions when supplied", () => {
    const error = {
      summary: "The selected model is not available.",
      hint: "Choose a different model or refresh the model list, then try again.",
      detail: "raw provider detail",
      key: "assistant-error-1:model"
    }
    const onRetry = vi.fn()
    const onEditProvider = vi.fn()
    const onSwitchProvider = vi.fn()
    const onDismiss = vi.fn()

    render(
      <MemoryRouter>
        <PlaygroundChatErrorBanner
          error={error}
          diagnosticsLabel="Health & diagnostics"
          retryLabel="Retry chat"
          editProviderLabel="Edit provider"
          switchProviderLabel="Switch provider"
          dismissLabel="Dismiss error"
          onRetry={onRetry}
          onEditProvider={onEditProvider}
          onSwitchProvider={onSwitchProvider}
          onDismiss={onDismiss}
        />
      </MemoryRouter>
    )

    fireEvent.click(screen.getByTestId("playground-chat-error-retry"))
    fireEvent.click(screen.getByTestId("playground-chat-error-edit-provider"))
    fireEvent.click(screen.getByTestId("playground-chat-error-switch-provider"))
    fireEvent.click(screen.getByRole("button", { name: "Dismiss error" }))

    expect(onRetry).toHaveBeenCalledOnce()
    expect(onEditProvider).toHaveBeenCalledOnce()
    expect(onSwitchProvider).toHaveBeenCalledOnce()
    expect(onDismiss).toHaveBeenCalledWith("assistant-error-1:model")
    expect(screen.queryByTestId("playground-chat-error-skip")).toBeNull()
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

  it("resolves encoded assistant errors loaded from server content fields", () => {
    const latest = getLatestChatErrorBannerEntry([
      {
        id: "server-loaded-error",
        role: "assistant",
        content: encodeError("Server-loaded error", "Retry from composer")
      } as any
    ])

    expect(latest?.summary).toBe("Server-loaded error")
    expect(latest?.hint).toBe("Retry from composer")
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

  it("does not dismiss a newly surfaced error when submit captured no prior error", () => {
    const { result } = renderHook(
      ({ messages }) => usePlaygroundChatErrorBanner(messages),
      {
        initialProps: {
          messages: [
            {
              id: "assistant-error-1",
              isBot: true,
              message: encodeError("Provider failed")
            }
          ]
        }
      }
    )

    act(() => {
      result.current.dismissAfterSuccessfulSubmit(null)
    })

    expect(result.current.visibleError?.summary).toBe("Provider failed")
  })
})
