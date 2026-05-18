// @vitest-environment jsdom
import { act, renderHook } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { useContextWindow, type UseContextWindowDeps } from "../hooks/useContextWindow"

function buildDeps(overrides: Partial<UseContextWindowDeps> = {}): UseContextWindowDeps {
  return {
    draftTokenCount: 0,
    conversationTokenCount: 0,
    resolvedMaxContext: 8192,
    modelContextLength: 8192,
    numCtx: undefined,
    updateChatModelSetting: vi.fn(),
    selectedCharacter: null,
    systemPrompt: "",
    selectedQuickPrompt: null,
    selectedSystemPrompt: null,
    ragPinnedResults: [],
    messages: [],
    selectedModel: "deepseek-chat",
    resolvedProviderKey: "deepseek",
    deferredComposerInput: "",
    modelCapabilities: {},
    webSearch: false,
    jsonMode: false,
    hasImageAttachment: false,
    measureComposerPerf: (_label, fn) => fn(),
    t: (_key, fallback) => (typeof fallback === "string" ? fallback : _key),
    ...overrides
  }
}

describe("useContextWindow model recommendation cleanup", () => {
  it("does not dispatch cleanup state when recommendation inputs are unstable", () => {
    expect(() => {
      renderHook(() =>
        useContextWindow(
          buildDeps({
            messages: [],
            modelCapabilities: {}
          })
        )
      )
    }).not.toThrow()
  })

  it("cleans stale dismissedRecommendationIds without throwing", () => {
    expect(() => {
      const { result, rerender } = renderHook(
        (deps: UseContextWindowDeps) => useContextWindow(deps),
        {
          initialProps: buildDeps({
            deferredComposerInput: "Return valid JSON.",
            jsonMode: false
          })
        }
      )

      act(() => {
        result.current.dismissModelRecommendation("enable-json-mode")
      })

      rerender(buildDeps({ deferredComposerInput: "", jsonMode: true }))
    }).not.toThrow()
  })
})
