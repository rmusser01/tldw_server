// @vitest-environment jsdom
import fs from "node:fs"
import path from "node:path"
import { renderHook } from "@testing-library/react"
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
  it("guards recommendation cleanup before dispatching dismissal state", () => {
    const source = fs.readFileSync(
      path.resolve(
        __dirname,
        "../hooks/useContextWindow.ts"
      ),
      "utf8"
    )

    expect(source).toContain("if (dismissedRecommendationIds.length === 0) return")
  })

  it("does not dispatch cleanup state when unstable recommendation inputs have no dismissed recommendations", () => {
    expect(() => {
      renderHook(() =>
        useContextWindow(
          buildDeps({
            messages: [],
            modelCapabilities: {}
          })
        )
      )
    }).not.toThrow(/Maximum update depth exceeded/)
  })
})
