// @vitest-environment jsdom
import { act, renderHook } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import {
  usePlaygroundRawPreview,
  type UsePlaygroundRawPreviewDeps
} from "../hooks/usePlaygroundRawPreview"

vi.mock("@/utils/resolve-api-provider", () => ({
  resolveApiProviderForModel: vi.fn(async () => "llama.cpp")
}))

describe("PlaygroundForm llama.cpp controls guard", () => {
  it.each(["library", "inline", "none"])(
    "keeps first-class llama.cpp fields in the %s preview request payload",
    async (grammarMode) => {
      const deps: UsePlaygroundRawPreviewDeps = {
        composerModels: [{ id: "llama.cpp:local", capabilities: [] }],
        selectedModel: "llama.cpp:local",
        compareModeActive: false,
        compareSelectedModels: [],
        compareMaxModels: 4,
        currentChatModelSettings: {
          apiProvider: "llama.cpp",
          llamaThinkingBudgetTokens: 256,
          llamaGrammarMode: grammarMode,
          llamaGrammarId: "json-grammar",
          llamaGrammarInline: 'root ::= "yes" | "no"',
          llamaGrammarOverride: 'root ::= "confirmed"'
        },
        history: [],
        systemPrompt: undefined,
        hasMcp: false,
        mcpHealthState: "unavailable",
        mcpTools: [],
        toolChoice: "auto",
        temporaryChat: true,
        serverChatId: null,
        serverChatState: null,
        serverChatSource: null,
        selectedCharacter: null,
        messageSteeringMode: "none",
        messageSteeringForceNarrate: false,
        ragMediaIds: null,
        selectedKnowledge: null,
        contextFiles: [],
        documentContext: [],
        selectedDocuments: [],
        imageBackendDefaultTrimmed: "",
        resolveSubmissionIntent: (message) => ({ message, isImageCommand: false }),
        formImage: "",
        formMessage: "Answer yes or no",
        notificationApi: { error: vi.fn() },
        t: (key, fallback) => typeof fallback === "string" ? fallback : key,
        setToolsPopoverOpen: vi.fn()
      }
      const { result } = renderHook(() => usePlaygroundRawPreview(deps))

      await act(async () => {
        await result.current.refreshRawRequestSnapshot()
      })

      expect(result.current.rawRequestSnapshot).toEqual(expect.objectContaining({
        endpoint: "/api/v1/chat/completions",
        body: expect.objectContaining({
          api_provider: "llama.cpp",
          thinking_budget_tokens: 256,
          grammar_mode: grammarMode,
          grammar_id: "json-grammar",
          grammar_inline: 'root ::= "yes" | "no"',
          grammar_override: 'root ::= "confirmed"'
        })
      }))
      expect(deps.notificationApi.error).not.toHaveBeenCalled()
    }
  )
})
