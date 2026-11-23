// @vitest-environment jsdom
import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { ModelParamsPanel } from "../ModelParamsPanel"
import { MODEL_PARAMS_SECTIONS_STORAGE_KEY } from "../model-params-sections"
import { useStoreChatModelSettings } from "@/store/model"
import { useUiModeStore } from "@/store/ui-mode"

const mocks = vi.hoisted(() => ({
  getLlmProviders: vi.fn(async () => ({
    providers: {
      "llama.cpp": {
        llama_cpp_controls: {
          grammar: { supported: true },
          thinking_budget: {
            supported: true,
            request_key: "reasoning_budget"
          },
          reserved_extra_body_keys: ["grammar", "reasoning_budget"]
        }
      }
    }
  })),
  grammarList: vi.fn(async () => ({
    items: [
      {
        id: "grammar_1",
        name: "JSON grammar",
        grammar_text: 'root ::= "ok"',
        version: 1
      }
    ],
    total: 1,
    limit: 100,
    offset: 0
  }))
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback || _key
  })
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getLlmProviders: mocks.getLlmProviders
  }
}))

vi.mock("@/services/tldw/TldwLlamaGrammars", () => ({
  tldwLlamaGrammars: {
    list: mocks.grammarList,
    create: vi.fn(),
    update: vi.fn(),
    remove: vi.fn()
  }
}))

describe("ModelParamsPanel llama.cpp controls", () => {
  beforeEach(() => {
    // Sections are remembered (XS-15); each test starts with them closed.
    localStorage.removeItem(MODEL_PARAMS_SECTIONS_STORAGE_KEY)
    vi.clearAllMocks()
    useStoreChatModelSettings.getState().reset()
    useUiModeStore.setState({ mode: "pro" })
  })

  it("shows llama.cpp advanced controls when the selected model resolves to llama.cpp", async () => {
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false }
      }
    })

    render(
      <QueryClientProvider client={queryClient}>
        <ModelParamsPanel selectedModel="llama.cpp/local-model" />
      </QueryClientProvider>
    )

    fireEvent.click(screen.getByRole("button", { name: "Provider & API" }))

    expect(
      await screen.findByText("llama.cpp advanced controls")
    ).toBeInTheDocument()
    expect(screen.getByText("Grammar source")).toBeInTheDocument()
    expect(screen.getByText("Thinking budget")).toBeInTheDocument()
  })

  it("warns when raw extra body contains reserved llama.cpp keys", async () => {
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false }
      }
    })

    useStoreChatModelSettings.getState().updateSettings({
      extraBody: '{"grammar":"root ::= \\"raw\\""}',
      llamaGrammarMode: "inline",
      llamaGrammarInline: 'root ::= "ui"'
    })

    render(
      <QueryClientProvider client={queryClient}>
        <ModelParamsPanel selectedModel="llama.cpp/local-model" />
      </QueryClientProvider>
    )

    fireEvent.click(screen.getByRole("button", { name: "Provider & API" }))

    expect(
      await screen.findByText(
        "First-class llama.cpp controls override reserved raw extra body keys."
      )
    ).toBeInTheDocument()
    expect(screen.getByText("grammar")).toBeInTheDocument()
  })
})
