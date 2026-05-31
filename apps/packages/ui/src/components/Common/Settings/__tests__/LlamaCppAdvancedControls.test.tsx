import React from "react"
import { cleanup, render, screen, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { afterEach, describe, expect, it, vi } from "vitest"

import { LlamaCppAdvancedControls } from "../LlamaCppAdvancedControls"

const mocks = vi.hoisted(() => ({
  getLlmProviders: vi.fn(),
  listGrammars: vi.fn(),
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getLlmProviders: mocks.getLlmProviders,
  },
}))

vi.mock("@/services/tldw/TldwLlamaGrammars", () => ({
  tldwLlamaGrammars: {
    list: mocks.listGrammars,
  },
}))

vi.mock("../LlamaGrammarLibraryModal", () => ({
  LlamaGrammarLibraryModal: () => null,
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, defaultValue?: string) => defaultValue || _key,
  }),
}))

const renderControls = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  })

  return render(
    <QueryClientProvider client={queryClient}>
      <LlamaCppAdvancedControls
        resolvedProvider="llama.cpp"
        grammarMode="none"
        extraBody='{"grammar":"root ::= \"ok\""}'
        onChange={vi.fn()}
      />
    </QueryClientProvider>
  )
}

describe("LlamaCppAdvancedControls", () => {
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it("renders llama.cpp notices through the canonical design-system Alert", async () => {
    mocks.getLlmProviders.mockResolvedValue({
      providers: {
        "llama.cpp": {
          llama_cpp_controls: {
            grammar: {
              supported: false,
              effective_reason: "Grammar support is disabled on this server.",
            },
            thinking_budget: {
              supported: true,
            },
            reserved_extra_body_keys: ["grammar", "grammar_file"],
          },
        },
      },
    })
    mocks.listGrammars.mockResolvedValue({ items: [] })

    const { container } = renderControls()

    await waitFor(() => {
      expect(
        container.querySelectorAll('[data-ds-component="Alert"]')
      ).toHaveLength(2)
    })

    expect(
      screen.getByText("Grammar support is disabled on this server.")
    ).toBeInTheDocument()
    expect(
      screen.getByText(
        "First-class llama.cpp controls override reserved raw extra body keys."
      )
    ).toBeInTheDocument()
    expect(screen.getByText("grammar")).toBeInTheDocument()
  })
})
