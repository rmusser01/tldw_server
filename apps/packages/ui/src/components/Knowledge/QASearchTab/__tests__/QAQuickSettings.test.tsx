import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { describe, expect, it, vi, beforeEach } from "vitest"

import { QAQuickSettings } from "../QAQuickSettings"
import { tldwClient } from "@/services/tldw/TldwApiClient"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback ?? _key,
  }),
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: vi.fn().mockResolvedValue(undefined),
    getProviders: vi.fn().mockResolvedValue({
      default_provider: "openai",
      providers: [
        {
          name: "openai",
          display_name: "OpenAI",
          models: ["gpt-4o-mini", "gpt-4.1"],
          default_model: "gpt-4o-mini",
        },
        {
          name: "anthropic",
          display_name: "Anthropic",
          models: ["claude-3-7-sonnet-20250219"],
          default_model: "claude-3-7-sonnet-20250219",
        },
      ],
    }),
  },
}))

function renderWithQueryClient(ui: React.ReactElement) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  })

  return render(
    <QueryClientProvider client={queryClient}>{ui}</QueryClientProvider>
  )
}

describe("QAQuickSettings", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("renders answer provider/model controls and updates callbacks", async () => {
    const onGenerationProviderChange = vi.fn()
    const onGenerationModelChange = vi.fn()

    renderWithQueryClient(
      <QAQuickSettings
        preset="balanced"
        onPresetChange={vi.fn()}
        strategy="standard"
        onStrategyChange={vi.fn()}
        selectedSources={["media_db"]}
        onSourcesChange={vi.fn()}
        generationProvider={null}
        onGenerationProviderChange={onGenerationProviderChange}
        generationModel=""
        onGenerationModelChange={onGenerationModelChange}
      />
    )

    expect(
      screen.getByRole("combobox", { name: "Answer provider" })
    ).toBeInTheDocument()
    expect(screen.getByRole("combobox", { name: "Answer model" })).toBeInTheDocument()

    fireEvent.mouseDown(screen.getByRole("combobox", { name: "Answer provider" }))
    await screen.findByText("OpenAI")
    fireEvent.click(screen.getByText("Anthropic"))

    expect(onGenerationProviderChange).toHaveBeenCalledWith("anthropic")
    expect(onGenerationModelChange).toHaveBeenCalledWith("claude-3-7-sonnet-20250219")

    const modelInput = screen.getByRole("combobox", { name: "Answer model" })
    fireEvent.change(modelInput, { target: { value: "claude-3-7-sonnet-20250219" } })

    await waitFor(() => {
      expect(onGenerationModelChange).toHaveBeenCalledWith("claude-3-7-sonnet-20250219")
    })
  })
})
