import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { AnswerModelMenu } from "../context/AnswerModelMenu"

vi.mock("@/libs/utils", () => ({
  cn: (...args: unknown[]) => args.filter(Boolean).join(" "),
}))

vi.mock("@/utils/provider-registry", () => ({
  getProviderDisplayName: (provider: string) => provider,
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
          models: ["gpt-4o-mini"],
          default_model: "gpt-4o-mini",
        },
      ],
    }),
  },
}))

function createDeferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })
  return { promise, resolve, reject }
}

function renderMenu(overrides: Partial<React.ComponentProps<typeof AnswerModelMenu>> = {}) {
  return render(
    <AnswerModelMenu
      generationProvider={null}
      generationModel={null}
      onGenerationProviderChange={vi.fn()}
      onGenerationModelChange={vi.fn()}
      {...overrides}
    />
  )
}

describe("AnswerModelMenu", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(tldwClient.initialize).mockResolvedValue(undefined)
    vi.mocked(tldwClient.getProviders).mockResolvedValue({
      default_provider: "openai",
      providers: [
        {
          name: "openai",
          display_name: "OpenAI",
          models: ["gpt-4o-mini"],
          default_model: "gpt-4o-mini",
        },
      ],
    })
  })

  it("shows provider loading state while model metadata is loading", () => {
    const providerCatalog = createDeferred<{
      default_provider: string
      providers: Array<{ name: string; display_name: string }>
    }>()
    vi.mocked(tldwClient.getProviders).mockReturnValueOnce(providerCatalog.promise)

    renderMenu()
    fireEvent.click(screen.getByRole("button", { name: "Choose answer model" }))

    expect(screen.getByText("Loading answer providers...")).toBeInTheDocument()

    act(() => {
      providerCatalog.resolve({
        default_provider: "openai",
        providers: [{ name: "openai", display_name: "OpenAI" }],
      })
    })
  })

  it("shows provider error recovery while preserving manual model entry", async () => {
    vi.mocked(tldwClient.getProviders).mockRejectedValueOnce(
      new Error("Provider metadata unavailable")
    )

    const onGenerationModelChange = vi.fn()
    renderMenu({ onGenerationModelChange })

    await waitFor(() => {
      expect(tldwClient.getProviders).toHaveBeenCalled()
    })

    fireEvent.click(screen.getByRole("button", { name: "Choose answer model" }))

    expect(
      screen.getByText(
        "Provider list failed to load. You can still type a model manually."
      )
    ).toBeInTheDocument()
    expect(screen.queryByText("Provider metadata unavailable")).not.toBeInTheDocument()

    fireEvent.change(screen.getByRole("combobox", { name: "Answer model" }), {
      target: { value: "local-manual-model" },
    })

    expect(onGenerationModelChange).toHaveBeenCalledWith("local-manual-model")
  })

  it("loads provider options and suggested models", async () => {
    const onGenerationProviderChange = vi.fn()
    renderMenu({ onGenerationProviderChange })

    await waitFor(() => {
      expect(tldwClient.getProviders).toHaveBeenCalled()
    })

    fireEvent.click(screen.getByRole("button", { name: "Choose answer model" }))
    fireEvent.change(screen.getByRole("combobox", { name: "Answer provider" }), {
      target: { value: "openai" },
    })

    expect(onGenerationProviderChange).toHaveBeenCalledWith("openai")
    expect(screen.getByRole("combobox", { name: "Answer model" })).toHaveAttribute(
      "placeholder",
      "Default: gpt-4o-mini"
    )
  })

  it("clears an explicit provider when Server default is selected", async () => {
    const onGenerationProviderChange = vi.fn()
    renderMenu({
      generationProvider: "openai",
      onGenerationProviderChange,
    })

    await waitFor(() => {
      expect(tldwClient.getProviders).toHaveBeenCalled()
    })

    fireEvent.click(screen.getByRole("button", { name: "Choose answer model" }))
    fireEvent.change(screen.getByRole("combobox", { name: "Answer provider" }), {
      target: { value: "__server_default__" },
    })

    expect(onGenerationProviderChange).toHaveBeenCalledWith(null)
  })

  it("summarizes restored manual model selection in the collapsed control", async () => {
    renderMenu({
      generationProvider: "openai",
      generationModel: "gpt-4o-mini",
    })

    await waitFor(() => {
      expect(tldwClient.getProviders).toHaveBeenCalled()
    })

    expect(screen.getByRole("button", { name: "Choose answer model" })).toHaveAttribute(
      "title",
      "Answer generation uses gpt-4o-mini"
    )
    expect(screen.getByText("AI: gpt-4o-mini")).toBeInTheDocument()
  })
})
