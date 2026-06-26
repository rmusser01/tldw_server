import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { AvailableModelsList } from "../AvailableModelsList"

const mocks = vi.hoisted(() => ({
  initialize: vi.fn(),
  getModelsMetadata: vi.fn(),
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, defaultValue?: string) => defaultValue ?? _key,
  }),
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: (...args: unknown[]) =>
      (mocks.initialize as (...args: unknown[]) => unknown)(...args),
    getModelsMetadata: (...args: unknown[]) =>
      (mocks.getModelsMetadata as (...args: unknown[]) => unknown)(...args),
  },
}))

vi.mock("@/components/Common/ProviderIcon", () => ({
  ProviderIcons: ({ provider }: { provider: string }) => (
    <span data-testid={`provider-icon-${provider}`} />
  ),
}))

const renderWithQueryClient = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  })

  return render(
    <QueryClientProvider client={queryClient}>
      <AvailableModelsList />
    </QueryClientProvider>
  )
}

describe("AvailableModelsList", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.initialize.mockResolvedValue(undefined)
  })

  it("renders models from object-shaped metadata responses", async () => {
    mocks.getModelsMetadata.mockResolvedValue({
      models: [
        {
          provider: "openai",
          id: "openai/gpt-4o",
          context_length: 128000,
          capabilities: ["vision", "tool_use"],
        },
      ],
      total: 1,
    })

    renderWithQueryClient()

    expect(await screen.findByText("openai/gpt-4o")).toBeInTheDocument()
    expect(screen.getByText("ctx 128000")).toBeInTheDocument()
    expect(
      screen.queryByText("Unable to load models from server")
    ).not.toBeInTheDocument()
  })

  it("treats aborted metadata loads as a non-fatal empty state", async () => {
    const abortError = Object.assign(new Error("The operation was aborted."), {
      name: "AbortError",
      code: "REQUEST_ABORTED",
      status: 0,
    })
    mocks.getModelsMetadata.mockRejectedValue(abortError)

    renderWithQueryClient()

    expect(
      await screen.findByText("No providers available.")
    ).toBeInTheDocument()
    expect(
      screen.queryByText("Unable to load models from server")
    ).not.toBeInTheDocument()
  })

  it("renders metadata load failures through sanitized shared recovery diagnostics", async () => {
    const rawMessage =
      "Provider registry failed at /api/v1/llm/models/metadata?token=sk_models_secret /Users/alice/private/models.json"
    const error = Object.assign(new Error(rawMessage), {
      status: 503,
    })
    mocks.getModelsMetadata.mockRejectedValue(error)

    renderWithQueryClient()

    const recovery = await screen.findByTestId("models-catalog-load-recovery")
    expect(recovery).toHaveAttribute("data-ds-component", "RecoveryCallout")
    expect(
      screen.getByRole("heading", { name: "Unable to load models from server" })
    ).toBeInTheDocument()
    expect(screen.getByText("GET")).toBeInTheDocument()
    expect(screen.getAllByText("[server-endpoint]").length).toBeGreaterThan(0)
    expect(screen.getByText("503")).toBeInTheDocument()
    expect(screen.getByText(/Provider registry failed at/)).toHaveTextContent(
      "[server-endpoint]"
    )
    expect(screen.getByText(/Provider registry failed at/)).toHaveTextContent(
      "[redacted-path]"
    )
    expect(screen.queryByText(/\/api\/v1\/llm\/models/)).not.toBeInTheDocument()
    expect(screen.queryByText(/\/Users\/alice/)).not.toBeInTheDocument()
    expect(screen.queryByText(/sk_models_secret/)).not.toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument()
    expect(screen.queryByText("Request details")).toBeNull()
  })
})
