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

  it.each([
    ["string errors", "Provider registry is offline"],
    ["plain object message errors", { message: "Provider registry returned 502" }],
  ])("keeps %s in request details", async (_name, error) => {
    mocks.getModelsMetadata.mockRejectedValue(error)

    renderWithQueryClient()

    expect(
      await screen.findByText("Unable to load models from server")
    ).toBeInTheDocument()
    expect(screen.getByText("Request details")).toBeInTheDocument()
    expect(
      screen.getByText(
        typeof error === "string" ? error : error.message
      )
    ).toBeInTheDocument()
  })
})
