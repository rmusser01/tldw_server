import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { MemoryRouter } from "react-router-dom"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { ModelsBody } from "../index"

const mocks = vi.hoisted(() => ({
  fetchChatModels: vi.fn(),
  getOpenAIOAuthStatus: vi.fn(),
  listUserProviderKeys: vi.fn(),
  startOpenAIOAuthAuthorize: vi.fn(),
  sendMessage: vi.fn(),
  warmCache: vi.fn(),
  notificationError: vi.fn(),
  notificationInfo: vi.fn(),
  notificationSuccess: vi.fn(),
  selectedModelSetter: vi.fn(),
  defaultProviderSetter: vi.fn(),
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string; [key: string]: unknown }
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      let value = fallbackOrOptions?.defaultValue ?? key
      for (const [name, replacement] of Object.entries(fallbackOrOptions ?? {})) {
        if (name === "defaultValue") continue
        value = value.replace(`{{${name}}}`, String(replacement))
      }
      return value
    }
  })
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (key: string) => {
    if (key === "selectedModel") return ["local-a", mocks.selectedModelSetter]
    if (key === "defaultApiProvider") return ["ollama", mocks.defaultProviderSetter]
    return [null, vi.fn()]
  }
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({
    error: (...args: unknown[]) => mocks.notificationError(...args),
    info: (...args: unknown[]) => mocks.notificationInfo(...args),
    success: (...args: unknown[]) => mocks.notificationSuccess(...args)
  })
}))

vi.mock("@/services/tldw-server", () => ({
  fetchChatModels: (...args: unknown[]) => mocks.fetchChatModels(...args)
}))

vi.mock("@/services/tldw", () => ({
  tldwClient: {
    getOpenAIOAuthStatus: (...args: unknown[]) =>
      mocks.getOpenAIOAuthStatus(...args),
    listUserProviderKeys: (...args: unknown[]) =>
      mocks.listUserProviderKeys(...args),
    startOpenAIOAuthAuthorize: (...args: unknown[]) =>
      mocks.startOpenAIOAuthAuthorize(...args),
    refreshOpenAIOAuth: vi.fn(),
    disconnectOpenAIOAuth: vi.fn(),
    switchOpenAICredentialSource: vi.fn(),
  },
  tldwModels: {
    warmCache: (...args: unknown[]) => mocks.warmCache(...args)
  }
}))

vi.mock("wxt/browser", () => ({
  browser: {
    runtime: {
      sendMessage: (...args: unknown[]) => mocks.sendMessage(...args)
    }
  }
}))

vi.mock("../AvailableModelsList", () => ({
  AvailableModelsList: () => <section>Available models</section>
}))

const renderModelsBody = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false
      }
    }
  })

  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        <ModelsBody />
      </MemoryRouter>
    </QueryClientProvider>
  )
}

describe("ModelsBody", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.fetchChatModels.mockResolvedValue([
      {
        model: "local-a",
        nickname: "Local A",
        provider: "ollama"
      },
      {
        model: "remote-b",
        provider: "openai",
        catalog_only: true,
        provider_is_configured: false
      }
    ])
    mocks.getOpenAIOAuthStatus.mockResolvedValue({
      connected: false,
      auth_source: "none"
    })
    mocks.listUserProviderKeys.mockResolvedValue({
      items: [
        {
          provider: "ollama",
          has_key: true,
          source: "user",
          key_hint: "oll-...",
          auth_source: null,
          last_used_at: null
        }
      ]
    })
    mocks.startOpenAIOAuthAuthorize.mockResolvedValue({
      auth_url: "https://example.test/oauth"
    })
    mocks.sendMessage.mockResolvedValue({ ok: true })
    mocks.warmCache.mockResolvedValue(undefined)
  })

  it("puts defaults and provider readiness before the full catalog", async () => {
    renderModelsBody()

    const defaults = await screen.findByText("Set your defaults")
    const readiness = await screen.findByText("Provider readiness")
    const catalog = await screen.findByText("Available models")

    expect(
      defaults.compareDocumentPosition(readiness) & Node.DOCUMENT_POSITION_FOLLOWING
    ).toBeTruthy()
    expect(
      readiness.compareDocumentPosition(catalog) & Node.DOCUMENT_POSITION_FOLLOWING
    ).toBeTruthy()
    expect(screen.getByText("1 configured")).toBeInTheDocument()
    expect(screen.getAllByText("1 usable").length).toBeGreaterThan(0)
    expect(screen.getByText("Default provider")).toBeInTheDocument()
    expect(screen.getAllByText("Ollama").length).toBeGreaterThan(0)
    expect(screen.getAllByText("Default model").length).toBeGreaterThan(0)
    expect(screen.getByText("local-a")).toBeInTheDocument()
  })

  it("uses server model configuration fields when provider-key details are empty", async () => {
    mocks.fetchChatModels.mockResolvedValue([
      {
        model: "configured-server-model",
        provider: "anthropic",
        is_configured: true,
        provider_is_configured: true
      },
      {
        model: "catalog-reference-model",
        provider: "openrouter",
        catalog_only: true,
        is_configured: true,
        provider_is_configured: true
      }
    ])
    mocks.listUserProviderKeys.mockResolvedValue({ items: [] })

    renderModelsBody()

    expect(await screen.findByText("2 configured")).toBeInTheDocument()
    expect(screen.getAllByText("1 usable").length).toBeGreaterThan(0)
  })

  it("does not show configured key counts when provider-key lookup fails", async () => {
    mocks.listUserProviderKeys.mockRejectedValue({ status: 500 })

    renderModelsBody()

    expect(await screen.findByText("Unable to load account keys")).toBeInTheDocument()
  })

  it("sanitizes refresh failure notifications before showing them", async () => {
    const user = userEvent.setup()
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => undefined)
    mocks.sendMessage.mockResolvedValue({ ok: false })
    mocks.warmCache.mockRejectedValue(
      new Error(
        "Request failed: 500 (GET /api/v1/llm/models/metadata) token=sk_secret_inline Authorization: Bearer sk-secret-inline /Users/alice/private/model-cache.json"
      )
    )

    try {
      renderModelsBody()

      await screen.findByText("Set your defaults")
      await user.click(screen.getByRole("button", { name: "Refresh" }))

      await waitFor(() => {
        expect(mocks.notificationError).toHaveBeenCalled()
      })
      const payload = mocks.notificationError.mock.calls.at(-1)?.[0] as {
        description?: string
      }
      const consoleOutput = consoleError.mock.calls
        .flatMap((call) =>
          call.map((entry) => entry instanceof Error ? entry.message : String(entry))
        )
        .join("\n")

      expect(payload.description).toContain("GET [server-endpoint]")
      expect(payload.description).toContain("[redacted-secret]")
      expect(payload.description).toContain("[redacted-path]")
      expect(payload.description).not.toContain("/api/v1/llm/models/metadata")
      expect(payload.description).not.toContain("sk_secret_inline")
      expect(payload.description).not.toContain("sk-secret-inline")
      expect(payload.description).not.toContain("/Users/alice")
      expect(consoleOutput).toContain("GET [server-endpoint]")
      expect(consoleOutput).not.toContain("/api/v1/llm/models/metadata")
      expect(consoleOutput).not.toContain("sk_secret_inline")
      expect(consoleOutput).not.toContain("sk-secret-inline")
      expect(consoleOutput).not.toContain("/Users/alice")
    } finally {
      consoleError.mockRestore()
    }
  })

  it("sanitizes OpenAI OAuth action failure notifications", async () => {
    const user = userEvent.setup()
    mocks.startOpenAIOAuthAuthorize.mockRejectedValue(
      new Error(
        "OAuth failed: POST /api/v1/openai/oauth/authorize api_key=sk-oauth-secret /home/alice/.config/openai.json"
      )
    )

    renderModelsBody()

    await user.click(await screen.findByRole("button", { name: "Connect OpenAI" }))

    await waitFor(() => {
      expect(mocks.notificationError).toHaveBeenCalled()
    })
    const payload = mocks.notificationError.mock.calls.at(-1)?.[0] as {
      description?: string
    }

    expect(payload.description).toContain("POST [server-endpoint]")
    expect(payload.description).toContain("api_key=[redacted-secret]")
    expect(payload.description).toContain("[redacted-path]")
    expect(payload.description).not.toContain("/api/v1/openai/oauth/authorize")
    expect(payload.description).not.toContain("sk-oauth-secret")
    expect(payload.description).not.toContain("/home/alice")
  })
})
