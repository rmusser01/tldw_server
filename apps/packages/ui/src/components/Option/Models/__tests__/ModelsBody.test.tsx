import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { ModelsBody } from "../index"

const mocks = vi.hoisted(() => ({
  fetchChatModels: vi.fn(),
  getOpenAIOAuthStatus: vi.fn(),
  listUserProviderKeys: vi.fn(),
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
    error: vi.fn(),
    info: vi.fn(),
    success: vi.fn()
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
    startOpenAIOAuthAuthorize: vi.fn(),
    refreshOpenAIOAuth: vi.fn(),
    disconnectOpenAIOAuth: vi.fn(),
    switchOpenAICredentialSource: vi.fn(),
  },
  tldwModels: {
    warmCache: vi.fn()
  }
}))

vi.mock("wxt/browser", () => ({
  browser: {
    runtime: {
      sendMessage: vi.fn()
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
        provider: "openai"
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
    expect(screen.getAllByText("2 usable").length).toBeGreaterThan(0)
    expect(screen.getByText("Default provider")).toBeInTheDocument()
    expect(screen.getAllByText("Ollama").length).toBeGreaterThan(0)
    expect(screen.getAllByText("Default model").length).toBeGreaterThan(0)
    expect(screen.getByText("local-a")).toBeInTheDocument()
  })
})
