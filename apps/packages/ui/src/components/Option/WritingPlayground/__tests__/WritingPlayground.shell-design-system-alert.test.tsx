import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

type QueryResult = {
  data?: unknown
  error?: unknown
  isLoading?: boolean
  isFetching?: boolean
}

const mockState = vi.hoisted(() => ({
  storageValues: new Map<string, unknown>(),
  queryResults: new Map<string, QueryResult>(),
  serverOnline: true,
  serverCapabilities: { hasChat: true },
  registryReadyLabel: "Registry Ready",
  resolveApiProviderForModel: vi.fn(async () => null as string | null),
  streamCalls: [] as Array<{ messages: unknown[]; options: Record<string, unknown> }>,
  sendCalls: [] as Array<{ messages: unknown[]; options: Record<string, unknown> }>,
}))

const serializeQueryKey = (queryKey: unknown): string =>
  JSON.stringify(Array.isArray(queryKey) ? queryKey : [queryKey])

const setQueryResult = (queryKey: unknown, result: QueryResult) => {
  mockState.queryResults.set(serializeQueryKey(queryKey), result)
}

vi.mock("@tanstack/react-query", () => ({
  useQuery: ({
    queryKey,
    enabled = true,
  }: {
    queryKey: unknown
    enabled?: boolean
  }) => {
    if (enabled === false) {
      return {
        data: undefined,
        isLoading: false,
        isFetching: false,
        error: null,
      }
    }

    const result = mockState.queryResults.get(serializeQueryKey(queryKey)) ?? {}
    return {
      data: result.data,
      isLoading: result.isLoading ?? false,
      isFetching: result.isFetching ?? false,
      error: result.error ?? null,
    }
  },
  useMutation: () => ({
    mutate: vi.fn(),
    mutateAsync: vi.fn(),
    isPending: false,
  }),
  useQueryClient: () => ({
    invalidateQueries: vi.fn(),
    setQueryData: vi.fn(),
  }),
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string },
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      if (fallbackOrOptions?.defaultValue) return fallbackOrOptions.defaultValue
      return key
    },
  }),
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: <T,>(key: string, initial?: T) =>
    React.useState<T | undefined>(() =>
      mockState.storageValues.has(key)
        ? (mockState.storageValues.get(key) as T)
        : initial,
    ),
}))

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => mockState.serverOnline,
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    capabilities: mockState.serverCapabilities,
    loading: false,
    refresh: async () => {},
  }),
}))

vi.mock("@/design-system", async (importActual) => {
  const actual = await importActual<typeof import("@/design-system")>()
  return {
    ...actual,
    READY_STATE_LABEL: mockState.registryReadyLabel,
  }
})

vi.mock("@/utils/resolve-api-provider", () => ({
  AUTO_MODEL_ID: "auto",
  resolveApiProviderForModel: mockState.resolveApiProviderForModel,
}))

vi.mock("@/components/Common/MarkdownPreview", () => ({
  MarkdownPreview: ({ content }: { content: string }) => <div>{content}</div>,
}))

vi.mock("@/services/tldw/TldwChat", () => ({
  TldwChatService: class TldwChatServiceMock {
    cancelStream() {}
    async *streamMessage(
      messages: unknown[],
      options: Record<string, unknown>,
    ) {
      mockState.streamCalls.push({ messages, options })
      yield "mocked stream token"
    }
    async sendMessage(messages: unknown[], options: Record<string, unknown>) {
      mockState.sendCalls.push({ messages, options })
      return "mocked completion"
    }
  },
}))

vi.mock("@/services/writing-playground", () => ({
  cloneWritingSession: vi.fn(),
  createWritingSession: vi.fn(),
  createWritingTemplate: vi.fn(),
  createWritingTheme: vi.fn(),
  createWritingWordcloud: vi.fn(),
  countWritingTokens: vi.fn(),
  deleteWritingSession: vi.fn(),
  deleteWritingTemplate: vi.fn(),
  deleteWritingTheme: vi.fn(),
  exportWritingSnapshot: vi.fn(),
  getWritingCapabilities: vi.fn(),
  getWritingDefaults: vi.fn(),
  getWritingSession: vi.fn(),
  getWritingWordcloud: vi.fn(),
  importWritingSnapshot: vi.fn(),
  listWritingSessions: vi.fn(),
  listWritingTemplates: vi.fn(),
  listWritingThemes: vi.fn(),
  tokenizeWritingText: vi.fn(),
  updateWritingSession: vi.fn(),
  updateWritingTemplate: vi.fn(),
  updateWritingTheme: vi.fn(),
}))

import { WritingPlayground } from "../index"
import { useWritingPlaygroundStore } from "@/store/writing-playground"

const DEFAULT_WRITING_CAPABILITIES = {
  server: {
    sessions: true,
    templates: true,
    themes: true,
    defaults_catalog: false,
    snapshots: false,
    tokenize: true,
    token_count: true,
  },
  requested: {
    provider: "openai",
    tokenizer_available: true,
    tokenizer: "mock-tokenizer",
    tokenizer_kind: "mock",
    tokenizer_source: "mock",
    detokenize_available: true,
    features: {
      logprobs: true,
    },
    supported_fields: ["top_logprobs"],
    extra_body_compat: {
      effective: true,
      source: "mock",
      notes: "mock",
    },
  },
}

const seedDefaultQueries = () => {
  setQueryResult(["writing-capabilities"], {
    data: DEFAULT_WRITING_CAPABILITIES,
  })
  setQueryResult(["writing-defaults"], {
    data: { templates: [], themes: [] },
  })
  setQueryResult(["writing-sessions"], {
    data: {
      sessions: [],
      total: 0,
      limit: 200,
      offset: 0,
    },
  })
  setQueryResult(["writing-templates"], {
    data: {
      templates: [],
      total: 0,
      limit: 200,
      offset: 0,
    },
  })
  setQueryResult(["writing-themes"], {
    data: {
      themes: [],
      total: 0,
      limit: 200,
      offset: 0,
    },
  })
  setQueryResult(["writing-session", null], { data: null })
}

const seedActiveSession = () => {
  useWritingPlaygroundStore.setState({
    activeSessionId: "session-auto",
    activeSessionName: "Auto Session",
  })
  setQueryResult(["writing-sessions"], {
    data: {
      sessions: [
        {
          id: "session-auto",
          name: "Auto Session",
          last_modified: "2026-03-16T12:00:00Z",
          version: 1,
        },
      ],
      total: 1,
      limit: 200,
      offset: 0,
    },
  })
}

const seedActiveSessionDetail = (
  payloadOverrides: Record<string, unknown> = {},
) => {
  seedActiveSession()
  setQueryResult(["writing-session", "session-auto"], {
    data: {
      id: "session-auto",
      name: "Auto Session",
      payload: {
        prompt: "Seed prompt",
        settings: {},
        template_name: null,
        theme_name: null,
        chat_mode: false,
        ...payloadOverrides,
      },
      schema_version: 1,
      version_parent_id: null,
      created_at: "2026-03-16T12:00:00Z",
      last_modified: "2026-03-16T12:00:00Z",
      deleted: false,
      client_id: "test-client",
      version: 1,
    },
  })
}

const seedRequestedCapabilities = (
  requestedOverrides: Record<string, unknown>,
) => {
  mockState.storageValues.set("selectedModel", "mock-model")
  setQueryResult(["writing-capabilities", "requested", "mock-model", ""], {
    data: {
      ...DEFAULT_WRITING_CAPABILITIES,
      requested: {
        ...DEFAULT_WRITING_CAPABILITIES.requested,
        ...requestedOverrides,
        features: {
          ...DEFAULT_WRITING_CAPABILITIES.requested.features,
          ...((requestedOverrides.features as Record<string, unknown> | undefined) ?? {}),
        },
        extra_body_compat: {
          ...DEFAULT_WRITING_CAPABILITIES.requested.extra_body_compat,
          ...((requestedOverrides.extra_body_compat as Record<string, unknown> | undefined) ?? {}),
        },
      },
    },
  })
}

const openSettingsPanel = () => {
  fireEvent.click(screen.getByLabelText("Toggle settings"))
}

describe("WritingPlayground shell product-state alerts", () => {
  beforeEach(() => {
    mockState.storageValues.clear()
    mockState.queryResults.clear()
    mockState.serverOnline = true
    mockState.serverCapabilities = { hasChat: true }
    mockState.resolveApiProviderForModel.mockReset()
    mockState.resolveApiProviderForModel.mockResolvedValue(null)
    mockState.streamCalls.length = 0
    mockState.sendCalls.length = 0
    seedDefaultQueries()

    useWritingPlaygroundStore.setState({
      activeSessionId: null,
      activeSessionName: null,
    })
  })

  it("renders the offline shell state through the design-system Alert", () => {
    mockState.serverOnline = false

    render(<WritingPlayground />)

    const offlineTitle = screen.getByText("Server required")
    expect(offlineTitle.closest('[data-ds-component="Alert"]')).toBeInTheDocument()
  })

  it("renders the unsupported shell state through the design-system Alert", () => {
    setQueryResult(["writing-capabilities"], {
      data: {
        ...DEFAULT_WRITING_CAPABILITIES,
        server: { ...DEFAULT_WRITING_CAPABILITIES.server, sessions: false },
      },
    })

    render(<WritingPlayground />)

    const unsupportedTitle = screen.getByText("Playground unavailable")
    expect(
      unsupportedTitle.closest('[data-ds-component="Alert"]'),
    ).toBeInTheDocument()
  })

  it("renders the sessions-load error through the design-system Alert", () => {
    setQueryResult(["writing-sessions"], {
      data: undefined,
      error: new Error("Session list unavailable"),
    })

    render(<WritingPlayground />)

    const sessionsError = screen.getByText("Unable to load sessions.")
    expect(sessionsError.closest('[data-ds-component="Alert"]')).toBeInTheDocument()
  })

  it("renders the active-session editor error through the design-system Alert", () => {
    seedActiveSession()
    setQueryResult(["writing-session", "session-auto"], {
      data: undefined,
      error: new Error("Session detail unavailable"),
    })

    render(<WritingPlayground />)

    const editorError = screen.getByText("Unable to load this session.")
    expect(editorError.closest('[data-ds-component="Alert"]')).toBeInTheDocument()
  })

  it("renders the logprobs unavailable advanced-settings state through the design-system Alert", () => {
    seedActiveSessionDetail()
    seedRequestedCapabilities({
      features: { logprobs: false },
    })

    render(<WritingPlayground />)
    openSettingsPanel()

    const logprobsAlert = screen.getByText(
      "Logprobs are not advertised for this provider/model.",
    )
    expect(logprobsAlert.closest('[data-ds-component="Alert"]')).toBeInTheDocument()
  })

  it("renders the unsupported advanced sampler state through the design-system Alert", () => {
    seedActiveSessionDetail()
    seedRequestedCapabilities({
      extra_body_compat: {
        supported: false,
        known_params: ["logit_bias"],
        effective_reason: "Advanced sampler controls are disabled for this runtime.",
      },
    })

    render(<WritingPlayground />)
    openSettingsPanel()

    const unsupportedAlert = screen.getByText(
      "Advanced sampler controls are disabled for this runtime.",
    )
    expect(
      unsupportedAlert.closest('[data-ds-component="Alert"]'),
    ).toBeInTheDocument()
  })

  it("renders the logit-bias validation error through the design-system Alert", () => {
    seedActiveSessionDetail()
    seedRequestedCapabilities({
      extra_body_compat: {
        supported: true,
        known_params: ["logit_bias"],
      },
    })

    render(<WritingPlayground />)
    openSettingsPanel()
    fireEvent.click(screen.getByText("Advanced sampler controls (extra_body)"))
    fireEvent.change(screen.getByPlaceholderText('{"50256": -100, "198": -1.5}'), {
      target: { value: "{" },
    })

    const logitBiasAlert = screen.getByText(/Invalid JSON:/)
    expect(logitBiasAlert.closest('[data-ds-component="Alert"]')).toBeInTheDocument()
  })
})
