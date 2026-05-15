import React from "react"
import { render, screen, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const mockState = vi.hoisted(() => ({
  storageValues: new Map<string, unknown>(),
  queryData: new Map<string, unknown>(),
  registryReadyLabel: "Registry Ready",
  resolveApiProviderForModel: vi.fn(async () => null as string | null),
  streamCalls: [] as Array<{ messages: unknown[]; options: Record<string, unknown> }>,
  sendCalls: [] as Array<{ messages: unknown[]; options: Record<string, unknown> }>
}))

vi.mock("@tanstack/react-query", () => {
  const resolveQueryData = (queryKey: unknown): unknown => {
    const key = Array.isArray(queryKey) ? queryKey[0] : queryKey
    if (key === "writing-session" && Array.isArray(queryKey)) {
      return mockState.queryData.get(
        `writing-session:${String(queryKey[1] || "")}`
      )
    }
    return mockState.queryData.get(String(key))
  }

  return {
    useQuery: ({ queryKey }: { queryKey: unknown }) => ({
      data: resolveQueryData(queryKey),
      isLoading: false,
      isFetching: false,
      error: null
    }),
    useMutation: () => ({
      mutate: vi.fn(),
      mutateAsync: vi.fn(),
      isPending: false
    }),
    useQueryClient: () => ({
      invalidateQueries: vi.fn(),
      setQueryData: vi.fn()
    })
  }
})

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string }
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      if (fallbackOrOptions?.defaultValue) return fallbackOrOptions.defaultValue
      return key
    }
  })
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: <T,>(key: string, initial?: T) =>
    React.useState<T | undefined>(() =>
      mockState.storageValues.has(key)
        ? (mockState.storageValues.get(key) as T)
        : initial
    )
}))

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => true
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    capabilities: { hasChat: true },
    loading: false,
    refresh: async () => {}
  })
}))

vi.mock("@/design-system", async (importActual) => {
  const actual = await importActual<typeof import("@/design-system")>()
  return {
    ...actual,
    READY_STATE_LABEL: mockState.registryReadyLabel
  }
})

vi.mock("@/utils/resolve-api-provider", () => ({
  AUTO_MODEL_ID: "auto",
  resolveApiProviderForModel: mockState.resolveApiProviderForModel
}))

vi.mock("@/components/Common/MarkdownPreview", () => ({
  MarkdownPreview: ({ content }: { content: string }) => <div>{content}</div>
}))

vi.mock("@/services/tldw/TldwChat", () => ({
  TldwChatService: class TldwChatServiceMock {
    cancelStream() {}
    async *streamMessage(
      messages: unknown[],
      options: Record<string, unknown>
    ) {
      mockState.streamCalls.push({ messages, options })
      yield "mocked stream token"
    }
    async sendMessage(messages: unknown[], options: Record<string, unknown>) {
      mockState.sendCalls.push({ messages, options })
      return "mocked completion"
    }
  }
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
  getWritingWordcloud: vi.fn(),
  getWritingSession: vi.fn(),
  importWritingSnapshot: vi.fn(),
  listWritingSessions: vi.fn(),
  listWritingTemplates: vi.fn(),
  listWritingThemes: vi.fn(),
  tokenizeWritingText: vi.fn(),
  updateWritingSession: vi.fn(),
  updateWritingTemplate: vi.fn(),
  updateWritingTheme: vi.fn()
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
    token_count: true
  },
  requested: {
    provider: "openai",
    tokenizer_available: true,
    tokenizer: "mock-tokenizer",
    tokenizer_kind: "mock",
    tokenizer_source: "mock",
    detokenize_available: true,
    features: {
      logprobs: true
    },
    supported_fields: ["top_logprobs"],
    extra_body_compat: {
      effective: true,
      source: "mock",
      notes: "mock"
    }
  }
}

beforeEach(() => {
  mockState.storageValues.clear()
  mockState.queryData.clear()
  mockState.resolveApiProviderForModel.mockReset()
  mockState.resolveApiProviderForModel.mockResolvedValue(null)
  mockState.streamCalls.length = 0
  mockState.sendCalls.length = 0

  mockState.queryData.set("writing-capabilities", DEFAULT_WRITING_CAPABILITIES)
  mockState.queryData.set("writing-defaults", { templates: [], themes: [] })
  mockState.queryData.set("writing-sessions", {
    sessions: [],
    total: 0,
    limit: 200,
    offset: 0
  })
  mockState.queryData.set("writing-templates", {
    templates: [],
    total: 0,
    limit: 200,
    offset: 0
  })
  mockState.queryData.set("writing-themes", {
    themes: [],
    total: 0,
    limit: 200,
    offset: 0
  })
  mockState.queryData.set("writing-session:", null)

  useWritingPlaygroundStore.setState({
    activeSessionId: null,
    activeSessionName: null
  })
})

describe("WritingPlayground topbar design-system state labels", () => {
  it("renders the ready diagnostics label from the design-system registry", () => {
    render(<WritingPlayground />)

    const topbar = screen.getByTestId("writing-playground-topbar")
    expect(within(topbar).getByText(mockState.registryReadyLabel)).toBeInTheDocument()
  })
})
