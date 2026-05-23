import React from "react"
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within
} from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const mockState = vi.hoisted(() => ({
  storageValues: new Map<string, unknown>(),
  queryData: new Map<string, unknown>(),
  queryKey: (queryKey: unknown) =>
    JSON.stringify(Array.isArray(queryKey) ? queryKey : [queryKey]),
  resolveApiProviderForModel: vi.fn(async () => null as string | null),
  streamCalls: [] as Array<{ messages: unknown[]; options: Record<string, unknown> }>,
  sendCalls: [] as Array<{ messages: unknown[]; options: Record<string, unknown> }>,
  sendResponses: [] as string[]
}))

type MockQueryResult = {
  data: unknown
  isLoading: boolean
  isFetching: boolean
  error: unknown
}

vi.mock("@tanstack/react-query", () => {
  const resolveQueryData = (queryKey: unknown): unknown => {
    return mockState.queryData.get(mockState.queryKey(queryKey))
  }

  return {
    useQuery: ({
      queryKey,
      enabled = true
    }: {
      queryKey: unknown
      enabled?: boolean
    }) => ({
      data: enabled === false ? undefined : resolveQueryData(queryKey),
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

vi.mock("@plasmohq/storage/hook", () => {
  return {
    useStorage: <T,>(key: string, initial?: T) =>
      React.useState<T | undefined>(() =>
        mockState.storageValues.has(key)
          ? (mockState.storageValues.get(key) as T)
          : initial
      )
  }
})

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
      return mockState.sendResponses.shift() ?? "mocked completion"
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

vi.mock("../WritingTipTapEditor", () => ({
  WritingTipTapEditor: ({
    onAdapterReady,
    onContentChange,
    placeholder
  }: {
    onAdapterReady: (adapter: {
      getSelection: () => { start: number; end: number }
      setSelection: (selection: { start: number; end: number }) => void
      getSelectedText: (currentValue: string) => string
      focus: () => void
    }) => void
    onContentChange: (json: Record<string, unknown>, plain: string) => void
    placeholder?: string
  }) => {
    const [selection, setSelection] = React.useState({ start: 0, end: 0 })

    React.useEffect(() => {
      onAdapterReady({
        getSelection: () => selection,
        setSelection,
        getSelectedText: (currentValue: string) =>
          currentValue.slice(selection.start, selection.end),
        focus: () => {}
      })
    }, [onAdapterReady, selection])

    return (
      <textarea
        aria-label="Mock rich editor"
        placeholder={placeholder}
        onChange={(event) =>
          onContentChange({ type: "doc" }, event.target.value)
        }
        onSelect={(event) => {
          const node = event.currentTarget
          setSelection({
            start: node.selectionStart,
            end: node.selectionEnd
          })
        }}
      />
    )
  }
}))

import { WritingPlayground } from "../index"
import { WRITING_REVISION_PRESETS } from "../writing-revision-presets"
import { useStoreChatModelSettings } from "@/store/model"
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

const structuredReplacement = (replacement: string, title = "Rewrite selection") =>
  JSON.stringify({
    title,
    replacement,
    rationale: "Clearer and more direct."
  })

const structuredAdvice = (rawText: string, title = "Outline advice") =>
  JSON.stringify({
    title,
    rawText,
    rationale: "This is an advisory planning pass."
  })

const seedWritingSession = (
  payloadOverrides: Record<string, unknown> = {}
) => {
  useWritingPlaygroundStore.setState({
    activeSessionId: "session-auto",
    activeSessionName: "Auto Session"
  })
  mockState.queryData.set(mockState.queryKey(["writing-sessions"]), {
    sessions: [
      {
        id: "session-auto",
        name: "Auto Session",
        last_modified: "2026-03-16T12:00:00Z",
        version: 1
      }
    ],
    total: 1,
    limit: 200,
    offset: 0
  })
  mockState.queryData.set("writing-session:session-auto", {
    id: "session-auto",
    name: "Auto Session",
    payload: {
      prompt: "Seed prompt",
      settings: {},
      template_name: null,
      theme_name: null,
      chat_mode: false,
      ...payloadOverrides
    },
    schema_version: 1,
    version_parent_id: null,
    created_at: "2026-03-16T12:00:00Z",
    last_modified: "2026-03-16T12:00:00Z",
    deleted: false,
    client_id: "test-client",
    version: 1
  })
}

const getEditor = () =>
  screen.getByPlaceholderText("Start writing your prompt...") as HTMLTextAreaElement

const selectEditorText = (editor: HTMLTextAreaElement, selectedText: string) => {
  const start = editor.value.indexOf(selectedText)
  expect(start).toBeGreaterThanOrEqual(0)
  editor.focus()
  editor.setSelectionRange(start, start + selectedText.length)
  fireEvent.select(editor)
}

const latestRevisionPrompt = () => {
  const lastCall = mockState.sendCalls.at(-1)
  expect(lastCall).toBeTruthy()
  const userMessage = (lastCall?.messages as Array<{ content?: unknown }>).find(
    (message) => typeof message.content === "string"
  )
  expect(userMessage?.content).toEqual(expect.any(String))
  return userMessage?.content as string
}

beforeEach(() => {
  mockState.storageValues.clear()
  mockState.queryData.clear()
  mockState.resolveApiProviderForModel.mockReset()
  mockState.resolveApiProviderForModel.mockResolvedValue(null)
  mockState.streamCalls.length = 0
  mockState.sendCalls.length = 0
  mockState.sendResponses.length = 0

  mockState.queryData.set(
    mockState.queryKey(["writing-capabilities"]),
    DEFAULT_WRITING_CAPABILITIES
  )
  mockState.queryData.set(
    mockState.queryKey(["writing-defaults"]),
    { templates: [], themes: [] }
  )
  mockState.queryData.set(mockState.queryKey(["writing-sessions"]), {
    sessions: [],
    total: 0,
    limit: 200,
    offset: 0
  })
  mockState.queryData.set(mockState.queryKey(["writing-templates"]), {
    templates: [],
    total: 0,
    limit: 200,
    offset: 0
  })
  mockState.queryData.set(mockState.queryKey(["writing-themes"]), {
    themes: [],
    total: 0,
    limit: 200,
    offset: 0
  })
  mockState.queryData.set(mockState.queryKey(["writing-session", null]), null)

  useWritingPlaygroundStore.setState({
    activeSessionId: null,
    activeSessionName: null,
    editorMode: "plain"
  })
  useStoreChatModelSettings.getState().reset()
})

afterEach(() => {
  cleanup()
})

describe("WritingPlayground phase1 baseline", () => {
  it("test mock returns empty query state when a query is disabled", () => {
    const result = useQuery({
      queryKey: ["writing-capabilities"],
      queryFn: vi.fn(),
      enabled: false
    } as never) as MockQueryResult

    expect(result).toEqual({
      data: undefined,
      isLoading: false,
      isFetching: false,
      error: null
    })
  })

  it("test mock distinguishes full array query keys", () => {
    mockState.queryData.clear()
    mockState.queryData.set(
      mockState.queryKey(["writing-capabilities"]),
      { source: "base" }
    )
    mockState.queryData.set(
      mockState.queryKey(["writing-capabilities", "requested", "model-a", ""]),
      { source: "requested" }
    )

    const baseResult = useQuery({
      queryKey: ["writing-capabilities"],
      queryFn: vi.fn()
    } as never) as { data: unknown }
    const requestedResult = useQuery({
      queryKey: ["writing-capabilities", "requested", "model-a", ""],
      queryFn: vi.fn()
    } as never) as { data: unknown }

    expect(baseResult.data).toEqual({ source: "base" })
    expect(requestedResult.data).toEqual({ source: "requested" })
  })

  it("renders key empty-state landmarks without crashing", () => {
    render(<WritingPlayground />)

    expect(
      screen.getByTestId("writing-playground-shell")
    ).toBeInTheDocument()
    expect(
      screen.getByTestId("writing-playground-editor-panel")
    ).toBeInTheDocument()
    expect(
      screen.getByTestId("writing-playground-topbar")
    ).toBeInTheDocument()
    expect(screen.getByText("Select a session to begin.")).toBeInTheDocument()
    expect(
      screen.getByTestId("writing-playground-main-grid")
    ).toBeInTheDocument()
  })

  it("updates shell layout mode on resize for compact behavior", () => {
    const originalWidth = window.innerWidth
    try {
      Object.defineProperty(window, "innerWidth", {
        configurable: true,
        writable: true,
        value: 1280
      })

      render(<WritingPlayground />)

      const shell = screen.getByTestId("writing-playground-shell")
      expect(shell).toHaveAttribute("data-layout-mode", "expanded")

      window.innerWidth = 960
      fireEvent(window, new Event("resize"))
      expect(shell).toHaveAttribute("data-layout-mode", "compact")
    } finally {
      Object.defineProperty(window, "innerWidth", {
        configurable: true,
        writable: true,
        value: originalWidth
      })
    }
  })

  it("surfaces auto-routing limits for token inspection", () => {
    mockState.storageValues.set("selectedModel", "auto")
    seedWritingSession()

    render(<WritingPlayground />)
    fireEvent.click(screen.getByRole("button", { name: "Toggle settings" }))
    fireEvent.click(screen.getByTestId("writing-inspector-tab-inspect"))

    expect(
      screen.getByRole("button", { name: "Count tokens" })
    ).toBeDisabled()
  })

  it("passes auto model selections through generation requests", async () => {
    mockState.storageValues.set("selectedModel", "auto")
    seedWritingSession()

    render(<WritingPlayground />)

    fireEvent.change(
      screen.getByPlaceholderText("Start writing your prompt..."),
      {
        target: { value: "Route this prompt on the server." }
      }
    )
    fireEvent.click(screen.getByTestId("writing-topbar-generate"))

    await waitFor(() => {
      expect(mockState.streamCalls).toHaveLength(1)
    })
    expect(mockState.streamCalls[0]?.options.model).toBe("auto")
    expect(mockState.streamCalls[0]?.messages).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          content: "Route this prompt on the server."
        })
      ])
    )
  })

  it("renders the writing revision action bar when an active session exists", () => {
    mockState.storageValues.set("selectedModel", "mock-model")
    seedWritingSession({ prompt: "Draft text." })

    render(<WritingPlayground />)

    expect(screen.getByTestId("writing-revision-action-bar")).toBeInTheDocument()
    expect(screen.getByTestId("writing-revision-queue")).toBeInTheDocument()
  })

  it("creates a pending Rewrite proposal for selected text and applies it without mutating early", async () => {
    mockState.storageValues.set("selectedModel", "mock-model")
    mockState.sendResponses.push(structuredReplacement("The sharper sentence."))
    seedWritingSession({ prompt: "Intro. The old sentence. Outro." })

    render(<WritingPlayground />)
    const editor = getEditor()
    selectEditorText(editor, "The old sentence.")

    fireEvent.click(screen.getByRole("button", { name: /rewrite/i }))

    await waitFor(() => {
      expect(screen.getByText("The sharper sentence.")).toBeInTheDocument()
    })
    expect(editor.value).toBe("Intro. The old sentence. Outro.")
    expect(mockState.streamCalls).toHaveLength(0)
    expect(mockState.sendCalls).toHaveLength(1)

    fireEvent.click(screen.getByRole("button", { name: /apply/i }))

    await waitFor(() => {
      expect(editor.value).toBe("Intro. The sharper sentence. Outro.")
    })
  })

  it("shows malformed model output as a raw suggestion without Apply", async () => {
    mockState.storageValues.set("selectedModel", "mock-model")
    mockState.sendResponses.push("not structured json")
    seedWritingSession({ prompt: "Rewrite this line." })

    render(<WritingPlayground />)
    const editor = getEditor()
    selectEditorText(editor, "Rewrite this line.")

    fireEvent.click(screen.getByRole("button", { name: /rewrite/i }))

    await waitFor(() => {
      expect(screen.getByText("not structured json")).toBeInTheDocument()
    })
    const queue = screen.getByTestId("writing-revision-queue")
    expect(within(queue).queryByRole("button", { name: /apply/i })).toBeNull()
  })

  it("creates Outline as an advisory proposal without Apply", async () => {
    mockState.storageValues.set("selectedModel", "mock-model")
    mockState.sendResponses.push(structuredAdvice("Add a midpoint reversal."))
    seedWritingSession({ prompt: "Act one opens quietly." })

    render(<WritingPlayground />)
    fireEvent.click(screen.getByRole("button", { name: /outline/i }))

    await waitFor(() => {
      expect(screen.getByText("Add a midpoint reversal.")).toBeInTheDocument()
    })
    const queue = screen.getByTestId("writing-revision-queue")
    expect(within(queue).queryByRole("button", { name: /apply/i })).toBeNull()
  })

  it("includes the selected workflow preset instruction in the proposed-edit prompt", async () => {
    mockState.storageValues.set("selectedModel", "mock-model")
    mockState.sendResponses.push(structuredReplacement("Concise line."))
    seedWritingSession({ prompt: "This line could use fewer extra words." })

    render(<WritingPlayground />)
    const makeConcise = WRITING_REVISION_PRESETS.find(
      (preset) => preset.id === "make_concise"
    )
    expect(makeConcise).toBeTruthy()

    fireEvent.click(screen.getByRole("radio", { name: /make concise/i }))
    selectEditorText(getEditor(), "This line could use fewer extra words.")
    fireEvent.click(screen.getByRole("button", { name: /rewrite/i }))

    await waitFor(() => {
      expect(mockState.sendCalls).toHaveLength(1)
    })
    expect(latestRevisionPrompt()).toContain(makeConcise!.instruction)
  })

  it("passes writing context, routing, and generation settings into the proposed-edit prompt", async () => {
    mockState.storageValues.set("selectedModel", "context-model")
    useStoreChatModelSettings.getState().setApiProvider("anthropic")
    mockState.sendResponses.push(structuredReplacement("Context-aware rewrite."))
    seedWritingSession({
      prompt: "Original line.",
      template_name: "Story template",
      theme_name: "Noir theme",
      chat_mode: false,
      settings: {
        temperature: 0.42,
        top_p: 0.88,
        top_k: 11,
        token_streaming: true,
        max_tokens: 333,
        presence_penalty: 0.2,
        frequency_penalty: 0.1,
        seed: 1234,
        stop: ["END"],
        advanced_extra_body: { safe_key: "kept", api_key: "do-not-leak" },
        memory_block: {
          enabled: true,
          prefix: "Memory:",
          text: "The lighthouse is important.",
          suffix: ""
        },
        author_note: {
          enabled: true,
          prefix: "Author:",
          text: "Keep the narrator restrained.",
          suffix: "",
          insertion_depth: 2
        },
        world_info: {
          enabled: true,
          search_range: 2000,
          prefix: "",
          suffix: "",
          entries: [
            {
              id: "wi-1",
              keys: ["lighthouse"],
              content: "The lighthouse marks the old border.",
              enabled: true
            }
          ]
        },
        context_order:
          "{memPrefix}{memText}{memSuffix}{wiPrefix}{wiText}{wiSuffix}{prompt}",
        context_length: 4096,
        author_note_depth_mode: "insertion",
        logprobs: true,
        top_logprobs: 3,
        use_basic_stopping_mode: false,
        basic_stopping_mode_type: "max_tokens"
      }
    })

    render(<WritingPlayground />)
    selectEditorText(getEditor(), "Original line.")
    fireEvent.click(screen.getByRole("button", { name: /rewrite/i }))

    await waitFor(() => {
      expect(mockState.sendCalls).toHaveLength(1)
    })
    const prompt = latestRevisionPrompt()
    expect(prompt).toContain("The lighthouse is important.")
    expect(prompt).toContain("Keep the narrator restrained.")
    expect(prompt).toContain("The lighthouse marks the old border.")
    expect(prompt).toContain("Template: Story template")
    expect(prompt).toContain("Theme: Noir theme")
    expect(prompt).toContain("Provider: anthropic")
    expect(prompt).toContain("Model: context-model")
    expect(prompt).toContain('"temperature":0.42')
    expect(prompt).toContain('"topP":0.88')
    expect(prompt).toContain('"maxTokens":333')
    expect(prompt).toContain('"safe_key":"kept"')
    expect(prompt).not.toContain("do-not-leak")
    expect(mockState.sendCalls[0]?.options).toMatchObject({
      model: "context-model",
      temperature: 0.42,
      maxTokens: 333
    })
  })

  it("regenerates by rejecting the old proposal and appending a regenerated proposal", async () => {
    mockState.storageValues.set("selectedModel", "mock-model")
    mockState.sendResponses.push(
      structuredReplacement("First replacement.", "First proposal"),
      structuredReplacement("Second replacement.", "Second proposal")
    )
    seedWritingSession({ prompt: "Replace this sentence." })

    render(<WritingPlayground />)
    selectEditorText(getEditor(), "Replace this sentence.")
    fireEvent.click(screen.getByRole("button", { name: /rewrite/i }))

    await waitFor(() => {
      expect(screen.getByText("First replacement.")).toBeInTheDocument()
    })
    fireEvent.click(screen.getByRole("button", { name: /regenerate/i }))

    await waitFor(() => {
      expect(screen.getByText("Second replacement.")).toBeInTheDocument()
    })
    const queue = screen.getByTestId("writing-revision-queue")
    expect(within(queue).getByText("rejected")).toBeInTheDocument()
    expect(within(queue).getAllByText("pending")).toHaveLength(1)
    expect(screen.getByText(/regenerated from/i)).toBeInTheDocument()
  })

  it("shows manual-apply guidance for rich editor apply without mutating content", async () => {
    mockState.storageValues.set("selectedModel", "mock-model")
    mockState.sendResponses.push(structuredReplacement("Rich replacement."))
    useWritingPlaygroundStore.setState({ editorMode: "tiptap" })
    seedWritingSession({ prompt: "Rich original." })

    render(<WritingPlayground />)
    fireEvent.click(screen.getByRole("radio", { name: "Rich" }))
    fireEvent.click(screen.getByRole("button", { name: /rewrite/i }))

    await waitFor(() => {
      expect(screen.getByText("Rich replacement.")).toBeInTheDocument()
    })
    fireEvent.click(screen.getByRole("button", { name: /apply/i }))

    await waitFor(() => {
      expect(
        screen.getByText(/copy the suggestion and apply it manually/i)
      ).toBeInTheDocument()
    })
    expect(screen.getByText(/rich editor/i)).toBeInTheDocument()
  })

  it("allows confirmed whole-document text-changing targets to create applyable proposals", async () => {
    mockState.storageValues.set("selectedModel", "mock-model")
    mockState.sendResponses.push(structuredReplacement("Whole document rewrite."))
    seedWritingSession({
      prompt: `${"Long paragraph ".repeat(180)}\n\nSecond paragraph.`
    })

    render(<WritingPlayground />)
    fireEvent.click(screen.getByLabelText(/confirm whole-document text change/i))
    fireEvent.click(screen.getByRole("button", { name: /rewrite/i }))

    await waitFor(() => {
      expect(screen.getByText("Whole document rewrite.")).toBeInTheDocument()
    })
    const queue = screen.getByTestId("writing-revision-queue")
    expect(within(queue).getByRole("button", { name: /apply/i })).toBeEnabled()
    expect(latestRevisionPrompt()).toContain("Target summary: whole document")
  })
})
