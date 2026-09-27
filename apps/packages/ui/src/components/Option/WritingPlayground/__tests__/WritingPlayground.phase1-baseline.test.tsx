import React from "react"
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within
} from "@testing-library/react"
import { useQuery } from "@tanstack/react-query"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { ServicePromptSnapshot } from "@/services/service-prompts"
import { message } from "antd"
import { createServicePromptScopeChangedError } from "@/services/tldw/service-prompt-scope-error"

const mockState = vi.hoisted(() => ({
  storageValues: new Map<string, unknown>(),
  queryData: new Map<string, unknown>(),
  queryKey: (queryKey: unknown) =>
    JSON.stringify(Array.isArray(queryKey) ? queryKey : [queryKey]),
  executeMutations: false,
  resolveApiProviderForModel: vi.fn(async () => null as string | null),
  streamCalls: [] as Array<{ messages: unknown[]; options: Record<string, unknown> }>,
  sendCalls: [] as Array<{ messages: unknown[]; options: Record<string, unknown> }>,
  sendResponses: [] as Array<string | Promise<string>>,
  loadSnapshot: vi.fn(),
  cancelStream: vi.fn(),
  streamResults: [] as AsyncGenerator<string>[],
  responseCallbacks: [] as Array<((chunk: unknown) => void) | undefined>
}))

vi.mock("@/services/service-prompts", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/services/service-prompts")>(),
  loadServicePromptSnapshot: mockState.loadSnapshot
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
    useMutation: (options?: {
      mutationFn?: (variables: unknown) => unknown | Promise<unknown>
      onMutate?: (variables: unknown) => unknown
      onSuccess?: (
        data: unknown,
        variables: unknown,
        context: unknown
      ) => void
      onError?: (
        error: unknown,
        variables: unknown,
        context: unknown
      ) => void
    }) => {
      const mutate = vi.fn(async (variables: unknown) => {
        if (!mockState.executeMutations) return undefined
        const context = options?.onMutate?.(variables)
        try {
          const result = await options?.mutationFn?.(variables)
          options?.onSuccess?.(result, variables, context)
          return result
        } catch (error) {
          options?.onError?.(error, variables, context)
          throw error
        }
      })
      return {
        mutate,
        mutateAsync: mutate,
        isPending: false
      }
    },
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
    cancelStream() { mockState.cancelStream() }
    async *streamMessage(
      messages: unknown[],
      options: Record<string, unknown>,
      callback?: (chunk: unknown) => void
    ) {
      mockState.streamCalls.push({ messages, options })
      mockState.responseCallbacks.push(callback)
      const result = mockState.streamResults.shift()
      if (result) { yield* result; return }
      yield "mocked stream token"
    }
    async sendMessage(messages: unknown[], options: Record<string, unknown>, callback?: (chunk: unknown) => void) {
      mockState.sendCalls.push({ messages, options })
      mockState.responseCallbacks.push(callback)
      return await (mockState.sendResponses.shift() ?? "mocked completion")
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
  getManuscriptScene: vi.fn(),
  getManuscriptStructure: vi.fn(),
  importWritingSnapshot: vi.fn(),
  listManuscriptProjects: vi.fn(),
  listWritingSessions: vi.fn(),
  listWritingTemplates: vi.fn(),
  listWritingThemes: vi.fn(),
  createManuscriptProject: vi.fn(),
  reorderManuscriptItems: vi.fn(),
  updateManuscriptScene: vi.fn(),
  tokenizeWritingText: vi.fn(),
  updateWritingSession: vi.fn(),
  updateWritingTemplate: vi.fn(),
  updateWritingTheme: vi.fn()
}))

vi.mock("../WritingTipTapEditor", () => ({
  WritingTipTapEditor: ({
    content,
    onAdapterReady,
    onSelectionChange,
    onContentChange,
    placeholder
  }: {
    content?: {
      type?: string
      text?: string
      content?: Array<{
        type?: string
        text?: string
        content?: Array<{ type?: string; text?: string }>
      }>
    } | null
    onAdapterReady: (adapter: {
      getSelection: () => { start: number; end: number }
      setSelection: (selection: { start: number; end: number }) => void
      getSelectedText: (currentValue: string) => string
      focus: () => void
      measureRange?: (selection: { start: number; end: number }) => {
        top: number
        bottom: number
        height: number
      } | null
    }) => void
    onSelectionChange?: (selection: { start: number; end: number }) => void
    onContentChange: (json: Record<string, unknown>, plain: string) => void
    placeholder?: string
  }) => {
    const [selection, setSelection] = React.useState({ start: 0, end: 0 })
    const [value, setValue] = React.useState("")
    const adapter = React.useMemo(
      () => ({
        getSelection: () => selection,
        setSelection,
        getSelectedText: (currentValue: string) =>
          currentValue.slice(selection.start, selection.end),
        focus: () => {},
        measureRange: (range: { start: number; end: number }) =>
          range.end > range.start
            ? { top: 40 + range.start, bottom: 56 + range.start, height: 16 }
            : null
      }),
      [selection]
    )

    React.useEffect(() => {
      const text =
        content?.content
          ?.map((node) =>
            node.text ??
            node.content?.map((child) => child.text ?? "").join("") ??
            ""
          )
          .join("\n") ?? ""
      setValue(text)
    }, [content])

    React.useEffect(() => {
      onAdapterReady(adapter)
    }, [adapter, onAdapterReady])

    return (
      <textarea
        aria-label="Mock rich editor"
        placeholder={placeholder}
        value={value}
        onChange={(event) => {
          setValue(event.target.value)
          onContentChange({ type: "doc" }, event.target.value)
        }}
        onSelect={(event) => {
          const node = event.currentTarget
          setSelection({
            start: node.selectionStart,
            end: node.selectionEnd
          })
          onSelectionChange?.({
            start: node.selectionStart,
            end: node.selectionEnd
          })
        }}
      />
    )
  }
}))

import { WritingPlayground } from "../index"
import { buildWritingAnnotationsQueryKey } from "../hooks/useWritingAnnotations"
import { WRITING_REVISION_PRESETS } from "../writing-revision-presets"
import { useStoreChatModelSettings } from "@/store/model"
import { useWritingPlaygroundStore } from "@/store/writing-playground"
import {
  updateWritingSession,
  type ManuscriptAnnotationResponse
} from "@/services/writing-playground"

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

const sceneRichContent = (text: string) => ({
  type: "doc",
  content: [
    {
      type: "paragraph",
      content: [{ type: "text", text }]
    }
  ]
})

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
  mockState.queryData.set(
    mockState.queryKey(["writing-session", "session-auto"]),
    {
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
    }
  )
}

const seedManuscriptStructure = () => {
  mockState.queryData.set(
    mockState.queryKey(["manuscript-structure", "project-1"]),
    {
      parts: [],
      unassigned_chapters: [
        {
          id: "chapter-1",
          title: "Chapter 1",
          word_count: 4,
          version: 1,
          scenes: [
            {
              id: "scene-1",
              title: "Scene A",
              word_count: 2,
              version: 3
            },
            {
              id: "scene-2",
              title: "Scene B",
              word_count: 2,
              version: 1
            }
          ]
        }
      ]
    }
  )
}

const seedManuscriptScene = (
  text: string,
  overrides: Record<string, unknown> = {}
) => {
  mockState.queryData.set(mockState.queryKey(["manuscript-scene", "scene-1"]), {
    id: "scene-1",
    chapter_id: "chapter-1",
    project_id: "project-1",
    title: "Scene 1",
    sort_order: 1,
    content: sceneRichContent(text),
    content_plain: text,
    synopsis: null,
    word_count: text.trim().split(/\s+/).filter(Boolean).length,
    pov_character_id: null,
    status: "draft",
    created_at: "2026-06-23T12:00:00Z",
    last_modified: "2026-06-23T12:00:00Z",
    deleted: false,
    client_id: "test-client",
    version: 3,
    ...overrides
  })
}

const makeManuscriptAnnotation = (
  overrides: Partial<ManuscriptAnnotationResponse> = {}
): ManuscriptAnnotationResponse => ({
  id: "annotation-1",
  project_id: "project-1",
  target_type: "scene",
  target_id: "scene-1",
  status: "open",
  category: "clarity",
  tags: [],
  source: "user",
  body: "Tighten this sentence.",
  suggested_fix: null,
  followup_note: null,
  metadata: {},
  scene_version: 3,
  anchor_start: 0,
  anchor_end: 6,
  selected_text: "target",
  anchor_status: "attached",
  derived_start: null,
  derived_end: null,
  scene_level: false,
  created_at: "2026-06-25T00:00:00Z",
  last_modified: "2026-06-25T00:00:00Z",
  deleted: false,
  client_id: "test-client",
  version: 1,
  ...overrides
})

const seedManuscriptAnnotations = (
  annotations: ManuscriptAnnotationResponse[]
) => {
  mockState.queryData.set(
    mockState.queryKey(
      buildWritingAnnotationsQueryKey({
        projectId: "project-1",
        targetContext: { targetType: "scene", targetId: "scene-1" }
      })
    ),
    {
      annotations,
      total: annotations.length,
      limit: 50,
      offset: 0
    }
  )
}

const getEditor = () =>
  screen.getByPlaceholderText("Start writing your prompt...") as HTMLTextAreaElement

const deferred = <T,>() => {
  let resolve!: (value: T) => void
  let reject!: (reason: unknown) => void
  const promise = new Promise<T>((res, rej) => { resolve = res; reject = rej })
  return { promise, resolve, reject }
}

const continuationSnapshot = (
  id = "writing.continuation.predict",
  system = "Continue in {my style}."
) => {
  const scope = new AbortController()
  const snapshot: ServicePromptSnapshot = {
    scopeKey: "test-user-42",
    requestScope: { config: { serverUrl: "http://localhost:8000", authMode: "multi-user" }, userId: 42 },
    capability: "supported",
    definitions: {
      [id]: {
        definition: { id, parts: [{ key: "system", mode: "literal", required_variables: [] }] },
        parts: { system }, source: "user", revision: "test-revision"
      }
    },
    scopeSignal: scope.signal,
    scopeInvalidatedSignal: scope.signal,
    release: vi.fn()
  }
  return { snapshot, scope }
}

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
  mockState.executeMutations = false
  mockState.resolveApiProviderForModel.mockReset()
  mockState.resolveApiProviderForModel.mockResolvedValue(null)
  mockState.streamCalls.length = 0
  mockState.sendCalls.length = 0
  mockState.sendResponses.length = 0
  mockState.streamResults.length = 0
  mockState.responseCallbacks.length = 0
  mockState.cancelStream.mockReset()
  mockState.loadSnapshot.mockReset()
  mockState.loadSnapshot.mockImplementation(async ([id]: string[]) =>
    continuationSnapshot(id).snapshot
  )
  vi.mocked(updateWritingSession).mockReset()

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
    activeProjectId: null,
    activeNodeId: null,
    activeNodeType: null,
    editorMode: "plain"
  })
  useStoreChatModelSettings.getState().reset()
})

afterEach(() => {
  vi.useRealTimers()
  cleanup()
  message.destroy()
})

describe("WritingPlayground phase1 baseline", () => {
  describe("scoped continuation", () => {
    it.each([
      ["preset", "partial"], ["reject", "partial"], ["apply", "partial"],
      ["preset", "admission"], ["reject", "admission"], ["apply", "admission"]
    ])("prevents %s from persisting provisional continuation text at %s and resumes idle mutations", async (mutation, timing) => {
      const original = "Intro. The old sentence. Outro."
      const rewritten = "Intro. The sharper sentence. Outro."
      const persistedPayloads: Record<string, unknown>[] = []
      mockState.executeMutations = true
      mockState.storageValues.set("selectedModel", "mock-model")
      seedWritingSession({ prompt: original })
      vi.mocked(updateWritingSession).mockImplementation(async (sessionId, patch, expectedVersion) => {
        const payload = patch.payload ?? {}
        persistedPayloads.push(payload)
        return {
          id: sessionId, name: "Auto Session", payload,
          schema_version: patch.schema_version ?? 1, version_parent_id: null,
          created_at: "2026-03-16T12:00:00Z", last_modified: "2026-03-16T12:00:01Z",
          deleted: false, client_id: "test-client", version: expectedVersion + 1
        } as Awaited<ReturnType<typeof updateWritingSession>>
      })
      mockState.sendResponses.push(structuredReplacement("The sharper sentence."))
      render(<WritingPlayground />)
      selectEditorText(getEditor(), "The old sentence.")
      fireEvent.click(screen.getByRole("button", { name: /^rewrite$/i }))
      await waitFor(() => expect(screen.getByText("The sharper sentence.")).toBeInTheDocument())
      await waitFor(() => {
        expect(persistedPayloads.map((payload) => payload.prompt)).toEqual([original])
      }, { timeout: 2000 })

      const tail = deferred<string>()
      mockState.streamResults.push((async function* () { yield " provisional"; yield await tail.promise })())
      const { snapshot, scope } = continuationSnapshot()
      mockState.loadSnapshot.mockResolvedValue(snapshot)
      const mutateRevision = () => {
        fireEvent.click(mutation === "preset"
          ? screen.getByRole("radio", { name: /make concise/i })
          : screen.getByRole("button", { name: new RegExp(`^${mutation}$`, "i") }))
      }
      vi.useFakeTimers()
      await act(async () => {
        fireEvent.click(screen.getByTestId("writing-topbar-generate"))
        if (timing === "admission") mutateRevision()
      })
      expect(getEditor()).toHaveValue(`${original} provisional`)
      if (timing === "partial") mutateRevision()
      act(() => scope.abort())
      await act(async () => { tail.resolve(" stale"); await vi.advanceTimersByTimeAsync(800) })
      vi.useRealTimers()
      // Observe the real session-save boundary after the debounce, not merely
      // whether the control looked disabled during generation.
      expect(persistedPayloads.map((payload) => payload.prompt)).toEqual([original])
      expect(getEditor()).toHaveValue(original)

      vi.useFakeTimers()
      mutateRevision()
      expect(getEditor()).toHaveValue(mutation === "apply" ? rewritten : original)
      await act(async () => { await vi.advanceTimersByTimeAsync(800) })
      vi.useRealTimers()
      expect(persistedPayloads).toHaveLength(2)
      expect(persistedPayloads[1]?.prompt).not.toContain("provisional")
      if (mutation !== "apply") expect(persistedPayloads[1]?.prompt).toBe(original)
      if (mutation === "preset") {
        expect(persistedPayloads[1]?.revision_preset_id).toBe("make_concise")
      } else {
        expect(JSON.stringify(persistedPayloads[1]?.revisions)).toContain(
          mutation === "apply" ? '"applied"' : '"rejected"'
        )
      }
    })

    it.each([
      ["predict", false, "Continue in {my style}."],
      ["predict", true, "Continue in {my style}."],
      ["fill", false, "Fill in {my style}."],
      ["fill", true, "Fill in {my style}."],
      ["predict", false, "Continue the text from the prompt. Respond with only the continuation."],
      ["predict", true, "Continue the text from the prompt. Respond with only the continuation."],
      ["fill", false, "Fill in the missing text between the prefix and suffix. Respond with only the missing text."],
      ["fill", true, "Fill in the missing text between the prefix and suffix. Respond with only the missing text."]
    ] as const)("uses selected %s instructions with streaming=%s: %s", async (mode, streaming, system) => {
      mockState.storageValues.set("selectedModel", "mock-model")
      const prompt = mode === "fill" ? "Opening{fill} tail" : "Opening"
      seedWritingSession({ prompt, settings: {
        token_streaming: streaming, temperature: 0.42, top_p: 0.88,
        max_tokens: 333, frequency_penalty: 0.1, presence_penalty: 0.2,
        top_k: 11, seed: 1234, stop: ["END"], use_basic_stopping_mode: false
      } })
      const { snapshot } = continuationSnapshot(`writing.continuation.${mode}`, system)
      mockState.loadSnapshot.mockResolvedValue(snapshot)
      mockState.sendResponses.push(" ending")
      mockState.streamResults.push((async function* () { yield " ending" })())
      render(<WritingPlayground />)
      fireEvent.click(screen.getByTestId("writing-topbar-generate"))
      const calls = streaming ? mockState.streamCalls : mockState.sendCalls
      await waitFor(() => expect(calls).toHaveLength(1))
      expect(mockState.loadSnapshot).toHaveBeenCalledExactlyOnceWith(
        [`writing.continuation.${mode}`], { signal: expect.any(AbortSignal) }
      )
      expect(calls[0]?.options).toMatchObject({
        systemPrompt: system, requestScope: { userId: 42 },
        signal: snapshot.scopeSignal, model: "mock-model", temperature: 0.42,
        topP: 0.88, maxTokens: 333, frequencyPenalty: 0.1, presencePenalty: 0.2,
        extraBody: { top_k: 11, seed: 1234, stop: ["END"] }
      })
      expect(calls[0]?.messages).toEqual([{ role: "user", content: mode === "fill"
        ? "Fill in the missing text between the prefix and suffix.\n\nPrefix:\nOpening\n\nSuffix:\n tail\n\nReturn only the missing text."
        : "Opening" }])
      await waitFor(() => expect(getEditor()).toHaveValue(mode === "fill" ? "Opening ending tail" : "Opening ending"))
      expect(snapshot.release).toHaveBeenCalledTimes(1)
    })

    it.each([false, true])("bypasses prompt lookup in chat mode with streaming=%s", async (streaming) => {
      mockState.storageValues.set("selectedModel", "mock-model")
      seedWritingSession({ prompt: "Opening", chat_mode: true, settings: {
        token_streaming: streaming,
        memory_block: { enabled: true, prefix: "", text: "Explicit instructions", suffix: "" }
      } })
      render(<WritingPlayground />)
      fireEvent.click(screen.getByTestId("writing-topbar-generate"))
      const calls = streaming ? mockState.streamCalls : mockState.sendCalls
      await waitFor(() => expect(calls).toHaveLength(1))
      expect(mockState.loadSnapshot).not.toHaveBeenCalled()
      expect(calls[0]?.options.systemPrompt).toBeUndefined()
      expect(calls[0]?.options.requestScope).toBeUndefined()
      expect(calls[0]?.messages).toEqual([
        { role: "system", content: "Explicit instructions" },
        { role: "user", content: "Opening" }
      ])
    })

    it.each([false, true])("preserves fill template, ordered context, stop mode and logprobs with streaming=%s", async (streaming) => {
      mockState.storageValues.set("selectedModel", "mock-model")
      seedWritingSession({ prompt: "Opening{fill}tail", template_name: "FIM", settings: {
        token_streaming: streaming, logprobs: true, top_logprobs: 3,
        use_basic_stopping_mode: true, basic_stopping_mode_type: "fill_suffix",
        memory_block: { enabled: true, prefix: "<memory>", text: "Remember", suffix: "</memory>" },
        context_order: "{memPrefix}{memText}{memSuffix}{prompt}"
      } })
      mockState.queryData.set(mockState.queryKey(["writing-templates"]), {
        templates: [{ id: "fim", name: "FIM", payload: { fim_template: "<prefix>{prefix}<suffix>{suffix}<middle>" } }],
        total: 1, limit: 200, offset: 0
      })
      const response = deferred<string>()
      mockState.sendResponses.push(response.promise)
      mockState.streamResults.push((async function* () { yield await response.promise })())
      const { snapshot } = continuationSnapshot("writing.continuation.fill", "Literal {braces}")
      const removeListener = vi.spyOn(snapshot.scopeInvalidatedSignal, "removeEventListener")
      mockState.loadSnapshot.mockResolvedValue(snapshot)
      render(<WritingPlayground />)
      fireEvent.click(screen.getByTestId("writing-topbar-generate"))
      const calls = streaming ? mockState.streamCalls : mockState.sendCalls
      await waitFor(() => expect(calls).toHaveLength(1))
      expect(calls[0]?.messages).toEqual([{ role: "user", content: "<memory>Remember</memory><prefix>Opening<suffix>tail<middle>" }])
      expect(calls[0]?.options).toMatchObject({
        systemPrompt: "Literal {braces}", logprobs: true, topLogprobs: 3,
        extraBody: { stop: ["ta"] }
      })
      await act(async () => {
        mockState.responseCallbacks[0]?.({ choices: [{ logprobs: { content: [{ token: "ACCEPTED", logprob: -0.2, top_logprobs: [] }] } }] })
        response.resolve(" middle ")
      })
      expect(getEditor()).toHaveValue("Opening middle tail")
      expect(screen.getByRole("button", { name: "ACCEPTED" })).toBeInTheDocument()
      expect(removeListener).toHaveBeenCalledWith("abort", expect.any(Function))
      expect(snapshot.release).toHaveBeenCalledTimes(1)
    })

    it("keeps parsed explicit system messages ahead of chat context", async () => {
      mockState.storageValues.set("selectedModel", "mock-model")
      seedWritingSession({ prompt: "<system>Explicit rules</system><user>Hello</user>", chat_mode: true, template_name: "Chat", settings: {
        token_streaming: false,
        memory_block: { enabled: true, prefix: "", text: "Memory rules", suffix: "" }
      } })
      mockState.queryData.set(mockState.queryKey(["writing-templates"]), {
        templates: [{ id: "chat", name: "Chat", payload: {
          system_prefix: "<system>", system_suffix: "</system>", user_prefix: "<user>", user_suffix: "</user>"
        } }], total: 1, limit: 200, offset: 0
      })
      render(<WritingPlayground />)
      fireEvent.click(screen.getByTestId("writing-topbar-generate"))
      await waitFor(() => expect(mockState.sendCalls).toHaveLength(1))
      expect(mockState.loadSnapshot).not.toHaveBeenCalled()
      expect(mockState.sendCalls[0]?.messages).toEqual([
        { role: "system", content: "Explicit rules" },
        { role: "system", content: "Memory rules" },
        { role: "user", content: "Hello" }
      ])
      expect(mockState.sendCalls[0]?.options.systemPrompt).toBeUndefined()
    })

    it("releases an old lookup without canceling a newer pending generation", async () => {
      mockState.storageValues.set("selectedModel", "mock-model")
      seedWritingSession({ prompt: "Opening" })
      const lookup = deferred<ServicePromptSnapshot>()
      const response = deferred<string>()
      const first = continuationSnapshot()
      const second = continuationSnapshot()
      mockState.loadSnapshot.mockReturnValueOnce(lookup.promise).mockResolvedValueOnce(second.snapshot)
      mockState.streamResults.push((async function* () { yield await response.promise })())
      render(<WritingPlayground />)
      fireEvent.click(screen.getByTestId("writing-topbar-generate"))
      fireEvent.click(screen.getByTestId("writing-topbar-generate"))
      fireEvent.click(screen.getByTestId("writing-topbar-generate"))
      await waitFor(() => expect(mockState.streamCalls).toHaveLength(1))
      await act(async () => { lookup.resolve(first.snapshot) })
      expect(first.snapshot.release).toHaveBeenCalledTimes(1)
      expect(second.snapshot.release).not.toHaveBeenCalled()
      expect(mockState.cancelStream).not.toHaveBeenCalled()
      expect(screen.getByTestId("writing-topbar-generate")).toHaveTextContent("Stop")
      await act(async () => { response.resolve(" fresh") })
      expect(getEditor()).toHaveValue("Opening fresh")
      expect(second.snapshot.release).toHaveBeenCalledTimes(1)
    })

    it.each([false, true])("rejects a scope-invalidated late lookup with streaming=%s", async (streaming) => {
      mockState.storageValues.set("selectedModel", "mock-model")
      seedWritingSession({ prompt: "Opening{fill} tail", settings: { token_streaming: streaming } })
      const lookup = deferred<ServicePromptSnapshot>()
      const { snapshot, scope } = continuationSnapshot("writing.continuation.fill")
      mockState.loadSnapshot.mockReturnValue(lookup.promise)
      render(<WritingPlayground />)
      fireEvent.click(screen.getByTestId("writing-topbar-generate"))
      await act(async () => { scope.abort(); lookup.resolve(snapshot) })
      expect(mockState.sendCalls).toHaveLength(0)
      expect(mockState.streamCalls).toHaveLength(0)
      expect(getEditor()).toHaveValue("Opening{fill} tail")
      expect(snapshot.release).toHaveBeenCalledTimes(1)
      expect(screen.getByTestId("writing-topbar-generate")).toHaveTextContent("Generate")
    })

    it.each(["missing", "empty", "rejected"])("rejects %s snapshot without dispatch or history", async (failure) => {
      mockState.storageValues.set("selectedModel", "mock-model")
      seedWritingSession({ prompt: "Opening{fill} tail" })
      const { snapshot } = continuationSnapshot(failure === "missing" ? "writing.continuation.predict" : "writing.continuation.fill", "  ")
      if (failure === "rejected") mockState.loadSnapshot.mockRejectedValue(new Error("lookup failed"))
      else mockState.loadSnapshot.mockResolvedValue(snapshot)
      const error = vi.spyOn(message, "error")
      render(<WritingPlayground />)
      fireEvent.click(screen.getByTestId("writing-topbar-generate"))
      await waitFor(() => expect(error).toHaveBeenCalledTimes(1))
      expect(mockState.sendCalls).toHaveLength(0)
      expect(mockState.streamCalls).toHaveLength(0)
      expect(getEditor()).toHaveValue("Opening{fill} tail")
      expect(screen.getByTitle("Undo generation")).toBeDisabled()
      expect(snapshot.release).toHaveBeenCalledTimes(failure === "rejected" ? 0 : 1)
      error.mockRestore()
    })

    it("treats a rejected scope-changing lookup as invalidation, not a current generation error", async () => {
      mockState.storageValues.set("selectedModel", "mock-model")
      seedWritingSession({ prompt: "Opening{fill} tail" })
      const lookup = deferred<ServicePromptSnapshot>()
      mockState.loadSnapshot.mockReturnValue(lookup.promise)
      const error = vi.spyOn(message, "error")
      render(<WritingPlayground />)
      fireEvent.click(screen.getByTestId("writing-topbar-generate"))
      await act(async () => { lookup.reject(createServicePromptScopeChangedError()) })
      expect(error).not.toHaveBeenCalled()
      expect(mockState.loadSnapshot.mock.calls[0]?.[1].signal.aborted).toBe(true)
      expect(getEditor()).toHaveValue("Opening{fill} tail")
      expect(mockState.streamCalls).toHaveLength(0)
      expect(screen.getByTitle("Undo generation")).toBeDisabled()
      error.mockRestore()
    })

    it("preserves manually stopped partial output and undo while ignoring late chunks and errors", async () => {
      mockState.storageValues.set("selectedModel", "mock-model")
      seedWritingSession({ prompt: "Opening", settings: { logprobs: true } })
      const tail = deferred<string>()
      mockState.streamResults.push((async function* () { yield " partial"; yield await tail.promise })())
      const { snapshot } = continuationSnapshot()
      mockState.loadSnapshot.mockResolvedValue(snapshot)
      const error = vi.spyOn(message, "error")
      render(<WritingPlayground />)
      fireEvent.click(screen.getByTestId("writing-topbar-generate"))
      await waitFor(() => expect(getEditor()).toHaveValue("Opening partial"))
      fireEvent.click(screen.getByTestId("writing-topbar-generate"))
      expect(getEditor()).toHaveValue("Opening partial")
      expect(snapshot.release).toHaveBeenCalledTimes(1)
      expect(mockState.cancelStream).toHaveBeenCalledTimes(1)
      await act(async () => { tail.reject(new Error("late failure")) })
      expect(error).not.toHaveBeenCalled()
      fireEvent.click(screen.getByTitle("Undo generation"))
      expect(getEditor()).toHaveValue("Opening")
      fireEvent.click(screen.getByTitle("Redo generation"))
      expect(getEditor()).toHaveValue("Opening partial")
      error.mockRestore()
    })

    it.each([false, true])("ignores old response/logprobs/finalization during a new request with streaming=%s", async (streaming) => {
      mockState.storageValues.set("selectedModel", "mock-model")
      seedWritingSession({ prompt: "Opening", settings: { token_streaming: streaming, logprobs: true } })
      const oldResponse = deferred<string>()
      const newResponse = deferred<string>()
      if (streaming) {
        mockState.streamResults.push((async function* () { yield await oldResponse.promise })())
        mockState.streamResults.push((async function* () { yield await newResponse.promise })())
      } else mockState.sendResponses.push(oldResponse.promise, newResponse.promise)
      const first = continuationSnapshot()
      const second = continuationSnapshot()
      mockState.loadSnapshot.mockResolvedValueOnce(first.snapshot).mockResolvedValueOnce(second.snapshot)
      render(<WritingPlayground />)
      const calls = streaming ? mockState.streamCalls : mockState.sendCalls
      fireEvent.click(screen.getByTestId("writing-topbar-generate"))
      await waitFor(() => expect(calls).toHaveLength(1))
      act(() => first.scope.abort())
      fireEvent.click(screen.getByTestId("writing-topbar-generate"))
      await waitFor(() => expect(calls).toHaveLength(2))
      await act(async () => {
        mockState.responseCallbacks[0]?.({ choices: [{ logprobs: { content: [{ token: "STALE", logprob: -0.2, top_logprobs: [] }] } }] })
        oldResponse.resolve(" stale")
      })
      expect(getEditor()).toHaveValue("Opening")
      expect(screen.getByTestId("writing-topbar-generate")).toHaveTextContent("Stop")
      expect(second.snapshot.release).not.toHaveBeenCalled()
      expect(mockState.cancelStream).toHaveBeenCalledTimes(1)
      await act(async () => { newResponse.resolve(" fresh") })
      expect(getEditor()).toHaveValue("Opening fresh")
      expect(screen.queryByText("STALE")).not.toBeInTheDocument()
      expect(first.snapshot.release).toHaveBeenCalledTimes(1)
      expect(second.snapshot.release).toHaveBeenCalledTimes(1)
      fireEvent.click(screen.getByTitle("Undo generation"))
      expect(getEditor()).toHaveValue("Opening")
    })

    it.each(["lookup", "send", "stream"])("cancels on unmount during %s and releases late leases", async (phase) => {
      mockState.storageValues.set("selectedModel", "mock-model")
      seedWritingSession({ prompt: "Opening", settings: { token_streaming: phase !== "send" } })
      const lookup = deferred<ServicePromptSnapshot>()
      const response = deferred<string>()
      const { snapshot } = continuationSnapshot()
      mockState.loadSnapshot.mockReturnValue(phase === "lookup" ? lookup.promise : Promise.resolve(snapshot))
      mockState.sendResponses.push(response.promise)
      mockState.streamResults.push((async function* () { yield await response.promise })())
      const view = render(<WritingPlayground />)
      fireEvent.click(screen.getByTestId("writing-topbar-generate"))
      if (phase !== "lookup") await waitFor(() => expect(mockState.responseCallbacks).toHaveLength(1))
      view.unmount()
      expect(mockState.loadSnapshot.mock.calls[0]?.[1].signal.aborted).toBe(true)
      await act(async () => { lookup.resolve(snapshot); response.resolve(" stale") })
      expect(snapshot.release).toHaveBeenCalledTimes(1)
    })

    it.each([false, true])("cancels session binding changes with streaming=%s", async (streaming) => {
      mockState.storageValues.set("selectedModel", "mock-model")
      seedWritingSession({ prompt: "Opening", settings: { token_streaming: streaming } })
      const response = deferred<string>()
      mockState.sendResponses.push(response.promise)
      mockState.streamResults.push((async function* () { yield " partial"; yield await response.promise })())
      const { snapshot } = continuationSnapshot()
      mockState.loadSnapshot.mockResolvedValue(snapshot)
      const current = mockState.queryData.get(mockState.queryKey(["writing-session", "session-auto"])) as Record<string, unknown>
      mockState.queryData.set(mockState.queryKey(["writing-session", "session-other"]), {
        ...current, id: "session-other", name: "Other", payload: { prompt: "Other draft", settings: {} }
      })
      mockState.queryData.set(mockState.queryKey(["writing-sessions"]), {
        sessions: [{ id: "session-auto", name: "Auto Session" }, { id: "session-other", name: "Other" }],
        total: 2, limit: 200, offset: 0
      })
      render(<WritingPlayground />)
      fireEvent.click(screen.getByTestId("writing-topbar-generate"))
      await waitFor(() => expect(mockState.responseCallbacks).toHaveLength(1))
      act(() => useWritingPlaygroundStore.setState({ activeSessionId: "session-other", activeSessionName: "Other" }))
      await waitFor(() => expect(getEditor()).toHaveValue("Other draft"))
      await act(async () => { response.resolve(" stale") })
      expect(getEditor()).toHaveValue("Other draft")
      expect(snapshot.release).toHaveBeenCalledTimes(1)
      expect(screen.getByTitle("Undo generation")).toBeDisabled()
    })

    it.each([false, true])("cancels scene binding changes with streaming=%s", async (streaming) => {
      mockState.storageValues.set("selectedModel", "mock-model")
      seedWritingSession({ prompt: "Session prompt", settings: { token_streaming: streaming } })
      seedManuscriptScene("Scene A")
      seedManuscriptScene("Scene B", { id: "scene-2" })
      const sceneB = mockState.queryData.get(mockState.queryKey(["manuscript-scene", "scene-1"]))
      mockState.queryData.set(mockState.queryKey(["manuscript-scene", "scene-2"]), sceneB)
      seedManuscriptScene("Scene A")
      useWritingPlaygroundStore.setState({ activeProjectId: "project-1", activeNodeType: "scene", activeNodeId: "scene-1" })
      const response = deferred<string>()
      mockState.sendResponses.push(response.promise)
      mockState.streamResults.push((async function* () { yield " partial"; yield await response.promise })())
      const { snapshot } = continuationSnapshot()
      mockState.loadSnapshot.mockResolvedValue(snapshot)
      render(<WritingPlayground />)
      await waitFor(() => expect(getEditor()).toHaveValue("Scene A"))
      fireEvent.click(screen.getByTestId("writing-topbar-generate"))
      await waitFor(() => expect(mockState.responseCallbacks).toHaveLength(1))
      act(() => useWritingPlaygroundStore.setState({ activeNodeId: "scene-2" }))
      await waitFor(() => expect(getEditor()).toHaveValue("Scene B"))
      await act(async () => { response.resolve(" stale") })
      expect(getEditor()).toHaveValue("Scene B")
      expect(snapshot.release).toHaveBeenCalledTimes(1)
    })

    it.each([false, true])("preserves a refreshed scene version before the first continuation response with streaming=%s", async (streaming) => {
      mockState.storageValues.set("selectedModel", "mock-model")
      seedWritingSession({ prompt: "Session prompt", settings: { token_streaming: streaming } })
      seedManuscriptScene("Original scene", { version: 1 })
      useWritingPlaygroundStore.setState({ activeProjectId: "project-1", activeNodeType: "scene", activeNodeId: "scene-1" })
      const response = deferred<string>()
      mockState.sendResponses.push(response.promise)
      mockState.streamResults.push((async function* () { yield await response.promise })())
      const view = render(<WritingPlayground />)
      await waitFor(() => expect(getEditor()).toHaveValue("Original scene"))
      fireEvent.click(screen.getByTestId("writing-topbar-generate"))
      await waitFor(() => expect(mockState.responseCallbacks).toHaveLength(1))

      seedManuscriptScene("Newer saved scene", { version: 2 })
      view.rerender(<WritingPlayground />)
      await waitFor(() => expect(getEditor()).toHaveValue("Newer saved scene"))
      await act(async () => { response.resolve(" stale continuation") })

      expect(getEditor()).toHaveValue("Newer saved scene")
      expect(screen.getByTitle("Undo generation")).toBeDisabled()
    })

    it("does not overwrite independent editor edits during invalidation", async () => {
      mockState.storageValues.set("selectedModel", "mock-model")
      seedWritingSession({ prompt: "Opening" })
      const response = deferred<string>()
      mockState.streamResults.push((async function* () { yield " partial"; yield await response.promise })())
      const { snapshot, scope } = continuationSnapshot()
      mockState.loadSnapshot.mockResolvedValue(snapshot)
      render(<WritingPlayground />)
      fireEvent.click(screen.getByTestId("writing-topbar-generate"))
      await waitFor(() => expect(getEditor()).toHaveValue("Opening partial"))
      fireEvent.change(getEditor(), { target: { value: "Independent edit" } })
      act(() => scope.abort())
      await act(async () => { response.resolve(" stale") })
      expect(getEditor()).toHaveValue("Independent edit")
      expect(screen.getByTitle("Undo generation")).toBeDisabled()
    })

    it("restores the actual manuscript, not a synthetic reroll prompt, on scope invalidation", async () => {
      mockState.storageValues.set("selectedModel", "mock-model")
      seedWritingSession({ prompt: "Opening", settings: { logprobs: true } })
      const firstResponse = deferred<string>()
      const rerollResponse = deferred<string>()
      mockState.streamResults.push((async function* () { yield await firstResponse.promise })())
      mockState.streamResults.push((async function* () { yield " rerolled"; yield await rerollResponse.promise })())
      const first = continuationSnapshot()
      const reroll = continuationSnapshot()
      mockState.loadSnapshot.mockResolvedValueOnce(first.snapshot).mockResolvedValueOnce(reroll.snapshot)
      render(<WritingPlayground />)
      fireEvent.click(screen.getByTestId("writing-topbar-generate"))
      await waitFor(() => expect(mockState.streamCalls).toHaveLength(1))
      await act(async () => {
        mockState.responseCallbacks[0]?.({ choices: [{ logprobs: { content: [{ token: "ending", logprob: -0.2, top_logprobs: [] }] } }] })
        firstResponse.resolve(" ending")
      })
      expect(getEditor()).toHaveValue("Opening ending")
      fireEvent.click(screen.getByRole("button", { name: "ending" }))
      await waitFor(() => expect(getEditor()).toHaveValue("Opening rerolled"))
      act(() => reroll.scope.abort())
      await act(async () => { rerollResponse.resolve(" stale") })
      expect(getEditor()).toHaveValue("Opening ending")
      expect(reroll.snapshot.release).toHaveBeenCalledTimes(1)
    })

    it.each([false, true])("releases the lease after generation failure with streaming=%s", async (streaming) => {
      mockState.storageValues.set("selectedModel", "mock-model")
      seedWritingSession({ prompt: "Opening", settings: { token_streaming: streaming } })
      const response = deferred<string>()
      mockState.sendResponses.push(response.promise)
      mockState.streamResults.push((async function* () { yield await response.promise })())
      const { snapshot } = continuationSnapshot()
      mockState.loadSnapshot.mockResolvedValue(snapshot)
      const error = vi.spyOn(message, "error")
      render(<WritingPlayground />)
      fireEvent.click(screen.getByTestId("writing-topbar-generate"))
      await waitFor(() => expect(mockState.responseCallbacks).toHaveLength(1))
      await act(async () => { response.reject(new Error("provider failed")) })
      expect(error).toHaveBeenCalledTimes(1)
      expect(snapshot.release).toHaveBeenCalledTimes(1)
      expect(getEditor()).toHaveValue("Opening")
      expect(screen.getByTestId("writing-topbar-generate")).toHaveTextContent("Generate")
      error.mockRestore()
    })

    it("cancels lookup without removing a placeholder and releases a late snapshot", async () => {
      mockState.storageValues.set("selectedModel", "mock-model")
      seedWritingSession({ prompt: "Opening{fill} tail" })
      const lookup = deferred<ServicePromptSnapshot>()
      const { snapshot } = continuationSnapshot("writing.continuation.fill")
      mockState.loadSnapshot.mockReturnValue(lookup.promise)
      render(<WritingPlayground />)
      fireEvent.click(screen.getByTestId("writing-topbar-generate"))
      expect(getEditor()).toHaveValue("Opening{fill} tail")
      fireEvent.click(screen.getByTestId("writing-topbar-generate"))
      expect(mockState.loadSnapshot.mock.calls[0]?.[1].signal.aborted).toBe(true)
      await act(async () => { lookup.resolve(snapshot) })
      expect(mockState.streamCalls).toHaveLength(0)
      expect(getEditor()).toHaveValue("Opening{fill} tail")
      expect(snapshot.release).toHaveBeenCalledTimes(1)
    })

    it("rolls back provisional output on scope invalidation without creating undo history", async () => {
      mockState.storageValues.set("selectedModel", "mock-model")
      seedWritingSession({ prompt: "Opening" })
      const tail = deferred<string>()
      mockState.streamResults.push((async function* () { yield " partial"; yield await tail.promise })())
      const { snapshot, scope } = continuationSnapshot()
      mockState.loadSnapshot.mockResolvedValue(snapshot)
      render(<WritingPlayground />)
      fireEvent.click(screen.getByTestId("writing-topbar-generate"))
      await waitFor(() => expect(getEditor()).toHaveValue("Opening partial"))
      act(() => scope.abort())
      await waitFor(() => expect(getEditor()).toHaveValue("Opening"))
      await act(async () => { tail.resolve(" stale") })
      expect(getEditor()).toHaveValue("Opening")
      expect(screen.getByTitle("Undo generation")).toBeDisabled()
      expect(snapshot.release).toHaveBeenCalledTimes(1)
    })
  })

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

  it("shows document and selected word counts in the status bar", async () => {
    mockState.storageValues.set("selectedModel", "mock-model")
    seedWritingSession({ prompt: "One two three four." })

    render(<WritingPlayground />)

    expect(screen.getByTestId("writing-status-word-count")).toHaveTextContent(
      "4 words"
    )
    expect(screen.queryByTestId("writing-status-selected-word-count")).toBeNull()

    selectEditorText(getEditor(), "One two")

    await waitFor(() => {
      expect(
        screen.getByTestId("writing-status-selected-word-count")
      ).toHaveTextContent("2 selected")
    })
  })

  it("keeps scene edits separate from session dirty state", async () => {
    seedWritingSession({ prompt: "Saved scene text" })
    useWritingPlaygroundStore.setState({
      activeProjectId: "project-1",
      activeNodeId: "scene-1",
      activeNodeType: "scene"
    })
    mockState.queryData.set(mockState.queryKey(["manuscript-scene", "scene-1"]), {
      id: "scene-1",
      chapter_id: "chapter-1",
      project_id: "project-1",
      title: "Scene 1",
      sort_order: 1,
      content: sceneRichContent("Saved scene text"),
      content_plain: "Saved scene text",
      synopsis: null,
      word_count: 3,
      pov_character_id: null,
      status: "draft",
      created_at: "2026-06-23T12:00:00Z",
      last_modified: "2026-06-23T12:00:00Z",
      deleted: false,
      client_id: "test-client",
      version: 3
    })

    render(<WritingPlayground />)

    await waitFor(() => {
      expect(screen.getByTestId("writing-scene-save-status")).toHaveTextContent(
        "Scene saved"
      )
    })

    fireEvent.change(getEditor(), {
      target: { value: "Edited scene text" }
    })

    expect(screen.getByTestId("writing-scene-save-status")).toHaveTextContent(
      "Scene unsaved"
    )
    expect(screen.queryByText("Unsaved changes")).not.toBeInTheDocument()
  })

  it("keeps a bound scene from being reclaimed by the active session prompt", async () => {
    seedWritingSession({ prompt: "Session draft should not win" })
    useWritingPlaygroundStore.setState({
      activeProjectId: "project-1",
      activeNodeId: "scene-1",
      activeNodeType: "scene"
    })
    mockState.queryData.set(mockState.queryKey(["manuscript-scene", "scene-1"]), {
      id: "scene-1",
      chapter_id: "chapter-1",
      project_id: "project-1",
      title: "Scene 1",
      sort_order: 1,
      content: sceneRichContent("Saved scene text"),
      content_plain: "Saved scene text",
      synopsis: null,
      word_count: 3,
      pov_character_id: null,
      status: "draft",
      created_at: "2026-06-23T12:00:00Z",
      last_modified: "2026-06-23T12:00:00Z",
      deleted: false,
      client_id: "test-client",
      version: 3
    })

    render(<WritingPlayground />)

    await waitFor(() => {
      expect(getEditor()).toHaveValue("Saved scene text")
    })

    await act(async () => {
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(getEditor()).toHaveValue("Saved scene text")

    fireEvent.change(getEditor(), {
      target: { value: "Edited scene text" }
    })

    await act(async () => {
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(getEditor()).toHaveValue("Edited scene text")
    expect(screen.getByTestId("writing-scene-save-status")).toHaveTextContent(
      "Scene unsaved"
    )
    expect(screen.queryByText("Unsaved changes")).not.toBeInTheDocument()
  })

  it("preserves the session prompt when settings autosave while a scene is bound", async () => {
    mockState.executeMutations = true
    vi.mocked(updateWritingSession).mockImplementation(
      async (sessionId, patch, expectedVersion) =>
        ({
          id: sessionId,
          name: "Auto Session",
          payload: patch.payload ?? {},
          schema_version: patch.schema_version ?? 1,
          version_parent_id: null,
          created_at: "2026-03-16T12:00:00Z",
          last_modified: "2026-03-16T12:00:01Z",
          deleted: false,
          client_id: "test-client",
          version: expectedVersion + 1
        }) as Awaited<ReturnType<typeof updateWritingSession>>
    )
    seedWritingSession({ prompt: "Session draft should stay" })
    useWritingPlaygroundStore.setState({
      activeProjectId: "project-1",
      activeNodeId: "scene-1",
      activeNodeType: "scene"
    })
    mockState.queryData.set(mockState.queryKey(["manuscript-scene", "scene-1"]), {
      id: "scene-1",
      chapter_id: "chapter-1",
      project_id: "project-1",
      title: "Scene 1",
      sort_order: 1,
      content: sceneRichContent("Scene body should not leak"),
      content_plain: "Scene body should not leak",
      synopsis: null,
      word_count: 5,
      pov_character_id: null,
      status: "draft",
      created_at: "2026-06-23T12:00:00Z",
      last_modified: "2026-06-23T12:00:00Z",
      deleted: false,
      client_id: "test-client",
      version: 3
    })

    render(<WritingPlayground />)

    await waitFor(() => {
      expect(getEditor()).toHaveValue("Scene body should not leak")
    })

    fireEvent.click(screen.getByRole("button", { name: "Toggle settings" }))
    const streamingToggle = await screen.findByLabelText("Streaming")
    vi.useFakeTimers()
    fireEvent.click(streamingToggle)

    await act(async () => {
      vi.advanceTimersByTime(800)
      await Promise.resolve()
      await Promise.resolve()
    })
    vi.useRealTimers()

    await waitFor(() => {
      expect(updateWritingSession).toHaveBeenCalled()
    })
    const [, updatePayload] = vi.mocked(updateWritingSession).mock.calls.at(-1) ?? []
    expect(updatePayload?.payload).toMatchObject({
      prompt: "Session draft should stay",
      settings: expect.objectContaining({ token_streaming: false })
    })
    expect(updatePayload?.payload?.prompt).not.toBe("Scene body should not leak")
  })

  it("restores the session prompt after leaving a scene during a settings autosave", async () => {
    seedWritingSession({ prompt: "Session draft should return" })
    useWritingPlaygroundStore.setState({
      activeProjectId: "project-1",
      activeNodeId: "scene-1",
      activeNodeType: "scene"
    })
    mockState.queryData.set(mockState.queryKey(["manuscript-scene", "scene-1"]), {
      id: "scene-1",
      chapter_id: "chapter-1",
      project_id: "project-1",
      title: "Scene 1",
      sort_order: 1,
      content: sceneRichContent("Scene body should not leak"),
      content_plain: "Scene body should not leak",
      synopsis: null,
      word_count: 5,
      pov_character_id: null,
      status: "draft",
      created_at: "2026-06-23T12:00:00Z",
      last_modified: "2026-06-23T12:00:00Z",
      deleted: false,
      client_id: "test-client",
      version: 3
    })

    render(<WritingPlayground />)

    await waitFor(() => {
      expect(getEditor()).toHaveValue("Scene body should not leak")
    })

    fireEvent.click(screen.getByRole("button", { name: "Toggle settings" }))
    const streamingToggle = await screen.findByLabelText("Streaming")
    fireEvent.click(streamingToggle)

    act(() => {
      useWritingPlaygroundStore.setState({
        activeNodeId: "chapter-1",
        activeNodeType: "chapter"
      })
    })

    await waitFor(() => {
      expect(getEditor()).toHaveValue("Session draft should return")
    })
  })

  it("blocks generation while a selected scene is still binding", async () => {
    mockState.storageValues.set("selectedModel", "mock-model")
    seedWritingSession({ prompt: "Session draft should not generate" })
    useWritingPlaygroundStore.setState({
      activeProjectId: "project-1",
      activeNodeId: "scene-1",
      activeNodeType: "scene"
    })

    render(<WritingPlayground />)

    const generate = screen.getByTestId("writing-topbar-generate")
    await waitFor(() => {
      expect(generate).toBeDisabled()
    })

    fireEvent.click(generate)

    expect(mockState.sendCalls).toHaveLength(0)
    expect(mockState.streamCalls).toHaveLength(0)
  })

  it("blocks revision queue mutations while a selected scene is still binding", async () => {
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

    act(() => {
      useWritingPlaygroundStore.setState({
        activeProjectId: "project-1",
        activeNodeId: "scene-1",
        activeNodeType: "scene"
      })
    })

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /apply/i })).toBeDisabled()
    })
    expect(screen.getByRole("button", { name: /reject/i })).toBeDisabled()
    expect(screen.getByRole("button", { name: /regenerate/i })).toBeDisabled()

    fireEvent.click(screen.getByRole("button", { name: /apply/i }))
    expect(getEditor()).toHaveValue("Intro. The old sentence. Outro.")
  })

  it("blocks manuscript node switching while generation is pending", async () => {
    mockState.storageValues.set("selectedModel", "mock-model")
    let resolveResponse: (value: string) => void = () => {}
    mockState.sendResponses.push(
      new Promise<string>((resolve) => {
        resolveResponse = resolve
      })
    )
    seedWritingSession({
      prompt: "Session draft.",
      settings: { token_streaming: false }
    })
    seedManuscriptStructure()
    useWritingPlaygroundStore.setState({
      activeProjectId: "project-1",
      activeNodeId: "scene-1",
      activeNodeType: "scene"
    })
    mockState.queryData.set(mockState.queryKey(["manuscript-scene", "scene-1"]), {
      id: "scene-1",
      chapter_id: "chapter-1",
      project_id: "project-1",
      title: "Scene A",
      sort_order: 1,
      content: sceneRichContent("Scene A text"),
      content_plain: "Scene A text",
      synopsis: null,
      word_count: 3,
      pov_character_id: null,
      status: "draft",
      created_at: "2026-06-23T12:00:00Z",
      last_modified: "2026-06-23T12:00:00Z",
      deleted: false,
      client_id: "test-client",
      version: 3
    })

    render(<WritingPlayground />)

    await waitFor(() => {
      expect(getEditor()).toHaveValue("Scene A text")
    })
    fireEvent.click(screen.getByText("Manuscript"))
    fireEvent.click(screen.getByTestId("writing-topbar-generate"))

    await waitFor(() => {
      expect(mockState.sendCalls).toHaveLength(1)
    })

    fireEvent.click(screen.getByText("Scene B (2w)"))

    await act(async () => {
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(useWritingPlaygroundStore.getState().activeNodeId).toBe("scene-1")
    expect(getEditor()).toHaveValue("Scene A text")

    await act(async () => {
      resolveResponse("Generated ending.")
    })
    await waitFor(() => {
      expect(getEditor()).toHaveValue("Scene A textGenerated ending.")
    })
  })

  it("blocks manuscript node switching while a revision request is pending", async () => {
    mockState.storageValues.set("selectedModel", "mock-model")
    let resolveResponse: (value: string) => void = () => {}
    mockState.sendResponses.push(
      new Promise<string>((resolve) => {
        resolveResponse = resolve
      })
    )
    seedWritingSession({ prompt: "Session draft." })
    seedManuscriptStructure()
    useWritingPlaygroundStore.setState({
      activeProjectId: "project-1",
      activeNodeId: "scene-1",
      activeNodeType: "scene"
    })
    mockState.queryData.set(mockState.queryKey(["manuscript-scene", "scene-1"]), {
      id: "scene-1",
      chapter_id: "chapter-1",
      project_id: "project-1",
      title: "Scene A",
      sort_order: 1,
      content: sceneRichContent("Scene A sentence."),
      content_plain: "Scene A sentence.",
      synopsis: null,
      word_count: 3,
      pov_character_id: null,
      status: "draft",
      created_at: "2026-06-23T12:00:00Z",
      last_modified: "2026-06-23T12:00:00Z",
      deleted: false,
      client_id: "test-client",
      version: 3
    })

    render(<WritingPlayground />)

    await waitFor(() => {
      expect(getEditor()).toHaveValue("Scene A sentence.")
    })
    selectEditorText(getEditor(), "Scene A sentence.")
    fireEvent.click(screen.getByText("Manuscript"))
    fireEvent.click(screen.getByRole("button", { name: /rewrite/i }))

    await waitFor(() => {
      expect(mockState.sendCalls).toHaveLength(1)
    })

    fireEvent.click(screen.getByText("Scene B (2w)"))

    await act(async () => {
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(useWritingPlaygroundStore.getState().activeNodeId).toBe("scene-1")
    expect(getEditor()).toHaveValue("Scene A sentence.")

    await act(async () => {
      resolveResponse(structuredReplacement("Scene A replacement."))
    })
    await waitFor(() => {
      expect(screen.getByText("Scene A replacement.")).toBeInTheDocument()
    })
  })

  it("initializes the workflow preset from the active session payload", async () => {
    mockState.storageValues.set("selectedModel", "mock-model")
    mockState.sendResponses.push(structuredReplacement("Voice-preserved line."))
    const preserveVoice = WRITING_REVISION_PRESETS.find(
      (preset) => preset.id === "preserve_voice"
    )
    expect(preserveVoice).toBeTruthy()
    seedWritingSession({
      prompt: "Keep this voice.",
      revision_preset_id: "preserve_voice"
    })

    render(<WritingPlayground />)

    expect(screen.getByText(preserveVoice!.instruction)).toBeInTheDocument()
    selectEditorText(getEditor(), "Keep this voice.")
    fireEvent.click(screen.getByRole("button", { name: /rewrite/i }))

    await waitFor(() => {
      expect(mockState.sendCalls).toHaveLength(1)
    })
    expect(latestRevisionPrompt()).toContain(preserveVoice!.instruction)
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
    expect(screen.getByTestId("writing-revision-pending-count")).toHaveTextContent(
      "1 pending"
    )

    fireEvent.click(screen.getByRole("button", { name: /apply/i }))

    await waitFor(() => {
      expect(editor.value).toBe("Intro. The sharper sentence. Outro.")
    })
  })

  it("creates an annotation suggested-fix proposal without mutating editor text", async () => {
    const sceneText = "Lead 😀 target tail"
    const annotatedPrefix = "Lead 😀 "
    const replacement = "  revised target\n"
    mockState.storageValues.set("selectedModel", "mock-model")
    useWritingPlaygroundStore.setState({
      activeProjectId: "project-1",
      activeNodeId: "scene-1",
      activeNodeType: "scene",
      editorMode: "tiptap"
    })
    seedWritingSession({ prompt: "Session draft should not win." })
    seedManuscriptScene(sceneText)
    seedManuscriptAnnotations([
      makeManuscriptAnnotation({
        id: "annotation-fix",
        body: "Replace the weak noun.",
        suggested_fix: replacement,
        selected_text: "target",
        anchor_start: Array.from(annotatedPrefix).length,
        anchor_end: Array.from(`${annotatedPrefix}target`).length
      })
    ])

    render(<WritingPlayground />)
    const richEditor = await screen.findByLabelText("Mock rich editor")

    await waitFor(() => {
      expect(richEditor).toHaveValue(sceneText)
    })
    fireEvent.click(screen.getByRole("button", { name: "Focus annotation annotation-fix" }))
    fireEvent.click(await screen.findByRole("button", { name: "Create revision" }))

    const queue = screen.getByTestId("writing-revision-queue")
    await waitFor(() => {
      expect(within(queue).getByText("Suggested fix: clarity")).toBeInTheDocument()
    })
    expect(richEditor).toHaveValue(sceneText)
    expect(screen.getByTestId("writing-revision-pending-count")).toHaveTextContent(
      "1 pending"
    )
    expect(
      within(queue).getByText((_, element) =>
        element?.tagName === "PRE" && element.textContent === "target"
      )
    ).toBeInTheDocument()
    expect(
      within(queue).getByText((_, element) =>
        element?.tagName === "PRE" && element.textContent === replacement
      )
    ).toBeInTheDocument()
  })

  it("continues from the selected text end and applies an insertion", async () => {
    mockState.storageValues.set("selectedModel", "mock-model")
    mockState.sendResponses.push(structuredReplacement(" continued"))
    seedWritingSession({ prompt: "Intro sentence. Outro." })

    render(<WritingPlayground />)
    const editor = getEditor()
    selectEditorText(editor, "sentence")
    fireEvent.click(screen.getByRole("button", { name: /continue/i }))

    await waitFor(() => {
      expect(mockState.sendCalls).toHaveLength(1)
    })
    fireEvent.click(screen.getByRole("button", { name: /apply/i }))

    await waitFor(() => {
      expect(editor.value).toBe("Intro sentence continued. Outro.")
    })
  })

  it("does not require broad-target confirmation for Continue", async () => {
    mockState.storageValues.set("selectedModel", "mock-model")
    mockState.sendResponses.push(structuredReplacement("Opening "))
    seedWritingSession({
      prompt: "Long paragraph ".repeat(120)
    })

    render(<WritingPlayground />)
    fireEvent.click(screen.getByRole("button", { name: /continue/i }))

    await waitFor(() => {
      expect(mockState.sendCalls).toHaveLength(1)
    })
    expect(latestRevisionPrompt()).toContain("Operation: insert")
  })

  it("refreshes the action-bar target when text is selected after a broad target renders", async () => {
    mockState.storageValues.set("selectedModel", "mock-model")
    mockState.sendResponses.push(structuredReplacement("Specific rewrite."))
    seedWritingSession({
      prompt: `${"Long paragraph ".repeat(180)}target phrase.`
    })

    render(<WritingPlayground />)
    expect(
      screen.getByLabelText(/confirm whole-document text change/i)
    ).toBeInTheDocument()

    selectEditorText(getEditor(), "target phrase.")

    await waitFor(() => {
      expect(
        screen.queryByLabelText(/confirm whole-document text change/i)
      ).toBeNull()
    })
    fireEvent.click(screen.getByRole("button", { name: /rewrite/i }))

    await waitFor(() => {
      expect(screen.getByText("Specific rewrite.")).toBeInTheDocument()
    })
    expect(latestRevisionPrompt()).toContain("Target summary: selection")
  })

  it("refreshes the rich-editor action-bar target when rich text is selected", async () => {
    mockState.storageValues.set("selectedModel", "mock-model")
    mockState.sendResponses.push(structuredReplacement("Rich specific rewrite."))
    useWritingPlaygroundStore.setState({ editorMode: "tiptap" })
    seedWritingSession({
      prompt: `${"Long paragraph ".repeat(180)}target phrase.`
    })

    render(<WritingPlayground />)
    fireEvent.click(screen.getByRole("radio", { name: "Rich" }))
    const richEditor = await screen.findByLabelText("Mock rich editor") as HTMLTextAreaElement
    expect(
      screen.getByLabelText(/confirm whole-document text change/i)
    ).toBeInTheDocument()

    selectEditorText(richEditor, "target phrase.")

    await waitFor(() => {
      expect(
        screen.queryByLabelText(/confirm whole-document text change/i)
      ).toBeNull()
    })
    fireEvent.click(screen.getByRole("button", { name: /rewrite/i }))

    await waitFor(() => {
      expect(screen.getByText("Rich specific rewrite.")).toBeInTheDocument()
    })
    expect(latestRevisionPrompt()).toContain("Target summary: selection")
  })

  it("keeps the topbar Generate control non-cancelable during revision proposal requests", async () => {
    mockState.storageValues.set("selectedModel", "mock-model")
    let resolveResponse: (value: string) => void = () => {}
    mockState.sendResponses.push(
      new Promise<string>((resolve) => {
        resolveResponse = resolve
      })
    )
    seedWritingSession({
      prompt: "Rewrite this line.",
      settings: { token_streaming: true }
    })

    render(<WritingPlayground />)
    selectEditorText(getEditor(), "Rewrite this line.")
    fireEvent.click(screen.getByRole("button", { name: /rewrite/i }))

    await waitFor(() => {
      expect(mockState.sendCalls).toHaveLength(1)
    })
    const generateButton = screen.getByTestId("writing-topbar-generate")
    expect(generateButton).toHaveTextContent("Generate")
    expect(generateButton).toBeDisabled()

    await act(async () => {
      resolveResponse(structuredReplacement("Resolved rewrite."))
    })
    await waitFor(() => {
      expect(screen.getByText("Resolved rewrite.")).toBeInTheDocument()
    })
  })

  it("blocks session switching while a revision proposal request is pending", async () => {
    mockState.storageValues.set("selectedModel", "mock-model")
    let resolveResponse: (value: string) => void = () => {}
    mockState.sendResponses.push(
      new Promise<string>((resolve) => {
        resolveResponse = resolve
      })
    )
    seedWritingSession({ prompt: "Rewrite this line." })
    mockState.queryData.set(mockState.queryKey(["writing-sessions"]), {
      sessions: [
        {
          id: "session-auto",
          name: "Auto Session",
          last_modified: "2026-03-16T12:00:00Z",
          version: 1
        },
        {
          id: "session-other",
          name: "Other Session",
          last_modified: "2026-03-16T12:05:00Z",
          version: 1
        }
      ],
      total: 2,
      limit: 200,
      offset: 0
    })
    mockState.queryData.set(
      mockState.queryKey(["writing-session", "session-other"]),
      {
        id: "session-other",
        name: "Other Session",
        payload: {
          prompt: "Other draft.",
          settings: {},
          template_name: null,
          theme_name: null,
          chat_mode: false
        },
        schema_version: 1,
        version_parent_id: null,
        created_at: "2026-03-16T12:00:00Z",
        last_modified: "2026-03-16T12:05:00Z",
        deleted: false,
        client_id: "test-client",
        version: 1
      }
    )

    render(<WritingPlayground />)
    selectEditorText(getEditor(), "Rewrite this line.")
    fireEvent.click(screen.getByRole("button", { name: /rewrite/i }))

    await waitFor(() => {
      expect(mockState.sendCalls).toHaveLength(1)
    })
    fireEvent.click(screen.getByRole("button", { name: /Other Session/i }))

    expect(useWritingPlaygroundStore.getState().activeSessionId).toBe(
      "session-auto"
    )
    expect(getEditor()).toHaveValue("Rewrite this line.")

    await act(async () => {
      resolveResponse(structuredReplacement("Resolved rewrite."))
    })
    await waitFor(() => {
      expect(screen.getByText("Resolved rewrite.")).toBeInTheDocument()
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
        advanced_extra_body: {
          safe_key: "kept",
          api_key: "do-not-leak",
          banned_tokens: ["cliche"]
        },
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
      maxTokens: 333,
      extraBody: {
        safe_key: "kept",
        banned_tokens: ["cliche"]
      }
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
    const proposals = within(queue).getAllByTestId("writing-revision-proposal")
    expect(proposals).toHaveLength(2)
    expect(proposals[1]).toHaveAttribute(
      "data-regenerated-from-id",
      proposals[0].getAttribute("data-proposal-id")
    )
  })

  it("shows manual-apply guidance for rich editor apply without mutating content", async () => {
    mockState.storageValues.set("selectedModel", "mock-model")
    mockState.sendResponses.push(structuredReplacement("Rich replacement."))
    useWritingPlaygroundStore.setState({ editorMode: "tiptap" })
    seedWritingSession({ prompt: "Rich original." })

    render(<WritingPlayground />)
    fireEvent.click(screen.getByRole("radio", { name: "Rich" }))
    const richEditor = await screen.findByLabelText("Mock rich editor")
    expect(richEditor).toHaveValue("Rich original.")
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
    expect(richEditor).toHaveValue("Rich original.")
  })

  it("allows confirmed whole-document text-changing targets to create applyable proposals", async () => {
    mockState.storageValues.set("selectedModel", "mock-model")
    mockState.sendResponses.push(structuredReplacement("Whole document rewrite."))
    seedWritingSession({
      prompt: `${"Long paragraph ".repeat(180)}\n\nSecond paragraph.`
    })

    render(<WritingPlayground />)
    const editor = getEditor()
    editor.focus()
    editor.setSelectionRange(0, 0)
    fireEvent.select(editor)
    fireEvent.click(
      await screen.findByLabelText(/confirm whole-document text change/i)
    )
    fireEvent.click(screen.getByRole("button", { name: /rewrite/i }))

    await waitFor(() => {
      expect(screen.getByText("Whole document rewrite.")).toBeInTheDocument()
    })
    const queue = screen.getByTestId("writing-revision-queue")
    expect(within(queue).getByRole("button", { name: /apply/i })).toBeEnabled()
    expect(latestRevisionPrompt()).toContain("Target summary: whole document")
  })
})
