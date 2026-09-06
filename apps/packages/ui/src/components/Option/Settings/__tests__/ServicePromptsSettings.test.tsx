import React from "react"
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within
} from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { App } from "antd"
import {
  BrowserRouter,
  HashRouter,
  Link,
  Route,
  Routes,
  useNavigate
} from "react-router-dom"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { ServicePromptsSettings } from "../ServicePromptsSettings"
import {
  ServicePromptApiError,
  type ServicePromptCatalogItem,
  type ServicePromptDetail
} from "@/services/tldw/domains/service-prompts"
import {
  requestSettingsNavigation,
  SETTINGS_NAVIGATION_REQUEST_EVENT
} from "@/utils/settings-return"

const mocks = vi.hoisted(() => ({
  confirmDanger: vi.fn(),
  initialize: vi.fn(),
  resolveScope: vi.fn(),
  readLegacy: vi.fn(),
  clearLegacy: vi.fn(),
  importLegacy: vi.fn(),
  subscribeConfig: vi.fn(),
  renderPart: vi.fn(),
  list: vi.fn(),
  get: vi.fn(),
  save: vi.fn(),
  reset: vi.fn()
}))

vi.mock("@/components/Common/confirm-danger", async () => {
  const actual = await vi.importActual<
    typeof import("@/components/Common/confirm-danger")
  >("@/components/Common/confirm-danger")
  return {
    ...actual,
    useConfirmDanger: () => {
      const confirmDanger = actual.useConfirmDanger()
      return (options: Parameters<typeof confirmDanger>[0]) =>
        mocks.confirmDanger(options, confirmDanger)
    }
  }
})

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: (...args: unknown[]) => mocks.initialize(...args),
    listServicePrompts: (...args: unknown[]) => mocks.list(...args),
    getServicePrompt: (...args: unknown[]) => mocks.get(...args),
    saveServicePrompt: (...args: unknown[]) => mocks.save(...args),
    resetServicePrompt: (...args: unknown[]) => mocks.reset(...args)
  }
}))

vi.mock("@/services/service-prompts", async () => {
  const actual = await vi.importActual<
    typeof import("@/services/service-prompts")
  >("@/services/service-prompts")
  return {
    ...actual,
    resolveServicePromptScope: (...args: unknown[]) =>
      mocks.resolveScope(...args),
    readLegacyServicePromptCandidates: (...args: unknown[]) =>
      mocks.readLegacy(...args),
    clearLegacyServicePromptCandidate: (...args: unknown[]) =>
      mocks.clearLegacy(...args),
    importLegacyServicePromptCandidate: (...args: unknown[]) =>
      mocks.importLegacy(...args),
    subscribeToServicePromptConfigChanges: (...args: unknown[]) =>
      mocks.subscribeConfig(...args),
    renderServicePromptPart: (...args: Parameters<typeof actual.renderServicePromptPart>) => {
      mocks.renderPart(...args)
      return actual.renderServicePromptPart(...args)
    }
  }
})

const translate = vi.hoisted(() => (
  _key: string,
  fallbackOrOptions?: string | { defaultValue?: string; [key: string]: unknown }
) => {
  const options = typeof fallbackOrOptions === "string"
    ? { defaultValue: fallbackOrOptions }
    : fallbackOrOptions
  let value = options?.defaultValue ?? _key
  for (const [key, replacement] of Object.entries(options ?? {})) {
    value = value.replaceAll(`{{${key}}}`, String(replacement))
  }
  return value
})

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: translate
  })
}))

const catalog: ServicePromptCatalogItem[] = [
  {
    id: "chat.rag.answer",
    label: "Server RAG answer",
    description: "Server description",
    parts: [{
      key: "template",
      label: "Template",
      mode: "template",
      required_variables: ["context", "question"]
    }],
    affected_workflows: [
      { id: "chat.main.rag", label: "Main Chat RAG" },
      { id: "chat.tab.rag", label: "Tab Chat RAG" },
      { id: "chat.document.rag", label: "Document Chat RAG" },
      { id: "chat.sidepanel.rag", label: "Legacy Sidepanel RAG" }
    ]
  },
  {
    id: "chat.rag.question_rewrite",
    label: "Server rewrite",
    description: "Server rewrite description",
    parts: [{
      key: "template",
      label: "Template",
      mode: "template",
      required_variables: ["chat_history", "question"]
    }],
    affected_workflows: [
      { id: "chat.main.rag", label: "Main Chat RAG" },
      { id: "chat.document.rag", label: "Document Chat RAG" },
      { id: "chat.sidepanel.rag", label: "Legacy Sidepanel RAG" }
    ]
  },
  {
    id: "chat.web_search.answer",
    label: "Server web answer",
    description: "Server web description",
    parts: [{
      key: "template",
      label: "Template",
      mode: "template",
      required_variables: ["current_date_time", "search_results"]
    }],
    affected_workflows: [
      { id: "chat.main.web_search", label: "Main Chat web search" },
      { id: "chat.compare.web_search", label: "Compare web search" }
    ]
  },
  {
    id: "media.text.translation",
    label: "Server translation",
    description: "Server translation description",
    parts: [
      {
        key: "system",
        label: "System instructions",
        mode: "literal",
        required_variables: []
      },
      {
        key: "user_template",
        label: "User template",
        mode: "template",
        required_variables: ["target_language", "text"]
      }
    ],
    affected_workflows: [
      { id: "media.text.translation", label: "Text translation" }
    ]
  },
  {
    id: "chat.title.generation",
    label: "Server title prompt",
    description: "Server title prompt description",
    parts: [{
      key: "user_template",
      label: "User template",
      mode: "template",
      required_variables: ["query"]
    }],
    affected_workflows: [
      { id: "chat.title.generation", label: "Server title workflow" }
    ]
  },
  {
    id: "image.prompt.refinement",
    label: "Server image refinement prompt",
    description: "Server image refinement prompt description",
    parts: [
      {
        key: "system_semantics",
        label: "Server refinement guidance",
        mode: "literal",
        required_variables: []
      },
      {
        key: "rewrite_semantics",
        label: "Server rewrite guidance",
        mode: "literal",
        required_variables: []
      }
    ],
    affected_workflows: [
      { id: "image.prompt.refinement", label: "Server image workflow" }
    ]
  },
  {
    id: "media.document.summarization",
    label: "Server document prompt",
    description: "Server document prompt description",
    parts: [{ key: "system", label: "System instructions", mode: "literal", required_variables: [] }],
    affected_workflows: [{ id: "media.document.summarization", label: "Server document workflow" }]
  },
  {
    id: "media.pdf.summarization",
    label: "Server PDF prompt",
    description: "Server PDF prompt description",
    parts: [{ key: "system", label: "System instructions", mode: "literal", required_variables: [] }],
    affected_workflows: [{ id: "media.pdf.summarization", label: "Server PDF workflow" }]
  },
  {
    id: "media.ebook.summarization",
    label: "Server EPUB prompt",
    description: "Server EPUB prompt description",
    parts: [{ key: "system", label: "System instructions", mode: "literal", required_variables: [] }],
    affected_workflows: [{ id: "media.ebook.summarization", label: "Server EPUB workflow" }]
  },
  {
    id: "media.email.summarization",
    label: "Server email prompt",
    description: "Server email prompt description",
    parts: [{ key: "system", label: "System instructions", mode: "literal", required_variables: [] }],
    affected_workflows: [{ id: "media.email.summarization", label: "Server email workflow" }]
  },
  {
    id: "media.audio.analysis",
    label: "Server audio prompt",
    description: "Server audio description",
    parts: [
      { key: "system", label: "System instructions", mode: "literal", required_variables: [] },
      { key: "user", label: "Server user label", mode: "literal", required_variables: [] }
    ],
    affected_workflows: [{ id: "media.audio.analysis", label: "Server audio workflow" }]
  },
  {
    id: "media.video.summarization",
    label: "Server video prompt",
    description: "Server video description",
    parts: [
      { key: "system", label: "System instructions", mode: "literal", required_variables: [] },
      { key: "final_summary", label: "Server final label", mode: "literal", required_variables: [] }
    ],
    affected_workflows: [{ id: "media.video.summarization", label: "Server video workflow" }]
  },
  {
    id: "media.web.summarization",
    label: "Server web prompt",
    description: "Server web description",
    parts: [
      { key: "system", label: "System instructions", mode: "literal", required_variables: [] },
      { key: "user", label: "User instructions", mode: "literal", required_variables: [] }
    ],
    affected_workflows: [{ id: "media.web.summarization", label: "Server web workflow" }]
  },
  {
    id: "notes.title.generate",
    label: "Server Notes title prompt",
    description: "Server Notes title prompt description",
    parts: [
      {
        key: "system",
        label: "Server system label",
        mode: "literal",
        required_variables: []
      },
      {
        key: "title_instruction",
        label: "Server title instruction label",
        mode: "literal",
        required_variables: []
      }
    ],
    affected_workflows: [
      { id: "notes.title.generate", label: "Server Notes title workflow" }
    ]
  },
  {
    id: "media.document.insights",
    label: "Server insights prompt",
    description: "Server insights description",
    parts: [
      { key: "analysis_guidance", label: "Analysis guidance", mode: "literal", required_variables: [] },
      { key: "presentation_guidance", label: "Presentation guidance", mode: "literal", required_variables: [] }
    ],
    affected_workflows: [{ id: "media.document.insights", label: "Server insights workflow" }]
  }
]

const detailFor = (
  definition: ServicePromptCatalogItem,
  options: {
    source?: "packaged" | "user"
    revision?: string | null
    parts?: Record<string, string>
  } = {}
): ServicePromptDetail => {
  const defaults = definition.id === "media.document.insights"
    ? { analysis_guidance: "Extract research insights.", presentation_guidance: "Use concise titles." }
    : definition.id === "media.web.summarization"
    ? { system: "Web system guidance.", user: "Web summary guidance." }
    : definition.id === "media.video.summarization"
    ? { system: "Video system guidance.", final_summary: "Combine video summaries." }
    : definition.id === "media.audio.analysis"
    ? { system: "Audio system guidance.", user: "Audio user guidance." }
    : definition.id === "media.email.summarization"
    ? { system: "Summarize the email clearly." }
    : definition.id === "media.ebook.summarization"
    ? { system: "Summarize the EPUB clearly." }
    : definition.id === "media.pdf.summarization"
    ? { system: "Summarize the PDF clearly." }
    : definition.id === "media.document.summarization"
    ? { system: "Summarize the document clearly." }
    : definition.id === "media.text.translation"
    ? {
        system: "Translate accurately. Literal {braces} stay literal.",
        user_template: "Translate to {target_language}:\n{text}"
      }
    : definition.id === "chat.rag.answer"
      ? { template: "Context: {context}\nQuestion: {question}" }
    : definition.id === "chat.rag.question_rewrite"
      ? { template: "History: {chat_history}\nQuestion: {question}" }
      : definition.id === "chat.title.generation"
        ? { user_template: "Create a short title for {query}" }
        : definition.id === "image.prompt.refinement"
          ? {
              system_semantics:
                "You refine image-generation prompts. Preserve intent.",
              rewrite_semantics: "Return a generation-ready prompt."
            }
          : definition.id === "notes.title.generate"
            ? {
                system: "Write concise document titles.",
                title_instruction: "Write a descriptive title"
              }
            : { template: "At {current_date_time}:\n{search_results}" }
  const effective = options.parts ?? defaults
  const source = options.source ?? "packaged"
  return {
    ...definition,
    default_parts: defaults,
    saved_parts: source === "user" ? effective : null,
    effective_parts: effective,
    source,
    revision: source === "user"
      ? options.revision ?? "11111111-1111-4111-8111-111111111111"
      : null
  }
}

const scopeOne = {
  config: {
    serverUrl: "https://research-one.test",
    authMode: "multi-user" as const
  },
  scopeKey: "server:research-one:auth:multi-user:org:none:user:42",
  userId: 42
}

const scopeTwo = {
  config: {
    serverUrl: "https://research-two.test",
    authMode: "multi-user" as const
  },
  scopeKey: "server:research-two:auth:multi-user:org:none:user:84",
  userId: 84
}

const rotatedScopeOne = {
  ...scopeOne,
  config: { ...scopeOne.config }
}

const accountTwoSameServer = {
  config: {
    ...scopeOne.config
  },
  scopeKey: "server:research-one:auth:multi-user:org:none:user:84",
  userId: 84
}

const legacyRagCandidate = {
  definitionId: "chat.rag.answer",
  partKey: "template",
  storageKey: "systemPromptForRag",
  value: "Legacy {context} {question}"
}

const deferred = <T,>() => {
  let resolve!: (value: T) => void
  let reject!: (reason: unknown) => void
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise
    reject = rejectPromise
  })
  return { promise, reject, resolve }
}

const createClient = () => new QueryClient({
  defaultOptions: {
    queries: { retry: false, gcTime: Infinity },
    mutations: { retry: false }
  }
})

const installNextStyleHistory = () => {
  const replaceState = window.history.replaceState.bind(window.history)
  const pushState = window.history.pushState.bind(window.history)
  let key = 0
  const nextState = (state: unknown) => {
    const record = state && typeof state === "object"
      ? { ...state as Record<string, unknown> }
      : {}
    delete record.idx
    return {
      ...record,
      __N: true,
      hostMarker: "preserved",
      key: typeof record.key === "string" ? record.key : `next-${++key}`
    }
  }
  vi.spyOn(window.history, "replaceState").mockImplementation(
    (state, unused, url) => replaceState(nextState(state), unused, url)
  )
  vi.spyOn(window.history, "pushState").mockImplementation(
    (state, unused, url) => pushState(nextState(state), unused, url)
  )
}

const renderSettings = (client = createClient()) => ({
  client,
  ...render(
    <BrowserRouter>
      <QueryClientProvider client={client}>
        <App>
          <Routes>
            <Route
              path="/settings/prompt"
              element={(
                <>
                  <ServicePromptsSettings />
                  <Link to="/settings/chat">Chat settings test route</Link>
                </>
              )}
            />
            <Route path="*" element={<p>Outside workflow prompt route</p>} />
          </Routes>
        </App>
      </QueryClientProvider>
    </BrowserRouter>
  )
})

const DelayedRouteLink = () => {
  const navigate = useNavigate()
  return (
    <a
      href="/settings/chat"
      onClick={(event) => {
        event.preventDefault()
        window.setTimeout(() => navigate("/settings/chat"), 0)
      }}
    >
      Delayed settings test route
    </a>
  )
}

const ProgrammaticSettingsNavigation = ({ destination }: {
  destination: string
}) => {
  const navigate = useNavigate()
  return (
    <button
      type="button"
      onClick={() => {
        if (requestSettingsNavigation(destination)) navigate(destination)
      }}
    >
      Programmatic settings navigation
    </button>
  )
}

const openPrompt = async (name: string) => {
  fireEvent.click(await screen.findByRole("button", { name }))
  await screen.findByRole("heading", { name })
}

describe("ServicePromptsSettings", () => {
  beforeEach(() => {
    vi.resetAllMocks()
    window.history.replaceState({}, "", "/settings/prompt")
    mocks.initialize.mockResolvedValue(undefined)
    mocks.resolveScope.mockResolvedValue(scopeOne)
    mocks.list.mockResolvedValue(catalog)
    mocks.get.mockImplementation(async (id: string) => {
      const definition = catalog.find((item) => item.id === id)
      if (!definition) throw new Error("unknown")
      return detailFor(definition)
    })
    mocks.save.mockImplementation(async (id: string, payload: {
      parts: Record<string, string>
    }) => detailFor(
      catalog.find((item) => item.id === id)!,
      { source: "user", parts: payload.parts }
    ))
    mocks.reset.mockImplementation(async (id: string) => detailFor(
      catalog.find((item) => item.id === id)!
    ))
    mocks.readLegacy.mockResolvedValue([])
    mocks.clearLegacy.mockResolvedValue(undefined)
    mocks.importLegacy.mockImplementation(async (candidate: {
      definitionId: string
      value: string
    }, detail: ServicePromptDetail) => detailFor(
      catalog.find((item) => item.id === candidate.definitionId)!,
      {
        source: "user",
        parts: { ...detail.effective_parts, template: candidate.value }
      }
    ))
    mocks.subscribeConfig.mockImplementation(() => () => undefined)
    mocks.confirmDanger.mockImplementation((options, confirmDanger) =>
      confirmDanger(options)
    )
    vi.spyOn(window, "confirm").mockReturnValue(true)
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("shows loading, a retryable disconnected error, and no migration probe on failure", async () => {
    let rejectScope!: (reason: unknown) => void
    mocks.resolveScope.mockReturnValue(new Promise((_resolve, reject) => {
      rejectScope = reject
    }))
    renderSettings()

    const loading = screen.getByRole("status", {
      name: "Loading server and account scope…"
    })
    expect(loading).toHaveAttribute("aria-busy", "true")
    expect(loading.querySelector(".ant-skeleton-active")).toBeNull()

    await act(async () => {
      rejectScope(new Error("offline"))
    })
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Unable to resolve the connected server and account."
    )
    expect(mocks.readLegacy).not.toHaveBeenCalled()

    mocks.resolveScope.mockResolvedValue(scopeOne)
    fireEvent.click(screen.getByRole("button", { name: "Retry" }))
    expect(await screen.findByRole("button", { name: "RAG answer" }))
      .toBeInTheDocument()
    expect(document.querySelector("h1")).toBeNull()
  })

  it("renders the localized definitions, query selection, status, workflows, and exact scope", async () => {
    window.history.replaceState(
      {},
      "",
      "/settings/prompt?prompt=media.text.translation"
    )
    renderSettings()

    expect(await screen.findAllByTestId("service-prompt-list-item"))
      .toHaveLength(15)
    expect(await screen.findByRole("heading", { name: "Text translation" }))
      .toBeInTheDocument()
    expect(screen.getByText("Server default")).toBeInTheDocument()
    expect(screen.getByText("Text translation", { selector: "li" }))
      .toBeInTheDocument()
    expect(screen.getByText("https://research-one.test")).toBeInTheDocument()
    expect(screen.getByText(scopeOne.scopeKey)).toBeInTheDocument()
    expect(screen.queryByText("Server translation")).not.toBeInTheDocument()
  })

  it("localizes and edits the conversation title prompt with Chat settings guidance", async () => {
    renderSettings()

    await openPrompt("Conversation title")

    expect(screen.getByRole("heading", { name: "Conversation title" }))
      .toBeVisible()
    const userTemplate = screen.getByLabelText("User template") as HTMLTextAreaElement
    expect(userTemplate.value).toContain("{query}")
    expect(screen.getByText("Automatic conversation titles")).toBeVisible()
    expect(screen.getByRole("link", { name: "Open Chat settings" }))
      .toHaveAttribute("href", "/settings/chat")
  })

  it("localizes and edits the Notes title prompt", async () => {
    renderSettings()

    await openPrompt("Notes title")

    expect(screen.getByRole("heading", { name: "Notes title" })).toBeVisible()
    expect(screen.getByLabelText("System instructions")).toHaveValue(
      "Write concise document titles."
    )
    expect(screen.getByLabelText("Title instruction")).toHaveValue(
      "Write a descriptive title"
    )
    expect(screen.getByText("Automatic Notes titles")).toBeVisible()
    expect(screen.queryByText("Server Notes title prompt")).not.toBeInTheDocument()
  })

  it("exposes document system guidance and identifies its synchronous scope", async () => {
    renderSettings()
    await openPrompt("Document summarization")
    expect(screen.getByLabelText("System instructions")).toHaveValue(
      "Summarize the document clearly."
    )
    expect(screen.getByText("Synchronous document analysis")).toBeVisible()
    expect(screen.getAllByText(/Without a saved override, server defaults apply/)[0]).toBeVisible()
  })

  it("exposes independent PDF system guidance and its synchronous scope", async () => {
    renderSettings()
    await openPrompt("PDF summarization")
    expect(screen.getByLabelText("System instructions")).toHaveValue(
      "Summarize the PDF clearly."
    )
    expect(screen.getByText("Synchronous PDF analysis")).toBeVisible()
    expect(screen.getAllByText(/Without a saved override, server defaults apply/)[0]).toBeVisible()
    expect(screen.queryByText("Server PDF prompt")).not.toBeInTheDocument()
  })

  it("edits document insights guidance without exposing its JSON contract", async () => {
    renderSettings()
    await openPrompt("Document Insights")
    expect(screen.getByLabelText("Analysis guidance")).toHaveValue("Extract research insights.")
    expect(screen.getByLabelText("Presentation guidance")).toHaveValue("Use concise titles.")
    expect(screen.getByText("Document workspace insights")).toBeVisible()
    expect(screen.getByText(/JSON output requirements and requested categories remain fixed/, { selector: "p" })).toBeVisible()
    fireEvent.change(screen.getByLabelText("Analysis guidance"), { target: { value: "Focus on methods {literally}" } })
    fireEvent.change(screen.getByLabelText("Presentation guidance"), { target: { value: "Write in French" } })
    fireEvent.click(screen.getByRole("button", { name: "Save changes" }))
    await waitFor(() => expect(mocks.save).toHaveBeenCalledWith(
      "media.document.insights",
      { parts: { analysis_guidance: "Focus on methods {literally}", presentation_guidance: "Write in French" }, expected_revision: null },
      { signal: expect.any(AbortSignal), requestScope: scopeOne }
    ))
  })

  it("edits web system and summary instructions as one pair", async () => {
    renderSettings()
    await openPrompt("Web article summarization")
    expect(screen.getByLabelText("System instructions")).toHaveValue("Web system guidance.")
    expect(screen.getByLabelText("User instructions")).toHaveValue("Web summary guidance.")
    expect(screen.getByText("Synchronous web scraping and ingestion")).toBeVisible()
    expect(screen.getByText(/Reset restores each scraping engine/, { selector: "p" })).toBeVisible()
    fireEvent.change(screen.getByLabelText("System instructions"), { target: { value: "Web {system}" } })
    fireEvent.change(screen.getByLabelText("User instructions"), { target: { value: "Web {summary}" } })
    fireEvent.click(screen.getByRole("button", { name: "Save changes" }))
    await waitFor(() => expect(mocks.save).toHaveBeenCalledWith(
      "media.web.summarization",
      { parts: { system: "Web {system}", user: "Web {summary}" }, expected_revision: null },
      { signal: expect.any(AbortSignal), requestScope: scopeOne }
    ))
  })

  it("edits video system and final-summary instructions as one pair", async () => {
    renderSettings()
    await openPrompt("Video summarization")
    expect(screen.getByLabelText("System instructions")).toHaveValue("Video system guidance.")
    expect(screen.getByLabelText("Final-summary instructions")).toHaveValue("Combine video summaries.")
    expect(screen.getByText("Synchronous video analysis")).toBeVisible()
    fireEvent.change(screen.getByLabelText("System instructions"), { target: { value: "Video {system}" } })
    fireEvent.change(screen.getByLabelText("Final-summary instructions"), { target: { value: "Video {final}" } })
    fireEvent.click(screen.getByRole("button", { name: "Save changes" }))
    await waitFor(() => expect(mocks.save).toHaveBeenCalledWith(
      "media.video.summarization",
      { parts: { system: "Video {system}", final_summary: "Video {final}" }, expected_revision: null },
      { signal: expect.any(AbortSignal), requestScope: scopeOne }
    ))
  })

  it("edits audio system and user instructions as one pair", async () => {
    renderSettings()
    await openPrompt("Audio summarization")
    expect(screen.getByLabelText("System instructions")).toHaveValue("Audio system guidance.")
    expect(screen.getByLabelText("User instructions")).toHaveValue("Audio user guidance.")
    expect(screen.getByText("Synchronous audio analysis")).toBeVisible()
    fireEvent.change(screen.getByLabelText("System instructions"), { target: { value: "Audio {system}" } })
    fireEvent.change(screen.getByLabelText("User instructions"), { target: { value: "Audio {user}" } })
    fireEvent.click(screen.getByRole("button", { name: "Save changes" }))
    await waitFor(() => expect(mocks.save).toHaveBeenCalledWith(
      "media.audio.analysis",
      { parts: { system: "Audio {system}", user: "Audio {user}" }, expected_revision: null },
      { signal: expect.any(AbortSignal), requestScope: scopeOne }
    ))
  })

  it("exposes independent email system guidance and its synchronous scope", async () => {
    renderSettings()
    await openPrompt("Email summarization")
    expect(screen.getByLabelText("System instructions")).toHaveValue(
      "Summarize the email clearly."
    )
    expect(screen.getByText("Synchronous email analysis")).toBeVisible()
    expect(screen.getAllByText(/Without a saved override, server defaults apply/)[0]).toBeVisible()
    expect(screen.queryByText("Server email prompt")).not.toBeInTheDocument()
  })

  it("exposes independent EPUB system guidance and its synchronous scope", async () => {
    renderSettings()
    await openPrompt("EPUB summarization")
    expect(screen.getByLabelText("System instructions")).toHaveValue(
      "Summarize the EPUB clearly."
    )
    expect(screen.getByText("Synchronous EPUB analysis")).toBeVisible()
    expect(screen.getAllByText(/Without a saved override, server defaults apply/)[0]).toBeVisible()
    expect(screen.queryByText("Server EPUB prompt")).not.toBeInTheDocument()
  })

  it("localizes and edits the image prompt refinement semantics", async () => {
    renderSettings()

    await openPrompt("Image prompt refinement")

    expect(screen.getByRole("heading", { name: "Image prompt refinement" }))
      .toBeVisible()
    expect(screen.getByLabelText("Refinement guidance")).toHaveValue(
      "You refine image-generation prompts. Preserve intent."
    )
    expect(screen.getByLabelText("Rewrite guidance")).toHaveValue(
      "Return a generation-ready prompt."
    )
    expect(screen.getByText("Image prompt refinement", { selector: "li" }))
      .toBeVisible()
    expect(screen.queryByText("Server image refinement prompt"))
      .not.toBeInTheDocument()
  })

  it("uses the Prompts Link and reverses dirty HashRouter Back and Forward", async () => {
    window.history.replaceState(
      { hostMarker: "hash-a" },
      "",
      "/options.html#/prompts"
    )
    window.history.pushState(
      { hostMarker: "hash-b" },
      "",
      "/options.html#/settings/prompt?prompt=chat.rag.answer"
    )
    const client = createClient()
    render(
      <HashRouter>
        <QueryClientProvider client={client}>
          <App>
            <Routes>
              <Route path="/settings/prompt" element={<ServicePromptsSettings />} />
              <Route path="/prompts" element={<p>Reusable prompts test route</p>} />
            </Routes>
          </App>
        </QueryClientProvider>
      </HashRouter>
    )

    const libraryLink = await screen.findByRole("link", {
      name: "Open reusable Prompts workspace"
    })
    expect(libraryLink).toHaveAttribute("href", "#/prompts")
    fireEvent.click(libraryLink)

    expect(await screen.findByText("Reusable prompts test route"))
      .toBeInTheDocument()
    expect(window.location.pathname).toBe("/options.html")
    expect(window.location.hash).toBe("#/prompts")
    const destinationToken = window.history.state
      .servicePromptHistoryEntryToken as string
    expect(destinationToken).toEqual(expect.any(String))
    const historyLength = window.history.length

    await act(async () => {
      window.history.back()
      await new Promise((resolve) => setTimeout(resolve, 0))
    })
    await screen.findByRole("heading", { name: "RAG answer" })
    expect(window.history.state).toMatchObject({
      hostMarker: "hash-b",
      servicePromptHistoryForwardEntryToken: destinationToken
    })

    const authoredValue = "Hash dirty {context} {question}"
    const editor = screen.getByRole("textbox", { name: "Template" })
    fireEvent.change(editor, { target: { value: authoredValue } })
    editor.focus()
    vi.mocked(window.confirm).mockReturnValue(false)

    const confirmationsBeforeBack = vi.mocked(window.confirm).mock.calls.length
    await act(async () => {
      window.history.back()
      await new Promise((resolve) => setTimeout(resolve, 0))
    })
    await waitFor(() => {
      expect(window.confirm).toHaveBeenCalledTimes(confirmationsBeforeBack + 1)
    })
    await waitFor(() => {
      expect(window.location.hash).toContain("#/settings/prompt")
    })
    await screen.findByRole("heading", { name: "RAG answer" })
    expect(screen.getByRole("textbox", { name: "Template" }))
      .toHaveValue(authoredValue)

    const confirmationsBeforeForward = vi.mocked(window.confirm).mock.calls.length
    await act(async () => {
      window.history.forward()
      await new Promise((resolve) => setTimeout(resolve, 0))
    })
    await waitFor(() => {
      expect(window.confirm).toHaveBeenCalledTimes(confirmationsBeforeForward + 1)
    })
    await waitFor(() => {
      expect(window.location.hash).toContain("#/settings/prompt")
    })
    await screen.findByRole("heading", { name: "RAG answer" })
    const restored = screen.getByRole("textbox", { name: "Template" })
    expect(restored).toHaveValue(authoredValue)
    await waitFor(() => expect(restored).toHaveFocus())
    expect(window.history.length).toBe(historyLength)
    expect(JSON.stringify(window.history.state)).not.toContain(authoredValue)
    for (const storage of [window.localStorage, window.sessionStorage]) {
      for (let index = 0; index < storage.length; index += 1) {
        expect(storage.getItem(storage.key(index)!) ?? "")
          .not.toContain(authoredValue)
      }
    }
  })

  it.each([
    {
      destination: "/settings/chat",
      host: "BrowserRouter",
      initialUrl: "/settings/prompt?prompt=chat.rag.answer"
    },
    {
      destination: "/prompts",
      host: "HashRouter",
      initialUrl: "/options.html#/settings/prompt?prompt=chat.rag.answer"
    }
  ] as const)(
    "guards and tokenizes programmatic Settings navigation in $host",
    async ({ destination, host, initialUrl }) => {
      window.history.replaceState({
        hostMarker: "programmatic-source",
        servicePromptHistoryForwardDestination: "https://stale.test/route"
      }, "", initialUrl)
      const Router = host === "HashRouter" ? HashRouter : BrowserRouter
      const client = createClient()
      render(
        <Router>
          <QueryClientProvider client={client}>
            <App>
              <Routes>
                <Route
                  path="/settings/prompt"
                  element={(
                    <>
                      <ServicePromptsSettings />
                      <ProgrammaticSettingsNavigation destination={destination} />
                    </>
                  )}
                />
                <Route path="*" element={<p>Programmatic outside route</p>} />
              </Routes>
            </App>
          </QueryClientProvider>
        </Router>
      )
      await screen.findByRole("heading", { name: "RAG answer" })

      fireEvent.change(screen.getByRole("textbox", { name: "Template" }), {
        target: { value: "Declined programmatic {context} {question}" }
      })
      vi.mocked(window.confirm).mockReturnValue(false)
      fireEvent.click(screen.getByRole("button", {
        name: "Programmatic settings navigation"
      }))
      expect(screen.getByRole("heading", { name: "RAG answer" }))
        .toBeInTheDocument()

      vi.mocked(window.confirm).mockReturnValue(true)
      fireEvent.click(screen.getByRole("button", {
        name: "Programmatic settings navigation"
      }))
      expect(await screen.findByText("Programmatic outside route"))
        .toBeInTheDocument()
      const destinationToken = window.history.state
        .servicePromptHistoryEntryToken as string
      expect(destinationToken).toEqual(expect.any(String))
      expect(window.history.state).not.toHaveProperty(
        "servicePromptHistoryForwardDestination"
      )
      const historyLength = window.history.length

      await act(async () => {
        window.history.back()
        await new Promise((resolve) => setTimeout(resolve, 0))
      })
      await screen.findByRole("heading", { name: "RAG answer" })
      expect(window.history.state).toMatchObject({
        hostMarker: "programmatic-source",
        servicePromptHistoryForwardEntryToken: destinationToken
      })
      expect(window.history.state).not.toHaveProperty(
        "servicePromptHistoryForwardDestination"
      )

      const dirtyUrl = window.location.href
      const authoredValue = "Programmatic forward dirty {context} {question}"
      const editor = screen.getByRole("textbox", { name: "Template" })
      fireEvent.change(editor, { target: { value: authoredValue } })
      editor.focus()
      vi.mocked(window.confirm).mockReturnValue(false)

      await act(async () => {
        window.history.forward()
        await new Promise((resolve) => setTimeout(resolve, 0))
      })
      await waitFor(() => expect(window.location.href).toBe(dirtyUrl))
      const restored = await screen.findByRole("textbox", { name: "Template" })
      expect(restored).toHaveValue(authoredValue)
      await waitFor(() => expect(restored).toHaveFocus())
      expect(window.history.length).toBe(historyLength)
      expect(JSON.stringify(window.history.state)).not.toContain(authoredValue)
      for (const storage of [window.localStorage, window.sessionStorage]) {
        for (let index = 0; index < storage.length; index += 1) {
          expect(storage.getItem(storage.key(index)!) ?? "")
            .not.toContain(authoredValue)
        }
      }
    }
  )

  it("uses escaped server English metadata for an unknown catalog ID", async () => {
    const unknown = {
      id: "media.future.summary",
      label: "Future <script>alert(1)</script>",
      description: "Plain <strong>server text</strong>",
      parts: [{
        key: "template",
        label: "Future body",
        mode: "template" as const,
        required_variables: ["text"]
      }],
      affected_workflows: [{ id: "future_flow", label: "Future <flow>" }]
    }
    mocks.list.mockResolvedValue([...catalog, unknown])
    mocks.get.mockResolvedValue(detailFor(unknown))
    renderSettings()

    await openPrompt("Future <script>alert(1)</script>")
    expect(screen.getAllByText("Plain <strong>server text</strong>")).toHaveLength(2)
    expect(document.querySelector("script")).toBeNull()
    expect(document.querySelector("strong")).toBeNull()
  })

  it("shows ordered editors and exact chips only for template parts", async () => {
    renderSettings()
    await openPrompt("Text translation")

    const editors = screen.getAllByRole("textbox")
    expect(editors).toHaveLength(2)
    expect(editors[0]).toHaveAccessibleName("System instructions")
    expect(editors[1]).toHaveAccessibleName("User template")
    const systemGroup = editors[0].closest("section")!
    expect(within(systemGroup).queryByText("target_language")).toBeNull()
    const templateGroup = editors[1].closest("section")!
    expect(within(templateGroup).getByText("{target_language}"))
      .toBeInTheDocument()
    expect(within(templateGroup).getByText("{text}")).toBeInTheDocument()
  })

  it("previews every part locally in registry order with visible markers and no request", async () => {
    renderSettings()
    await openPrompt("Text translation")
    const callsBefore = {
      list: mocks.list.mock.calls.length,
      get: mocks.get.mock.calls.length,
      save: mocks.save.mock.calls.length
    }

    fireEvent.click(screen.getByRole("button", { name: "Preview" }))

    const preview = await screen.findByLabelText("Prompt preview")
    const outputs = within(preview).getAllByRole("code")
    expect(outputs[0]).toHaveTextContent(
      "Translate accurately. Literal {braces} stay literal."
    )
    expect(outputs[1]).toHaveTextContent(
      "Translate to [target_language]: [text]"
    )
    for (const output of outputs) {
      expect(output.closest("pre")).toHaveAttribute("tabindex", "0")
    }
    expect(mocks.renderPart).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({ id: "media.text.translation" }),
      "system",
      expect.any(String),
      {}
    )
    expect(mocks.renderPart).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({ id: "media.text.translation" }),
      "user_template",
      expect.any(String),
      { target_language: "[target_language]", text: "[text]" }
    )
    expect({
      list: mocks.list.mock.calls.length,
      get: mocks.get.mock.calls.length,
      save: mocks.save.mock.calls.length
    }).toEqual(callsBefore)
  })

  it("saves the complete atomic Translation draft with the observed revision", async () => {
    const translation = detailFor(catalog[3], {
      source: "user",
      revision: "22222222-2222-4222-8222-222222222222"
    })
    mocks.get.mockResolvedValue(translation)
    renderSettings()
    await openPrompt("Text translation")

    const [system, userTemplate] = screen.getAllByRole("textbox")
    fireEvent.change(system, { target: { value: "New system text" } })
    fireEvent.change(userTemplate, {
      target: { value: "Use {target_language}: {text}" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Save changes" }))

    await waitFor(() => {
      expect(mocks.save).toHaveBeenCalledWith(
        "media.text.translation",
        {
          parts: {
            system: "New system text",
            user_template: "Use {target_language}: {text}"
          },
          expected_revision: "22222222-2222-4222-8222-222222222222"
        },
        {
          signal: expect.any(AbortSignal),
          requestScope: scopeOne
        }
      )
    })
    expect(await screen.findByText("Customized")).toBeInTheDocument()
    expect(screen.getByRole("status")).toHaveTextContent(
      "Workflow prompt saved."
    )
  })

  it("disables competing actions during save and clears loading when it finishes", async () => {
    const user = userEvent.setup()
    mocks.readLegacy.mockResolvedValue([legacyRagCandidate])
    mocks.get.mockResolvedValue(detailFor(catalog[0], { source: "user" }))
    const pendingSave = deferred<ServicePromptDetail>()
    mocks.save.mockReturnValue(pendingSave.promise)
    renderSettings()
    await screen.findByText("Browser-local workflow prompts found")
    await openPrompt("RAG answer")
    const editor = screen.getByRole("textbox", { name: "Template" })
    const submittedValue = "Saving {context} {question}"
    fireEvent.change(editor, {
      target: { value: submittedValue }
    })
    fireEvent.click(screen.getByRole("button", { name: "Save changes" }))
    await waitFor(() => expect(mocks.save).toHaveBeenCalledTimes(1))
    expect(editor).toBeDisabled()
    await user.type(editor, " Ignored newer text")
    expect(editor).toHaveValue(submittedValue)
    expect(mocks.save).toHaveBeenCalledWith(
      "chat.rag.answer",
      expect.objectContaining({ parts: { template: submittedValue } }),
      {
        signal: expect.any(AbortSignal),
        requestScope: scopeOne
      }
    )
    expect(screen.getByRole("button", { name: /Save changes/ })).toBeDisabled()
    expect(screen.getByRole("button", { name: "Reset to default" })).toBeDisabled()
    expect(screen.getByRole("button", { name: "Import to this server" }))
      .toBeDisabled()
    expect(screen.getByRole("button", { name: "Discard local values" }))
      .toBeDisabled()
    expect(document.querySelector(".ant-btn-loading")).not.toBeNull()
    await act(async () => {
      pendingSave.resolve(detailFor(catalog[0], {
        source: "user",
        parts: { template: submittedValue }
      }))
    })
    await waitFor(() => expect(document.querySelector(".ant-btn-loading"))
      .toBeNull())
    expect(screen.getByRole("button", { name: "Reset to default" }))
      .toBeEnabled()
    expect(screen.getByRole("button", { name: "Import to this server" }))
      .toBeEnabled()
    expect(screen.getByRole("textbox", { name: "Template" }))
      .toHaveValue(submittedValue)
  })

  it("aborts an in-flight save when another definition is selected and ignores its late result", async () => {
    const pendingSave = deferred<ServicePromptDetail>()
    mocks.save.mockReturnValue(pendingSave.promise)
    renderSettings()
    await openPrompt("RAG answer")
    fireEvent.change(screen.getByRole("textbox", { name: "Template" }), {
      target: { value: "Saving {context} {question}" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Save changes" }))
    await waitFor(() => expect(mocks.save).toHaveBeenCalledTimes(1))
    const signal = mocks.save.mock.calls[0][2].signal as AbortSignal

    fireEvent.click(screen.getByRole("button", { name: "RAG follow-up rewrite" }))
    await screen.findByRole("heading", { name: "RAG follow-up rewrite" })
    expect(signal.aborted).toBe(true)
    await act(async () => {
      pendingSave.resolve(detailFor(catalog[0], {
        source: "user",
        parts: { template: "Late {context} {question}" }
      }))
    })
    expect(screen.getByRole("textbox", { name: "Template" }))
      .toHaveValue("History: {chat_history}\nQuestion: {question}")
  })

  it("aborts an in-flight reset when another definition is selected and ignores its late result", async () => {
    mocks.get.mockImplementation(async (id: string) => detailFor(
      catalog.find((item) => item.id === id)!,
      id === "chat.rag.answer" ? { source: "user" } : {}
    ))
    const pendingReset = deferred<ServicePromptDetail>()
    mocks.reset.mockReturnValue(pendingReset.promise)
    renderSettings()
    await openPrompt("RAG answer")
    fireEvent.click(screen.getByRole("button", { name: "Reset to default" }))
    fireEvent.click(within(await screen.findByRole("dialog"))
      .getByRole("button", { name: "Reset" }))
    await waitFor(() => expect(mocks.reset).toHaveBeenCalledTimes(1))
    const signal = mocks.reset.mock.calls[0][2].signal as AbortSignal

    fireEvent.click(screen.getByRole("button", { name: "RAG follow-up rewrite" }))
    await screen.findByRole("heading", { name: "RAG follow-up rewrite" })
    expect(signal.aborted).toBe(true)
    await act(async () => {
      pendingReset.resolve(detailFor(catalog[0]))
    })
    expect(screen.getByRole("textbox", { name: "Template" }))
      .toHaveValue("History: {chat_history}\nQuestion: {question}")
  })

  it("claims reset confirmation synchronously and ignores approval after unmount", async () => {
    mocks.get.mockResolvedValue(detailFor(catalog[0], { source: "user" }))
    const pendingConfirmation = deferred<boolean>()
    mocks.confirmDanger.mockReturnValue(pendingConfirmation.promise)
    const view = renderSettings()
    await openPrompt("RAG answer")
    const resetButton = screen.getByRole("button", { name: "Reset to default" })

    act(() => {
      resetButton.click()
      resetButton.click()
    })

    view.unmount()
    await act(async () => {
      pendingConfirmation.resolve(true)
      await pendingConfirmation.promise
    })
    expect(mocks.confirmDanger).toHaveBeenCalledTimes(1)
    expect(mocks.reset).not.toHaveBeenCalled()
  })

  it("names permanent customization removal before a conditional reset", async () => {
    mocks.get.mockResolvedValue(detailFor(catalog[0], { source: "user" }))
    renderSettings()
    await openPrompt("RAG answer")

    fireEvent.click(screen.getByRole("button", { name: "Reset to default" }))
    const dialog = await screen.findByRole("dialog")
    expect(dialog).toHaveTextContent("Reset RAG answer?")
    expect(dialog).toHaveTextContent(
      "permanently remove the saved customization"
    )
    expect(dialog).toHaveTextContent("There is no history or undo.")
    fireEvent.click(within(dialog).getByRole("button", { name: "Reset" }))

    await waitFor(() => {
      expect(mocks.reset).toHaveBeenCalledWith(
        "chat.rag.answer",
        "11111111-1111-4111-8111-111111111111",
        {
          signal: expect.any(AbortSignal),
          requestScope: scopeOne
        }
      )
    })
    expect(await screen.findByRole("status")).toHaveTextContent(
      "Workflow prompt reset to the server default."
    )
  })

  it("shows local and authoritative field validation without dropping the draft", async () => {
    renderSettings()
    await openPrompt("RAG answer")
    const editor = screen.getByRole("textbox", { name: "Template" })
    fireEvent.change(editor, { target: { value: "missing variables" } })
    fireEvent.click(screen.getByRole("button", { name: "Save changes" }))
    expect(await screen.findByText(
      "Template variables must match the registered variables exactly once."
    )).toBeInTheDocument()
    const localError = screen.getByText(
      "Template variables must match the registered variables exactly once."
    )
    expect(editor).toHaveAttribute("aria-invalid", "true")
    expect(editor).toHaveAttribute("aria-describedby", localError.id)
    expect(localError.id).toBe("service-prompt-chat-rag-answer-template-error")
    expect(mocks.save).not.toHaveBeenCalled()

    fireEvent.change(editor, {
      target: { value: "Context {context}; question {question}" }
    })
    mocks.save.mockRejectedValueOnce(new ServicePromptApiError("Invalid", {
      status: 422,
      code: "service_prompt_validation_failed",
      fieldErrors: { template: "Server rejected this template." }
    }))
    fireEvent.click(screen.getByRole("button", { name: "Save changes" }))
    expect(await screen.findByText("Server rejected this template."))
      .toBeInTheDocument()
    const serverError = screen.getByText("Server rejected this template.")
    expect(editor).toHaveAttribute("aria-describedby", serverError.id)
    expect(editor).toHaveValue("Context {context}; question {question}")
  })

  it("shows a generic save error for structural 422 responses without dropping the draft", async () => {
    renderSettings()
    await openPrompt("RAG answer")
    const editor = screen.getByRole("textbox", { name: "Template" })
    const draft = "Context {context}; question {question}"
    fireEvent.change(editor, { target: { value: draft } })
    mocks.save.mockRejectedValueOnce(new ServicePromptApiError(
      "Framework validation failed",
      { status: 422 }
    ))

    fireEvent.click(screen.getByRole("button", { name: "Save changes" }))

    expect(await screen.findByText("Unable to save this workflow prompt."))
      .toBeInTheDocument()
    expect(editor).toHaveValue(draft)
    expect(screen.getByText("Unsaved")).toBeInTheDocument()
  })

  it.each([
    {
      name: "wrong validation code",
      options: {
        status: 422,
        code: "other_validation_failed",
        fieldErrors: { template: "Rejected." }
      }
    },
    {
      name: "unknown field",
      options: {
        status: 422,
        code: "service_prompt_validation_failed",
        fieldErrors: { stale_part: "Rejected." }
      }
    },
    {
      name: "blank field message",
      options: {
        status: 422,
        code: "service_prompt_validation_failed",
        fieldErrors: { template: "   " }
      }
    }
  ])("shows a generic save error for $name", async ({ options }) => {
    renderSettings()
    await openPrompt("RAG answer")
    const editor = screen.getByRole("textbox", { name: "Template" })
    const draft = "Context {context}; question {question}"
    fireEvent.change(editor, { target: { value: draft } })
    mocks.save.mockRejectedValueOnce(new ServicePromptApiError(
      "Malformed validation response",
      options
    ))

    fireEvent.click(screen.getByRole("button", { name: "Save changes" }))

    expect(await screen.findByText("Unable to save this workflow prompt."))
      .toBeInTheDocument()
    expect(editor).toHaveValue(draft)
    expect(screen.getByText("Unsaved")).toBeInTheDocument()
  })

  it("preserves the whole draft on conflict and reloads only on request", async () => {
    renderSettings()
    await openPrompt("RAG answer")
    const editor = screen.getByRole("textbox", { name: "Template" })
    const draft = "Draft {context} and {question}"
    fireEvent.change(editor, { target: { value: draft } })
    mocks.save.mockRejectedValueOnce(new ServicePromptApiError("Conflict", {
      status: 409,
      code: "service_prompt_revision_conflict",
      currentRevision: "33333333-3333-4333-8333-333333333333"
    }))
    fireEvent.click(screen.getByRole("button", { name: "Save changes" }))

    expect(await screen.findByText("This prompt changed on the server."))
      .toBeInTheDocument()
    expect(editor).toHaveValue(draft)
    expect(mocks.get).toHaveBeenCalledTimes(1)

    mocks.get.mockRejectedValueOnce(new Error("reload offline"))
    fireEvent.click(screen.getByRole("button", { name: "Reload server value" }))
    expect(await screen.findByText("Unable to reload the server value."))
      .toBeInTheDocument()
    expect(screen.getByRole("textbox", { name: "Template" })).toHaveValue(draft)
    expect(screen.getByText("Unsaved")).toBeInTheDocument()

    mocks.get.mockResolvedValue(detailFor(catalog[0], {
      source: "user",
      parts: { template: "Server {context} {question}" },
      revision: "33333333-3333-4333-8333-333333333333"
    }))
    fireEvent.click(screen.getByRole("button", { name: "Reload server value" }))
    await waitFor(() => expect(editor).toHaveValue("Server {context} {question}"))
  })

  it("aborts conflict reload on selection and cannot bind its late result to the new editor", async () => {
    renderSettings()
    await openPrompt("RAG answer")
    fireEvent.change(screen.getByRole("textbox", { name: "Template" }), {
      target: { value: "Draft {context} {question}" }
    })
    mocks.save.mockRejectedValueOnce(new ServicePromptApiError("Conflict", {
      status: 409,
      code: "service_prompt_revision_conflict"
    }))
    fireEvent.click(screen.getByRole("button", { name: "Save changes" }))
    await screen.findByText("This prompt changed on the server.")

    const pendingReload = deferred<ServicePromptDetail>()
    mocks.get.mockReturnValueOnce(pendingReload.promise)
    fireEvent.click(screen.getByRole("button", { name: "Reload server value" }))
    await waitFor(() => expect(mocks.get).toHaveBeenCalledTimes(2))
    const signal = mocks.get.mock.calls[1][1].signal as AbortSignal
    fireEvent.click(screen.getByRole("button", { name: "RAG follow-up rewrite" }))
    await screen.findByRole("heading", { name: "RAG follow-up rewrite" })
    expect(signal.aborted).toBe(true)

    await act(async () => {
      pendingReload.resolve(detailFor(catalog[0], {
        source: "user",
        parts: { template: "Late server {context} {question}" }
      }))
    })
    expect(screen.getByRole("textbox", { name: "Template" }))
      .toHaveValue("History: {chat_history}\nQuestion: {question}")
  })

  it("offers revision-bound recovery for a corrupt override", async () => {
    const revision = "44444444-4444-4444-8444-444444444444"
    mocks.get.mockRejectedValue(new ServicePromptApiError(
      "The saved Service Prompt override is corrupt.",
      {
        status: 500,
        code: "service_prompt_corrupt_override",
        revision,
        canReset: true
      }
    ))
    renderSettings()
    fireEvent.click(await screen.findByRole("button", { name: "RAG answer" }))

    expect(await screen.findByText("Saved customization is unavailable"))
      .toBeInTheDocument()
    expect(screen.getByText(
      "The saved value cannot be read safely. Reset it to restore the server default."
    )).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Reset corrupt customization" }))
    const dialog = await screen.findByRole("dialog")
    fireEvent.click(within(dialog).getByRole("button", { name: "Reset" }))
    await waitFor(() => {
      expect(mocks.reset).toHaveBeenCalledWith(
        "chat.rag.answer",
        revision,
        {
          signal: expect.any(AbortSignal),
          requestScope: scopeOne
        }
      )
    })
  })

  it("shows a failed corrupt reset inside the corrupt recovery callout", async () => {
    const revision = "44444444-4444-4444-8444-444444444444"
    mocks.get.mockRejectedValue(new ServicePromptApiError("Corrupt", {
      status: 500,
      code: "service_prompt_corrupt_override",
      revision,
      canReset: true
    }))
    mocks.reset.mockRejectedValueOnce(new Error("reset unavailable"))
    renderSettings()
    fireEvent.click(await screen.findByRole("button", { name: "RAG answer" }))
    const resetButton = await screen.findByRole("button", {
      name: "Reset corrupt customization"
    })
    fireEvent.click(resetButton)
    fireEvent.click(within(await screen.findByRole("dialog"))
      .getByRole("button", { name: "Reset" }))

    const recovery = (await screen.findByText("Saved customization is unavailable"))
      .closest<HTMLElement>('[data-ds-component="RecoveryCallout"]')!
    expect(within(recovery).getByText("Unable to reset this workflow prompt."))
      .toBeInTheDocument()
  })

  it("refetches a corrupt revision after reset conflict before offering reset again", async () => {
    const firstRevision = "44444444-4444-4444-8444-444444444444"
    const secondRevision = "55555555-5555-4555-8555-555555555555"
    mocks.get
      .mockRejectedValueOnce(new ServicePromptApiError("Corrupt", {
        status: 500,
        code: "service_prompt_corrupt_override",
        revision: firstRevision,
        canReset: true
      }))
      .mockRejectedValueOnce(new ServicePromptApiError("Corrupt again", {
        status: 500,
        code: "service_prompt_corrupt_override",
        revision: secondRevision,
        canReset: true
      }))
    mocks.reset
      .mockRejectedValueOnce(new ServicePromptApiError("Conflict", {
        status: 409,
        code: "service_prompt_revision_conflict"
      }))
      .mockResolvedValueOnce(detailFor(catalog[0]))
    renderSettings()
    fireEvent.click(await screen.findByRole("button", { name: "RAG answer" }))
    fireEvent.click(await screen.findByRole("button", {
      name: "Reset corrupt customization"
    }))
    const dialogs = await screen.findAllByRole("dialog")
    fireEvent.click(within(dialogs[dialogs.length - 1])
      .getByRole("button", { name: "Reset" }))
    await waitFor(() => expect(mocks.get).toHaveBeenCalledTimes(2))
    const rebindMessage =
      "The saved customization changed. The latest revision was loaded. Retry reset."
    const recovery = screen.getByText("Saved customization is unavailable")
      .closest<HTMLElement>('[data-ds-component="RecoveryCallout"]')!
    expect(within(recovery).getByText(rebindMessage)).toBeInTheDocument()
    expect(within(recovery).getByRole("alert")).toHaveTextContent(rebindMessage)
    expect(screen.getAllByText(rebindMessage)).toHaveLength(1)
    expect(screen.getByRole("status")).not.toHaveTextContent(rebindMessage)

    fireEvent.click(await screen.findByRole("button", {
      name: "Reset corrupt customization"
    }))
    const retryDialogs = await screen.findAllByRole("dialog")
    fireEvent.click(within(retryDialogs[retryDialogs.length - 1])
      .getByRole("button", { name: "Reset" }))
    await waitFor(() => {
      expect(mocks.reset).toHaveBeenNthCalledWith(
        2,
        "chat.rag.answer",
        secondRevision,
        {
          signal: expect.any(AbortSignal),
          requestScope: scopeOne
        }
      )
    })
  })

  it("distinguishes an older catalog 404 from other catalog failures", async () => {
    mocks.list.mockRejectedValueOnce(new ServicePromptApiError("Not found", {
      status: 404
    }))
    const first = renderSettings()
    expect(await screen.findByText("Workflow prompts require a server update"))
      .toBeInTheDocument()
    expect(screen.getByText(/existing browser-local prompt behavior remains active/i))
      .toBeInTheDocument()
    expect(mocks.readLegacy).not.toHaveBeenCalled()
    first.unmount()

    mocks.list.mockRejectedValueOnce(new ServicePromptApiError("Forbidden", {
      status: 403
    }))
    renderSettings()
    expect(await screen.findByText("Unable to load workflow prompts"))
      .toBeInTheDocument()
    expect(screen.queryByText("Workflow prompts require a server update"))
      .not.toBeInTheDocument()
  })

  it.each(["catalog", "detail"] as const)(
    "re-resolves the connected scope when a %s query is rejected as stale",
    async (query) => {
      const scopeError = new ServicePromptApiError(
        "The server or authenticated account changed before the request was sent.",
        { status: 412, code: "request_config_scope_changed" }
      )
      mocks.resolveScope
        .mockResolvedValueOnce(scopeOne)
        .mockResolvedValue(scopeTwo)
      if (query === "catalog") {
        mocks.list.mockRejectedValueOnce(scopeError).mockResolvedValue(catalog)
      } else {
        mocks.get.mockRejectedValueOnce(scopeError)
      }

      renderSettings()
      if (query === "detail") {
        fireEvent.click(await screen.findByRole("button", { name: "RAG answer" }))
        await screen.findByText("Unable to load this workflow prompt")
      }

      expect(await screen.findByText(/Server or account changed/i))
        .toBeInTheDocument()
      expect(mocks.resolveScope).toHaveBeenCalledTimes(2)
      expect(mocks.list).toHaveBeenLastCalledWith(expect.objectContaining({
        requestScope: scopeTwo
      }))
    }
  )

  it("shows and retries a legacy probe failure even when no candidates were returned", async () => {
    mocks.readLegacy
      .mockRejectedValueOnce(new Error("storage unavailable"))
      .mockResolvedValueOnce([legacyRagCandidate])
    renderSettings()

    expect(await screen.findByText("Unable to read browser-local workflow prompts."))
      .toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Retry" }))
    expect(await screen.findByText("Browser-local workflow prompts found"))
      .toBeInTheDocument()
    expect(mocks.readLegacy).toHaveBeenCalledTimes(2)
  })

  it("preserves candidates and allows retry when migration detail prefetch fails", async () => {
    mocks.readLegacy.mockResolvedValue([legacyRagCandidate])
    mocks.get.mockRejectedValueOnce(new Error("detail unavailable"))
    renderSettings()
    await screen.findByText("Browser-local workflow prompts found")

    fireEvent.click(await screen.findByRole("button", {
      name: "Import to this server"
    }))
    expect(await screen.findByText(
      "Unable to prepare this import. The browser-local values were preserved."
    )).toBeInTheDocument()
    expect(screen.getByText("1 browser-local prompt still needs attention."))
      .toBeInTheDocument()
    expect(screen.getByRole("textbox", { name: "Repair RAG answer" }))
      .toHaveValue(legacyRagCandidate.value)

    mocks.get.mockResolvedValue(detailFor(catalog[0]))
    const retryImport = screen.getByText("Import to this server").closest("button")!
    await waitFor(() => expect(retryImport).not.toBeDisabled())
    fireEvent.click(retryImport)
    await waitFor(() => expect(mocks.importLegacy).toHaveBeenCalledTimes(1))
  })

  it("handles replacement-confirm rejection without losing candidates and remains retryable", async () => {
    mocks.readLegacy.mockResolvedValue([legacyRagCandidate])
    mocks.get.mockResolvedValue(detailFor(catalog[0], {
      source: "user",
      parts: { template: "Saved {context} {question}" }
    }))
    mocks.confirmDanger.mockRejectedValueOnce(new Error("dialog unavailable"))
    renderSettings()
    await screen.findByText("Browser-local workflow prompts found")

    const retryImport = screen.getByText("Import to this server").closest("button")!
    await waitFor(() => expect(retryImport).not.toBeDisabled())
    fireEvent.click(retryImport)
    expect(await screen.findByText(
      "Unable to prepare this import. The browser-local values were preserved."
    )).toBeInTheDocument()
    expect(screen.getByRole("textbox", { name: "Repair RAG answer" }))
      .toHaveValue(legacyRagCandidate.value)

    await waitFor(() => expect(retryImport).not.toBeDisabled())
    fireEvent.click(retryImport)
    fireEvent.click(within(await screen.findByRole("dialog"))
      .getByRole("button", { name: "Replace and import" }))
    await waitFor(() => expect(mocks.importLegacy).toHaveBeenCalledTimes(1))
  })

  it("imports raw migration values and removes each only after confirmed save", async () => {
    mocks.readLegacy.mockResolvedValue([
      {
        definitionId: "chat.rag.answer",
        partKey: "template",
        storageKey: "systemPromptForRag",
        value: "Legacy {context} {question}"
      }
    ])
    renderSettings()
    expect(await screen.findByText("Browser-local workflow prompts found"))
      .toBeInTheDocument()
    expect(screen.getByText("https://research-one.test")).toBeInTheDocument()
    expect(screen.getByText(/Portable backups do not include Service Prompt overrides/i))
      .toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Import to this server" }))
    await waitFor(() => expect(mocks.importLegacy).toHaveBeenCalledWith(
      expect.objectContaining({
        definitionId: "chat.rag.answer",
        value: "Legacy {context} {question}"
      }),
      expect.objectContaining({ id: "chat.rag.answer" }),
      {
        signal: expect.any(AbortSignal),
        requestScope: scopeOne
      }
    ))
    expect(screen.queryByText("Browser-local workflow prompts found"))
      .not.toBeInTheDocument()
  })

  it("keeps migration editors immutable for the complete deferred import", async () => {
    const user = userEvent.setup()
    mocks.readLegacy.mockResolvedValue([legacyRagCandidate])
    const pendingDetail = deferred<ServicePromptDetail>()
    const pendingImport = deferred<ServicePromptDetail>()
    mocks.get.mockReturnValueOnce(pendingDetail.promise)
    mocks.importLegacy.mockReturnValueOnce(pendingImport.promise)
    renderSettings()
    await screen.findByText("Browser-local workflow prompts found")
    const repair = screen.getByRole("textbox", { name: "Repair RAG answer" })

    fireEvent.click(screen.getByRole("button", { name: "Import to this server" }))
    await waitFor(() => expect(mocks.get).toHaveBeenCalledTimes(1))
    expect(repair).toBeDisabled()
    await user.type(repair, " Ignored newer text")
    expect(repair).toHaveValue(legacyRagCandidate.value)

    await act(async () => {
      pendingDetail.resolve(detailFor(catalog[0]))
    })
    await waitFor(() => expect(mocks.importLegacy).toHaveBeenCalledWith(
      expect.objectContaining({ value: legacyRagCandidate.value }),
      expect.objectContaining({ id: "chat.rag.answer" }),
      {
        signal: expect.any(AbortSignal),
        requestScope: scopeOne
      }
    ))
    expect(repair).toBeDisabled()
    expect(repair).toHaveValue(legacyRagCandidate.value)

    await act(async () => {
      pendingImport.resolve(detailFor(catalog[0], {
        source: "user",
        parts: { template: legacyRagCandidate.value }
      }))
    })
    await waitFor(() => expect(screen.queryByText(
      "Browser-local workflow prompts found"
    )).not.toBeInTheDocument())
  })

  it("confirms replacements, exposes invalid raw text for repair, and preserves failed imports", async () => {
    mocks.readLegacy.mockResolvedValue([
      {
        definitionId: "chat.rag.answer",
        partKey: "template",
        storageKey: "systemPromptForRag",
        value: "invalid legacy"
      },
      {
        definitionId: "chat.web_search.answer",
        partKey: "template",
        storageKey: "webSearchPrompt",
        value: "Web {current_date_time} {search_results}"
      }
    ])
    mocks.get.mockImplementation(async (id: string) => detailFor(
      catalog.find((item) => item.id === id)!,
      id === "chat.web_search.answer" ? { source: "user" } : {}
    ))
    mocks.importLegacy.mockImplementation(async (candidate: { definitionId: string }) => {
      if (candidate.definitionId === "chat.web_search.answer") {
        throw new Error("synthetic import failure")
      }
      return detailFor(catalog[0], { source: "user" })
    })
    renderSettings()
    await screen.findByText("Browser-local workflow prompts found")

    fireEvent.click(screen.getByRole("button", { name: "Import to this server" }))
    expect(await screen.findByText(
      "Template variables must match the registered variables exactly once."
    )).toBeInTheDocument()
    const repair = screen.getByRole("textbox", { name: "Repair RAG answer" })
    expect(repair).toHaveValue("invalid legacy")
    const migrationError = screen.getByText(
      "Template variables must match the registered variables exactly once."
    )
    expect(repair).toHaveAttribute("aria-invalid", "true")
    expect(repair).toHaveAttribute("aria-describedby", migrationError.id)
    expect(migrationError.id).toBe("migration-chat-rag-answer-error")
    fireEvent.change(repair, {
      target: { value: "Repaired {context} {question}" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Import to this server" }))

    const dialog = await screen.findByRole("dialog")
    expect(dialog).toHaveTextContent("Web-search answer")
    fireEvent.click(within(dialog).getByRole("button", { name: "Replace and import" }))
    expect(await screen.findByText("1 browser-local prompt still needs attention."))
      .toBeInTheDocument()
    expect(screen.queryByRole("textbox", { name: "Repair RAG answer" }))
      .not.toBeInTheDocument()
    expect(screen.getByRole("textbox", { name: "Repair Web-search answer" }))
      .toHaveValue("Web {current_date_time} {search_results}")
  })

  it("discards only the mapped raw values after confirmation and reports partial cleanup", async () => {
    mocks.readLegacy.mockResolvedValue([
      {
        definitionId: "chat.rag.answer",
        partKey: "template",
        storageKey: "systemPromptForRag",
        value: "RAG {context} {question}"
      },
      {
        definitionId: "chat.web_search.answer",
        partKey: "template",
        storageKey: "webSearchPrompt",
        value: "Web {current_date_time} {search_results}"
      }
    ])
    mocks.clearLegacy.mockImplementation(async (id: string) => {
      if (id === "chat.web_search.answer") throw new Error("cleanup failure")
    })
    renderSettings()
    await screen.findByText("Browser-local workflow prompts found")
    mocks.initialize.mockClear()

    fireEvent.click(screen.getByRole("button", { name: "Discard local values" }))
    const dialog = await screen.findByRole("dialog")
    expect(dialog).toHaveTextContent("Discard browser-local workflow prompts?")
    fireEvent.click(within(dialog).getByRole("button", { name: "Discard" }))

    await waitFor(() => {
      expect(mocks.clearLegacy).toHaveBeenCalledWith("chat.rag.answer")
      expect(mocks.clearLegacy).toHaveBeenCalledWith("chat.web_search.answer")
    })
    expect(mocks.initialize).not.toHaveBeenCalled()
    expect(await screen.findByText("1 browser-local prompt still needs attention."))
      .toBeInTheDocument()
  })

  it("claims discard confirmation synchronously and ignores approval after unmount", async () => {
    mocks.readLegacy.mockResolvedValue([legacyRagCandidate])
    const pendingConfirmation = deferred<boolean>()
    mocks.confirmDanger.mockReturnValue(pendingConfirmation.promise)
    const view = renderSettings()
    await screen.findByText("Browser-local workflow prompts found")
    const discardButton = screen.getByRole("button", {
      name: "Discard local values"
    })

    act(() => {
      discardButton.click()
      discardButton.click()
    })

    view.unmount()
    await act(async () => {
      pendingConfirmation.resolve(true)
      await pendingConfirmation.promise
    })
    expect(mocks.confirmDanger).toHaveBeenCalledTimes(1)
    expect(mocks.clearLegacy).not.toHaveBeenCalled()
  })

  it.each(["resolve", "reject"] as const)(
    "does not restore old migration state when non-abortable discard %s settles after scope change",
    async (outcome) => {
      const secondCandidate = {
        definitionId: "chat.web_search.answer",
        partKey: "template",
        storageKey: "webSearchPrompt",
        value: "Web {current_date_time} {search_results}"
      }
      mocks.readLegacy
        .mockResolvedValueOnce([legacyRagCandidate, secondCandidate])
        .mockResolvedValueOnce([])
      const pendingClear = deferred<void>()
      mocks.clearLegacy.mockReturnValueOnce(pendingClear.promise)
      mocks.resolveScope
        .mockResolvedValueOnce(scopeOne)
        .mockResolvedValueOnce(scopeTwo)
        .mockResolvedValueOnce(scopeTwo)
      renderSettings()
      await screen.findByText("Browser-local workflow prompts found")
      fireEvent.click(screen.getByRole("button", { name: "Discard local values" }))
      fireEvent.click(within(await screen.findByRole("dialog"))
        .getByRole("button", { name: "Discard" }))
      await waitFor(() => expect(mocks.clearLegacy).toHaveBeenCalledTimes(1))

      window.dispatchEvent(new Event("tldw:config-updated"))
      await waitFor(() => {
        expect(mocks.resolveScope).toHaveBeenCalledTimes(3)
        expect(mocks.readLegacy).toHaveBeenCalledTimes(2)
      })
      await act(async () => {
        if (outcome === "resolve") pendingClear.resolve()
        else pendingClear.reject(new Error("late clear failure"))
      })

      await waitFor(() => {
        expect(screen.queryByText("Browser-local workflow prompts found"))
          .not.toBeInTheDocument()
        expect(screen.queryByText(/browser-local prompt still needs attention/i))
          .not.toBeInTheDocument()
      })
    }
  )

  it("cancels the old scope, clears migration, and makes its dirty draft unsaveable", async () => {
    mocks.readLegacy.mockResolvedValue([
      {
        definitionId: "chat.rag.answer",
        partKey: "template",
        storageKey: "systemPromptForRag",
        value: "Legacy {context} {question}"
      }
    ])
    const { client } = renderSettings()
    const cancel = vi.spyOn(client, "cancelQueries")
    const invalidate = vi.spyOn(client, "invalidateQueries")
    await openPrompt("RAG answer")
    fireEvent.change(screen.getByRole("textbox", { name: "Template" }), {
      target: { value: "Dirty {context} {question}" }
    })
    mocks.resolveScope.mockResolvedValue(scopeTwo)

    window.dispatchEvent(new Event("tldw:config-updated"))

    await waitFor(() => expect(cancel).toHaveBeenCalledWith({
      queryKey: ["service-prompts", scopeOne.scopeKey]
    }))
    expect(invalidate).toHaveBeenCalledWith({
      queryKey: ["service-prompts", scopeOne.scopeKey],
      refetchType: "none"
    })
    await waitFor(() => {
      expect(mocks.readLegacy).toHaveBeenCalledTimes(2)
      expect(screen.getByText("https://research-two.test")).toBeInTheDocument()
      expect(screen.queryByRole("button", { name: "Save changes" }))
        .not.toBeInTheDocument()
    })
    expect(await screen.findByText(/Server or account changed/i))
      .toBeInTheDocument()
  })

  it("reconciles cross-tab credentials without discarding a same-user dirty draft", async () => {
    const unsubscribe = vi.fn()
    mocks.subscribeConfig.mockReturnValue(unsubscribe)
    const view = renderSettings()
    await openPrompt("RAG answer")
    const editor = screen.getByRole("textbox", { name: "Template" })
    const authored = "Dirty after rotation {context} {question}"
    fireEvent.change(editor, { target: { value: authored } })
    await waitFor(() => expect(mocks.subscribeConfig).toHaveBeenCalled())
    const latestSubscription = mocks.subscribeConfig.mock.calls.at(-1)
    const notifyConfigChanged = latestSubscription?.[0] as () => void

    mocks.resolveScope.mockResolvedValue(rotatedScopeOne)
    act(() => notifyConfigChanged())

    await waitFor(() => expect(mocks.resolveScope).toHaveBeenCalledTimes(2))
    expect(screen.getByRole("textbox", { name: "Template" })).toHaveValue(authored)
    expect(screen.getByText("Unsaved")).toBeInTheDocument()
    expect(screen.queryByText(/Server or account changed/i)).not.toBeInTheDocument()

    mocks.resolveScope.mockResolvedValue(accountTwoSameServer)
    act(() => notifyConfigChanged())

    expect(await screen.findByText(/Server or account changed/i))
      .toBeInTheDocument()
    await waitFor(() => {
      expect(mocks.list).toHaveBeenLastCalledWith(expect.objectContaining({
        requestScope: accountTwoSameServer
      }))
      expect(screen.queryByRole("textbox", { name: "Template" }))
        .not.toBeInTheDocument()
    })

    view.unmount()
    expect(unsubscribe).toHaveBeenCalled()
  })

  it.each([
    "tldw:config-updated",
    "tldw:auth-credentials-changed"
  ] as const)(
    "preserves a dirty draft when %s reconciliation temporarily fails",
    async (eventName) => {
      renderSettings()
      await openPrompt("RAG answer")
      const editor = screen.getByRole("textbox", { name: "Template" })
      const authored = "Dirty while offline {context} {question}"
      fireEvent.change(editor, { target: { value: authored } })
      mocks.resolveScope.mockRejectedValueOnce(new Error("temporarily offline"))

      window.dispatchEvent(new Event(eventName))

      await waitFor(() => expect(mocks.resolveScope).toHaveBeenCalledTimes(2))
      expect(screen.queryByRole("textbox", { name: "Template" }))
        .not.toBeInTheDocument()
      expect(screen.queryByText("Unsaved")).not.toBeInTheDocument()
      expect(screen.queryByText(/Server or account changed/i))
        .not.toBeInTheDocument()
      expect(screen.queryByRole("button", { name: "Save changes" }))
        .not.toBeInTheDocument()
      expect(mocks.save).not.toHaveBeenCalled()

      mocks.resolveScope.mockResolvedValueOnce(rotatedScopeOne)
      window.dispatchEvent(new Event(eventName))

      await waitFor(() => expect(mocks.resolveScope).toHaveBeenCalledTimes(3))
      const saveButton = await screen.findByRole("button", {
        name: "Save changes"
      })
      await waitFor(() => expect(saveButton).toBeEnabled())
      expect(screen.getByRole("textbox", { name: "Template" }))
        .toHaveValue(authored)
      expect(screen.getByText("Unsaved")).toBeInTheDocument()
    }
  )

  it("ignores a late save result while same-scope verification is pending", async () => {
    const pendingSave = deferred<ServicePromptDetail>()
    mocks.save.mockReturnValueOnce(pendingSave.promise)
    renderSettings()
    await openPrompt("RAG answer")
    const authored = "Dirty during verification {context} {question}"
    fireEvent.change(screen.getByRole("textbox", { name: "Template" }), {
      target: { value: authored }
    })
    fireEvent.click(screen.getByRole("button", { name: "Save changes" }))
    await waitFor(() => expect(mocks.save).toHaveBeenCalledOnce())

    mocks.resolveScope.mockRejectedValueOnce(new Error("temporarily offline"))
    window.dispatchEvent(new Event("tldw:config-updated"))
    await waitFor(() => expect(mocks.resolveScope).toHaveBeenCalledTimes(2))

    await act(async () => {
      pendingSave.resolve(detailFor(catalog[0], {
        source: "user",
        parts: { template: "Late server value {context} {question}" }
      }))
    })

    mocks.resolveScope.mockResolvedValueOnce(rotatedScopeOne)
    fireEvent.click(screen.getByRole("button", { name: "Retry" }))
    expect(await screen.findByRole("textbox", { name: "Template" }))
      .toHaveValue(authored)
    expect(screen.getByText("Unsaved")).toBeInTheDocument()
  })

  it("conceals migration values while the server and account are unverified", async () => {
    mocks.readLegacy.mockResolvedValue([legacyRagCandidate])
    renderSettings()
    expect(await screen.findByText("Browser-local workflow prompts found"))
      .toBeInTheDocument()
    mocks.resolveScope.mockRejectedValueOnce(new Error("temporarily offline"))

    window.dispatchEvent(new Event("tldw:auth-credentials-changed"))

    await waitFor(() => expect(mocks.resolveScope).toHaveBeenCalledTimes(2))
    expect(screen.queryByText("Browser-local workflow prompts found"))
      .not.toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Discard local values" }))
      .not.toBeInTheDocument()
    expect(mocks.clearLegacy).not.toHaveBeenCalled()
  })

  it("invalidates a dirty draft when authenticated scope becomes unresolved", async () => {
    const { client } = renderSettings()
    const cancel = vi.spyOn(client, "cancelQueries")
    await openPrompt("RAG answer")
    fireEvent.change(screen.getByRole("textbox", { name: "Template" }), {
      target: { value: "Dirty before logout {context} {question}" }
    })
    mocks.resolveScope.mockRejectedValueOnce(Object.assign(
      new Error("redacted"),
      { code: "service_prompt_scope_unresolved" }
    ))

    window.dispatchEvent(new Event("tldw:auth-credentials-changed"))

    expect(await screen.findByText(/Server or account changed/i))
      .toBeInTheDocument()
    expect(cancel).toHaveBeenCalledWith({
      queryKey: ["service-prompts", scopeOne.scopeKey]
    })
    await waitFor(() => {
      expect(screen.queryByRole("textbox", { name: "Template" }))
        .not.toBeInTheDocument()
    })
  })

  it("invalidates the page when the server rejects a changed request scope", async () => {
    renderSettings()
    await openPrompt("RAG answer")
    fireEvent.change(screen.getByRole("textbox", { name: "Template" }), {
      target: { value: "Dirty {context} {question}" }
    })
    mocks.save.mockRejectedValueOnce(new ServicePromptApiError(
      "The server or authenticated account changed before the request was sent.",
      { status: 412, code: "request_config_scope_changed" }
    ))

    fireEvent.click(screen.getByRole("button", { name: "Save changes" }))

    expect(await screen.findByText(/Server or account changed/i))
      .toBeInTheDocument()
    expect(screen.queryByText("This prompt changed on the server."))
      .not.toBeInTheDocument()
  })

  it("guards dirty query selection and eligible same-origin anchor navigation", async () => {
    renderSettings()
    await openPrompt("RAG answer")
    fireEvent.change(screen.getByRole("textbox", { name: "Template" }), {
      target: { value: "Dirty {context} {question}" }
    })
    vi.mocked(window.confirm).mockReturnValue(false)

    fireEvent.click(screen.getByRole("button", { name: "RAG follow-up rewrite" }))
    expect(screen.getByRole("heading", { name: "RAG answer" })).toBeInTheDocument()
    expect(window.location.search).toContain("chat.rag.answer")

    const settingsLink = document.createElement("a")
    settingsLink.href = "/settings/chat"
    settingsLink.textContent = "Chat settings"
    document.body.append(settingsLink)
    const event = new MouseEvent("click", { bubbles: true, cancelable: true, button: 0 })
    settingsLink.dispatchEvent(event)
    expect(event.defaultPrevented).toBe(true)
    expect(window.confirm).toHaveBeenCalled()
    settingsLink.remove()
  })

  it("guards same-extension anchors and ignores different extension hosts", async () => {
    renderSettings()
    await openPrompt("RAG answer")
    fireEvent.change(screen.getByRole("textbox", { name: "Template" }), {
      target: { value: "Dirty {context} {question}" }
    })

    const NativeURL = globalThis.URL
    const browserHref = window.location.href
    const extensionHref =
      "moz-extension://profile/options.html#/settings/prompt?prompt=chat.rag.answer"
    class OpaqueExtensionURL extends NativeURL {
      constructor(input: string | URL, base?: string | URL) {
        super(
          String(input) === browserHref && base === undefined
            ? extensionHref
            : input,
          base
        )
      }

      get origin() {
        return this.protocol === "moz-extension:"
          ? window.location.origin
          : super.origin
      }
    }
    vi.stubGlobal("URL", OpaqueExtensionURL)

    try {
      vi.mocked(window.confirm).mockReturnValue(false)
      const sameHostLink = document.createElement("a")
      sameHostLink.href = "moz-extension://profile/options.html#/settings/chat"
      document.body.append(sameHostLink)
      const sameHostClick = new MouseEvent("click", {
        bubbles: true,
        cancelable: true,
        button: 0
      })
      sameHostLink.dispatchEvent(sameHostClick)
      sameHostLink.remove()

      expect(sameHostClick.defaultPrevented).toBe(true)
      expect(window.confirm).toHaveBeenCalledOnce()

      vi.mocked(window.confirm).mockClear()
      vi.mocked(window.confirm).mockReturnValue(true)
      const otherHostLink = document.createElement("a")
      otherHostLink.href = "moz-extension://other/options.html#/settings/chat"
      document.body.append(otherHostLink)
      const otherHostClick = new MouseEvent("click", {
        bubbles: true,
        cancelable: true,
        button: 0
      })
      otherHostLink.dispatchEvent(otherHostClick)
      otherHostLink.remove()

      expect(otherHostClick.defaultPrevented).toBe(false)
      expect(window.confirm).not.toHaveBeenCalled()
      expect(window.history.state).not.toHaveProperty(
        "servicePromptHistoryForwardEntryToken"
      )
      const unload = new Event("beforeunload", { cancelable: true })
      window.dispatchEvent(unload)
      expect(unload.defaultPrevented).toBe(true)
    } finally {
      vi.stubGlobal("URL", NativeURL)
    }
  })

  it("ignores ineligible anchors while dirty", async () => {
    renderSettings()
    await openPrompt("RAG answer")
    fireEvent.change(screen.getByRole("textbox", { name: "Template" }), {
      target: { value: "Dirty {context} {question}" }
    })
    vi.mocked(window.confirm).mockClear()
    const cases = [
      { href: "https://external.test/path" },
      { href: "/settings/chat", target: "_blank" },
      { href: "/settings/chat", download: "file.txt" },
      { href: "/settings/chat", ctrlKey: true },
      { href: "/settings/chat", button: 1 },
      { href: "/settings/chat", prevented: true }
    ]
    for (const item of cases) {
      const link = document.createElement("a")
      link.href = item.href
      if (item.target) link.target = item.target
      if (item.download) link.download = item.download
      document.body.append(link)
      const event = new MouseEvent("click", {
        bubbles: true,
        cancelable: true,
        button: item.button ?? 0,
        ctrlKey: item.ctrlKey ?? false
      })
      if (item.prevented) event.preventDefault()
      link.dispatchEvent(event)
      link.remove()
    }
    expect(window.confirm).not.toHaveBeenCalled()
  })

  it("reverses declined popstate once, registers beforeunload, and cleans up listeners", async () => {
    const go = vi.spyOn(window.history, "go").mockImplementation(() => undefined)
    const addWindow = vi.spyOn(window, "addEventListener")
    const removeWindow = vi.spyOn(window, "removeEventListener")
    const removeDocument = vi.spyOn(document, "removeEventListener")
    const view = renderSettings()
    await openPrompt("RAG answer")
    const popstateRegistrations = addWindow.mock.calls.filter(
      ([eventName]) => eventName === "popstate"
    ).length
    fireEvent.change(screen.getByRole("textbox", { name: "Template" }), {
      target: { value: "Dirty {context} {question}" }
    })
    expect(addWindow.mock.calls.filter(
      ([eventName]) => eventName === "popstate"
    )).toHaveLength(popstateRegistrations)
    vi.mocked(window.confirm).mockReturnValue(false)

    const unload = new Event("beforeunload", { cancelable: true })
    window.dispatchEvent(unload)
    expect(unload.defaultPrevented).toBe(true)

    window.history.replaceState({}, "", "/settings/prompt")
    window.dispatchEvent(new PopStateEvent("popstate", {
      state: {}
    }))
    expect(go).toHaveBeenCalledTimes(1)
    expect(go).toHaveBeenCalledWith(1)
    const confirmations = vi.mocked(window.confirm).mock.calls.length
    window.dispatchEvent(new PopStateEvent("popstate", {
      state: { servicePromptHistoryIndex: 1 }
    }))
    expect(window.confirm).toHaveBeenCalledTimes(confirmations)

    view.unmount()
    expect(removeWindow).toHaveBeenCalledWith("beforeunload", expect.any(Function))
    expect(removeWindow).toHaveBeenCalledWith("popstate", expect.any(Function))
    expect(removeWindow).toHaveBeenCalledWith(
      SETTINGS_NAVIGATION_REQUEST_EVENT,
      expect.any(Function)
    )
    expect(removeDocument).toHaveBeenCalledWith(
      "click",
      expect.any(Function),
      true
    )
  })

  it("uses router history idx to reverse a real declined back navigation without adding history", async () => {
    window.history.replaceState({ idx: 40 }, "", "/settings")
    window.history.pushState({ idx: 41 }, "", "/settings/prompt")
    renderSettings()
    await openPrompt("RAG answer")
    const historyLength = window.history.length
    const editor = screen.getByRole("textbox", { name: "Template" })
    fireEvent.change(editor, {
      target: { value: "Dirty {context} {question}" }
    })
    editor.focus()
    vi.mocked(window.confirm).mockReturnValue(false)

    await act(async () => {
      window.history.back()
      await new Promise((resolve) => setTimeout(resolve, 0))
    })
    await waitFor(() => {
      expect(window.confirm).toHaveBeenCalled()
      expect(window.location.pathname).toBe("/settings/prompt")
      expect(new URLSearchParams(window.location.search).get("prompt"))
        .toBe("chat.rag.answer")
    })
    expect(window.history.length).toBe(historyLength)
    const restoredEditor = await screen.findByRole("textbox", { name: "Template" })
    await waitFor(() => expect(restoredEditor).toHaveFocus())
    expect(restoredEditor).toHaveValue("Dirty {context} {question}")
  })

  it("restores a dirty Next-style prompt entry after declined Back", async () => {
    window.history.replaceState(
      { __N: true, key: "next-root", hostMarker: "preserved" },
      "",
      "/settings/prompt"
    )
    installNextStyleHistory()
    renderSettings()
    await openPrompt("RAG answer")
    expect(window.history.state).toMatchObject({
      __N: true,
      hostMarker: "preserved",
      servicePromptHistoryIndex: 1
    })
    expect(window.history.state).not.toHaveProperty("idx")
    await openPrompt("RAG follow-up rewrite")
    expect(window.history.state).toMatchObject({
      __N: true,
      hostMarker: "preserved",
      servicePromptHistoryIndex: 2
    })
    expect(window.history.state).not.toHaveProperty("idx")

    const dirtyUrl = window.location.href
    const historyLength = window.history.length
    const editor = screen.getByRole("textbox", { name: "Template" })
    fireEvent.change(editor, {
      target: { value: "Back dirty {chat_history} {question}" }
    })
    editor.focus()
    vi.mocked(window.confirm).mockReturnValue(false)

    await act(async () => {
      window.history.back()
      await new Promise((resolve) => setTimeout(resolve, 0))
    })
    await waitFor(() => expect(window.location.href).toBe(dirtyUrl))
    expect(window.history.length).toBe(historyLength)
    const restored = await screen.findByRole("textbox", { name: "Template" })
    expect(restored).toHaveValue("Back dirty {chat_history} {question}")
    await waitFor(() => expect(restored).toHaveFocus())
  })

  it("restores a dirty Next-style prompt entry after declined Forward", async () => {
    window.history.replaceState(
      { __N: true, key: "next-root", hostMarker: "preserved" },
      "",
      "/settings/prompt"
    )
    installNextStyleHistory()
    renderSettings()
    await openPrompt("RAG answer")
    await openPrompt("RAG follow-up rewrite")
    const historyLength = window.history.length

    await act(async () => {
      window.history.back()
      await new Promise((resolve) => setTimeout(resolve, 0))
    })
    await screen.findByRole("heading", { name: "RAG answer" })
    expect(window.history.state).toMatchObject({
      __N: true,
      hostMarker: "preserved",
      servicePromptHistoryIndex: 1
    })
    expect(window.history.state).not.toHaveProperty("idx")

    const dirtyUrl = window.location.href
    const editor = screen.getByRole("textbox", { name: "Template" })
    fireEvent.change(editor, {
      target: { value: "Forward dirty {context} {question}" }
    })
    editor.focus()
    vi.mocked(window.confirm).mockReturnValue(false)

    await act(async () => {
      window.history.forward()
      await new Promise((resolve) => setTimeout(resolve, 0))
    })
    await waitFor(() => expect(window.location.href).toBe(dirtyUrl))
    expect(window.history.length).toBe(historyLength)
    const restored = await screen.findByRole("textbox", { name: "Template" })
    expect(restored).toHaveValue("Forward dirty {context} {question}")
    await waitFor(() => expect(restored).toHaveFocus())
  })

  it("stamps a delayed SPA destination during cleanup before declined reversal", async () => {
    window.history.replaceState(
      { __N: true, key: "delayed-prompt", hostMarker: "preserved" },
      "",
      "/settings/prompt?prompt=chat.rag.answer"
    )
    installNextStyleHistory()
    const client = createClient()
    render(
      <BrowserRouter>
        <QueryClientProvider client={client}>
          <App>
            <Routes>
              <Route
                path="/settings/prompt"
                element={(
                  <>
                    <ServicePromptsSettings />
                    <DelayedRouteLink />
                  </>
                )}
              />
              <Route path="*" element={<p>Delayed outside route</p>} />
            </Routes>
          </App>
        </QueryClientProvider>
      </BrowserRouter>
    )
    await screen.findByRole("heading", { name: "RAG answer" })

    fireEvent.click(screen.getByRole("link", {
      name: "Delayed settings test route"
    }))
    await screen.findByText("Delayed outside route")
    const destinationToken = window.history.state
      .servicePromptHistoryEntryToken as string
    expect(destinationToken).toEqual(expect.any(String))
    const historyLength = window.history.length

    await act(async () => {
      window.history.back()
      await new Promise((resolve) => setTimeout(resolve, 0))
    })
    await screen.findByRole("heading", { name: "RAG answer" })
    expect(window.history.state).toMatchObject({
      hostMarker: "preserved",
      servicePromptHistoryForwardEntryToken: destinationToken
    })

    const dirtyUrl = window.location.href
    const authoredValue = "Delayed route dirty {context} {question}"
    const editor = screen.getByRole("textbox", { name: "Template" })
    fireEvent.change(editor, { target: { value: authoredValue } })
    editor.focus()
    vi.mocked(window.confirm).mockReturnValue(false)

    await act(async () => {
      window.history.forward()
      await new Promise((resolve) => setTimeout(resolve, 0))
    })
    await screen.findByRole("heading", { name: "RAG answer" })
    await waitFor(() => expect(window.location.href).toBe(dirtyUrl))
    const restored = screen.getByRole("textbox", { name: "Template" })
    expect(restored).toHaveValue(authoredValue)
    await waitFor(() => expect(restored).toHaveFocus())
    expect(window.history.length).toBe(historyLength)
    expect(JSON.stringify(window.history.state)).not.toContain(authoredValue)
    for (const storage of [window.localStorage, window.sessionStorage]) {
      for (let index = 0; index < storage.length; index += 1) {
        expect(storage.getItem(storage.key(index)!) ?? "")
          .not.toContain(authoredValue)
      }
    }
  })

  it.each(["native UUID", "fallback token"] as const)(
    "uses %s entry identity for identical outside URLs",
    async (tokenSource) => {
    const randomUuidDescriptor = Object.getOwnPropertyDescriptor(
      window.crypto,
      "randomUUID"
    )
    if (tokenSource === "fallback token") {
      Object.defineProperty(window.crypto, "randomUUID", {
        configurable: true,
        value: undefined
      })
    }
    window.history.replaceState(
      { __N: true, key: "outside-a", hostMarker: "outside-a" },
      "",
      "/settings/chat"
    )
    window.history.pushState(
      { __N: true, key: "prompt-b", hostMarker: "prompt-b" },
      "",
      "/settings/prompt?prompt=chat.rag.answer"
    )
    installNextStyleHistory()
    const visitedStates: unknown[] = []
    const captureState = (event: PopStateEvent) => visitedStates.push(event.state)
    window.addEventListener("popstate", captureState)
    try {
      renderSettings()
      await screen.findByRole("heading", { name: "RAG answer" })

      fireEvent.click(screen.getByRole("link", {
        name: "Chat settings test route"
      }))
      expect(await screen.findByText("Outside workflow prompt route"))
        .toBeInTheDocument()
      const outsideUrl = window.location.href
      const destinationState = window.history.state as Record<string, unknown>
      expect(destinationState).toMatchObject({
        __N: true,
        hostMarker: "preserved",
        servicePromptHistoryEntryToken: expect.any(String)
      })
      const destinationToken = destinationState.servicePromptHistoryEntryToken
      const historyLength = window.history.length

      await act(async () => {
        window.history.back()
        await new Promise((resolve) => setTimeout(resolve, 0))
      })
      await screen.findByRole("heading", { name: "RAG answer" })
      expect(window.history.state).toMatchObject({
        __N: true,
        hostMarker: "preserved",
        servicePromptHistoryForwardEntryToken: destinationToken
      })

      const dirtyUrl = window.location.href
      const authoredValue = "Duplicate URL dirty {context} {question}"
      const editor = screen.getByRole("textbox", { name: "Template" })
      fireEvent.change(editor, { target: { value: authoredValue } })
      editor.focus()
      vi.mocked(window.confirm).mockReturnValue(false)

      await act(async () => {
        window.history.back()
        await new Promise((resolve) => setTimeout(resolve, 0))
      })
      await screen.findByRole("heading", { name: "RAG answer" })
      await waitFor(() => expect(window.location.href).toBe(dirtyUrl))
      expect(screen.getByRole("textbox", { name: "Template" }))
        .toHaveValue(authoredValue)
      expect(visitedStates).toContainEqual(expect.objectContaining({
        hostMarker: "outside-a"
      }))

      await act(async () => {
        window.history.forward()
        await new Promise((resolve) => setTimeout(resolve, 0))
      })
      await screen.findByRole("heading", { name: "RAG answer" })
      await waitFor(() => expect(window.location.href).toBe(dirtyUrl))
      expect(window.history.length).toBe(historyLength)
      expect(screen.getByRole("textbox", { name: "Template" }))
        .toHaveValue(authoredValue)
      expect(visitedStates).toContainEqual(expect.objectContaining({
        hostMarker: "preserved",
        servicePromptHistoryEntryToken: destinationToken
      }))
      expect(window.history.state).toMatchObject({
        __N: true,
        hostMarker: "preserved",
        servicePromptHistoryForwardEntryToken: destinationToken
      })
      expect(window.location.href).not.toBe(outsideUrl)
      expect(JSON.stringify([window.history.state, ...visitedStates]))
        .not.toContain(authoredValue)
      for (const storage of [window.localStorage, window.sessionStorage]) {
        for (let index = 0; index < storage.length; index += 1) {
          expect(storage.getItem(storage.key(index)!) ?? "")
            .not.toContain(authoredValue)
        }
      }
    } finally {
      window.removeEventListener("popstate", captureState)
      if (tokenSource === "fallback token") {
        if (randomUuidDescriptor) {
          Object.defineProperty(window.crypto, "randomUUID", randomUuidDescriptor)
        } else {
          Reflect.deleteProperty(window.crypto, "randomUUID")
        }
      }
    }
    }
  )

  it.each([
    { resolvedScope: scopeOne, restoresDraft: true, scopeCase: "matching scope" },
    { resolvedScope: scopeTwo, restoresDraft: false, scopeCase: "mismatched scope" }
  ] as const)(
    "claims the RAM handoff before slow $scopeCase resolution",
    async ({ resolvedScope, restoresDraft }) => {
      const delayedScope = deferred<typeof scopeOne | typeof scopeTwo>()
      mocks.resolveScope
        .mockResolvedValueOnce(scopeOne)
        .mockResolvedValueOnce(scopeOne)
        .mockReturnValueOnce(delayedScope.promise)
      window.history.replaceState(
        { __N: true, key: "next-root", hostMarker: "preserved" },
        "",
        "/settings/prompt"
      )
      installNextStyleHistory()
      renderSettings()
      await openPrompt("RAG answer")
      fireEvent.click(screen.getByRole("link", {
        name: "Chat settings test route"
      }))
      await screen.findByText("Outside workflow prompt route")
      await act(async () => {
        window.history.back()
        await new Promise((resolve) => setTimeout(resolve, 0))
      })
      await screen.findByRole("heading", { name: "RAG answer" })

      const realSetTimeout = globalThis.setTimeout
      let capsuleTimerCount = 0
      let expireCapsule: (() => void) | null = null
      vi.spyOn(globalThis, "setTimeout").mockImplementation(((
        ...args: Parameters<typeof setTimeout>
      ) => {
        const [handler, delay, ...handlerArgs] = args
        if (delay === 2_000 && typeof handler === "function") {
          capsuleTimerCount += 1
          expireCapsule = () => handler(...handlerArgs)
          return Number.MAX_SAFE_INTEGER as unknown as ReturnType<typeof setTimeout>
        }
        return realSetTimeout(...args)
      }) as typeof setTimeout)

      const dirtyUrl = window.location.href
      const authoredValue = "Slow scope dirty {context} {question}"
      fireEvent.change(screen.getByRole("textbox", { name: "Template" }), {
        target: { value: authoredValue }
      })
      vi.mocked(window.confirm).mockReturnValue(false)

      await act(async () => {
        window.history.forward()
        await new Promise((resolve) => setTimeout(resolve, 0))
      })
      await waitFor(() => expect(mocks.resolveScope).toHaveBeenCalledTimes(3))
      expect(window.location.href).toBe(dirtyUrl)
      expect(capsuleTimerCount).toBe(1)
      expect(expireCapsule).not.toBeNull()
      act(() => expireCapsule!())
      await act(async () => {
        delayedScope.resolve(resolvedScope)
        await delayedScope.promise
      })

      const restored = await screen.findByRole("textbox", { name: "Template" })
      if (restoresDraft) {
        expect(restored).toHaveValue(authoredValue)
        expect(screen.getByText("Unsaved")).toBeInTheDocument()
      } else {
        expect(restored).toHaveValue("Context: {context}\nQuestion: {question}")
        expect(restored).not.toHaveValue(authoredValue)
        expect(screen.queryByText("Unsaved")).not.toBeInTheDocument()
      }
      expect(JSON.stringify(window.history.state)).not.toContain(authoredValue)
      for (const storage of [window.localStorage, window.sessionStorage]) {
        for (let index = 0; index < storage.length; index += 1) {
          expect(storage.getItem(storage.key(index)!) ?? "")
            .not.toContain(authoredValue)
        }
      }
    }
  )

  it.each([
    {
      acceptsLeave: false,
      resolvedScope: scopeOne,
      restoresDraft: true,
      scopeCase: "matching scope"
    },
    {
      acceptsLeave: false,
      resolvedScope: scopeTwo,
      restoresDraft: false,
      scopeCase: "mismatched scope"
    },
    {
      acceptsLeave: true,
      resolvedScope: scopeOne,
      restoresDraft: false,
      scopeCase: "accepted leave"
    }
  ] as const)(
    "handles the claimed RAM handoff through scope rejection and Retry for $scopeCase",
    async ({ acceptsLeave, resolvedScope, restoresDraft }) => {
      mocks.resolveScope
        .mockResolvedValueOnce(scopeOne)
        .mockResolvedValueOnce(scopeOne)
        .mockRejectedValueOnce(new Error("scope unavailable"))
        .mockResolvedValueOnce(resolvedScope)
      window.history.replaceState(
        { __N: true, key: "next-root", hostMarker: "preserved" },
        "",
        "/settings/prompt"
      )
      installNextStyleHistory()
      renderSettings()
      await openPrompt("RAG answer")
      fireEvent.click(screen.getByRole("link", {
        name: "Chat settings test route"
      }))
      await screen.findByText("Outside workflow prompt route")
      await act(async () => {
        window.history.back()
        await new Promise((resolve) => setTimeout(resolve, 0))
      })
      await screen.findByRole("heading", { name: "RAG answer" })

      const dirtyUrl = window.location.href
      const authoredValue = "Retry scope dirty {context} {question}"
      fireEvent.change(screen.getByRole("textbox", { name: "Template" }), {
        target: { value: authoredValue }
      })
      vi.mocked(window.confirm).mockReturnValue(false)

      await act(async () => {
        window.history.forward()
        await new Promise((resolve) => setTimeout(resolve, 0))
      })
      expect(await screen.findByText(
        "Unable to resolve the connected server and account."
      )).toBeInTheDocument()
      expect(window.location.href).toBe(dirtyUrl)

      const hiddenUnload = new Event("beforeunload", { cancelable: true })
      window.dispatchEvent(hiddenUnload)
      expect(hiddenUnload.defaultPrevented).toBe(true)

      if (acceptsLeave) {
        vi.mocked(window.confirm).mockReturnValue(true)
        window.dispatchEvent(new PopStateEvent("popstate", {
          state: window.history.state
        }))
      }
      fireEvent.click(screen.getByRole("button", { name: "Retry" }))
      const restored = await screen.findByRole("textbox", { name: "Template" })
      if (restoresDraft) {
        expect(restored).toHaveValue(authoredValue)
        expect(screen.getByText("Unsaved")).toBeInTheDocument()
      } else {
        expect(restored).toHaveValue("Context: {context}\nQuestion: {question}")
        expect(restored).not.toHaveValue(authoredValue)
        expect(screen.queryByText("Unsaved")).not.toBeInTheDocument()
        const cleanUnload = new Event("beforeunload", { cancelable: true })
        window.dispatchEvent(cleanUnload)
        expect(cleanUnload.defaultPrevented).toBe(false)
      }
      expect(JSON.stringify(window.history.state)).not.toContain(authoredValue)
    }
  )

  it.each([
    { focusCase: "stable editor ID", expectedFocus: "editor" },
    { focusCase: "no saved ID", expectedFocus: "detail" },
    { focusCase: "a stale saved ID", expectedFocus: "detail" }
  ] as const)(
    "restores dirty prompt state and focus with $focusCase after declined outside-route Forward",
    async ({ focusCase, expectedFocus }) => {
    window.history.replaceState(
      { __N: true, key: "next-root", hostMarker: "preserved" },
      "",
      "/settings/prompt"
    )
    installNextStyleHistory()
    renderSettings()
    await openPrompt("RAG answer")
    const outsideUrl = new URL("/settings/chat", window.location.href).href
    fireEvent.change(screen.getByRole("textbox", { name: "Template" }), {
      target: { value: "Accepted leave {context} {question}" }
    })

    fireEvent.click(screen.getByRole("link", { name: "Chat settings test route" }))
    expect(await screen.findByText("Outside workflow prompt route"))
      .toBeInTheDocument()
    expect(window.location.href).toBe(outsideUrl)
    expect(window.history.state).toMatchObject({
      __N: true,
      hostMarker: "preserved"
    })
    expect(window.history.state).not.toHaveProperty("idx")
    const historyLength = window.history.length

    await act(async () => {
      window.history.back()
      await new Promise((resolve) => setTimeout(resolve, 0))
    })
    await screen.findByRole("heading", { name: "RAG answer" })
    expect(window.history.state).toMatchObject({
      __N: true,
      hostMarker: "preserved",
      servicePromptHistoryForwardEntryToken: expect.any(String),
      servicePromptHistoryIndex: 1
    })
    expect(window.history.state).not.toHaveProperty(
      "servicePromptHistoryForwardDestination"
    )

    const dirtyUrl = window.location.href
    const authoredValue = "Outside forward dirty {context} {question}"
    const editor = screen.getByRole("textbox", { name: "Template" })
    fireEvent.change(editor, {
      target: { value: authoredValue }
    })
    if (focusCase === "stable editor ID") {
      editor.focus()
    } else {
      const temporaryTarget = document.createElement("button")
      temporaryTarget.type = "button"
      temporaryTarget.textContent = "Temporary history focus target"
      if (focusCase === "a stale saved ID") {
        temporaryTarget.id = "removed-history-focus-target"
      }
      screen.getByRole("region", { name: "Workflow prompt details" })
        .append(temporaryTarget)
      temporaryTarget.focus()
    }
    vi.mocked(window.confirm).mockReturnValue(false)

    await act(async () => {
      window.history.forward()
      await new Promise((resolve) => setTimeout(resolve, 0))
    })
    await waitFor(() => expect(window.location.href).toBe(dirtyUrl))
    expect(window.history.length).toBe(historyLength)
    const restored = await screen.findByRole("textbox", { name: "Template" })
    expect(restored).toHaveValue(authoredValue)
    const restoredFocus = expectedFocus === "editor"
      ? restored
      : screen.getByRole("region", { name: "Workflow prompt details" })
    await waitFor(() => expect(restoredFocus).toHaveFocus())
    expect(JSON.stringify(window.history.state)).not.toContain(authoredValue)
    for (const storage of [window.localStorage, window.sessionStorage]) {
      for (let index = 0; index < storage.length; index += 1) {
        expect(storage.getItem(storage.key(index)!) ?? "").not.toContain(authoredValue)
      }
    }
    }
  )

  it.each(["success", "error"] as const)(
    "focuses the shared narrow-layout detail target after explicit selection %s",
    async (outcome) => {
      Object.defineProperty(window, "innerWidth", {
        value: 390,
        configurable: true
      })
      if (outcome === "error") {
        mocks.get.mockRejectedValueOnce(new Error("detail unavailable"))
      }
      renderSettings()
      fireEvent.click(await screen.findByRole("button", { name: "RAG answer" }))
      if (outcome === "success") {
        await screen.findByRole("heading", { name: "RAG answer" })
      } else {
        await screen.findByText("Unable to load this workflow prompt")
      }
      const detailTarget = screen.getByRole("region", {
        name: "Workflow prompt details"
      })
      await waitFor(() => expect(detailTarget).toHaveFocus())
    }
  )

  it("keeps the selected editor structurally usable on narrow layouts and renders prompt text as text only", async () => {
    Object.defineProperty(window, "innerWidth", { value: 390, configurable: true })
    renderSettings()
    await openPrompt("Text translation")

    const editorRegion = screen.getByRole("region", { name: "Text translation editor" })
    expect(editorRegion).toHaveClass("min-w-0")
    const system = screen.getByRole("textbox", { name: "System instructions" })
    fireEvent.change(system, { target: { value: "<img src=x onerror=alert(1)>" } })
    fireEvent.click(screen.getByRole("button", { name: "Preview" }))
    expect(await screen.findByText("<img src=x onerror=alert(1)>", { selector: "code" }))
      .toBeInTheDocument()
    expect(document.querySelector("img")).toBeNull()
  })
})
