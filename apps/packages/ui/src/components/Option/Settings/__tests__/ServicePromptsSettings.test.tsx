import React from "react"
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within
} from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { App } from "antd"
import { BrowserRouter } from "react-router-dom"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { ServicePromptsSettings } from "../ServicePromptsSettings"
import {
  ServicePromptApiError,
  type ServicePromptCatalogItem,
  type ServicePromptDetail
} from "@/services/tldw/domains/service-prompts"

const mocks = vi.hoisted(() => ({
  confirmDanger: vi.fn(),
  resolveScope: vi.fn(),
  readLegacy: vi.fn(),
  clearLegacy: vi.fn(),
  importLegacy: vi.fn(),
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
    renderServicePromptPart: (...args: Parameters<typeof actual.renderServicePromptPart>) => {
      mocks.renderPart(...args)
      return actual.renderServicePromptPart(...args)
    }
  }
})

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
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
    }
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
  const defaults = definition.id === "media.text.translation"
    ? {
        system: "Translate accurately. Literal {braces} stay literal.",
        user_template: "Translate to {target_language}:\n{text}"
      }
    : definition.id === "chat.rag.answer"
      ? { template: "Context: {context}\nQuestion: {question}" }
      : definition.id === "chat.rag.question_rewrite"
        ? { template: "History: {chat_history}\nQuestion: {question}" }
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
    authMode: "multi-user" as const,
    accessToken: "secret"
  },
  scopeKey: "server:research-one:auth:multi-user:org:none:user:42"
}

const scopeTwo = {
  config: {
    serverUrl: "https://research-two.test",
    authMode: "multi-user" as const,
    accessToken: "other-secret"
  },
  scopeKey: "server:research-two:auth:multi-user:org:none:user:84"
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

const renderSettings = (client = createClient()) => ({
  client,
  ...render(
    <BrowserRouter>
      <QueryClientProvider client={client}>
        <App>
          <ServicePromptsSettings />
        </App>
      </QueryClientProvider>
    </BrowserRouter>
  )
})

const openPrompt = async (name: string) => {
  fireEvent.click(await screen.findByRole("button", { name }))
  await screen.findByRole("heading", { name })
}

describe("ServicePromptsSettings", () => {
  beforeEach(() => {
    vi.resetAllMocks()
    window.history.replaceState({}, "", "/settings/prompt")
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

  it("renders the four localized definitions, query selection, status, workflows, and exact scope", async () => {
    window.history.replaceState(
      {},
      "",
      "/settings/prompt?prompt=media.text.translation"
    )
    renderSettings()

    expect(await screen.findAllByTestId("service-prompt-list-item"))
      .toHaveLength(4)
    expect(await screen.findByRole("heading", { name: "Text translation" }))
      .toBeInTheDocument()
    expect(screen.getByText("Server default")).toBeInTheDocument()
    expect(screen.getByText("Text translation", { selector: "li" }))
      .toBeInTheDocument()
    expect(screen.getByText("https://research-one.test")).toBeInTheDocument()
    expect(screen.getByText(scopeOne.scopeKey)).toBeInTheDocument()
    expect(screen.queryByText("Server translation")).not.toBeInTheDocument()
  })

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
        { signal: expect.any(AbortSignal) }
      )
    })
    expect(await screen.findByText("Customized")).toBeInTheDocument()
    expect(screen.getByRole("status")).toHaveTextContent(
      "Workflow prompt saved."
    )
  })

  it("replaces an in-flight save with migration import without leaving save loading stuck", async () => {
    mocks.readLegacy.mockResolvedValue([legacyRagCandidate])
    const pendingSave = deferred<ServicePromptDetail>()
    mocks.save.mockReturnValue(pendingSave.promise)
    const { client } = renderSettings()
    await screen.findByText("Browser-local workflow prompts found")
    await openPrompt("RAG answer")
    fireEvent.change(screen.getByRole("textbox", { name: "Template" }), {
      target: { value: "Saving {context} {question}" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Save changes" }))
    await waitFor(() => expect(mocks.save).toHaveBeenCalledTimes(1))
    const saveSignal = mocks.save.mock.calls[0][2].signal as AbortSignal

    fireEvent.click(screen.getByRole("button", { name: "Import to this server" }))
    await waitFor(() => expect(mocks.importLegacy).toHaveBeenCalledTimes(1))
    expect(saveSignal.aborted).toBe(true)
    await act(async () => {
      pendingSave.resolve(detailFor(catalog[0], {
        source: "user",
        parts: { template: "Late {context} {question}" }
      }))
    })
    await waitFor(() => expect(document.querySelector(".ant-btn-loading"))
      .toBeNull())
    expect((client.getQueryData([
      "service-prompts",
      scopeOne.scopeKey,
      "detail",
      "chat.rag.answer"
    ]) as ServicePromptDetail).effective_parts.template)
      .toBe(legacyRagCandidate.value)
  })

  it("replaces an in-flight save with reset and ignores the late save result", async () => {
    mocks.get.mockResolvedValue(detailFor(catalog[0], { source: "user" }))
    const pendingSave = deferred<ServicePromptDetail>()
    mocks.save.mockReturnValue(pendingSave.promise)
    const { client } = renderSettings()
    await openPrompt("RAG answer")
    fireEvent.change(screen.getByRole("textbox", { name: "Template" }), {
      target: { value: "Saving {context} {question}" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Save changes" }))
    await waitFor(() => expect(mocks.save).toHaveBeenCalledTimes(1))
    const saveSignal = mocks.save.mock.calls[0][2].signal as AbortSignal

    fireEvent.click(screen.getByRole("button", { name: "Reset to default" }))
    fireEvent.click(within(await screen.findByRole("dialog"))
      .getByRole("button", { name: "Reset" }))
    await waitFor(() => expect(mocks.reset).toHaveBeenCalledTimes(1))
    expect(saveSignal.aborted).toBe(true)
    await act(async () => {
      pendingSave.resolve(detailFor(catalog[0], {
        source: "user",
        parts: { template: "Late {context} {question}" }
      }))
    })
    await waitFor(() => expect(document.querySelector(".ant-btn-loading"))
      .toBeNull())
    expect((client.getQueryData([
      "service-prompts",
      scopeOne.scopeKey,
      "detail",
      "chat.rag.answer"
    ]) as ServicePromptDetail).effective_parts.template)
      .toBe("Context: {context}\nQuestion: {question}")
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
        { signal: expect.any(AbortSignal) }
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
        { signal: expect.any(AbortSignal) }
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
        { signal: expect.any(AbortSignal) }
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
      { signal: expect.any(AbortSignal) }
    ))
    expect(screen.queryByText("Browser-local workflow prompts found"))
      .not.toBeInTheDocument()
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

    fireEvent.click(screen.getByRole("button", { name: "Discard local values" }))
    const dialog = await screen.findByRole("dialog")
    expect(dialog).toHaveTextContent("Discard browser-local workflow prompts?")
    fireEvent.click(within(dialog).getByRole("button", { name: "Discard" }))

    await waitFor(() => {
      expect(mocks.clearLegacy).toHaveBeenCalledWith("chat.rag.answer")
      expect(mocks.clearLegacy).toHaveBeenCalledWith("chat.web_search.answer")
    })
    expect(await screen.findByText("1 browser-local prompt still needs attention."))
      .toBeInTheDocument()
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
      mocks.resolveScope.mockResolvedValueOnce(scopeOne).mockResolvedValueOnce(scopeTwo)
      renderSettings()
      await screen.findByText("Browser-local workflow prompts found")
      fireEvent.click(screen.getByRole("button", { name: "Discard local values" }))
      fireEvent.click(within(await screen.findByRole("dialog"))
        .getByRole("button", { name: "Discard" }))
      await waitFor(() => expect(mocks.clearLegacy).toHaveBeenCalledTimes(1))

      window.dispatchEvent(new Event("tldw:config-updated"))
      await waitFor(() => {
        expect(mocks.resolveScope).toHaveBeenCalledTimes(2)
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
    const removeWindow = vi.spyOn(window, "removeEventListener")
    const removeDocument = vi.spyOn(document, "removeEventListener")
    const view = renderSettings()
    await openPrompt("RAG answer")
    fireEvent.change(screen.getByRole("textbox", { name: "Template" }), {
      target: { value: "Dirty {context} {question}" }
    })
    vi.mocked(window.confirm).mockReturnValue(false)

    const unload = new Event("beforeunload", { cancelable: true })
    window.dispatchEvent(unload)
    expect(unload.defaultPrevented).toBe(true)

    window.history.replaceState({ servicePromptHistoryIndex: 0 }, "", "/settings/prompt")
    window.dispatchEvent(new PopStateEvent("popstate", {
      state: { servicePromptHistoryIndex: 0 }
    }))
    expect(go).toHaveBeenCalledTimes(1)
    const confirmations = vi.mocked(window.confirm).mock.calls.length
    window.dispatchEvent(new PopStateEvent("popstate", {
      state: { servicePromptHistoryIndex: 1 }
    }))
    expect(window.confirm).toHaveBeenCalledTimes(confirmations)

    view.unmount()
    expect(removeWindow).toHaveBeenCalledWith("beforeunload", expect.any(Function))
    expect(removeWindow).toHaveBeenCalledWith("popstate", expect.any(Function))
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
