// @vitest-environment jsdom
import React from "react"
import { act, fireEvent, render, renderHook, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import {
  MemoryRouter,
  HashRouter,
  useLocation,
  useNavigate,
} from "react-router-dom"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { useSimpleForm } from "@/hooks/useSimpleForm"
import { SidepanelImportedContextBanner } from "../SidepanelImportedContextBanner"
import { usePlaygroundQueueManagement } from "../hooks/usePlaygroundQueueManagement"
import { usePlaygroundSubmit } from "../hooks/usePlaygroundSubmit"
import { useSidepanelChatHandoffImport } from "../hooks/useSidepanelChatHandoffImport"
import { buildQueuedRequest, type QueuedRequest } from "@/utils/chat-request-queue"
import type {
  SidepanelChatHandoffPackage,
  SidepanelChatHandoffPageContext,
} from "@/services/sidepanel-chat-handoff"

const serviceMocks = vi.hoisted(() => ({
  readSidepanelChatHandoff: vi.fn(),
  consumeSidepanelChatHandoff: vi.fn(),
  buildSidepanelHandoffMessageForModel: vi.fn(
    (visibleDraft: string, context?: SidepanelChatHandoffPageContext) =>
      context
        ? `MODEL MESSAGE\nTitle: ${context.title ?? ""}\nUser draft: ${visibleDraft}`
        : visibleDraft,
  ),
}))

const storageMocks = vi.hoisted(() => ({ get: vi.fn(), remove: vi.fn() }))
vi.mock("@/utils/safe-storage", () => ({ createSafeStorage: () => storageMocks }))
vi.mock("@/services/sidepanel-chat-handoff", () => serviceMocks)

vi.mock("~/services/tldw-server", () => ({
  defaultEmbeddingModelForRag: vi.fn(async () => "embedding-model"),
}))

vi.mock("@/services/search", () => ({
  getIsSimpleInternetSearch: vi.fn(async () => false),
}))

vi.mock("@/utils/rag-format", () => ({
  formatPinnedResults: vi.fn(() => ""),
}))

vi.mock("@/utils/chat-model-availability", () => ({
  buildAvailableChatModelIds: vi.fn(() => new Set(["openai:gpt-4o-mini"])),
  findUnavailableChatModel: vi.fn(() => null),
  normalizeChatModelId: vi.fn((model: string | null | undefined) => model ?? ""),
}))

vi.mock("@/utils/image-generation-chat", () => ({
  IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE: "image-generation-assistant",
  IMAGE_GENERATION_USER_MESSAGE_TYPE: "image-generation-user",
}))

vi.mock("../usage-metrics", () => ({
  projectTokenBudget: vi.fn(() => ({
    isOverLimit: false,
    isNearLimit: false,
  })),
}))

vi.mock("@/hooks/chat/chat-action-utils", () => ({
  isChatSubmitSuccess: vi.fn((result: { status?: string }) =>
    result?.status === "submitted",
  ),
  normalizeChatSubmitResult: vi.fn((result) => result ?? { status: "submitted" }),
  throwIfChatSubmitUnsuccessful: vi.fn(),
}))

vi.mock("lucide-react", () => {
  const Icon = (props: React.SVGProps<SVGSVGElement>) => <svg {...props} />
  return new Proxy(
    { X: Icon },
    {
      get: (target, prop) => {
        if (prop === "then") return undefined
        return prop in target ? target[prop as keyof typeof target] : Icon
      },
    },
  )
})

const importedPageContext: SidepanelChatHandoffPageContext = {
  title: "Imported article",
  url: "https://example.test/article",
  snippets: [
    {
      kind: "visible-context",
      label: "Current page",
      text: "Important imported page context.",
    },
  ],
}

const buildPackage = (
  overrides: Partial<SidepanelChatHandoffPackage> = {},
): SidepanelChatHandoffPackage => ({
  id: "handoff-1",
  source: "sidepanel-chat",
  createdAt: "2026-05-29T10:00:00.000Z",
  expiresAt: "2026-05-29T10:10:00.000Z",
  draft: { text: "Imported draft" },
  pageContext: importedPageContext,
  ...overrides,
})

const notificationApi = {
  warning: vi.fn(),
  error: vi.fn(),
  info: vi.fn(),
}

const submitHarnessSendMessage = vi.fn<(payload: any) => Promise<null>>(
  async () => null,
)

const t = (_key: string, fallback?: any) =>
  typeof fallback === "string" ? fallback : _key

const LocationProbe = ({
  onChange,
}: {
  onChange?: (location: ReturnType<typeof useLocation>) => void
}) => {
  const location = useLocation()
  React.useEffect(() => {
    onChange?.(location)
  }, [location, onChange])

  return <><div data-testid="location-search">{location.search}</div><div data-testid="location-path">{location.pathname}</div><div data-testid="location-key">{location.key}</div></>
}

const HookHarness = ({
  initialDraft = "",
  onLocationChange,
}: {
  initialDraft?: string
  onLocationChange?: (location: ReturnType<typeof useLocation>) => void
}) => {
  const form = useSimpleForm({ initialValues: { message: initialDraft, image: "" } })
  const draft = form.values.message
  const setDraft = (value: string) => form.setFieldValue("message", value)
  const navigate = useNavigate()
  const handoffImport = useSidepanelChatHandoffImport({
    draftValue: draft,
    getDraftRevision: form.getRevision,
    setMessageValue: (value) => setDraft(value),
    notificationApi,
    t,
  })

  return (
    <div>
      <LocationProbe onChange={onLocationChange} />
      <button type="button" onClick={() => { setDraft("OTHER VIEW DRAFT  "); navigate("/other") }}>Leave chat</button>
      <button type="button" onClick={() => { setDraft("TRANSIENT EDIT"); setDraft(draft) }}>Edit then Undo</button>
      <textarea
        aria-label="Composer"
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
      />
      {handoffImport.importedContext ? (
        <SidepanelImportedContextBanner
          context={handoffImport.importedContext}
          onRemove={handoffImport.removeImportedContext}
        />
      ) : null}
      {handoffImport.conflict ? (
        <div role="status" aria-label="Sidepanel handoff conflict">
          {handoffImport.importPending ? <p>Checking copy</p> : null}
          {handoffImport.conflictFeedback ? <p>{handoffImport.conflictFeedback}</p> : null}
          <button type="button" disabled={handoffImport.importPending} onClick={handoffImport.insertHandoffDraft}>
            Insert
          </button>
          <button type="button" disabled={handoffImport.importPending} onClick={handoffImport.replaceWithHandoffDraft}>
            Replace
          </button>
          <button type="button" onClick={handoffImport.cancelHandoffImport}>
            Cancel
          </button>
        </div>
      ) : null}
    </div>
  )
}

const baseSubmitDeps = (overrides: Record<string, unknown> = {}) => ({
  form: {
    onSubmit: (callback: (value: any) => void | Promise<void>) =>
      () => void callback({ message: "Ask about this", image: "" }),
    setFieldValue: vi.fn(),
    setFieldError: vi.fn(),
    reset: vi.fn(),
  },
  draftReady: true,
  restorePromptAssistAttempt: vi.fn(),
  beginPromptAssistReset: vi.fn(() => 1),
  markPromptAssistAttemptSaved: vi.fn(),
  isSending: false,
  isConnectionReady: true,
  webSearch: false,
  compareModeActive: false,
  compareSelectedModels: [],
  selectedModel: "openai:gpt-4o-mini",
  fileRetrievalEnabled: false,
  ragPinnedResults: [],
  selectedDocuments: [],
  uploadedFiles: [],
  currentContextSnapshot: {},
  conversationTokenCount: 0,
  characterContextTokenEstimate: 0,
  pinnedSourceTokenEstimate: 0,
  resolvedMaxContext: 100_000,
  jsonMode: false,
  sendMessage: vi.fn(async () => null),
  clearSelectedDocuments: vi.fn(),
  clearUploadedFiles: vi.fn(),
  textAreaFocus: vi.fn(),
  setLastSubmittedContext: vi.fn(),
  estimateTokensForText: vi.fn(() => 1),
  resolveSubmissionIntent: vi.fn((message: string) => ({
    message,
    handled: false,
    invalidImageCommand: false,
    isImageCommand: false,
  })),
  queueSubmission: vi.fn(),
  validateSelectedChatModelsAvailability: vi.fn(() => true),
  compareModelsSupportCapability: vi.fn(() => true),
  notificationApi,
  t,
  ...overrides,
})

const SubmitImportHarness = ({
  submitOverrides,
}: {
  submitOverrides?: Record<string, unknown>
}) => {
  const form = useSimpleForm({ initialValues: { message: "", image: "" } })
  const draft = form.values.message
  const setDraft = (value: string) => form.setFieldValue("message", value)
  const handoffImport = useSidepanelChatHandoffImport({
    draftValue: draft,
    getDraftRevision: form.getRevision,
    setMessageValue: (value) => setDraft(value),
    notificationApi,
    t,
  })
  const { submitForm } = usePlaygroundSubmit(
    baseSubmitDeps({
      form,
      sendMessage: submitHarnessSendMessage,
      importedSidepanelContext: handoffImport.importedContext,
      clearImportedSidepanelContext: handoffImport.removeImportedContext,
      ...(submitOverrides ?? {}),
    }) as any,
  )

  return (
    <div>
      <textarea aria-label="Composer" value={draft} readOnly />
      {handoffImport.importedContext ? (
        <SidepanelImportedContextBanner
          context={handoffImport.importedContext}
          onRemove={handoffImport.removeImportedContext}
        />
      ) : null}
      <button type="button" onClick={() => submitForm()}>
        Send
      </button>
    </div>
  )
}

const renderWithRoute = (
  ui: React.ReactElement,
  initialEntry = "/chat?handoff=handoff-1",
) => render(<MemoryRouter initialEntries={[initialEntry]}>{ui}</MemoryRouter>)

describe("sidepanel chat handoff import", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    storageMocks.get.mockReset()
    storageMocks.remove.mockReset()
    submitHarnessSendMessage.mockResolvedValue(null)
    serviceMocks.readSidepanelChatHandoff.mockResolvedValue(buildPackage())
    serviceMocks.consumeSidepanelChatHandoff.mockResolvedValue(buildPackage())
  })

  it("imports a valid handoff, pre-fills the composer, renders context, then consumes the package", async () => {
    renderWithRoute(<HookHarness />)

    await waitFor(() =>
      expect(screen.getByLabelText("Composer")).toHaveValue("Imported draft"),
    )

    expect(
      screen.getByRole("region", { name: "Imported sidepanel context" }),
    ).toHaveTextContent("Imported article")
    expect(screen.getByTestId("location-search").textContent).toBe("")
    await waitFor(() =>
      expect(serviceMocks.consumeSidepanelChatHandoff).toHaveBeenCalledWith(
        "handoff-1",
      ),
    )
  })

  it.each([
    ["Destination  ", "  Incoming\n\n"],
    ["  \n\t", "Incoming"],
    ["Destination\n", "   "]
  ])("inserts exact whitespace in destination %j and copy %j", async (destination, incoming) => {
    serviceMocks.readSidepanelChatHandoff.mockResolvedValue(buildPackage({ draft: { text: incoming } }))
    renderWithRoute(<HookHarness initialDraft={destination} />)
    const insert = await screen.findByRole("button", { name: "Insert" })
    expect(screen.getByLabelText("Composer")).toHaveValue(destination)
    expect(serviceMocks.consumeSidepanelChatHandoff).not.toHaveBeenCalled()
    fireEvent.click(insert)
    await waitFor(() => expect(screen.getByLabelText("Composer")).toHaveValue(`${destination}\n\n${incoming}`))
  })

  it.each(["Insert", "Replace"])("rechecks availability before %s instead of applying an expired in-memory copy", async choice => {
    renderWithRoute(<HookHarness initialDraft="KEEP  " />)
    const action = await screen.findByRole("button", { name: choice })
    serviceMocks.readSidepanelChatHandoff.mockResolvedValueOnce(null)
    fireEvent.click(action)
    await waitFor(() => expect(notificationApi.warning).toHaveBeenCalled())
    expect(serviceMocks.readSidepanelChatHandoff).toHaveBeenCalledTimes(2)
    expect(screen.getByLabelText("Composer")).toHaveValue("KEEP  ")
    expect(serviceMocks.consumeSidepanelChatHandoff).not.toHaveBeenCalled()
  })

  it("checks one pending choice, keeps Cancel available, and retires its late read", async () => {
    let resolve!: (value: SidepanelChatHandoffPackage) => void
    renderWithRoute(<HookHarness initialDraft="KEEP  " />)
    const insert = await screen.findByRole("button", { name: "Insert" })
    serviceMocks.readSidepanelChatHandoff.mockReturnValueOnce(new Promise<SidepanelChatHandoffPackage>(r => { resolve = r }))
    fireEvent.click(insert); fireEvent.click(insert)
    expect(insert).toBeDisabled()
    expect(screen.getByRole("button", { name: "Replace" })).toBeDisabled()
    expect(screen.getByText("Checking copy")).toBeVisible()
    expect(serviceMocks.readSidepanelChatHandoff).toHaveBeenCalledTimes(2)
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }))
    await act(async () => { resolve(buildPackage()) })
    expect(screen.getByLabelText("Composer")).toHaveValue("KEEP  ")
    expect(screen.queryByRole("region", { name: "Imported sidepanel context" })).toBeNull()
  })

  it.each(["Insert", "Replace"])("keeps edits made during a pending %s and offers the choice again", async choice => {
    let resolve!: (value: SidepanelChatHandoffPackage) => void
    renderWithRoute(<HookHarness initialDraft="OLD  " />)
    const action = await screen.findByRole("button", { name: choice })
    serviceMocks.readSidepanelChatHandoff.mockReturnValueOnce(new Promise<SidepanelChatHandoffPackage>(r => { resolve = r }))
    fireEvent.click(action)
    fireEvent.change(screen.getByLabelText("Composer"), { target: { value: "NEW EXACT  \n" } })
    await act(async () => { resolve(buildPackage()) })
    expect(screen.getByLabelText("Composer")).toHaveValue("NEW EXACT  \n")
    expect(serviceMocks.consumeSidepanelChatHandoff).not.toHaveBeenCalled()
    expect(screen.getByText(/Your draft changed while the copy was checked/)).toBeVisible()
    expect(action).not.toBeDisabled()
  })

  it.each(["Insert", "Replace"])("retires a pending %s after an edit and Undo in the same batch", async choice => {
    let resolve!: (value: SidepanelChatHandoffPackage) => void
    renderWithRoute(<HookHarness initialDraft="KEEP  " />)
    const action = await screen.findByRole("button", { name: choice })
    serviceMocks.readSidepanelChatHandoff.mockReturnValueOnce(new Promise<SidepanelChatHandoffPackage>(r => { resolve = r }))
    fireEvent.click(action)
    fireEvent.click(screen.getByRole("button", { name: "Edit then Undo" }))
    await act(async () => { resolve(buildPackage()) })
    expect(screen.getByLabelText("Composer")).toHaveValue("KEEP  ")
    expect(serviceMocks.consumeSidepanelChatHandoff).not.toHaveBeenCalled()
    expect(screen.getByText(/Your draft changed while the copy was checked/)).toBeVisible()
    expect(action).not.toBeDisabled()
  })

  it("retires a choice on route departure without returning to the old chat or consuming its copy", async () => {
    let resolve!: (value: SidepanelChatHandoffPackage) => void
    renderWithRoute(<HookHarness initialDraft="KEEP  " />)
    const action = await screen.findByRole("button", { name: "Replace" })
    serviceMocks.readSidepanelChatHandoff.mockReturnValueOnce(new Promise<SidepanelChatHandoffPackage>(r => { resolve = r }))
    fireEvent.click(action)
    fireEvent.click(screen.getByRole("button", { name: "Leave chat" }))
    await act(async () => { resolve(buildPackage()) })
    expect(screen.getByLabelText("Composer")).toHaveValue("OTHER VIEW DRAFT  ")
    expect(screen.queryByRole("button", { name: "Insert" })).toBeNull()
    expect(serviceMocks.consumeSidepanelChatHandoff).not.toHaveBeenCalled()
    expect(notificationApi.warning).not.toHaveBeenCalled()
  })

  it("keeps a real storage read failure retryable without consuming or changing either draft", async () => {
    const actual = await vi.importActual<typeof import("@/services/sidepanel-chat-handoff")>("@/services/sidepanel-chat-handoff")
    const now = Date.now()
    const copy = buildPackage({ createdAt: new Date(now).toISOString(), expiresAt: new Date(now + actual.SIDEPANEL_CHAT_HANDOFF_TTL_MS).toISOString() })
    storageMocks.get.mockResolvedValue(copy)
    serviceMocks.readSidepanelChatHandoff.mockImplementation(actual.readSidepanelChatHandoff)
    renderWithRoute(<HookHarness initialDraft="KEEP  " />)
    const insert = await screen.findByRole("button", { name: "Insert" })
    storageMocks.get.mockRejectedValueOnce(new Error("read failed"))
    fireEvent.click(insert)
    await screen.findByText(/The copy could not be checked/)
    expect(screen.getByLabelText("Composer")).toHaveValue("KEEP  ")
    expect(screen.getByTestId("location-search")).toHaveTextContent("handoff=handoff-1")
    expect(serviceMocks.consumeSidepanelChatHandoff).not.toHaveBeenCalled()
    expect(storageMocks.remove).not.toHaveBeenCalled()
    expect(insert).not.toBeDisabled()
    fireEvent.click(insert)
    await waitFor(() => expect(screen.getByLabelText("Composer")).toHaveValue("KEEP  \n\nImported draft"))
    expect(storageMocks.get).toHaveBeenCalledTimes(3)
  })

  it("retires an acceptance read on unmount without consuming or announcing its result", async () => {
    let resolve!: (value: SidepanelChatHandoffPackage) => void
    const view = renderWithRoute(<HookHarness initialDraft="KEEP  " />)
    const insert = await screen.findByRole("button", { name: "Insert" })
    serviceMocks.readSidepanelChatHandoff.mockReturnValueOnce(new Promise<SidepanelChatHandoffPackage>(r => { resolve = r }))
    fireEvent.click(insert)
    view.unmount()
    await act(async () => { resolve(buildPackage()) })
    expect(serviceMocks.consumeSidepanelChatHandoff).not.toHaveBeenCalled()
    expect(notificationApi.warning).not.toHaveBeenCalled()
  })

  it("retires an extension hash-route departure even when both entries use the default router key", async () => {
    const originalUrl = window.location.href
    const originalState = window.history.state
    window.history.replaceState(null, "", "/options.html#/chat?handoff=handoff-1")
    let resolve!: (value: SidepanelChatHandoffPackage) => void
    const view = render(<HashRouter><HookHarness initialDraft="KEEP  " /></HashRouter>)
    try {
      const replace = await screen.findByRole("button", { name: "Replace" })
      expect(screen.getByTestId("location-key")).toHaveTextContent("default")
      serviceMocks.readSidepanelChatHandoff.mockReturnValueOnce(new Promise<SidepanelChatHandoffPackage>(r => { resolve = r }))
      fireEvent.click(replace)
      await act(async () => { window.location.hash = "/other?handoff=handoff-1" })
      await waitFor(() => expect(screen.getByTestId("location-path")).toHaveTextContent("/other"))
      expect(screen.getByTestId("location-key")).toHaveTextContent("default")
      await act(async () => { resolve(buildPackage()) })
      expect(screen.getByLabelText("Composer")).toHaveValue("KEEP  ")
      expect(serviceMocks.consumeSidepanelChatHandoff).not.toHaveBeenCalled()
      expect(screen.queryByRole("button", { name: "Insert" })).toBeNull()
    } finally {
      view.unmount()
      window.history.replaceState(originalState, "", originalUrl)
    }
  })

  it("does not consume before successful import when an existing local draft requires conflict resolution", async () => {
    renderWithRoute(<HookHarness initialDraft="Local draft" />)

    await screen.findByRole("status", {
      name: "Sidepanel handoff conflict",
    })

    expect(screen.getByLabelText("Composer")).toHaveValue("Local draft")
    expect(serviceMocks.consumeSidepanelChatHandoff).not.toHaveBeenCalled()
  })

  it("offers insert, replace, and cancel when a local draft exists", async () => {
    const user = userEvent.setup()
    let view = renderWithRoute(<HookHarness initialDraft="Local draft" />)

    expect(
      await screen.findByRole("button", { name: "Insert" }),
    ).toBeVisible()
    expect(screen.getByRole("button", { name: "Replace" })).toBeVisible()
    expect(screen.getByRole("button", { name: "Cancel" })).toBeVisible()

    await user.click(screen.getByRole("button", { name: "Insert" }))
    expect(screen.getByLabelText("Composer")).toHaveValue(
      "Local draft\n\nImported draft",
    )
    await waitFor(() =>
      expect(serviceMocks.consumeSidepanelChatHandoff).toHaveBeenCalledWith(
        "handoff-1",
      ),
    )

    view.unmount()
    vi.clearAllMocks()
    serviceMocks.readSidepanelChatHandoff.mockResolvedValue(buildPackage())
    view = renderWithRoute(<HookHarness initialDraft="Local draft" />)
    await user.click(await screen.findByRole("button", { name: "Replace" }))
    expect(screen.getByLabelText("Composer")).toHaveValue("Imported draft")
    await waitFor(() =>
      expect(serviceMocks.consumeSidepanelChatHandoff).toHaveBeenCalledWith(
        "handoff-1",
      ),
    )

    view.unmount()
    vi.clearAllMocks()
    serviceMocks.readSidepanelChatHandoff.mockResolvedValue(buildPackage())
    renderWithRoute(<HookHarness initialDraft="Local draft" />)
    await user.click(await screen.findByRole("button", { name: "Cancel" }))
    expect(screen.getByLabelText("Composer")).toHaveValue("Local draft")
    await waitFor(() =>
      expect(serviceMocks.consumeSidepanelChatHandoff).toHaveBeenCalledWith(
        "handoff-1",
      ),
    )
  })

  it("cleans only handoff from the hash-route query and preserves character params", async () => {
    renderWithRoute(
      <HookHarness />,
      "/chat?mode=character&characterId=char-1&handoff=handoff-1",
    )

    await waitFor(() =>
      expect(screen.getByLabelText("Composer")).toHaveValue("Imported draft"),
    )

    expect(screen.getByTestId("location-search").textContent).toBe(
      "?mode=character&characterId=char-1",
    )
  })

  it("includes imported page context in requestOverrides.messageForModel on submit", async () => {
    const user = userEvent.setup()
    renderWithRoute(<SubmitImportHarness />)

    await waitFor(() =>
      expect(screen.getByLabelText("Composer")).toHaveValue("Imported draft"),
    )
    await user.click(screen.getByRole("button", { name: "Send" }))

    await waitFor(() =>
      expect(
        serviceMocks.buildSidepanelHandoffMessageForModel,
      ).toHaveBeenCalledWith("Imported draft", importedPageContext),
    )
    expect(submitHarnessSendMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        message: "Imported draft",
        requestOverrides: {
          messageForModel:
            "MODEL MESSAGE\nTitle: Imported article\nUser draft: Imported draft",
        },
      }),
    )
  })

  it("merges OpenUI request mode with imported page context request overrides", async () => {
    const user = userEvent.setup()
    const clearOpenUIRequestMode = vi.fn()
    renderWithRoute(
      <SubmitImportHarness
        submitOverrides={{
          openUIRequestMode: true,
          clearOpenUIRequestMode,
        }}
      />,
    )

    await waitFor(() =>
      expect(screen.getByLabelText("Composer")).toHaveValue("Imported draft"),
    )
    await user.click(screen.getByRole("button", { name: "Send" }))

    expect(submitHarnessSendMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        message: "Imported draft",
        requestOverrides: {
          messageForModel:
            "MODEL MESSAGE\nTitle: Imported article\nUser draft: Imported draft",
          dynamicUIRequest: { renderer: "openui" },
        },
      }),
    )
    expect(clearOpenUIRequestMode).toHaveBeenCalledTimes(1)
  })

  it("sends a context-only handoff with a visible fallback prompt", async () => {
    const user = userEvent.setup()
    serviceMocks.readSidepanelChatHandoff.mockResolvedValue(
      buildPackage({ draft: { text: "" } }),
    )

    renderWithRoute(<SubmitImportHarness />)

    await screen.findByRole("region", { name: "Imported sidepanel context" })
    expect(screen.getByLabelText("Composer")).toHaveValue("")

    await user.click(screen.getByRole("button", { name: "Send" }))

    await waitFor(() =>
      expect(
        serviceMocks.buildSidepanelHandoffMessageForModel,
      ).toHaveBeenCalledWith("Summarize this page.", importedPageContext),
    )
    expect(submitHarnessSendMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        message: "Summarize this page.",
        requestOverrides: {
          messageForModel:
            "MODEL MESSAGE\nTitle: Imported article\nUser draft: Summarize this page.",
        },
      }),
    )
  })

  it("preserves imported page context when the send is queued", async () => {
    const user = userEvent.setup()
    const queueSubmission = vi.fn(() => ({ id: "queued-1" }))
    renderWithRoute(
      <SubmitImportHarness
        submitOverrides={{ isSending: true, queueSubmission }}
      />,
    )

    await waitFor(() =>
      expect(screen.getByLabelText("Composer")).toHaveValue("Imported draft"),
    )
    await user.click(screen.getByRole("button", { name: "Send" }))

    expect(queueSubmission).toHaveBeenCalledWith(
      expect.objectContaining({
        promptText: "Imported draft",
        requestOverrides: {
          messageForModel:
            "MODEL MESSAGE\nTitle: Imported article\nUser draft: Imported draft",
        },
      }),
    )
    expect(submitHarnessSendMessage).not.toHaveBeenCalled()
    expect(
      screen.queryByRole("region", { name: "Imported sidepanel context" }),
    ).not.toBeInTheDocument()
  })

  it("queues OpenUI request mode with imported page context and clears the one-shot mode", async () => {
    const user = userEvent.setup()
    const queueSubmission = vi.fn(() => ({ id: "queued-1" }))
    const clearOpenUIRequestMode = vi.fn()
    renderWithRoute(
      <SubmitImportHarness
        submitOverrides={{
          isSending: true,
          openUIRequestMode: true,
          clearOpenUIRequestMode,
          queueSubmission,
        }}
      />,
    )

    await waitFor(() =>
      expect(screen.getByLabelText("Composer")).toHaveValue("Imported draft"),
    )
    await user.click(screen.getByRole("button", { name: "Send" }))

    expect(queueSubmission).toHaveBeenCalledWith(
      expect.objectContaining({
        promptText: "Imported draft",
        requestOverrides: {
          messageForModel:
            "MODEL MESSAGE\nTitle: Imported article\nUser draft: Imported draft",
          dynamicUIRequest: { renderer: "openui" },
        },
      }),
    )
    expect(clearOpenUIRequestMode).toHaveBeenCalledTimes(1)
    expect(submitHarnessSendMessage).not.toHaveBeenCalled()
  })

  it("keeps imported page context when a resolved submit reports failure", async () => {
    const user = userEvent.setup()
    const sendMessage = vi.fn(async () => ({
      status: "failed" as const,
      errorMessage: "provider unavailable",
    }))
    renderWithRoute(<SubmitImportHarness submitOverrides={{ sendMessage }} />)

    await waitFor(() =>
      expect(screen.getByLabelText("Composer")).toHaveValue("Imported draft"),
    )
    await user.click(screen.getByRole("button", { name: "Send" }))

    await waitFor(() => expect(sendMessage).toHaveBeenCalledTimes(1))
    expect(
      screen.getByRole("region", { name: "Imported sidepanel context" }),
    ).toBeInTheDocument()
  })

  it("replays queued sidepanel context into requestOverrides.messageForModel", async () => {
    const queued = buildQueuedRequest({
      promptText: "Imported draft",
      sourceContext: {
        documents: [],
        isImageCommand: false,
        requestOverrides: {
          messageForModel: "MODEL MESSAGE\nUser draft: Imported draft",
        },
      },
      snapshot: {
        selectedModel: "openai:gpt-4o-mini",
        chatMode: "normal",
        webSearch: false,
        compareMode: false,
        compareSelectedModels: [],
        selectedSystemPrompt: null,
        selectedQuickPrompt: null,
        toolChoice: null,
        useOCR: false,
      },
    })
    const queueRef: { current: QueuedRequest[] } = { current: [queued] }
    const setQueuedMessages = vi.fn(
      (
        next:
          | QueuedRequest[]
          | ((prev: QueuedRequest[]) => QueuedRequest[]),
      ) => {
        queueRef.current =
          typeof next === "function" ? next(queueRef.current) : next
      },
    )
    const sendMessage = vi.fn(async () => ({ status: "submitted" as const }))

    renderHook(() =>
      usePlaygroundQueueManagement({
        composerModels: [{ id: "openai:gpt-4o-mini", is_configured: true }],
        isConnectionReady: true,
        isSending: false,
        selectedModel: "openai:gpt-4o-mini",
        chatMode: "normal",
        webSearch: false,
        compareMode: false,
        compareModeActive: false,
        compareSelectedModels: [],
        selectedSystemPrompt: "",
        selectedQuickPrompt: null,
        toolChoice: "auto",
        useOCR: false,
        selectedDocuments: [],
        uploadedFiles: [],
        contextFiles: [],
        documentContext: [],
        queuedMessages: queueRef.current,
        setQueuedMessages,
        historyId: null,
        serverChatId: null,
        conversationTokenCount: 0,
        resolvedMaxContext: 100_000,
        estimateTokensForText: vi.fn(() => 1),
        characterContextTokenEstimate: 0,
        pinnedSourceTokenEstimate: 0,
        currentContextSnapshot: {},
        setLastSubmittedContext: vi.fn(),
        setSelectedModel: vi.fn(),
        setChatMode: vi.fn(),
        setWebSearch: vi.fn(),
        setCompareMode: vi.fn(),
        setCompareSelectedModels: vi.fn(),
        setSelectedSystemPrompt: vi.fn(),
        setSelectedQuickPrompt: vi.fn(),
        setToolChoice: vi.fn(),
        setUseOCR: vi.fn(),
        compareModelsSupportCapability: vi.fn(() => true),
        sendMessage,
        stopStreamingRequest: vi.fn(),
        form: {
          setFieldError: vi.fn(),
          reset: vi.fn(),
        },
        clearSelectedDocuments: vi.fn(),
        clearUploadedFiles: vi.fn(),
        textAreaFocus: vi.fn(),
        notificationApi,
        t,
      }),
    )

    await waitFor(() => expect(sendMessage).toHaveBeenCalledTimes(1))
    expect(sendMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        message: "Imported draft",
        requestOverrides: expect.objectContaining({
          messageForModel: "MODEL MESSAGE\nUser draft: Imported draft",
        }),
      }),
    )
  })

  it("omits imported page context after the user removes the banner", async () => {
    const user = userEvent.setup()
    const sendMessage = vi.fn<(payload: any) => Promise<null>>(async () => null)
    const SubmitHarness = () => {
      const form = useSimpleForm({ initialValues: { message: "", image: "" } })
      const draft = form.values.message
      const setDraft = (value: string) => form.setFieldValue("message", value)
      const handoffImport = useSidepanelChatHandoffImport({
        draftValue: draft,
        getDraftRevision: form.getRevision,
        setMessageValue: (value) => setDraft(value),
        notificationApi,
        t,
      })
      const { submitForm } = usePlaygroundSubmit(
        baseSubmitDeps({
          form,
          sendMessage,
          importedSidepanelContext: handoffImport.importedContext,
          clearImportedSidepanelContext: handoffImport.removeImportedContext,
        }) as any,
      )

      return (
        <div>
          <textarea aria-label="Composer" value={draft} readOnly />
          {handoffImport.importedContext ? (
            <SidepanelImportedContextBanner
              context={handoffImport.importedContext}
              onRemove={handoffImport.removeImportedContext}
            />
          ) : null}
          <button type="button" onClick={() => submitForm()}>
            Send
          </button>
        </div>
      )
    }

    renderWithRoute(<SubmitHarness />)

    await screen.findByRole("region", { name: "Imported sidepanel context" })
    await user.click(
      screen.getByRole("button", {
        name: /remove imported context from imported article/i,
      }),
    )
    expect(
      screen.queryByRole("region", { name: "Imported sidepanel context" }),
    ).not.toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Send" }))

    await waitFor(() => expect(sendMessage).toHaveBeenCalledTimes(1))
    expect(sendMessage.mock.calls[0][0]).not.toHaveProperty("requestOverrides")
  })

  it("shows non-blocking feedback for expired or malformed handoffs", async () => {
    serviceMocks.readSidepanelChatHandoff.mockResolvedValue(null)

    renderWithRoute(<HookHarness initialDraft="Keep local" />)

    await waitFor(() => expect(notificationApi.warning).toHaveBeenCalled())
    expect(screen.getByLabelText("Composer")).toHaveValue("Keep local")
    expect(screen.getByTestId("location-search").textContent).toBe("")
    expect(serviceMocks.consumeSidepanelChatHandoff).not.toHaveBeenCalled()
  })
})
