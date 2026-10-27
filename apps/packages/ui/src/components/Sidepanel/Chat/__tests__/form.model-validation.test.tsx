import { resolveEffectiveAssistantState } from "@/hooks/chat/effective-assistant-state"
import { isProtectedTrackedPersonaUnavailable } from "@/types/assistant-selection"
import { isSavedRoleplayConversation } from "@/hooks/chat/roleplay-resume-source"
import { readFileSync } from "node:fs"
import path from "node:path"
import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import ts from "typescript"
import {
  QueryClient,
  QueryClientProvider,
  useQuery
} from "@tanstack/react-query"
import { X } from "lucide-react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { useChatModelCatalog } from "@/hooks/useChatModelCatalog"
import { EmptySidePanel } from "../empty"
import { useConnectionStore } from "@/store/connection"
import { ConnectionPhase } from "@/types/connection"
import { useSimpleForm } from "@/hooks/useSimpleForm"
import { ModelSelect } from "@/components/Common/ModelSelect"
import {
  buildAvailableChatModelIds,
  findUnavailableChatModel,
  normalizeChatModelId
} from "@/utils/chat-model-availability"

const mocks = vi.hoisted(() => ({
  useMessage: vi.fn(),
  fetchChatModels: vi.fn(),
  send: vi.fn(),
  queue: vi.fn(),
  loading: vi.fn()
}))
const models = [{ model: "ready-model", provider: "llamacpp" }]
const t = (key: string, fallback?: string) => fallback ?? key
vi.mock("@/hooks/useMessage", () => ({ useMessage: mocks.useMessage }))
vi.mock("@/hooks/chat/useSelectedModel", () => ({
  useSelectedModel: () => ({
    ...mocks.useMessage(),
    selectedModelSaveFailed: false,
    retrySelectedModelSave: async () => false
  })
}))
vi.mock("@/services/tldw-server", () => ({
  fetchChatModels: mocks.fetchChatModels
}))
vi.mock("react-i18next", () => ({ useTranslation: () => ({ t }) }))
vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, initial: unknown) => React.useState(initial)
}))

// Execute the real sidebar submit callback, validation effects and inline JSX.
// This keeps unrelated audio/ingestion hooks out of this regression without
// copying the failing logic or adding a production abstraction just for tests.
const source = readFileSync(path.resolve(__dirname, "../form.tsx"), "utf8")
const ast = ts.createSourceFile(
  "form.tsx",
  source,
  ts.ScriptTarget.Latest,
  true,
  ts.ScriptKind.TSX
)
const nodes: ts.Node[] = []
const visit = (node: ts.Node) => {
  nodes.push(node)
  ts.forEachChild(node, visit)
}
visit(ast)
const initializer = (name: string) => {
  const declaration = nodes.find(
    (node): node is ts.VariableDeclaration =>
      ts.isVariableDeclaration(node) && node.name.getText(ast) === name
  )
  if (!declaration?.initializer)
    throw new Error(`Missing sidebar declaration: ${name}`)
  return declaration.initializer.getText(ast)
}
const optionalInitializer = (name: string, fallback: string) => {
  const node = nodes.find((node): node is ts.VariableDeclaration => ts.isVariableDeclaration(node) && node.name.getText(ast) === name)
  return node?.initializer?.getText(ast) ?? fallback
}
const effects = nodes
  .filter(
    (node) =>
      ts.isExpressionStatement(node) &&
      ts.isCallExpression(node.expression) &&
      node.expression.expression.getText(ast) === "React.useEffect" &&
      node.getText(ast).includes('form.clearFieldError("message")')
  )
  .map((node) => node.getText(ast))
  .join("\n")
if (!effects) throw new Error("Missing sidebar validation effects")
const catalogQuery = nodes.find(
  (node): node is ts.VariableDeclaration =>
    ts.isVariableDeclaration(node) &&
    ts.isObjectBindingPattern(node.name) &&
    Boolean(
      node.initializer
        ?.getText(ast)
        .includes("useChatModelCatalog(isConnectionReady)")
    )
)
if (!catalogQuery) throw new Error("Missing sidebar model catalog query")
const catalogDataName = (catalogQuery.name as ts.ObjectBindingPattern).elements
  .find((element) => element.propertyName?.getText(ast) === "data")
  ?.name.getText(ast)
if (!catalogDataName) throw new Error("Missing sidebar model catalog data")
const trackedLoadingNode = nodes.find((node): node is ts.JsxAttribute =>
  ts.isJsxAttribute(node) && node.name.getText(ast) === "trackedSelectionLoading")
if (!trackedLoadingNode?.initializer || !ts.isJsxExpression(trackedLoadingNode.initializer) || !trackedLoadingNode.initializer.expression)
  throw new Error("Missing actual compact tracked picker loading binding")
const trackedLoadingExpression = trackedLoadingNode.initializer.expression.getText(ast)
// XS-02: in Pro mode the model chip lives in ControlRow. Read which ref the
// form hands it, so the harness can mount the chip on that same ref.
const controlRowModelPickerRef = nodes
  .filter((node): node is ts.JsxSelfClosingElement =>
    ts.isJsxSelfClosingElement(node) && node.tagName.getText(ast) === "ControlRow")
  .flatMap((node) => node.attributes.properties)
  .find((attr): attr is ts.JsxAttribute =>
    ts.isJsxAttribute(attr) && attr.name.getText(ast) === "modelPickerRef")
  ?.initializer?.getText(ast)
  .replace(/^\{|\}$/g, "")
  .trim()
const compiled = ts.transpileModule(
  `
  return function useSidebarValidation(scope) {
    const { form, selectedModel, isSending, isConnectionReady, isProMode, draftReady, protectedHistory = false, personaMode = "none" } = scope;
    const audioHealthEnabled = false, greetingHistorySelection = protectedHistory ? { getCurrent: () => ({ status: "ready", owner: null, view: null }) } : null;
    const effectiveAssistantState = resolveEffectiveAssistantState({ tracked: personaMode === "tracked" ? { assistantKind: "persona", assistantId: "loaded-persona" } : null,
      settings: personaMode === "overlay" ? { assistantOverlay: { kind: "persona", id: "loaded-persona", name: "Persona overlay", system_prompt: "Fixture overlay" } } : null });
    const personaHistoryUnavailable = ${optionalInitializer("personaHistoryUnavailable", "false")};
    const personaSubmitDisabledReason = ${optionalInitializer("personaSubmitDisabledReason", "null")};
    const personaSubmitDisabledReasonRef = { current: personaSubmitDisabledReason }, greetingSubmitGateRef = { current: () => false };
    const getSubmitDisabledReason = ${optionalInitializer("getSubmitDisabledReason", "() => null")};
    const fetchChatModels = mocks.fetchChatModels;
    const ${catalogQuery.getText(ast)};
    const availableChatModelIds = ${initializer("availableChatModelIds")};
    const modelPickerRef = ${initializer("modelPickerRef")};
    const contextFiles = [], selectedDocuments = [];
    const streaming = false, sendWhenEnter = true, greetingStarting = false;
    const connectionReadyRef = { current: isConnectionReady }, connectionGenerationRef = { current: 0 };
    const resolveSubmissionIntent = (combinedMessage) => ({ combinedMessage, isImageCommand: false });
    const notification = { error: () => {}, info: mocks.loading };
    const queueSubmission = mocks.queue;
    const stopListening = async () => {};
    const stopServerDictation = () => {};
    const sendCurrentFormMessageRef = { current: mocks.send };
    ${effects}
    const submit = ${initializer("submitCurrentRequest")};
    const inline = ${initializer("composerInlineMessagesNode")};
    return { submit, inline, modelPickerRef, catalogReady: ${catalogDataName} !== undefined, trackedSelectionLoading: ${trackedLoadingExpression} };
  }
`,
  { compilerOptions: { jsx: ts.JsxEmit.React, target: ts.ScriptTarget.ES2022 } }
).outputText
const useSidebarValidation = new Function(
  "React",
  "ModelSelect",
  "X",
  "t",
  "useQuery",
  "useChatModelCatalog",
  "mocks",
  "buildAvailableChatModelIds",
  "findUnavailableChatModel",
  "normalizeChatModelId",
  "isSavedRoleplayConversation",
  "resolveEffectiveAssistantState",
  "isProtectedTrackedPersonaUnavailable",
  compiled
)(
  React,
  ModelSelect,
  X,
  t,
  useQuery,
  useChatModelCatalog,
  mocks,
  buildAvailableChatModelIds,
  findUnavailableChatModel,
  normalizeChatModelId,
  isSavedRoleplayConversation,
  resolveEffectiveAssistantState,
  isProtectedTrackedPersonaUnavailable
) as (scope: {
  form: ReturnType<typeof useSimpleForm<{ message: string; image: string }>>
  selectedModel: string
  isSending: boolean
  isConnectionReady: boolean
  isProMode: boolean
  draftReady: boolean
  protectedHistory?: boolean
  personaMode?: "none" | "tracked" | "overlay"
}) => {
  submit: (message: string, image: string) => Promise<void>
  inline: React.ReactNode
  modelPickerRef: React.RefObject<React.ComponentRef<typeof ModelSelect>>
  catalogReady: boolean
  trackedSelectionLoading: boolean
}

// Execute the deployed composer-message effect with the real controlled form.
const messageEffect = nodes.find(node => ts.isExpressionStatement(node) &&
  ts.isCallExpression(node.expression) && node.expression.expression.getText(ast) === "React.useEffect" &&
  node.getText(ast).includes('"tldw:set-composer-message"'));
if (!messageEffect) throw new Error("Missing actual composer message effect");
const useActualComposerMessage = new Function("React", ts.transpileModule(
  `return function useActualComposerMessage(form, textareaRef) { ${messageEffect.getText(ast)} }`,
  { compilerOptions: { target: ts.ScriptTarget.ES2022 } }
).outputText)(React) as (form: ReturnType<typeof useSimpleForm<{message: string; image: string}>>, ref: React.RefObject<HTMLTextAreaElement>) => void;

function Harness({
  model = "",
  busy = false,
  connected = true,
  pro = false,
  draftReady = true,
  text = "keep my draft",
  image = "",
  protectedHistory = false,
  starters = false,
  personaMode = "none" as "none" | "tracked" | "overlay"
}) {
  const form = useSimpleForm({ initialValues: { message: text, image } })
  const textareaRef = React.useRef<HTMLTextAreaElement>(null);
  useActualComposerMessage(form, textareaRef);
  const [controlsOpen, setControlsOpen] = React.useState(false);
  const [selectedModel, setSelectedModel] = React.useState(model)
  mocks.useMessage.mockReturnValue({ selectedModel, setSelectedModel: async (next: string) => { setSelectedModel(next); return true } })
  const { submit, inline, modelPickerRef, catalogReady, trackedSelectionLoading } = useSidebarValidation({
    form,
    selectedModel,
    isSending: busy,
    isConnectionReady: connected,
    isProMode: pro,
    draftReady,
    protectedHistory,
    personaMode
  })
  return (
    <>
      {starters && <>
        <EmptySidePanel inputRef={textareaRef} />
        <button onClick={() => setControlsOpen(true)}>Character controls</button>
        {controlsOpen && <div role="dialog" aria-label="Character controls"><button onClick={() => setControlsOpen(false)}>Close character controls</button></div>}
      </>}
      <textarea ref={textareaRef} aria-label="Message" {...form.getInputProps("message")} />
      <button
        onClick={() => void submit(form.values.message, form.values.image)}
      >
        Send
      </button>
      <output data-testid="model-catalog">
        {catalogReady ? "ready" : "pending"}
      </output>
      <output data-testid="tracked-selection-loading">{String(trackedSelectionLoading)}</output>
      {inline}
      {pro && controlRowModelPickerRef === "modelPickerRef" && (
        <div data-testid="pro-control-row">
          <ModelSelect ref={modelPickerRef} showSelectedName />
        </div>
      )}
    </>
  )
}

let queryClient: QueryClient
const renderHarness = (ui: React.ReactElement) =>
  render(ui, {
    wrapper: ({ children }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    )
  })
const waitForCatalog = () =>
  waitFor(() =>
    expect(screen.getByTestId("model-catalog")).toHaveTextContent("ready")
  )

describe("sidebar model validation", () => {
  it.each([["chat-suggestion-1", "Summarize this page"], ["chat-suggestion-2", "What are the key points?"]])("retains starter %s through controls and submits it exactly once", async (id, message) => {
    const previous = useConnectionStore.getState().state;
    useConnectionStore.setState({ state: { ...previous, phase: ConnectionPhase.CONNECTED, isConnected: true, serverUrl: "https://starter-fixture.example" } });
    try {
      renderHarness(<Harness model="ready-model" text="" starters />);
      await waitForCatalog();
      fireEvent.click(screen.getByTestId(id));
      expect(screen.getByLabelText("Message")).toHaveValue(message);
      fireEvent.click(screen.getByRole("button", { name: "Character controls", exact: true }));
      fireEvent.click(screen.getByRole("button", { name: "Close character controls" }));
      expect(screen.getByLabelText("Message")).toHaveValue(message);
      fireEvent.click(screen.getByRole("button", { name: "Send", exact: true }));
      await waitFor(() => expect(mocks.send).toHaveBeenCalledExactlyOnceWith(message, "", undefined));
      expect(mocks.queue).not.toHaveBeenCalled();
    } finally { useConnectionStore.setState({ state: previous }); }
  });

  it.each(["failed", "offline"])("ends compact tracked waiting when discovery is %s", async mode => {
    if (mode === "failed") mocks.fetchChatModels.mockRejectedValueOnce(new Error("Declared model discovery failure"))
    renderHarness(<Harness protectedHistory connected={mode !== "offline"} />)
    if (mode === "failed") await waitFor(() => expect(queryClient.getQueryCache().find({ queryKey: ["playground:chatModels"], exact: false })?.state.status).toBe("error"))
    await waitFor(() => expect(screen.getByTestId("tracked-selection-loading")).toHaveTextContent("false"))
    expect(screen.getByLabelText("Message")).toHaveValue("keep my draft")
    expect(mocks.send).not.toHaveBeenCalled()
  })

  it("holds compact tracked waiting only through its actual model fetch", async () => {
    let settle!: (value: typeof models) => void
    mocks.fetchChatModels.mockImplementationOnce(() => new Promise(resolve => { settle = resolve }))
    renderHarness(<Harness protectedHistory />)
    await waitFor(() => expect(screen.getByTestId("tracked-selection-loading")).toHaveTextContent("true"))
    await act(async () => { settle(models) })
    await waitForCatalog()
    expect(screen.getByTestId("tracked-selection-loading")).toHaveTextContent("false")
  })

  it.each([false, true])("refuses loaded protected persona before direct or queued preparation (busy=%s)", async busy => {
    renderHarness(<Harness model="ready-model" busy={busy} protectedHistory personaMode="tracked" />)
    await waitForCatalog()
    fireEvent.click(screen.getByText("Send"))
    await waitFor(() => expect(screen.getByText(/Tracked persona chat is unavailable here/)).toBeVisible())
    expect(mocks.send).not.toHaveBeenCalled()
    expect(mocks.queue).not.toHaveBeenCalled()
    expect(screen.getByRole("textbox", { name: "Message" })).toHaveValue("keep my draft")
  })

  it.each([{ protectedHistory: false, personaMode: "tracked" as const }, { protectedHistory: true, personaMode: "overlay" as const }])(
    "keeps supported persona lane available ($protectedHistory/$personaMode)", async props => {
      renderHarness(<Harness model="ready-model" {...props} />)
      await waitForCatalog()
      fireEvent.click(screen.getByText("Send"))
      await waitFor(() => expect(mocks.send).toHaveBeenCalledOnce())
    }
  )

  beforeEach(() => {
    vi.clearAllMocks()
    mocks.fetchChatModels.mockResolvedValue(models)
    queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false, staleTime: Infinity } }
    })
  })
  afterEach(() => queryClient.clear())

  it.each(["", "  ", "tldw:"])(
    "keeps missing-model feedback and draft for selection %j",
    async (model) => {
      const view = renderHarness(<Harness model={model} />)
      fireEvent.click(screen.getByText("Send"))
      await waitFor(() =>
        expect(screen.getByRole("alert")).toHaveTextContent("formError.noModel")
      )
      view.rerender(<Harness model={model} />)
      expect(screen.getByRole("alert")).toBeVisible()
      expect(screen.getByRole("textbox", { name: "Message" })).toHaveValue("keep my draft")
      expect(mocks.send).not.toHaveBeenCalled()
      expect(mocks.queue).not.toHaveBeenCalled()
    }
  )

  it("keeps unavailable-model feedback despite a nonempty selection", async () => {
    renderHarness(<Harness model="removed-model" />)
    await waitForCatalog()
    fireEvent.click(screen.getByText("Send"))
    await waitFor(() =>
      expect(screen.getByRole("alert")).toHaveTextContent(
        "Selected model is not available"
      )
    )
    expect(mocks.send).not.toHaveBeenCalled()
  })

  it.each(["", "removed-model"])(
    "offers the real Casual picker and sends the retained draft after selection %j",
    async (model) => {
      renderHarness(<Harness model={model} />)
      await waitForCatalog()
      fireEvent.click(screen.getByText("Send"))
      // XS-02: the blocked Send opens the picker itself.
      fireEvent.click(
        await screen.findByRole("menuitem", { name: /ready-model/ })
      )
      await waitFor(() =>
        expect(screen.queryByRole("alert")).not.toBeInTheDocument()
      )
      expect(screen.getByRole("textbox", { name: "Message" })).toHaveValue("keep my draft")
      fireEvent.click(screen.getByText("Send"))
      await waitFor(() =>
        expect(mocks.send).toHaveBeenCalledWith("keep my draft", "", undefined)
      )
    }
  )

  it("clears an error after a draft edit, then validates the next Send again", async () => {
    renderHarness(<Harness />)
    fireEvent.click(screen.getByText("Send"))
    await screen.findByRole("alert")
    fireEvent.change(screen.getByRole("textbox", { name: "Message" }), {
      target: { value: "edited draft" }
    })
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
    fireEvent.click(screen.getByText("Send"))
    await waitFor(() => expect(screen.getByRole("alert")).toBeVisible())
  })

  it("supports explicit error dismissal without discarding the draft", async () => {
    renderHarness(<Harness />)
    fireEvent.click(screen.getByText("Send"))
    fireEvent.click(await screen.findByRole("button", { name: "Dismiss" }))
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
    expect(screen.getByRole("textbox", { name: "Message" })).toHaveValue("keep my draft")
  })

  it("validates image-only input in Pro mode", async () => {
    renderHarness(
      <Harness pro text="" image="data:image/png;base64,attachment" />
    )
    fireEvent.click(screen.getByText("Send"))
    await waitFor(() =>
      expect(screen.getByRole("alert")).toHaveTextContent("formError.noModel")
    )
    expect(mocks.send).not.toHaveBeenCalled()
  })

  it("loads the ordinary catalog without activating optional audio controls", async () => {
    renderHarness(<Harness model="ready-model" />)
    await waitForCatalog()
    expect(mocks.fetchChatModels).toHaveBeenCalledTimes(1)
  })

  it("does not fetch a model catalog before connection", async () => {
    renderHarness(<Harness connected={false} />)
    await act(async () => {})
    expect(mocks.fetchChatModels).not.toHaveBeenCalled()
  })

  it("ignores empty input", async () => {
    renderHarness(<Harness text="" />)
    await act(async () => fireEvent.click(screen.getByText("Send")))
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
    expect(mocks.send).not.toHaveBeenCalled()
    expect(mocks.queue).not.toHaveBeenCalled()
  })

  it.each([{}, { busy: true }, { connected: false }])("retains an unrestored draft without send or queue for %j", async props => {
    renderHarness(<Harness model="ready-model" draftReady={false} {...props} />)
    await act(async () => fireEvent.click(screen.getByText("Send")))
    expect(mocks.send).not.toHaveBeenCalled()
    expect(mocks.queue).not.toHaveBeenCalled()
    expect(mocks.loading).toHaveBeenCalledOnce()
    expect(screen.getByLabelText("Message")).toHaveValue("keep my draft")
  })

  it("XS-02: sends with an available model and keeps showing which model it uses", async () => {
    renderHarness(<Harness model="ready-model" />)
    await waitForCatalog()
    fireEvent.click(screen.getByText("Send"))
    await waitFor(() =>
      expect(mocks.send).toHaveBeenCalledWith("keep my draft", "", undefined)
    )
    expect(await screen.findByTestId("chat-model-select")).toHaveTextContent("ready-model")
  })

  it("XS-02: an empty model chip is named by its visible 'Choose model' text", async () => {
    renderHarness(<Harness model="" />)
    await waitForCatalog()
    const chip = await screen.findByTestId("chat-model-select")
    expect(chip).toHaveTextContent("Choose model")
    expect(chip).toHaveAccessibleName("Choose model")
  })

  it.each(["", "removed-model"])(
    "XS-02: opens the Casual model picker when Send is blocked by the model %j",
    async (model) => {
      renderHarness(<Harness model={model} />)
      await waitForCatalog()
      await screen.findByTestId("chat-model-select")
      fireEvent.click(screen.getByText("Send"))
      await screen.findByRole("alert")
      const search = await screen.findByPlaceholderText("Search models")
      // Keyboard users land in the picker's search, and stay there.
      await waitFor(() => expect(search).toHaveFocus())
      await act(() => new Promise((resolve) => setTimeout(resolve, 100)))
      expect(search).toHaveFocus()
      expect(await screen.findByRole("menuitem", { name: /ready-model/ })).toBeInTheDocument()
      expect(mocks.send).not.toHaveBeenCalled()
    }
  )

  it("XS-02: opens the Pro control row's model picker when Send is blocked by the model", async () => {
    renderHarness(<Harness model="" pro />)
    await waitForCatalog()
    const row = await screen.findByTestId("pro-control-row")
    expect(row).toContainElement(screen.getByTestId("chat-model-select"))
    fireEvent.click(screen.getByText("Send"))
    await screen.findByRole("alert")
    const search = await screen.findByPlaceholderText("Search models")
    await waitFor(() => expect(search).toHaveFocus())
    expect(mocks.send).not.toHaveBeenCalled()
  })

  it("XS-02: Escape in the picker's search closes it and returns focus to the model chip", async () => {
    renderHarness(<Harness model="" />)
    await waitForCatalog()
    const chip = await screen.findByTestId("chat-model-select")
    fireEvent.click(screen.getByText("Send"))
    const search = await screen.findByPlaceholderText("Search models")
    await waitFor(() => expect(search).toHaveFocus())
    fireEvent.keyDown(search, { key: "Escape", code: "Escape", keyCode: 27 })
    await waitFor(() => expect(chip).toHaveFocus())
    // The popup starts its close animation (jsdom never finishes it).
    await waitFor(() =>
      expect(search.closest(".ant-dropdown")?.className).toMatch(
        /ant-dropdown-hidden|-leave/
      )
    )
  })

  it.each([{ busy: true }])(
    "preserves queued dispatch for %j",
    async (props) => {
      renderHarness(<Harness model="ready-model" {...props} />)
      await waitForCatalog()
      fireEvent.click(screen.getByText("Send"))
      await waitFor(() =>
        expect(mocks.queue).toHaveBeenCalledWith(
          expect.objectContaining({ promptText: "keep my draft" })
        )
      )
      expect(mocks.send).not.toHaveBeenCalled()
    }
  )
  it("keeps disconnected input without placing it on the queue", async () => {
    renderHarness(<Harness model="ready-model" connected={false} />)
    fireEvent.click(screen.getByText("Send"))
    expect(mocks.queue).not.toHaveBeenCalled()
    expect(mocks.send).not.toHaveBeenCalled()
    expect(screen.getByLabelText("Message")).toHaveValue("keep my draft")
  })

  it.each(["failed", "empty", "failed-refetch"])("preserves input and never dispatches from an unverified %s catalog", async mode => {
    if (mode === "failed") mocks.fetchChatModels.mockRejectedValueOnce(new Error("Unavailable"))
    if (mode === "empty") mocks.fetchChatModels.mockResolvedValueOnce([])
    renderHarness(<Harness model="ready-model" />)
    if (mode === "failed") {
      await waitFor(() => expect(queryClient.getQueryCache().find({ queryKey: ["playground:chatModels"], exact: false })?.state.status).toBe("error"))
    } else {
      await waitForCatalog()
      if (mode === "failed-refetch") {
        mocks.fetchChatModels.mockRejectedValueOnce(new Error("Unavailable"))
        await act(async () => { await queryClient.invalidateQueries({ queryKey: ["playground:chatModels"] }) })
        await waitFor(() => expect(queryClient.getQueryCache().find({ queryKey: ["playground:chatModels"], exact: false })?.state.status).toBe("error"))
        await waitFor(() => expect(screen.getByTestId("model-catalog")).toHaveTextContent("pending"))
      }
    }
    fireEvent.click(screen.getByText("Send"))
    await waitFor(() => expect(screen.getByText(/Selected model is not available/)).toBeInTheDocument())
    expect(screen.getByRole("textbox", { name: "Message" })).toHaveValue("keep my draft")
    expect(mocks.send).not.toHaveBeenCalled()
    expect(mocks.queue).not.toHaveBeenCalled()
  })
})

it.each(["compact", "full"].flatMap(surface => ["greeting", "persona"].map(reason => ({ surface, reason }))))("$reason readiness gates ordinary, Run now, and Run next entry points on $surface", async ({ surface, reason }) => {
  const formSource = surface === "compact" ? source : readFileSync(path.resolve(__dirname, "../../../Option/Playground/PlaygroundForm.tsx"), "utf8")
  const tree = ts.createSourceFile("form.tsx", formSource, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX)
  const initializers = new Map<string, string>()
  const walk = (node: ts.Node) => {
    if (ts.isVariableDeclaration(node) && node.initializer) initializers.set(node.name.getText(tree), node.initializer.getText(tree))
    ts.forEachChild(node, walk)
  }
  walk(tree)
  const dispatchNames = ["onSubmit", "handleRunQueuedRequest", "handleRunNextQueuedRequest"]
  dispatchNames.unshift("getSubmitDisabledReason")
  const code = dispatchNames.map(name => {
    const value = initializers.get(name)
    if (!value) throw new Error(`Missing actual dispatch entry: ${name}`)
    return `const ${name} = ${value};`
  }).join("\n") + "\nreturn { onSubmit, handleRunQueuedRequest, handleRunNextQueuedRequest };"
  const transpiled = ts.transpileModule(code, { compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText
  let preparing = reason === "greeting"
  const personaSubmitDisabledReasonRef = { current: reason === "persona" ? "Tracked persona chat is unavailable here" : null }
  const submitMessageAction = vi.fn(async () => ({ status: "submitted" })), run = vi.fn(), next = vi.fn()
  const actions = new Function("React", "greetingSubmitGateRef", "characterSubmitBlockerRef", "personaSubmitDisabledReasonRef", "submitMessageAction", "runQueuedRequest", "runNextQueuedRequest", "queue", "t", transpiled)(
    { useCallback: (callback: unknown) => callback }, { current: () => preparing }, { current: null }, personaSubmitDisabledReasonRef, submitMessageAction, run, next,
    { handleRunQueuedRequest: run, handleRunNextQueuedRequest: next }, t)
  expect(await actions.onSubmit({ message: "UNSENT", image: "" })).toMatchObject({ status: "skipped" })
  actions.handleRunQueuedRequest("queued"); actions.handleRunNextQueuedRequest()
  expect(submitMessageAction).not.toHaveBeenCalled(); expect(run).not.toHaveBeenCalled(); expect(next).not.toHaveBeenCalled()
  preparing = false
  personaSubmitDisabledReasonRef.current = null
  await actions.onSubmit({ message: "UNSENT", image: "" }); actions.handleRunQueuedRequest("queued"); actions.handleRunNextQueuedRequest()
  expect(submitMessageAction).toHaveBeenCalledOnce(); expect(run).toHaveBeenCalledWith("queued"); expect(next).toHaveBeenCalledOnce()


})
