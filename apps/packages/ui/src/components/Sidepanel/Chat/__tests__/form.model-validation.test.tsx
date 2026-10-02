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
  queue: vi.fn()
}))
const models = [{ model: "ready-model", provider: "llamacpp" }]
const t = (key: string, fallback?: string) => fallback ?? key
vi.mock("@/hooks/useMessage", () => ({ useMessage: mocks.useMessage }))
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
        .includes("fetchChatModels({ returnEmpty: true })")
    )
)
if (!catalogQuery) throw new Error("Missing sidebar model catalog query")
const catalogDataName = (catalogQuery.name as ts.ObjectBindingPattern).elements
  .find((element) => element.propertyName?.getText(ast) === "data")
  ?.name.getText(ast)
if (!catalogDataName) throw new Error("Missing sidebar model catalog data")
const compiled = ts.transpileModule(
  `
  return function useSidebarValidation(scope) {
    const { form, selectedModel, isSending, isConnectionReady, isProMode } = scope;
    const audioHealthEnabled = false;
    const fetchChatModels = mocks.fetchChatModels;
    const ${catalogQuery.getText(ast)};
    const availableChatModelIds = ${initializer("availableChatModelIds")};
    const contextFiles = [], selectedDocuments = [];
    const streaming = false, sendWhenEnter = true;
    const resolveSubmissionIntent = (combinedMessage) => ({ combinedMessage, isImageCommand: false });
    const notification = { error: () => {} };
    const queueSubmission = mocks.queue;
    const stopListening = async () => {};
    const stopServerDictation = () => {};
    const sendCurrentFormMessageRef = { current: mocks.send };
    ${effects}
    const submit = ${initializer("submitCurrentRequest")};
    const inline = ${initializer("composerInlineMessagesNode")};
    return { submit, inline, catalogReady: ${catalogDataName} !== undefined };
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
  "mocks",
  "buildAvailableChatModelIds",
  "findUnavailableChatModel",
  "normalizeChatModelId",
  compiled
)(
  React,
  ModelSelect,
  X,
  t,
  useQuery,
  mocks,
  buildAvailableChatModelIds,
  findUnavailableChatModel,
  normalizeChatModelId
) as (scope: {
  form: ReturnType<typeof useSimpleForm<{ message: string; image: string }>>
  selectedModel: string
  isSending: boolean
  isConnectionReady: boolean
  isProMode: boolean
}) => {
  submit: (message: string, image: string) => Promise<void>
  inline: React.ReactNode
  catalogReady: boolean
}

function Harness({
  model = "",
  busy = false,
  connected = true,
  pro = false,
  text = "keep my draft",
  image = ""
}) {
  const form = useSimpleForm({ initialValues: { message: text, image } })
  const [selectedModel, setSelectedModel] = React.useState(model)
  mocks.useMessage.mockReturnValue({ selectedModel, setSelectedModel })
  const { submit, inline, catalogReady } = useSidebarValidation({
    form,
    selectedModel,
    isSending: busy,
    isConnectionReady: connected,
    isProMode: pro
  })
  return (
    <>
      <textarea aria-label="Message" {...form.getInputProps("message")} />
      <button
        onClick={() => void submit(form.values.message, form.values.image)}
      >
        Send
      </button>
      <output data-testid="model-catalog">
        {catalogReady ? "ready" : "pending"}
      </output>
      {inline}
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
      expect(screen.getByRole("textbox")).toHaveValue("keep my draft")
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
      fireEvent.click(screen.getByTestId("chat-model-select"))
      fireEvent.click(
        await screen.findByRole("menuitem", { name: /ready-model/ })
      )
      await waitFor(() =>
        expect(screen.queryByRole("alert")).not.toBeInTheDocument()
      )
      expect(screen.getByRole("textbox")).toHaveValue("keep my draft")
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
    fireEvent.change(screen.getByRole("textbox"), {
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
    expect(screen.getByRole("textbox")).toHaveValue("keep my draft")
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

  it("sends with an available model and keeps the Casual picker out of the way", async () => {
    renderHarness(<Harness model="ready-model" />)
    fireEvent.click(screen.getByText("Send"))
    await waitFor(() =>
      expect(mocks.send).toHaveBeenCalledWith("keep my draft", "", undefined)
    )
    expect(screen.queryByTestId("chat-model-select")).not.toBeInTheDocument()
  })

  it.each([{ busy: true }, { connected: false }])(
    "preserves queued dispatch for %j",
    async (props) => {
      renderHarness(<Harness model="ready-model" {...props} />)
      fireEvent.click(screen.getByText("Send"))
      await waitFor(() =>
        expect(mocks.queue).toHaveBeenCalledWith(
          expect.objectContaining({ promptText: "keep my draft" })
        )
      )
      expect(mocks.send).not.toHaveBeenCalled()
    }
  )
})
