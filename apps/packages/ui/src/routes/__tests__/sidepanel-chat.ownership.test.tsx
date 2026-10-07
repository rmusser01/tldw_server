import { ChatDraftRecovery, type RestoreChatDraft } from "@/components/Chat/composer/ChatDraftRecovery"
import { buildQueuedRequest } from "@/utils/chat-request-queue"
import { useSelectedModel } from "@/hooks/chat/useSelectedModel"
import { useStoreChatModelSettings } from "@/store/model"
import { useStoreMessageOption } from "@/store/option"
import {
  type SidepanelChatSnapshot,
  useSidepanelChatTabsStore
} from "@/store/sidepanel-chat-tabs"
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { readFileSync } from "node:fs"
import path from "node:path"
import ts from "typescript"
import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import LegacySidepanelChat from "../../../../../tldw-frontend/extension/routes/sidepanel-chat"
import SidepanelChat from "../sidepanel-chat"
import { SidepanelRouteShell } from "../sidepanel-route-shell"
import { MemoryRouter } from "react-router-dom"
import { createServicePromptScopeChangedError } from "@/services/tldw/service-prompt-scope-error"
import { useHistorySelectionContext, parseHistorySelectionHandoff, type HistorySelectionController } from "@/hooks/chat/useHistorySelection"
import { requestSettingsNavigation, SETTINGS_NAVIGATION_REQUEST_EVENT } from "@/utils/settings-return"
import type { HistoryBookmarkScope } from "@/db/dexie/types"
import type { HistoryOwnerV1 } from "@/services/chat-history-selection"
import type { HistoryCaptureRequestV1, HistoryViewSelectionV1 } from "@/types/history-selection"
import { chatDraftOwnerKey } from "@/hooks/useChatDraftOwner"
import { useComposerText } from "@/components/Chat/composer/hooks/useComposerText"
import { chatDraftAddressKey, conversationDraftKey } from "@/utils/chat-draft-address"
import { decodeSidepanelChatWebUiHandoff } from "@/services/tldw/sidepanel-chat-webui-handoff"

const io = vi.hoisted(() => ({
  noop: () => {},
  t: (key: string, fallback?: string) =>
    typeof fallback === "string" ? fallback : key,
  user: "alice",
  sequence: 0,
  localId: "foreign",
  serverId: "server-chat",
  captureNative: false,
  useComposer: false,
  useRecovery: false,
  write: vi.fn(),
  physicalTabId: 7 as number | null,
  controller: null as HistorySelectionController | null,
  keyboardNew: null as (() => void) | null,
  data: new Map<string, unknown>(),
  read: vi.fn(),
  local: vi.fn(),
  scope: vi.fn(),
  server: vi.fn()
}))
vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    get: (key: string) => io.read(key),
    set: (key: string, value: unknown) => io.write(key, value),
    getAll: async () => Object.fromEntries(io.data),
    watch: io.noop,
    unwatch: io.noop,
    remove: async (key: string) => {
      io.data.delete(key)
    }
  }),
  safeStorageSerde: {
    deserializer: (value: unknown) =>
      typeof value === "string" ? JSON.parse(value) : value
  }
}))
vi.mock("@/hooks/useMessage", () => ({
  useMessage: () => {
    const state = useStoreMessageOption()
    const historySelection = useHistorySelectionContext()
    const selection = useSelectedModel()
    return new Proxy(state, {
      get: (target, key) =>
        key in selection
          ? Reflect.get(selection, key)
          : key in target
            ? Reflect.get(target, key)
            : key === "clearChat"
              ? (beforeReset?: () => void) => {
                  if (!requestSettingsNavigation("/")) return false
                  beforeReset?.()
                  useStoreMessageOption.getState().startNewChatDraft()
                  historySelection?.reset()
                  useStoreMessageOption.setState({
                    messages: [],
                    history: [],
                    historyId: null,
                    serverChatId: null,
                    queuedMessages: []
                  })
                  return true
                }
              : String(key).startsWith("set") ||
                  ["stopStreamingRequest", "onSubmit"].includes(String(key))
                ? io.noop
                : null
    })
  }
}))
vi.mock("@/services/service-prompts", async original => ({
  ...await original<typeof import("@/services/service-prompts")>(),
  loadServicePromptSnapshot: () => io.scope(),
  resolveServicePromptScope: async () => (await io.scope()).requestScope,
  subscribeToServicePromptConfigChanges: () => io.noop
}))
// Storage/capture I/O only; the mounted logical-view controller and route remain real.
vi.mock("@/db/dexie/chat", async original => {
  const actual = await original<typeof import("@/db/dexie/chat")>()
  return { ...actual, PageAssistDatabase: class extends actual.PageAssistDatabase {
    async getHistoryInfo(id: string) { return (await io.local(id))?.historyInfo }
  } }
})
vi.mock("@/db/dexie/history-selection", async original => ({
  ...await original<typeof import("@/db/dexie/history-selection")>(),
  ensureLocalProfileId: async () => "profile",
  getLocalHistoryOwner: async (id: string) => ({ kind: "local", profile_id: "profile", owner_key: "local-profile", conversation_id: id }),
  loadHistoryBookmark: async (scope: HistoryBookmarkScope, owner: { owner_key: string; conversation_id: string }) => io.data.get(`bookmark:${JSON.stringify([scope.profile_id, scope.client_session_id, owner.owner_key, owner.conversation_id])}`) ?? null,
  saveHistoryBookmark: async (scope: HistoryBookmarkScope, view: HistoryViewSelectionV1) => {
    io.data.set(`bookmark:${JSON.stringify([scope.profile_id, scope.client_session_id, view.owner_key, view.conversation_id])}`, { ...scope, view })
  },
  loadHistoryTurnRecoveries: async () => []
}))
vi.mock("@/services/chat-history-selection", async original => ({
  ...await original<typeof import("@/services/chat-history-selection")>(),
  captureHistorySnapshot: async (owner: HistoryOwnerV1, incoming: HistoryCaptureRequestV1["view"], purpose: string) => {
    if (owner.kind === "unavailable" || (owner.kind === "native" && (!io.captureNative || owner.conversation_id === "failed"))) throw new Error("unsupported_history_capability")
    const cached = owner.kind === "native" ? { historyInfo: { id: owner.conversation_id }, messages: [] } : await io.local(owner.conversation_id)
    if (!cached || cached.historyInfo.id !== owner.conversation_id) throw new Error("missing_conversation")
    const view = { ...incoming, owner_key: owner.owner_key || `native:${owner.conversation_id}` }
    const rows = cached.messages.map((row: { id: string; role: string }) => ({ id: row.id, revision: "1", role: row.role, parent_id: null, settled: true }))
    return { status: "captured", view, purpose, rows,
      selected_content: cached.messages.map((row: { id: string; content: string }) => ({ id: row.id, revision: "1", message: row.content, images: [] })),
      storage_context_digest: "storage",
      snapshot: { version: 1, owner_key: view.owner_key, conversation_id: owner.conversation_id, nodes: rows,
        fences: { conversation: "1", history: "1", settings: "1" }, source_digest: "source", storage_context_digest: "storage", interpretation_status: { kind: "parent_graph_v1" } }
    }
  }
}))
vi.mock("@/hooks/useBackgroundMessage", () => ({ default: () => null }))
vi.mock("@/db/dexie/fork-operations", async original => ({
  ...await original<typeof import("@/db/dexie/fork-operations")>(),
  findForkCandidate: async () => null,
  loadForkOperations: async () => []
}))
vi.mock("@/hooks/useMigration", () => ({ useMigration: () => {} }))
vi.mock("@/hooks/useSmartScroll", () => ({
  useSmartScroll: () => ({
    containerRef: { current: null },
    autoScrollToBottom: io.noop
  })
}))
vi.mock("@/hooks/keyboard/useKeyboardShortcuts", () => ({
  useChatShortcuts: (action: () => void) => { io.keyboardNew = action },
  useSidebarShortcuts: io.noop,
  useChatModeShortcuts: io.noop,
  useWebSearchShortcuts: io.noop
}))
vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionActions: () => ({ checkOnce: io.noop })
}))
vi.mock("@/hooks/useServerOnline", () => ({ useServerOnline: io.noop }))
vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({ warning: io.noop, error: io.noop })
}))
vi.mock("@/hooks/useCharacterGreeting", () => ({
  useCharacterGreeting: io.noop
}))
vi.mock("@/hooks/useTTS", () => ({ useTTS: () => ({ cancel: io.noop }) }))
vi.mock("@/hooks/useSelectedCharacter", () => ({
  useSelectedCharacter: () => [null]
}))
vi.mock("@/hooks/useSelectedAssistant", () => ({
  useSelectedAssistant: () => [null]
}))
vi.mock("@/hooks/useSetting", () => ({ useSetting: () => [100] }))
const storageWrite = vi.hoisted(() => async () => {})
vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: unknown, fallback: unknown) => [
    fallback,
    storageWrite,
    { isLoading: false }
  ]
}))
vi.mock("@/store/ui-mode", () => ({
  useUiModeStore: (select: (state: unknown) => unknown) =>
    select({ mode: "pro" })
}))
vi.mock("@/store/artifacts", () => ({
  useArtifactsStore: (select: (state: unknown) => unknown) =>
    select({ isOpen: false, closeArtifact: io.noop })
}))
vi.mock("@/db/dexie/helpers", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/db/dexie/helpers")>()),
  generateID: () => `new-${++io.sequence}`,
  getTitleById: async () => "",
  getRecentChatFromCopilot: async () => null,
  getFullChatData: (id: string) => io.local(id)
}))
vi.mock("@/services/app", () => ({ copilotResumeLastChat: async () => false }))
vi.mock("@/services/web-clipper/enrichment", () => ({
  readPendingWebClipAnalyzeRequest: () => null,
  clearPendingWebClipAnalyzeRequest: io.noop
}))
vi.mock("@/components/Sidepanel/Chat/body", () => ({
  SidePanelBody: () => {
    io.controller = useHistorySelectionContext()
    return <div>
      {useStoreMessageOption((state) => state.messages).map((item, index) => (
        <p key={index}>{item.message}</p>
      ))}
    </div>
  }
}))
// Read the production option rather than giving the probe stronger isolation
// than the deployed form. The mounted composer/registry remain real.
const formAst = ts.createSourceFile("form.tsx", readFileSync(path.resolve(__dirname,
  "../../components/Sidepanel/Chat/form.tsx"), "utf8"), ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX)
let sidepanelTabScopedDraft = false
let sidepanelRecoverySurface: string | undefined
let sidepanelHasDraftAddress = false
let sidepanelRecoverySurfaceTabs = false
const findComposerOptions = (node: ts.Node) => {
  if (ts.isCallExpression(node) && node.expression.getText(formAst) === "useComposerText") {
    const options = node.arguments[0]
    if (options && ts.isObjectLiteralExpression(options)) {
      sidepanelHasDraftAddress = options.properties.some(property => ts.isPropertyAssignment(property) && property.name.getText(formAst) === "draftAddress" && property.initializer.getText(formAst) === "draftAddress ?? undefined")
      sidepanelRecoverySurfaceTabs = options.properties.some(property => ts.isPropertyAssignment(property) && property.name.getText(formAst) === "recoverySurfaceTabs" && property.initializer.kind === ts.SyntaxKind.TrueKeyword)
      sidepanelTabScopedDraft = options.properties.some(property => ts.isPropertyAssignment(property) &&
        property.name.getText(formAst) === "tabScopedDraft" && property.initializer.kind === ts.SyntaxKind.TrueKeyword)
      const property = options.properties.find((property): property is ts.PropertyAssignment =>
        ts.isPropertyAssignment(property) && property.name.getText(formAst) === "recoverySurface")
      if (property && ts.isStringLiteral(property.initializer)) sidepanelRecoverySurface = property.initializer.text
    }
  }
  ts.forEachChild(node, findComposerOptions)
}
findComposerOptions(formAst)
const tabStorage = (): Storage => {
  const entries = new Map<string, string>()
  return { get length() { return entries.size }, key: index => [...entries.keys()][index] ?? null,
    getItem: key => entries.get(key) ?? null, setItem: (key, value) => { entries.set(key, value) },
    removeItem: key => { entries.delete(key) }, clear: () => entries.clear() }
}
// Reuse the production Form source capture, opener and ready registration.
// The real composer/registry/H1 remain mounted; document IO is a separate suite.
const formOpeningNodes: ts.Node[] = []
const collectOpening = (node: ts.Node) => { formOpeningNodes.push(node); ts.forEachChild(node, collectOpening) }
collectOpening(formAst)
const openingDeclaration = (name: string) => formOpeningNodes.find((node): node is ts.VariableDeclaration => ts.isVariableDeclaration(node) && node.name.getText(formAst) === name)
const openingNames = ["documentWebUiHandoffRef", "getHandoffInputRevision", "handoffInputRevision", "handoffSelectionIsCurrent", "captureHandoffGuard", "openChatInWebUiWithDocumentHandoff"]
const registrations = formOpeningNodes.filter(node => ts.isExpressionStatement(node) && node.getText(formAst).startsWith("React.useLayoutEffect") && node.getText(formAst).includes("onWebUiOpenReady"))
const openingCleanup = formOpeningNodes.find(node => ts.isExpressionStatement(node) && /React\.use(?:Layout)?Effect\(\(\) => \(\) => documentWebUiHandoffRef/.test(node.getText(formAst)))
const openingCode = ts.transpileModule(`function useFormOpening(props, composer) {
  const { form, draftReady, isDraftOwnerCurrent } = composer;
  const { chatDraftAddress: draftAddress, chatDraftRevision: sourceDraftRevision, historyId, serverChatId } = useStoreMessageOption();
  const greetingHistorySelection = useHistorySelectionContext();
  const contextFiles = [];
  const currentDocumentSelectionKey = "no-files";
  const currentDocumentSelectionKeyRef = React.useRef(currentDocumentSelectionKey);
  const onOpenChatInWebUi = props.onOpenChatInWebUi;
  const onWebUiOpenReady = props.onWebUiOpenReady;
  ${openingNames.map(name => `const ${openingDeclaration(name)!.getText(formAst)};`).join("\n")}
  ${openingCleanup?.getText(formAst) ?? ""}
  ${registrations.map(node => node.getText(formAst)).join("\n")}
}`, { compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText
const useFormOpening = new Function("React", "useStoreMessageOption", "useHistorySelectionContext", "notification", "t", "tldwClient", `${openingCode}; return useFormOpening;`)(React, useStoreMessageOption, useHistorySelectionContext, { error: io.noop }, io.t, {})

const ComposerProbe = (props: { draftKey: string; inputRef?: React.RefObject<HTMLTextAreaElement>; onRecoverDraft?: RestoreChatDraft }) => {
  const { draftKey, inputRef, onRecoverDraft } = props
  const address = useStoreMessageOption(state => state.chatDraftAddress)
  React.useLayoutEffect(() => {
    if (!useStoreMessageOption.getState().chatDraftAddress) useStoreMessageOption.getState().startNewChatDraft()
  }, [address])
  const transition = useStoreMessageOption(state => state.chatDraftTransition)
  const composer = useComposerText({
    draftKey: address ? conversationDraftKey(draftKey, address) : draftKey,
    draftAddress: sidepanelHasDraftAddress ? address ?? undefined : undefined,
    recoverySurface: sidepanelRecoverySurface,
    recoverySurfaceTabs: sidepanelRecoverySurfaceTabs,
    draftEnabled: address !== null,
    tabScopedDraft: sidepanelTabScopedDraft,
    bindFromDraftKey: address && transition?.kind === "bind" && transition.sourceAddress &&
      transition.destinationKey === chatDraftAddressKey(address)
      ? conversationDraftKey(draftKey, transition.sourceAddress) : undefined,
    textareaRef: inputRef ?? React.createRef<HTMLTextAreaElement>()
  })
  useFormOpening(props, composer)
  return <>{io.useRecovery && <ChatDraftRecovery listDrafts={composer.listRetainedDrafts} copyDraft={composer.copyRetainedDraft} onRestore={onRecoverDraft} disabled={!composer.draftReady} />}<textarea ref={inputRef} aria-label="Composer draft" disabled={!composer.draftReady}
    value={composer.form.values.message}
    onChange={event => composer.setMessageValue(event.target.value)} /></>
}
vi.mock("@/components/Sidepanel/Chat/form", () => ({
  SidepanelForm: (props: { draftKey: string }) => io.useComposer ? <ComposerProbe {...props} /> : null
}))
vi.mock("@/components/Sidepanel/Chat/SidepanelHeaderSimple", () => ({
  SidepanelHeaderSimple: (props: { chatReady?: boolean; onOpenChatInWebUi?: () => void }) => <button disabled={!props.chatReady}
    onClick={() => props.onOpenChatInWebUi?.()}>Open full chat in WebUI</button>
}))
vi.mock("@/components/Sidepanel/Chat/ConnectionBanner", () => ({
  ConnectionBanner: () => null
}))
vi.mock("@/components/Sidepanel/Chat/Sidebar", () => ({
  SidepanelChatSidebar: (props: {
    tabs: { id: string; label: string }[]
    onSelectTab: (id: string) => void
    onCloseTab: (id: string) => void
    onNewTab: () => void
    onOpenLocalHistory: (id: string) => void
    onOpenServerChat: (chat: { id: string; title: string }) => void
  }) => (
    <nav>
      <button onClick={props.onNewTab}>New conversation</button>
      {props.tabs.map((tab) => (
        <React.Fragment key={tab.id}><button onClick={() => props.onSelectTab(tab.id)}>
          {tab.label}
        </button><button aria-label={`Close ${tab.label}`} onClick={() => props.onCloseTab(tab.id)}>Close</button></React.Fragment>
      ))}
      <button onClick={() => props.onOpenLocalHistory(io.localId)}>
        Open cached history
      </button>
      <button
        onClick={() =>
          props.onOpenServerChat({ id: io.serverId, title: "Server chat" })
        }>
        Open server history
      </button>
    </nav>
  )
}))
vi.mock("@/components/Common/CommandPaletteHost", () => ({
  CommandPaletteHost: ({ commandPaletteProps }: { commandPaletteProps: { onNewChat: () => void } }) =>
    <button onClick={() => commandPaletteProps.onNewChat()}>Palette New chat</button>
}))
vi.mock("@/components/Common/CommandPalette", () => ({
  CommandPalette: () => null
}))
vi.mock("@/components/Timeline", () => ({ TimelineModal: () => null }))
vi.mock("@/components/Sidepanel/Notes/NoteQuickSaveModal", () => ({
  default: () => null
}))
vi.mock("react-i18next", () => ({ useTranslation: () => ({ t: io.t, i18n: { language: "en", resolvedLanguage: "en" } }) }))
vi.mock("~/hooks/useDarkmode", () => ({ useDarkMode: () => ({ mode: "light" }) }))
vi.mock("@/hooks/useAutoButtonTitles", () => ({ useAutoButtonTitles: io.noop }))
vi.mock("@/i18n", () => ({ ensureI18nNamespaces: async () => {} }))
vi.mock("@/hooks/useServerCapabilities", () => ({ useServerCapabilities: () => ({ capabilities: null, loading: false }) }))
vi.mock("@/utils/ui-diagnostics", () => ({ registerUiDiagnostics: io.noop }))
vi.mock("~/components/Layouts/Layout", () => ({ default: ({ children }: { children: React.ReactNode }) => <>{children}</> }))
vi.mock(
  "@plasmohq/storage",
  async () =>
    import("../../../../../tldw-frontend/extension/shims/plasmo-storage")
)
vi.mock("@/services/tldw/deployment-mode", () => ({
  isHostedTldwDeployment: () => false
}))
vi.mock("@/utils/browser-runtime", async original => ({ ...await original<typeof import("@/utils/browser-runtime")>(), isExtensionRuntime: () => false }))
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: async () => {},
    listChatMessages: (...args: unknown[]) => io.server(...args),
    ensureConfigForRequest: async () => ({
      serverUrl: "http://chat.test",
      authMode: "multi-user"
    })
  }
}))
vi.mock("wxt/browser", () => ({
  browser: { runtime: { sendMessage: async () => ({ tabId: io.physicalTabId }) } }
}))

const ownerKey = (user: string) =>
  `["http://chat.test","multi-user","manual",null,"${user}",null]`
const storedKey = (user: string) =>
  `sidepanelChatTabsState:v2:${encodeURIComponent(ownerKey(user))}:tab-7`
const snapshot = (text = "ALICE TRANSCRIPT"): SidepanelChatSnapshot => ({
  history: [{ role: "user", content: text }],
  messages: [
    { role: "user", isBot: false, name: "User", message: text, sources: [] }
  ],
  chatMode: "normal",
  historyId: null,
  webSearch: false,
  toolChoice: "none",
  selectedModel: "llama:one",
  selectedSystemPrompt: null,
  selectedQuickPrompt: null,
  temporaryChat: false,
  useOCR: false,
  serverChatId: null,
  serverChatState: null,
  serverChatTopic: null,
  serverChatClusterId: null,
  serverChatSource: null,
  serverChatExternalRef: null,
  queuedMessages: [],
  modelSettings: { systemPrompt: "ALICE PRIVATE PROMPT" }
})
const tabsState = (user = "alice") => ({
  version: 2,
  ownerKey: ownerKey(user),
  tabs: [
    {
      id: `${user}-one`,
      label: `${user} one`,
      labelSource: "manual",
      historyId: null,
      serverChatId: null,
      serverChatTopic: null,
      updatedAt: 1
    },
    {
      id: `${user}-two`,
      label: `${user} two`,
      labelSource: "manual",
      historyId: null,
      serverChatId: null,
      serverChatTopic: null,
      updatedAt: 1
    }
  ],
  activeTabId: `${user}-one`,
  snapshotsById: {
    [`${user}-one`]: snapshot(`${user} first`),
    [`${user}-two`]: snapshot(`${user} inactive`)
  }
})
const switchAccount = (user: string) =>
  act(() => {
    io.user = user
    window.dispatchEvent(
      new CustomEvent("tldw:config-updated", {
        detail: { authorityChanged: true }
      })
    )
  })

const completedReplyFixture = () => {
  const saved = tabsState()
  const stale = saved.snapshotsById["alice-one"]
  stale.historyId = "owned-history"
  stale.serverChatId = "owned-chat"
  stale.messages = [{ id: "reply-id", isBot: true, role: "assistant", name: "Character",
    message: "1\n2\n▋", sources: [], parentMessageId: "user-id" }]
  stale.queuedMessages = [buildQueuedRequest({ id: "queue-id", clientRequestId: "client-id",
    conversationId: "owned-chat", promptText: "Retained queued request" })]
  const mirror = {
    historyInfo: { id: "owned-history", server_chat_id: "owned-chat", server_scope_key: ownerKey("alice") },
    messages: [{ id: "reply-id", history_id: "owned-history", role: "assistant", name: "Character",
      content: "1\n2\n3\n4", parent_message_id: "user-id", createdAt: 1 }]
  }
  return { saved, stale, mirror }
}

// The real auth watcher also writes the model-cache invalidation tombstone.
// Recovery must not write chat state; retain every other write in this assertion.
const recoveryWrites = () => io.write.mock.calls.filter(([key, value]) => {
  if (key !== "tldwModelsCache") return true
  expect(value).toMatchObject({ models: null, timestamp: 0, scope: null, invalidationToken: expect.any(String) })
  return false
})

describe.each([
  ["shared", SidepanelChat],
  ["legacy", LegacySidepanelChat]
] as const)("%s sidepanel snapshot ownership", (_name, Route) => {
  beforeEach(() => {
    vi.unstubAllGlobals()
    vi.spyOn(window, "open").mockReset().mockImplementation(() => ({}) as Window)
    io.user = "alice"
    io.sequence = 0
    io.localId = "foreign"
    io.serverId = "server-chat"
    io.captureNative = false
    io.useComposer = false
    io.useRecovery = false
    io.write.mockReset().mockImplementation(async (key, value) => { io.data.set(key, value) })
    io.physicalTabId = 7
    sessionStorage.clear()
    io.controller = null
    io.keyboardNew = null
    io.data.clear()
    io.read.mockReset().mockImplementation(async (key) => io.data.get(key))
    io.local.mockReset()
    io.server.mockReset()
    io.scope
      .mockReset()
      .mockImplementation(async () => ({
        requestScope: {
          config: { serverUrl: "http://chat.test", authMode: "multi-user" },
          userId: io.user
        },
        scopeSignal: new AbortController().signal,
        scopeInvalidatedSignal: new AbortController().signal,
        release: io.noop
      }))
    window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed"))
    useSidepanelChatTabsStore.getState().clear()
    useStoreChatModelSettings.getState().reset()
  })

  it("keeps recovery mounted during cold Retry, then mounts the verified chat", async () => {
    const alice = await io.scope()
    io.scope.mockRejectedValueOnce(new Error("offline"))
    render(<Route />)
    const retry = await screen.findByRole("button", { name: "Retry" })
    const panel = screen.getByRole("alert")
    expect(recoveryWrites()).toEqual([])
    expect(screen.queryByRole("textbox", { name: "Composer draft" })).not.toBeInTheDocument()
    let finish!: (value: unknown) => void
    io.scope.mockImplementationOnce(() => new Promise(resolve => { finish = resolve }))
    retry.focus()
    fireEvent.click(retry)
    expect(retry).toBeDisabled()
    expect(screen.getByRole("alert")).toBe(panel)
    expect(screen.getByRole("button", { name: "Retry" })).toBe(retry)
    expect(panel).toHaveTextContent("Checking the chat connection")
    await act(async () => { finish(alice) })
    await screen.findByRole("button", { name: "Open cached history" })
    expect(useSidepanelChatTabsStore.getState().ownerKey).toBe(ownerKey("alice"))
  })

  it("offers another Retry after a repeated cold verification failure", async () => {
    io.scope.mockRejectedValue(new Error("offline"))
    render(<Route />)
    const retry = await screen.findByRole("button", { name: "Retry" })
    fireEvent.click(retry)
    await waitFor(() => expect(retry).not.toBeDisabled())
    expect(io.scope).toHaveBeenCalledTimes(2)
    expect(recoveryWrites()).toEqual([])
    expect(screen.getByRole("alert")).toHaveTextContent("We couldn’t verify the chat connection")
  })

  it("keeps cold recovery when server-settings navigation is declined", async () => {
    io.scope.mockRejectedValue(new Error("offline"))
    render(<Route />)
    const setup = await screen.findByRole("button", { name: "Server settings" })
    const decline = (event: Event) => event.preventDefault()
    window.addEventListener(SETTINGS_NAVIGATION_REQUEST_EVENT, decline)
    try { fireEvent.click(setup) } finally { window.removeEventListener(SETTINGS_NAVIGATION_REQUEST_EVENT, decline) }
    expect(window.open).not.toHaveBeenCalled()
    expect(screen.getByRole("alert")).toBeInTheDocument()
    expect(recoveryWrites()).toEqual([])
  })

  if (_name === "shared") it.each(["extension", "webui"])("opens full server settings from the actual compact shell in %s", async surface => {
    io.scope.mockRejectedValue(new Error("offline"))
    if (surface === "extension") vi.stubGlobal("browser", { runtime: { id: "test-extension", getURL: (route: string) => `chrome-extension://test${route}` } })
    else {
      // The deployed WebUI installs the browser shim, including getURL without an extension id.
      vi.stubGlobal("browser", { runtime: { id: undefined, getURL: (route: string) => `https://webui.test${route}` } })
      vi.stubGlobal("chrome", undefined)
    }
    render(<MemoryRouter initialEntries={["/chat"]}><SidepanelRouteShell /></MemoryRouter>)
    fireEvent.click(await screen.findByRole("button", { name: "Server settings" }))
    expect(window.open).toHaveBeenCalledWith(surface === "extension" ? "chrome-extension://test/options.html#/settings/tldw" : "/settings/tldw", "_blank", "noopener,noreferrer")
    expect(screen.queryByTestId("not-found-recovery-panel")).not.toBeInTheDocument()
    expect(screen.getByRole("alert")).toBeInTheDocument()
    // The real shell persists its header preference, never chat/draft content.
    expect(recoveryWrites()).toEqual([["headerShortcutsExpanded", false]])
  })

  it.each(["account", "unmount"])("ignores a failed cold Retry after %s retirement", async ending => {
    io.scope.mockRejectedValueOnce(new Error("offline"))
    const page = render(<Route />)
    const retry = await screen.findByRole("button", { name: "Retry" })
    let reject!: (error: Error) => void
    io.scope.mockImplementationOnce(() => new Promise((_resolve, rejectRead) => { reject = rejectRead }))
    fireEvent.click(retry)
    if (ending === "account") {
      switchAccount("bob")
      await screen.findByRole("button", { name: "Open cached history" })
    } else page.unmount()
    const writes = io.write.mock.calls.length
    await act(async () => { reject(Object.assign(new Error("Expired Alice credentials"), { code: "service_prompt_scope_unresolved" })) })
    expect(io.write).toHaveBeenCalledTimes(writes)
    if (ending === "account") expect(useSidepanelChatTabsStore.getState().ownerKey).toBe(ownerKey("bob"))
  })

  it.each([
    ["unresolved", Object.assign(new Error("Expired session"), { code: "service_prompt_scope_unresolved" })],
    ["scope changed", createServicePromptScopeChangedError()],
    ["401", Object.assign(new Error("Unauthenticated"), { status: 401 })]
  ])("revokes the mounted chat for positive %s authority loss without deleting saved tabs", async (_reason, error) => {
    io.data.set(storedKey("alice"), tabsState())
    render(<Route />)
    fireEvent.click(await screen.findByRole("button", { name: "alice two" }))
    const durable = io.data.get(storedKey("alice"))
    io.scope.mockRejectedValue(error)
    act(() => window.dispatchEvent(new Event("focus")))
    await screen.findByRole("button", { name: "Retry" })
    expect(useSidepanelChatTabsStore.getState().ownerKey).toBeNull()
    expect(useStoreMessageOption.getState().messages).toEqual([])
    expect(io.data.get(storedKey("alice"))).toEqual(durable)
  })

  it("offers restore Retry without overwriting unread owned tabs", async () => {
    const saved = tabsState()
    io.data.set(storedKey("alice"), saved)
    io.read.mockImplementation(async key => {
      if (key === storedKey("alice")) throw new Error("Disk temporarily unavailable")
      return io.data.get(key)
    })
    render(<Route />)
    const alert = await screen.findByRole("alert")
    expect(alert).toHaveTextContent("Could not restore your saved chat. Retry to load it.")
    expect(screen.queryByLabelText("Restoring previous chat")).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "New conversation" }))
    expect(io.data.get(storedKey("alice"))).toEqual(saved)
    expect(useSidepanelChatTabsStore.getState().tabs).toEqual([])
    io.read.mockImplementation(async key => io.data.get(key))
    fireEvent.click(screen.getByRole("button", { name: "Retry" }))
    await screen.findByRole("button", { name: "alice one" })
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
    expect(useStoreMessageOption.getState().messages[0]?.message).toBe("alice first")
    expect(useSidepanelChatTabsStore.getState().tabs).toHaveLength(2)
  })

  it("retains an owned Temporary transcript and locator while excluding queued sends and derived labels from durable tabs", async () => {
    render(<Route />)
    await waitFor(() =>
      expect(useSidepanelChatTabsStore.getState().activeTabId).toBeTruthy()
    )
    act(() => {
      useStoreMessageOption.getState().setTemporaryChat(true)
      useStoreMessageOption.getState().startNewChatDraft(true)
      useStoreMessageOption.getState().setHistoryId("temp")
      useStoreMessageOption
        .getState()
        .setMessages(snapshot("TEMPORARY PRIVATE TURN").messages)
      useStoreMessageOption
        .getState()
        .setHistory(snapshot("TEMPORARY PRIVATE TURN").history)
      useStoreMessageOption
        .getState()
        .setQueuedMessages([
          buildQueuedRequest({ promptText: "TEMPORARY QUEUED TURN" })
        ])
    })
    const id = useSidepanelChatTabsStore.getState().activeTabId!
    const address = useStoreMessageOption.getState().chatDraftAddress!
    act(() => window.dispatchEvent(new Event("beforeunload")))
    await waitFor(() => {
      const durable = io.data.get(storedKey("alice")) as ReturnType<
        typeof tabsState
      >
      expect(durable.snapshotsById[id]).toMatchObject({
        chatDraftHandle: {
          address,
          ownerHash: chatDraftOwnerKey({
            config: { serverUrl: "http://chat.test", authMode: "multi-user" },
            userId: "alice"
          })
        },
        temporaryChat: true,
        messages: snapshot("TEMPORARY PRIVATE TURN").messages,
        history: snapshot("TEMPORARY PRIVATE TURN").history,
        queuedMessages: [],
        historyId: null,
        serverChatId: null,
        historySelectionReference: null
      })
      expect(JSON.stringify(durable)).toContain("TEMPORARY PRIVATE TURN")
      expect(durable.tabs.find((tab) => tab.id === id)?.label).toBe("New chat")
      expect(JSON.stringify(durable)).not.toContain("TEMPORARY QUEUED TURN")
    })
    expect(
      useSidepanelChatTabsStore.getState().getSnapshot(id)?.messages[0]?.message
    ).toBe("TEMPORARY PRIVATE TURN")
    expect(useStoreMessageOption.getState().messages[0]?.message).toBe(
      "TEMPORARY PRIVATE TURN"
    )
  })

  it.each([false, true])("recovers a departed unsent draft into a new compact tab (Temporary %s)", async temporary => {
    io.useComposer = true; io.useRecovery = true
    const view = render(<Route />)
    await waitFor(() => expect(screen.getByLabelText("Composer draft")).toBeEnabled())
    act(() => useStoreMessageOption.getState().setTemporaryChat(temporary))
    const source = useStoreMessageOption.getState().chatDraftAddress
    const sourceTab = useSidepanelChatTabsStore.getState().activeTabId!
    fireEvent.change(screen.getByLabelText("Composer draft"), { target: { value: "RETAINED BEFORE DEBOUNCE" } })
    fireEvent.click(screen.getByRole("button", { name: "New conversation" }))
    await waitFor(() => expect(screen.getByLabelText("Composer draft")).toHaveValue(""))
    fireEvent.click(screen.getByRole("button", { name: "Recover drafts" }))
    fireEvent.click(await screen.findByRole("button", { name: /RETAINED BEFORE DEBOUNCE/ }))
    await waitFor(() => expect(screen.getByLabelText("Composer draft")).toHaveValue("RETAINED BEFORE DEBOUNCE"))
    expect(useStoreMessageOption.getState().chatDraftAddress).toMatchObject({ kind: "fresh", scope: source!.kind === "fresh" ? source.scope : undefined, temporary })
    expect(useStoreMessageOption.getState().chatDraftAddress?.id).not.toBe(source?.id)
    expect(useStoreMessageOption.getState().temporaryChat).toBe(temporary)
    expect(useStoreMessageOption.getState().historyId).toBeNull()
    expect(useSidepanelChatTabsStore.getState().activeTabId).not.toBe(sourceTab)
    expect(useSidepanelChatTabsStore.getState().tabs.some(tab => tab.id === sourceTab)).toBe(true)
    const recovered = useStoreMessageOption.getState().chatDraftAddress
    const recoveredTab = useSidepanelChatTabsStore.getState().activeTabId
    act(() => window.dispatchEvent(new Event("pagehide")))
    view.unmount(); await act(async () => {})
    sessionStorage.clear()
    useSidepanelChatTabsStore.getState().clear()
    useStoreMessageOption.getState().startNewChatDraft()
    render(<Route />)
    await waitFor(() => expect(screen.getByLabelText("Composer draft")).toHaveValue("RETAINED BEFORE DEBOUNCE"))
    expect(useSidepanelChatTabsStore.getState().activeTabId).toBe(recoveredTab)
    expect(useStoreMessageOption.getState().chatDraftAddress).toEqual(recovered)
    expect(useStoreMessageOption.getState().temporaryChat).toBe(temporary)
  })


  it("allows a later local destination preparation to supersede a held recovery copy", async () => {
    io.useComposer = true; io.useRecovery = true
    render(<Route />)
    await waitFor(() => expect(screen.getByLabelText("Composer draft")).toBeEnabled())
    fireEvent.change(screen.getByLabelText("Composer draft"), { target: { value: "OLDER RECOVERY TEXT" } })
    fireEvent.click(screen.getByRole("button", { name: "New conversation" }))
    await waitFor(() => expect(screen.getByLabelText("Composer draft")).toHaveValue(""))
    const sourceTab = useSidepanelChatTabsStore.getState().activeTabId
    const source = useStoreMessageOption.getState().chatDraftAddress
    let finishCopy!: () => void
    io.write.mockImplementation(async (key, value) => {
      if (key.startsWith("registry:draft:") && value.value?.content === "OLDER RECOVERY TEXT")
        await new Promise<void>(resolve => { finishCopy = resolve })
      io.data.set(key, value)
    })
    fireEvent.click(screen.getByRole("button", { name: "Recover drafts" }))
    fireEvent.click(await screen.findByRole("button", { name: /OLDER RECOVERY TEXT/ }))
    await waitFor(() => expect(finishCopy).toBeTypeOf("function"))
    io.localId = "B"
    let finishLoad!: (value: unknown) => void
    const load = new Promise(resolve => { finishLoad = resolve })
    io.local.mockReturnValue(load)
    fireEvent.click(screen.getByRole("button", { name: "Open cached history" }))
    await waitFor(() => expect(finishLoad).toBeTypeOf("function"))
    await act(async () => { finishCopy() })
    expect(useSidepanelChatTabsStore.getState().activeTabId).toBe(sourceTab)
    expect(useStoreMessageOption.getState().chatDraftAddress).toBe(source)
    await act(async () => { finishLoad({ historyInfo: { id: "B", title: "B", server_scope_key: ownerKey("alice") },
      messages: [{ id: "B-message", role: "user", content: "LATER B TRANSCRIPT" }] }) })
    await screen.findByText("LATER B TRANSCRIPT")
    expect(useStoreMessageOption.getState().chatDraftAddress).toMatchObject({ kind: "local", id: "B" })
  })
  it.each(["copy failure", "declined navigation", "New during copy"])("retains the active compact draft after %s recovery", async boundary => {
    io.useComposer = true; io.useRecovery = true
    render(<Route />)
    await waitFor(() => expect(screen.getByLabelText("Composer draft")).toBeEnabled())
    fireEvent.change(screen.getByLabelText("Composer draft"), { target: { value: "OLD RETAINED TEXT" } })
    fireEvent.click(screen.getByRole("button", { name: "New conversation" }))
    await waitFor(() => expect(screen.getByLabelText("Composer draft")).toHaveValue(""))
    fireEvent.change(screen.getByLabelText("Composer draft"), { target: { value: "CURRENT PRIVATE TEXT" } })
    fireEvent.click(screen.getByRole("button", { name: "Recover drafts" }))
    const choice = await screen.findByRole("button", { name: /OLD RETAINED TEXT/ })
    const sourceTab = useSidepanelChatTabsStore.getState().activeTabId
    const source = useStoreMessageOption.getState().chatDraftAddress
    let finish!: () => void
    if (boundary === "copy failure") io.write.mockImplementation(async (key, value) => {
      if (key.startsWith("registry:draft:")) throw new Error("quota")
      io.data.set(key, value)
    })
    if (boundary === "declined navigation") window.addEventListener(SETTINGS_NAVIGATION_REQUEST_EVENT, event => event.preventDefault(), { once: true })
    if (boundary === "New during copy") io.write.mockImplementation(async (key, value) => {
      if (key.startsWith("registry:draft:") && value.value?.content === "OLD RETAINED TEXT")
        await new Promise<void>(resolve => { finish = resolve })
      io.data.set(key, value)
    })
    fireEvent.click(choice)
    if (boundary === "New during copy") {
      await waitFor(() => expect(finish).toBeTypeOf("function"))
      fireEvent.click(screen.getByRole("button", { name: "New conversation" }))
      await waitFor(() => expect(screen.getByLabelText("Composer draft")).toHaveValue(""))
      fireEvent.change(screen.getByLabelText("Composer draft"), { target: { value: "NEW ACCEPTED TEXT" } })
      const accepted = useStoreMessageOption.getState().chatDraftAddress
      await act(async () => { finish() })
      expect(useStoreMessageOption.getState().chatDraftAddress).toBe(accepted)
      expect(screen.getByLabelText("Composer draft")).toHaveValue("NEW ACCEPTED TEXT")
    } else {
      await screen.findByText("Could not open this draft. Your current text and saved draft are kept.")
      expect(useSidepanelChatTabsStore.getState().activeTabId).toBe(sourceTab)
      expect(useStoreMessageOption.getState().chatDraftAddress).toBe(source)
      expect(screen.getByLabelText("Composer draft")).toHaveValue("CURRENT PRIVATE TEXT")
    }
  })

  it.each([false, true])("cold restores a fresh handle before stale H1, transcript and recent fallback (Temporary %s)", async temporary => {
    const saved = tabsState()
    const address = { kind: "fresh" as const, id: "retained-unsent", scope: { type: "global" as const }, temporary }
    Object.assign(saved.snapshotsById["alice-one"], {
      temporaryChat: temporary,
      chatDraftHandle: { address, ownerHash: chatDraftOwnerKey({ config: { serverUrl: "http://chat.test", authMode: "multi-user" }, userId: "alice" }) },
      historyId: "stale-local", serverChatId: "stale-native",
      historySelectionReference: { profile_id: "profile", client_session_id: "old-tab", conversation_id: "stale-native", owner_key: "old-owner", owner_kind: "native" },
      queuedMessages: temporary ? [buildQueuedRequest({ promptText: "OLD TEMP QUEUE" })] : []
    })
    saved.snapshotsById["alice-one"].messages[0].message = "STALE PRIVATE TRANSCRIPT▋"
    io.data.set(storedKey("alice"), saved)
    render(<Route />)
    await screen.findByRole("button", { name: "alice two" })
    await waitFor(() => expect(useStoreMessageOption.getState()).toMatchObject({ chatDraftAddress: address, temporaryChat: temporary, messages: [], history: [], historyId: null, serverChatId: null, queuedMessages: [] }))
    expect(io.local).not.toHaveBeenCalled()
    expect(io.server).not.toHaveBeenCalled()
    expect(io.controller?.getCurrent().owner).toBeNull()
  })

  it("reloads a New tab and its first unsent edit before debounce without further core changes", async () => {
    io.useComposer = true
    const view = render(<Route />)
    await screen.findByRole("button", { name: "New conversation" })
    await waitFor(() => expect(screen.getByLabelText("Composer draft")).toBeEnabled())
    fireEvent.click(screen.getByRole("button", { name: "New conversation" }))
    const id = useSidepanelChatTabsStore.getState().activeTabId!
    const address = useStoreMessageOption.getState().chatDraftAddress!
    await waitFor(() => expect(screen.getByLabelText("Composer draft")).toBeEnabled())
    fireEvent.change(screen.getByLabelText("Composer draft"), { target: { value: "FIRST UNSENT EDIT" } })
    act(() => window.dispatchEvent(new Event("pagehide")))
    view.unmount()
    await act(async () => {})
    const saved = io.data.get(storedKey("alice")) as ReturnType<typeof tabsState>
    expect(saved.activeTabId).toBe(id)
    expect(saved.snapshotsById[id].chatDraftHandle?.address).toEqual(address)
    useSidepanelChatTabsStore.getState().clear()
    useStoreMessageOption.getState().startNewChatDraft()
    render(<Route />)
    await waitFor(() => expect(screen.getByLabelText("Composer draft")).toHaveValue("FIRST UNSENT EDIT"))
    expect(useStoreMessageOption.getState().chatDraftAddress).toEqual(address)
  })

  it("keeps same-fallback logical tab drafts independent across physical session storage areas", async () => {
    useStoreMessageOption.getState().startNewChatDraft(false)
    const firstArea = tabStorage(), secondArea = tabStorage()
    const key = "tldw:sidepanelChatDraft:shared-fallback-logical-id"
    vi.stubGlobal("sessionStorage", firstArea)
    let first = render(<ComposerProbe draftKey={key} />)
    await waitFor(() => expect(within(first.container).getByLabelText("Composer draft")).toBeEnabled())
    fireEvent.change(within(first.container).getByLabelText("Composer draft"), { target: { value: "FIRST PANEL TEXT" } })
    act(() => window.dispatchEvent(new Event("pagehide")))
    vi.stubGlobal("sessionStorage", secondArea)
    let second = render(<ComposerProbe draftKey={key} />)
    await waitFor(() => expect(within(second.container).getByLabelText("Composer draft")).toBeEnabled())
    fireEvent.change(within(second.container).getByLabelText("Composer draft"), { target: { value: "SECOND PANEL TEXT" } })
    act(() => window.dispatchEvent(new Event("pagehide")))
    first.unmount(); second.unmount()
    vi.stubGlobal("sessionStorage", firstArea)
    first = render(<ComposerProbe draftKey={key} />)
    await waitFor(() => expect(within(first.container).getByLabelText("Composer draft")).toHaveValue("FIRST PANEL TEXT"))
    fireEvent.change(within(first.container).getByLabelText("Composer draft"), { target: { value: "" } })
    act(() => window.dispatchEvent(new Event("pagehide")))
    vi.stubGlobal("sessionStorage", secondArea)
    second = render(<ComposerProbe draftKey={key} />)
    await waitFor(() => expect(within(second.container).getByLabelText("Composer draft")).toHaveValue("SECOND PANEL TEXT"))
    fireEvent.change(within(second.container).getByLabelText("Composer draft"), { target: { value: "SECOND PANEL NEWER TEXT" } })
    act(() => window.dispatchEvent(new Event("pagehide")))
    first.unmount(); second.unmount()
    vi.stubGlobal("sessionStorage", firstArea)
    first = render(<ComposerProbe draftKey={key} />)
    await waitFor(() => expect(within(first.container).getByLabelText("Composer draft")).toBeEnabled())
    expect(within(first.container).getByLabelText("Composer draft")).toHaveValue("")
    first.unmount()
  })

  it.each(["history", "messages"])(
    "cold restores a safe fresh view from malformed owned Temporary %s",
    async (field) => {
      const saved = tabsState()
      const address = {
        kind: "fresh" as const,
        id: "malformed-temp",
        scope: { type: "global" as const },
        temporary: true
      }
      saved.activeTabId = "alice-two"
      Object.assign(saved.snapshotsById["alice-two"], {
        chatDraftHandle: {
          address,
          ownerHash: chatDraftOwnerKey({
            config: { serverUrl: "http://chat.test", authMode: "multi-user" },
            userId: "alice"
          })
        },
        temporaryChat: true,
        historyId: "temp",
        [field]: [null]
      })
      io.data.set(storedKey("alice"), saved)
      render(<Route />)
      await waitFor(() =>
        expect(useStoreMessageOption.getState().chatDraftAddress).toEqual(address)
      )
      expect(useStoreMessageOption.getState()).toMatchObject({
        temporaryChat: true,
        history: [],
        messages: [],
        historyId: null,
        serverChatId: null
      })
      expect(screen.queryByRole("alert")).not.toBeInTheDocument()
    }
  )

  it("retains owned Temporary transcript and next unsent text on warm return and cold reload", async () => {
    io.useComposer = true
    let view = render(<Route />)
    await waitFor(() =>
      expect(screen.getByLabelText("Composer draft")).toBeEnabled()
    )
    act(() => {
      useStoreMessageOption.getState().setTemporaryChat(true)
      useStoreMessageOption.getState().startNewChatDraft(true)
      useStoreMessageOption.getState().setHistoryId("temp")
      useStoreMessageOption
        .getState()
        .setHistory(snapshot("TEMP PRIVATE REPLY").history)
      useStoreMessageOption
        .getState()
        .setMessages(snapshot("TEMP PRIVATE REPLY").messages)
      useStoreMessageOption
        .getState()
        .setQueuedMessages([
          buildQueuedRequest({ promptText: "TEMP PRIVATE QUEUE" })
        ])
    })
    const address = useStoreMessageOption.getState().chatDraftAddress!
    await waitFor(() =>
      expect(screen.getByLabelText("Composer draft")).toBeEnabled()
    )
    fireEvent.change(screen.getByLabelText("Composer draft"), {
      target: { value: "NEXT TEMP UNSENT" }
    })
    view.unmount()
    await act(async () => {})
    expect(JSON.stringify(io.data.get(storedKey("alice")))).toContain(
      "TEMP PRIVATE REPLY"
    )
    expect(JSON.stringify(io.data.get(storedKey("alice")))).not.toContain(
      "TEMP PRIVATE QUEUE"
    )
    view = render(<Route />)
    await screen.findByText("TEMP PRIVATE REPLY")
    await waitFor(() =>
      expect(screen.getByLabelText("Composer draft")).toHaveValue(
        "NEXT TEMP UNSENT"
      )
    )
    expect(useStoreMessageOption.getState().chatDraftAddress).toEqual(address)
    expect(useStoreMessageOption.getState().historyId).toBe("temp")
    view.unmount()
    await act(async () => {})
    useSidepanelChatTabsStore.getState().clear()
    useStoreMessageOption.getState().startNewChatDraft(false)
    render(<Route />)
    await waitFor(() =>
      expect(screen.getByLabelText("Composer draft")).toHaveValue(
        "NEXT TEMP UNSENT"
      )
    )
    expect(useStoreMessageOption.getState()).toMatchObject({
      chatDraftAddress: address,
      temporaryChat: true,
      history: snapshot("TEMP PRIVATE REPLY").history,
      messages: snapshot("TEMP PRIVATE REPLY").messages,
      queuedMessages: [],
      historyId: null,
      serverChatId: null
    })
    expect(
      within(screen.getByRole("log")).getByText("TEMP PRIVATE REPLY")
    ).toBeInTheDocument()
  })

  it.each(["local", "native"])("loads the exact qualified %s handle and its retained composer text", async kind => {
    io.useComposer = true
    io.captureNative = true
    const saved = tabsState()
    const address = kind === "local"
      ? { kind: "local" as const, id: "saved-B", profileId: "profile" }
      : { kind: "native" as const, id: "saved-B", ownerKey: "native:saved-B", scope: { type: "workspace" as const, workspaceId: "workspace-B" } }
    const ownerHash = chatDraftOwnerKey({ config: { serverUrl: "http://chat.test", authMode: "multi-user" }, userId: "alice" })
    Object.assign(saved.snapshotsById["alice-two"], {
      chatDraftHandle: { address, ownerHash }, historyId: kind === "local" ? "saved-B" : "obsolete-mirror",
      serverChatId: kind === "native" ? "saved-B" : null
    })
    io.data.set(storedKey("alice"), saved)
    io.data.set(`registry:draft:${conversationDraftKey("tldw:sidepanelChatDraft:alice-two", address)}:owner:${ownerHash}`,
      { value: { content: "SAVED B UNSENT" }, updatedAt: Date.now() })
    io.local.mockResolvedValue({ historyInfo: { id: "saved-B", local_owner_key: "local-profile" }, messages: [] })
    render(<Route />)
    await screen.findByText("alice first")
    fireEvent.click(await screen.findByRole("button", { name: "alice two" }))
    await waitFor(() => expect(screen.getByLabelText("Composer draft")).toHaveValue("SAVED B UNSENT"))
    expect(useStoreMessageOption.getState().chatDraftAddress).toEqual(address)
    expect(useSidepanelChatTabsStore.getState().activeTabId).toBe("alice-two")
    await waitFor(() => expect(io.controller?.getCurrent().status).toBe("ready"))
    expect(io.controller?.getCurrent().owner?.kind).toBe(kind)
    if (kind === "native") expect(io.controller?.getCurrent().owner).toMatchObject({ scope: address.scope })
  })

  it("keeps setup edits made on A while saved B is being qualified", async () => {
    const saved = tabsState()
    const ownerHash = chatDraftOwnerKey({ config: { serverUrl: "http://chat.test", authMode: "multi-user" }, userId: "alice" })
    Object.assign(saved.snapshotsById["alice-two"], {
      historyId: "saved-B", chatDraftHandle: { ownerHash, address: { kind: "local", id: "saved-B", profileId: "profile" } }
    })
    io.data.set(storedKey("alice"), saved)
    const cached = { historyInfo: { id: "saved-B", local_owner_key: "local-profile" }, messages: [] }
    let finish!: (value: typeof cached) => void
    io.local.mockImplementationOnce(() => new Promise(resolve => { finish = resolve })).mockResolvedValue(cached)
    render(<Route />)
    fireEvent.click(await screen.findByRole("button", { name: "alice two" }))
    await waitFor(() => expect(finish).toBeTypeOf("function"))
    act(() => {
      useStoreMessageOption.getState().setWebSearch(true)
      useStoreMessageOption.getState().setSelectedModel("changed-source-model")
      useStoreMessageOption.getState().setSelectedSystemPrompt("changed-source-prompt")
    })
    await waitFor(() => expect(useSidepanelChatTabsStore.getState().getSnapshot("alice-one")?.webSearch).toBe(true))
    await act(async () => { finish(cached) })
    await waitFor(() => expect(useSidepanelChatTabsStore.getState().activeTabId).toBe("alice-two"))
    expect(useSidepanelChatTabsStore.getState().getSnapshot("alice-one")).toMatchObject({
      webSearch: true, selectedModel: "changed-source-model", selectedSystemPrompt: "changed-source-prompt" })
    fireEvent.click(screen.getByRole("button", { name: "alice one" }))
    await waitFor(() => expect(useSidepanelChatTabsStore.getState().activeTabId).toBe("alice-one"))
    expect(useStoreMessageOption.getState()).toMatchObject({ webSearch: true,
      selectedModel: "changed-source-model", selectedSystemPrompt: "changed-source-prompt" })
  })

  it.each(["New", "another tab"])("recovers from an invalid active saved handle using %s without replacing retained records", async action => {
    const saved = tabsState()
    const ownerHash = chatDraftOwnerKey({ config: { serverUrl: "http://chat.test", authMode: "multi-user" }, userId: "alice" })
    Object.assign(saved.snapshotsById["alice-one"], {
      historyId: "stale-A", chatDraftHandle: { ownerHash, address: { kind: "local", id: "stale-A", profileId: "old-profile" } }
    })
    Object.assign(saved.snapshotsById["alice-two"], {
      historyId: "valid-B", chatDraftHandle: { ownerHash, address: { kind: "local", id: "valid-B", profileId: "profile" } }
    })
    io.local.mockImplementation(async id => ({ historyInfo: { id, local_owner_key: "local-profile" }, messages: [] }))
    io.data.set(storedKey("alice"), saved)
    render(<Route />)
    const alert = await screen.findByRole("alert")
    expect(alert).toHaveTextContent("Your saved chats and drafts are kept.")
    expect(io.data.get(storedKey("alice"))).toEqual(saved)
    if (action === "New") fireEvent.click(screen.getByRole("button", { name: "Start a new conversation" }))
    else fireEvent.click(await screen.findByRole("button", { name: "alice two" }))
    await waitFor(() => expect(screen.queryByRole("alert")).not.toBeInTheDocument())
    expect(useSidepanelChatTabsStore.getState().tabs.some(tab => tab.id === "alice-one")).toBe(true)
    expect(useSidepanelChatTabsStore.getState().getSnapshot("alice-one")?.chatDraftHandle).toEqual(saved.snapshotsById["alice-one"].chatDraftHandle)
    if (action === "New") expect(useStoreMessageOption.getState().chatDraftAddress?.kind).toBe("fresh")
    else expect(useStoreMessageOption.getState().chatDraftAddress).toEqual(saved.snapshotsById["alice-two"].chatDraftHandle?.address)
  })

  it("keeps the ready native source lease while preparing B and cancels B when A is selected again", async () => {
    io.captureNative = true
    const saved = tabsState()
    const ownerHash = chatDraftOwnerKey({ config: { serverUrl: "http://chat.test", authMode: "multi-user" }, userId: "alice" })
    Object.assign(saved.snapshotsById["alice-one"], {
      serverChatId: "source-A", chatDraftHandle: { ownerHash,
        address: { kind: "native", id: "source-A", ownerKey: "native:source-A", scope: { type: "global" } } }
    })
    Object.assign(saved.snapshotsById["alice-two"], {
      historyId: "saved-B", chatDraftHandle: { ownerHash, address: { kind: "local", id: "saved-B", profileId: "profile" } }
    })
    io.data.set(storedKey("alice"), saved)
    let finish!: (value: unknown) => void
    io.local.mockImplementation(() => new Promise(resolve => { finish = resolve }))
    render(<Route />)
    await screen.findByRole("button", { name: "alice two" })
    await waitFor(() => expect(io.controller?.getCurrent().status).toBe("ready"))
    const sourceOwner = io.controller!.getCurrent().owner!
    expect(sourceOwner.validate_lease?.()).toBe(true)
    fireEvent.click(screen.getByRole("button", { name: "alice two" }))
    await waitFor(() => expect(finish).toBeTypeOf("function"))
    expect(sourceOwner.validate_lease?.()).toBe(true)
    fireEvent.click(screen.getByRole("button", { name: "alice one" }))
    await act(async () => { finish({ historyInfo: { id: "saved-B", local_owner_key: "local-profile" }, messages: [] }) })
    expect(useSidepanelChatTabsStore.getState().activeTabId).toBe("alice-one")
    expect(io.controller!.getCurrent().owner).toBe(sourceOwner)
    expect(sourceOwner.validate_lease?.()).toBe(true)
  })

  it.each([0, null])("does not carry another physical panel's RAM transcript into empty context %s", async physicalTabId => {
    io.useComposer = true
    io.data.set(storedKey("alice"), tabsState())
    const first = render(<Route />)
    await screen.findByText("alice first")
    await waitFor(() => expect(screen.getByLabelText("Composer draft")).toBeEnabled())
    const address = useStoreMessageOption.getState().chatDraftAddress
    fireEvent.change(screen.getByLabelText("Composer draft"), { target: { value: "PANEL SEVEN UNSENT" } })
    first.unmount()
    io.physicalTabId = physicalTabId
    render(<Route />)
    await waitFor(() => expect(screen.getByLabelText("Composer draft")).toBeEnabled())
    expect(useStoreMessageOption.getState().messages).toEqual([])
    expect(useStoreMessageOption.getState().chatDraftAddress).not.toEqual(address)
    expect(screen.getByLabelText("Composer draft")).toHaveValue("")
    const key = `sidepanelChatTabsState:v2:${encodeURIComponent(ownerKey("alice"))}:${physicalTabId === null ? "global" : "tab-0"}`
    expect(useSidepanelChatTabsStore.getState().storageKey).toBe(key)
    expect(io.read).toHaveBeenCalledWith(key)
  })

  it.each(["profile", "replacement", "closed"])("keeps the source tab and composer when a saved handle has a %s mismatch", async reason => {
    io.useComposer = true
    const saved = tabsState()
    const target = saved.snapshotsById["alice-two"]
    Object.assign(target, {
      historyId: "saved-B",
      chatDraftHandle: { address: { kind: "local", id: "saved-B", profileId: reason === "profile" ? "another-profile" : "profile" },
        ownerHash: chatDraftOwnerKey({ config: { serverUrl: "http://chat.test", authMode: "multi-user" }, userId: "alice" }) }
    })
    io.data.set(storedKey("alice"), saved)
    let finish!: (value: unknown) => void
    io.local.mockImplementation(() => new Promise(resolve => { finish = resolve }))
    render(<Route />)
    await screen.findByText("alice first")
    await waitFor(() => expect(screen.getByLabelText("Composer draft")).toBeEnabled())
    const source = useStoreMessageOption.getState().chatDraftAddress
    fireEvent.change(screen.getByLabelText("Composer draft"), { target: { value: "A UNSENT TEXT" } })
    fireEvent.click(screen.getByRole("button", { name: "alice two" }))
    await waitFor(() => expect(finish).toBeTypeOf("function"))
    expect(useSidepanelChatTabsStore.getState().activeTabId).toBe("alice-one")
    expect(useStoreMessageOption.getState().chatDraftAddress).toBe(source)
    expect(screen.getByLabelText("Composer draft")).toHaveValue("A UNSENT TEXT")
    if (reason === "replacement") {
      fireEvent.click(screen.getByRole("button", { name: "New conversation" }))
      await waitFor(() => expect(screen.getByLabelText("Composer draft")).toBeEnabled())
      fireEvent.change(screen.getByLabelText("Composer draft"), { target: { value: "NEW UNSENT TEXT" } })
    }
    if (reason === "closed") fireEvent.click(screen.getByRole("button", { name: "Close alice two" }))
    const accepted = useStoreMessageOption.getState().chatDraftAddress
    const active = useSidepanelChatTabsStore.getState().activeTabId
    await act(async () => { finish({ historyInfo: { id: "saved-B", local_owner_key: "local-profile" }, messages: [] }) })
    expect(useSidepanelChatTabsStore.getState().activeTabId).toBe(active)
    expect(useStoreMessageOption.getState().chatDraftAddress).toBe(accepted)
    expect(screen.getByLabelText("Composer draft")).toHaveValue(reason === "replacement" ? "NEW UNSENT TEXT" : "A UNSENT TEXT")
  })

  it("preserves the configured WebUI route, setup and unsent draft through the shared legacy entry", async () => {
    io.useComposer = true
    io.data.set("tldwConfig", { serverUrl: "http://chat.test", webUiUrl: "https://webui.test/base" })
    render(<Route />)
    await waitFor(() => expect(screen.getByLabelText("Composer draft")).toBeEnabled())
    fireEvent.change(screen.getByLabelText("Composer draft"), { target: { value: "CARRY THIS UNSENT TEXT" } })
    act(() => useStoreMessageOption.getState().setWebSearch(true))
    fireEvent.click(screen.getByRole("button", { name: "Open full chat in WebUI" }))
    await waitFor(() => expect(window.open).toHaveBeenCalled())
    const url = new URL(String(vi.mocked(window.open).mock.calls.at(-1)![0]))
    expect(url.origin + url.pathname).toBe("https://webui.test/base/chat")
    const encoded = new URLSearchParams(url.hash.slice(1)).get("handoff")!
    const raw = JSON.parse(Buffer.from(encoded, "base64url").toString("utf8"))
    expect(raw).not.toHaveProperty("state")
    expect(raw).not.toHaveProperty("isCurrent")
    const payload = decodeSidepanelChatWebUiHandoff(encoded)
    expect(payload).toMatchObject({ draft: "CARRY THIS UNSENT TEXT", webSearch: true,
      historyId: null, serverChatId: null, temporaryChat: false })
    expect(screen.getByLabelText("Composer draft")).toHaveValue("CARRY THIS UNSENT TEXT")
  })

  it.each(["keyboard", "palette"])("uses the same restoration boundary for %s New", async entry => {
    const saved = tabsState()
    let finish!: (value: unknown) => void
    io.read.mockImplementation(key => key === storedKey("alice")
      ? new Promise(resolve => { finish = resolve }) : Promise.resolve(io.data.get(key)))
    render(<Route />)
    await waitFor(() => expect(finish).toBeTypeOf("function"))
    const source = useStoreMessageOption.getState().chatDraftAddress
    const newChat = () => { if (entry === "keyboard") io.keyboardNew?.()
      else fireEvent.click(screen.getByRole("button", { name: "Palette New chat" })) }
    act(newChat)
    expect(useStoreMessageOption.getState().chatDraftAddress).toBe(source)
    await act(async () => { finish(saved) })
    await screen.findByText("alice first")
    act(newChat)
    await waitFor(() => expect(useSidepanelChatTabsStore.getState().activeTabId).not.toBe("alice-one"))
    expect(useStoreMessageOption.getState().chatDraftAddress?.kind).toBe("fresh")
    expect(useSidepanelChatTabsStore.getState().tabs.some(tab => tab.id === "alice-one")).toBe(true)
  })

  it("expands the selected history review through its qualified bookmark in the web compact view", async () => {
    io.localId = "owned"
    io.local.mockResolvedValue({ historyInfo: { id: "owned", title: "Owned review", server_scope_key: ownerKey("alice") },
      messages: [{ id: "row", role: "user", content: "OWNED REVIEW TEXT" }] })
    render(<Route />)
    fireEvent.click(await screen.findByRole("button", { name: "Open cached history" }))
    await screen.findByText("OWNED REVIEW TEXT")
    fireEvent.click(screen.getByRole("button", { name: "Expand in full page" }))
    await waitFor(() => expect(window.open).toHaveBeenCalled())
    const route = String(vi.mocked(window.open).mock.calls.at(-1)![0])
    expect(parseHistorySelectionHandoff(route)).toMatchObject({ owner_kind: "local", conversation_id: "owned" })
    expect(route).not.toContain("handoff=")
  })

  it.each(["pending", "target failure"])("does not expand unresolved %s restoration from the header", async state => {
    const saved = tabsState()
    if (state === "target failure") {
      const ownerHash = chatDraftOwnerKey({ config: { serverUrl: "http://chat.test", authMode: "multi-user" }, userId: "alice" })
      Object.assign(saved.snapshotsById["alice-one"], { historyId: "old-A",
        chatDraftHandle: { ownerHash, address: { kind: "local", id: "old-A", profileId: "old-profile" } } })
    }
    io.data.set(storedKey("alice"), saved)
    io.data.set("tldwConfig", { webUiUrl: "https://webui.test" })
    if (state === "pending") io.read.mockImplementation(key => key === storedKey("alice")
      ? new Promise(() => {}) : Promise.resolve(io.data.get(key)))
    render(<Route />)
    if (state === "target failure") await screen.findByRole("alert")
    else await waitFor(() => expect(io.read).toHaveBeenCalledWith(storedKey("alice")))
    await act(async () => { fireEvent.click(screen.getByRole("button", { name: "Open full chat in WebUI" })) })
    expect(window.open).not.toHaveBeenCalled()
  })

  it("ignores a configured WebUI handoff config read after accepted New", async () => {
    io.useComposer = true
    let finish!: (value: unknown) => void
    io.read.mockImplementation(key => key === "tldwConfig"
      ? new Promise(resolve => { finish = resolve }) : Promise.resolve(io.data.get(key)))
    render(<Route />)
    await waitFor(() => expect(screen.getByLabelText("Composer draft")).toBeEnabled())
    fireEvent.click(screen.getByRole("button", { name: "Open full chat in WebUI" }))
    await waitFor(() => expect(finish).toBeTypeOf("function"))
    fireEvent.click(screen.getByRole("button", { name: "New conversation" }))
    await act(async () => { finish({ webUiUrl: "https://old-webui.test" }) })
    expect(window.open).not.toHaveBeenCalled()
  })

  it.each(["input", "edit undo", "H1"])("retires WebUI config waiting when the actual composer changes %s", async boundary => {
    io.useComposer = true
    let finish!: (value: unknown) => void
    io.read.mockImplementation(key => key === "tldwConfig" ? new Promise(resolve => { finish = resolve }) : Promise.resolve(io.data.get(key)))
    render(<Route />)
    await waitFor(() => expect(screen.getByLabelText("Composer draft")).toBeEnabled())
    fireEvent.change(screen.getByLabelText("Composer draft"), { target: { value: "EXACT SOURCE  " } })
    fireEvent.click(screen.getByRole("button", { name: "Open full chat in WebUI" }))
    await waitFor(() => expect(finish).toBeTypeOf("function"))
    if (boundary === "H1") act(() => { io.controller!.beginLoad() })
    else {
      fireEvent.change(screen.getByLabelText("Composer draft"), { target: { value: "NEW EDIT  " } })
      if (boundary === "edit undo") fireEvent.change(screen.getByLabelText("Composer draft"), { target: { value: "EXACT SOURCE  " } })
    }
    await act(async () => { finish({ webUiUrl: "https://old-webui.test" }) })
    expect(window.open).not.toHaveBeenCalled()
  })

  it.each(["account", "unmount"])("ignores a restore rejection after %s replacement", async boundary => {
    let rejectRead!: (error: Error) => void
    io.read.mockImplementation(key => key === storedKey("alice")
      ? new Promise((_resolve, reject) => { rejectRead = reject })
      : Promise.resolve(io.data.get(key)))
    const view = render(<Route />)
    await waitFor(() => expect(rejectRead).toBeTypeOf("function"))
    if (boundary === "account") switchAccount("bob")
    else view.unmount()
    await act(async () => { rejectRead(new Error("Late Alice read failure")) })
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
  })

  it("recovers a completed owned reply from its stale streaming snapshot", async () => {
    const { saved, stale, mirror } = completedReplyFixture()
    io.data.set(storedKey("alice"), saved)
    io.local.mockResolvedValue(mirror)
    const restoreStarted = Date.now()
    render(<Route />)
    await waitFor(() => expect(useStoreMessageOption.getState().messages[0]?.message).toBe("1\n2\n3\n4"))
    expect(useStoreMessageOption.getState().messages[0]?.id).toBe("reply-id")
    const restoredQueue = useStoreMessageOption.getState().queuedMessages
    expect(restoredQueue).toEqual(stale.queuedMessages.map(request => ({ ...request,
      status: "blocked", blockedReason: "restored_request", updatedAt: expect.any(Number) })))
    for (const request of restoredQueue) {
      expect(request.updatedAt).toBeGreaterThanOrEqual(restoreStarted)
      expect(request.updatedAt).toBeLessThanOrEqual(Date.now())
    }
    expect(useStoreMessageOption.getState().history).toEqual(stale.history)
    expect(useStoreChatModelSettings.getState().systemPrompt).toBe("ALICE PRIVATE PROMPT")
  })

  it.each(["foreign", "unowned", "history", "conversation", "row-history", "id", "parent", "prefix", "unfinished", "missing", "acknowledged", "offline"])("retains the original snapshot for %s local recovery", async reason => {
    const { saved, stale, mirror } = completedReplyFixture()
    if (reason === "foreign") mirror.historyInfo.server_scope_key = ownerKey("bob")
    if (reason === "unowned") mirror.historyInfo.server_scope_key = ""
    if (reason === "history") mirror.historyInfo.id = "another-history"
    if (reason === "conversation") mirror.historyInfo.server_chat_id = "another-chat"
    if (reason === "row-history") mirror.messages[0].history_id = "another-history"
    if (reason === "id") mirror.messages[0].id = "another-reply"
    if (reason === "parent") mirror.messages[0].parent_message_id = "another-user"
    if (reason === "prefix") mirror.messages[0].content = "Changed saved content"
    if (reason === "unfinished") mirror.messages[0].content = "1\n2\n3\n▋"
    if (reason === "missing") mirror.messages = []
    if (reason === "acknowledged") stale.messages[0].serverMessageId = "already-saved"
    io.data.set(storedKey("alice"), saved)
    if (reason === "offline") io.local.mockRejectedValue(new Error("Mirror unavailable"))
    else io.local.mockResolvedValue(mirror)
    const restoreStarted = Date.now()
    render(<Route />)
    await screen.findByRole("button", { name: "alice one" })
    await act(async () => {})
    expect(useStoreMessageOption.getState().messages).toEqual(stale.messages)
    const restoredQueue = useStoreMessageOption.getState().queuedMessages
    expect(restoredQueue).toEqual(stale.queuedMessages.map(request => ({ ...request,
      status: "blocked", blockedReason: "restored_request", updatedAt: expect.any(Number) })))
    for (const request of restoredQueue) {
      expect(request.updatedAt).toBeGreaterThanOrEqual(restoreStarted)
      expect(request.updatedAt).toBeLessThanOrEqual(Date.now())
    }
  })

  it.each(["account", "unmount"])("rejects a completed local reply arriving after %s", async boundary => {
    const { saved, mirror } = completedReplyFixture()
    io.data.set(storedKey("alice"), saved)
    let finish!: (value: typeof mirror) => void
    io.local.mockImplementation(() => new Promise(resolve => { finish = resolve }))
    const mounted = render(<Route />)
    await waitFor(() => expect(finish).toBeTypeOf("function"))
    if (boundary === "account") switchAccount("bob")
    else mounted.unmount()
    const before = useStoreMessageOption.getState().messages
    await act(async () => { finish(mirror) })
    expect(useStoreMessageOption.getState().messages).toEqual(before)
    expect(useStoreMessageOption.getState().messages.some(message => message.message === "1\n2\n3\n4")).toBe(false)
  })

  it("does not publish an unowned legacy snapshot or overwrite it", async () => {
    const legacy = tabsState()
    io.data.set("sidepanelChatTabsState:tab-7", legacy)
    render(<Route />)
    await screen.findByRole("button", { name: "Open cached history" })
    await act(async () => {})
    expect(useStoreMessageOption.getState().messages).toEqual([])
    expect(useStoreChatModelSettings.getState().systemPrompt).not.toBe(
      "ALICE PRIVATE PROMPT"
    )
    expect(io.data.get("sidepanelChatTabsState:tab-7")).toBe(legacy)
  })

  it("rejects foreign local history before publishing its content", async () => {
    io.local.mockResolvedValue({
      historyInfo: {
        id: "foreign",
        title: "Foreign private",
        server_scope_key: ownerKey("bob")
      },
      messages: [
        {
          id: "foreign-message",
          role: "user",
          content: "FOREIGN LOCAL TRANSCRIPT"
        }
      ]
    })
    render(<Route />)
    fireEvent.click(
      await screen.findByRole("button", { name: "Open cached history" })
    )
    await act(async () => {})
    expect(useStoreMessageOption.getState().messages).toEqual([])
  })

  it("restores and selects same-owner tabs, clears them for Bob and recovers Alice on return", async () => {
    io.data.set(storedKey("alice"), tabsState())
    render(<Route />)
    fireEvent.click(await screen.findByRole("button", { name: "alice two" }))
    await waitFor(() =>
      expect(useStoreMessageOption.getState().messages[0]?.message).toBe(
        "alice inactive"
      )
    )
    expect(useStoreChatModelSettings.getState().systemPrompt).toBe(
      "ALICE PRIVATE PROMPT"
    )
    switchAccount("bob")
    await waitFor(() =>
      expect(
        useSidepanelChatTabsStore
          .getState()
          .tabs.some((tab) => tab.id.startsWith("alice"))
      ).toBe(false)
    )
    expect(useStoreMessageOption.getState().messages).toEqual([])
    expect(useStoreChatModelSettings.getState().systemPrompt).not.toBe(
      "ALICE PRIVATE PROMPT"
    )
    switchAccount("alice")
    await screen.findByRole("button", { name: "alice two" })
    expect(useStoreMessageOption.getState().messages[0]?.message).toBe(
      "alice inactive"
    )
  })

  it("restores owned tabs during StrictMode effect replay", async () => {
    io.data.set(storedKey("alice"), tabsState())
    render(
      <React.StrictMode>
        <Route />
      </React.StrictMode>
    )
    await screen.findByText("alice first")
    expect(useStoreChatModelSettings.getState().systemPrompt).toBe(
      "ALICE PRIVATE PROMPT"
    )
    expect(useSidepanelChatTabsStore.getState().tabs).toHaveLength(2)
  })

  it("does not restart durable restore when the real model-selection setter changes", async () => {
    const saved = tabsState()
    io.read.mockImplementation(async (key) =>
      key === storedKey("alice") ? saved : undefined
    )
    render(<Route />)
    await screen.findByText("alice first")
    act(() => {
      useStoreMessageOption
        .getState()
        .setMessages(snapshot("LIVE UNSAVED CONTENT").messages)
      useStoreMessageOption.getState().setSelectedModel("llama:two")
    })
    await act(async () => {})
    expect(useStoreMessageOption.getState().selectedModel).toBe("llama:two")
    expect(useStoreMessageOption.getState().messages[0]?.message).toBe(
      "LIVE UNSAVED CONTENT"
    )
  })

  it("finishes a held owned restore when the real model-selection setter changes", async () => {
    let finish!: (value: unknown) => void
    io.read.mockImplementation((key) =>
      key === storedKey("alice")
        ? new Promise((resolve) => {
            finish = resolve
          })
        : Promise.resolve(undefined)
    )
    render(<Route />)
    await waitFor(() => expect(finish).toBeTypeOf("function"))
    act(() => useStoreMessageOption.getState().setSelectedModel("llama:two"))
    await act(async () => {
      finish(tabsState())
    })
    await screen.findByText("alice first")
  })

  it("restores the last selected owned tab after a fresh mount", async () => {
    io.data.set(storedKey("alice"), tabsState())
    const view = render(<Route />)
    fireEvent.click(await screen.findByRole("button", { name: "alice two" }))
    await screen.findByText("alice inactive")
    view.unmount()
    useSidepanelChatTabsStore.getState().clear()
    useStoreMessageOption.setState({
      history: [],
      messages: [],
      historyId: null
    })
    useStoreChatModelSettings.getState().reset()
    render(<Route />)
    await screen.findByText("alice inactive")
    expect(useSidepanelChatTabsStore.getState().activeTabId).toBe("alice-two")
    expect(useStoreChatModelSettings.getState().systemPrompt).toBe(
      "ALICE PRIVATE PROMPT"
    )
  })

  it.each(["local", "server"])(
    "ignores a held %s response after selecting another owned tab",
    async (source) => {
      io.data.set(storedKey("alice"), tabsState())
      let finish!: (value: unknown) => void
      const pending = new Promise((resolve) => {
        finish = resolve
      })
      if (source === "local") io.local.mockReturnValueOnce(pending)
      else io.server.mockReturnValueOnce(pending)
      render(<Route />)
      await screen.findByText("alice first")
      fireEvent.click(
        screen.getByRole("button", {
          name:
            source === "local" ? "Open cached history" : "Open server history"
        })
      )
      fireEvent.click(screen.getByRole("button", { name: "alice two" }))
      await act(async () => {
        finish(
          source === "local"
            ? {
                historyInfo: {
                  id: "alice-old",
                  title: "Alice old",
                  server_scope_key: ownerKey("alice")
                },
                messages: [{ id: "old", role: "user", content: "ALICE LATE" }]
              }
            : [{ id: "old", role: "user", content: "ALICE LATE" }]
        )
      })
      expect(useStoreMessageOption.getState().messages[0]?.message).toBe(
        "alice inactive"
      )
      expect(useStoreMessageOption.getState().isLoading).toBe(false)
    }
  )

  it.each(["account", "unmount"])(
    "ignores a held owned storage read after %s",
    async (ending) => {
      let finish!: (value: unknown) => void
      io.read.mockImplementation((key) =>
        key === storedKey("alice")
          ? new Promise((resolve) => {
              finish = resolve
            })
          : Promise.resolve(io.data.get(key))
      )
      const view = render(<Route />)
      await waitFor(() => expect(finish).toBeTypeOf("function"))
      if (ending === "account") {
        switchAccount("bob")
        await waitFor(() =>
          expect(useSidepanelChatTabsStore.getState().ownerKey).toBe(
            ownerKey("bob")
          )
        )
      } else view.unmount()
      await act(async () => {
        finish(tabsState())
      })
      expect(useStoreMessageOption.getState().messages).toEqual([])
      expect(
        useSidepanelChatTabsStore
          .getState()
          .tabs.some((tab) => tab.id.startsWith("alice"))
      ).toBe(false)
      expect(useStoreChatModelSettings.getState().systemPrompt).not.toBe(
        "ALICE PRIVATE PROMPT"
      )
    }
  )

  it("rejects a late Alice identity result after switching to Bob during preflight", async () => {
    const alice = await io.scope()
    let finish!: (value: unknown) => void
    io.scope.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          finish = resolve
        })
    )
    io.data.set(storedKey("alice"), tabsState())
    render(<Route />)
    await waitFor(() => expect(finish).toBeTypeOf("function"))
    switchAccount("bob")
    await screen.findByRole("button", { name: "Open cached history" })
    await act(async () => {
      finish(alice)
    })
    expect(useSidepanelChatTabsStore.getState().ownerKey).toBe(ownerKey("bob"))
    expect(useStoreMessageOption.getState().messages).toEqual([])
    expect(io.data.get(storedKey("alice"))).toMatchObject({
      activeTabId: "alice-one"
    })
  })

  it.each(["identity", "storage"])(
    "preserves saved owner data when %s is temporarily unavailable",
    async (failure) => {
      const saved = tabsState()
      io.data.set(storedKey("alice"), saved)
      if (failure === "identity")
        io.scope.mockRejectedValueOnce(new Error("temporarily unavailable"))
      else io.read.mockRejectedValueOnce(new Error("temporarily unavailable"))
      render(<Route />)
      await act(async () => {})
      act(() => window.dispatchEvent(new Event("beforeunload")))
      expect(io.data.get(storedKey("alice"))).toBe(saved)
      expect(useStoreMessageOption.getState().messages).toEqual([])
    }
  )

  it("retains the active tab on same-owner rotation and a failed focus recheck", async () => {
    io.data.set(storedKey("alice"), tabsState())
    render(<Route />)
    fireEvent.click(await screen.findByRole("button", { name: "alice two" }))
    io.scope.mockRejectedValueOnce(new Error("offline"))
    act(() => {
      window.dispatchEvent(
        new CustomEvent("tldw:config-updated", {
          detail: { authorityChanged: false }
        })
      )
      window.dispatchEvent(new Event("focus"))
    })
    await act(async () => {})
    expect(useStoreMessageOption.getState().messages[0]?.message).toBe(
      "alice inactive"
    )
    expect(useSidepanelChatTabsStore.getState().activeTabId).toBe("alice-two")
  })

  it.each(["local", "server"])(
    "rejects a held %s history response after account replacement",
    async (source) => {
      let finish!: (value: unknown) => void
      const response = new Promise((resolve) => {
        finish = resolve
      })
      if (source === "local") io.local.mockReturnValueOnce(response)
      else io.server.mockReturnValueOnce(response)
      render(<Route />)
      fireEvent.click(
        await screen.findByRole("button", {
          name:
            source === "local" ? "Open cached history" : "Open server history"
        })
      )
      switchAccount("bob")
      await waitFor(() =>
        expect(useSidepanelChatTabsStore.getState().ownerKey).toBe(
          ownerKey("bob")
        )
      )
      await act(async () => {
        finish(
          source === "local"
            ? {
                historyInfo: {
                  id: "alice-old",
                  title: "Alice old",
                  server_scope_key: ownerKey("alice")
                },
                messages: [{ id: "old", role: "user", content: "ALICE LATE" }]
              }
            : [{ id: "old", role: "user", content: "ALICE LATE" }]
        )
      })
      expect(useStoreMessageOption.getState().messages).toEqual([])
      expect(
        useSidepanelChatTabsStore
          .getState()
          .tabs.some(
            (tab) =>
              tab.historyId === "alice-old" ||
              tab.serverChatId === "server-chat"
          )
      ).toBe(false)
    }
  )

  it("restores an owned local history and its title", async () => {
    io.localId = "owned"
    io.local.mockResolvedValue({
      historyInfo: {
        id: "owned",
        title: "Owned cached title",
        server_scope_key: ownerKey("alice")
      },
      messages: [
        {
          id: "owned-message",
          role: "user",
          content: "OWNED CACHED TRANSCRIPT"
        }
      ]
    })
    render(<Route />)
    fireEvent.click(
      await screen.findByRole("button", { name: "Open cached history" })
    )
    await screen.findByText("OWNED CACHED TRANSCRIPT")
    expect(useStoreMessageOption.getState().historyId).toBe("owned")
  })

  if (_name === "shared") it("loads a destination in its own logical view without overwriting the source tab", async () => {
    io.data.set(storedKey("alice"), tabsState())
    io.localId = "B"
    const data = (id: string) => ({ historyInfo: { id, title: id, server_scope_key: ownerKey("alice") },
      messages: [{ id: `${id}-message`, role: "user", content: `${id} TRANSCRIPT` }] })
    let release!: () => void
    const held = new Promise<void>(resolve => { release = resolve })
    let opening = false
    io.local.mockImplementation(async id => { if (id === "B" && opening) await held; return data(id) })
    render(<Route />)
    await screen.findByRole("button", { name: "alice one" })
    await act(async () => {
      await io.controller!.loadConversation({ historyId: "A" }, null)
      useStoreMessageOption.getState().setHistoryId("A")
      useStoreMessageOption.getState().setChatDraftAddress({ kind: "local", id: "A", profileId: "profile" })
    })
    await screen.findByText("A TRANSCRIPT")
    opening = true
    fireEvent.click(screen.getByRole("button", { name: "Open cached history" }))
    await waitFor(() => expect(io.local).toHaveBeenCalledWith("B"))
    expect(io.controller!.getCurrent().view?.conversation_id).toBe("A")
    expect(io.controller!.getCurrent().status).toBe("ready")
    expect(useSidepanelChatTabsStore.getState().activeTabId).toBe("alice-one")
    expect(useStoreMessageOption.getState().messages[0]?.message).toBe("A TRANSCRIPT")
    act(() => useStoreChatModelSettings.getState().setSystemPrompt("EDITED A SETUP"))
    await waitFor(() => expect(useSidepanelChatTabsStore.getState().snapshotsById["alice-one"].modelSettings?.systemPrompt).toBe("EDITED A SETUP"))
    await act(async () => { release(); await held })
    await screen.findByText("B TRANSCRIPT")
    fireEvent.click(screen.getByRole("button", { name: "alice one" }))
    await screen.findByText("A TRANSCRIPT")
    expect(useStoreChatModelSettings.getState().systemPrompt).toBe("EDITED A SETUP")
    expect(useStoreMessageOption.getState().chatDraftAddress).toMatchObject({ kind: "local", id: "A" })
    await waitFor(() => expect(io.controller!.getCurrent().view?.conversation_id).toBe("A"))
  })

  if (_name === "shared") it("keeps the live source owner when a second opening fails after native preparation", async () => {
    io.data.set(storedKey("alice"), tabsState())
    io.captureNative = true
    io.local.mockImplementation(async id => ({ historyInfo: { id, title: id, server_scope_key: ownerKey("alice") },
      messages: [{ id: `${id}-message`, role: "user", content: `${id} TRANSCRIPT` }] }))
    let release!: () => void
    const held = new Promise<void>(resolve => { release = resolve })
    io.server.mockImplementation(async () => { await held; return [] })
    render(<Route />)
    await screen.findByRole("button", { name: "alice one" })
    await act(async () => {
      await io.controller!.loadConversation({ historyId: "A" }, null)
      useStoreMessageOption.getState().setHistoryId("A")
      useStoreMessageOption.getState().setChatDraftAddress({ kind: "local", id: "A", profileId: "profile" })
    })
    await screen.findByText("A TRANSCRIPT")
    const source = io.controller!.getCurrent()
    const sourceCurrent = io.controller!.fence()
    fireEvent.click(screen.getByRole("button", { name: "Open server history" }))
    // The route lists messages only after B's native H1 capture is ready.
    await waitFor(() => expect(io.server).toHaveBeenCalledWith("server-chat", expect.anything(), expect.anything()))
    expect(io.controller!.getCurrent()).toBe(source)
    expect(sourceCurrent()).toBe(true)
    io.serverId = "failed"
    fireEvent.click(screen.getByRole("button", { name: "Open server history" }))
    await waitFor(() => expect(useStoreMessageOption.getState().isLoading).toBe(false))
    await act(async () => { release(); await held })
    expect(io.controller!.getCurrent()).toBe(source)
    expect(sourceCurrent()).toBe(true)
    expect(useSidepanelChatTabsStore.getState().activeTabId).toBe("alice-one")
    expect(useStoreMessageOption.getState().chatDraftAddress).toMatchObject({ kind: "local", id: "A" })
  })

  if (_name === "shared") it("resets the new logical tab after preserving the departing selection", async () => {
    io.data.set(storedKey("alice"), tabsState())
    io.local.mockResolvedValue({ historyInfo: { id: "A", title: "A" },
      messages: [{ id: "A-message", role: "user", content: "A TRANSCRIPT" }] })
    render(<Route />)
    await screen.findByRole("button", { name: "alice one" })
    await act(async () => {
      await io.controller!.open({ kind: "local", profile_id: "profile", owner_key: "local-profile", conversation_id: "A" }, null)
      useStoreMessageOption.getState().setHistoryId("A")
      useStoreMessageOption.getState().setChatDraftAddress({ kind: "local", id: "A", profileId: "profile" })
    })
    await screen.findByText("A TRANSCRIPT")
    fireEvent.click(screen.getByRole("button", { name: "New conversation" }))
    expect(useStoreMessageOption.getState().chatDraftAddress?.kind).toBe("fresh")
    expect(io.controller!.getCurrent().status).toBe("idle")
    act(() => { io.controller!.activate("alice-one") })
    expect(io.controller!.getCurrent().view?.conversation_id).toBe("A")
    expect(io.controller!.getCurrent().view?.cursor).toEqual({ kind: "after_message", message_id: "A-message" })
  })

  if (_name === "shared") it.each([1, 2])("keeps the active tab when closing %s tab(s) requires declined navigation", async count => {
    const saved = tabsState()
    saved.tabs = saved.tabs.slice(0, count)
    io.data.set(storedKey("alice"), saved)
    render(<Route />)
    await screen.findByRole("button", { name: "alice one" })
    const source = useSidepanelChatTabsStore.getState().getSnapshot("alice-one")
    const decline = (event: Event) => event.preventDefault()
    window.addEventListener(SETTINGS_NAVIGATION_REQUEST_EVENT, decline, { once: true })
    fireEvent.click(screen.getByRole("button", { name: "Close alice one" }))
    expect(useSidepanelChatTabsStore.getState().tabs).toHaveLength(count)
    expect(useSidepanelChatTabsStore.getState().activeTabId).toBe("alice-one")
    expect(useSidepanelChatTabsStore.getState().getSnapshot("alice-one")).toEqual(source)
  })

  it("does not persist an invalidated render into the replacement owner's durable key", async () => {
    io.data.set(storedKey("alice"), tabsState())
    render(<Route />)
    await screen.findByRole("button", { name: "alice two" })
    act(() => {
      useStoreMessageOption
        .getState()
        .setMessages(snapshot("ALICE PENDING CLOSURE").messages)
      io.user = "bob"
      window.dispatchEvent(
        new CustomEvent("tldw:config-updated", {
          detail: { authorityChanged: true }
        })
      )
      window.dispatchEvent(new Event("beforeunload"))
    })
    await screen.findByRole("button", { name: "Open cached history" })
    await act(async () => {})
    expect(JSON.stringify(io.data.get(storedKey("bob")))).not.toContain("ALICE")
    expect(JSON.stringify(io.data.get(storedKey("alice")))).not.toContain(
      "ALICE PENDING CLOSURE"
    )
  })

  it.each(["local", "server", "new"])(
    "keeps the initial owned restore intact when %s is requested during a held read",
    async (action) => {
      let finish!: (value: unknown) => void
      io.read.mockImplementation((key) =>
        key === storedKey("alice")
          ? new Promise((resolve) => {
              finish = resolve
            })
          : Promise.resolve(undefined)
      )
      render(<Route />)
      await waitFor(() => expect(finish).toBeTypeOf("function"))
      fireEvent.click(
        screen.getByRole("button", {
          name:
            action === "local"
              ? "Open cached history"
              : action === "server"
                ? "Open server history"
                : "New conversation"
        })
      )
      await act(async () => {
        finish(tabsState())
      })
      await screen.findByText("alice first")
      expect(io.local).not.toHaveBeenCalled()
      expect(io.server).not.toHaveBeenCalled()
    }
  )

  it("clears live Alice state when focus discovers Bob without a local auth event", async () => {
    io.data.set(storedKey("alice"), tabsState())
    render(<Route />)
    await screen.findByText("alice first")
    act(() => {
      io.user = "bob"
      window.dispatchEvent(new Event("focus"))
    })
    await waitFor(() =>
      expect(useSidepanelChatTabsStore.getState().ownerKey).toBe(
        ownerKey("bob")
      )
    )
    await act(async () => {})
    expect(useStoreMessageOption.getState().messages).toEqual([])
    expect(useStoreChatModelSettings.getState().systemPrompt).not.toBe(
      "ALICE PRIVATE PROMPT"
    )
    expect(JSON.stringify(io.data.get(storedKey("bob")))).not.toContain(
      "alice first"
    )
    expect(JSON.stringify(io.data.get(storedKey("bob")))).not.toContain(
      "ALICE PRIVATE PROMPT"
    )
  })
})
