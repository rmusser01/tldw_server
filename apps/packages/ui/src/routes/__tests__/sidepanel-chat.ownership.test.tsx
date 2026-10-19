import {
  ChatDraftRecovery,
  type RestoreChatDraft
} from "@/components/Chat/composer/ChatDraftRecovery"
import {
  normalizePendingWebClipAnalyzeRequest,
  readPendingWebClipAnalyzeRequest,
  writePendingWebClipAnalyzeRequest
} from "@/services/web-clipper/enrichment"
import { WEB_CLIPPER_ANALYZE_MESSAGE_TYPE } from "@/services/web-clipper/analyze-handoff"
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
import * as historyController from "@/hooks/chat/useHistorySelection"
import * as historyRecoveryStorage from "@/db/dexie/history-selection"
import { resolveHistorySelection } from "@/utils/history-selection"
import type { HistoryTurnRecoveryEntry } from "@/services/history-turn-keep"
import { resetHistoryTurnRegistry } from "@/services/history-turn-keep"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
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
  uiMode: "pro" as "pro" | "casual",
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
  realChatShortcuts: false,
  stubMirror: false,
  realSidebar: false,
  productionTextareaReadiness: false,
  holdDrawerPresentation: false,
  drawerCallbacks: [] as { onClose?: (...args: unknown[]) => void; afterOpenChange?: (open: boolean) => void }[],
  sidebarHold: null as Promise<void> | null,
  sidebarError: null as Error | null,
  closeTab: null as ((id: string, options?: { discardTemporaryConfirmed: true; isCurrent?: () => boolean }) => Promise<boolean | void>) | null,
  data: new Map<string, unknown>(),
  read: vi.fn(),
  local: vi.fn(),
  scope: vi.fn(),
  server: vi.fn(),
  submit: vi.fn(),
  notifyError: vi.fn(),
  capture: vi.fn(),
  settle: vi.fn()
}))
// Keep real Drawer/content and expose retained motion callbacks for boundary controls.
vi.mock("antd", async original => {
  const actual = await original<typeof import("antd")>()
  return {
    ...actual,
    Drawer: (props: React.ComponentProps<typeof actual.Drawer>) => {
      io.drawerCallbacks.push(props)
      return (
        <actual.ConfigProvider theme={{ token: { motion: false } }}>
          <actual.Drawer {...props} afterOpenChange={io.holdDrawerPresentation ? undefined : props.afterOpenChange} />
        </actual.ConfigProvider>
      )
    }
  }
})
vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    hasPersistentBackend: true,
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
            : key === "onSubmit"
              ? io.submit
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
    async getRecentChatHistories() { return [] }
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
// Only the declared native mirror IO is replaced in the overlay handoff controls.
vi.mock("@/db/dexie/server-chat-mirror", async original => {
  const actual = await original<typeof import("@/db/dexie/server-chat-mirror")>()
  return { ...actual,
    linkServerChatMirror: (...args: Parameters<typeof actual.linkServerChatMirror>) => io.stubMirror ? Promise.resolve(`local-${args[0].chatId}`) : actual.linkServerChatMirror(...args),
    reconcileServerChatMirror: (...args: Parameters<typeof actual.reconcileServerChatMirror>) => io.stubMirror ? Promise.resolve() : actual.reconcileServerChatMirror(...args)
  }
})
vi.mock("@/hooks/useMigration", () => ({ useMigration: () => {} }))
vi.mock("@/hooks/useSmartScroll", () => ({
  useSmartScroll: () => ({
    containerRef: { current: null },
    autoScrollToBottom: io.noop
  })
}))
vi.mock("@/hooks/keyboard/useKeyboardShortcuts", async (original) => {
  const actual =
    await original<typeof import("@/hooks/keyboard/useKeyboardShortcuts")>()
  return {
    useChatShortcuts: (action: () => void, enabled: boolean) => {
      io.keyboardNew = action
      if (io.realChatShortcuts) actual.useChatShortcuts(action, enabled)
    },
    useSidebarShortcuts: actual.useSidebarShortcuts,
    useChatModeShortcuts: io.noop,
    useWebSearchShortcuts: io.noop
  }
})
vi.mock("@/hooks/useServerChatHistory", () => ({ useServerChatHistory: () => ({ data: [], isSuccess: true, isPending: false, isFetching: false, isError: false, fetchStatus: "idle", refetch: io.noop }) }))
vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionActions: () => ({ checkOnce: io.noop }),
  useConnectionState: () => ({ isConnected: true })
}))
vi.mock("@/hooks/useServerOnline", () => ({ useServerOnline: io.noop }))
vi.mock("@/hooks/useAntdNotification", () => {
  const notification = { warning: io.noop, error: io.notifyError }
  return { useAntdNotification: () => notification }
})
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
    select({ mode: io.uiMode })
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
const composerReadyRegistration = formOpeningNodes.find(node => ts.isExpressionStatement(node) && /React\.use(?:Layout)?Effect/.test(node.getText(formAst)) && node.getText(formAst).includes("onComposerReady"))!
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
  const onComposerReady = props.onComposerReady;
  const keyboardShortcutsEnabled = props.keyboardShortcutsEnabled ?? true;
  ${openingNames.map(name => `const ${openingDeclaration(name)!.getText(formAst)};`).join("\n")}
  ${openingCleanup?.getText(formAst) ?? ""}
  ${registrations.map(node => node.getText(formAst)).join("\n")}
  ${composerReadyRegistration.getText(formAst)}
}`, { compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText
const useFormOpening = new Function("React", "useStoreMessageOption", "useHistorySelectionContext", "notification", "t", "tldwClient", `${openingCode}; return useFormOpening;`)(React, useStoreMessageOption, useHistorySelectionContext, { error: io.noop }, io.t, {})

const ComposerProbe = (props: { draftKey: string; inputRef?: React.RefObject<HTMLTextAreaElement>; onRecoverDraft?: RestoreChatDraft; keyboardShortcutsEnabled?: boolean; onComposerReady?: () => void }) => {
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
  return <>{io.useRecovery && <ChatDraftRecovery listDrafts={composer.listRetainedDrafts} copyDraft={composer.copyRetainedDraft} onRestore={onRecoverDraft} disabled={!composer.draftReady} />}<textarea ref={inputRef} aria-label="Composer draft" disabled={!composer.draftReady && !io.productionTextareaReadiness}
    value={composer.form.values.message}
    onChange={event => composer.setMessageValue(event.target.value)} /></>
}
vi.mock("@/components/Sidepanel/Chat/form", () => ({
  SidepanelForm: (props: { draftKey: string }) => io.useComposer ? <ComposerProbe {...props} /> : null
}))
vi.mock("@/components/Sidepanel/Chat/SidepanelHeaderSimple", () => ({
  SidepanelHeaderSimple: (props: {
    chatReady?: boolean
    onOpenChatInWebUi?: () => void
    sidebarOpen: boolean
    setSidebarOpen: React.Dispatch<React.SetStateAction<boolean>>
  }) => (
    <>
      <button
        aria-label="Toggle conversation sidebar"
        aria-expanded={props.sidebarOpen}
        onClick={() => props.setSidebarOpen((previous) => !previous)}>
        Toggle sidebar
      </button>
      <button
        disabled={!props.chatReady}
        onClick={() => props.onOpenChatInWebUi?.()}>
        Open full chat in WebUI
      </button>
    </>
  )
}))
vi.mock("@/components/Sidepanel/Chat/ConnectionBanner", () => ({
  ConnectionBanner: () => null
}))
vi.mock("@/components/Sidepanel/Chat/Sidebar", async original => {
 const actual = await original<typeof import("@/components/Sidepanel/Chat/Sidebar")>()
 return {
  SidepanelChatSidebar: (props: {
    onClose: () => void
    variant: "docked" | "overlay"
    tabs: { id: string; label: string }[]
    onSelectTab: (id: string) => void
    onCloseTab: (
      id: string,
      options?: { discardTemporaryConfirmed: true; isCurrent?: () => boolean }
    ) => Promise<boolean | void>
    onNewTab: () => void
    onOpenLocalHistory: (id: string) => void
    onOpenServerChat: (chat: { id: string; title: string }) => void
  }) => {
    if (io.sidebarHold) throw io.sidebarHold
    if (io.sidebarError) throw io.sidebarError
    if (io.realSidebar) return <actual.SidepanelChatSidebar {...props as React.ComponentProps<typeof actual.SidepanelChatSidebar>} />
    io.closeTab = props.onCloseTab
    return (
      <nav data-testid="compact-sidebar" data-variant={props.variant}>
        <button onClick={props.onClose}>Collapse conversation sidebar</button>
        <button onClick={props.onNewTab}>New conversation</button>
        {props.tabs.map((tab) => (
          <React.Fragment key={tab.id}>
            <button onClick={() => props.onSelectTab(tab.id)}>
              {tab.label}
            </button>
            <button
              aria-label={`Close ${tab.label}`}
              onClick={() => props.onCloseTab(tab.id)}>
              Close
            </button>
          </React.Fragment>
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
  }
}})
vi.mock("@/components/Common/CommandPaletteHost", () => ({
  CommandPaletteHost: ({
    commandPaletteProps
  }: {
    commandPaletteProps: {
      onNewChat: () => void
      onToggleSidebar: () => void
      onSearchHistory: () => void
    }
  }) => (
    <>
      <button onClick={() => commandPaletteProps.onNewChat()}>
        Palette New chat
      </button>
      <button onClick={commandPaletteProps.onToggleSidebar}>
        Palette toggle sidebar
      </button>
      <button onClick={commandPaletteProps.onSearchHistory}>
        Palette find conversations
      </button>
    </>
  )
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
    captureHistorySelection: (...args: unknown[]) => io.capture(...args),
    addChatMessage: (...args: unknown[]) => io.settle(...args),
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

it("mounted shared Sidepanel keeps a stale completed Regenerate through the guarded service", async () => {
  io.user = "alice"
  io.data.clear()
  io.read.mockImplementation(async key => io.data.get(key))
  io.scope.mockImplementation(async () => ({
    requestScope: { config: { serverUrl: "http://chat.test", authMode: "multi-user" }, userId: "alice" },
    scopeSignal: new AbortController().signal,
    scopeInvalidatedSignal: new AbortController().signal,
    release: io.noop
  }))
  useSidepanelChatTabsStore.getState().clear()
  resetHistoryTurnRegistry()
  const scope = { profile_id: "sidepanel-profile", client_session_id: "sidepanel-client" }
  const view = {
    view_session_id: "retained-sidepanel", owner_key: "server-owner", conversation_id: "sidepanel-server",
    interpretation: { kind: "legacy_linear_v1" as const, projection_id: "reviewed" },
    cursor: { kind: "after_message" as const, message_id: "question" }, selection_revision: 3
  }
  const source = {
    version: 1 as const, owner_key: view.owner_key, conversation_id: view.conversation_id,
    fences: { conversation: "3", history: "3", settings: "1" },
    nodes: [{ id: "question", revision: "q1", role: "user" as const, parent_id: null,
      legacy_projection_id: "reviewed", settled: true }],
    source_digest: "sidepanel-source", storage_context_digest: "sidepanel-storage",
    interpretation_status: { kind: "legacy_linear_v1" as const, projection_id: "reviewed", ordered_path_ids: ["question"] }
  }
  const selected = resolveHistorySelection(source, view, "send", "original-prepared")
  if (selected.status !== "ready") throw new Error("invalid_sidepanel_recovery_fixture")
  const assistant = {
    id: "sidepanel-alternative", history_id: view.conversation_id, role: "assistant" as const,
    name: "Original model", content: "Complete Sidepanel alternative", createdAt: 7,
    images: [], parent_message_id: "question"
  }
  const entry: HistoryTurnRecoveryEntry = { scope, turn: {
    operation_id: "sidepanel-regenerate", origin_view: view,
    selection_digest: selected.selection.selection_digest,
    request_context_digest: selected.selection.request_context_digest,
    owner_key: view.owner_key, conversation_id: view.conversation_id,
    input_id: "question", assistant_id: assistant.id, created_at: 7,
    input_text: "Original question", input_images: [], result_text: assistant.content,
    state: "generated_unsaved", outcome: "complete",
    existing_selection: selected.selection, settlement_payload: assistant
  } }
  const state = {
    status: "stale_selection" as const, settingsQualified: true, view, bookmarkScope: scope,
    owner: { kind: "native" as const, conversation_id: view.conversation_id, owner_key: view.owner_key,
      request_scope: { config: { serverUrl: "http://chat.test", authMode: "multi-user" as const }, userId: "alice" },
      validate_lease: () => true }
  }
  // This fixture isolates Sidepanel's controller consumption. Capability
  // parsing, owner binding and Keep's fences still use production services.
  const controller = {
    ...state, capture: null, error: "stale_projection", recoveries: [entry],
    getCurrent: () => state, fence: () => () => true, getReference: () => null,
    activate: io.noop, reset: io.noop, beginLoad: io.noop,
    getSignal: () => new AbortController().signal,
    loadConversation: async () => false, prepareExpansionPath: async () => null,
    choose: async () => false, refreshRecovery: async () => {}, dismissRecovery: async () => {}
  } as unknown as historyController.HistorySelectionController
  const selection = vi.spyOn(historyController, "useHistorySelection").mockReturnValue(controller)
  const load = vi.spyOn(historyRecoveryStorage, "loadHistoryTurnRecoveries").mockResolvedValue([entry])
  const dismiss = vi.spyOn(historyRecoveryStorage, "dismissHistoryTurnRecovery").mockResolvedValue()
  io.capture.mockReset().mockImplementation(async (_id, request) => ({
    status: "legacy_review_required", code: "legacy_review_required",
    acknowledged_capability: "tldw_history_existing_input_v1",
    snapshot: { ...source, interpretation_status: { kind: "legacy_review_required" } },
    view: request.view
  }))
  io.settle.mockReset().mockResolvedValue({ id: assistant.id })
  const mounted = render(<SidepanelChat />)
  try {
    await waitFor(() => expect(io.settle).toHaveBeenCalledOnce())
    expect(io.capture.mock.calls[0][1]).toMatchObject({
      required_capability: "tldw_history_existing_input_v1",
      view: { cursor: { kind: "empty" }, interpretation: { kind: "parent_graph_v1" } }
    })
    expect(io.settle.mock.calls[0]).toMatchObject([
      view.conversation_id,
      { id: assistant.id, content: assistant.content, parent_message_id: "question",
        role: "assistant", tldw_history_existing_input_v1: selected.selection },
      { requestScope: state.owner.request_scope }
    ])
    await waitFor(() => expect(dismiss).toHaveBeenCalledOnce())
  } finally {
    mounted.unmount()
    selection.mockRestore()
    load.mockRestore()
    dismiss.mockRestore()
  }
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

const installSidebarViewport = (initialNarrow: boolean) => {
  let narrow = initialNarrow
  const listeners = new Set<() => void>()
  const media = {
    get matches() {
      return narrow
    },
    media: "(max-width: 400px)",
    onchange: null,
    addEventListener: (_type: string, listener: () => void) =>
      listeners.add(listener),
    removeEventListener: (_type: string, listener: () => void) =>
      listeners.delete(listener),
    addListener: () => {},
    removeListener: () => {},
    dispatchEvent: () => true
  }
  vi.stubGlobal("matchMedia", () => media)
  return {
    setNarrow: (value: boolean) =>
      act(() => {
        narrow = value
        listeners.forEach((listener) => listener())
      })
  }
}

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

describe("current shared Clipper handoff consumer", () => {
  beforeEach(() => {
    io.write.mockReset().mockImplementation(async (key, value) => { io.data.set(key, value) })
    io.user = "alice"
    io.uiMode = "pro"
    io.sequence = 0
    sessionStorage.clear()
    io.submit.mockReset().mockResolvedValue(undefined)
    io.notifyError.mockReset()
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

  it("fails closed for an old unowned pending clip while preserving a verified owner", () => {
    const pending = {
      id: "pending", clipId: "clip", noteId: "note", pageUrl: "https://example.test/clip",
      pageTitle: "Clip", image: "", message: "Captured text", requestOverrides: { chatMode: "normal", useOCR: false }
    }
    expect(normalizePendingWebClipAnalyzeRequest(pending)).toBeNull()
    const owned = { ...pending, ownerKey: ownerKey("alice") }
    expect(normalizePendingWebClipAnalyzeRequest(owned)).toEqual(owned)
  })

  it.each([
    ["another account", ownerKey("bob")],
    ["another origin", ownerKey("alice").replace("http://chat.test", "http://other.test")]
  ])("does not dispatch retained clip text from %s", async (_label, pendingOwnerKey) => {
    io.data.set(storedKey("alice"), tabsState())
    writePendingWebClipAnalyzeRequest({
      id: "private-clip", ownerKey: pendingOwnerKey, clipId: "clip", noteId: "note",
      pageUrl: "https://example.test/private", pageTitle: "Private clip", image: "",
      message: "Another owner's private capture", requestOverrides: { chatMode: "normal", useOCR: false }
    })
    render(<SidepanelChat />)
    await screen.findByText("alice first")
    await act(async () => { await new Promise(resolve => setTimeout(resolve, 0)) })
    expect(io.submit).not.toHaveBeenCalled()
  })

  it("does not retire a replacement owner's pending clip after the consumer unmounts", async () => {
    io.data.set(storedKey("alice"), tabsState())
    const pending = {
      id: "same-id", ownerKey: ownerKey("alice"), clipId: "clip", noteId: "note",
      pageUrl: "https://example.test/private", pageTitle: "Private clip", image: "",
      message: "Alice capture", requestOverrides: { chatMode: "normal" as const, useOCR: false }
    }
    let release!: () => void
    io.submit.mockImplementationOnce(() => new Promise<void>(resolve => { release = resolve }))
    writePendingWebClipAnalyzeRequest(pending)
    const view = render(<SidepanelChat />)
    await waitFor(() => expect(io.submit).toHaveBeenCalledTimes(1))
    act(() => useStoreMessageOption.setState(state => ({ messages: [...state.messages, {
      id: "accepted-clip", isBot: false, name: "You", message: pending.message,
      messageType: WEB_CLIPPER_ANALYZE_MESSAGE_TYPE
    }] })))
    view.unmount()
    const replacement = { ...pending, ownerKey: ownerKey("bob"), message: "Bob capture" }
    writePendingWebClipAnalyzeRequest(replacement)
    await act(async () => { release(); await new Promise(resolve => setTimeout(resolve, 0)) })
    expect(readPendingWebClipAnalyzeRequest()).toEqual(replacement)
    expect(io.notifyError).not.toHaveBeenCalled()
  })

  it("keeps a rejected Clipper handoff and current work, then retires one accepted retry", async () => {
    const pending = {
      id: "pending-clip", ownerKey: ownerKey("alice"), clipId: "clip", noteId: "note", pageUrl: "https://example.test/clip",
      pageTitle: "Captured page", image: "", message: "Analyze retained page",
      requestOverrides: { chatMode: "normal" as const, useOCR: false }
    }
    io.data.set(storedKey("alice"), tabsState())
    io.submit.mockRejectedValueOnce(new Error("unsupported_history_action_context"))
    writePendingWebClipAnalyzeRequest(pending)
    const view = render(<SidepanelChat />)
    await waitFor(() => expect(io.notifyError).toHaveBeenCalledWith(expect.objectContaining({
      description: "Start a new chat to analyze this clip. Your current conversation and saved clip are kept."
    })))
    expect(readPendingWebClipAnalyzeRequest()).toEqual(pending)
    expect(await screen.findByText("alice first")).toBeInTheDocument()
    expect(useSidepanelChatTabsStore.getState().activeTabId).toBe("alice-one")
    expect(io.submit).toHaveBeenCalledTimes(1)
    io.submit.mockImplementationOnce(async () => {
      useStoreMessageOption.setState(state => ({ messages: [...state.messages, {
        id: "accepted-clip", isBot: false, name: "You", message: pending.message,
        messageType: WEB_CLIPPER_ANALYZE_MESSAGE_TYPE
      }] }))
      await new Promise(resolve => setTimeout(resolve, 0))
    })
    // A changed model retries the pending handoff through the existing effect.
    act(() => useStoreMessageOption.setState({ selectedModel: "llama:retry" }))
    await waitFor(() => expect(readPendingWebClipAnalyzeRequest()).toBeNull())
    expect(screen.getByText(pending.message)).toBeInTheDocument()
    view.rerender(<SidepanelChat />)
    await act(async () => { await new Promise(resolve => setTimeout(resolve, 0)) })
    expect(io.submit).toHaveBeenCalledTimes(2)
  })

})

describe.each([
  ["shared", SidepanelChat],
  ["legacy", LegacySidepanelChat]
] as const)("%s sidepanel snapshot ownership", (_name, Route) => {
  beforeEach(() => {
    vi.unstubAllGlobals()
    vi.spyOn(window, "open").mockReset().mockImplementation(() => ({}) as Window)
    io.user = "alice"
    io.uiMode = "pro"
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
    io.realChatShortcuts = false
    io.stubMirror = false
    io.holdDrawerPresentation = false
    io.drawerCallbacks = []
    io.realSidebar = false
    io.productionTextareaReadiness = false
    io.sidebarHold = null
    io.sidebarError = null
    io.closeTab = null
    io.submit.mockReset().mockResolvedValue(undefined)
    io.notifyError.mockReset()
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

  it.each(["keyboard", "header", "palette", "close"])(
    "wide Pro honors %s collapse and Find reopens the same draft",
    async (entry) => {
      installSidebarViewport(false)
      io.data.set(storedKey("alice"), tabsState())
      const page = render(<Route />)
      await screen.findByText("alice first")
      expect(await screen.findByTestId("compact-sidebar")).toHaveAttribute(
        "data-variant",
        "docked"
      )
      if (entry === "keyboard")
        fireEvent.keyDown(document, { key: "b", ctrlKey: true })
      else
        fireEvent.click(
          screen.getByRole("button", {
            name:
              entry === "header"
                ? "Toggle conversation sidebar"
                : entry === "palette"
                  ? "Palette toggle sidebar"
                  : "Collapse conversation sidebar"
          })
        )
      expect(screen.queryByTestId("compact-sidebar")).not.toBeInTheDocument()
      page.rerender(<Route />)
      expect(screen.queryByTestId("compact-sidebar")).not.toBeInTheDocument()
      fireEvent.click(
        screen.getByRole("button", { name: "Palette find conversations" })
      )
      expect(await screen.findByTestId("compact-sidebar")).toHaveAttribute(
        "data-variant",
        "docked"
      )
      expect(
        screen.getByRole("button", { name: "Toggle conversation sidebar" })
      ).toHaveAttribute("aria-expanded", "true")
      expect(screen.getByText("alice first")).toBeVisible()
      expect(useSidepanelChatTabsStore.getState().activeTabId).toBe("alice-one")
      expect(io.submit).not.toHaveBeenCalled()
    }
  )

  it.each(["New conversation", "alice two"])("closes an accepted overlay %s and focuses its owned composer", async action => {
    installSidebarViewport(true)
    io.useComposer = true
    io.data.set(storedKey("alice"), tabsState())
    render(<Route />)
    await screen.findByText("alice first")
    const opener = screen.getByRole("button", { name: "Toggle conversation sidebar" })
    opener.focus()
    fireEvent.click(opener)
    fireEvent.click(await screen.findByRole("button", { name: action }))
    await waitFor(() => expect(screen.queryByTestId("compact-sidebar")).not.toBeInTheDocument())
    await waitFor(() => expect(screen.getByRole("textbox", { name: "Composer draft" })).toHaveFocus())
    expect(io.submit).not.toHaveBeenCalled()
  })

  it("hands an already-ready active tab back to its composer after the native inert boundary clears", async () => {
    installSidebarViewport(true)
    io.useComposer = true
    io.data.set(storedKey("alice"), tabsState())
    render(<Route />)
    await screen.findByText("alice first")
    const composer = screen.getByRole("textbox", { name: "Composer draft" }) as HTMLTextAreaElement
    await waitFor(() => expect(composer).not.toBeDisabled())
    const nativeFocus = composer.focus.bind(composer)
    const blocked = vi.fn()
    vi.spyOn(composer, "focus").mockImplementation(() => {
      // jsdom does not enforce native inert; retain the browser's refusal boundary.
      if (composer.closest("[inert]")) blocked()
      else nativeFocus()
    })
    fireEvent.click(screen.getByRole("button", { name: "Toggle conversation sidebar" }))
    const sidebar = await screen.findByTestId("compact-sidebar")
    within(sidebar).getByRole("button", { name: "alice one" }).focus()
    fireEvent.click(within(sidebar).getByRole("button", { name: "alice one" }))
    await waitFor(() => expect(composer).toHaveFocus())
    expect(blocked).not.toHaveBeenCalled()
    expect(useSidepanelChatTabsStore.getState().activeTabId).toBe("alice-one")
  })

  it("keeps a refused New overlay and suppresses the background New shortcut", async () => {
    installSidebarViewport(true)
    io.realChatShortcuts = true
    io.data.set(storedKey("alice"), tabsState())
    render(<Route />)
    await screen.findByText("alice first")
    fireEvent.click(screen.getByRole("button", { name: "Toggle conversation sidebar" }))
    const sidebar = await screen.findByTestId("compact-sidebar")
    const block = (event: Event) => event.preventDefault()
    window.addEventListener(SETTINGS_NAVIGATION_REQUEST_EVENT, block)
    try { fireEvent.click(within(sidebar).getByRole("button", { name: "New conversation" })) }
    finally { window.removeEventListener(SETTINGS_NAVIGATION_REQUEST_EVENT, block) }
    expect(screen.getByTestId("compact-sidebar")).toBe(sidebar)
    fireEvent.keyDown(document, { key: "u", ctrlKey: true, shiftKey: true })
    expect(useSidepanelChatTabsStore.getState().activeTabId).toBe("alice-one")
    expect(screen.getByTestId("compact-sidebar")).toBe(sidebar)
  })

  it.each(["local", "server"])("dismisses a held accepted %s open only after its publication", async source => {
    installSidebarViewport(true)
    io.data.set(storedKey("alice"), tabsState())
    io.captureNative = true
    io.stubMirror = source === "server"
    io.localId = "B"
    let finish!: () => void
    const held = new Promise<void>(resolve => { finish = resolve })
    const data = { historyInfo: { id: "B", title: "B", server_scope_key: ownerKey("alice") }, messages: [{ id: "B-message", role: "user", content: "B TRANSCRIPT" }] }
    if (source === "local") io.local.mockImplementation(async () => { await held; return data })
    else io.server.mockImplementation(async () => { await held; return [] })
    render(<Route />)
    await screen.findByText("alice first")
    fireEvent.click(screen.getByRole("button", { name: "Toggle conversation sidebar" }))
    const sidebar = await screen.findByTestId("compact-sidebar")
    fireEvent.click(within(sidebar).getByRole("button", { name: source === "local" ? "Open cached history" : "Open server history" }))
    await waitFor(() => expect(source === "local" ? io.local : io.server).toHaveBeenCalled())
    expect(screen.getByTestId("compact-sidebar")).toBe(sidebar)
    await act(async () => { finish(); await held })
    await waitFor(() => expect(useSidepanelChatTabsStore.getState().activeTabId).not.toBe("alice-one"))
    expect(screen.queryByTestId("compact-sidebar")).not.toBeInTheDocument()
    expect(io.submit).not.toHaveBeenCalled()
  })

  it("does not let an old accepted load dismiss a reopened overlay or steal its focus", async () => {
    installSidebarViewport(true)
    io.useComposer = true
    io.data.set(storedKey("alice"), tabsState())
    io.localId = "B"
    let finish!: () => void
    const held = new Promise<void>(resolve => { finish = resolve })
    io.local.mockImplementation(async () => { await held; return { historyInfo: { id: "B", title: "B", server_scope_key: ownerKey("alice") }, messages: [{ id: "B-message", role: "user", content: "B TRANSCRIPT" }] } })
    render(<Route />)
    await screen.findByText("alice first")
    fireEvent.click(screen.getByRole("button", { name: "Toggle conversation sidebar" }))
    fireEvent.click(await screen.findByRole("button", { name: "Open cached history" }))
    await waitFor(() => expect(io.local).toHaveBeenCalledWith("B"))
    fireEvent.click(screen.getByRole("button", { name: "Collapse conversation sidebar" }))
    fireEvent.click(screen.getByRole("button", { name: "Toggle conversation sidebar" }))
    const reopened = await screen.findByTestId("compact-sidebar")
    const control = within(reopened).getByRole("button", { name: "New conversation" })
    control.focus()
    await act(async () => { finish(); await held })
    await screen.findByText("B TRANSCRIPT")
    expect(screen.getByTestId("compact-sidebar")).toBe(reopened)
    expect(control).toHaveFocus()
  })

  it.each(["local", "server"])("retains the overlay when a %s target is refused", async source => {
    installSidebarViewport(true)
    io.data.set(storedKey("alice"), tabsState())
    io.local.mockResolvedValue(null)
    render(<Route />)
    await screen.findByText("alice first")
    fireEvent.click(screen.getByRole("button", { name: "Toggle conversation sidebar" }))
    const sidebar = await screen.findByTestId("compact-sidebar")
    fireEvent.click(within(sidebar).getByRole("button", { name: source === "local" ? "Open cached history" : "Open server history" }))
    await waitFor(() => expect(useStoreMessageOption.getState().isLoading).toBe(false))
    expect(screen.getByTestId("compact-sidebar")).toBe(sidebar)
    expect(useSidepanelChatTabsStore.getState().activeTabId).toBe("alice-one")
  })

  it("waits for an existing local tab's preparation before dismissing its overlay", async () => {
    installSidebarViewport(true)
    const saved = tabsState()
    saved.tabs[1].historyId = "B"
    saved.snapshotsById["alice-two"].historyId = "B"
    saved.snapshotsById["alice-two"].chatDraftHandle = { address: { kind: "local", id: "B", profileId: "profile" }, ownerHash: chatDraftOwnerKey({ config: { serverUrl: "http://chat.test", authMode: "multi-user" }, userId: "alice" }) }
    io.data.set(storedKey("alice"), saved)
    io.localId = "B"
    let release!: () => void
    const held = new Promise<void>(resolve => { release = resolve })
    io.local.mockImplementation(async () => { await held; return { historyInfo: { id: "B", title: "B", server_scope_key: ownerKey("alice") }, messages: [{ id: "B-message", role: "user", content: "EXISTING B" }] } })
    render(<Route />)
    await screen.findByText("alice first")
    fireEvent.click(screen.getByRole("button", { name: "Toggle conversation sidebar" }))
    const sidebar = await screen.findByTestId("compact-sidebar")
    fireEvent.click(within(sidebar).getByRole("button", { name: "Open cached history" }))
    await waitFor(() => expect(io.local).toHaveBeenCalledWith("B"))
    expect(screen.getByTestId("compact-sidebar")).toBe(sidebar)
    expect(useSidepanelChatTabsStore.getState().activeTabId).toBe("alice-one")
    await act(async () => { release(); await held })
    await screen.findByText("EXISTING B")
    expect(useSidepanelChatTabsStore.getState().activeTabId).toBe("alice-two")
    expect(useSidepanelChatTabsStore.getState().tabs).toHaveLength(2)
    expect(screen.queryByTestId("compact-sidebar")).not.toBeInTheDocument()
  })

  it("does not dismiss a replacement account's overlay after A to B to A retirement", async () => {
    installSidebarViewport(true)
    io.data.set(storedKey("alice"), tabsState())
    io.data.set(storedKey("bob"), tabsState("bob"))
    io.localId = "B"
    let release!: () => void
    const held = new Promise<void>(resolve => { release = resolve })
    io.local.mockImplementation(async () => { await held; return { historyInfo: { id: "B", title: "B", server_scope_key: ownerKey("alice") }, messages: [{ id: "B-message", role: "user", content: "RETIRED B" }] } })
    render(<Route />)
    await screen.findByText("alice first")
    fireEvent.click(screen.getByRole("button", { name: "Toggle conversation sidebar" }))
    fireEvent.click(await screen.findByRole("button", { name: "Open cached history" }))
    await waitFor(() => expect(io.local).toHaveBeenCalledWith("B"))
    switchAccount("bob")
    await screen.findByText("bob first")
    switchAccount("alice")
    await screen.findByText("alice first")
    fireEvent.click(screen.getByRole("button", { name: "Toggle conversation sidebar" }))
    const replacement = await screen.findByTestId("compact-sidebar")
    await act(async () => { release(); await held })
    expect(screen.getByTestId("compact-sidebar")).toBe(replacement)
    expect(screen.queryByText("RETIRED B")).not.toBeInTheDocument()
    expect(io.submit).not.toHaveBeenCalled()
  })

  it("focuses Find through the real Sidebar and Drawer once, without replaying it on reopen", async () => {
    installSidebarViewport(true)
    io.realSidebar = true
    io.data.set(storedKey("alice"), tabsState())
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const view = render(<QueryClientProvider client={client}><Route /></QueryClientProvider>)
    await screen.findByText("alice first")
    const find = screen.getByRole("button", { name: "Palette find conversations" })
    find.focus()
    fireEvent.click(find)
    const input = await screen.findByPlaceholderText("Search chats...")
    await waitFor(() => expect(input).toHaveFocus())
    fireEvent.click(within(screen.getByRole("dialog", { name: "Chats" })).getAllByRole("button", { name: "Close", exact: true })[0])
    const toggle = screen.getByRole("button", { name: "Toggle conversation sidebar" })
    toggle.focus()
    fireEvent.click(toggle)
    const dialog = await screen.findByRole("dialog", { name: "Chats" })
    expect(dialog).toHaveAttribute("aria-modal", "true")
    expect(screen.getByPlaceholderText("Search chats...")).not.toHaveFocus()
    view.unmount()
    client.clear()
  })

  it("waits for ready before focusing the same enabled textarea the production form exposes", async () => {
    installSidebarViewport(true)
    io.useComposer = true
    io.productionTextareaReadiness = true
    io.data.set(storedKey("alice"), tabsState())
    let release!: () => void
    const held = new Promise<void>(resolve => { release = resolve })
    io.read.mockImplementation(async key => { if (String(key).startsWith("registry:draft:tldw:sidepanelChatDraft:new-")) await held; return io.data.get(key) })
    render(<Route />)
    await screen.findByText("alice first")
    await waitFor(() => expect(screen.getByRole("textbox", { name: "Composer draft" })).toBeEnabled())
    fireEvent.click(screen.getByRole("button", { name: "Toggle conversation sidebar" }))
    const create = await screen.findByRole("button", { name: "New conversation" })
    create.focus()
    fireEvent.click(create)
    await waitFor(() => expect(io.read.mock.calls.some(([key]) => String(key).startsWith("registry:draft:tldw:sidepanelChatDraft:new-"))).toBe(true))
    const input = screen.getByRole("textbox", { name: "Composer draft" })
    expect(input).toBeEnabled()
    expect(input).not.toHaveFocus()
    await act(async () => { release(); await held })
    await waitFor(() => expect(input).toHaveFocus())
  })

  it("keeps a held Sidebar content boundary dismissible and never resurrects its late content", async () => {
    installSidebarViewport(true)
    io.data.set(storedKey("alice"), tabsState())
    render(<Route />)
    await screen.findByText("alice first")
    let release!: () => void
    const held = new Promise<void>(resolve => { release = resolve })
    io.sidebarHold = held
    const opener = screen.getByRole("button", { name: "Toggle conversation sidebar" })
    opener.focus()
    fireEvent.click(opener)
    const dialog = await screen.findByRole("dialog", { name: "Chats" })
    expect(screen.getByRole("status")).toHaveTextContent("Loading chats")
    expect(screen.getByTestId("chat-main")).toHaveAttribute("inert")
    fireEvent.keyDown(dialog, { key: "Escape", code: "Escape" })
    expect(screen.queryByRole("dialog", { name: "Chats" })).not.toBeInTheDocument()
    expect(screen.getByTestId("chat-main")).not.toHaveAttribute("inert")
    expect(opener).toHaveFocus()
    await act(async () => { io.sidebarHold = null; release(); await held })
    expect(screen.queryByTestId("compact-sidebar")).not.toBeInTheDocument()
    expect(useSidepanelChatTabsStore.getState().activeTabId).toBe("alice-one")
    expect(io.submit).not.toHaveBeenCalled()
  })

  it("keeps a rejected Sidebar content boundary dismissible without losing the active chat", async () => {
    installSidebarViewport(true)
    io.data.set(storedKey("alice"), tabsState())
    render(<Route />)
    await screen.findByText("alice first")
    io.sidebarError = new Error("Declared Sidebar chunk failure")
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => {})
    try {
      fireEvent.click(screen.getByRole("button", { name: "Toggle conversation sidebar" }))
      const dialog = await screen.findByRole("dialog", { name: "Chats" })
      expect(screen.getByRole("alert")).toHaveTextContent("Could not load chats")
      fireEvent.click(within(dialog).getByRole("button", { name: "Close", exact: true }))
      expect(screen.queryByRole("dialog", { name: "Chats" })).not.toBeInTheDocument()
      expect(screen.getByText("alice first")).toBeVisible()
      expect(useSidepanelChatTabsStore.getState().activeTabId).toBe("alice-one")
    } finally { consoleError.mockRestore(); io.sidebarError = null }
  })

  it("lets newer loading-drawer focus retire Find before deferred Sidebar content mounts", async () => {
    installSidebarViewport(true)
    io.realSidebar = true
    io.data.set(storedKey("alice"), tabsState())
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const view = render(<QueryClientProvider client={client}><Route /></QueryClientProvider>)
    await screen.findByText("alice first")
    let release!: () => void
    const held = new Promise<void>(resolve => { release = resolve })
    io.sidebarHold = held
    const find = screen.getByRole("button", { name: "Palette find conversations" })
    find.focus()
    fireEvent.click(find)
    const dialog = await screen.findByRole("dialog", { name: "Chats" })
    const close = within(dialog).getByRole("button", { name: "Close", exact: true })
    close.focus()
    await act(async () => { io.sidebarHold = null; release(); await held })
    await screen.findByRole("textbox", { name: "Search chats" })
    expect(close).toHaveFocus()
    view.unmount()
    client.clear()
  })

  it("wraps the actual Drawer Tab boundaries without sending focus to the background", async () => {
    installSidebarViewport(true)
    io.realSidebar = true
    io.data.set(storedKey("alice"), tabsState())
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const view = render(<QueryClientProvider client={client}><Route /></QueryClientProvider>)
    await screen.findByText("alice first")
    fireEvent.click(screen.getByRole("button", { name: "Toggle conversation sidebar" }))
    const drawer = await screen.findByRole("dialog", { name: "Chats" })
    const first = within(drawer).getAllByRole("button", { name: "Close", exact: true })[0]
    const last = await within(drawer).findByRole("button", { name: "Pro", exact: true })
    const outer = drawer.closest<HTMLElement>(".sidepanel-chat-overlay")!
    outer.focus()
    const initialBack = new KeyboardEvent("keydown", { key: "Tab", shiftKey: true, bubbles: true, cancelable: true })
    fireEvent(outer, initialBack)
    expect(initialBack.defaultPrevented).toBe(true)
    expect(last).toHaveFocus()
    last.focus()
    const next = new KeyboardEvent("keydown", { key: "Tab", bubbles: true, cancelable: true })
    fireEvent(last, next)
    expect(next.defaultPrevented).toBe(true)
    expect(first).toHaveFocus()
    const previous = new KeyboardEvent("keydown", { key: "Tab", shiftKey: true, bubbles: true, cancelable: true })
    fireEvent(first, previous)
    expect(previous.defaultPrevented).toBe(true)
    expect(last).toHaveFocus()
    view.unmount()
    client.clear()
  })

  it("lets the actual nested Rename modal own the first Escape while Chats stays open", async () => {
    installSidebarViewport(true)
    io.realSidebar = true
    io.data.set(storedKey("alice"), tabsState())
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const view = render(<QueryClientProvider client={client}><Route /></QueryClientProvider>)
    await screen.findByText("alice first")
    fireEvent.click(screen.getByRole("button", { name: "Toggle conversation sidebar" }))
    const sidebar = await screen.findByTestId("sidepanel-chat-sidebar")
    fireEvent.contextMenu(within(sidebar).getByRole("button", { name: "alice one" }))
    fireEvent.click(await screen.findByRole("menuitem", { name: "Rename" }))
    // rc-component reuses "test-id" for both titles in test mode; locate the real nested title.
    const rename = (await screen.findByText("Rename conversation")).closest('[role="dialog"]')!
    fireEvent.keyDown(rename, { key: "Escape", code: "Escape" })
    await waitFor(() => expect(screen.queryByText("Rename conversation")).not.toBeInTheDocument())
    expect(screen.getByRole("dialog", { name: "Chats" })).toBeInTheDocument()
    expect(useSidepanelChatTabsStore.getState().activeTabId).toBe("alice-one")
    expect(io.submit).not.toHaveBeenCalled()
    view.unmount()
    client.clear()
  })

  it("rejects retained Close and presentation callbacks after a new opening", async () => {
    installSidebarViewport(true)
    io.realSidebar = true
    io.holdDrawerPresentation = true
    io.data.set(storedKey("alice"), tabsState())
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const view = render(<QueryClientProvider client={client}><Route /></QueryClientProvider>)
    await screen.findByText("alice first")
    fireEvent.click(screen.getByRole("button", { name: "Toggle conversation sidebar" }))
    const first = await screen.findByRole("dialog", { name: "Chats" })
    const old = io.drawerCallbacks.at(-1)!
    fireEvent.click(within(first).getAllByRole("button", { name: "Close", exact: true })[0])
    fireEvent.click(screen.getByRole("button", { name: "Palette find conversations" }))
    const search = await screen.findByRole("textbox", { name: "Search chats" })
    const current = io.drawerCallbacks.at(-1)!
    await act(async () => { old.onClose?.(); old.afterOpenChange?.(true) })
    expect(screen.getByRole("dialog", { name: "Chats" })).toBeInTheDocument()
    expect(search).not.toHaveFocus()
    await act(async () => { current.afterOpenChange?.(true) })
    expect(search).toHaveFocus()
    await act(async () => { old.afterOpenChange?.(false) })
    const find = screen.getByRole("button", { name: "Palette find conversations" })
    find.focus()
    fireEvent.click(find)
    expect(search).toHaveFocus()
    view.unmount()
    client.clear()
  })

  it("returns canceled overlay focus to its original initiator", async () => {
    installSidebarViewport(true)
    io.data.set(storedKey("alice"), tabsState())
    render(<Route />)
    await screen.findByText("alice first")
    const opener = screen.getByRole("button", { name: "Toggle conversation sidebar" })
    opener.focus()
    fireEvent.click(opener)
    const close = await screen.findByRole("button", { name: "Collapse conversation sidebar" })
    close.focus()
    fireEvent.click(close)
    await waitFor(() => expect(opener).toHaveFocus())
  })

  it("cold narrow Pro starts collapsed and its keyboard opens an overlay", async () => {
    installSidebarViewport(true)
    io.data.set(storedKey("alice"), tabsState())
    render(<Route />)
    await screen.findByText("alice first")
    expect(screen.queryByTestId("compact-sidebar")).not.toBeInTheDocument()
    fireEvent.keyDown(document, { key: "b", ctrlKey: true })
    expect(await screen.findByTestId("compact-sidebar")).toHaveAttribute(
      "data-variant",
      "overlay"
    )
    fireEvent.keyDown(document, { key: "b", ctrlKey: true })
    expect(screen.getByTestId("compact-sidebar")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Collapse conversation sidebar" }))
    expect(screen.queryByTestId("compact-sidebar")).not.toBeInTheDocument()
  })

  it("keeps current docked Collapse usable after an open overlay changes layout, while rejecting its old shell callbacks", async () => {
    const viewport = installSidebarViewport(true)
    io.data.set(storedKey("alice"), tabsState())
    render(<Route />)
    await screen.findByText("alice first")
    fireEvent.click(screen.getByRole("button", { name: "Toggle conversation sidebar" }))
    await screen.findByRole("dialog", { name: "Chats" })
    const old = io.drawerCallbacks.at(-1)!
    viewport.setNarrow(false)
    expect(await screen.findByTestId("compact-sidebar")).toHaveAttribute("data-variant", "docked")
    await act(async () => { old.onClose?.(); old.afterOpenChange?.(true); old.afterOpenChange?.(false) })
    expect(screen.getByTestId("compact-sidebar")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Collapse conversation sidebar" }))
    expect(screen.queryByTestId("compact-sidebar")).not.toBeInTheDocument()
  })

  it("changes default visibility only at docked-layout transitions", async () => {
    const viewport = installSidebarViewport(false)
    io.uiMode = "casual"
    io.data.set(storedKey("alice"), tabsState())
    const page = render(<Route />)
    await screen.findByText("alice first")
    expect(screen.queryByTestId("compact-sidebar")).not.toBeInTheDocument()
    io.uiMode = "pro"
    page.rerender(<Route />)
    expect(await screen.findByTestId("compact-sidebar")).toHaveAttribute(
      "data-variant",
      "docked"
    )
    expect(
      screen.getByRole("button", { name: "Toggle conversation sidebar" })
    ).toHaveAttribute("aria-expanded", "true")
    fireEvent.keyDown(document, { key: "b", ctrlKey: true })
    expect(screen.queryByTestId("compact-sidebar")).not.toBeInTheDocument()
    viewport.setNarrow(true)
    expect(screen.queryByTestId("compact-sidebar")).not.toBeInTheDocument()
    fireEvent.keyDown(document, { key: "b", ctrlKey: true })
    expect(await screen.findByTestId("compact-sidebar")).toHaveAttribute(
      "data-variant",
      "overlay"
    )
    viewport.setNarrow(false)
    expect(await screen.findByTestId("compact-sidebar")).toHaveAttribute(
      "data-variant",
      "docked"
    )
    io.uiMode = "casual"
    page.rerender(<Route />)
    expect(screen.queryByTestId("compact-sidebar")).not.toBeInTheDocument()
    expect(screen.getByText("alice first")).toBeVisible()
  })

  if (_name === "shared")
    it("final capture cannot overwrite a synchronous New/reset replacement", async () => {
      io.data.set(storedKey("alice"), tabsState())
      io.local.mockImplementation(async (id) => ({
        historyInfo: { id, title: id, server_scope_key: ownerKey("alice") },
        messages: [
          { id: `${id}-message`, role: "user", content: `${id} TRANSCRIPT` }
        ]
      }))
      render(<Route />)
      await screen.findByRole("button", { name: "alice one" })
      await act(async () => {
        await io.controller!.loadConversation({ historyId: "A" }, null)
      })
      const origin = io.controller!.getCurrent().view!
      const replacement = [
        {
          id: "new-user",
          isBot: false,
          name: "You",
          message: "NEW COMPACT DRAFT"
        }
      ]
      let replaced = false
      const unsubscribe = useStoreMessageOption.subscribe((state, previous) => {
        if (
          !replaced &&
          state.history !== previous.history &&
          state.history.some((row) => row.content === "A TRANSCRIPT")
        ) {
          replaced = true
          io.controller!.reset()
          state.startNewChatDraft()
          useStoreMessageOption.setState({
            historyId: null,
            serverChatId: null,
            history: [{ role: "user", content: "NEW COMPACT DRAFT" }],
            messages: replacement
          })
        }
      })
      try {
        await act(async () => {
          await io.controller!.followResult(origin, "A-message", () => true)
        })
        expect(replaced).toBe(true)
        expect(useStoreMessageOption.getState().messages).toEqual(replacement)
        expect(useStoreMessageOption.getState().history).toEqual([
          { role: "user", content: "NEW COMPACT DRAFT" }
        ])
      } finally {
        unsubscribe()
      }
    })

  it.each([
    ["active", false],
    ["active", true],
    ["inactive", false],
    ["inactive", true],
    ["last", false],
    ["last", true]
  ] as const)(
    "admits %s Temporary tab closure only after discard accepts (%s)",
    async (placement, accepted) => {
      io.useComposer = true
      const saved = tabsState()
      if (placement === "last") saved.tabs = saved.tabs.slice(0, 1)
      io.data.set(storedKey("alice"), saved)
      render(<Route />)
      await screen.findByText("alice first")
      // The lazy sidebar actions mount independently of transcript hydration.
      await waitFor(() => expect(io.closeTab).toBeTypeOf("function"))
      act(() => {
        const state = useStoreMessageOption.getState()
        state.setTemporaryChat(true)
        state.startNewChatDraft(true)
        state.setHistoryId("temp")
        state.setHistory(snapshot("TEMPORARY CLOSE TARGET").history)
        state.setMessages(snapshot("TEMPORARY CLOSE TARGET").messages)
      })
      await waitFor(() =>
        expect(screen.getByLabelText("Composer draft")).toBeEnabled()
      )
      fireEvent.change(screen.getByLabelText("Composer draft"), {
        target: { value: "TEMP NEXT UNSENT  " }
      })
      if (placement === "inactive") {
        fireEvent.click(screen.getByRole("button", { name: "alice two" }))
        await screen.findByText("alice inactive")
        await waitFor(() =>
          expect(screen.getByLabelText("Composer draft")).toBeEnabled()
        )
      }
      const tabsBefore = useSidepanelChatTabsStore.getState().tabs
      const targetBefore = useSidepanelChatTabsStore
        .getState()
        .getSnapshot("alice-one")
      const currentBefore = useStoreMessageOption.getState()
      const activeBefore = useSidepanelChatTabsStore.getState().activeTabId
      const draftBefore = (
        screen.getByLabelText("Composer draft") as HTMLTextAreaElement
      ).value
      const confirm = vi.spyOn(window, "confirm").mockReturnValue(accepted)
      try {
        let result: boolean | void
        await act(async () => {
          result = await io.closeTab!("alice-one")
        })
        expect(confirm).toHaveBeenCalledTimes(1)
        expect(result!).toBe(accepted)
        if (!accepted) {
          expect(useSidepanelChatTabsStore.getState().tabs).toEqual(tabsBefore)
          expect(
            useSidepanelChatTabsStore.getState().getSnapshot("alice-one")
          ).toEqual(targetBefore)
        } else {
          expect(
            useSidepanelChatTabsStore
              .getState()
              .tabs.some((tab) => tab.id === "alice-one")
          ).toBe(false)
          expect(
            useSidepanelChatTabsStore.getState().getSnapshot("alice-one")
          ).toBeUndefined()
        }
        if (!accepted || placement === "inactive") {
          expect(useSidepanelChatTabsStore.getState().activeTabId).toBe(
            activeBefore
          )
          expect(useStoreMessageOption.getState()).toMatchObject({
            chatDraftAddress: currentBefore.chatDraftAddress,
            chatDraftRevision: currentBefore.chatDraftRevision,
            history: currentBefore.history,
            messages: currentBefore.messages,
            temporaryChat: currentBefore.temporaryChat
          })
          expect(screen.getByLabelText("Composer draft")).toHaveValue(
            draftBefore
          )
        } else if (placement === "active") {
          expect(useSidepanelChatTabsStore.getState().activeTabId).toBe(
            "alice-two"
          )
          expect(useStoreMessageOption.getState().messages).toEqual(
            snapshot("alice inactive").messages
          )
        } else {
          expect(useSidepanelChatTabsStore.getState().tabs).toHaveLength(1)
          expect(useStoreMessageOption.getState().messages).toEqual([])
        }
      } finally {
        confirm.mockRestore()
      }
    }
  )

  it("rejects Temporary closure when confirmation retires the original H1 view under unchanged IDs", async () => {
    io.data.set(storedKey("alice"), tabsState())
    render(<Route />)
    await screen.findByText("alice first")
    act(() => {
      const state = useStoreMessageOption.getState()
      state.setTemporaryChat(true)
      state.startNewChatDraft(true)
      state.setHistoryId("temp")
      state.setMessages(snapshot("TEMPORARY H1 TARGET").messages)
    })
    const confirm = vi.spyOn(window, "confirm").mockImplementation(() => {
      io.controller!.reset()
      return true
    })
    try {
      let result: boolean | void
      await act(async () => {
        result = await io.closeTab!("alice-one")
      })
      expect(confirm).toHaveBeenCalledTimes(1)
      expect(result!).toBe(false)
      expect(useSidepanelChatTabsStore.getState().activeTabId).toBe("alice-one")
      expect(useStoreMessageOption.getState().messages).toEqual(
        snapshot("TEMPORARY H1 TARGET").messages
      )
    } finally {
      confirm.mockRestore()
    }
  })

  it("rejects a missing last-tab target without creating a replacement", async () => {
    const saved = tabsState()
    saved.tabs = saved.tabs.slice(0, 1)
    io.data.set(storedKey("alice"), saved)
    render(<Route />)
    await screen.findByText("alice first")
    const before = useSidepanelChatTabsStore.getState().tabs
    let result: boolean | void
    await act(async () => {
      result = await io.closeTab!("missing")
    })
    expect(result!).toBe(false)
    expect(useSidepanelChatTabsStore.getState().tabs).toEqual(before)
    expect(useSidepanelChatTabsStore.getState().activeTabId).toBe("alice-one")
  })

  it.each(["new", "switch"])(
    "retains Temporary content without discard confirmation during %s",
    async (action) => {
      io.data.set(storedKey("alice"), tabsState())
      render(<Route />)
      await screen.findByText("alice first")
      act(() => {
        const state = useStoreMessageOption.getState()
        state.setTemporaryChat(true)
        state.startNewChatDraft(true)
        state.setHistoryId("temp")
        state.setMessages(snapshot("RETAINED TEMPORARY SOURCE").messages)
      })
      const confirm = vi.spyOn(window, "confirm").mockReturnValue(false)
      try {
        fireEvent.click(
          screen.getByRole("button", {
            name: action === "new" ? "New conversation" : "alice two"
          })
        )
        await waitFor(() =>
          expect(useSidepanelChatTabsStore.getState().activeTabId).not.toBe(
            "alice-one"
          )
        )
        expect(confirm).not.toHaveBeenCalled()
        expect(
          useSidepanelChatTabsStore.getState().getSnapshot("alice-one")
            ?.messages
        ).toEqual(snapshot("RETAINED TEMPORARY SOURCE").messages)
      } finally {
        confirm.mockRestore()
      }
    }
  )

  it("uses already acknowledged bulk discard without asking a second question", async () => {
    io.data.set(storedKey("alice"), tabsState())
    render(<Route />)
    await screen.findByRole("button", { name: "alice one" })
    await screen.findByText("alice first")
    act(() => {
      const state = useStoreMessageOption.getState()
      state.setTemporaryChat(true)
      state.startNewChatDraft(true)
      state.setHistoryId("temp")
      state.setMessages(snapshot("BULK ACKNOWLEDGED TEMPORARY").messages)
    })
    await waitFor(() =>
      expect(
        useSidepanelChatTabsStore.getState().getSnapshot("alice-one")
          ?.temporaryChat
      ).toBe(true)
    )
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(false)
    try {
      let accepted: boolean | void
      await act(async () => {
        accepted = await io.closeTab!("alice-one", {
          discardTemporaryConfirmed: true
        })
      })
      expect(accepted!).toBe(true)
      expect(confirm).not.toHaveBeenCalled()
      expect(
        useSidepanelChatTabsStore
          .getState()
          .tabs.some((tab) => tab.id === "alice-one")
      ).toBe(false)
    } finally {
      confirm.mockRestore()
    }
  })

  it("retains Temporary source without discard confirmation when opening owned cached history", async () => {
    io.data.set(storedKey("alice"), tabsState())
    io.localId = "cached"
    io.local.mockResolvedValue({
      historyInfo: {
        id: "cached",
        title: "Cached history",
        server_scope_key: ownerKey("alice")
      },
      messages: [
        { id: "cached-row", role: "user", content: "CACHED DESTINATION" }
      ]
    })
    render(<Route />)
    await screen.findByText("alice first")
    act(() => {
      const state = useStoreMessageOption.getState()
      state.setTemporaryChat(true)
      state.startNewChatDraft(true)
      state.setHistoryId("temp")
      state.setMessages(snapshot("RETAINED HISTORY SOURCE").messages)
    })
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(false)
    try {
      fireEvent.click(
        screen.getByRole("button", { name: "Open cached history" })
      )
      await screen.findByText("CACHED DESTINATION")
      expect(confirm).not.toHaveBeenCalled()
      expect(
        useSidepanelChatTabsStore.getState().getSnapshot("alice-one")?.messages
      ).toEqual(snapshot("RETAINED HISTORY SOURCE").messages)
    } finally {
      confirm.mockRestore()
    }
  })

  it.each(["replacement", "caller"])(
    "keeps the source after %s retirement while admitted close waits for owned history",
    async (ending) => {
      const saved = tabsState()
      Object.assign(saved.snapshotsById["alice-two"], {
        historyId: "saved-B",
        chatDraftHandle: {
          address: { kind: "local", id: "saved-B", profileId: "profile" },
          ownerHash: chatDraftOwnerKey({
            config: { serverUrl: "http://chat.test", authMode: "multi-user" },
            userId: "alice"
          })
        }
      })
      io.data.set(storedKey("alice"), saved)
      let finish!: (value: unknown) => void
      io.local
        .mockResolvedValue({
          historyInfo: { id: "saved-B", local_owner_key: "local-profile" },
          messages: []
        })
        .mockImplementationOnce(
          () =>
            new Promise((resolve) => {
              finish = resolve
            })
        )
      render(<Route />)
      await screen.findByRole("button", { name: "alice one" })
      await screen.findByText("alice first")
      act(() => {
        const state = useStoreMessageOption.getState()
        state.setTemporaryChat(true)
        state.startNewChatDraft(true)
        state.setHistoryId("temp")
        state.setMessages(snapshot("OLD CLOSING TEMPORARY").messages)
      })
      const confirm = vi.spyOn(window, "confirm").mockReturnValue(true)
      try {
        let current = true
        let closing!: Promise<boolean | void>
        act(() => {
          closing = io.closeTab!("alice-one", {
            discardTemporaryConfirmed: true,
            isCurrent: () => current
          })
        })
        await waitFor(() => expect(finish).toBeTypeOf("function"))
        if (ending === "replacement")
          act(() => {
            useStoreMessageOption.getState().startNewChatDraft(true)
            useStoreMessageOption
              .getState()
              .setMessages(snapshot("NEW SOURCE WHILE CLOSE WAITS").messages)
          })
        else current = false
        let accepted: boolean | void
        await act(async () => {
          finish({
            historyInfo: { id: "saved-B", local_owner_key: "local-profile" },
            messages: []
          })
          accepted = await closing
        })
        expect(accepted!).toBe(false)
        expect(useSidepanelChatTabsStore.getState().activeTabId).toBe(
          "alice-one"
        )
        expect(useSidepanelChatTabsStore.getState().tabs).toHaveLength(2)
        expect(useStoreMessageOption.getState().messages).toEqual(
          snapshot(
            ending === "replacement"
              ? "NEW SOURCE WHILE CLOSE WAITS"
              : "OLD CLOSING TEMPORARY"
          ).messages
        )
      } finally {
        confirm.mockRestore()
      }
    }
  )

  it.each(["active", "last"])(
    "retains a same-ID replacement published during its own %s close transition",
    async (placement) => {
      const saved = tabsState()
      if (placement === "last") saved.tabs = saved.tabs.slice(0, 1)
      io.data.set(storedKey("alice"), saved)
      render(<Route />)
      await screen.findByText("alice first")
      act(() => {
        const state = useStoreMessageOption.getState()
        state.setTemporaryChat(true)
        state.startNewChatDraft(true)
        state.setHistoryId("temp")
        state.setMessages(snapshot("APPROVED OLD TARGET").messages)
      })
      const address = useStoreMessageOption.getState().chatDraftAddress!
      if (address.kind !== "fresh")
        throw new Error("Expected the real Temporary draft address")
      let replaced = false
      const unsubscribe = useSidepanelChatTabsStore.subscribe((state) => {
        if (replaced || state.activeTabId === "alice-one") return
        replaced = true
        state.setSnapshot("alice-one", {
          ...state.getSnapshot("alice-one")!,
          chatDraftAddress: { ...address, id: "replacement-draft" },
          messages: snapshot("SAME ID NEW TARGET").messages
        })
      })
      const confirm = vi.spyOn(window, "confirm").mockReturnValue(true)
      try {
        let accepted: boolean | void
        await act(async () => {
          accepted = await io.closeTab!("alice-one")
        })
        expect(replaced).toBe(true)
        expect(accepted!).toBe(false)
        expect(
          useSidepanelChatTabsStore
            .getState()
            .tabs.some((tab) => tab.id === "alice-one")
        ).toBe(true)
        expect(
          useSidepanelChatTabsStore.getState().getSnapshot("alice-one")
            ?.messages
        ).toEqual(snapshot("SAME ID NEW TARGET").messages)
      } finally {
        unsubscribe()
        confirm.mockRestore()
      }
    }
  )
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

  if (_name === "shared")
    it.each(["extension", "webui"])(
      "opens full server settings from the actual compact shell in %s",
      async (surface) => {
        io.scope.mockRejectedValue(new Error("offline"))
        if (surface === "extension")
          vi.stubGlobal("browser", {
            runtime: {
              id: "test-extension",
              getURL: (route: string) => `chrome-extension://test${route}`
            }
          })
        else {
          // The deployed WebUI installs the browser shim, including getURL without an extension id.
          vi.stubGlobal("browser", {
            runtime: {
              id: undefined,
              getURL: (route: string) => `https://webui.test${route}`
            }
          })
          vi.stubGlobal("chrome", undefined)
        }
        render(
          <MemoryRouter initialEntries={["/chat"]}>
            <SidepanelRouteShell />
          </MemoryRouter>
        )
        fireEvent.click(
          await screen.findByRole("button", { name: "Server settings" })
        )
        expect(window.open).toHaveBeenCalledWith(
          surface === "extension"
            ? "chrome-extension://test/options.html#/settings/tldw"
            : "/settings/tldw",
          "_blank",
          "noopener,noreferrer"
        )
        expect(
          screen.queryByTestId("not-found-recovery-panel")
        ).not.toBeInTheDocument()
        expect(screen.getByRole("alert")).toBeInTheDocument()
        // The real shell persists its header preference, never chat/draft content.
        expect(recoveryWrites()).toEqual([["headerShortcutsExpanded", false]])
      }
    )

  it.each(["account", "unmount"])("ignores a failed cold Retry after %s retirement", async ending => {
    const alice = tabsState()
    io.data.set(storedKey("alice"), structuredClone(alice))
    io.scope.mockRejectedValueOnce(new Error("offline"))
    const page = render(<Route />)
    const retry = await screen.findByRole("button", { name: "Retry" })
    let reject!: (error: Error) => void
    io.scope.mockImplementationOnce(() => new Promise((_resolve, rejectRead) => { reject = rejectRead }))
    fireEvent.click(retry)
    if (ending === "account") {
      switchAccount("bob")
      await screen.findByRole("button", { name: "Open cached history" })
      await waitFor(() => expect(io.data.get(storedKey("bob"))).toMatchObject({ ownerKey: ownerKey("bob") }))
    } else page.unmount()
    const writes = recoveryWrites().length
    const bob = structuredClone(io.data.get(storedKey("bob")))
    const revision = useSidepanelChatTabsStore.getState().revision
    await act(async () => { reject(Object.assign(new Error("Expired Alice credentials"), { code: "service_prompt_scope_unresolved" })) })
    expect(recoveryWrites()).toHaveLength(writes)
    expect(io.data.get(storedKey("alice"))).toEqual(alice)
    expect(io.data.get(storedKey("bob"))).toEqual(bob)
    expect(useSidepanelChatTabsStore.getState().revision).toBe(revision)
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
    await waitFor(() => expect(screen.getByRole("button", { name: "Recover drafts" })).toBeEnabled())
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
    await waitFor(() => expect(screen.getByRole("button", { name: "Recover drafts" })).toBeEnabled())
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
    await waitFor(() => expect(screen.getByRole("button", { name: "Recover drafts" })).toBeEnabled())
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

  it.each([undefined, 'anthropic'])(
    'restores a saved tab provider %s without borrowing the departing pair',
    async (provider) => {
      const saved = tabsState();
      saved.snapshotsById['alice-one'].modelSettings.apiProvider = 'custom';
      saved.snapshotsById['alice-two'].modelSettings = {
        apiProvider: provider,
        systemPrompt: 'Destination instructions',
      };
      io.data.set(storedKey('alice'), saved);
      render(<Route />);
      await screen.findByText('alice first');
      expect(useStoreChatModelSettings.getState().apiProvider).toBe('custom');
      fireEvent.click(await screen.findByRole('button', { name: 'alice two' }));
      await screen.findByText('alice inactive');
      expect(useStoreChatModelSettings.getState().apiProvider).toBe(provider);
      expect(useStoreChatModelSettings.getState().systemPrompt).toBe('Destination instructions');
    }
  );

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
