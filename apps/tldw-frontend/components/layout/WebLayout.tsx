import React, { lazy, Suspense, useState, useContext, useEffect, useCallback } from "react"

import { Drawer, Tooltip } from "antd"
import { EraserIcon, XIcon } from "lucide-react"
import { IconButton } from "@/components/Common/IconButton"
import { useLocation, useNavigate } from "react-router-dom"
import { useTranslation } from "react-i18next"
import { useQueryClient } from "@tanstack/react-query"
import { useShallow } from "zustand/react/shallow"

import { classNames } from "@/libs/class-name"
import { PageAssistDatabase } from "@/db/dexie/chat"
import { useMessageOption } from "@/hooks/useMessageOption"
import {
  useChatShortcuts,
  useSidebarShortcuts,
  useQuickChatShortcuts
} from "@/hooks/keyboard/useKeyboardShortcuts"
import { useQuickChatStore } from "@/store/quick-chat"
import { useStoreMessageOption } from "@/store/option"
import { useLayoutUiStore } from "@/store/layout-ui"
import { useRouteTransitionStore } from "@/store/route-transition"
import { QuickChatHelperButton } from "@/components/Common/QuickChatHelper"
import { NotesDockHost } from "@/components/Common/NotesDock"
import {
  BuddyShellHost,
  BuddyShellRenderContextProvider
} from "@/components/Common/PersonaBuddy"
import { CurrentChatModelSettings } from "@/components/Common/Settings/CurrentChatModelSettings"
import { Sidebar } from "@/components/Option/Sidebar"
import { Header } from "@/components/Layouts/Header"
import { QuickIngestModalHost } from "@/components/Layouts/QuickIngestButton"
import { useMigration } from "@/hooks/useMigration"
import { useStorageMigrations } from "@/hooks/useStorageMigrations"
import { useLayoutEffectsOwner } from "@/hooks/useLayoutEffectsOwner"
import { useChatSidebar } from "@/hooks/useFeatureFlags"
import { useMobile } from "@/hooks/useMediaQuery"
import { useSetting } from "@/hooks/useSetting"
import { useServerOnline } from "@/hooks/useServerOnline"
import { ChatSidebar } from "@/components/Common/ChatSidebar"
import { EventOnlyHosts } from "@/components/Common/EventHosts"
import { PageAssistLoader } from "@/components/Common/PageAssistLoader"
import { setSettingsReturnTo } from "@/utils/settings-return"
import { WorkflowIntegrationHost } from "@/components/Common/Workflow"
import {
  VIEWPORT_CONSTRAINED_PATHS
} from "@/routes/route-paths"
import {
  CHAT_BACKGROUND_IMAGE_SETTING,
  HEADER_SHORTCUTS_EXPANDED_SETTING
} from "@/services/settings/ui-settings"
import { setSetting } from "@/services/settings/registry"
import {
  BACKEND_UNREACHABLE_EVENT,
  type BackendUnreachableDetail
} from "@/services/request-events"
import { useBackendRecoveryUi } from "@/components/Common/BackendRecoveryUiContext"
import { BackendUnavailableModalGate } from "@web/components/layout/BackendUnavailableModalGate"
import { getUnreadCount } from "@web/lib/api/notifications"
import { CommandPalette } from "@/components/Common/CommandPalette"
import {
  useConnectionActions,
  useConnectionState,
  useConnectionUxState
} from "@/hooks/useConnectionState"
import { ConnectionPhase } from "@/types/connection"

// Lazy-load Timeline to reduce initial bundle size (~1.2MB cytoscape)
const TimelineModal = lazy(() =>
  import("@/components/Timeline").then((m) => ({ default: m.TimelineModal }))
)

const KeyboardShortcutsModal = lazy(() =>
  import("@/components/Common/KeyboardShortcutsModal").then((m) => ({
    default: m.KeyboardShortcutsModal
  }))
)
import { useConfirmDanger } from "@/components/Common/confirm-danger"
import { DemoModeProvider, useDemoMode } from "@/context/demo-mode"

type OptionLayoutProps = {
  children: React.ReactNode
  hideHeader?: boolean
  hideSidebar?: boolean
  allowNestedHideHeader?: boolean
  allowNestedHideSidebar?: boolean
}

const SHORTCUT_LOADING_MIN_MS = 0
const SHORTCUT_LOADING_MAX_MS = 2500

const OptionLayoutInner: React.FC<OptionLayoutProps> = ({
  children,
  hideHeader = false,
  hideSidebar = false
}) => {
  const confirmDanger = useConfirmDanger()
  const isLayoutEffectsOwner = useLayoutEffectsOwner({ prefer: true })
  useStorageMigrations(isLayoutEffectsOwner)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const chatSidebarCollapsed = useLayoutUiStore(
    (state) => state.chatSidebarCollapsed
  )
  const setChatSidebarCollapsed = useLayoutUiStore(
    (state) => state.setChatSidebarCollapsed
  )
  const { t } = useTranslation(["option", "common", "settings"])
  const navigate = useNavigate()
  const [openModelSettings, setOpenModelSettings] = useState(false)
  const { isLoading: migrationLoading } = useMigration()
  const { demoEnabled } = useDemoMode()
  const [showChatSidebar] = useChatSidebar()
  const isMobileViewport = useMobile()
  useServerOnline()

  // Notification unread count for header bell
  const [notificationCount, setNotificationCount] = useState(0)
  useEffect(() => {
    if (demoEnabled) return
    let cancelled = false
    const poll = () => {
      getUnreadCount()
        .then((res) => { if (!cancelled) setNotificationCount(res?.unread_count ?? 0) })
        .catch(() => {})
    }
    poll()
    const id = setInterval(poll, 30_000)
    return () => { cancelled = true; clearInterval(id) }
  }, [demoEnabled])
  const handleOpenNotifications = useCallback(() => navigate("/notifications"), [navigate])
  const location = useLocation()
  const { historyId, serverChatId } = useStoreMessageOption((state) => ({
    historyId: state.historyId,
    serverChatId: state.serverChatId
  }))
  const mobileSidebarPathRef = React.useRef(location.pathname)
  const { phase, isConnected } = useConnectionState()
  const { checkOnce } = useConnectionActions()
  const { isChecking } = useConnectionUxState()
  const { fatalBackendRecoveryActive } = useBackendRecoveryUi()
  const [backendUnavailableDetail, setBackendUnavailableDetail] =
    useState<BackendUnreachableDetail | null>(null)
  const suppressBackendUnavailableModal = React.useMemo(() => {
    if (typeof window === "undefined") return false
    try {
      return window.localStorage.getItem("__tldw_test_bypass") === "true"
    } catch {
      return false
    }
  }, [])
  const [chatBackgroundImage] = useSetting(CHAT_BACKGROUND_IMAGE_SETTING)
  const isChatScreen = location.pathname === "/chat"
  const isViewportConstrainedRoute = (VIEWPORT_CONSTRAINED_PATHS as readonly string[]).includes(location.pathname)
  const chatScreenBackgroundStyle =
    isChatScreen && chatBackgroundImage
      ? {
          backgroundImage: `url(${chatBackgroundImage})`,
          backgroundSize: "cover",
          backgroundPosition: "center",
          backgroundRepeat: "no-repeat"
        }
      : undefined
  const { clearChat, useOCR, chatMode, setChatMode, webSearch, setWebSearch } =
    useMessageOption()
  const queryClient = useQueryClient()
  const {
    active: shortcutLoading,
    pendingPath: shortcutPendingPath,
    startedAt: shortcutStartedAt,
    stop: stopShortcutLoading
  } = useRouteTransitionStore(
    useShallow((state) => ({
      active: state.active,
      pendingPath: state.pendingPath,
      startedAt: state.startedAt,
      stop: state.stop
    }))
  )

  React.useEffect(() => {
    if (!isLayoutEffectsOwner) return
    const path = `${location.pathname}${location.search}${location.hash}`
    setSettingsReturnTo(path, { historyId, serverChatId })
  }, [
    historyId,
    isLayoutEffectsOwner,
    location.hash,
    location.pathname,
    location.search,
    serverChatId
  ])

  React.useEffect(() => {
    setChatSidebarCollapsed(true)
    void setSetting(HEADER_SHORTCUTS_EXPANDED_SETTING, false).catch(() => {
      // ignore storage write failures
    })
  }, [location.pathname, setChatSidebarCollapsed])

  React.useEffect(() => {
    if (typeof window === "undefined") return
    const onBackendUnreachable = (event: Event) => {
      if (suppressBackendUnavailableModal) return
      const detail = (event as CustomEvent<BackendUnreachableDetail | undefined>)
        ?.detail
      if (!detail || typeof detail !== "object") return
      setBackendUnavailableDetail(detail)
      void checkOnce().catch(() => undefined)
    }

    window.addEventListener(
      BACKEND_UNREACHABLE_EVENT,
      onBackendUnreachable as EventListener
    )
    return () => {
      window.removeEventListener(
        BACKEND_UNREACHABLE_EVENT,
        onBackendUnreachable as EventListener
      )
    }
  }, [checkOnce, suppressBackendUnavailableModal])

  React.useEffect(() => {
    if (isConnected && phase === ConnectionPhase.CONNECTED) {
      setBackendUnavailableDetail(null)
    }
  }, [isConnected, phase])

  const closeBackendUnavailableModal = React.useCallback(() => {
    setBackendUnavailableDetail(null)
  }, [])

  const openHealthDiagnostics = React.useCallback(() => {
    setBackendUnavailableDetail(null)
    navigate("/settings/health")
  }, [navigate])

  const retryConnectionCheck = React.useCallback(() => {
    void checkOnce().catch(() => undefined)
  }, [checkOnce])

  // Create toggle function for sidebar
  const toggleSidebar = () => {
    if (hideSidebar) return
    if (showChatSidebar && !hideHeader) {
      if (isMobileViewport) {
        setSidebarOpen((prev) => !prev)
        return
      }
      setChatSidebarCollapsed((prev) => !prev)
      return
    }
    setSidebarOpen((prev) => !prev)
  }

  React.useEffect(() => {
    if (isMobileViewport && !showChatSidebar) return
    if (!isMobileViewport && sidebarOpen) {
      setSidebarOpen(false)
    }
  }, [isMobileViewport, showChatSidebar, sidebarOpen])

  React.useEffect(() => {
    if (!isMobileViewport || !showChatSidebar) {
      mobileSidebarPathRef.current = location.pathname
      return
    }
    if (location.pathname !== mobileSidebarPathRef.current && sidebarOpen) {
      setSidebarOpen(false)
    }
    mobileSidebarPathRef.current = location.pathname
  }, [isMobileViewport, showChatSidebar, sidebarOpen, location.pathname])

  const handleIngestPage = () => {
    if (typeof window !== "undefined") {
      window.dispatchEvent(new CustomEvent("tldw:open-quick-ingest"))
    }
  }

  const commandPaletteProps = {
    onNewChat: clearChat,
    onToggleRag: () => setChatMode(chatMode === "rag" ? "normal" : "rag"),
    onToggleWebSearch: () => setWebSearch(!webSearch),
    onIngestPage: handleIngestPage,
    onSwitchModel: () => setOpenModelSettings(true),
    onToggleSidebar: toggleSidebar
  }

  // Quick Chat Helper toggle
  const { isOpen: quickChatOpen, setIsOpen: setQuickChatOpen } = useQuickChatStore()
  const toggleQuickChat = () => {
    setQuickChatOpen(!quickChatOpen)
  }

  // Initialize shortcuts
  useChatShortcuts(clearChat, true)
  useSidebarShortcuts(toggleSidebar, true)
  useQuickChatShortcuts(toggleQuickChat, true)

  React.useEffect(() => {
    if (!shortcutLoading) return
    if (
      shortcutPendingPath &&
      shortcutPendingPath !== location.pathname
    ) {
      return
    }
    const elapsed = shortcutStartedAt ? Date.now() - shortcutStartedAt : 0
    const delay = Math.max(SHORTCUT_LOADING_MIN_MS - elapsed, 0)
    if (delay <= 0) {
      stopShortcutLoading()
      return
    }
    const timeoutId = window.setTimeout(() => {
      stopShortcutLoading()
    }, delay)
    return () => window.clearTimeout(timeoutId)
  }, [
    shortcutLoading,
    shortcutPendingPath,
    location.pathname,
    shortcutStartedAt,
    stopShortcutLoading
  ])

  React.useEffect(() => {
    if (!shortcutLoading) return
    const timeoutId = window.setTimeout(() => {
      stopShortcutLoading()
    }, SHORTCUT_LOADING_MAX_MS)
    return () => window.clearTimeout(timeoutId)
  }, [shortcutLoading, stopShortcutLoading])



  if (migrationLoading) {
    return (
      <div className="flex h-screen w-full items-center justify-center bg-bg ">
        <div className="text-center space-y-2">
          <div className="text-base font-medium text-text ">
            Migrating your chat history…
          </div>
          <div className="text-xs text-text-muted ">
            This runs once after an update and will reload the extension when finished.
          </div>
        </div>
      </div>
    )
  }

  const renderShortcutOverlay = () => (
    <div className="pointer-events-none absolute inset-0 z-30 flex justify-center px-4 sm:px-6 lg:px-8">
      <div className="pointer-events-auto relative w-full max-w-6xl">
        <div className="absolute inset-0 rounded-2xl bg-bg/70 backdrop-blur-[1px]">
          <PageAssistLoader
            label="Loading..."
            fullScreen={false}
            autoFocus={false}
          />
        </div>
      </div>
    </div>
  )

  return (
    <BuddyShellRenderContextProvider>
      <div
        className={classNames(
          "relative flex w-full",
          isViewportConstrainedRoute ? "h-screen min-h-0" : "min-h-screen"
        )}
        style={chatScreenBackgroundStyle}
      >
      {/* Persistent ChatSidebar when feature flag enabled */}
      {showChatSidebar && !hideHeader && !hideSidebar && !isMobileViewport && (
        <ChatSidebar
          collapsed={chatSidebarCollapsed}
          onToggleCollapse={() => setChatSidebarCollapsed((prev) => !prev)}
          className="sticky top-0 shrink-0 border-r border-border"
        />
      )}
      <main
        className={classNames(
          "relative flex-1 flex flex-col",
          hideHeader ? "bg-bg " : ""
        )}
        data-demo-mode={demoEnabled ? "on" : "off"}>
        {hideHeader ? (
          <div className="relative flex min-h-screen flex-1 flex-col items-center justify-center px-4 py-10 sm:px-8 overflow-auto">
            {children}
            {shortcutLoading && renderShortcutOverlay()}
          </div>
        ) : isViewportConstrainedRoute ? (
          <div className="relative flex min-h-0 flex-1 flex-col overflow-y-auto">
            <div className="relative z-20 w-full shrink-0">
              <Header
                onToggleSidebar={hideSidebar ? undefined : toggleSidebar}
                sidebarCollapsed={
                  showChatSidebar && isMobileViewport
                    ? !sidebarOpen
                    : chatSidebarCollapsed
                }
                notificationCount={notificationCount}
                onOpenNotifications={handleOpenNotifications}
              />
            </div>
            <div className="relative flex min-h-0 flex-1 flex-col">
              {children}
              {shortcutLoading && renderShortcutOverlay()}
            </div>
          </div>
        ) : (
          <div className="relative flex flex-col min-h-[135vh]">
            <div className="relative z-20 w-full">
              <Header
                onToggleSidebar={hideSidebar ? undefined : toggleSidebar}
                sidebarCollapsed={
                  showChatSidebar && isMobileViewport
                    ? !sidebarOpen
                    : chatSidebarCollapsed
                }
                notificationCount={notificationCount}
                onOpenNotifications={handleOpenNotifications}
              />
            </div>
            <div className="relative flex min-h-0 flex-1 flex-col">
              {children}
              {shortcutLoading && renderShortcutOverlay()}
            </div>
          </div>
        )}
        {/* Mobile Drawer for ChatSidebar when the persistent sidebar is enabled */}
        {!hideHeader && showChatSidebar && !hideSidebar && isMobileViewport && (
          <Drawer
            title={
              <div className="flex items-center justify-between">
                <span>{t("common:chatSidebar.title", "Chats")}</span>
                <IconButton
                  onClick={() => setSidebarOpen(false)}
                  ariaLabel={t("common:close", { defaultValue: "Close" }) as string}
                  title={t("common:close", { defaultValue: "Close" }) as string}
                  className="h-10 w-10 sm:h-7 sm:w-7 sm:min-h-0 sm:min-w-0"
                >
                  <XIcon className="h-5 w-5 text-text-muted" />
                </IconButton>
              </div>
            }
            placement="left"
            closeIcon={null}
            onClose={() => setSidebarOpen(false)}
            open={sidebarOpen}
          >
            <ChatSidebar
              collapsed={false}
              onToggleCollapse={() => setSidebarOpen(false)}
            />
          </Drawer>
        )}
        {/* Legacy Drawer sidebar - only shown when new ChatSidebar feature is disabled */}
        {!hideHeader && !showChatSidebar && !hideSidebar && (
          <Drawer
            title={
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <IconButton
                    onClick={() => setSidebarOpen(false)}
                    ariaLabel={t('common:close', { defaultValue: 'Close' }) as string}
                    title={t('common:close', { defaultValue: 'Close' }) as string}
                    className="-ml-1 h-11 w-11 sm:h-7 sm:w-7 sm:min-w-0 sm:min-h-0">
                    <XIcon className="h-5 w-5 text-text-muted " />
                  </IconButton>
                  <span>{t("sidebarTitle")}</span>
                </div>

                <div className="flex items-center space-x-3">
                  <Tooltip
                    title={t(
                      "settings:generalSettings.systemData.deleteChatHistory.label",
                      { defaultValue: t("settings:generalSettings.system.deleteChatHistory.label") as string }
                    )}
                    placement="left">
                    <IconButton
                      ariaLabel={t(
                        "settings:generalSettings.systemData.deleteChatHistory.label",
                        { defaultValue: t("settings:generalSettings.system.deleteChatHistory.label") as string }
                      ) as string}
                      onClick={async () => {
                        const ok = await confirmDanger({
                          title: t("common:confirmTitle", {
                            defaultValue: "Please confirm"
                          }),
                          content: t(
                            "settings:generalSettings.systemData.deleteChatHistory.confirm",
                            {
                              defaultValue: t(
                                "settings:generalSettings.system.deleteChatHistory.confirm"
                              ) as string
                            }
                          ),
                          okText: t("common:delete", { defaultValue: "Delete" }),
                          cancelText: t("common:cancel", { defaultValue: "Cancel" })
                        })

                        if (!ok) return

                        const db = new PageAssistDatabase()
                        await db.deleteAllChatHistory()
                        await queryClient.invalidateQueries({
                          queryKey: ["fetchChatHistory"]
                        })
                        clearChat()
                      }}
                      className="text-text-muted hover:text-text h-11 w-11 sm:h-7 sm:w-7 sm:min-w-0 sm:min-h-0">
                      <EraserIcon className="size-5" />
                    </IconButton>
                  </Tooltip>
                </div>
              </div>
            }
            placement="left"
            closeIcon={null}
          onClose={() => setSidebarOpen(false)}
          open={sidebarOpen}>
          <Sidebar
            isOpen={sidebarOpen}
            onClose={() => setSidebarOpen(false)}
          />
        </Drawer>
        )}

        {!hideHeader && (
          <CurrentChatModelSettings
            open={openModelSettings}
            setOpen={setOpenModelSettings}
            useDrawer
            isOCREnabled={useOCR}
          />
        )}

        {/* Quick Chat Helper floating button (legacy layout only) */}
        {!hideHeader && !showChatSidebar && !hideSidebar && (
          <QuickChatHelperButton />
        )}

        {/* Timeline Modal - lazy-loaded */}
        {!hideHeader && (
          <Suspense fallback={null}>
            <TimelineModal />
          </Suspense>
        )}

        {/* Command Palette - global keyboard shortcut ⌘K */}
        {!hideHeader && (
          <CommandPalette
            {...commandPaletteProps}
          />
        )}

        {/* Keyboard Shortcuts Help Modal - triggered by ? */}
        {!hideHeader && (
          <Suspense fallback={null}>
            <KeyboardShortcutsModal />
          </Suspense>
        )}

        {/* Quick Ingest Modal Host - listens for global open events */}
        <QuickIngestModalHost />

        {/* Notes Dock Host - floating notes panel */}
        <NotesDockHost />

        <BuddyShellHost root="web" />

        {/* Ensure event-driven modals are available even when the header is hidden */}
        {hideHeader && <EventOnlyHosts commandPaletteProps={commandPaletteProps} />}

        {/* Workflow landing modal + active workflow overlay */}
        <WorkflowIntegrationHost autoShowPaths={["/"]} />

        <BackendUnavailableModalGate
          backendUnavailableDetail={backendUnavailableDetail}
          fatalBackendRecoveryActive={fatalBackendRecoveryActive}
          isChecking={isChecking}
          onClose={closeBackendUnavailableModal}
          onConsumeHiddenDetail={closeBackendUnavailableModal}
          onOpenHealth={openHealthDiagnostics}
          onRetry={retryConnectionCheck}
          t={t}
        />
      </main>
    </div>
    </BuddyShellRenderContextProvider>
  )
}

type LayoutShellOverrides = {
  hideHeader?: boolean
  hideSidebar?: boolean
  sourcePath?: string
}

type LayoutShellContextValue = {
  inShell: boolean
  setOverrides?: (overrides: LayoutShellOverrides | null) => void
}

type LayoutShellGlobal = {
  mounted: boolean
  ownerId?: string
  setOverrides?: (overrides: LayoutShellOverrides | null) => void
}

const LayoutShellContext = React.createContext<LayoutShellContextValue>({
  inShell: false
})

const getGlobalShell = (): LayoutShellGlobal | null => {
  if (typeof globalThis === "undefined") return null
  const scope = globalThis as typeof globalThis & {
    __tldwOptionShell?: LayoutShellGlobal
  }
  if (!scope.__tldwOptionShell) {
    scope.__tldwOptionShell = { mounted: false }
  }
  return scope.__tldwOptionShell
}

/**
 * Component for when OptionLayout is nested inside an existing shell.
 * Extracted to satisfy React's Rules of Hooks (hooks must not be conditional).
 */
function NestedLayoutContent({
  props,
  shell,
  globalShell
}: {
  props: OptionLayoutProps
  shell: LayoutShellContextValue
  globalShell: LayoutShellGlobal | null
}) {
  const location = useLocation()
  const requestedOverrides = React.useMemo(() => {
    const overrides: LayoutShellOverrides = {}
    if (props.hideHeader) overrides.hideHeader = true
    if (props.hideSidebar) overrides.hideSidebar = true
    if (Object.keys(overrides).length === 0) return null
    overrides.sourcePath = location.pathname
    return overrides
  }, [location.pathname, props.hideHeader, props.hideSidebar])

  React.useEffect(() => {
    const setOverrides = shell.setOverrides || globalShell?.setOverrides
    if (!setOverrides || !requestedOverrides) return
    setOverrides(requestedOverrides)
    return () => setOverrides?.(null)
  }, [globalShell?.setOverrides, requestedOverrides, shell.setOverrides])

  return <>{props.children}</>
}

/**
 * Component for when OptionLayout is the root shell.
 * Extracted to satisfy React's Rules of Hooks (hooks must not be conditional).
 */
function RootLayoutShell({
  props,
  globalShell,
  ownerId
}: {
  props: OptionLayoutProps
  globalShell: LayoutShellGlobal | null
  ownerId: string
}) {
  const [overrides, setOverrides] = React.useState<LayoutShellOverrides | null>(
    null
  )
  const location = useLocation()
  const allowNestedHideHeader = props.allowNestedHideHeader !== false
  const allowNestedHideSidebar = props.allowNestedHideSidebar !== false
  const overridesMatch =
    !overrides?.sourcePath || overrides.sourcePath === location.pathname
  const effectiveHideHeader =
    (allowNestedHideHeader && overridesMatch && overrides?.hideHeader) ||
    props.hideHeader ||
    false
  const effectiveHideSidebar =
    (allowNestedHideSidebar && overridesMatch && overrides?.hideSidebar) ||
    props.hideSidebar ||
    false

  if (globalShell) {
    globalShell.mounted = true
    globalShell.ownerId = ownerId
    globalShell.setOverrides = setOverrides
  }

  React.useEffect(() => {
    if (!globalShell) return
    return () => {
      if (
        globalShell.setOverrides === setOverrides &&
        globalShell.ownerId === ownerId
      ) {
        globalShell.setOverrides = undefined
        globalShell.mounted = false
        globalShell.ownerId = undefined
      }
    }
  }, [globalShell, ownerId, setOverrides])

  React.useEffect(() => {
    if (!overrides?.sourcePath) return
    if (overrides.sourcePath !== location.pathname) {
      setOverrides(null)
    }
  }, [location.pathname, overrides?.sourcePath])

  return (
    <DemoModeProvider>
      <LayoutShellContext.Provider value={{ inShell: true, setOverrides }}>
        <OptionLayoutInner
          {...props}
          hideHeader={effectiveHideHeader}
          hideSidebar={effectiveHideSidebar}
        />
      </LayoutShellContext.Provider>
    </DemoModeProvider>
  )
}

export default function OptionLayout(props: OptionLayoutProps) {
  const shell = useContext(LayoutShellContext)
  const globalShell = getGlobalShell()
  const ownerId = React.useId()
  const externalShell =
    Boolean(globalShell?.mounted) &&
    (globalShell?.ownerId == null || globalShell.ownerId !== ownerId)

  // Check if we're nested inside an existing shell
  if (shell.inShell || externalShell) {
    return <NestedLayoutContent props={props} shell={shell} globalShell={globalShell} />
  }

  // We're the root shell
  return (
    <RootLayoutShell
      props={props}
      globalShell={globalShell}
      ownerId={ownerId}
    />
  )
}
