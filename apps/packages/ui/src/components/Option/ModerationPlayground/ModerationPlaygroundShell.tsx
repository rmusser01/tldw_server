import React from "react"
import { useTranslation } from "react-i18next"
import { message, Skeleton } from "antd"
import { useNavigate, useSearchParams } from "react-router-dom"
import { useServerOnline } from "@/hooks/useServerOnline"
import { useConnectionUxState } from "@/hooks/useConnectionState"
import { testModeration } from "@/services/moderation"
import { ModerationContextBar } from "./ModerationContextBar"
import { useModerationContext } from "./hooks/useModerationContext"
import { useModerationSettings } from "./hooks/useModerationSettings"
import { useBlocklist } from "./hooks/useBlocklist"
import { useUserOverrides } from "./hooks/useUserOverrides"
import { useModerationTest } from "./hooks/useModerationTest"
import { ONBOARDING_KEY, getErrorStatus } from "./moderation-utils"
import { useMilestoneStore } from "@/store/milestones"
import { useTutorialStore } from "@/store/tutorials"

// Lazy panel imports — replace with real components in Tasks 5-9
const PolicySettingsPanel = React.lazy(() => import("./PolicySettingsPanel"))
const BlocklistStudioPanel = React.lazy(() => import("./BlocklistStudioPanel"))
const UserOverridesPanel = React.lazy(() => import("./UserOverridesPanel"))
const TestSandboxPanel = React.lazy(() => import("./TestSandboxPanel"))
const AdvancedPanel = React.lazy(() => import("./AdvancedPanel"))

const TABS = [
  { key: "policy", label: "Policy & Settings" },
  { key: "blocklist", label: "Blocklist Studio" },
  { key: "overrides", label: "User Overrides" },
  { key: "test", label: "Test Sandbox" },
  { key: "advanced", label: "Advanced" }
] as const

type TabKey = (typeof TABS)[number]["key"]
const TAB_KEYS = new Set<TabKey>(TABS.map((tab) => tab.key))

const isTabKey = (value: string | null): value is TabKey =>
  Boolean(value && TAB_KEYS.has(value as TabKey))

const HERO_STYLE: React.CSSProperties = {
  background:
    "linear-gradient(180deg, var(--moderation-hero-start) 0%, var(--moderation-hero-end) 100%)",
  border: "1px solid var(--moderation-hero-border)",
  boxShadow: "0 24px 70px var(--moderation-hero-shadow)"
}
const HERO_GRID_STYLE: React.CSSProperties = {
  backgroundImage:
    "linear-gradient(var(--moderation-hero-grid-1) 1px, transparent 1px), linear-gradient(90deg, var(--moderation-hero-grid-2) 1px, transparent 1px)",
  backgroundSize: "28px 28px",
  opacity: "var(--moderation-hero-grid-opacity)"
}

export const ModerationPlaygroundShell: React.FC = () => {
  const { t } = useTranslation(["option", "common"])
  const online = useServerOnline()
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const { uxState } = useConnectionUxState()
  const [messageApi, contextHolder] = message.useMessage()
  const [activeTab, setActiveTab] = React.useState<TabKey>("policy")
  const requestedTab = searchParams.get("tab")
  const tabPanelBaseId = React.useId()
  const tabRefs = React.useRef<Record<TabKey, HTMLButtonElement | null>>({
    policy: null,
    blocklist: null,
    overrides: null,
    test: null,
    advanced: null
  })
  const markMilestone = useMilestoneStore((s) => s.markMilestone)
  const startTutorial = useTutorialStore((s) => s.startTutorial)
  const isTutorialCompleted = useTutorialStore((s) => s.isCompleted)
  const tutorialInitializedRef = React.useRef(false)

  const ctx = useModerationContext()
  const settings = useModerationSettings(ctx.activeUserId)
  const blocklist = useBlocklist()
  const overrides = useUserOverrides(ctx.activeUserId)
  const tester = useModerationTest()

  const policy = settings.policyQuery.data || {}
  const hasUnsavedChanges = settings.isDirty || overrides.isDirty || blocklist.isDirtyRaw

  const getTabId = (tab: TabKey) => `${tabPanelBaseId}-${tab}-tab`
  const getTabPanelId = (tab: TabKey) => `${tabPanelBaseId}-${tab}-panel`

  const activateTab = (tab: TabKey) => {
    setActiveTab(tab)
    tabRefs.current[tab]?.focus()
  }

  React.useEffect(() => {
    if (isTabKey(requestedTab) && requestedTab !== activeTab) {
      setActiveTab(requestedTab)
    }
  }, [activeTab, requestedTab])

  const handleTabKeyDown = (
    event: React.KeyboardEvent<HTMLButtonElement>,
    currentTab: TabKey
  ) => {
    const currentIndex = TABS.findIndex((tab) => tab.key === currentTab)
    let nextIndex: number | null = null
    if (event.key === "ArrowRight" || event.key === "ArrowDown") {
      nextIndex = (currentIndex + 1) % TABS.length
    } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
      nextIndex = (currentIndex - 1 + TABS.length) % TABS.length
    } else if (event.key === "Home") {
      nextIndex = 0
    } else if (event.key === "End") {
      nextIndex = TABS.length - 1
    }
    if (nextIndex == null) return
    event.preventDefault()
    activateTab(TABS[nextIndex].key)
  }

  // Authorization check — show error if backend returns 401/403
  const hasPermissionError = [settings.settingsQuery?.error, settings.policyQuery?.error, overrides.overridesQuery?.error]
    .map(getErrorStatus)
    .some((status) => status === 401 || status === 403)

  const [showOnboarding, setShowOnboarding] = React.useState(() => {
    if (typeof window === "undefined") return false
    return !localStorage.getItem(ONBOARDING_KEY)
  })
  const dismissOnboarding = () => {
    setShowOnboarding(false)
    if (typeof window !== "undefined") localStorage.setItem(ONBOARDING_KEY, "true")
  }

  React.useEffect(() => {
    if (!hasPermissionError) markMilestone("content_rules_reviewed")
  }, [markMilestone, hasPermissionError])

  React.useEffect(() => {
    if (activeTab === "test" && !hasPermissionError) {
      markMilestone("content_rules_tested")
    }
  }, [activeTab, markMilestone, hasPermissionError])

  // Ctrl+S save shortcut
  React.useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "s") {
        e.preventDefault()
        if (settings.isDirty)
          void settings
            .save()
            .then(() => messageApi.success("Settings saved"))
            .catch((err: any) => messageApi.error(err?.message || "Save failed"))
        if (overrides.isDirty)
          void overrides
            .save()
            .then(() => messageApi.success("Override saved"))
            .catch((err: any) => messageApi.error(err?.message || "Save failed"))
      }
    }
    window.addEventListener("keydown", handleKey)
    return () => window.removeEventListener("keydown", handleKey)
  }, [settings, overrides, messageApi])

  // Auto-start tutorial on first visit
  React.useEffect(() => {
    if (tutorialInitializedRef.current) return
    if (hasPermissionError) return
    tutorialInitializedRef.current = true
    if (typeof window === "undefined") return
    if (isTutorialCompleted("moderation-basics")) return
    try {
      const dismissed = window.localStorage.getItem(ONBOARDING_KEY)
      if (!dismissed) {
        startTutorial("moderation-basics")
      }
    } catch (err) {
      console.debug("[ModerationPlayground] localStorage unavailable, skipping tutorial auto-start", err)
    }
  }, [hasPermissionError, startTutorial, isTutorialCompleted])

  // beforeunload warning
  React.useEffect(() => {
    if (!hasUnsavedChanges) return
    const handler = (e: BeforeUnloadEvent) => {
      e.preventDefault()
      e.returnValue = ""
    }
    window.addEventListener("beforeunload", handler)
    return () => window.removeEventListener("beforeunload", handler)
  }, [hasUnsavedChanges])

  // Sync test userId from context — always track active user
  React.useEffect(() => {
    tester.setUserId(ctx.activeUserId ?? "")
  }, [ctx.activeUserId]) // eslint-disable-line react-hooks/exhaustive-deps

  const handleReload = async () => {
    try {
      await settings.reload()
      messageApi.success("Reloaded moderation config")
    } catch (err: any) {
      messageApi.error(err?.message || "Reload failed")
    }
  }

  const handleQuickTest = async (text: string, phase: "input" | "output") => {
    try {
      return await testModeration({
        user_id: ctx.activeUserId || undefined,
        phase,
        text
      })
    } catch (err: any) {
      messageApi.error(err?.message || "Test failed")
      return undefined
    }
  }

  const renderPanel = () => {
    switch (activeTab) {
      case "policy":
        return (
          <React.Suspense fallback={<div className="px-4 py-8"><Skeleton active paragraph={{ rows: 4 }} /></div>}>
            <PolicySettingsPanel settings={settings} messageApi={messageApi} />
          </React.Suspense>
        )
      case "blocklist":
        return (
          <React.Suspense fallback={<div className="px-4 py-8"><Skeleton active paragraph={{ rows: 4 }} /></div>}>
            <BlocklistStudioPanel blocklist={blocklist} messageApi={messageApi} />
          </React.Suspense>
        )
      case "overrides":
        return (
          <React.Suspense fallback={<div className="px-4 py-8"><Skeleton active paragraph={{ rows: 4 }} /></div>}>
            <UserOverridesPanel ctx={ctx} overrides={overrides} messageApi={messageApi} />
          </React.Suspense>
        )
      case "test":
        return (
          <React.Suspense fallback={<div className="px-4 py-8"><Skeleton active paragraph={{ rows: 4 }} /></div>}>
            <TestSandboxPanel tester={tester} messageApi={messageApi} />
          </React.Suspense>
        )
      case "advanced":
        return (
          <React.Suspense fallback={<div className="px-4 py-8"><Skeleton active paragraph={{ rows: 4 }} /></div>}>
            <AdvancedPanel
              settings={settings}
              blocklist={blocklist}
              overrides={overrides}
              messageApi={messageApi}
            />
          </React.Suspense>
        )
    }
  }

  const offlineWarning =
    !online && uxState !== "testing"
      ? uxState === "error_auth" || uxState === "configuring_auth"
        ? (
      <div className="mx-4 sm:mx-6 lg:mx-8 p-3 border border-yellow-200 dark:border-yellow-800 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg text-sm text-yellow-800 dark:text-yellow-300">
        <div>Add your credentials to use moderation controls.</div>
        <button
          type="button"
          onClick={() => navigate("/settings/tldw")}
          className="mt-2 text-sm font-medium underline underline-offset-2"
        >
          Open Settings
        </button>
      </div>
          )
        : uxState === "unconfigured" || uxState === "configuring_url"
          ? (
      <div className="mx-4 sm:mx-6 lg:mx-8 p-3 border border-yellow-200 dark:border-yellow-800 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg text-sm text-yellow-800 dark:text-yellow-300">
        <div>Finish setup to use moderation controls.</div>
        <button
          type="button"
          onClick={() => navigate("/")}
          className="mt-2 text-sm font-medium underline underline-offset-2"
        >
          Finish Setup
        </button>
      </div>
            )
          : uxState === "error_unreachable"
            ? (
      <div className="mx-4 sm:mx-6 lg:mx-8 p-3 border border-yellow-200 dark:border-yellow-800 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg text-sm text-yellow-800 dark:text-yellow-300">
        <div>Can&apos;t reach your tldw server right now.</div>
        <div className="mt-2 flex flex-wrap gap-2">
          <button
            type="button"
            onClick={() => navigate("/settings/health")}
            className="text-sm font-medium underline underline-offset-2"
          >
            Health & diagnostics
          </button>
          <button
            type="button"
            onClick={() => navigate("/settings/tldw")}
            className="text-sm font-medium underline underline-offset-2"
          >
            Open Settings
          </button>
        </div>
      </div>
              )
            : (
      <div className="mx-4 sm:mx-6 lg:mx-8 p-3 border border-yellow-200 dark:border-yellow-800 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg text-sm text-yellow-800 dark:text-yellow-300">
        Connect to your tldw server to use moderation controls.
      </div>
              )
      : null

  return (
    <div className="space-y-0">
      {contextHolder}

      {/* Authorization gate — block UI if 401/403 */}
      {hasPermissionError && (
        <div className="mx-4 sm:mx-6 lg:mx-8 mt-4 p-4 border border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-900/20 rounded-lg">
          <p className="text-sm font-semibold text-red-800 dark:text-red-300">Admin moderation access required</p>
          <p className="text-sm text-red-700 dark:text-red-400 mt-1">
            Moderation controls require an admin account with SYSTEM_CONFIGURE permission.
          </p>
        </div>
      )}

      {!hasPermissionError && <>
      {/* Onboarding */}
      {showOnboarding && (
        <div className="mx-4 sm:mx-6 lg:mx-8 mb-4 p-5 border border-blue-200 dark:border-blue-800 bg-blue-50 dark:bg-blue-900/20 rounded-xl">
            <p className="text-base font-semibold">
            {t("option:moderationPlayground.onboarding.title", "Welcome to Content Rules")}
          </p>
          <p className="text-sm text-text-muted mt-1">
            {t(
              "option:moderationPlayground.onboarding.description",
              "Set up content safety rules, blocklists, overrides, and test cases."
            )}
          </p>

          <div className="mt-3 flex flex-col gap-2 sm:flex-row sm:items-center sm:gap-4">
            <button
              type="button"
              data-testid="moderation-family-guardrails-link"
              onClick={() => navigate("/settings/family-guardrails")}
              className="inline-flex items-center gap-1.5 rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600"
            >
              {t("option:moderationPlayground.onboarding.guardrailsCta", "Set up Family Guardrails")}
            </button>
            <button
              type="button"
              onClick={dismissOnboarding}
              className="text-sm text-text-muted hover:text-text transition"
            >
              {t("option:moderationPlayground.onboarding.dismiss", "Skip \u2014 I\u2019ll explore on my own")}
            </button>
          </div>

          <p className="mt-3 text-xs text-text-muted">
            {t(
              "option:moderationPlayground.onboarding.tabHint",
              "Start here: Policy & Settings tab sets your base rules. Then test them in the Test Sandbox."
            )}
          </p>
        </div>
      )}

      {/* Hero */}
      <div
        data-testid="moderation-hero"
        className="relative overflow-hidden rounded-[28px] mx-4 sm:mx-6 lg:mx-8 p-6 sm:p-8 text-text"
        style={HERO_STYLE}
      >
        <div className="absolute inset-0" style={HERO_GRID_STYLE} />
        <div className="relative flex min-w-0 flex-wrap items-center justify-between gap-4">
          <div className="min-w-0 max-w-full">
            <h2 className="text-xl sm:text-2xl font-display font-bold">
              {t("option:moderationPlayground.title", "Content Rules")}
            </h2>
            <p className="text-text-muted text-sm mt-1 break-words">
              {t(
                "option:moderationPlayground.subtitle",
                "Configure moderation policies, blocklists, user overrides, and rule tests."
              )}
            </p>
            <div className="mt-2">
              <span
                className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${
                  online
                    ? "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300"
                    : "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300"
                }`}
              >
                {online ? "Server online" : "Server offline"}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Context bar */}
      <ModerationContextBar
        scope={ctx.scope}
        onScopeChange={ctx.setScope}
        userIdDraft={ctx.userIdDraft}
        onUserIdDraftChange={ctx.setUserIdDraft}
        onLoadUser={ctx.loadUser}
        activeUserId={ctx.activeUserId}
        onClearUser={ctx.clearUser}
        userLoading={overrides.loading}
        policy={policy}
        hasUnsavedChanges={hasUnsavedChanges}
        onReload={handleReload}
        onRunQuickTest={handleQuickTest}
        onOpenTestTab={() => setActiveTab("test")}
      />

      {/* Offline warning */}
      {offlineWarning}

      {/* Tab bar */}
      <div className="border-b border-border mx-4 sm:mx-6 lg:mx-8">
        <div className="flex max-w-full overflow-x-auto -mb-px" role="tablist" aria-label="Content rules sections">
          {TABS.map((tab) => (
            <button
              key={tab.key}
              id={getTabId(tab.key)}
              ref={(node) => {
                tabRefs.current[tab.key] = node
              }}
              data-testid={`moderation-tab-${tab.key}`}
              role="tab"
              aria-selected={activeTab === tab.key}
              aria-controls={activeTab === tab.key ? getTabPanelId(tab.key) : undefined}
              tabIndex={activeTab === tab.key ? 0 : -1}
              onClick={() => setActiveTab(tab.key)}
              onKeyDown={(event) => handleTabKeyDown(event, tab.key)}
              className={`
                px-4 py-2.5 text-sm font-medium whitespace-nowrap border-b-2 transition-colors
                ${
                  activeTab === tab.key
                    ? "border-blue-500 text-blue-600 dark:text-blue-400"
                    : "border-transparent text-text-muted hover:text-text hover:border-gray-300"
                }
              `}
            >
              {tab.label}
              {tab.key === "policy" && showOnboarding && (
                <span className="ml-1.5 rounded bg-blue-100 dark:bg-blue-900/40 px-1.5 py-0.5 text-[10px] font-semibold text-blue-700 dark:text-blue-300">
                  {t("option:moderationPlayground.onboarding.startHereBadge", "Start here")}
                </span>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Tab content */}
      <div
        id={getTabPanelId(activeTab)}
        role="tabpanel"
        aria-labelledby={getTabId(activeTab)}
        className="mx-4 min-w-0 sm:mx-6 lg:mx-8 py-6"
      >
        {renderPanel()}
      </div>
      </>}
    </div>
  )
}
