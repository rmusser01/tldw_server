import React from "react"
import { Button, Tabs } from "antd"
import { useTranslation } from "react-i18next"
import {
  BarChartOutlined,
  EditOutlined,
  PlayCircleOutlined,
  SettingOutlined,
  ThunderboltOutlined
} from "@ant-design/icons"
import { TakeQuizTab } from "./tabs/TakeQuizTab"
import type { TakeTabNavigationIntent } from "./navigation"
import { RESULTS_FILTER_PREFS_KEY, TAKE_QUIZ_LIST_PREFS_KEY } from "./stateKeys"
import { useAttemptsQuery, useQuizzesQuery } from "./hooks"
import { parseQuizAssessmentIntentFromLocation } from "@/services/tldw/quiz-flashcards-handoff"

type QuizTabKey = "take" | "generate" | "create" | "manage" | "results"

const INITIAL_TAB_RESET_VERSION: Record<QuizTabKey, number> = {
  take: 0,
  generate: 0,
  create: 0,
  manage: 0,
  results: 0
}

const LazyGenerateTab = React.lazy(() =>
  import("./tabs/GenerateTab").then((module) => ({ default: module.GenerateTab }))
)
const LazyCreateTab = React.lazy(() =>
  import("./tabs/CreateTab").then((module) => ({ default: module.CreateTab }))
)
const LazyManageTab = React.lazy(() =>
  import("./tabs/ManageTab").then((module) => ({ default: module.ManageTab }))
)
const LazyResultsTab = React.lazy(() =>
  import("./tabs/ResultsTab").then((module) => ({ default: module.ResultsTab }))
)

/**
 * QuizPlayground contains all the tabs and core quiz logic.
 * Connection state is handled by QuizWorkspace.
 */
export const QuizPlayground: React.FC = () => {
  const { t } = useTranslation(["option", "common"])
  const initialAssessmentIntent = React.useMemo(
    () =>
      typeof window !== "undefined"
        ? parseQuizAssessmentIntentFromLocation(window.location)
        : null,
    []
  )
  const [activeTab, setActiveTab] = React.useState<QuizTabKey>("take")
  const [createTabDirty, setCreateTabDirty] = React.useState(false)
  const [takeTabIntent, setTakeTabIntent] = React.useState<TakeTabNavigationIntent | null>(() =>
    initialAssessmentIntent
      ? {
        startQuizId: initialAssessmentIntent.startQuizId ?? null,
        highlightQuizId:
            initialAssessmentIntent.highlightQuizId ??
            initialAssessmentIntent.startQuizId ??
            null,
        forceShowWorkspaceItems: initialAssessmentIntent.forceShowWorkspaceItems ?? false,
        sourceTab: initialAssessmentIntent.assignmentMode === "shared"
          ? "assignment"
          : (initialAssessmentIntent.deckId != null ||
              initialAssessmentIntent.deckName != null ||
              initialAssessmentIntent.sourceAttemptId != null)
            ? "flashcards"
            : null,
        attemptId: initialAssessmentIntent.sourceAttemptId ?? null,
        assignmentMode: initialAssessmentIntent.assignmentMode ?? null,
        assignmentDueAt: initialAssessmentIntent.assignmentDueAt ?? null,
        assignmentNote: initialAssessmentIntent.assignmentNote ?? null,
        assignedByRole: initialAssessmentIntent.assignedByRole ?? null
      }
      : null
  )
  const [globalSearchQuery, setGlobalSearchQuery] = React.useState("")
  const [takeSearchIntent, setTakeSearchIntent] = React.useState<{ query: string; token: number } | null>(null)
  const [manageSearchIntent, setManageSearchIntent] = React.useState<{ query: string; token: number } | null>(null)
  const [tabResetVersion, setTabResetVersion] = React.useState<Record<QuizTabKey, number>>(
    INITIAL_TAB_RESET_VERSION
  )
  const [loadedTabs, setLoadedTabs] = React.useState<Record<QuizTabKey, boolean>>({
    take: true,
    generate: false,
    create: false,
    manage: false,
    results: false
  })
  const searchTokenCounter = React.useRef(0)
  const tabsRef = React.useRef<HTMLDivElement | null>(null)

  React.useEffect(() => {
    if (!initialAssessmentIntent?.deckName) return
    searchTokenCounter.current += 1
    setTakeSearchIntent({
      query: initialAssessmentIntent.deckName,
      token: searchTokenCounter.current
    })
    setActiveTab("take")
  }, [initialAssessmentIntent?.deckName])

  React.useEffect(() => {
    setLoadedTabs((current) =>
      current[activeTab]
        ? current
        : {
          ...current,
          [activeTab]: true
        }
    )
  }, [activeTab])

  const { data: quizCounts } = useQuizzesQuery({ limit: 1, offset: 0 })
  const { data: attemptCounts } = useAttemptsQuery({ limit: 1, offset: 0 })
  const totalQuizzes = quizCounts?.count ?? 0
  const totalAttempts = attemptCounts?.count ?? 0

  // FTUX: default to "generate" tab when no quizzes exist yet
  const defaultTabResolved = React.useRef(false)
  React.useEffect(() => {
    if (defaultTabResolved.current) return
    if (quizCounts === undefined) return // still loading
    defaultTabResolved.current = true
    if (totalQuizzes === 0 && !initialAssessmentIntent) {
      setActiveTab("generate")
    }
  }, [quizCounts, totalQuizzes, initialAssessmentIntent])

  const renderTabLabel = React.useCallback((
    label: string,
    shortLabel: string,
    icon: React.ReactNode,
    count?: number
  ) => {
    const countBadge = typeof count === "number" && count >= 0 ? (
      <span aria-hidden className="rounded bg-surface2 px-1.5 py-0.5 text-xs text-text">
        {count}
      </span>
    ) : null

    return (
      <span className="inline-flex items-center gap-1.5" aria-label={label}>
        <span aria-hidden>{icon}</span>
        <span aria-hidden className="sm:hidden">{shortLabel}</span>
        <span aria-hidden className="hidden sm:inline">{label}</span>
        {countBadge}
      </span>
    )
  }, [])

  React.useEffect(() => {
    const root = tabsRef.current
    if (!root) return
    const activeTabNode = root.querySelector<HTMLElement>(".ant-tabs-tab-active")
    activeTabNode?.scrollIntoView({
      block: "nearest",
      inline: "center"
    })
  }, [activeTab])

  const navigateToTake = React.useCallback((intent?: TakeTabNavigationIntent) => {
    setTakeTabIntent({
      startQuizId: intent?.startQuizId ?? null,
      highlightQuizId: intent?.highlightQuizId ?? intent?.startQuizId ?? null,
      forceShowWorkspaceItems: intent?.forceShowWorkspaceItems ?? false,
      sourceTab: intent?.sourceTab ?? null,
      attemptId: intent?.attemptId ?? null,
      assignmentMode: intent?.assignmentMode ?? null,
      assignmentDueAt: intent?.assignmentDueAt ?? null,
      assignmentNote: intent?.assignmentNote ?? null,
      assignedByRole: intent?.assignedByRole ?? null
    })
    setActiveTab("take")
  }, [])

  const handleResetActiveTab = React.useCallback(() => {
    if (typeof window !== "undefined") {
      if (activeTab === "take") {
        window.sessionStorage.removeItem(TAKE_QUIZ_LIST_PREFS_KEY)
      }
      if (activeTab === "results") {
        window.sessionStorage.removeItem(RESULTS_FILTER_PREFS_KEY)
      }
    }
    if (activeTab === "take") {
      setTakeTabIntent(null)
    }
    if (activeTab === "create") {
      setCreateTabDirty(false)
    }
    setTabResetVersion((current) => ({
      ...current,
      [activeTab]: current[activeTab] + 1
    }))
  }, [activeTab])

  const handleTabChange = React.useCallback((nextTabRaw: string) => {
    const nextTab = nextTabRaw as QuizTabKey
    if (activeTab === "create" && nextTab !== "create" && createTabDirty) {
      const shouldLeave = window.confirm(
        t("option:quiz.unsavedCreateConfirm", {
          defaultValue: "You have unsaved quiz changes. Leave Create tab?"
        })
      )
      if (!shouldLeave) {
        return
      }
      setCreateTabDirty(false)
    }
    setLoadedTabs((current) =>
      current[nextTab]
        ? current
        : {
          ...current,
          [nextTab]: true
        }
    )
    setActiveTab(nextTab)
  }, [activeTab, createTabDirty, t])

  const renderLazyTab = React.useCallback(
    (tabKey: Exclude<QuizTabKey, "take">, content: React.ReactNode) => {
      if (!loadedTabs[tabKey] && activeTab !== tabKey) {
        return null
      }

      return (
        <React.Suspense fallback={null}>
          {content}
        </React.Suspense>
      )
    },
    [activeTab, loadedTabs]
  )

  const handleApplyGlobalSearch = React.useCallback(() => {
    const nextQuery = globalSearchQuery.trim()
    searchTokenCounter.current += 1
    const token = searchTokenCounter.current

    if (activeTab === "manage") {
      setManageSearchIntent({ query: nextQuery, token })
      return
    }

    setTakeSearchIntent({ query: nextQuery, token })
    setActiveTab("take")
  }, [activeTab, globalSearchQuery])

  return (
    <div className="mx-auto max-w-6xl p-4">
      <div className="mb-3 flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
        <div className="flex w-full gap-2 md:max-w-xl">
          <input
            type="text"
            className="w-full rounded border border-border bg-background px-3 py-1.5 text-sm"
            placeholder={t("option:quiz.globalSearchPlaceholder", { defaultValue: "Search quizzes across tabs..." })}
            value={globalSearchQuery}
            onChange={(event) => setGlobalSearchQuery(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                handleApplyGlobalSearch()
              }
            }}
            data-testid="quiz-global-search-input"
          />
          <Button onClick={handleApplyGlobalSearch} size="small" data-testid="quiz-global-search-apply">
            {t("common:search", { defaultValue: "Search" })}
          </Button>
        </div>
        {(totalQuizzes > 0 || totalAttempts > 0) && (
          <Button
            onClick={handleResetActiveTab}
            size="small"
            data-testid="quiz-reset-current-tab"
          >
            {t("option:quiz.resetCurrentTab", { defaultValue: "Reset Current Tab" })}
          </Button>
        )}
      </div>
      <div ref={tabsRef}>
        <Tabs
          data-testid="quiz-playground-tabs"
          className="quiz-tabs [&_.ant-tabs-nav-wrap]:overflow-x-auto [&_.ant-tabs-nav-wrap]:scroll-smooth [&_.ant-tabs-nav-list]:min-w-max [&_.ant-tabs-tab]:px-1 [&_.ant-tabs-tab-btn]:whitespace-nowrap"
          activeKey={activeTab}
          destroyInactiveTabPane={false}
          onChange={handleTabChange}
          items={[
          {
            key: "take",
            label: <span data-testid="quiz-tab-take">{renderTabLabel(
              t("option:quiz.take", { defaultValue: "Take Quiz" }),
              t("option:quiz.takeShort", { defaultValue: "Take" }),
              <PlayCircleOutlined />,
              totalQuizzes
            )}</span>,
            children: (
              <TakeQuizTab
                key={`take-${tabResetVersion.take}`}
                startQuizId={takeTabIntent?.startQuizId ?? null}
                highlightQuizId={takeTabIntent?.highlightQuizId ?? null}
                navigationSource={takeTabIntent?.sourceTab ?? null}
                assignmentMode={takeTabIntent?.assignmentMode ?? null}
                assignmentDueAt={takeTabIntent?.assignmentDueAt ?? null}
                assignmentNote={takeTabIntent?.assignmentNote ?? null}
                assignedByRole={takeTabIntent?.assignedByRole ?? null}
                externalSearchQuery={takeSearchIntent?.query ?? null}
                externalSearchToken={takeSearchIntent?.token ?? null}
                onStartHandled={() =>
                  setTakeTabIntent((current) =>
                    current
                      ? {
                        ...current,
                        startQuizId: null
                      }
                      : current
                  )
                }
                onHighlightHandled={() =>
                  setTakeTabIntent((current) =>
                    current
                      ? {
                        ...current,
                        highlightQuizId: null,
                        sourceTab: null,
                        attemptId: null
                      }
                      : current
                  )
                }
                onExternalSearchHandled={() => {
                  setTakeSearchIntent(null)
                }}
                onNavigateToGenerate={() => setActiveTab("generate")}
                onNavigateToCreate={() => setActiveTab("create")}
              />
            )
          },
          {
            key: "generate",
            label: <span data-testid="quiz-tab-generate">{renderTabLabel(
              quizCounts !== undefined && totalQuizzes === 0
                ? t("option:quiz.generateStartHere", { defaultValue: "Generate \u2190 Start here" })
                : t("option:quiz.generate", { defaultValue: "Generate" }),
              quizCounts !== undefined && totalQuizzes === 0
                ? t("option:quiz.generateStartHereShort", { defaultValue: "Gen \u2190" })
                : t("option:quiz.generateShort", { defaultValue: "Gen" }),
              <ThunderboltOutlined />
            )}</span>,
            children: (
              renderLazyTab(
                "generate",
                <LazyGenerateTab
                  key={`generate-${tabResetVersion.generate}`}
                  onNavigateToTake={(intent) => navigateToTake(intent)}
                  onNavigateToManage={() => setActiveTab("manage")}
                />
              )
            )
          },
          {
            key: "create",
            label: <span data-testid="quiz-tab-create">{renderTabLabel(
              t("option:quiz.create", { defaultValue: "Create" }),
              t("option:quiz.createShort", { defaultValue: "Create" }),
              <EditOutlined />
            )}</span>,
            children: (
              renderLazyTab(
                "create",
                <LazyCreateTab
                  key={`create-${tabResetVersion.create}`}
                  onDirtyStateChange={setCreateTabDirty}
                  onNavigateToTake={(intent) => navigateToTake(intent)}
                />
              )
            )
          },
          {
            key: "manage",
            label: <span data-testid="quiz-tab-manage">{renderTabLabel(
              t("option:quiz.manage", { defaultValue: "Manage" }),
              t("option:quiz.manageShort", { defaultValue: "Manage" }),
              <SettingOutlined />,
              totalQuizzes
            )}</span>,
            children: (
              renderLazyTab(
                "manage",
                <LazyManageTab
                  key={`manage-${tabResetVersion.manage}`}
                  externalSearchQuery={manageSearchIntent?.query ?? null}
                  externalSearchToken={manageSearchIntent?.token ?? null}
                  onExternalSearchHandled={() => {
                    setManageSearchIntent(null)
                  }}
                  onNavigateToCreate={() => setActiveTab("create")}
                  onNavigateToGenerate={() => setActiveTab("generate")}
                  onStartQuiz={(quizId) => {
                    navigateToTake({
                      startQuizId: quizId,
                      highlightQuizId: quizId,
                      sourceTab: "manage"
                    })
                  }}
                />
              )
            )
          },
          {
            key: "results",
            label: <span data-testid="quiz-tab-results">{renderTabLabel(
              t("option:quiz.results", { defaultValue: "Results" }),
              t("option:quiz.resultsShort", { defaultValue: "Results" }),
              <BarChartOutlined />,
              totalAttempts
            )}</span>,
            children: (
              renderLazyTab(
                "results",
                <LazyResultsTab
                  key={`results-${tabResetVersion.results}`}
                  onRetakeQuiz={(intent) => navigateToTake(intent)}
                />
              )
            )
          }
          ]}
        />
      </div>
    </div>
  )
}

export default QuizPlayground
