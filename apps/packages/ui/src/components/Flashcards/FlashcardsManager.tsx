import React from "react"
import { Button, Card, Empty, Space, Tabs, Tooltip, Typography } from "antd"
import { HelpCircle } from "lucide-react"
import { useTranslation } from "react-i18next"
import { useLocation, useNavigate } from "react-router-dom"
import {
  ReviewTab,
  ManageTab,
  ImportExportTab,
  SchedulerTab,
  TemplatesTab,
  type TransferTaskKey
} from "./tabs"
import { KeyboardShortcutsModal } from "./components"
import { useDecksQuery, type UseFlashcardQueriesOptions } from "./hooks"
import type { Flashcard } from "@/services/flashcards"
import { parseFlashcardsGenerateIntentFromLocation } from "@/services/tldw/flashcards-generate-handoff"
import { parseStudyPackIntentFromLocation } from "@/services/tldw/study-pack-handoff"
import {
  buildQuizAssessmentRouteFromFlashcards,
  parseFlashcardsStudyIntentFromLocation
} from "@/services/tldw/quiz-flashcards-handoff"

const { Text } = Typography

const parseInitialFlashcardsTab = (locationLike: { search?: string; hash?: string }): string | null => {
  const search = locationLike.search || ""
  const normalized = search.startsWith("?") ? search.slice(1) : search
  const params = new URLSearchParams(normalized)
  const rawTab = params.get("tab")?.trim().toLowerCase()

  switch (rawTab) {
    case "review":
    case "study":
      return "review"
    case "manage":
    case "cards":
      return "cards"
    case "transfer":
    case "importexport":
      return "importExport"
    case "template":
    case "templates":
      return "templates"
    case "scheduler":
      return "scheduler"
    default:
      return null
  }
}

/**
 * FlashcardsManager contains all the tabs and core flashcard logic.
 * Connection state is handled by FlashcardsWorkspace.
 *
 * Structure: Study | Manage | Transfer
 * - Study: Spaced repetition review and cram loops
 * - Manage: Browse, filter, create, edit, bulk operations
 * - Transfer: Import, generate, Study Pack, and export workflows
 * - Scheduler: Deck-level scheduler policy editing and queue visibility
 */
export const FlashcardsManager: React.FC = () => {
  const { t } = useTranslation(["option", "common"])
  const location = useLocation()
  const navigate = useNavigate()
  const currentGenerateIntent = React.useMemo(
    () => parseFlashcardsGenerateIntentFromLocation(location),
    [location]
  )
  const currentStudyPackIntent = React.useMemo(
    () => parseStudyPackIntentFromLocation(location),
    [location]
  )
  const currentStudyIntent = React.useMemo(
    () => parseFlashcardsStudyIntentFromLocation(location),
    [location]
  )
  const currentTab = React.useMemo(() => parseInitialFlashcardsTab(location), [location])
  const [activeTab, setActiveTab] = React.useState<string>(() =>
    currentTab ?? (currentGenerateIntent || currentStudyPackIntent ? "importExport" : "review")
  )
  const deckVisibilityOptions = React.useMemo<UseFlashcardQueriesOptions>(() => ({
    includeWorkspaceItems: currentStudyIntent?.forceShowWorkspaceItems ?? false
  }), [currentStudyIntent?.forceShowWorkspaceItems])
  const { data: initialDecks } = useDecksQuery(deckVisibilityOptions)
  const hasNoInitialDecks = initialDecks !== undefined && initialDecks.length === 0
  const [reviewDeckId, setReviewDeckId] = React.useState<number | null | undefined>(
    currentStudyIntent?.deckId ?? undefined
  )
  const schedulerHandoffDeckId =
    currentTab === "scheduler" && currentStudyIntent?.deckId != null
      ? currentStudyIntent.deckId
      : (reviewDeckId ?? null)
  const schedulerHandoffKey =
    currentTab === "scheduler" && currentStudyIntent?.deckId != null
      ? (location.key ?? `${location.pathname}:${location.search}:${location.hash}`)
      : schedulerHandoffDeckId != null
        ? `review:${schedulerHandoffDeckId}`
        : null
  const [reviewOverrideCard, setReviewOverrideCard] = React.useState<Flashcard | null>(null)
  const [openCreateSignal, setOpenCreateSignal] = React.useState(0)
  const [shortcutsModalOpen, setShortcutsModalOpen] = React.useState(false)
  const [schedulerDirty, setSchedulerDirty] = React.useState(false)
  const [schedulerDiscardSignal, setSchedulerDiscardSignal] = React.useState(0)
  const [manageDeckHandoff, setManageDeckHandoff] = React.useState<{
    deckId: number
    key: string
    showWorkspaceDecks: boolean
  } | null>(null)
  const [createDeckHandoff, setCreateDeckHandoff] = React.useState<{
    deckId: number | null
    showWorkspaceDecks: boolean
  } | null>(null)
  const [schedulerDeckHandoff, setSchedulerDeckHandoff] = React.useState<{
    deckId: number
    key: string
  } | null>(null)
  const [exportDeckHandoff, setExportDeckHandoff] = React.useState<{
    deckId: number
    key: string
  } | null>(null)
  const [transferTaskHandoff, setTransferTaskHandoff] = React.useState<{
    task: TransferTaskKey
    key: string
  } | null>(null)
  const deckHandoffCounterRef = React.useRef(0)
  const transferTaskHandoffCounterRef = React.useRef(0)
  const nextDeckHandoffKey = React.useCallback((prefix: string, deckId: number) => {
    deckHandoffCounterRef.current += 1
    return `${prefix}:${deckId}:${deckHandoffCounterRef.current}`
  }, [])
  const nextTransferTaskHandoffKey = React.useCallback((task: TransferTaskKey) => {
    transferTaskHandoffCounterRef.current += 1
    return `${task}:${transferTaskHandoffCounterRef.current}`
  }, [])
  const clearDeckHandoffs = React.useCallback(() => {
    setManageDeckHandoff(null)
    setCreateDeckHandoff(null)
    setSchedulerDeckHandoff(null)
    setExportDeckHandoff(null)
  }, [])
  const clearCreateDeckHandoff = React.useCallback(() => {
    setCreateDeckHandoff(null)
  }, [])
  const discardSchedulerChanges = React.useCallback(() => {
    setSchedulerDirty(false)
    setSchedulerDiscardSignal((current) => current + 1)
  }, [])
  const applyReviewDeckChange = React.useCallback(
    (deckId: number | null | undefined) => {
      setReviewDeckId(deckId)
      clearDeckHandoffs()
    },
    [clearDeckHandoffs]
  )

  // Listen for "?" key to open keyboard shortcuts modal
  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't trigger when typing in inputs
      const target = e.target as HTMLElement
      if (
        target.tagName === "INPUT" ||
        target.tagName === "TEXTAREA" ||
        target.isContentEditable
      ) {
        return
      }

      if (e.key === "?" || (e.shiftKey && e.key === "/")) {
        e.preventDefault()
        setShortcutsModalOpen(true)
      }
    }

    window.addEventListener("keydown", handleKeyDown)
    return () => window.removeEventListener("keydown", handleKeyDown)
  }, [])

  React.useEffect(() => {
    const nextTab =
      currentTab ?? (currentGenerateIntent || currentStudyPackIntent ? "importExport" : null)
    if (nextTab) {
      if (
        nextTab === "importExport" &&
        (currentTab === "importExport" || currentGenerateIntent || currentStudyPackIntent)
      ) {
        setTransferTaskHandoff(null)
      }
      setActiveTab(nextTab)
    }
    if (currentStudyIntent?.deckId !== undefined) {
      applyReviewDeckChange(currentStudyIntent.deckId ?? undefined)
    }
  }, [applyReviewDeckChange, currentGenerateIntent, currentStudyIntent?.deckId, currentStudyPackIntent, currentTab])

  React.useEffect(() => {
    if (hasNoInitialDecks && activeTab === "scheduler" && schedulerDirty) {
      discardSchedulerChanges()
    }
  }, [activeTab, discardSchedulerChanges, hasNoInitialDecks, schedulerDirty])

  const handleReviewCard = React.useCallback(
    (card: Flashcard) => {
      applyReviewDeckChange(card.deck_id ?? undefined)
      setReviewOverrideCard(card)
      setActiveTab("review")
    },
    [applyReviewDeckChange]
  )

  const routeToCreateEntryPoint = React.useCallback(() => {
    const createDeckId = reviewDeckId ?? null
    setCreateDeckHandoff({
      deckId: createDeckId,
      showWorkspaceDecks:
        createDeckId != null && currentStudyIntent?.forceShowWorkspaceItems === true
    })
    setActiveTab("cards")
    setOpenCreateSignal((prev) => prev + 1)
  }, [currentStudyIntent?.forceShowWorkspaceItems, reviewDeckId])
  const navigateToManageDeck = React.useCallback(
    (deckId: number) => {
      applyReviewDeckChange(deckId)
      setManageDeckHandoff({
        deckId,
        key: nextDeckHandoffKey("manage", deckId),
        showWorkspaceDecks: currentStudyIntent?.forceShowWorkspaceItems ?? false
      })
      setActiveTab("cards")
    },
    [applyReviewDeckChange, currentStudyIntent?.forceShowWorkspaceItems, nextDeckHandoffKey]
  )
  const navigateToSchedulerDeck = React.useCallback(
    (deckId: number) => {
      applyReviewDeckChange(deckId)
      setSchedulerDeckHandoff({
        deckId,
        key: nextDeckHandoffKey("scheduler", deckId)
      })
      setActiveTab("scheduler")
    },
    [applyReviewDeckChange, nextDeckHandoffKey]
  )
  const requestTransferTask = React.useCallback(
    (task: TransferTaskKey) => {
      setTransferTaskHandoff({
        task,
        key: nextTransferTaskHandoffKey(task)
      })
    },
    [nextTransferTaskHandoffKey]
  )
  const navigateToTransferTask = React.useCallback(
    (task: TransferTaskKey) => {
      clearDeckHandoffs()
      requestTransferTask(task)
      setActiveTab("importExport")
    },
    [clearDeckHandoffs, requestTransferTask]
  )
  const navigateToImportTask = React.useCallback(() => {
    navigateToTransferTask("import")
  }, [navigateToTransferTask])
  const navigateToGenerateTask = React.useCallback(() => {
    navigateToTransferTask("create")
  }, [navigateToTransferTask])
  const navigateToExportDeck = React.useCallback(
    (deckId: number) => {
      applyReviewDeckChange(deckId)
      setExportDeckHandoff({
        deckId,
        key: nextDeckHandoffKey("export", deckId)
      })
      requestTransferTask("export")
      setActiveTab("importExport")
    },
    [applyReviewDeckChange, nextDeckHandoffKey, requestTransferTask]
  )

  const quizCtaRoute = React.useMemo(() => {
    const startQuizId = currentStudyIntent?.quizId
    return buildQuizAssessmentRouteFromFlashcards({
      startQuizId,
      highlightQuizId: startQuizId,
      deckId: reviewDeckId ?? currentStudyIntent?.deckId,
      sourceAttemptId: currentStudyIntent?.attemptId,
      forceShowWorkspaceItems: currentStudyIntent?.forceShowWorkspaceItems ?? false
    })
  }, [currentStudyIntent?.attemptId, currentStudyIntent?.deckId, currentStudyIntent?.forceShowWorkspaceItems, currentStudyIntent?.quizId, reviewDeckId])
  const canOpenQuizCta =
    currentStudyIntent?.quizId !== undefined ||
    reviewDeckId != null ||
    currentStudyIntent?.deckId != null
  const effectiveActiveTab = activeTab
  const effectiveSchedulerDeckId = schedulerDeckHandoff?.deckId ?? schedulerHandoffDeckId
  const effectiveSchedulerHandoffKey =
    schedulerDeckHandoff?.key ?? schedulerHandoffKey

  const handleTabChange = React.useCallback(
    (nextTab: string) => {
      if (activeTab === "scheduler" && nextTab !== "scheduler" && schedulerDirty) {
        const shouldDiscard = window.confirm(
          t("option:flashcards.schedulerDiscardChangesPrompt", {
            defaultValue: "Discard unsaved scheduler changes?"
          })
        )
        if (!shouldDiscard) return
        discardSchedulerChanges()
      }

      if (nextTab === "importExport") {
        setTransferTaskHandoff(null)
      }

      setActiveTab(nextTab)
    },
    [activeTab, discardSchedulerChanges, schedulerDirty, t]
  )

  const schedulerEmptyPreview = (
    <Card size="small" data-testid="flashcards-scheduler-empty-preview">
      <Empty
        image={Empty.PRESENTED_IMAGE_SIMPLE}
        description={
          <Space orientation="vertical" size={8} align="center">
            <Text strong>
              {t("option:flashcards.schedulerEmptyTitle", {
                defaultValue: "Create a deck before tuning scheduler rules."
              })}
            </Text>
            <Text type="secondary" className="max-w-xl text-center">
              {t("option:flashcards.schedulerEmptyDescription", {
                defaultValue:
                  "Scheduler policies control review timing per deck. Start with a new deck, or import and generate cards first."
              })}
            </Text>
          </Space>
        }
      >
        <Space wrap>
          <Button type="primary" onClick={routeToCreateEntryPoint}>
            {t("option:flashcards.schedulerEmptyCreateDeck", {
              defaultValue: "Create a deck"
            })}
          </Button>
          <Button onClick={navigateToImportTask}>
            {t("option:flashcards.schedulerEmptyImportGenerate", {
              defaultValue: "Import or generate cards"
            })}
          </Button>
        </Space>
      </Empty>
    </Card>
  )

  return (
    <div className="mx-auto max-w-6xl p-4">
      <Tabs
        data-testid="flashcards-tabs"
        className="flashcards-responsive-tabs [&_.ant-tabs-extra-content]:min-w-0 [&_.ant-tabs-extra-content]:max-w-full [&_.ant-tabs-nav-list]:min-w-max [&_.ant-tabs-nav-wrap]:min-w-0 [&_.ant-tabs-nav-wrap]:overflow-x-auto"
        activeKey={effectiveActiveTab}
        onChange={handleTabChange}
        tabBarExtraContent={(
          <div
            className="flex min-w-0 max-w-full flex-wrap items-center justify-end gap-1 sm:flex-nowrap"
            data-testid="flashcards-tab-actions"
          >
            <Tooltip
              title={
                canOpenQuizCta
                  ? undefined
                  : t("option:flashcards.quizCtaNeedsContext", {
                      defaultValue: "Select a review deck before testing with Quiz."
                    })
              }
            >
              <span>
                <Button
                  size="small"
                  data-testid="flashcards-to-quiz-cta"
                  disabled={!canOpenQuizCta}
                  onClick={() => {
                    if (!canOpenQuizCta) return
                    navigate(quizCtaRoute)
                  }}
                >
                  {t("option:flashcards.testWithQuiz", {
                    defaultValue: "Test with Quiz"
                  })}
                </Button>
              </span>
            </Tooltip>
            <Tooltip
              title={t("option:flashcards.keyboardShortcutsHelp", {
                defaultValue: "Press ? to show shortcuts"
              })}
            >
              <Button
                type="text"
                size="small"
                icon={<HelpCircle className="size-4" />}
                onClick={() => setShortcutsModalOpen(true)}
                aria-label={t("option:flashcards.keyboardShortcutsTitle", {
                  defaultValue: "Keyboard Shortcuts"
                })}
              />
            </Tooltip>
          </div>
        )}
        items={[
          {
            key: "review",
            label: t("option:flashcards.tabStudy", { defaultValue: "Study" }),
            children: (
              <ReviewTab
                onNavigateToCreate={routeToCreateEntryPoint}
                onNavigateToImport={navigateToImportTask}
                onNavigateToGenerate={navigateToGenerateTask}
                reviewDeckId={reviewDeckId}
                onReviewDeckChange={applyReviewDeckChange}
                reviewOverrideCard={reviewOverrideCard}
                onClearOverride={() => setReviewOverrideCard(null)}
                isActive={effectiveActiveTab === "review"}
                forceShowWorkspaceItems={currentStudyIntent?.forceShowWorkspaceItems ?? false}
                onNavigateToManageDeck={navigateToManageDeck}
                onNavigateToSchedulerDeck={navigateToSchedulerDeck}
                onNavigateToExportDeck={navigateToExportDeck}
              />
            )
          },
          {
            key: "cards",
            label: t("option:flashcards.tabManage", { defaultValue: "Manage" }),
            children: (
              <ManageTab
                onNavigateToImport={navigateToImportTask}
                onNavigateToGenerate={navigateToGenerateTask}
                onReviewCard={handleReviewCard}
                openCreateSignal={openCreateSignal}
                isActive={effectiveActiveTab === "cards"}
                initialDeckId={
                  manageDeckHandoff?.deckId ??
                  (currentTab === "cards" ? currentStudyIntent?.deckId : undefined)
                }
                initialDeckHandoffKey={manageDeckHandoff?.key ?? null}
                initialShowWorkspaceDecks={
                  manageDeckHandoff?.showWorkspaceDecks ??
                  (currentTab === "cards" ? (currentStudyIntent?.forceShowWorkspaceItems ?? false) : false)
                }
                createInitialDeckId={createDeckHandoff?.deckId ?? null}
                createInitialShowWorkspaceDecks={createDeckHandoff?.showWorkspaceDecks ?? false}
                onCreateHandoffConsumed={clearCreateDeckHandoff}
              />
            )
          },
          {
            key: "importExport",
            label: t("option:flashcards.importExport", { defaultValue: "Import / Export" }),
            children: (
              <ImportExportTab
                generateIntent={currentGenerateIntent}
                studyPackIntent={currentStudyPackIntent}
                initialTask={transferTaskHandoff?.task ?? null}
                initialTaskHandoffKey={transferTaskHandoff?.key ?? null}
                initialExportDeckId={exportDeckHandoff?.deckId ?? null}
                initialExportDeckHandoffKey={exportDeckHandoff?.key ?? null}
              />
            )
          },
          {
            key: "templates",
            label: t("option:flashcards.tabTemplates", { defaultValue: "Templates" }),
            children: <TemplatesTab />
          },
          {
            key: "scheduler",
            label: t("option:flashcards.tabScheduler", { defaultValue: "Scheduler" }),
            children: hasNoInitialDecks ? (
              schedulerEmptyPreview
            ) : (
              <SchedulerTab
                isActive={effectiveActiveTab === "scheduler"}
                initialDeckId={effectiveSchedulerDeckId}
                initialDeckHandoffKey={effectiveSchedulerHandoffKey}
                deckVisibilityOptions={deckVisibilityOptions}
                onDirtyChange={setSchedulerDirty}
                discardSignal={schedulerDiscardSignal}
              />
            )
          }
        ]}
      />

      <KeyboardShortcutsModal
        open={shortcutsModalOpen}
        onClose={() => setShortcutsModalOpen(false)}
        activeTab={
          effectiveActiveTab === "importExport"
            ? "import"
            : effectiveActiveTab === "scheduler"
              ? "scheduler"
              : effectiveActiveTab === "templates"
                ? "templates"
                : (effectiveActiveTab as "review" | "cards")
        }
      />
    </div>
  )
}

export default FlashcardsManager
