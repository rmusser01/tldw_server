import React from "react"
import { Button, Space, Tabs, Tooltip } from "antd"
import { HelpCircle } from "lucide-react"
import { useTranslation } from "react-i18next"
import { useLocation, useNavigate } from "react-router-dom"
import { ReviewTab, ManageTab, ImportExportTab, SchedulerTab, TemplatesTab } from "./tabs"
import { KeyboardShortcutsModal } from "./components"
import { useDecksQuery } from "./hooks"
import type { Flashcard } from "@/services/flashcards"
import { parseFlashcardsGenerateIntentFromLocation } from "@/services/tldw/flashcards-generate-handoff"
import { parseStudyPackIntentFromLocation } from "@/services/tldw/study-pack-handoff"
import {
  buildQuizAssessmentRouteFromFlashcards,
  parseFlashcardsStudyIntentFromLocation
} from "@/services/tldw/quiz-flashcards-handoff"

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
  const { data: initialDecks } = useDecksQuery({
    includeWorkspaceItems: currentStudyIntent?.forceShowWorkspaceItems ?? false
  })
  const showSchedulerTab = initialDecks === undefined || initialDecks.length > 0
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
  const [schedulerDeckHandoff, setSchedulerDeckHandoff] = React.useState<{
    deckId: number
    key: string
  } | null>(null)
  const [exportDeckHandoff, setExportDeckHandoff] = React.useState<{
    deckId: number
    key: string
  } | null>(null)
  const deckHandoffCounterRef = React.useRef(0)
  const nextDeckHandoffKey = React.useCallback((prefix: string, deckId: number) => {
    deckHandoffCounterRef.current += 1
    return `${prefix}:${deckId}:${deckHandoffCounterRef.current}`
  }, [])
  const clearDeckHandoffs = React.useCallback(() => {
    setManageDeckHandoff(null)
    setSchedulerDeckHandoff(null)
    setExportDeckHandoff(null)
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
      setActiveTab(nextTab)
    }
    if (currentStudyIntent?.deckId !== undefined) {
      applyReviewDeckChange(currentStudyIntent.deckId ?? undefined)
    }
  }, [applyReviewDeckChange, currentGenerateIntent, currentStudyIntent?.deckId, currentStudyPackIntent, currentTab])

  React.useEffect(() => {
    if (!showSchedulerTab && activeTab === "scheduler") {
      setActiveTab("review")
    }
  }, [activeTab, showSchedulerTab])

  const handleReviewCard = React.useCallback(
    (card: Flashcard) => {
      applyReviewDeckChange(card.deck_id ?? undefined)
      setReviewOverrideCard(card)
      setActiveTab("review")
    },
    [applyReviewDeckChange]
  )

  const routeToCreateEntryPoint = React.useCallback(() => {
    setActiveTab("cards")
    setOpenCreateSignal((prev) => prev + 1)
  }, [])
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
  const navigateToExportDeck = React.useCallback(
    (deckId: number) => {
      applyReviewDeckChange(deckId)
      setExportDeckHandoff({
        deckId,
        key: nextDeckHandoffKey("export", deckId)
      })
      setActiveTab("importExport")
    },
    [applyReviewDeckChange, nextDeckHandoffKey]
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
  const canOpenQuizCta = currentStudyIntent?.quizId !== undefined
  const effectiveActiveTab =
    !showSchedulerTab && activeTab === "scheduler" ? "review" : activeTab
  const effectiveSchedulerDeckId = schedulerDeckHandoff?.deckId ?? schedulerHandoffDeckId
  const effectiveSchedulerHandoffKey =
    schedulerDeckHandoff?.key ?? schedulerHandoffKey

  const handleTabChange = React.useCallback(
    (nextTab: string) => {
      if (nextTab === "scheduler" && !showSchedulerTab) return

      if (activeTab === "scheduler" && nextTab !== "scheduler" && schedulerDirty) {
        const shouldDiscard = window.confirm(
          t("option:flashcards.schedulerDiscardChangesPrompt", {
            defaultValue: "Discard unsaved scheduler changes?"
          })
        )
        if (!shouldDiscard) return
        setSchedulerDirty(false)
        setSchedulerDiscardSignal((current) => current + 1)
      }

      setActiveTab(nextTab)
    },
    [activeTab, schedulerDirty, showSchedulerTab, t]
  )

  return (
    <div className="mx-auto max-w-6xl p-4">
      <Tabs
        data-testid="flashcards-tabs"
        activeKey={effectiveActiveTab}
        onChange={handleTabChange}
        tabBarExtraContent={(
          <Space size={4}>
            <Tooltip
              title={
                canOpenQuizCta
                  ? undefined
                  : t("option:flashcards.quizCtaNeedsContext", {
                      defaultValue: "Open a Quiz-linked flashcard session before testing with Quiz."
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
          </Space>
        )}
        items={[
          {
            key: "review",
            label: t("option:flashcards.tabStudy", { defaultValue: "Study" }),
            children: (
              <ReviewTab
                onNavigateToCreate={routeToCreateEntryPoint}
                onNavigateToImport={() => setActiveTab("importExport")}
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
                onNavigateToImport={() => setActiveTab("importExport")}
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
              />
            )
          },
          {
            key: "importExport",
            label: t("option:flashcards.tabTransfer", { defaultValue: "Transfer" }),
            children: (
              <ImportExportTab
                generateIntent={currentGenerateIntent}
                studyPackIntent={currentStudyPackIntent}
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
          ...(showSchedulerTab
            ? [
                {
                  key: "scheduler",
                  label: t("option:flashcards.tabScheduler", { defaultValue: "Scheduler" }),
                  children: (
                    <SchedulerTab
                      isActive={effectiveActiveTab === "scheduler"}
                      initialDeckId={effectiveSchedulerDeckId}
                      initialDeckHandoffKey={effectiveSchedulerHandoffKey}
                      onDirtyChange={setSchedulerDirty}
                      discardSignal={schedulerDiscardSignal}
                    />
                  )
                }
              ]
            : [])
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
