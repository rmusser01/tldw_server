import React from "react"
import { Button, Input } from "antd"
import { useStorage } from "@plasmohq/storage/hook"
import {
  Check,
  Compass,
  GraduationCap,
  Lock,
  MessageCircleQuestion,
  Play,
  RotateCcw,
  Route
} from "lucide-react"
import { useTranslation } from "react-i18next"
import { useTutorialStore } from "../../../store/tutorials"
import { getTutorialsForRoute } from "../../../tutorials/registry"
import type { TutorialDefinition } from "../../../tutorials/registry"
import {
  QUICK_CHAT_WORKFLOW_GUIDES,
  QUICK_CHAT_WORKFLOW_GUIDES_STORAGE_KEY,
  filterQuickChatWorkflowGuides,
  normalizeQuickChatRoutePath,
  resolveQuickChatWorkflowGuides
} from "./workflow-guides"

type Props = {
  onAskGuide: (question: string) => void
  onOpenRoute: (route: string) => void
  askDisabled?: boolean
  currentRoute?: string | null
  onStartTutorial?: (tutorialId: string) => void
}

export type QuickChatPageTutorialEntry = {
  tutorial: TutorialDefinition
  isLocked: boolean
  isCompleted: boolean
}

export const buildQuickChatPageTutorialEntries = (
  currentRoute: string | null | undefined,
  completedTutorialIds: string[]
): QuickChatPageTutorialEntry[] => {
  const normalizedRoute = normalizeQuickChatRoutePath(currentRoute)
  if (!normalizedRoute) {
    return []
  }

  const tutorials = getTutorialsForRoute(normalizedRoute, {
    ignoreRuntimeSuppression: true,
    completedTutorialIds,
    includeLocked: true
  })
  const completedSet = new Set(completedTutorialIds)

  return tutorials.map((tutorial) => {
    const unmetPrereqs = (tutorial.prerequisites ?? []).filter(
      (prereqId) => !completedSet.has(prereqId)
    )
    const isLocked = unmetPrereqs.length > 0
    const isCompleted = completedSet.has(tutorial.id)
    return {
      tutorial,
      isLocked,
      isCompleted
    }
  })
}

export const QuickChatGuidesPanel: React.FC<Props> = ({
  onAskGuide,
  onOpenRoute,
  askDisabled = false,
  currentRoute = null,
  onStartTutorial
}) => {
  const { t } = useTranslation(["option", "common", "tutorials"])
  const [storedGuidesRaw] = useStorage<unknown>(
    QUICK_CHAT_WORKFLOW_GUIDES_STORAGE_KEY,
    QUICK_CHAT_WORKFLOW_GUIDES
  )
  const completedTutorials = useTutorialStore((state) => state.completedTutorials)
  const startTutorial = useTutorialStore((state) => state.startTutorial)
  const [search, setSearch] = React.useState("")
  const guides = React.useMemo(
    () => resolveQuickChatWorkflowGuides(storedGuidesRaw),
    [storedGuidesRaw]
  )
  const filteredGuides = React.useMemo(
    () => filterQuickChatWorkflowGuides(search, guides),
    [guides, search]
  )
  const pageTutorials = React.useMemo(
    () => buildQuickChatPageTutorialEntries(currentRoute, completedTutorials),
    [currentRoute, completedTutorials]
  )

  const handleStartTutorial = React.useCallback(
    (tutorialId: string) => {
      if (onStartTutorial) {
        onStartTutorial(tutorialId)
        return
      }
      startTutorial(tutorialId)
    },
    [onStartTutorial, startTutorial]
  )

  return (
    <div className="flex h-full flex-col gap-3">
      <div
        className="rounded-md border border-border bg-surface p-3"
        data-testid="quick-chat-guides-tutorials-section"
      >
        <div className="mb-2 flex items-center gap-2 text-sm font-medium text-text">
          <GraduationCap className="h-4 w-4 text-primary" />
          <span>
            {t(
              "option:quickChatHelper.guides.tutorialsTitle",
              "Tutorials for this page"
            )}
          </span>
        </div>
        <p className="mb-2 text-xs text-text-muted">
          {t(
            "option:quickChatHelper.guides.tutorialsSubtitle",
            "Launch a guided walkthrough of the current page before diving into workflow cards."
          )}
        </p>
        {pageTutorials.length === 0 ? (
          <div
            className="rounded-md border border-dashed border-border bg-surface2 p-3 text-xs text-text-muted"
            data-testid="quick-chat-guides-tutorials-empty"
          >
            {t(
              "option:quickChatHelper.guides.tutorialsEmpty",
              "No tutorials are available for this page yet."
            )}
          </div>
        ) : (
          <div className="space-y-2">
            {pageTutorials.map(({ tutorial, isLocked, isCompleted }) => {
              return (
                <div
                  key={tutorial.id}
                  data-testid={`quick-chat-guides-tutorial-${tutorial.id}`}
                  data-locked={isLocked ? "true" : "false"}
                  data-completed={isCompleted ? "true" : "false"}
                  className={`flex items-start gap-2 rounded-md border border-border px-2.5 py-2 ${
                    isLocked ? "bg-surface2/40 opacity-80" : "bg-surface2/70"
                  }`}
                >
                  <div
                    className={`mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded ${
                      isLocked
                        ? "bg-surface2 text-text-subtle"
                        : isCompleted
                          ? "bg-success/10 text-success"
                          : "bg-primary/10 text-primary"
                    }`}
                  >
                    {isLocked ? (
                      <Lock className="h-3.5 w-3.5" />
                    ) : isCompleted ? (
                      <Check className="h-3.5 w-3.5" />
                    ) : (
                      <GraduationCap className="h-3.5 w-3.5" />
                    )}
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-medium text-text">
                      {t(tutorial.labelKey, tutorial.labelFallback)}
                    </p>
                    <p className="mt-0.5 text-xs text-text-muted line-clamp-2">
                      {t(tutorial.descriptionKey, tutorial.descriptionFallback)}
                    </p>
                    <p className="mt-1 text-[11px] text-text-subtle">
                      {t("tutorials:steps", { count: tutorial.steps.length })}
                    </p>
                  </div>
                  <Button
                    size="small"
                    type={isCompleted ? "default" : "primary"}
                    ghost={isCompleted}
                    data-testid={`quick-chat-guides-tutorial-action-${tutorial.id}`}
                    icon={
                      isLocked ? (
                        <Lock className="h-3.5 w-3.5" />
                      ) : isCompleted ? (
                        <RotateCcw className="h-3.5 w-3.5" />
                      ) : (
                        <Play className="h-3.5 w-3.5" />
                      )
                    }
                    disabled={isLocked}
                    onClick={() => handleStartTutorial(tutorial.id)}
                  >
                    {isLocked
                      ? t("tutorials:actions.locked", "Locked")
                      : isCompleted
                        ? t("tutorials:actions.replay", "Replay")
                        : t("tutorials:actions.start", "Start")}
                  </Button>
                </div>
              )
            })}
          </div>
        )}
      </div>

      <div
        className="rounded-md border border-border bg-surface p-3"
        data-testid="quick-chat-guides-workflow-section"
      >
        <div className="mb-2 flex items-center gap-2 text-sm font-medium text-text">
          <Compass className="h-4 w-4 text-primary" />
          <span>
            {t("option:quickChatHelper.guides.title", "Workflow guide browser")}
          </span>
        </div>
        <p className="mb-2 text-xs text-text-muted">
          {t(
            "option:quickChatHelper.guides.subtitle",
            "Browse common goals, then open the right page or ask the helper to explain the workflow."
          )}
        </p>
        <Input
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          placeholder={t(
            "option:quickChatHelper.guides.searchPlaceholder",
            "Search guides by goal, page, or keyword..."
          )}
          allowClear
        />
      </div>

      <div className="flex-1 space-y-3 overflow-y-auto pr-1">
        {filteredGuides.length === 0 ? (
          <div className="rounded-md border border-dashed border-border bg-surface p-4 text-sm text-text-muted">
            {t(
              "option:quickChatHelper.guides.empty",
              "No matching guides. Try a broader term like “workflow”, “RAG”, or “media”."
            )}
          </div>
        ) : (
          filteredGuides.map((guide) => (
            <section
              key={guide.id}
              className="rounded-md border border-border bg-surface p-3"
            >
              <h4 className="text-sm font-semibold text-text">{guide.title}</h4>
              <p className="mt-1 text-xs font-medium text-text-muted">
                {guide.question}
              </p>
              <p className="mt-2 text-sm text-text">{guide.answer}</p>
              <div className="mt-2 flex flex-wrap gap-1">
                {guide.tags.map((tag) => (
                  <span
                    key={`${guide.id}-${tag}`}
                    className="inline-flex items-center rounded-full border border-border bg-surface2 px-2 py-0.5 text-[11px] text-text-muted"
                  >
                    {tag}
                  </span>
                ))}
              </div>
              <div className="mt-3 flex flex-wrap gap-2">
                <Button
                  size="small"
                  type="primary"
                  ghost
                  icon={<MessageCircleQuestion className="h-3.5 w-3.5" />}
                  onClick={() => onAskGuide(guide.question)}
                  disabled={askDisabled}
                >
                  {t("option:quickChatHelper.guides.askDocs", "Ask docs mode")}
                </Button>
                <Button
                  size="small"
                  icon={<Route className="h-3.5 w-3.5" />}
                  onClick={() => onOpenRoute(guide.route)}
                >
                  {t("option:quickChatHelper.guides.openPage", {
                    defaultValue: "Open {{page}}",
                    page: guide.routeLabel
                  })}
                </Button>
              </div>
            </section>
          ))
        )}
      </div>
    </div>
  )
}

export default QuickChatGuidesPanel
