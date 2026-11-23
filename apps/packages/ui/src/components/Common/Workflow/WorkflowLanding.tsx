import React, { useCallback, useEffect, useRef, useState } from "react"
import { Button, Checkbox } from "antd"
import {
  X,
  Sparkles,
  MessageSquare,
  Workflow,
  Search,
  BarChart2
} from "lucide-react"
import { useTranslation } from "react-i18next"
import { useNavigate } from "react-router-dom"
import { useWorkflowsStore } from "@/store/workflows"
import { WorkflowCard } from "./WorkflowCard"
import {
  ALL_WORKFLOWS,
  CATEGORY_LABELS,
  CATEGORY_ORDER,
  getWorkflowsByCategory
} from "./workflow-definitions"
import type { WorkflowCategory, WorkflowId } from "@/types/workflows"

interface WorkflowLandingProps {
  onClose?: () => void
  onJustChat?: () => void
}

const FOCUSABLE_SELECTOR =
  'a[href], button:not([disabled]), textarea:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])'

const getFocusableElements = (container: HTMLElement | null) => {
  if (!container) return []
  return Array.from(
    container.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR)
  ).filter((el) => !el.hasAttribute("aria-hidden"))
}

/**
 * WorkflowLanding
 *
 * A welcoming landing page that appears for new users after first login.
 * Presents available workflows organized by category with a goal-first approach.
 *
 * Features:
 * - Grid of workflow cards organized by category
 * - "Don't show again" checkbox
 * - Quick access to "Just Chat" mode
 */
export const WorkflowLanding: React.FC<WorkflowLandingProps> = ({
  onClose,
  onJustChat
}) => {
  const { t } = useTranslation(["workflows", "common"])
  const navigate = useNavigate()
  const [view, setView] = useState<"hub" | "workflows">("hub")

  // Store state
  const showLanding = useWorkflowsStore((s) => s.showLanding)
  const landingConfig = useWorkflowsStore((s) => s.landingConfig)

  // Store actions
  const setShowLanding = useWorkflowsStore((s) => s.setShowLanding)
  const dismissLanding = useWorkflowsStore((s) => s.dismissLanding)
  const startWorkflow = useWorkflowsStore((s) => s.startWorkflow)
  const loadLandingConfig = useWorkflowsStore((s) => s.loadLandingConfig)

  // Load config on mount
  useEffect(() => {
    loadLandingConfig()
  }, [loadLandingConfig])

  useEffect(() => {
    if (showLanding) {
      setView("hub")
    }
  }, [showLanding])

  const handleSelectWorkflow = useCallback(
    (workflowId: WorkflowId) => {
      startWorkflow(workflowId)
    },
    [startWorkflow]
  )

  const handleDontShowAgain = useCallback(
    (checked: boolean) => {
      if (checked) {
        dismissLanding()
      }
    },
    [dismissLanding]
  )

  const handleClose = useCallback(() => {
    setShowLanding(false)
    onClose?.()
  }, [setShowLanding, onClose])

  const handleJustChat = useCallback(() => {
    setShowLanding(false)
    onJustChat?.()
  }, [setShowLanding, onJustChat])

  const handleGoToWorkflows = useCallback(() => {
    setView("workflows")
  }, [])

  const handleResearchClick = useCallback(() => {
    setShowLanding(false)
    navigate("/research")
    onClose?.()
  }, [navigate, onClose, setShowLanding])

  const handleAnalysisClick = useCallback(() => {
    setShowLanding(false)
    navigate("/media-multi")
    onClose?.()
  }, [navigate, onClose, setShowLanding])

  const handleChatClick = useCallback(() => {
    setShowLanding(false)
    onJustChat?.()
  }, [onJustChat, setShowLanding])

  // Check if a workflow has been completed before
  const isCompleted = (workflowId: WorkflowId) =>
    landingConfig.completedWorkflows.includes(workflowId)

  if (!showLanding) {
    return null
  }

  if (view === "hub") {
    return (
      <div className="flex flex-col h-full p-6 sm:p-8 overflow-auto">
        {/* Header */}
        <div className="relative mb-8 text-center">
          <h1
            id="workflow-landing-title"
            className="text-2xl font-semibold text-text inline-flex items-center justify-center gap-2"
          >
            <Sparkles className="h-6 w-6 text-primary" />
            {t("workflows:landing.title", "Welcome to tldw")}
          </h1>
          <p className="text-base text-text-muted mt-2">
            {t("workflows:landing.subtitle", "What would you like to do?")}
          </p>
          <Button
            type="text"
            size="small"
            icon={<X className="h-4 w-4" />}
            onClick={handleClose}
            aria-label={t("common:close", "Close")}
            className="absolute right-0 top-0"
          />
        </div>

        {/* 2x2 Grid of Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 w-full">
          <HubCard
            icon={<Workflow className="h-5 w-5" />}
            title={t(
              "common:landingHub.workflow.title",
              "Get Started with a Workflow"
            )}
            description={t(
              "common:landingHub.workflow.description",
              "Follow guided steps to accomplish a specific goal"
            )}
            onClick={handleGoToWorkflows}
          />
          <HubCard
            icon={<Search className="h-5 w-5" />}
            title={t("common:landingHub.research.title", "Do Research")}
            description={t(
              "common:landingHub.research.description",
              "Launch the deep research console for long-form cited runs"
            )}
            onClick={handleResearchClick}
          />
          <HubCard
            icon={<BarChart2 className="h-5 w-5" />}
            title={t(
              "common:landingHub.analysis.title",
              "Perform Analysis"
            )}
            description={t(
              "common:landingHub.analysis.description",
              "Review and compare multiple media items"
            )}
            onClick={handleAnalysisClick}
          />
          <HubCard
            icon={<MessageSquare className="h-5 w-5" />}
            title={t("common:landingHub.chat.title", "Start Chatting")}
            description={t(
              "common:landingHub.chat.description",
              "Jump straight into conversation (Pro users)"
            )}
            onClick={handleChatClick}
          />
        </div>

        {/* Footer */}
        <div className="mt-8 pt-4 border-t border-border">
          <div className="flex items-center justify-center">
            <Checkbox onChange={(e) => handleDontShowAgain(e.target.checked)}>
              <span className="text-sm text-text-muted">
                {t(
                  "common:landingHub.skipCheckbox",
                  "Don't show this again - go straight to chat"
                )}
              </span>
            </Checkbox>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full p-6 sm:p-8 overflow-auto">
      {/* Header */}
      <div className="relative mb-8 text-center">
        <h1
          id="workflow-landing-title"
          className="text-2xl font-semibold text-text inline-flex items-center justify-center gap-2"
        >
          <Sparkles className="h-6 w-6 text-primary" />
          {t("workflows:landing.title", "Welcome to tldw")}
        </h1>
        <p className="text-base text-text-muted mt-2">
          {t("workflows:landing.subtitle", "What would you like to do?")}
        </p>
        <Button
          type="text"
          size="small"
          icon={<X className="h-4 w-4" />}
          onClick={handleClose}
          aria-label={t("common:close", "Close")}
          className="absolute right-0 top-0"
        />
      </div>

      {/* Workflow categories */}
      <div className="flex-1 space-y-8 overflow-auto">
        {CATEGORY_ORDER.map((category) => {
          const workflows = getWorkflowsByCategory(category)
          if (workflows.length === 0) return null

          return (
            <div key={category} className="space-y-4">
              <h2 className="text-sm font-medium text-text-muted">
                {t(CATEGORY_LABELS[category], category)}
              </h2>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                {workflows.map((workflow) => (
                  <WorkflowCard
                    key={workflow.id}
                    workflow={workflow}
                    onSelect={handleSelectWorkflow}
                    isCompleted={isCompleted(workflow.id)}
                  />
                ))}
              </div>
            </div>
          )
        })}
      </div>

      {/* Footer */}
      <div className="mt-8 pt-4 border-t border-border">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <Checkbox onChange={(e) => handleDontShowAgain(e.target.checked)}>
            <span className="text-sm text-text-muted">
              {t("workflows:landing.dontShowAgain", "Don't show this again")}
            </span>
          </Checkbox>
          <Button
            type="link"
            icon={<MessageSquare className="h-4 w-4" />}
            onClick={handleJustChat}
          >
            {t("workflows:landing.justChat", "Just Chat")}
          </Button>
        </div>
      </div>
    </div>
  )
}

const HubCard: React.FC<{
  icon: React.ReactNode
  title: string
  description: string
  onClick: () => void
}> = ({ icon, title, description, onClick }) => {
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault()
      onClick()
    }
  }

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={onClick}
      onKeyDown={handleKeyDown}
      aria-label={title}
      className="group cursor-pointer transition-all border border-border rounded-xl p-5 bg-surface hover:border-primary hover:shadow-md"
    >
      <div className="flex flex-col items-start text-left gap-3">
        <div className="p-3 rounded-lg bg-primary/10 text-primary group-hover:bg-primary group-hover:text-white transition-colors">
          {icon}
        </div>
        <h3 className="text-lg font-semibold text-text">{title}</h3>
        <p className="text-sm text-text-muted">{description}</p>
      </div>
    </div>
  )
}

/**
 * WorkflowLandingModal
 *
 * A modal wrapper for the landing page, used when showing
 * the landing as an overlay on existing content.
 */
export const WorkflowLandingModal: React.FC<WorkflowLandingProps> = (props) => {
  const showLanding = useWorkflowsStore((s) => s.showLanding)
  const setShowLanding = useWorkflowsStore((s) => s.setShowLanding)
  const dialogRef = useRef<HTMLDivElement>(null)
  const lastFocusedElementRef = useRef<HTMLElement | null>(null)

  useEffect(() => {
    if (!showLanding) return

    lastFocusedElementRef.current = document.activeElement as HTMLElement | null
    const dialog = dialogRef.current
    if (!dialog) return

    const focusable = getFocusableElements(dialog)
    const initialFocus = focusable[0] || dialog
    initialFocus.focus()

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.stopPropagation()
        setShowLanding(false)
        props.onClose?.()
        return
      }

      if (event.key !== "Tab") return

      const focusableElements = getFocusableElements(dialog)
      if (focusableElements.length === 0) {
        event.preventDefault()
        dialog.focus()
        return
      }

      const first = focusableElements[0]
      const last = focusableElements[focusableElements.length - 1]
      const active = document.activeElement
      const isShift = event.shiftKey

      if (isShift && (active === first || !dialog.contains(active))) {
        event.preventDefault()
        last.focus()
        return
      }

      if (!isShift && active === last) {
        event.preventDefault()
        first.focus()
      }
    }

    document.addEventListener("keydown", handleKeyDown)
    return () => {
      document.removeEventListener("keydown", handleKeyDown)
      lastFocusedElementRef.current?.focus()
    }
  }, [showLanding, setShowLanding, props.onClose])

  if (!showLanding) {
    return null
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="workflow-landing-title"
        tabIndex={-1}
        className="bg-bg rounded-lg shadow-xl w-full max-w-3xl max-h-[80vh] overflow-hidden"
      >
        <WorkflowLanding {...props} />
      </div>
    </div>
  )
}
