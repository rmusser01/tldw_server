import React, { useCallback, useEffect } from "react"
import { useNavigate } from "react-router-dom"
import { useStorage } from "@plasmohq/storage/hook"
import { Checkbox } from "antd"
import { useTranslation } from "react-i18next"
import { Workflow, Search, BarChart2, MessageSquare, Sparkles } from "lucide-react"
import { useWorkflowsStore } from "@/store/workflows"
import { SKIP_LANDING_HUB_KEY } from "@/utils/storage-migrations"
import { LandingCard } from "./LandingCard"

/**
 * LandingHub
 *
 * The main landing page shown after login, presenting 4 primary actions:
 * 1. Get Started with a Workflow - Opens workflow modal
 * 2. Do Research - Navigate to the deep research console
 * 3. Perform Analysis - Navigate to media-multi
 * 4. Start Chatting - Navigate to chat playground
 *
 * Includes a "Don't show again" checkbox that skips directly to chat.
 */
export const LandingHub: React.FC = () => {
  const { t } = useTranslation(["common", "workflows"])
  const navigate = useNavigate()
  const [skipLanding, setSkipLanding, skipMeta] = useStorage<boolean>(
    SKIP_LANDING_HUB_KEY,
    false
  )

  // Workflow store actions
  const setShowLanding = useWorkflowsStore((s) => s.setShowLanding)

  const isSkipLoading = skipMeta?.isLoading ?? false

  const handleWorkflowClick = useCallback(() => {
    setShowLanding(true)
  }, [setShowLanding])

  const handleResearchClick = useCallback(() => {
    navigate("/research")
  }, [navigate])

  const handleAnalysisClick = useCallback(() => {
    navigate("/media-multi")
  }, [navigate])

  const handleChatClick = useCallback(() => {
    navigate("/chat")
  }, [navigate])

  const handleSkipChange = useCallback(
    (checked: boolean) => {
      setSkipLanding(checked)
    },
    [setSkipLanding]
  )

  useEffect(() => {
    if (isSkipLoading) {
      return
    }
    if (skipLanding) {
      navigate("/chat", { replace: true })
    }
  }, [isSkipLoading, skipLanding, navigate])

  // Wait for skip preference to hydrate before rendering
  if (isSkipLoading) {
    return null
  }

  if (skipLanding) {
    return null
  }

  return (
    <div className="flex flex-col items-center justify-center min-h-full p-8">
      {/* Welcome Header */}
      <div className="text-center mb-8">
        <h1 className="text-2xl font-semibold text-text flex items-center justify-center gap-2">
          <Sparkles className="h-6 w-6 text-primary" />
          {t("common:landingHub.title", "Welcome to tldw")}
        </h1>
        <p className="text-base text-text-muted mt-2">
          {t("common:landingHub.subtitle", "What would you like to do today?")}
        </p>
      </div>

      {/* 2x2 Grid of Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-4xl w-full">
        <LandingCard
          icon={<Workflow className="h-6 w-6" />}
          title={t("common:landingHub.workflow.title", "Get Started with a Workflow")}
          description={t(
            "common:landingHub.workflow.description",
            "Follow guided steps to accomplish a specific goal"
          )}
          onClick={handleWorkflowClick}
        />
        <LandingCard
          icon={<Search className="h-6 w-6" />}
          title={t("common:landingHub.research.title", "Do Research")}
          description={t(
            "common:landingHub.research.description",
            "Launch the deep research console for long-form cited runs"
          )}
          onClick={handleResearchClick}
        />
        <LandingCard
          icon={<BarChart2 className="h-6 w-6" />}
          title={t("common:landingHub.analysis.title", "Perform Analysis")}
          description={t(
            "common:landingHub.analysis.description",
            "Review and compare multiple media items"
          )}
          onClick={handleAnalysisClick}
        />
        <LandingCard
          icon={<MessageSquare className="h-6 w-6" />}
          title={t("common:landingHub.chat.title", "Start Chatting")}
          description={t(
            "common:landingHub.chat.description",
            "Jump straight into conversation (Pro users)"
          )}
          onClick={handleChatClick}
        />
      </div>

      {/* Skip Checkbox */}
      <div className="mt-8">
        <Checkbox
          checked={skipLanding}
          onChange={(e) => handleSkipChange(e.target.checked)}
        >
          <span className="text-sm text-text-muted">
            {t("common:landingHub.skipCheckbox", "Don't show this again - go straight to chat")}
          </span>
        </Checkbox>
      </div>

    </div>
  )
}

export default LandingHub
