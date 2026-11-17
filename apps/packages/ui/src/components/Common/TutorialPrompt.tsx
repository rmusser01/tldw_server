import { useStartTutorial } from "@/hooks/useStartTutorial"
/**
 * TutorialPrompt Component
 * Shows a toast notification on first visit to pages with tutorials
 */

import React, { useEffect, useLayoutEffect, useRef } from "react"
import { useLocation } from "react-router-dom"
import { useTranslation } from "react-i18next"
import { Button, ConfigProvider, notification, Space, theme } from "antd"
import { GraduationCap } from "lucide-react"
import { CHAT_PATH } from "@/routes/route-paths"
import { useTutorialStore } from "@/store/tutorials"
import { getPrimaryTutorialForRoute, normalizeTutorialRoute } from "@/tutorials"
import { isHiddenBySelfOrAncestor } from "@/utils/focus-return"

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/** Delay before showing the prompt (ms) to let the page settle */
const PROMPT_DELAY_MS = 2000
/** Minimum time between prompt appearances to avoid rapid-route spam */
const PROMPT_COOLDOWN_MS = 10000

/** Duration the notification stays open (seconds); auto-dismiss after 15s */
const NOTIFICATION_DURATION = 15
const NOTIFICATION_KEY = "tutorial-prompt-global"
// Place the Chat offer below the workspace toolbar and clear of the composer.
const CHAT_PROMPT_TOP_PX = 128

// ─────────────────────────────────────────────────────────────────────────────
// Component
// ─────────────────────────────────────────────────────────────────────────────

export const TutorialPrompt: React.FC<{ excludedRoutes?: readonly string[] }> = ({
  excludedRoutes
}) => {
  const { t } = useTranslation(["tutorials", "common"])
  const location = useLocation()
  const { token } = theme.useToken()
  const [api, contextHolder] = notification.useNotification({ top: CHAT_PROMPT_TOP_PX })
  const shownRef = useRef<Set<string>>(new Set())
  const lastPromptShownAtRef = useRef(0)

  const hasSeenPromptForPage = useTutorialStore(
    (state) => state.hasSeenPromptForPage
  )
  const markPromptSeen = useTutorialStore((state) => state.markPromptSeen)
  // The prompt is about the page, so a same-page address change (opening a
  // note writes /notes?note=<id>) or a chat change (switch, New chat) neither
  // closes it nor stales its Start; leaving the page does both.
  const startTutorial = useStartTutorial({ fence: "page" })
  const launcherRef = useRef(startTutorial)
  useLayoutEffect(() => {
    launcherRef.current = startTutorial
  }, [startTutorial])
  const isHelpModalOpen = useTutorialStore((state) => state.isHelpModalOpen)
  const activeTutorialId = useTutorialStore((state) => state.activeTutorialId)
  const completedTutorials = useTutorialStore(
    (state) => state.completedTutorials
  )

  useEffect(() => {
    // Keep the prompt's original source; unrelated renders must not cancel its delay.
    const launchTutorial = launcherRef.current
    const pageKey = normalizeTutorialRoute(location.pathname)
    const tutorialLookupOptions = {
      completedTutorialIds: completedTutorials
    }

    // Skip prompts while a tutorial/help modal is active.
    if (isHelpModalOpen || activeTutorialId || excludedRoutes?.includes(pageKey)) {
      return
    }

    // Skip if we've already shown a prompt for this page in this session
    if (shownRef.current.has(pageKey)) {
      return
    }

    // Skip if user has already seen the prompt for this page
    if (hasSeenPromptForPage(pageKey)) {
      return
    }

    // Get the primary tutorial for this page
    const primaryTutorial = getPrimaryTutorialForRoute(
      pageKey,
      tutorialLookupOptions
    )
    if (!primaryTutorial || completedTutorials.includes(primaryTutorial.id)) {
      return
    }

    // The page counts as prompted once the prompt appears; leaving the page
    // before then cancels the pending prompt, and this lets it be scheduled
    // again on a later visit.
    const now = Date.now()
    const elapsedSinceLastPrompt = now - lastPromptShownAtRef.current
    const cooldownRemaining = Math.max(
      PROMPT_COOLDOWN_MS - elapsedSinceLastPrompt,
      0
    )
    const delayMs = Math.max(PROMPT_DELAY_MS, cooldownRemaining)

    const firstTarget = primaryTutorial.requireFirstTargetForPrompt
      ? primaryTutorial.steps[0]?.target
      : undefined
    let currentPrompt: symbol | undefined
    const closePrompt = () => {
      if (!currentPrompt) return
      currentPrompt = undefined
      api.destroy(NOTIFICATION_KEY)
    }
    const updatePrompt = () => {
      const overlayOpen = Array.from(
        document.querySelectorAll<HTMLElement>(
          '[aria-modal="true"], [data-tutorial-prompt-blocker][aria-expanded="true"]'
        )
      ).some((element) => !isHiddenBySelfOrAncestor(element))
      if (overlayOpen) {
        closePrompt()
        return
      }
      if (currentPrompt || (firstTarget && !document.querySelector(firstTarget))) {
        return
      }

      const prompt = Symbol()
      currentPrompt = prompt
      shownRef.current.add(pageKey)
      lastPromptShownAtRef.current = Date.now()

      const finishPrompt = () => {
        promptObserver.disconnect()
        markPromptSeen(pageKey)
      }
      const handleDismiss = () => {
        if (currentPrompt !== prompt) return
        closePrompt()
        finishPrompt()
      }

      const handleStartTour = () => {
        if (currentPrompt !== prompt || !launchTutorial(primaryTutorial.id)) return
        handleDismiss()
      }

      api.info({
        key: NOTIFICATION_KEY,
        message: (
          <span className="font-medium">
            {t("tutorials:prompt.title", "New to this page?")}
          </span>
        ),
        description: (
          <span className="text-text-muted">
            {t(primaryTutorial.descriptionKey, primaryTutorial.descriptionFallback)}
          </span>
        ),
        icon: (
          <div className="flex size-8 items-center justify-center rounded-lg bg-primary/10">
            <GraduationCap className="size-4 text-primary" />
          </div>
        ),
        btn: (
          <Space>
            <Button size="small" onClick={handleDismiss}>
              {t("tutorials:prompt.dismiss", "Not now")}
            </Button>
            <Button type="primary" size="small" onClick={handleStartTour}>
              {t("tutorials:prompt.startTour", "Start tour")}
            </Button>
          </Space>
        ),
        duration: NOTIFICATION_DURATION,
        placement: pageKey === CHAT_PATH ? "topRight" : "bottomRight",
        className: "tutorial-prompt-notification",
        onClose: () => {
          if (currentPrompt !== prompt) return
          currentPrompt = undefined
          finishPrompt()
        }
      })
    }

    const promptObserver = new MutationObserver(updatePrompt)
    const timeout = globalThis.setTimeout(() => {
      promptObserver.observe(document.body, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ["aria-expanded", "aria-hidden", "aria-modal", "class", "hidden", "style"]
      })
      updatePrompt()
    }, delayMs)

    return () => {
      globalThis.clearTimeout(timeout)
      promptObserver.disconnect()
      closePrompt()
    }
  }, [
    location.pathname,
    activeTutorialId,
    api,
    excludedRoutes,
    isHelpModalOpen,
    completedTutorials,
    t,
    hasSeenPromptForPage,
    markPromptSeen
  ])

  // A dialog can open after its page's offer is already visible.
  return (
    <ConfigProvider
      theme={{
        components: { Notification: { zIndexPopup: token.zIndexPopupBase - 1 } }
      }}
    >
      {contextHolder}
    </ConfigProvider>
  )
}

export default TutorialPrompt
