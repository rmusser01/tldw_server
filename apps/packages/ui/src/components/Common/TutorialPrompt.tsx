/**
 * TutorialPrompt Component
 * Shows a toast notification on first visit to pages with tutorials
 */

import React, { useEffect, useRef } from "react"
import { useLocation } from "react-router-dom"
import { useTranslation } from "react-i18next"
import { notification, Button, Space } from "antd"
import { GraduationCap } from "lucide-react"
import { useTutorialStore } from "@/store/tutorials"
import { getPrimaryTutorialForRoute, hasTutorialsForRoute } from "@/tutorials"

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

// ─────────────────────────────────────────────────────────────────────────────
// Component
// ─────────────────────────────────────────────────────────────────────────────

export const TutorialPrompt: React.FC = () => {
  const { t } = useTranslation(["tutorials", "common"])
  const location = useLocation()
  const [api, contextHolder] = notification.useNotification()
  const timeoutRef = useRef<ReturnType<typeof globalThis.setTimeout> | null>(null)
  const notificationKeyRef = useRef<string | null>(null)
  const shownRef = useRef<Set<string>>(new Set())
  const lastPromptShownAtRef = useRef(0)
  const activePromptPageRef = useRef<string | null>(null)
  const suppressCloseMarkRef = useRef(false)

  const hasSeenPromptForPage = useTutorialStore((state) => state.hasSeenPromptForPage)
  const markPromptSeen = useTutorialStore((state) => state.markPromptSeen)
  const startTutorial = useTutorialStore((state) => state.startTutorial)
  const isHelpModalOpen = useTutorialStore((state) => state.isHelpModalOpen)
  const activeTutorialId = useTutorialStore((state) => state.activeTutorialId)
  const completedTutorials = useTutorialStore((state) => state.completedTutorials)

  useEffect(() => {
    const pageKey = location.pathname
    const tutorialLookupOptions = {
      completedTutorialIds: completedTutorials
    }

    // Clean up previous timeout
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current)
      timeoutRef.current = null
    }

    // Always keep a single active notification instance.
    if (notificationKeyRef.current) {
      suppressCloseMarkRef.current = true
      api.destroy(notificationKeyRef.current)
      notificationKeyRef.current = null
      activePromptPageRef.current = null
    } else {
      suppressCloseMarkRef.current = false
    }

    // Skip prompts while a tutorial/help modal is active.
    if (isHelpModalOpen || activeTutorialId) {
      return
    }

    // Skip if we've already shown a prompt for this page in this session
    if (shownRef.current.has(pageKey)) {
      return
    }

    // Skip if page has no tutorials
    if (!hasTutorialsForRoute(pageKey, tutorialLookupOptions)) {
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
    if (!primaryTutorial) {
      return
    }

    // Mark as shown for this session immediately
    shownRef.current.add(pageKey)

    const now = Date.now()
    const elapsedSinceLastPrompt = now - lastPromptShownAtRef.current
    const cooldownRemaining = Math.max(PROMPT_COOLDOWN_MS - elapsedSinceLastPrompt, 0)
    const delayMs = Math.max(PROMPT_DELAY_MS, cooldownRemaining)

    // Show notification after a delay
    timeoutRef.current = globalThis.setTimeout(() => {
      lastPromptShownAtRef.current = Date.now()
      activePromptPageRef.current = pageKey

      const handleDismiss = () => {
        if (notificationKeyRef.current) {
          api.destroy(notificationKeyRef.current)
          notificationKeyRef.current = null
        }
        markPromptSeen(pageKey)
        activePromptPageRef.current = null
      }

      const handleStartTour = () => {
        if (notificationKeyRef.current) {
          api.destroy(notificationKeyRef.current)
          notificationKeyRef.current = null
        }
        markPromptSeen(pageKey)
        startTutorial(primaryTutorial.id)
        activePromptPageRef.current = null
      }

      notificationKeyRef.current = NOTIFICATION_KEY
      api.info({
        key: NOTIFICATION_KEY,
        message: (
          <span className="font-medium">
            {t("tutorials:prompt.title", "New to this page?")}
          </span>
        ),
        description: (
          <span className="text-text-muted">
            {t("tutorials:prompt.description", {
              defaultValue: "Take a quick tour to learn the key features.",
              tutorialName: t(primaryTutorial.labelKey, primaryTutorial.labelFallback)
            })}
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
        placement: "bottomRight",
        className: "tutorial-prompt-notification",
        onClose: () => {
          notificationKeyRef.current = null
          if (suppressCloseMarkRef.current) {
            suppressCloseMarkRef.current = false
            return
          }

          if (activePromptPageRef.current) {
            markPromptSeen(activePromptPageRef.current)
          }
          activePromptPageRef.current = null
        }
      })
    }, delayMs)

    return () => {
      if (timeoutRef.current) {
        window.clearTimeout(timeoutRef.current)
        timeoutRef.current = null
      }
      if (notificationKeyRef.current) {
        suppressCloseMarkRef.current = true
        api.destroy(notificationKeyRef.current)
        notificationKeyRef.current = null
        activePromptPageRef.current = null
      }
    }
  }, [
    location.pathname,
    activeTutorialId,
    api,
    isHelpModalOpen,
    completedTutorials,
    t,
    hasSeenPromptForPage,
    markPromptSeen,
    startTutorial
  ])

  return contextHolder
}

export default TutorialPrompt
