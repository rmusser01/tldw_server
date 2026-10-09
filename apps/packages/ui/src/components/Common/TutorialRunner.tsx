/**
 * TutorialRunner Component
 * Wraps React Joyride to run tutorials from the registry
 */

import React from "react"
import Joyride, {
  CallBackProps,
  STATUS,
  EVENTS,
  ACTIONS,
  type Step
} from "react-joyride"
import { useTranslation } from "react-i18next"
import { useActiveTutorial } from "@/store/tutorials"
import { getTutorialById, type TutorialDefinition } from "@/tutorials"
import { isHiddenBySelfOrAncestor } from "@/utils/focus-return"

const TARGET_RETRY_DELAY_MS = 350
const MAX_TARGET_RETRY_ATTEMPTS = 4

const resolveTargetElement = (target: Step["target"]): HTMLElement | null => {
  if (!target || typeof document === "undefined") return null

  if (typeof target !== "string") {
    return target instanceof HTMLElement && isElementVisible(target)
      ? target
      : null
  }

  const selectors = target
    .split(",")
    .map((selector) => selector.trim())
    .filter((selector) => selector.length > 0)

  for (const selector of selectors) {
    for (const element of document.querySelectorAll(selector)) {
      if (element instanceof HTMLElement && isElementVisible(element)) {
        return element
      }
    }
  }

  return null
}

const isElementVisible = (element: HTMLElement): boolean => {
  const style = window.getComputedStyle(element)
  if (isHiddenBySelfOrAncestor(element) || Number(style.opacity || "1") === 0) {
    return false
  }

  const rect = element.getBoundingClientRect()
  return rect.width > 0 && rect.height > 0
}

// ─────────────────────────────────────────────────────────────────────────────
// Styles
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Joyride styles matching the app's design system
 */
const tutorialStyles = {
  options: {
    primaryColor: "rgb(var(--color-primary, 99 102 241))",
    textColor: "rgb(var(--color-text, 31 41 55))",
    backgroundColor: "rgb(var(--color-surface, 255 255 255))",
    arrowColor: "rgb(var(--color-surface, 255 255 255))",
    overlayColor: "rgba(0, 0, 0, 0.5)",
    zIndex: 10000
  },
  tooltip: {
    borderRadius: 12,
    padding: 16
  },
  tooltipContainer: {
    textAlign: "left" as const
  },
  tooltipTitle: {
    fontSize: 16,
    fontWeight: 600,
    marginBottom: 8
  },
  tooltipContent: {
    fontSize: 14,
    lineHeight: 1.5
  },
  buttonNext: {
    borderRadius: 8,
    padding: "8px 16px",
    fontSize: 14,
    fontWeight: 500
  },
  buttonBack: {
    marginRight: 8,
    fontSize: 14
  },
  buttonSkip: {
    color: "rgb(var(--color-text-muted, 107 114 128))",
    fontSize: 14
  },
  buttonClose: {
    padding: 8
  },
  spotlight: {
    borderRadius: 8
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Component
// ─────────────────────────────────────────────────────────────────────────────

export const TutorialRunner: React.FC = () => {
  const { t } = useTranslation(["tutorials", "common"])
  const {
    activeTutorialId,
    activeStepIndex,
    endTutorial,
    setStepIndex,
    markComplete
  } = useActiveTutorial()
  const retryAttemptsRef = React.useRef<Map<string, number>>(new Map())
  const retryTimersRef = React.useRef<Map<string, number>>(new Map())
  const activeTutorialRef = React.useRef<string | null>(null)
  const [retryNonce, setRetryNonce] = React.useState(0)

  const clearRetryState = React.useCallback(() => {
    retryTimersRef.current.forEach((timerId) => {
      window.clearTimeout(timerId)
    })
    retryTimersRef.current.clear()
    retryAttemptsRef.current.clear()
  }, [])

  React.useEffect(() => {
    activeTutorialRef.current = activeTutorialId
  }, [activeTutorialId])

  React.useEffect(() => {
    clearRetryState()
  }, [activeTutorialId, clearRetryState])

  React.useEffect(() => {
    return () => {
      clearRetryState()
    }
  }, [clearRetryState])

  // Get the tutorial definition
  const tutorial: TutorialDefinition | undefined = activeTutorialId
    ? getTutorialById(activeTutorialId)
    : undefined

  /**
   * Convert tutorial steps to Joyride format with i18n
   */
  const joyrideSteps: Step[] = (tutorial?.steps ?? []).map((step) => ({
    target: resolveTargetElement(step.target) ?? step.target,
    title: t(step.titleKey, step.titleFallback),
    content: t(step.contentKey, step.contentFallback),
    placement: step.placement || "auto",
    disableBeacon: step.disableBeacon ?? false,
    spotlightClicks: step.spotlightClicks ?? false,
    isFixed: step.isFixed ?? false
  }))

  /**
   * Joyride locale strings
   */
  const joyrideLocale = React.useMemo(
    () => ({
      back: t("tutorials:controls.back", "Back"),
      close: t("tutorials:controls.close", "Close"),
      last: t("tutorials:controls.finish", "Finish"),
      next: t("tutorials:controls.next", "Next"),
      skip: t("tutorials:controls.skip", "Skip tour"),
      open: t("tutorials:controls.open", "Open the dialog")
    }),
    [t]
  )

  /**
   * Handle Joyride callbacks
   */
  const handleCallback = React.useCallback(
    (data: CallBackProps) => {
      const { status, index, type, action } = data

      if (action === ACTIONS.CLOSE) {
        clearRetryState()
        endTutorial()
        return
      }

      // Tutorial finished or skipped
      if (status === STATUS.FINISHED || status === STATUS.SKIPPED) {
        clearRetryState()
        if (status === STATUS.FINISHED && activeTutorialId) {
          markComplete(activeTutorialId)
        }
        endTutorial()
        return
      }

      // Step progression
      if (type === EVENTS.STEP_AFTER) {
        const retryKey = `${activeTutorialId ?? "unknown"}:${index}`
        retryAttemptsRef.current.delete(retryKey)
        const retryTimer = retryTimersRef.current.get(retryKey)
        if (typeof retryTimer === "number") {
          window.clearTimeout(retryTimer)
          retryTimersRef.current.delete(retryKey)
        }

        if (action === ACTIONS.NEXT) {
          setStepIndex(index + 1)
        } else if (action === ACTIONS.PREV) {
          setStepIndex(index - 1)
        }
      }

      // Target not found - retry briefly for delayed mount/collapsed UI, then fallback.
      if (type === EVENTS.TARGET_NOT_FOUND) {
        const retryKey = `${activeTutorialId ?? "unknown"}:${index}`
        const currentAttempts = retryAttemptsRef.current.get(retryKey) ?? 0
        const currentStep = tutorial?.steps[index]

        if (currentStep && currentAttempts < MAX_TARGET_RETRY_ATTEMPTS) {
          retryAttemptsRef.current.set(retryKey, currentAttempts + 1)

          const existingTimer = retryTimersRef.current.get(retryKey)
          if (typeof existingTimer === "number") {
            window.clearTimeout(existingTimer)
          }

          const tutorialIdAtSchedule = activeTutorialId
          const timerId = window.setTimeout(() => {
            retryTimersRef.current.delete(retryKey)
            if (activeTutorialRef.current !== tutorialIdAtSchedule) {
              return
            }

            const refreshedTarget = resolveTargetElement(currentStep.target)
            if (refreshedTarget) {
              refreshedTarget.scrollIntoView({
                block: "center",
                inline: "nearest",
                behavior: "smooth"
              })
            }

            if (refreshedTarget) {
              retryAttemptsRef.current.delete(retryKey)
            }

            setStepIndex(index)
            setRetryNonce((value) => value + 1)
          }, TARGET_RETRY_DELAY_MS)

          retryTimersRef.current.set(retryKey, timerId)
          return
        }

        retryAttemptsRef.current.delete(retryKey)
        const finalRetryTimer = retryTimersRef.current.get(retryKey)
        if (typeof finalRetryTimer === "number") {
          window.clearTimeout(finalRetryTimer)
          retryTimersRef.current.delete(retryKey)
        }

        console.warn(
          `[TutorialRunner] Target not found for step ${index}:`,
          currentStep?.target
        )

        // After retries are exhausted, skip to next step (or end without completion).
        if (index < joyrideSteps.length - 1) {
          setStepIndex(index + 1)
        } else {
          // Last step target never resolved, so end without granting completion.
          endTutorial()
        }
      }
    },
    [
      activeTutorialId,
      clearRetryState,
      endTutorial,
      markComplete,
      setStepIndex,
      tutorial,
      joyrideSteps.length
    ]
  )

  // Don't render if no active tutorial
  if (!tutorial || joyrideSteps.length === 0) {
    return null
  }

  return (
    <Joyride
      key={`${activeTutorialId}-${retryNonce}`}
      steps={joyrideSteps}
      run={true}
      stepIndex={activeStepIndex}
      continuous
      showProgress
      showSkipButton
      scrollToFirstStep
      disableScrolling={false}
      callback={handleCallback}
      styles={tutorialStyles}
      locale={joyrideLocale}
      floaterProps={{
        disableAnimation: false
      }}
    />
  )
}

export default TutorialRunner
