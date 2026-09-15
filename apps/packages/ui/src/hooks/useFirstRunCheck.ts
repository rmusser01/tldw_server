import React from "react"

import { apiSend } from "@/services/api-send"
import type { PersonaSetupStep } from "@/hooks/usePersonaSetupWizard"

const DISMISSED_KEY = "assistant_setup_dismissed"

type UseFirstRunCheckResult = {
  shouldShowSetup: boolean
  resumeStep: PersonaSetupStep | null
  loading: boolean
}

/**
 * Determines whether the first-run setup wizard should be presented.
 *
 * Fetches `GET /api/v1/persona/profiles`. If no profiles exist AND the
 * user has not previously dismissed the wizard (tracked via localStorage
 * key `assistant_setup_dismissed`), `shouldShowSetup` is `true`.
 * Source destinations can opt in to continuing without optional personalization.
 *
 * If any profile has `setup.status === "in_progress"`, `resumeStep`
 * returns that profile's `current_step` so the wizard can resume where
 * the user left off.
 */
export function useFirstRunCheck({
  allowCompletedSetup = false
}: { allowCompletedSetup?: boolean } = {}): UseFirstRunCheckResult {
  const [shouldShowSetup, setShouldShowSetup] = React.useState(false)
  const [resumeStep, setResumeStep] = React.useState<PersonaSetupStep | null>(
    null
  )
  const [loading, setLoading] = React.useState(true)

  React.useEffect(() => {
    let cancelled = false

    const check = async () => {
      setLoading(true)
      try {
        const path = "/api/v1/persona/profiles" as const
        const res = await apiSend<
          Array<{
            setup?: { status?: string; current_step?: string }
          }>
        >({
          path,
          method: "GET"
        })
        if (cancelled) return

        if (!res.ok || !Array.isArray(res.data)) {
          setShouldShowSetup(false)
          setResumeStep(null)
          return
        }
        const profiles = res.data

        // Look for an in-progress setup to allow resuming
        let foundResumeStep: PersonaSetupStep | null = null
        for (const profile of profiles) {
          const setup = profile?.setup
          if (
            setup &&
            typeof setup === "object" &&
            setup.status === "in_progress" &&
            typeof setup.current_step === "string"
          ) {
            foundResumeStep = setup.current_step as PersonaSetupStep
            break
          }
        }
        setResumeStep(foundResumeStep)

        // Show setup if zero profiles and not previously dismissed
        if (profiles.length === 0) {
          // Unified server setup owns its own gate. Optional personalization
          // must not interrupt source destinations or probe admin-only setup.
          if (allowCompletedSetup) {
            setShouldShowSetup(false)
            return
          }
          let dismissed = false
          try {
            dismissed = localStorage.getItem(DISMISSED_KEY) === "true"
          } catch {
            // localStorage unavailable -- treat as not dismissed
          }
          setShouldShowSetup(!dismissed)
        } else {
          setShouldShowSetup(false)
        }
      } catch {
        if (!cancelled) {
          // On network failure, don't block the user with the setup screen
          setShouldShowSetup(false)
          setResumeStep(null)
        }
      } finally {
        if (!cancelled) {
          setLoading(false)
        }
      }
    }

    void check()

    return () => {
      cancelled = true
    }
  }, [allowCompletedSetup])

  return { shouldShowSetup, resumeStep, loading }
}
