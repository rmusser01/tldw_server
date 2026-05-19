import React from "react"

export type PersonaSetupStep =
  | "archetype"
  | "persona"
  | "voice"
  | "commands"
  | "safety"
  | "test"

export type PersonaSetupState = {
  status?: "not_started" | "in_progress" | "completed" | null
  version?: number | null
  run_id?: string | null
  current_step?: PersonaSetupStep | null
  completed_steps?: PersonaSetupStep[] | null
  completed_at?: string | null
  last_test_type?: "dry_run" | "live_session" | null
}

type UsePersonaSetupWizardArgs = {
  selectedPersonaId: string
  isCompanionMode: boolean
  loading: boolean
  setup: PersonaSetupState | null
}

const VALID_SETUP_STEPS = new Set<PersonaSetupStep>([
  "archetype",
  "persona",
  "voice",
  "commands",
  "safety",
  "test"
])

export const usePersonaSetupWizard = ({
  selectedPersonaId,
  isCompanionMode,
  loading,
  setup
}: UsePersonaSetupWizardArgs) => {
  const normalizedPersonaId = String(selectedPersonaId || "").trim()
  const currentStep = React.useMemo<PersonaSetupStep>(() => {
    const candidate = String(setup?.current_step || "").trim()
    return VALID_SETUP_STEPS.has(candidate as PersonaSetupStep)
      ? (candidate as PersonaSetupStep)
      : "persona"
  }, [setup?.current_step])

  const status = String(setup?.status || "").trim()
  const isSetupRequired =
    !isCompanionMode &&
    !loading &&
    normalizedPersonaId.length > 0 &&
    setup !== null &&
    status !== "completed"

  return {
    isSetupRequired,
    currentStep
  }
}
