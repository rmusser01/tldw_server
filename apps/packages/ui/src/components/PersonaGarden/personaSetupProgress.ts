import type {
  PersonaSetupState,
  PersonaSetupStep
} from "@/hooks/usePersonaSetupWizard"

export type PersonaSetupProgressStatus = "completed" | "current" | "pending"

export type PersonaSetupProgressItem = {
  step: PersonaSetupStep
  label: string
  status: PersonaSetupProgressStatus
  summary: string | null
}

export const PERSONA_SETUP_STEPS: PersonaSetupStep[] = [
  "archetype",
  "persona",
  "voice",
  "commands",
  "safety",
  "test"
]

const PERSONA_SETUP_LABELS: Record<PersonaSetupStep, string> = {
  archetype: "Pick a starting point",
  persona: "Choose persona",
  voice: "Voice defaults",
  commands: "Starter commands",
  safety: "Safety and connections",
  test: "Test and finish"
}

const COMPLETED_SUMMARIES: Record<PersonaSetupStep, string> = {
  archetype: "Archetype selected",
  persona: "Persona selected",
  voice: "Voice defaults saved",
  commands: "Starter commands selected",
  safety: "Safety choices saved",
  test: "Setup verified"
}

const CURRENT_SUMMARIES: Record<PersonaSetupStep, string> = {
  archetype: "Choose an archetype to get started",
  persona: "Choose or create a persona",
  voice: "Save assistant defaults",
  commands: "Pick starter commands",
  safety: "Review approvals and connections",
  test: "Run a setup test"
}

function getCompletedSummary(setup: PersonaSetupState | null, step: PersonaSetupStep): string {
  if (step === "test") {
    if (setup?.last_test_type === "dry_run") {
      return "Finished with dry run"
    }
    if (setup?.last_test_type === "live_session") {
      return "Finished with live session"
    }
  }
  return COMPLETED_SUMMARIES[step]
}

export function buildPersonaSetupProgress(
  setup: PersonaSetupState | null | undefined
): PersonaSetupProgressItem[] {
  const status = setup?.status ?? "not_started"
  const currentStep = setup?.current_step ?? "archetype"
  const completedSteps = new Set<PersonaSetupStep>(
    Array.isArray(setup?.completed_steps) ? setup.completed_steps : []
  )

  return PERSONA_SETUP_STEPS.map((step) => {
    const itemStatus: PersonaSetupProgressStatus = completedSteps.has(step)
      ? "completed"
      : status === "in_progress" && currentStep === step
        ? "current"
        : "pending"

    return {
      step,
      label: PERSONA_SETUP_LABELS[step],
      status: itemStatus,
      summary:
        itemStatus === "completed"
          ? getCompletedSummary(setup ?? null, step)
          : itemStatus === "current"
            ? CURRENT_SUMMARIES[step]
            : null
    }
  })
}
