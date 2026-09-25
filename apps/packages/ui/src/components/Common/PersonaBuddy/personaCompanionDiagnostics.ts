export type PersonaCompanionDiagnosticName =
  | "ambient_selected"
  | "ambient_skipped"
  | "ambient_preempted"
  | "stale_generation"

export type PersonaCompanionFailureClass =
  | "empty_set"
  | "cooldown"
  | "preempted"
  | "stale_action"
  | "stale_timer"

export type PersonaCompanionDiagnosticEvent = {
  event: PersonaCompanionDiagnosticName
  personaId?: string
  packId?: string
  state?: string
  failureClass?: PersonaCompanionFailureClass
}

const SAFE_DIAGNOSTIC_ID = /^[a-zA-Z0-9_.:-]{1,128}$/

export const createPersonaCompanionDiagnostic = ({
  event,
  personaId,
  packId,
  state,
  failureClass
}: PersonaCompanionDiagnosticEvent): PersonaCompanionDiagnosticEvent => ({
  event,
  ...(personaId && SAFE_DIAGNOSTIC_ID.test(personaId) ? { personaId } : {}),
  ...(packId && SAFE_DIAGNOSTIC_ID.test(packId) ? { packId } : {}),
  ...(state && SAFE_DIAGNOSTIC_ID.test(state) ? { state } : {}),
  ...(failureClass ? { failureClass } : {})
})
