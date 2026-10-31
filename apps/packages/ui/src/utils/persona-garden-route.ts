export type PersonaGardenTabKey =
  | "commands"
  | "test-lab"
  | "live"
  | "profiles"
  | "voice"
  | "visuals"
  | "connections"
  | "state"
  | "scopes"
  | "policies"

const PERSONA_GARDEN_TAB_KEYS = new Set<PersonaGardenTabKey>([
  "commands",
  "test-lab",
  "live",
  "profiles",
  "voice",
  "visuals",
  "connections",
  "state",
  "scopes",
  "policies"
])

export const buildPersonaGardenRoute = ({
  personaId,
  tab,
  sessionId
}: {
  personaId?: string | number | null
  tab?: PersonaGardenTabKey | null
  sessionId?: string | null
} = {}): string => {
  const params = new URLSearchParams()
  const normalizedPersonaId = String(personaId ?? "").trim()
  if (normalizedPersonaId) {
    params.set("persona_id", normalizedPersonaId)
  }
  if (tab && PERSONA_GARDEN_TAB_KEYS.has(tab)) {
    params.set("tab", tab)
  }
  if (tab === "live" && sessionId?.trim()) {
    params.set("session_id", sessionId.trim())
  }
  const query = params.toString()
  return query ? `/persona?${query}` : "/persona"
}

export const readPersonaGardenSearch = (
  search: string
): {
  personaId: string | null
  tab: PersonaGardenTabKey | null
  sessionId: string | null
} => {
  const params = new URLSearchParams(search)
  const personaId = params.get("persona_id")?.trim() || null
  const tabCandidate = params.get("tab")?.trim() || null
  const tab =
    tabCandidate &&
    PERSONA_GARDEN_TAB_KEYS.has(tabCandidate as PersonaGardenTabKey)
      ? (tabCandidate as PersonaGardenTabKey)
      : null
  const sessionId =
    tab === "live" ? params.get("session_id")?.trim() || null : null
  return { personaId, tab, sessionId }
}
