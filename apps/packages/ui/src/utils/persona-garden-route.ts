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
  tab
}: {
  personaId?: string | number | null
  tab?: PersonaGardenTabKey | null
} = {}): string => {
  const params = new URLSearchParams()
  const normalizedPersonaId = String(personaId ?? "").trim()
  if (normalizedPersonaId) {
    params.set("persona_id", normalizedPersonaId)
  }
  if (tab && PERSONA_GARDEN_TAB_KEYS.has(tab)) {
    params.set("tab", tab)
  }
  const query = params.toString()
  return query ? `/persona?${query}` : "/persona"
}

export const readPersonaGardenSearch = (
  search: string
): {
  personaId: string | null
  tab: PersonaGardenTabKey | null
} => {
  const params = new URLSearchParams(search)
  const personaId = params.get("persona_id")?.trim() || null
  const tabCandidate = params.get("tab")?.trim() || null
  const tab =
    tabCandidate && PERSONA_GARDEN_TAB_KEYS.has(tabCandidate as PersonaGardenTabKey)
      ? (tabCandidate as PersonaGardenTabKey)
      : null
  return { personaId, tab }
}
