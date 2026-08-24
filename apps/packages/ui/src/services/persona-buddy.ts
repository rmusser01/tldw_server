import type {
  PersonaBuddyPreferences,
  PersonaBuddyPreferencesOverrideUpdate,
  PersonaBuddyPreferencesUpdate
} from "@/types/persona-buddy"
import { fetchPersonaVisualJson } from "@/services/persona-visuals"

const buddyPreferencesPath = "/api/v1/persona/buddy/preferences"

export const getBuddyPreferences = (): Promise<PersonaBuddyPreferences> =>
  fetchPersonaVisualJson<PersonaBuddyPreferences>(buddyPreferencesPath)

export const updateBuddyPreferences = (
  input: PersonaBuddyPreferencesUpdate
): Promise<PersonaBuddyPreferences> =>
  fetchPersonaVisualJson<PersonaBuddyPreferences>(buddyPreferencesPath, {
    method: "PATCH",
    body: input
  })

export const updatePersonaBuddyPreferences = (
  personaId: string,
  input: PersonaBuddyPreferencesOverrideUpdate
): Promise<PersonaBuddyPreferences> =>
  fetchPersonaVisualJson<PersonaBuddyPreferences>(
    `/api/v1/persona/profiles/${encodeURIComponent(personaId)}/buddy/preferences`,
    {
      method: "PATCH",
      body: input
    }
  )
