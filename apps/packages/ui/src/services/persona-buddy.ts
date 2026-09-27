import type {
  PersonaBuddyOverridePreferences,
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
): Promise<PersonaBuddyOverridePreferences> =>
  fetchPersonaVisualJson<PersonaBuddyOverridePreferences>(
    `/api/v1/persona/profiles/${encodeURIComponent(personaId)}/buddy/preferences`,
    {
      method: "PATCH",
      body: input
    }
  )

export const getPersonaBuddyPreferences = (
  personaId: string
): Promise<PersonaBuddyOverridePreferences> =>
  fetchPersonaVisualJson<PersonaBuddyOverridePreferences>(
    `/api/v1/persona/profiles/${encodeURIComponent(personaId)}/buddy/preferences`
  )
