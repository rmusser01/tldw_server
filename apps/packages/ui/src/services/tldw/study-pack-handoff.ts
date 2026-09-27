import { readFlashcardsGenerateRoute } from "./flashcards-generate-handoff"
export type { StudyPackHandoffSourceType, StudyPackHandoffSourceItem, StudyPackIntent } from "./study-pack-intent"

// Legacy URLs contain no verified owner. Never recover their private metadata.
export const parseStudyPackIntentFromSearch = (_search: string): null => null
export const parseStudyPackIntentFromLocation = (_location: { search?: string; hash?: string }): null => null

export const buildStudyPackRoute = (token?: string | null): string => {
  if (!token) return "/flashcards?tab=importExport"
  if (!/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(token)) {
    throw new Error("The study pack transfer could not be opened. Reopen it from the source.")
  }
  return `/flashcards?tab=importExport&study_pack_handoff=${token}`
}

export const readStudyPackRoute = (location: { pathname?: string; search?: string; hash?: string }) =>
  readFlashcardsGenerateRoute(location, "study-pack")
