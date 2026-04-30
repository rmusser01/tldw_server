/**
 * Tutorials Module
 * Exports all tutorial-related types, registry, and helpers
 */

// Types
export type {
  TutorialStep,
  TutorialDefinition,
  TutorialSequence
} from "./registry"

// Registry and helpers
export {
  TUTORIAL_REGISTRY,
  getTutorialsForRoute,
  getTutorialById,
  getPrimaryTutorialForRoute,
  getNextTutorialInSequence,
  areTutorialPrerequisitesMet,
  hasTutorialsForRoute,
  getTutorialCountForRoute,
  normalizeTutorialRoute
} from "./registry"

// Tutorial definitions (for direct access if needed)
export { playgroundTutorials } from "./definitions/playground"
export { workspacePlaygroundTutorials } from "./definitions/workspace-playground"
export { mediaTutorials } from "./definitions/media"
export { knowledgeTutorials } from "./definitions/knowledge"
export { charactersTutorials } from "./definitions/characters"
export { promptsTutorials } from "./definitions/prompts"
export { evaluationsTutorials } from "./definitions/evaluations"
export { notesTutorials } from "./definitions/notes"
export { flashcardsTutorials } from "./definitions/flashcards"
export { worldBooksTutorials } from "./definitions/world-books"
export { documentWorkspaceTutorials } from "./definitions/document-workspace"
export { gettingStartedTutorials } from "./definitions/getting-started"
export { ttsTutorials } from "./definitions/tts"
export { sttTutorials } from "./definitions/stt"
export { watchlistsTutorials } from "./definitions/watchlists"
export { monitoringTutorials } from "./definitions/monitoring"
