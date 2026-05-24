/**
 * Extension route inventory for page review automation.
 * Mirrors packages/ui/src/routes/route-registry.tsx.
 */

export type ExtensionRouteKind = "options" | "sidepanel"

export interface ExtensionRouteEntry {
  kind: ExtensionRouteKind
  path: string
  name: string
  skip?: string
}

export const EXTENSION_ROUTES: ExtensionRouteEntry[] = [
  // Options routes
  { kind: "options", path: "/", name: "Options Home" },
  { kind: "options", path: "/onboarding-test", name: "Onboarding Test" },
  { kind: "options", path: "/settings", name: "Settings" },
  { kind: "options", path: "/settings/tldw", name: "Server Settings" },
  { kind: "options", path: "/settings/model", name: "Model Settings" },
  { kind: "options", path: "/settings/prompt", name: "Prompt Settings" },
  { kind: "options", path: "/settings/evaluations", name: "Evaluations Settings" },
  { kind: "options", path: "/settings/chat", name: "Chat Settings" },
  { kind: "options", path: "/settings/ui", name: "UI Settings" },
  { kind: "options", path: "/settings/quick-ingest", name: "Quick Ingest Settings" },
  { kind: "options", path: "/settings/speech", name: "Speech Settings" },
  { kind: "options", path: "/settings/image-generation", name: "Image Generation Settings" },
  { kind: "options", path: "/settings/share", name: "Share Settings" },
  { kind: "options", path: "/settings/processed", name: "Processed Settings" },
  { kind: "options", path: "/settings/health", name: "Health Settings" },
  { kind: "options", path: "/settings/prompt-studio", name: "Prompt Studio Settings" },
  { kind: "options", path: "/settings/knowledge", name: "Knowledge Settings" },
  { kind: "options", path: "/settings/family-guardrails", name: "Family Guardrails Wizard" },
  { kind: "options", path: "/settings/guardian", name: "Guardian Settings" },
  { kind: "options", path: "/settings/chatbooks", name: "Chatbooks Settings" },
  { kind: "options", path: "/settings/characters", name: "Characters Settings" },
  { kind: "options", path: "/settings/world-books", name: "World Books Settings" },
  { kind: "options", path: "/settings/chat-dictionaries", name: "Dictionaries Settings" },
  { kind: "options", path: "/settings/rag", name: "RAG Settings" },
  { kind: "options", path: "/settings/about", name: "About" },
  { kind: "options", path: "/chunking-playground", name: "Chunking Playground" },
  { kind: "options", path: "/documentation", name: "Documentation" },
  { kind: "options", path: "/review", name: "Review (Media Multi)" },
  { kind: "options", path: "/flashcards", name: "Flashcards" },
  { kind: "options", path: "/quiz", name: "Quiz" },
  { kind: "options", path: "/writing-playground", name: "Writing Playground" },
  { kind: "options", path: "/model-playground", name: "Model Playground" },
  { kind: "options", path: "/chatbooks", name: "Chatbooks" },
  { kind: "options", path: "/watchlists", name: "Watchlists" },
  { kind: "options", path: "/integrations", name: "Integrations" },
  { kind: "options", path: "/admin/integrations", name: "Admin Integrations" },
  { kind: "options", path: "/scheduled-tasks", name: "Scheduled Tasks" },
  { kind: "options", path: "/kanban", name: "Kanban" },
  { kind: "options", path: "/data-tables", name: "Data Tables" },
  { kind: "options", path: "/collections", name: "Collections" },
  { kind: "options", path: "/media", name: "Media" },
  { kind: "options", path: "/media-trash", name: "Media Trash" },
  { kind: "options", path: "/media-multi", name: "Media Multi" },
  { kind: "options", path: "/content-review", name: "Content Review" },
  { kind: "options", path: "/notes", name: "Notes" },
  { kind: "options", path: "/knowledge", name: "Knowledge" },
  { kind: "options", path: "/world-books", name: "World Books" },
  { kind: "options", path: "/dictionaries", name: "Dictionaries" },
  { kind: "options", path: "/characters", name: "Characters" },
  { kind: "options", path: "/prompts", name: "Prompts" },
  { kind: "options", path: "/prompt-studio", name: "Prompt Studio" },
  { kind: "options", path: "/tts", name: "TTS" },
  { kind: "options", path: "/stt", name: "STT" },
  { kind: "options", path: "/speech", name: "Speech" },
  { kind: "options", path: "/evaluations", name: "Evaluations" },
  { kind: "options", path: "/audiobook-studio", name: "Audiobook Studio" },
  { kind: "options", path: "/presentation-studio/start", name: "Presentation Studio Quick Start" },
  { kind: "options", path: "/workflow-editor", name: "Workflow Editor" },
  { kind: "options", path: "/research-workspace", name: "Research Workspace" },
  { kind: "options", path: "/moderation-playground", name: "Moderation Playground" },
  { kind: "options", path: "/admin/server", name: "Admin Server" },
  { kind: "options", path: "/admin/llamacpp", name: "Admin LlamaCpp" },
  { kind: "options", path: "/admin/mlx", name: "Admin MLX" },
  { kind: "options", path: "/quick-chat-popout", name: "Quick Chat Popout" },

  // Sidepanel routes
  { kind: "sidepanel", path: "/", name: "Sidepanel Chat" },
  { kind: "sidepanel", path: "/agent", name: "Sidepanel Agent" },
  { kind: "sidepanel", path: "/companion", name: "Sidepanel Companion" },
  { kind: "sidepanel", path: "/persona", name: "Sidepanel Persona" },
  { kind: "sidepanel", path: "/settings", name: "Sidepanel Settings" },
  { kind: "sidepanel", path: "/error-boundary-test", name: "Sidepanel Error Boundary Test" }
]

export const ACTIVE_EXTENSION_ROUTES = EXTENSION_ROUTES.filter((r) => !r.skip)
export const OPTION_ROUTES = ACTIVE_EXTENSION_ROUTES.filter((r) => r.kind === "options")
export const SIDEPANEL_ROUTES = ACTIVE_EXTENSION_ROUTES.filter((r) => r.kind === "sidepanel")
