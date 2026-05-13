/**
 * Page inventory for smoke tests
 * Generated from actual pages in the tldw-frontend/pages directory
 */

export type PageCategory =
  | "chat"
  | "media"
  | "settings"
  | "admin"
  | "workspace"
  | "knowledge"
  | "audio"
  | "connectors"
  | "other"

export interface PageEntry {
  /** Route path */
  path: string
  /** Human-readable name */
  name: string
  /** Category for grouping */
  category: PageCategory
  /** Optional data-testid to verify page loaded correctly */
  expectedTestId?: string
  /** Skip reason if page is known to be broken or requires special setup */
  skip?: string
}

/**
 * All pages in the tldw-frontend application
 */
export const PAGES: PageEntry[] = [
  // ═══════════════════════════════════════════════════════════════════════════
  // Chat
  // ═══════════════════════════════════════════════════════════════════════════
  { path: "/chat", name: "Chat", category: "chat", expectedTestId: "chat-input" },
  { path: "/chat/agent", name: "Agent Chat", category: "chat" },
  { path: "/persona", name: "Persona Chat", category: "chat" },
  {
    path: "/chat/settings",
    name: "Chat Settings (Page)",
    category: "chat",
    skip: "Covered in Stage 5 release gate; intermittently stalls all-pages webpack route traversal in CI.",
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // Media
  // ═══════════════════════════════════════════════════════════════════════════
  { path: "/media", name: "Media", category: "media" },
  { path: "/media-multi", name: "Media Multi", category: "media" },
  { path: "/media-trash", name: "Media Trash", category: "media" },
  { path: "/media/123/view", name: "Media View (Redirect)", category: "media" },

  // ═══════════════════════════════════════════════════════════════════════════
  // Settings (20+ pages)
  // ═══════════════════════════════════════════════════════════════════════════
  { path: "/settings", name: "Settings", category: "settings" },
  { path: "/settings/tldw", name: "TLDW Settings", category: "settings" },
  { path: "/settings/model", name: "Model Settings", category: "settings" },
  { path: "/settings/chat", name: "Chat Settings", category: "settings" },
  { path: "/settings/prompt", name: "Prompt Settings", category: "settings" },
  { path: "/settings/knowledge", name: "Knowledge Settings", category: "settings" },
  { path: "/settings/rag", name: "RAG Settings", category: "settings" },
  { path: "/settings/speech", name: "Speech Settings", category: "settings" },
  { path: "/settings/evaluations", name: "Evaluations Settings", category: "settings" },
  { path: "/settings/family-guardrails", name: "Family Guardrails Wizard", category: "settings" },
  { path: "/settings/guardian", name: "Guardian Settings", category: "settings" },
  { path: "/settings/chatbooks", name: "Chatbooks Settings", category: "settings" },
  { path: "/settings/characters", name: "Characters Settings", category: "settings" },
  { path: "/settings/world-books", name: "World Books Settings", category: "settings" },
  { path: "/settings/chat-dictionaries", name: "Dictionaries Settings", category: "settings" },
  { path: "/settings/health", name: "Health Settings", category: "settings" },
  { path: "/settings/processed", name: "Processed Settings", category: "settings" },
  { path: "/settings/about", name: "About", category: "settings" },
  { path: "/settings/share", name: "Share Settings", category: "settings" },
  { path: "/settings/quick-ingest", name: "Quick Ingest Settings", category: "settings" },
  { path: "/settings/prompt-studio", name: "Prompt Studio Settings", category: "settings" },

  // ═══════════════════════════════════════════════════════════════════════════
  // Admin
  // ═══════════════════════════════════════════════════════════════════════════
  { path: "/admin", name: "Admin", category: "admin" },
  { path: "/admin/server", name: "Server Admin", category: "admin" },
  { path: "/admin/llamacpp", name: "LlamaCpp Admin", category: "admin" },
  { path: "/admin/mlx", name: "MLX Admin", category: "admin" },
  { path: "/admin/orgs", name: "Orgs Admin", category: "admin" },
  { path: "/admin/data-ops", name: "Data Ops Admin", category: "admin" },
  { path: "/admin/watchlists-items", name: "Watchlists Items", category: "admin" },
  { path: "/admin/watchlists-runs", name: "Watchlists Runs", category: "admin" },
  { path: "/admin/maintenance", name: "Maintenance Admin", category: "admin" },

  // ═══════════════════════════════════════════════════════════════════════════
  // Workspace / Tools
  // ═══════════════════════════════════════════════════════════════════════════
  { path: "/flashcards", name: "Flashcards", category: "workspace" },
  { path: "/quiz", name: "Quiz", category: "workspace" },
  { path: "/moderation", name: "Moderation Review", category: "workspace", expectedTestId: "moderation-review-shell" },
  { path: "/moderation/rules", name: "Content Rules", category: "workspace" },
  {
    path: "/moderation-playground",
    name: "Moderation Playground Legacy Redirect",
    category: "workspace",
    skip: "Legacy alias covered by the route alias contract."
  },
  { path: "/kanban", name: "Kanban", category: "workspace" },
  { path: "/data-tables", name: "Data Tables", category: "workspace" },
  { path: "/content-review", name: "Content Review", category: "workspace" },
  { path: "/claims-review", name: "Claims Review", category: "workspace" },
  { path: "/watchlists", name: "Watchlists", category: "workspace" },
  { path: "/chat-workspace", name: "Chat Workspace", category: "workspace" },
  {
    path: "/chatbooks",
    name: "Chatbooks",
    category: "workspace",
    skip: "Covered in Stage 5 release gate; intermittently stalls all-pages webpack route traversal in CI.",
  },
  { path: "/notes", name: "Notes", category: "workspace" },
  { path: "/collections", name: "Collections", category: "workspace" },
  { path: "/evaluations", name: "Evaluations", category: "workspace" },
  { path: "/search", name: "Search", category: "workspace" },
  { path: "/review", name: "Review", category: "workspace" },
  { path: "/reading", name: "Reading", category: "workspace" },
  { path: "/items", name: "Items", category: "workspace" },
  { path: "/chunking-playground", name: "Chunking Playground", category: "workspace" },

  // ═══════════════════════════════════════════════════════════════════════════
  // Knowledge
  // ═══════════════════════════════════════════════════════════════════════════
  { path: "/knowledge", name: "Knowledge", category: "knowledge" },
  { path: "/world-books", name: "World Books", category: "knowledge" },
  { path: "/dictionaries", name: "Dictionaries", category: "knowledge" },
  { path: "/characters", name: "Characters", category: "knowledge" },
  { path: "/prompts", name: "Prompts", category: "knowledge" },
  { path: "/prompt-studio", name: "Prompt Studio", category: "knowledge" },

  // ═══════════════════════════════════════════════════════════════════════════
  // Audio
  // ═══════════════════════════════════════════════════════════════════════════
  { path: "/tts", name: "TTS", category: "audio" },
  { path: "/stt", name: "STT", category: "audio" },
  { path: "/speech", name: "Speech", category: "audio" },
  { path: "/audio", name: "Audio", category: "audio" },

  // ═══════════════════════════════════════════════════════════════════════════
  // Connectors
  // ═══════════════════════════════════════════════════════════════════════════
  { path: "/connectors", name: "Connectors", category: "connectors" },
  { path: "/connectors/browse", name: "Connectors Browse", category: "connectors" },
  { path: "/connectors/jobs", name: "Connectors Jobs", category: "connectors" },
  { path: "/connectors/sources", name: "Connectors Sources", category: "connectors" },

  // ═══════════════════════════════════════════════════════════════════════════
  // Other / Core Pages
  // ═══════════════════════════════════════════════════════════════════════════
  { path: "/", name: "Home", category: "other" },
  { path: "/login", name: "Login", category: "other" },
  { path: "/setup", name: "Setup", category: "other" },
  { path: "/config", name: "Config", category: "other" },
  { path: "/documentation", name: "Documentation", category: "other" },
  { path: "/profile", name: "Profile", category: "other" },
  { path: "/privileges", name: "Privileges", category: "other" },
  { path: "/quick-chat-popout", name: "Quick Chat Popout", category: "other" },
  { path: "/onboarding-test", name: "Onboarding Test", category: "other" },
  { path: "/for/journalists", name: "For Journalists", category: "other" },
  { path: "/for/osint", name: "For OSINT", category: "other" },
  { path: "/for/researchers", name: "For Researchers", category: "other" },
  { path: "/__debug__/sidepanel-error-boundary", name: "Debug Error Boundary", category: "other" },
  { path: "/__debug__/sidepanel-chat", name: "Debug Sidepanel Chat", category: "other" },
  { path: "/404", name: "Not Found", category: "other" },
  { path: "/account", name: "Account", category: "other" },
  { path: "/notifications", name: "Notifications", category: "other" },
  { path: "/signup", name: "Sign Up", category: "other" },
  { path: "/shared", name: "Shared", category: "other" },
  { path: "/skills", name: "Skills", category: "other" },

  // ═══════════════════════════════════════════════════════════════════════════
  // Workspace / Tools (missing)
  // ═══════════════════════════════════════════════════════════════════════════
  { path: "/acp-playground", name: "ACP Playground", category: "workspace" },
  { path: "/audiobook-studio", name: "Audiobook Studio", category: "workspace" },
  { path: "/chatbooks-playground", name: "Chatbooks Playground", category: "workspace" },
  { path: "/document-workspace", name: "Document Workspace", category: "workspace" },
  { path: "/model-playground", name: "Model Playground", category: "workspace" },
  { path: "/presentation-studio", name: "Presentation Studio", category: "workspace" },
  { path: "/repo2txt", name: "Repo2Txt", category: "workspace" },
  { path: "/workflow-editor", name: "Workflow Editor", category: "workspace" },
  { path: "/workspace-playground", name: "Workspace Playground", category: "workspace" },
  {
    path: "/writing-playground",
    name: "Writing Playground",
    category: "workspace",
    skip: "Covered in Stage 5 release gate; intermittently trips the global error boundary during full all-pages traversal in CI.",
  },
  { path: "/sources", name: "Sources", category: "workspace" },

  // ═══════════════════════════════════════════════════════════════════════════
  // Chat / Agent (missing)
  // ═══════════════════════════════════════════════════════════════════════════
  { path: "/agent-tasks", name: "Agent Tasks", category: "chat" },
  { path: "/agents", name: "Agents", category: "chat" },
  { path: "/chat-workflows", name: "Chat Workflows", category: "chat" },
  { path: "/companion", name: "Companion", category: "chat" },

  // ═══════════════════════════════════════════════════════════════════════════
  // Knowledge (missing)
  // ═══════════════════════════════════════════════════════════════════════════
  { path: "/mcp-hub", name: "MCP Hub", category: "knowledge" },

  // ═══════════════════════════════════════════════════════════════════════════
  // Admin (missing)
  // ═══════════════════════════════════════════════════════════════════════════
  { path: "/admin/api-keys", name: "API Keys Admin", category: "admin" },
  { path: "/admin/billing", name: "Billing Admin", category: "admin" },
  { path: "/admin/monitoring", name: "Monitoring Admin", category: "admin" },
  { path: "/admin/rate-limiting", name: "Rate Limiting Admin", category: "admin" },
  { path: "/admin/rbac", name: "RBAC Admin", category: "admin" },
  { path: "/admin/sources", name: "Sources Admin", category: "admin" },
  { path: "/admin/usage", name: "Usage Admin", category: "admin" },

  // ═══════════════════════════════════════════════════════════════════════════
  // Settings (missing)
  // ═══════════════════════════════════════════════════════════════════════════
  { path: "/settings/image-gen", name: "Image Gen Settings", category: "settings" },
  { path: "/settings/image-generation", name: "Image Generation Settings", category: "settings" },
  { path: "/settings/mcp-hub", name: "MCP Hub Settings", category: "settings" },
  { path: "/settings/splash", name: "Splash Settings", category: "settings" },
  { path: "/settings/ui", name: "UI Settings", category: "settings" },

  // ═══════════════════════════════════════════════════════════════════════════
  // Auth (missing)
  // ═══════════════════════════════════════════════════════════════════════════
  { path: "/auth/magic-link", name: "Magic Link", category: "other" },
  { path: "/auth/reset-password", name: "Reset Password", category: "other" },
  { path: "/auth/verify-email", name: "Verify Email", category: "other" },

  // ═══════════════════════════════════════════════════════════════════════════
  // Billing (missing)
  // ═══════════════════════════════════════════════════════════════════════════
  { path: "/billing", name: "Billing", category: "other" },
  { path: "/billing/success", name: "Billing Success", category: "other" },
  { path: "/billing/cancel", name: "Billing Cancel", category: "other" },

  // ═══════════════════════════════════════════════════════════════════════════
  // Settings Chat (page-level — distinct from settings/chat)
  // ═══════════════════════════════════════════════════════════════════════════
  { path: "/chat/settings", name: "Chat Settings Page", category: "chat", skip: "Covered in Stage 5 release gate" }
]

/**
 * Get pages filtered by category
 */
export function getPagesByCategory(category: PageCategory): PageEntry[] {
  return PAGES.filter((p) => p.category === category && !p.skip)
}

/**
 * Get all non-skipped pages
 */
export function getActivePages(): PageEntry[] {
  return PAGES.filter((p) => !p.skip)
}

/**
 * Get pages that are skipped with reasons
 */
export function getSkippedPages(): PageEntry[] {
  return PAGES.filter((p) => p.skip)
}

/**
 * Total count of pages
 */
export const PAGE_COUNT = PAGES.length
export const ACTIVE_PAGE_COUNT = getActivePages().length
