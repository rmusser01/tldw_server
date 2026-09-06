import { HOSTED_VISIBLE_OPTION_PATHS } from "./route-hosted-visibility"

export type RouteSurface =
  | "default_self_hosted"
  | "advanced_self_hosted"
  | "hosted_only"
  | "admin_operator"
  | "extension_sidepanel"
  | "labs_beta"
  | "internal_qa_debug"
  | "legacy_alias"
  | "redirect"
  | "deprecated"

export type RouteGroup =
  | "start"
  | "chat"
  | "knowledge"
  | "media_library"
  | "settings"
  | "operations"
  | "workspace"
  | "audio"
  | "study"
  | "safety"
  | "specialized"
  | "documentation"
  | "account"
  | "extension"

export type RouteAvailability =
  | "web"
  | "extension_options"
  | "extension_sidepanel"

export type RouteSmokePolicy = "include" | "exclude" | "manual"

export type RouteNavPolicy = "primary" | "secondary" | "hidden"
export type HostedOptionVisibility = "visible" | "hidden"

export type RouteHeadingPolicy = {
  requiresH1: boolean
  exceptionReason?: string
}

export type RouteMetadata = {
  path: string
  canonicalPath: string
  label: string
  group: RouteGroup
  surface: RouteSurface
  availability: RouteAvailability[]
  aliases?: string[]
  redirectsTo?: string
  smoke: RouteSmokePolicy
  commandPalette: "show" | "hide" | "alias_only"
  nav: RouteNavPolicy
  hostedOptionVisibility: HostedOptionVisibility
  requiresH1?: boolean
  h1ExceptionReason?: string
  requiresAuth?: boolean
  requiresBackend?: boolean
  rationale: string
}

type RouteMetadataInput = Omit<
  RouteMetadata,
  | "canonicalPath"
  | "availability"
  | "smoke"
  | "commandPalette"
  | "nav"
  | "hostedOptionVisibility"
> &
  Partial<
    Pick<RouteMetadata, "canonicalPath" | "smoke" | "commandPalette" | "nav">
  > & {
    availability?: readonly RouteAvailability[]
  }

const web = ["web"] as const
const webAndExtension = ["web", "extension_options"] as const
const extensionOptions = ["extension_options"] as const
const sidepanel = ["extension_sidepanel"] as const

const headingExceptionSurfaces = new Set<RouteSurface>([
  "hosted_only",
  "extension_sidepanel",
  "internal_qa_debug",
  "legacy_alias",
  "redirect",
  "deprecated"
])

const defineRoute = ({
  canonicalPath,
  availability = web,
  smoke = "include",
  commandPalette = "hide",
  nav = "hidden",
  ...metadata
}: RouteMetadataInput): RouteMetadata => ({
  ...metadata,
  canonicalPath: canonicalPath ?? metadata.path,
  availability: [...availability],
  smoke,
  commandPalette,
  nav,
  hostedOptionVisibility: HOSTED_VISIBLE_OPTION_PATHS.has(metadata.path)
    ? "visible"
    : "hidden"
})

export const AUDITED_ROOT_ROUTE_PATHS = [
  "/",
  "/setup",
  "/login",
  "/signup",
  "/account",
  "/profile",
  "/privileges",
  "/config",
  "/billing",
  "/404",
  "/chat",
  "/quick-chat-popout",
  "/persona",
  "/characters",
  "/companion",
  "/agents",
  "/agent-tasks",
  "/chat-workflows",
  "/chat-workspace",
  "/knowledge",
  "/search",
  "/research",
  "/workspaces",
  "/research-workspace",
  "/document-workspace",
  "/repo2txt",
  "/model-playground",
  "/writing-playground",
  "/presentation-studio",
  "/audio-studio",
  "/audiobook-studio",
  "/media",
  "/media-multi",
  "/review",
  "/media-trash",
  "/items",
  "/collections",
  "/reading",
  "/notes",
  "/shared",
  "/chatbooks",
  "/chatbooks-playground",
  "/sources",
  "/connectors",
  "/integrations",
  "/scheduled-tasks",
  "/watchlists",
  "/workflow-editor",
  "/settings",
  "/admin",
  "/mcp-hub",
  "/acp-playground",
  "/prompts",
  "/prompt-studio",
  "/dictionaries",
  "/world-books",
  "/speech",
  "/stt",
  "/tts",
  "/audio",
  "/evaluations",
  "/flashcards",
  "/quiz",
  "/moderation-playground",
  "/content-review",
  "/claims-review",
  "/data-tables",
  "/chunking-playground",
  "/kanban",
  "/skills",
  "/vn-assets",
  "/vn-play",
  "/documentation",
  "/notifications",
  "/composer-variants-preview",
  "/onboarding-test"
] as const

export const ROUTE_METADATA = [
  defineRoute({
    path: "/",
    label: "Home",
    group: "start",
    surface: "default_self_hosted",
    availability: ["web", "extension_options", "extension_sidepanel"],
    commandPalette: "hide",
    nav: "primary",
    requiresBackend: false,
    rationale: "Entry resolver for setup, home, or the extension sidepanel home state."
  }),
  defineRoute({
    path: "/setup",
    label: "Setup",
    group: "start",
    surface: "default_self_hosted",
    availability: webAndExtension,
    nav: "secondary",
    requiresBackend: false,
    rationale: "Connection and first-run setup route."
  }),
  defineRoute({
    path: "/login",
    label: "Login",
    group: "account",
    surface: "hosted_only",
    smoke: "manual",
    requiresBackend: true,
    rationale: "Hosted or multi-user authentication entry point."
  }),
  defineRoute({
    path: "/signup",
    label: "Sign Up",
    group: "account",
    surface: "hosted_only",
    smoke: "manual",
    requiresBackend: true,
    rationale: "Hosted account creation entry point."
  }),
  defineRoute({
    path: "/account",
    label: "Account",
    group: "account",
    surface: "hosted_only",
    smoke: "manual",
    requiresAuth: true,
    rationale: "Hosted account management route."
  }),
  defineRoute({
    path: "/profile",
    label: "Profile",
    group: "account",
    surface: "hosted_only",
    smoke: "manual",
    requiresAuth: true,
    rationale: "User profile and account identity route."
  }),
  defineRoute({
    path: "/privileges",
    label: "Privileges",
    group: "account",
    surface: "advanced_self_hosted",
    smoke: "manual",
    requiresBackend: true,
    rationale: "Privilege inspection route for deployments with authorization features."
  }),
  defineRoute({
    path: "/config",
    label: "Config",
    group: "settings",
    surface: "advanced_self_hosted",
    smoke: "manual",
    requiresBackend: true,
    rationale: "Deployment configuration route surfaced outside the main settings area."
  }),
  defineRoute({
    path: "/billing",
    label: "Billing",
    group: "account",
    surface: "hosted_only",
    smoke: "manual",
    requiresAuth: true,
    rationale: "Hosted billing entry point."
  }),
  defineRoute({
    path: "/404",
    label: "Not Found",
    group: "start",
    surface: "default_self_hosted",
    smoke: "manual",
    rationale: "Recovery route for unknown paths."
  }),
  defineRoute({
    path: "/chat",
    label: "Chat",
    group: "chat",
    surface: "default_self_hosted",
    availability: ["web", "extension_options", "extension_sidepanel"],
    commandPalette: "show",
    nav: "primary",
    requiresBackend: true,
    rationale: "Primary conversation route."
  }),
  defineRoute({
    path: "/quick-chat-popout",
    label: "Quick Chat Popout",
    group: "chat",
    surface: "advanced_self_hosted",
    availability: webAndExtension,
    requiresBackend: true,
    rationale: "Focused popout route for quick chat."
  }),
  defineRoute({
    path: "/persona",
    label: "Persona",
    group: "chat",
    surface: "default_self_hosted",
    availability: ["web", "extension_options", "extension_sidepanel"],
    nav: "secondary",
    requiresBackend: true,
    rationale: "Persona chat and extension persona sidepanel route."
  }),
  defineRoute({
    path: "/characters",
    label: "Characters",
    group: "chat",
    surface: "default_self_hosted",
    availability: webAndExtension,
    nav: "secondary",
    requiresBackend: true,
    rationale: "Character library and role-play setup route."
  }),
  defineRoute({
    path: "/companion",
    label: "Companion",
    group: "chat",
    surface: "default_self_hosted",
    availability: ["web", "extension_options", "extension_sidepanel"],
    nav: "secondary",
    requiresBackend: true,
    rationale: "Companion route shared by WebUI and sidepanel."
  }),
  defineRoute({
    path: "/agents",
    label: "Agents",
    group: "chat",
    surface: "advanced_self_hosted",
    availability: webAndExtension,
    requiresBackend: true,
    rationale: "Agent registry and orchestration entry point."
  }),
  defineRoute({
    path: "/agent-tasks",
    label: "Agent Tasks",
    group: "chat",
    surface: "advanced_self_hosted",
    availability: webAndExtension,
    requiresBackend: true,
    rationale: "Agent task status and unsupported-state route."
  }),
  defineRoute({
    path: "/chat-workflows",
    label: "Chat Workflows",
    group: "chat",
    surface: "advanced_self_hosted",
    availability: webAndExtension,
    requiresBackend: true,
    rationale: "Workflow list entry point from chat-related routes."
  }),
  defineRoute({
    path: "/chat-workspace",
    label: "Chat Workspace",
    group: "workspace",
    surface: "advanced_self_hosted",
    availability: webAndExtension,
    nav: "secondary",
    requiresBackend: true,
    rationale: "Workspace-focused chat route with runtime and approvals."
  }),
  defineRoute({
    path: "/knowledge",
    label: "Knowledge QA",
    group: "knowledge",
    surface: "default_self_hosted",
    availability: webAndExtension,
    commandPalette: "show",
    nav: "primary",
    requiresBackend: true,
    rationale: "Ask questions over selected knowledge sources."
  }),
  defineRoute({
    path: "/search",
    canonicalPath: "/knowledge",
    label: "Search",
    group: "knowledge",
    surface: "legacy_alias",
    aliases: ["/knowledge"],
    commandPalette: "alias_only",
    nav: "hidden",
    requiresBackend: true,
    rationale: "Legacy search-facing entry to the Knowledge QA surface."
  }),
  defineRoute({
    path: "/research",
    label: "Research",
    group: "knowledge",
    surface: "advanced_self_hosted",
    smoke: "manual",
    requiresBackend: true,
    rationale: "Research run route discovered outside the original smoke inventory."
  }),
  defineRoute({
    path: "/workspaces",
    label: "Workspaces",
    group: "workspace",
    surface: "advanced_self_hosted",
    availability: webAndExtension,
    smoke: "manual",
    nav: "secondary",
    requiresBackend: true,
    rationale: "Canonical Workspace manager for research and project workspaces."
  }),
  defineRoute({
    path: "/research-workspace",
    label: "Research Workspace",
    group: "workspace",
    surface: "labs_beta",
    availability: webAndExtension,
    nav: "secondary",
    requiresBackend: true,
    rationale: "Research workspace and source orchestration route."
  }),
  defineRoute({
    path: "/document-workspace",
    label: "Document Workspace",
    group: "workspace",
    surface: "advanced_self_hosted",
    availability: webAndExtension,
    requiresBackend: true,
    rationale: "Document-centered workspace route."
  }),
  defineRoute({
    path: "/repo2txt",
    label: "Repo2Txt",
    group: "workspace",
    surface: "advanced_self_hosted",
    availability: webAndExtension,
    requiresBackend: true,
    rationale: "Repository text export route."
  }),
  defineRoute({
    path: "/model-playground",
    label: "Model Playground",
    group: "workspace",
    surface: "advanced_self_hosted",
    availability: webAndExtension,
    requiresBackend: true,
    rationale: "Model testing and comparison route."
  }),
  defineRoute({
    path: "/writing-playground",
    label: "Writing Playground",
    group: "workspace",
    surface: "advanced_self_hosted",
    availability: webAndExtension,
    smoke: "manual",
    requiresBackend: true,
    rationale: "Writing-focused playground route with known full-suite smoke instability."
  }),
  defineRoute({
    path: "/presentation-studio",
    label: "Presentation Studio",
    group: "workspace",
    surface: "advanced_self_hosted",
    availability: webAndExtension,
    requiresBackend: true,
    rationale: "Presentation creation and editing route."
  }),
  defineRoute({
    path: "/presentation-studio/new",
    label: "New Presentation",
    group: "workspace",
    surface: "advanced_self_hosted",
    availability: web,
    smoke: "manual",
    requiresBackend: true,
    rationale: "WebUI-only presentation authoring entry point."
  }),
  defineRoute({
    path: "/presentation-studio/start",
    label: "Presentation Studio Quick Start",
    group: "workspace",
    surface: "advanced_self_hosted",
    availability: extensionOptions,
    smoke: "manual",
    requiresBackend: true,
    rationale: "Extension-safe structured presentation quick start."
  }),
  defineRoute({
    path: "/presentation-studio/:projectId",
    label: "Presentation Studio Project",
    group: "workspace",
    surface: "advanced_self_hosted",
    availability: webAndExtension,
    smoke: "manual",
    requiresBackend: true,
    rationale:
      "Kind-first project route with a source-free extension handoff for non-structured presentations."
  }),
  defineRoute({
    path: "/audio-studio",
    label: "Audio Studio",
    group: "audio",
    surface: "advanced_self_hosted",
    availability: webAndExtension,
    commandPalette: "show",
    nav: "secondary",
    requiresBackend: true,
    aliases: ["/audiobook-studio"],
    rationale:
      "Workflow-first audio production route for narration, podcast, briefing, and music."
  }),
  defineRoute({
    path: "/audiobook-studio",
    label: "Audiobook Studio",
    canonicalPath: "/audio-studio",
    group: "audio",
    surface: "legacy_alias",
    availability: webAndExtension,
    redirectsTo: "/audio-studio?workflow=narration",
    commandPalette: "alias_only",
    nav: "hidden",
    requiresBackend: true,
    rationale:
      "Compatibility route that sends legacy audiobook users into Audio Studio Narration."
  }),
  defineRoute({
    path: "/media",
    label: "Media",
    group: "media_library",
    surface: "default_self_hosted",
    availability: webAndExtension,
    commandPalette: "show",
    nav: "primary",
    requiresBackend: true,
    rationale: "Primary library browsing and media inspection route."
  }),
  defineRoute({
    path: "/media-multi",
    label: "Media Multi Select",
    group: "media_library",
    surface: "advanced_self_hosted",
    availability: webAndExtension,
    requiresBackend: true,
    rationale: "Bulk media selection and review route."
  }),
  defineRoute({
    path: "/review",
    label: "Review",
    group: "media_library",
    surface: "advanced_self_hosted",
    availability: webAndExtension,
    requiresBackend: true,
    rationale: "Media review route and queue-style workflow entry point."
  }),
  defineRoute({
    path: "/media-trash",
    label: "Media Trash",
    group: "media_library",
    surface: "advanced_self_hosted",
    availability: webAndExtension,
    requiresBackend: true,
    rationale: "Deleted media recovery and cleanup route."
  }),
  defineRoute({
    path: "/items",
    label: "Items",
    group: "media_library",
    surface: "advanced_self_hosted",
    requiresBackend: true,
    rationale: "Item workspace route for library objects."
  }),
  defineRoute({
    path: "/collections",
    label: "Collections",
    group: "media_library",
    surface: "default_self_hosted",
    availability: webAndExtension,
    nav: "secondary",
    requiresBackend: true,
    rationale: "Collection organization route."
  }),
  defineRoute({
    path: "/reading",
    label: "Reading",
    group: "media_library",
    surface: "default_self_hosted",
    requiresBackend: true,
    rationale: "Reading-list and reading workspace route."
  }),
  defineRoute({
    path: "/notes",
    label: "Notes",
    group: "media_library",
    surface: "default_self_hosted",
    availability: webAndExtension,
    commandPalette: "show",
    nav: "primary",
    requiresBackend: true,
    rationale: "Notes workspace route."
  }),
  defineRoute({
    path: "/shared",
    label: "Shared",
    group: "media_library",
    surface: "advanced_self_hosted",
    availability: webAndExtension,
    requiresBackend: true,
    rationale: "Shared content and workspaces route."
  }),
  defineRoute({
    path: "/chatbooks",
    label: "Chatbooks Backup & Import",
    group: "media_library",
    surface: "advanced_self_hosted",
    availability: webAndExtension,
    smoke: "manual",
    requiresBackend: true,
    rationale: "Primary account backup, archive import, and selective Chatbook export route with known full-suite smoke instability."
  }),
  defineRoute({
    path: "/chatbooks-playground",
    canonicalPath: "/chatbooks",
    label: "Chatbooks Backup & Import",
    group: "media_library",
    surface: "legacy_alias",
    availability: webAndExtension,
    smoke: "manual",
    commandPalette: "alias_only",
    redirectsTo: "/chatbooks",
    requiresBackend: true,
    rationale: "Legacy route name retained for compatibility; use /chatbooks for the Backup & Import workflow."
  }),
  defineRoute({
    path: "/sources",
    label: "Sources",
    group: "operations",
    surface: "default_self_hosted",
    availability: webAndExtension,
    nav: "secondary",
    requiresBackend: true,
    rationale: "Source management route called out by the audit for capability-state remediation."
  }),
  defineRoute({
    path: "/connectors",
    label: "Connectors",
    group: "operations",
    surface: "advanced_self_hosted",
    requiresBackend: true,
    rationale: "Connector hub route."
  }),
  defineRoute({
    path: "/integrations",
    label: "Integrations",
    group: "operations",
    surface: "advanced_self_hosted",
    availability: webAndExtension,
    nav: "secondary",
    requiresBackend: true,
    rationale: "Integration setup and discovery route."
  }),
  defineRoute({
    path: "/scheduled-tasks",
    label: "Scheduled Tasks",
    group: "operations",
    surface: "advanced_self_hosted",
    availability: webAndExtension,
    nav: "secondary",
    requiresBackend: true,
    rationale: "Automation schedule management route."
  }),
  defineRoute({
    path: "/watchlists",
    label: "Watchlists",
    group: "operations",
    surface: "labs_beta",
    availability: webAndExtension,
    nav: "secondary",
    requiresBackend: true,
    rationale: "Watchlist monitoring route with beta workflow framing."
  }),
  defineRoute({
    path: "/workflow-editor",
    label: "Workflow Editor",
    group: "operations",
    surface: "advanced_self_hosted",
    availability: webAndExtension,
    requiresBackend: true,
    rationale: "Workflow editing route."
  }),
  defineRoute({
    path: "/settings",
    label: "Settings",
    group: "settings",
    surface: "default_self_hosted",
    availability: ["web", "extension_options", "extension_sidepanel"],
    commandPalette: "show",
    nav: "primary",
    requiresBackend: false,
    rationale: "General settings route and sidepanel settings target."
  }),
  defineRoute({
    path: "/settings/model",
    label: "Model Settings",
    group: "settings",
    surface: "default_self_hosted",
    availability: webAndExtension,
    nav: "secondary",
    requiresBackend: false,
    rationale: "Model provider configuration route in the settings area."
  }),
  defineRoute({
    path: "/settings/health",
    label: "Health & Diagnostics",
    group: "settings",
    surface: "default_self_hosted",
    availability: webAndExtension,
    commandPalette: "show",
    nav: "secondary",
    requiresBackend: false,
    rationale: "Connection diagnostics, health checks, and recovery route."
  }),
  defineRoute({
    path: "/settings/image-generation",
    label: "Image Generation Settings",
    group: "settings",
    surface: "advanced_self_hosted",
    availability: webAndExtension,
    requiresBackend: true,
    rationale: "Image generation provider and model configuration route."
  }),
  defineRoute({
    path: "/admin",
    label: "Admin",
    group: "settings",
    surface: "admin_operator",
    availability: webAndExtension,
    smoke: "manual",
    commandPalette: "show",
    requiresBackend: true,
    rationale: "Admin landing route for operator-only workflows."
  }),
  defineRoute({
    path: "/admin/server",
    label: "Server Admin",
    group: "settings",
    surface: "admin_operator",
    availability: webAndExtension,
    commandPalette: "show",
    requiresBackend: true,
    rationale: "Operator server status, user, role, and maintenance route."
  }),
  defineRoute({
    path: "/admin/mlx",
    label: "Admin MLX",
    group: "settings",
    surface: "admin_operator",
    smoke: "manual",
    commandPalette: "show",
    requiresBackend: true,
    rationale: "Admin route for local MLX model status and operator controls."
  }),
  // Every admin module is registered here so the command palette and any
  // metadata-driven navigation can find the full admin surface — the 2026-09
  // UX audit (finding S1) flagged that most admin routes were URL-only.
  defineRoute({
    path: "/admin/llamacpp",
    label: "Admin Llama.cpp",
    group: "settings",
    surface: "admin_operator",
    availability: webAndExtension,
    smoke: "manual",
    commandPalette: "show",
    requiresBackend: true,
    rationale: "Admin route for the managed llama.cpp inference server."
  }),
  defineRoute({
    path: "/admin/runtime-config",
    label: "Admin Runtime Config",
    group: "settings",
    surface: "admin_operator",
    availability: webAndExtension,
    smoke: "manual",
    commandPalette: "show",
    requiresBackend: true,
    rationale: "Admin route for runtime configuration inspection."
  }),
  defineRoute({
    path: "/admin/monitoring",
    label: "Monitoring",
    group: "settings",
    surface: "admin_operator",
    availability: webAndExtension,
    smoke: "manual",
    commandPalette: "show",
    requiresBackend: true,
    rationale: "Admin route for health metrics, alerts, and telemetry."
  }),
  defineRoute({
    path: "/admin/integrations",
    label: "Workspace Integrations",
    group: "settings",
    surface: "admin_operator",
    availability: webAndExtension,
    smoke: "manual",
    commandPalette: "show",
    requiresBackend: true,
    rationale: "Admin route for Slack/Discord/Telegram workspace policy."
  }),
  defineRoute({
    path: "/admin/sources",
    label: "Admin Sources",
    group: "settings",
    surface: "admin_operator",
    availability: webAndExtension,
    smoke: "manual",
    commandPalette: "show",
    requiresBackend: true,
    rationale: "Admin route for ingestion source availability and sync state."
  }),
  defineRoute({
    path: "/admin/api-keys",
    label: "Admin API Keys",
    group: "settings",
    surface: "admin_operator",
    availability: webAndExtension,
    smoke: "manual",
    commandPalette: "show",
    requiresBackend: true,
    rationale: "Admin route for creating, rotating, and revoking user API keys."
  }),
  defineRoute({
    path: "/admin/rbac",
    label: "Admin Roles & Permissions",
    group: "settings",
    surface: "admin_operator",
    availability: webAndExtension,
    smoke: "manual",
    commandPalette: "show",
    requiresBackend: true,
    rationale: "Admin route for the RBAC permission matrix and role grants."
  }),
  defineRoute({
    path: "/admin/orgs",
    label: "Admin Organizations",
    group: "settings",
    surface: "admin_operator",
    availability: webAndExtension,
    smoke: "manual",
    commandPalette: "show",
    requiresBackend: true,
    rationale: "Admin route for organizations and teams in multi-user mode."
  }),
  defineRoute({
    path: "/admin/rate-limiting",
    label: "Admin Rate Limiting",
    group: "settings",
    surface: "admin_operator",
    availability: webAndExtension,
    smoke: "manual",
    commandPalette: "show",
    requiresBackend: true,
    rationale: "Admin route for resource governor policy and rate limits."
  }),
  defineRoute({
    path: "/admin/data-ops",
    label: "Admin Data Operations",
    group: "settings",
    surface: "admin_operator",
    availability: webAndExtension,
    smoke: "manual",
    commandPalette: "show",
    requiresBackend: true,
    rationale: "Admin route for backups, retention, and data subject requests."
  }),
  defineRoute({
    path: "/admin/usage",
    label: "Admin Usage Analytics",
    group: "settings",
    surface: "admin_operator",
    availability: webAndExtension,
    smoke: "manual",
    commandPalette: "show",
    requiresBackend: true,
    rationale: "Admin route for request, storage, and LLM usage analytics."
  }),
  defineRoute({
    path: "/admin/billing",
    label: "Admin Billing",
    group: "settings",
    surface: "admin_operator",
    availability: webAndExtension,
    smoke: "manual",
    commandPalette: "show",
    requiresBackend: true,
    rationale: "Admin route for subscriptions and billing events."
  }),
  defineRoute({
    path: "/admin/maintenance",
    label: "Admin Maintenance",
    group: "settings",
    surface: "admin_operator",
    availability: webAndExtension,
    smoke: "manual",
    commandPalette: "show",
    requiresBackend: true,
    rationale: "Admin route for maintenance mode, feature flags, and incidents."
  }),
  defineRoute({
    path: "/admin/watchlists-items",
    label: "Admin Watchlist Items",
    group: "settings",
    surface: "admin_operator",
    availability: webAndExtension,
    smoke: "manual",
    commandPalette: "show",
    requiresBackend: true,
    rationale: "Admin route for reviewing collected watchlist updates."
  }),
  defineRoute({
    path: "/admin/watchlists-runs",
    label: "Admin Watchlist Runs",
    group: "settings",
    surface: "admin_operator",
    smoke: "manual",
    commandPalette: "show",
    requiresBackend: true,
    rationale: "Admin route for watchlist run history (planned surface)."
  }),
  defineRoute({
    path: "/mcp-hub",
    label: "MCP Hub",
    group: "operations",
    surface: "advanced_self_hosted",
    availability: webAndExtension,
    commandPalette: "show",
    nav: "secondary",
    requiresBackend: true,
    rationale: "MCP setup and operations hub."
  }),
  defineRoute({
    path: "/acp-playground",
    label: "ACP Playground",
    group: "operations",
    surface: "advanced_self_hosted",
    availability: webAndExtension,
    requiresBackend: true,
    rationale: "ACP testing and agent protocol playground route."
  }),
  defineRoute({
    path: "/prompts",
    label: "Prompts",
    group: "knowledge",
    surface: "default_self_hosted",
    availability: webAndExtension,
    commandPalette: "show",
    nav: "primary",
    requiresBackend: true,
    rationale: "Prompt library and prompt-studio home route."
  }),
  defineRoute({
    path: "/prompt-studio",
    canonicalPath: "/prompts",
    label: "Prompt Studio",
    group: "knowledge",
    surface: "legacy_alias",
    availability: webAndExtension,
    aliases: ["/prompts?tab=studio"],
    redirectsTo: "/prompts?tab=studio",
    commandPalette: "alias_only",
    nav: "hidden",
    requiresBackend: true,
    rationale: "Legacy prompt-studio route that redirects into the Prompts surface."
  }),
  defineRoute({
    path: "/dictionaries",
    label: "Dictionaries",
    group: "knowledge",
    surface: "default_self_hosted",
    availability: webAndExtension,
    nav: "secondary",
    requiresBackend: true,
    rationale: "Chat dictionary management route."
  }),
  defineRoute({
    path: "/world-books",
    label: "World Books",
    group: "knowledge",
    surface: "default_self_hosted",
    availability: webAndExtension,
    nav: "secondary",
    requiresBackend: true,
    rationale: "World book and lorebook management route."
  }),
  defineRoute({
    path: "/speech",
    label: "Speech",
    group: "audio",
    surface: "default_self_hosted",
    availability: webAndExtension,
    nav: "secondary",
    requiresBackend: true,
    rationale: "Speech overview route for STT/TTS readiness."
  }),
  defineRoute({
    path: "/stt",
    label: "Speech to Text",
    group: "audio",
    surface: "default_self_hosted",
    availability: webAndExtension,
    requiresBackend: true,
    rationale: "Speech-to-text workflow route."
  }),
  defineRoute({
    path: "/tts",
    label: "Text to Speech",
    group: "audio",
    surface: "default_self_hosted",
    availability: webAndExtension,
    requiresBackend: true,
    rationale: "Text-to-speech workflow route."
  }),
  defineRoute({
    path: "/audio",
    canonicalPath: "/speech",
    label: "Audio",
    group: "audio",
    surface: "legacy_alias",
    aliases: ["/speech"],
    commandPalette: "alias_only",
    nav: "hidden",
    requiresBackend: true,
    rationale: "Legacy audio alias retained for compatibility with the speech route family."
  }),
  defineRoute({
    path: "/evaluations",
    label: "Evaluations",
    group: "study",
    surface: "advanced_self_hosted",
    availability: webAndExtension,
    requiresBackend: true,
    rationale: "Evaluation setup and result inspection route."
  }),
  defineRoute({
    path: "/flashcards",
    label: "Flashcards",
    group: "study",
    surface: "default_self_hosted",
    availability: ["web", "extension_options", "extension_sidepanel"],
    commandPalette: "show",
    nav: "secondary",
    requiresBackend: true,
    rationale: "Flashcard study and sidepanel review route."
  }),
  defineRoute({
    path: "/quiz",
    label: "Quiz",
    group: "study",
    surface: "default_self_hosted",
    availability: webAndExtension,
    requiresBackend: true,
    rationale: "Quiz and study route."
  }),
  defineRoute({
    path: "/moderation-playground",
    canonicalPath: "/moderation/rules",
    redirectsTo: "/moderation/rules",
    label: "Moderation Playground",
    group: "safety",
    surface: "redirect",
    availability: webAndExtension,
    smoke: "exclude",
    commandPalette: "alias_only",
    requiresBackend: true,
    rationale:
      "Legacy alias that redirects to the canonical moderation rules workflow."
  }),
  defineRoute({
    path: "/content-review",
    label: "Content Review",
    group: "safety",
    surface: "advanced_self_hosted",
    smoke: "manual",
    requiresBackend: true,
    rationale: "Content review queue route."
  }),
  defineRoute({
    path: "/claims-review",
    canonicalPath: "/content-review",
    redirectsTo: "/content-review",
    label: "Claims Review",
    group: "safety",
    surface: "redirect",
    smoke: "exclude",
    commandPalette: "alias_only",
    requiresBackend: true,
    rationale: "Legacy alias that redirects to the content review queue."
  }),
  defineRoute({
    path: "/data-tables",
    label: "Data Tables",
    group: "specialized",
    surface: "advanced_self_hosted",
    availability: webAndExtension,
    requiresBackend: true,
    rationale: "Structured data table route."
  }),
  defineRoute({
    path: "/chunking-playground",
    label: "Chunking Playground",
    group: "specialized",
    surface: "advanced_self_hosted",
    availability: webAndExtension,
    requiresBackend: true,
    rationale: "Chunking strategy playground route."
  }),
  defineRoute({
    path: "/kanban",
    label: "Kanban",
    group: "specialized",
    surface: "advanced_self_hosted",
    availability: webAndExtension,
    requiresBackend: true,
    rationale: "Kanban board route."
  }),
  defineRoute({
    path: "/skills",
    label: "Skills",
    group: "operations",
    surface: "advanced_self_hosted",
    availability: webAndExtension,
    requiresBackend: true,
    rationale: "Skills discovery and management route."
  }),
  defineRoute({
    path: "/vn-assets",
    label: "VN Assets",
    group: "specialized",
    surface: "labs_beta",
    smoke: "manual",
    requiresBackend: true,
    rationale: "Visual novel asset route discovered outside the original smoke inventory."
  }),
  defineRoute({
    path: "/vn-play",
    label: "VN Play",
    group: "specialized",
    surface: "labs_beta",
    smoke: "manual",
    requiresBackend: true,
    rationale: "Visual novel play route discovered outside the original smoke inventory."
  }),
  defineRoute({
    path: "/documentation",
    label: "Documentation",
    group: "documentation",
    surface: "default_self_hosted",
    availability: webAndExtension,
    commandPalette: "show",
    nav: "secondary",
    requiresBackend: false,
    rationale: "In-app documentation route."
  }),
  defineRoute({
    path: "/notifications",
    label: "Notifications",
    group: "operations",
    surface: "default_self_hosted",
    requiresBackend: true,
    rationale: "Notification inbox and alert route."
  }),
  defineRoute({
    path: "/composer-variants-preview",
    label: "Composer Variants Preview",
    group: "specialized",
    surface: "internal_qa_debug",
    smoke: "exclude",
    rationale: "Internal composer preview route, not normal product navigation."
  }),
  defineRoute({
    path: "/onboarding-test",
    label: "Onboarding Test Harness",
    group: "start",
    surface: "internal_qa_debug",
    availability: webAndExtension,
    smoke: "exclude",
    rationale: "Internal onboarding QA route."
  }),
  defineRoute({
    path: "/agent",
    canonicalPath: "/agents",
    label: "Agent Sidepanel",
    group: "extension",
    surface: "extension_sidepanel",
    availability: sidepanel,
    smoke: "exclude",
    rationale: "Extension-only sidepanel agent route."
  }),
  defineRoute({
    path: "/clipper",
    label: "Clipper",
    group: "extension",
    surface: "extension_sidepanel",
    availability: sidepanel,
    smoke: "exclude",
    rationale: "Extension-only page clipping route."
  }),
  defineRoute({
    path: "/companion/conversation",
    canonicalPath: "/companion",
    label: "Companion Conversation",
    group: "extension",
    surface: "extension_sidepanel",
    availability: sidepanel,
    smoke: "exclude",
    rationale: "Extension-only companion conversation sidepanel route."
  }),
  defineRoute({
    path: "/error-boundary-test",
    label: "Error Boundary Test",
    group: "extension",
    surface: "internal_qa_debug",
    availability: sidepanel,
    smoke: "exclude",
    rationale: "Sidepanel QA route for error boundary testing."
  }),
  defineRoute({
    path: "/__debug__/sidepanel-chat",
    label: "Debug Sidepanel Chat",
    group: "extension",
    surface: "internal_qa_debug",
    smoke: "exclude",
    rationale: "Web debug route for sidepanel chat rendering."
  }),
  defineRoute({
    path: "/__debug__/mermaid-chat-cards",
    label: "Debug Mermaid Chat Cards",
    group: "extension",
    surface: "internal_qa_debug",
    smoke: "exclude",
    rationale: "Web debug route for Mermaid chat-card browser QA."
  }),
  defineRoute({
    path: "/__debug__/sidepanel-error-boundary",
    label: "Debug Sidepanel Error Boundary",
    group: "extension",
    surface: "internal_qa_debug",
    smoke: "exclude",
    rationale: "Web debug route for sidepanel error-boundary rendering."
  }),
  // ── Governance backfill (2026-09): every active smoke-inventory and
  // shared-registry route is registered so metadata-coverage and heading
  // governance hold. requiresH1 is explicitly false with an exception
  // reason until each page passes a heading audit.
  defineRoute({
    path: "/settings/preferences",
    label: "Preferences Settings",
    group: "settings",
    surface: "default_self_hosted",
    requiresH1: false,
    h1ExceptionReason: "Pre-governance page pending heading audit.",
    rationale: "Startup and general preference settings route."
  }),
  defineRoute({
    path: "/settings/tldw",
    label: "TLDW Settings",
    group: "settings",
    surface: "default_self_hosted",
    requiresH1: false,
    h1ExceptionReason: "Pre-governance page pending heading audit.",
    rationale: "tldw server connection settings route."
  }),
  defineRoute({
    path: "/settings/chat",
    label: "Chat Settings",
    group: "settings",
    surface: "default_self_hosted",
    requiresH1: false,
    h1ExceptionReason: "Pre-governance page pending heading audit.",
    rationale: "Chat behavior settings route."
  }),
  defineRoute({
    path: "/settings/chat-macros",
    label: "Chat Macros Settings",
    group: "settings",
    surface: "default_self_hosted",
    requiresH1: false,
    h1ExceptionReason: "Pre-governance page pending heading audit.",
    rationale: "Chat macro management settings route."
  }),
  defineRoute({
    path: "/settings/prompt",
    label: "Workflow Prompts Settings",
    group: "settings",
    surface: "default_self_hosted",
    requiresH1: false,
    h1ExceptionReason: "Pre-governance page pending heading audit.",
    rationale: "Service and workflow prompt settings route."
  }),
  defineRoute({
    path: "/settings/knowledge",
    label: "Knowledge Settings",
    group: "settings",
    surface: "default_self_hosted",
    requiresH1: false,
    h1ExceptionReason: "Pre-governance page pending heading audit.",
    rationale: "Knowledge base settings route."
  }),
  defineRoute({
    path: "/settings/rag",
    label: "RAG Settings",
    group: "settings",
    surface: "default_self_hosted",
    requiresH1: false,
    h1ExceptionReason: "Pre-governance page pending heading audit.",
    rationale: "Retrieval-augmented generation settings route."
  }),
  defineRoute({
    path: "/settings/speech",
    label: "Speech Settings",
    group: "settings",
    surface: "default_self_hosted",
    requiresH1: false,
    h1ExceptionReason: "Pre-governance page pending heading audit.",
    rationale: "Speech-to-text and text-to-speech settings route."
  }),
  defineRoute({
    path: "/settings/evaluations",
    label: "Evaluations Settings",
    group: "settings",
    surface: "default_self_hosted",
    requiresH1: false,
    h1ExceptionReason: "Pre-governance page pending heading audit.",
    rationale: "Evaluation configuration settings route."
  }),
  defineRoute({
    path: "/settings/family-guardrails",
    label: "Family Guardrails Wizard",
    group: "settings",
    surface: "default_self_hosted",
    requiresH1: false,
    h1ExceptionReason: "Pre-governance page pending heading audit.",
    rationale: "Family guardrails setup wizard route."
  }),
  defineRoute({
    path: "/settings/guardian",
    label: "Guardian Settings",
    group: "settings",
    surface: "default_self_hosted",
    requiresH1: false,
    h1ExceptionReason: "Pre-governance page pending heading audit.",
    rationale: "Guardian oversight settings route."
  }),
  defineRoute({
    path: "/settings/chatbooks",
    label: "Chatbooks Settings",
    group: "settings",
    surface: "default_self_hosted",
    requiresH1: false,
    h1ExceptionReason: "Pre-governance page pending heading audit.",
    rationale: "Chatbook export/import settings route."
  }),
  defineRoute({
    path: "/settings/characters",
    label: "Characters Settings",
    group: "settings",
    surface: "default_self_hosted",
    requiresH1: false,
    h1ExceptionReason: "Pre-governance page pending heading audit.",
    rationale: "Character defaults settings route."
  }),
  defineRoute({
    path: "/settings/world-books",
    label: "World Books Settings",
    group: "settings",
    surface: "default_self_hosted",
    requiresH1: false,
    h1ExceptionReason: "Pre-governance page pending heading audit.",
    rationale: "World book settings route."
  }),
  defineRoute({
    path: "/settings/chat-dictionaries",
    label: "Dictionaries Settings",
    group: "settings",
    surface: "default_self_hosted",
    requiresH1: false,
    h1ExceptionReason: "Pre-governance page pending heading audit.",
    rationale: "Chat dictionary settings route."
  }),
  defineRoute({
    path: "/settings/processed",
    label: "Processed Settings",
    group: "settings",
    surface: "default_self_hosted",
    requiresH1: false,
    h1ExceptionReason: "Pre-governance page pending heading audit.",
    rationale: "Processed content review settings route."
  }),
  defineRoute({
    path: "/settings/data",
    label: "Data Management Settings",
    group: "settings",
    surface: "default_self_hosted",
    requiresH1: false,
    h1ExceptionReason: "Pre-governance page pending heading audit.",
    rationale: "Data management and cleanup settings route."
  }),
  defineRoute({
    path: "/settings/about",
    label: "About",
    group: "settings",
    surface: "default_self_hosted",
    requiresH1: false,
    h1ExceptionReason: "Pre-governance page pending heading audit.",
    rationale: "About and version information route."
  }),
  defineRoute({
    path: "/settings/share",
    label: "Share Settings",
    group: "settings",
    surface: "default_self_hosted",
    requiresH1: false,
    h1ExceptionReason: "Pre-governance page pending heading audit.",
    rationale: "Content sharing settings route."
  }),
  defineRoute({
    path: "/settings/quick-ingest",
    label: "Quick Ingest Settings",
    group: "settings",
    surface: "default_self_hosted",
    requiresH1: false,
    h1ExceptionReason: "Pre-governance page pending heading audit.",
    rationale: "Quick ingest defaults settings route."
  }),
  defineRoute({
    path: "/settings/prompt-studio",
    label: "Prompt Studio Settings",
    group: "settings",
    surface: "default_self_hosted",
    requiresH1: false,
    h1ExceptionReason: "Pre-governance page pending heading audit.",
    rationale: "Prompt studio settings route."
  }),
  defineRoute({
    path: "/settings/mcp-hub",
    label: "MCP Hub Settings",
    group: "settings",
    surface: "advanced_self_hosted",
    requiresH1: false,
    h1ExceptionReason: "Pre-governance page pending heading audit.",
    rationale: "MCP hub settings alias inside the settings area."
  }),
  defineRoute({
    path: "/settings/splash",
    label: "Splash Settings",
    group: "settings",
    surface: "default_self_hosted",
    requiresH1: false,
    h1ExceptionReason: "Pre-governance page pending heading audit.",
    rationale: "Splash screen settings route."
  }),
  defineRoute({
    path: "/settings/ui",
    label: "UI Settings",
    group: "settings",
    surface: "default_self_hosted",
    requiresH1: false,
    h1ExceptionReason: "Pre-governance page pending heading audit.",
    rationale: "Interface appearance settings route."
  }),
  defineRoute({
    path: "/settings/image-gen",
    label: "Image Gen Settings",
    group: "settings",
    surface: "default_self_hosted",
    smoke: "manual",
    requiresH1: false,
    h1ExceptionReason: "Pre-governance page pending heading audit.",
    rationale: "Image generation settings alias; smoke-skipped pending Stage 5 gate."
  }),
  defineRoute({
    path: "/media/123/view",
    label: "Media View (Redirect)",
    group: "media_library",
    surface: "redirect",
    smoke: "manual",
    requiresH1: false,
    h1ExceptionReason: "Pre-governance page pending heading audit.",
    rationale: "Sample media detail redirect covered by the media route contract."
  }),
  defineRoute({
    path: "/chat/agent",
    label: "Agent Chat",
    group: "chat",
    surface: "advanced_self_hosted",
    requiresH1: false,
    h1ExceptionReason: "Pre-governance page pending heading audit.",
    rationale: "Agent-driven chat surface route."
  }),
  defineRoute({
    path: "/moderation",
    label: "Moderation Review",
    group: "safety",
    surface: "advanced_self_hosted",
    requiresH1: false,
    h1ExceptionReason: "Pre-governance page pending heading audit.",
    rationale: "Moderation review queue route."
  }),
  defineRoute({
    path: "/moderation/rules",
    label: "Content Rules",
    group: "safety",
    surface: "advanced_self_hosted",
    requiresH1: false,
    h1ExceptionReason: "Pre-governance page pending heading audit.",
    rationale: "Moderation content rules route."
  }),
  defineRoute({
    path: "/connectors/browse",
    label: "Connector Catalog",
    group: "operations",
    surface: "advanced_self_hosted",
    requiresH1: false,
    h1ExceptionReason: "Pre-governance page pending heading audit.",
    rationale: "Connector browsing route."
  }),
  defineRoute({
    path: "/connectors/jobs",
    label: "Connector Jobs",
    group: "operations",
    surface: "advanced_self_hosted",
    requiresH1: false,
    h1ExceptionReason: "Pre-governance page pending heading audit.",
    rationale: "Connector job monitoring route."
  }),
  defineRoute({
    path: "/connectors/sources",
    label: "Connector Sources",
    group: "operations",
    surface: "advanced_self_hosted",
    requiresH1: false,
    h1ExceptionReason: "Pre-governance page pending heading audit.",
    rationale: "Connector source management route."
  }),
  defineRoute({
    path: "/for/journalists",
    label: "For Journalists",
    group: "documentation",
    surface: "default_self_hosted",
    requiresH1: false,
    h1ExceptionReason: "Pre-governance page pending heading audit.",
    rationale: "Persona landing page for journalists."
  }),
  defineRoute({
    path: "/for/osint",
    label: "For OSINT",
    group: "documentation",
    surface: "default_self_hosted",
    requiresH1: false,
    h1ExceptionReason: "Pre-governance page pending heading audit.",
    rationale: "Persona landing page for OSINT analysts."
  }),
  defineRoute({
    path: "/for/researchers",
    label: "For Researchers",
    group: "documentation",
    surface: "default_self_hosted",
    requiresH1: false,
    h1ExceptionReason: "Pre-governance page pending heading audit.",
    rationale: "Persona landing page for researchers."
  }),
  defineRoute({
    path: "/auth/magic-link",
    label: "Magic Link",
    group: "account",
    surface: "default_self_hosted",
    requiresH1: false,
    h1ExceptionReason: "Pre-governance page pending heading audit.",
    rationale: "Magic-link sign-in landing route."
  }),
  defineRoute({
    path: "/auth/reset-password",
    label: "Reset Password",
    group: "account",
    surface: "default_self_hosted",
    requiresH1: false,
    h1ExceptionReason: "Pre-governance page pending heading audit.",
    rationale: "Password reset flow route."
  }),
  defineRoute({
    path: "/auth/verify-email",
    label: "Verify Email",
    group: "account",
    surface: "default_self_hosted",
    requiresH1: false,
    h1ExceptionReason: "Pre-governance page pending heading audit.",
    rationale: "Email verification flow route."
  }),
  defineRoute({
    path: "/billing/success",
    label: "Billing Success",
    group: "account",
    surface: "hosted_only",
    requiresH1: false,
    h1ExceptionReason: "Pre-governance page pending heading audit.",
    rationale: "Billing checkout success landing route."
  }),
  defineRoute({
    path: "/billing/cancel",
    label: "Billing Cancel",
    group: "account",
    surface: "hosted_only",
    requiresH1: false,
    h1ExceptionReason: "Pre-governance page pending heading audit.",
    rationale: "Billing checkout cancellation landing route."
  }),
  defineRoute({
    path: "/sources/new",
    label: "New Source",
    group: "operations",
    surface: "default_self_hosted",
    requiresH1: false,
    h1ExceptionReason: "Pre-governance page pending heading audit.",
    rationale: "Source creation route."
  }),
  defineRoute({
    path: "/sources/:sourceId",
    label: "Source Detail",
    group: "operations",
    surface: "default_self_hosted",
    smoke: "manual",
    requiresH1: false,
    h1ExceptionReason: "Pre-governance page pending heading audit.",
    rationale: "Source detail route addressed by source id."
  }),
  defineRoute({
    path: "/share/:token",
    label: "Shared Content",
    group: "media_library",
    surface: "advanced_self_hosted",
    smoke: "manual",
    requiresH1: false,
    h1ExceptionReason: "Pre-governance page pending heading audit.",
    rationale: "Public share link viewer route."
  }),
  defineRoute({
    path: "/knowledge/shared/:shareToken",
    label: "Shared Knowledge",
    group: "knowledge",
    surface: "advanced_self_hosted",
    smoke: "manual",
    requiresH1: false,
    h1ExceptionReason: "Pre-governance page pending heading audit.",
    rationale: "Shared knowledge viewer route addressed by share token."
  }),
  defineRoute({
    path: "/knowledge/thread/:threadId",
    label: "Knowledge Thread",
    group: "knowledge",
    surface: "advanced_self_hosted",
    smoke: "manual",
    requiresH1: false,
    h1ExceptionReason: "Pre-governance page pending heading audit.",
    rationale: "Knowledge thread route addressed by thread id."
  }),
  defineRoute({
    path: "/prototype-workspaces",
    label: "Prototype Workspaces",
    group: "workspace",
    surface: "labs_beta",
    smoke: "manual",
    requiresH1: false,
    h1ExceptionReason: "Pre-governance page pending heading audit.",
    rationale: "Prototype workspace experiments route."
  }),
  defineRoute({
    path: "/scheduled-tasks/results",
    label: "Scheduled Task Results",
    group: "operations",
    surface: "advanced_self_hosted",
    smoke: "manual",
    requiresH1: false,
    h1ExceptionReason: "Pre-governance page pending heading audit.",
    rationale: "Scheduled task result inspection route."
  })
] as const satisfies readonly RouteMetadata[]

export const normalizeRoutePath = (path: string): string => {
  const [pathname] = path.split("?")
  if (!pathname || pathname === "/") {
    return "/"
  }
  return pathname.endsWith("/") ? pathname.slice(0, -1) : pathname
}

const metadataByPath = new Map<string, RouteMetadata>()

for (const metadata of ROUTE_METADATA) {
  metadataByPath.set(normalizeRoutePath(metadata.path), metadata)
}

const COMMAND_PALETTE_LABEL_OVERRIDES = new Map<string, string>([
  ["/chat", "Go to Chat"],
  ["/knowledge", "Go to Knowledge"],
  ["/media", "Go to Media"],
  ["/notes", "Go to Notes"],
  ["/prompts", "Go to Prompts"],
  ["/flashcards", "Go to Flashcards"],
  ["/documentation", "Go to Documentation"],
  ["/settings", "Go to Settings"],
  ["/settings/health", "Go to Health & Diagnostics"],
  ["/mcp-hub", "Go to MCP Hub"]
])

export function getRouteMetadata(path: string): RouteMetadata | undefined {
  return metadataByPath.get(normalizeRoutePath(path))
}

export function getRouteHeadingPolicy(
  metadata: RouteMetadata
): RouteHeadingPolicy {
  if (typeof metadata.requiresH1 === "boolean") {
    return {
      requiresH1: metadata.requiresH1,
      exceptionReason: metadata.h1ExceptionReason
    }
  }

  if (metadata.smoke !== "include" || headingExceptionSurfaces.has(metadata.surface)) {
    return {
      requiresH1: false,
      exceptionReason: metadata.h1ExceptionReason ?? metadata.rationale
    }
  }

  return { requiresH1: true }
}

export function getCanonicalRoutePath(path: string): string | undefined {
  return getRouteMetadata(path)?.canonicalPath
}

export function isRouteAvailableForSurface(
  path: string,
  availability: RouteAvailability
): boolean {
  return getRouteMetadata(path)?.availability.includes(availability) ?? false
}

export function getRoutesForSmokeInventory(): string[] {
  return ROUTE_METADATA.filter((metadata) => metadata.smoke === "include").map(
    (metadata) => metadata.path
  )
}

export function getCommandPaletteRoutes(): RouteMetadata[] {
  return ROUTE_METADATA.filter((metadata) => metadata.commandPalette === "show")
}

export function getCommandPaletteTarget(path: string): string {
  return getRouteMetadata(path)?.canonicalPath ?? path
}

export function getRouteCommandPaletteLabel(
  route: RouteMetadata | string
): string {
  const metadata = typeof route === "string" ? getRouteMetadata(route) : route
  if (!metadata) return typeof route === "string" ? route : route.label
  return (
    COMMAND_PALETTE_LABEL_OVERRIDES.get(normalizeRoutePath(metadata.path)) ??
    metadata.label
  )
}

export function isAuditedRootRoute(path: string): boolean {
  return (AUDITED_ROOT_ROUTE_PATHS as readonly string[]).includes(
    normalizeRoutePath(path)
  )
}
