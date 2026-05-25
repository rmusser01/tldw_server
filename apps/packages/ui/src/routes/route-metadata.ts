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
const sidepanel = ["extension_sidepanel"] as const

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
  "/research-workspace",
  "/document-workspace",
  "/repo2txt",
  "/model-playground",
  "/writing-playground",
  "/presentation-studio",
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
    path: "/research-workspace",
    label: "Research Workspace",
    group: "workspace",
    surface: "labs_beta",
    availability: webAndExtension,
    smoke: "manual",
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
    path: "/audiobook-studio",
    label: "Audiobook Studio",
    group: "audio",
    surface: "advanced_self_hosted",
    availability: webAndExtension,
    requiresBackend: true,
    rationale: "Long-form audio production route tied to TTS readiness."
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
    label: "Chatbooks",
    group: "media_library",
    surface: "advanced_self_hosted",
    availability: webAndExtension,
    smoke: "manual",
    requiresBackend: true,
    rationale: "Chatbook management entry route with known full-suite smoke instability."
  }),
  defineRoute({
    path: "/chatbooks-playground",
    label: "Chatbooks Playground",
    group: "media_library",
    surface: "advanced_self_hosted",
    availability: webAndExtension,
    requiresBackend: true,
    rationale: "Chatbook authoring and testing workspace."
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
    path: "/admin",
    label: "Admin",
    group: "settings",
    surface: "admin_operator",
    smoke: "manual",
    requiresBackend: true,
    rationale: "Admin landing route for operator-only workflows."
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
    label: "STT",
    group: "audio",
    surface: "default_self_hosted",
    availability: webAndExtension,
    requiresBackend: true,
    rationale: "Speech-to-text workflow route."
  }),
  defineRoute({
    path: "/tts",
    label: "TTS",
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
    label: "Moderation Playground",
    group: "safety",
    surface: "advanced_self_hosted",
    availability: webAndExtension,
    requiresBackend: true,
    rationale: "Moderation and safety testing route."
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
    label: "Claims Review",
    group: "safety",
    surface: "advanced_self_hosted",
    smoke: "manual",
    requiresBackend: true,
    rationale: "Claims review and verification route."
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
    path: "/__debug__/sidepanel-error-boundary",
    label: "Debug Sidepanel Error Boundary",
    group: "extension",
    surface: "internal_qa_debug",
    smoke: "exclude",
    rationale: "Web debug route for sidepanel error-boundary rendering."
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

export function getRouteMetadata(path: string): RouteMetadata | undefined {
  return metadataByPath.get(normalizeRoutePath(path))
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
  return typeof route === "string"
    ? getRouteMetadata(route)?.label ?? route
    : route.label
}

export function isAuditedRootRoute(path: string): boolean {
  return (AUDITED_ROOT_ROUTE_PATHS as readonly string[]).includes(
    normalizeRoutePath(path)
  )
}
