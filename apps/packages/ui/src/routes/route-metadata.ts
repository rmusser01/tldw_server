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
  | "marketing"
  | "account"

export type RouteAvailability =
  | "web"
  | "extension_options"
  | "extension_sidepanel"

export type HostedOptionVisibility = "visible"

export type RouteMetadata = {
  path: string
  canonicalPath: string
  label: string
  commandLabel?: string
  group: RouteGroup
  surface: RouteSurface
  availability: RouteAvailability[]
  aliases?: string[]
  redirectsTo?: string
  smoke: "include" | "exclude" | "manual"
  commandPalette: "show" | "hide" | "alias_only"
  hostedOptionVisibility?: HostedOptionVisibility
  nav: "primary" | "secondary" | "hidden"
  requiresAuth?: boolean
  requiresBackend?: boolean
  requiresH1?: boolean
  h1ExceptionReason?: string
  rationale: string
}

export type RouteHeadingPolicy = {
  requiresH1: boolean
  exceptionReason?: string
}

const webAndExtensionOptions: RouteAvailability[] = ["web", "extension_options"]
const webOnly: RouteAvailability[] = ["web"]

const AUDITED_ROUTE_METADATA: RouteMetadata[] = [
  {
    path: "/",
    canonicalPath: "/",
    label: "Home",
    group: "start",
    surface: "default_self_hosted",
    availability: [...webAndExtensionOptions, "extension_sidepanel"],
    smoke: "include",
    commandPalette: "show",
    hostedOptionVisibility: "visible",
    nav: "primary",
    requiresBackend: true,
    rationale: "Default entry route for self-hosted WebUI and extension shells."
  },
  {
    path: "/setup",
    canonicalPath: "/setup",
    label: "Setup",
    group: "start",
    surface: "default_self_hosted",
    availability: webAndExtensionOptions,
    smoke: "manual",
    commandPalette: "hide",
    nav: "hidden",
    rationale: "Initial configuration route is stateful and only relevant before setup completes."
  },
  {
    path: "/login",
    canonicalPath: "/login",
    label: "Login",
    group: "account",
    surface: "default_self_hosted",
    availability: webOnly,
    smoke: "manual",
    commandPalette: "hide",
    nav: "hidden",
    rationale: "Authentication entry point appears only when auth mode requires it."
  },
  {
    path: "/signup",
    canonicalPath: "/signup",
    label: "Sign Up",
    group: "account",
    surface: "hosted_only",
    availability: webOnly,
    smoke: "manual",
    commandPalette: "hide",
    nav: "hidden",
    rationale: "Account creation is deployment-mode dependent and should not be default self-hosted navigation."
  },
  {
    path: "/account",
    canonicalPath: "/account",
    label: "Account",
    group: "account",
    surface: "hosted_only",
    availability: webOnly,
    smoke: "manual",
    commandPalette: "hide",
    nav: "hidden",
    requiresAuth: true,
    rationale: "Account management is user-authenticated and primarily hosted-facing."
  },
  {
    path: "/profile",
    canonicalPath: "/profile",
    label: "Profile",
    group: "account",
    surface: "advanced_self_hosted",
    availability: webOnly,
    smoke: "manual",
    commandPalette: "hide",
    nav: "hidden",
    requiresAuth: true,
    rationale: "Profile state depends on authentication and is not a primary self-hosted workflow."
  },
  {
    path: "/privileges",
    canonicalPath: "/privileges",
    label: "Privileges",
    group: "account",
    surface: "advanced_self_hosted",
    availability: webOnly,
    smoke: "manual",
    commandPalette: "hide",
    nav: "hidden",
    requiresAuth: true,
    rationale: "Privilege review is account-scoped and should stay outside default navigation."
  },
  {
    path: "/config",
    canonicalPath: "/config",
    label: "Configuration",
    group: "settings",
    surface: "admin_operator",
    availability: webOnly,
    smoke: "manual",
    commandPalette: "hide",
    nav: "hidden",
    requiresBackend: true,
    rationale: "Configuration state is operator-facing and overlaps with settings/admin surfaces."
  },
  {
    path: "/billing",
    canonicalPath: "/billing",
    label: "Billing",
    group: "account",
    surface: "hosted_only",
    availability: webOnly,
    smoke: "manual",
    commandPalette: "hide",
    nav: "hidden",
    requiresAuth: true,
    rationale: "Billing is hosted/account-specific and not part of default self-hosted IA."
  },
  {
    path: "/404",
    canonicalPath: "/404",
    label: "Not Found",
    group: "start",
    surface: "default_self_hosted",
    availability: webOnly,
    smoke: "include",
    commandPalette: "hide",
    nav: "hidden",
    rationale: "Error-state route is useful for coverage but should not be user navigation."
  },
  {
    path: "/chat",
    canonicalPath: "/chat",
    label: "Chat",
    group: "chat",
    surface: "default_self_hosted",
    availability: [...webAndExtensionOptions, "extension_sidepanel"],
    smoke: "include",
    commandPalette: "show",
    hostedOptionVisibility: "visible",
    nav: "primary",
    requiresBackend: true,
    rationale: "Primary assistant conversation route across WebUI and extension sidepanel."
  },
  {
    path: "/quick-chat-popout",
    canonicalPath: "/quick-chat-popout",
    label: "Quick Chat Popout",
    group: "chat",
    surface: "advanced_self_hosted",
    availability: webAndExtensionOptions,
    smoke: "manual",
    commandPalette: "hide",
    nav: "hidden",
    rationale: "Detached chat surface is a secondary convenience entry rather than primary IA."
  },
  {
    path: "/persona",
    canonicalPath: "/persona",
    label: "Persona",
    group: "chat",
    surface: "advanced_self_hosted",
    availability: [...webAndExtensionOptions, "extension_sidepanel"],
    smoke: "include",
    commandPalette: "show",
    nav: "secondary",
    requiresBackend: true,
    rationale: "Persona chat is a real route but secondary to the core chat path."
  },
  {
    path: "/characters",
    canonicalPath: "/characters",
    label: "Characters",
    group: "knowledge",
    surface: "advanced_self_hosted",
    availability: webAndExtensionOptions,
    smoke: "include",
    commandPalette: "show",
    nav: "secondary",
    requiresBackend: true,
    rationale: "Character library is a knowledge asset manager used by chat workflows."
  },
  {
    path: "/companion",
    canonicalPath: "/companion",
    label: "Companion",
    group: "chat",
    surface: "labs_beta",
    availability: [...webAndExtensionOptions, "extension_sidepanel"],
    smoke: "manual",
    commandPalette: "show",
    nav: "secondary",
    rationale: "Companion is discoverable but still a specialized persona surface."
  },
  {
    path: "/agents",
    canonicalPath: "/agents",
    label: "Agents",
    group: "chat",
    surface: "advanced_self_hosted",
    availability: webAndExtensionOptions,
    smoke: "manual",
    commandPalette: "show",
    nav: "secondary",
    requiresBackend: true,
    rationale: "Agent management is an advanced assistant workflow."
  },
  {
    path: "/agent-tasks",
    canonicalPath: "/agent-tasks",
    label: "Agent Tasks",
    group: "chat",
    surface: "advanced_self_hosted",
    availability: webAndExtensionOptions,
    smoke: "manual",
    commandPalette: "show",
    nav: "secondary",
    requiresBackend: true,
    rationale: "Agent task tracking is useful after agent workflows are configured."
  },
  {
    path: "/chat-workflows",
    canonicalPath: "/chat-workflows",
    label: "Chat Workflows",
    group: "chat",
    surface: "advanced_self_hosted",
    availability: webAndExtensionOptions,
    smoke: "include",
    commandPalette: "show",
    nav: "secondary",
    requiresBackend: true,
    rationale: "Chat workflow authoring is adjacent to chat but more advanced than daily chat."
  },
  {
    path: "/chat-workspace",
    canonicalPath: "/chat-workspace",
    label: "Chat Workspace",
    group: "chat",
    surface: "advanced_self_hosted",
    availability: webAndExtensionOptions,
    smoke: "include",
    commandPalette: "show",
    hostedOptionVisibility: "visible",
    nav: "secondary",
    requiresBackend: true,
    rationale: "Workspace-style chat is available for users who need denser working context."
  },
  {
    path: "/knowledge",
    canonicalPath: "/knowledge",
    label: "Knowledge",
    group: "knowledge",
    surface: "default_self_hosted",
    availability: webAndExtensionOptions,
    smoke: "include",
    commandPalette: "show",
    hostedOptionVisibility: "visible",
    nav: "primary",
    requiresBackend: true,
    rationale: "Primary knowledge and RAG route for self-hosted users."
  },
  {
    path: "/search",
    canonicalPath: "/search",
    label: "Search",
    group: "knowledge",
    surface: "default_self_hosted",
    availability: webOnly,
    smoke: "include",
    commandPalette: "show",
    nav: "primary",
    requiresBackend: true,
    rationale: "Top-level search is a primary retrieval workflow for existing content."
  },
  {
    path: "/research",
    canonicalPath: "/research",
    label: "Research",
    group: "knowledge",
    surface: "advanced_self_hosted",
    availability: webOnly,
    smoke: "manual",
    commandPalette: "show",
    nav: "secondary",
    requiresBackend: true,
    rationale: "Research runs are backend-stateful and should be discoverable after setup."
  },
  {
    path: "/workspace-playground",
    canonicalPath: "/workspace-playground",
    label: "Workspace Playground",
    group: "workspace",
    surface: "advanced_self_hosted",
    availability: webAndExtensionOptions,
    smoke: "include",
    commandPalette: "show",
    nav: "secondary",
    requiresBackend: true,
    rationale: "Workspace playground is a power-user composition surface."
  },
  {
    path: "/document-workspace",
    canonicalPath: "/document-workspace",
    label: "Document Workspace",
    group: "workspace",
    surface: "advanced_self_hosted",
    availability: webAndExtensionOptions,
    smoke: "include",
    commandPalette: "show",
    nav: "secondary",
    requiresBackend: true,
    rationale: "Document workspace supports document-heavy workflows outside the main library."
  },
  {
    path: "/repo2txt",
    canonicalPath: "/repo2txt",
    label: "Repo2Txt",
    group: "workspace",
    surface: "advanced_self_hosted",
    availability: webAndExtensionOptions,
    smoke: "include",
    commandPalette: "show",
    nav: "secondary",
    rationale: "Repository conversion is a specialized utility for technical users."
  },
  {
    path: "/model-playground",
    canonicalPath: "/model-playground",
    label: "Model Playground",
    group: "workspace",
    surface: "advanced_self_hosted",
    availability: webAndExtensionOptions,
    smoke: "include",
    commandPalette: "show",
    nav: "secondary",
    requiresBackend: true,
    rationale: "Model experimentation is an advanced workflow that depends on configured providers."
  },
  {
    path: "/writing-playground",
    canonicalPath: "/writing-playground",
    label: "Writing Playground",
    group: "workspace",
    surface: "advanced_self_hosted",
    availability: webAndExtensionOptions,
    smoke: "manual",
    commandPalette: "show",
    nav: "secondary",
    requiresBackend: true,
    rationale: "Writing tools are useful but have broader setup and state dependencies."
  },
  {
    path: "/presentation-studio",
    canonicalPath: "/presentation-studio",
    label: "Presentation Studio",
    group: "workspace",
    surface: "labs_beta",
    availability: webAndExtensionOptions,
    smoke: "include",
    commandPalette: "show",
    nav: "secondary",
    rationale: "Presentation creation is a labs workflow with dedicated project state."
  },
  {
    path: "/audiobook-studio",
    canonicalPath: "/audiobook-studio",
    label: "Audiobook Studio",
    group: "audio",
    surface: "labs_beta",
    availability: webAndExtensionOptions,
    smoke: "include",
    commandPalette: "show",
    nav: "secondary",
    requiresBackend: true,
    rationale: "Audiobook generation is an audio-specific studio workflow."
  },
  {
    path: "/media",
    canonicalPath: "/media",
    label: "Media Library",
    commandLabel: "Go to Media",
    group: "media_library",
    surface: "default_self_hosted",
    availability: webAndExtensionOptions,
    smoke: "include",
    commandPalette: "show",
    hostedOptionVisibility: "visible",
    nav: "primary",
    requiresBackend: true,
    rationale: "Primary route for ingested media review and management."
  },
  {
    path: "/media-multi",
    canonicalPath: "/media-multi",
    label: "Media Multi",
    group: "media_library",
    surface: "default_self_hosted",
    availability: webAndExtensionOptions,
    smoke: "include",
    commandPalette: "show",
    nav: "secondary",
    requiresBackend: true,
    rationale: "Dense media workflow for batch review and multi-item operations."
  },
  {
    path: "/review",
    canonicalPath: "/media-multi",
    label: "Review",
    group: "media_library",
    surface: "redirect",
    availability: webAndExtensionOptions,
    redirectsTo: "/media-multi",
    smoke: "exclude",
    commandPalette: "alias_only",
    nav: "hidden",
    requiresBackend: true,
    rationale: "Legacy review route redirects to the current media multi workflow."
  },
  {
    path: "/media-trash",
    canonicalPath: "/media-trash",
    label: "Media Trash",
    group: "media_library",
    surface: "advanced_self_hosted",
    availability: webAndExtensionOptions,
    smoke: "include",
    commandPalette: "show",
    nav: "secondary",
    requiresBackend: true,
    rationale: "Recovery route for deleted media and library cleanup."
  },
  {
    path: "/items",
    canonicalPath: "/items",
    label: "Items",
    group: "media_library",
    surface: "advanced_self_hosted",
    availability: webOnly,
    smoke: "include",
    commandPalette: "show",
    nav: "secondary",
    requiresBackend: true,
    rationale: "Item-level content management is a secondary library workflow."
  },
  {
    path: "/collections",
    canonicalPath: "/collections",
    label: "Collections",
    group: "media_library",
    surface: "default_self_hosted",
    availability: webAndExtensionOptions,
    smoke: "include",
    commandPalette: "show",
    hostedOptionVisibility: "visible",
    nav: "secondary",
    requiresBackend: true,
    rationale: "Collection organization is a common media-library workflow."
  },
  {
    path: "/reading",
    canonicalPath: "/reading",
    label: "Reading",
    group: "media_library",
    surface: "default_self_hosted",
    availability: webOnly,
    smoke: "include",
    commandPalette: "show",
    nav: "primary",
    requiresBackend: true,
    rationale: "Reading route supports document and article review for knowledge workflows."
  },
  {
    path: "/notes",
    canonicalPath: "/notes",
    label: "Notes",
    group: "knowledge",
    surface: "default_self_hosted",
    availability: webAndExtensionOptions,
    smoke: "include",
    commandPalette: "show",
    nav: "primary",
    requiresBackend: true,
    requiresH1: false,
    h1ExceptionReason:
      "Notes can render user-authored document headings, including h1, inside the editor and preview surface.",
    rationale: "Notes are a primary knowledge capture surface."
  },
  {
    path: "/shared",
    canonicalPath: "/shared",
    label: "Shared",
    group: "knowledge",
    surface: "advanced_self_hosted",
    availability: webAndExtensionOptions,
    smoke: "manual",
    commandPalette: "hide",
    nav: "hidden",
    requiresBackend: true,
    rationale: "Shared content depends on share state and should not lead default navigation."
  },
  {
    path: "/chatbooks",
    canonicalPath: "/chatbooks",
    label: "Chatbooks",
    group: "workspace",
    surface: "advanced_self_hosted",
    availability: webAndExtensionOptions,
    smoke: "manual",
    commandPalette: "show",
    nav: "secondary",
    requiresBackend: true,
    rationale: "Chatbook import/export is a power-user workflow with larger state dependencies."
  },
  {
    path: "/chatbooks-playground",
    canonicalPath: "/chatbooks-playground",
    label: "Chatbooks Playground",
    group: "workspace",
    surface: "labs_beta",
    availability: webAndExtensionOptions,
    smoke: "include",
    commandPalette: "show",
    nav: "hidden",
    requiresBackend: true,
    rationale: "Experimental chatbook surface should be testable without primary IA prominence."
  },
  {
    path: "/sources",
    canonicalPath: "/sources",
    label: "Sources",
    group: "knowledge",
    surface: "default_self_hosted",
    availability: webAndExtensionOptions,
    smoke: "include",
    commandPalette: "show",
    nav: "primary",
    requiresBackend: true,
    rationale: "Source management is a primary path into ingestion and knowledge setup."
  },
  {
    path: "/connectors",
    canonicalPath: "/connectors",
    label: "Connectors",
    group: "operations",
    surface: "labs_beta",
    availability: webOnly,
    smoke: "manual",
    commandPalette: "show",
    nav: "secondary",
    requiresBackend: true,
    rationale: "Connector routes are planned/operator workflows with placeholder coverage."
  },
  {
    path: "/integrations",
    canonicalPath: "/integrations",
    label: "Integrations",
    group: "operations",
    surface: "advanced_self_hosted",
    availability: webAndExtensionOptions,
    smoke: "include",
    commandPalette: "show",
    nav: "secondary",
    requiresBackend: true,
    rationale: "Integration management is an operator-adjacent configuration workflow."
  },
  {
    path: "/scheduled-tasks",
    canonicalPath: "/scheduled-tasks",
    label: "Scheduled Tasks",
    group: "operations",
    surface: "advanced_self_hosted",
    availability: webAndExtensionOptions,
    smoke: "include",
    commandPalette: "show",
    nav: "secondary",
    requiresBackend: true,
    rationale: "Scheduled work is a returning-user operations workflow."
  },
  {
    path: "/watchlists",
    canonicalPath: "/watchlists",
    label: "Watchlists",
    group: "operations",
    surface: "advanced_self_hosted",
    availability: webAndExtensionOptions,
    smoke: "include",
    commandPalette: "show",
    nav: "secondary",
    requiresBackend: true,
    rationale: "Watchlists are a recurring automation workflow for power users."
  },
  {
    path: "/workflow-editor",
    canonicalPath: "/workflow-editor",
    label: "Workflow Editor",
    group: "operations",
    surface: "advanced_self_hosted",
    availability: webAndExtensionOptions,
    smoke: "include",
    commandPalette: "show",
    nav: "secondary",
    requiresBackend: true,
    rationale: "Workflow editing is an advanced orchestration surface."
  },
  {
    path: "/settings",
    canonicalPath: "/settings",
    label: "Settings",
    group: "settings",
    surface: "default_self_hosted",
    availability: [...webAndExtensionOptions, "extension_sidepanel"],
    smoke: "include",
    commandPalette: "show",
    nav: "primary",
    rationale: "Primary route for user and server configuration."
  },
  {
    path: "/admin",
    canonicalPath: "/admin",
    label: "Admin",
    group: "operations",
    surface: "admin_operator",
    availability: webOnly,
    smoke: "include",
    commandPalette: "show",
    nav: "secondary",
    requiresAuth: true,
    requiresBackend: true,
    rationale: "Admin hub is operator-only and should remain separate from regular user navigation."
  },
  {
    path: "/mcp-hub",
    canonicalPath: "/mcp-hub",
    label: "MCP Hub",
    group: "operations",
    surface: "advanced_self_hosted",
    availability: webAndExtensionOptions,
    smoke: "include",
    commandPalette: "show",
    nav: "secondary",
    requiresBackend: true,
    rationale: "MCP hub is a configured integration surface for advanced automation users."
  },
  {
    path: "/acp-playground",
    canonicalPath: "/acp-playground",
    label: "ACP Playground",
    group: "operations",
    surface: "labs_beta",
    availability: webAndExtensionOptions,
    smoke: "include",
    commandPalette: "show",
    nav: "hidden",
    requiresBackend: true,
    rationale: "ACP testing is useful for verification but should not be default navigation."
  },
  {
    path: "/prompts",
    canonicalPath: "/prompts",
    label: "Prompts",
    group: "knowledge",
    surface: "default_self_hosted",
    availability: webAndExtensionOptions,
    smoke: "include",
    commandPalette: "show",
    nav: "primary",
    requiresBackend: true,
    rationale: "Prompt library and prompt-studio tools live under the prompts workspace."
  },
  {
    path: "/prompt-studio",
    canonicalPath: "/prompts",
    label: "Prompt Studio",
    group: "knowledge",
    surface: "redirect",
    availability: webAndExtensionOptions,
    redirectsTo: "/prompts?tab=studio",
    smoke: "exclude",
    commandPalette: "alias_only",
    nav: "hidden",
    requiresBackend: true,
    rationale: "Legacy prompt-studio route redirects to the prompts workspace studio tab."
  },
  {
    path: "/dictionaries",
    canonicalPath: "/dictionaries",
    label: "Dictionaries",
    group: "knowledge",
    surface: "advanced_self_hosted",
    availability: webAndExtensionOptions,
    smoke: "include",
    commandPalette: "show",
    nav: "secondary",
    requiresBackend: true,
    rationale: "Dictionaries support character and prompt workflows but are not a primary start point."
  },
  {
    path: "/world-books",
    canonicalPath: "/world-books",
    label: "World Books",
    group: "knowledge",
    surface: "advanced_self_hosted",
    availability: webAndExtensionOptions,
    smoke: "include",
    commandPalette: "show",
    nav: "secondary",
    requiresBackend: true,
    rationale: "World books are reusable context assets for advanced knowledge and persona workflows."
  },
  {
    path: "/speech",
    canonicalPath: "/speech",
    label: "Speech",
    group: "audio",
    surface: "default_self_hosted",
    availability: webAndExtensionOptions,
    smoke: "include",
    commandPalette: "show",
    nav: "primary",
    requiresBackend: true,
    rationale: "Speech is the canonical audio workspace for STT and TTS."
  },
  {
    path: "/stt",
    canonicalPath: "/stt",
    label: "Speech to Text",
    group: "audio",
    surface: "default_self_hosted",
    availability: webAndExtensionOptions,
    smoke: "include",
    commandPalette: "show",
    hostedOptionVisibility: "visible",
    nav: "secondary",
    requiresBackend: true,
    rationale: "STT remains a direct task route inside the broader speech workflow."
  },
  {
    path: "/tts",
    canonicalPath: "/tts",
    label: "Text to Speech",
    group: "audio",
    surface: "default_self_hosted",
    availability: webAndExtensionOptions,
    smoke: "include",
    commandPalette: "show",
    hostedOptionVisibility: "visible",
    nav: "secondary",
    requiresBackend: true,
    rationale: "TTS remains a direct task route inside the broader speech workflow."
  },
  {
    path: "/audio",
    canonicalPath: "/speech",
    label: "Audio",
    group: "audio",
    surface: "redirect",
    availability: webOnly,
    redirectsTo: "/speech",
    smoke: "exclude",
    commandPalette: "alias_only",
    nav: "hidden",
    requiresBackend: true,
    rationale: "Legacy audio route redirects to the canonical speech workspace."
  },
  {
    path: "/evaluations",
    canonicalPath: "/evaluations",
    label: "Evaluations",
    group: "workspace",
    surface: "advanced_self_hosted",
    availability: webAndExtensionOptions,
    smoke: "include",
    commandPalette: "show",
    nav: "secondary",
    requiresBackend: true,
    rationale: "Evaluation workflows are advanced and depend on configured models and datasets."
  },
  {
    path: "/flashcards",
    canonicalPath: "/flashcards",
    label: "Flashcards",
    group: "study",
    surface: "default_self_hosted",
    availability: [...webAndExtensionOptions, "extension_sidepanel"],
    smoke: "include",
    commandPalette: "show",
    nav: "secondary",
    requiresBackend: true,
    rationale: "Flashcards are a direct study workflow reachable from web and sidepanel contexts."
  },
  {
    path: "/quiz",
    canonicalPath: "/quiz",
    label: "Quiz",
    group: "study",
    surface: "default_self_hosted",
    availability: webAndExtensionOptions,
    smoke: "include",
    commandPalette: "show",
    nav: "secondary",
    requiresBackend: true,
    rationale: "Quizzes are a direct study workflow for ingested knowledge."
  },
  {
    path: "/moderation-playground",
    canonicalPath: "/moderation/rules",
    label: "Moderation Playground",
    group: "safety",
    surface: "redirect",
    availability: webOnly,
    redirectsTo: "/moderation/rules",
    smoke: "exclude",
    commandPalette: "alias_only",
    nav: "hidden",
    requiresBackend: true,
    rationale: "Legacy moderation playground redirects to the current content rules route."
  },
  {
    path: "/content-review",
    canonicalPath: "/content-review",
    label: "Content Review",
    group: "safety",
    surface: "advanced_self_hosted",
    availability: webAndExtensionOptions,
    smoke: "include",
    commandPalette: "show",
    nav: "secondary",
    requiresBackend: true,
    rationale: "Content review is a safety workflow for flagged or queued material."
  },
  {
    path: "/claims-review",
    canonicalPath: "/content-review",
    label: "Claims Review",
    group: "safety",
    surface: "redirect",
    availability: webOnly,
    redirectsTo: "/content-review",
    smoke: "exclude",
    commandPalette: "alias_only",
    nav: "hidden",
    requiresBackend: true,
    rationale: "Legacy claims review route redirects to the canonical content review workflow."
  },
  {
    path: "/data-tables",
    canonicalPath: "/data-tables",
    label: "Data Tables",
    group: "workspace",
    surface: "advanced_self_hosted",
    availability: webAndExtensionOptions,
    smoke: "include",
    commandPalette: "show",
    nav: "secondary",
    rationale: "Structured table work is an advanced utility surface."
  },
  {
    path: "/chunking-playground",
    canonicalPath: "/chunking-playground",
    label: "Chunking Playground",
    group: "workspace",
    surface: "labs_beta",
    availability: webAndExtensionOptions,
    smoke: "include",
    commandPalette: "show",
    nav: "hidden",
    rationale: "Chunking experiments are useful for RAG tuning but should remain lab-scoped."
  },
  {
    path: "/kanban",
    canonicalPath: "/kanban",
    label: "Kanban",
    group: "workspace",
    surface: "advanced_self_hosted",
    availability: webAndExtensionOptions,
    smoke: "include",
    commandPalette: "show",
    nav: "secondary",
    rationale: "Kanban is a secondary organization surface."
  },
  {
    path: "/skills",
    canonicalPath: "/skills",
    label: "Skills",
    group: "operations",
    surface: "advanced_self_hosted",
    availability: webAndExtensionOptions,
    smoke: "include",
    commandPalette: "show",
    nav: "secondary",
    requiresBackend: true,
    rationale: "Skill management is an advanced automation and assistant configuration workflow."
  },
  {
    path: "/vn-assets",
    canonicalPath: "/vn-assets",
    label: "VN Assets",
    group: "specialized",
    surface: "labs_beta",
    availability: webOnly,
    smoke: "include",
    commandPalette: "hide",
    nav: "hidden",
    rationale: "Visual novel assets are a specialized labs surface."
  },
  {
    path: "/vn-play",
    canonicalPath: "/vn-play",
    label: "VN Play",
    group: "specialized",
    surface: "labs_beta",
    availability: webOnly,
    smoke: "include",
    commandPalette: "hide",
    nav: "hidden",
    rationale: "Visual novel play is a specialized labs surface."
  },
  {
    path: "/documentation",
    canonicalPath: "/documentation",
    label: "Documentation",
    group: "documentation",
    surface: "default_self_hosted",
    availability: webAndExtensionOptions,
    smoke: "include",
    commandPalette: "show",
    nav: "secondary",
    rationale: "Documentation is a support surface reachable when users need help."
  },
  {
    path: "/notifications",
    canonicalPath: "/notifications",
    label: "Notifications",
    group: "account",
    surface: "advanced_self_hosted",
    availability: webOnly,
    smoke: "manual",
    commandPalette: "hide",
    nav: "hidden",
    requiresBackend: true,
    rationale: "Notifications are user-stateful and should not be primary navigation."
  },
  {
    path: "/composer-variants-preview",
    canonicalPath: "/composer-variants-preview",
    label: "Composer Variants Preview",
    group: "specialized",
    surface: "internal_qa_debug",
    availability: webOnly,
    smoke: "manual",
    commandPalette: "hide",
    nav: "hidden",
    rationale: "Composer variants preview is an internal QA route for visual comparison."
  },
  {
    path: "/onboarding-test",
    canonicalPath: "/onboarding-test",
    label: "Onboarding Test",
    group: "start",
    surface: "internal_qa_debug",
    availability: webAndExtensionOptions,
    smoke: "manual",
    commandPalette: "hide",
    nav: "hidden",
    rationale: "Onboarding test route is for verification and should not appear in normal IA."
  }
]

type RegistryRouteMetadataInput = Pick<
  RouteMetadata,
  "path" | "label" | "group" | "rationale"
> &
  Partial<Omit<RouteMetadata, "path" | "label" | "group" | "rationale">>

const registryRoute = ({
  path,
  canonicalPath = path,
  label,
  commandLabel,
  group,
  surface = "advanced_self_hosted",
  availability = webAndExtensionOptions,
  aliases,
  redirectsTo,
  smoke = "include",
  commandPalette = "show",
  hostedOptionVisibility,
  nav = "secondary",
  requiresAuth,
  requiresBackend,
  requiresH1,
  h1ExceptionReason,
  rationale
}: RegistryRouteMetadataInput): RouteMetadata => ({
  path,
  canonicalPath,
  label,
  commandLabel,
  group,
  surface,
  availability,
  aliases,
  redirectsTo,
  smoke,
  commandPalette,
  hostedOptionVisibility,
  nav,
  requiresAuth,
  requiresBackend,
  requiresH1,
  h1ExceptionReason,
  rationale
})

const settingsRoute = (
  path: string,
  label: string,
  rationale: string,
  overrides: Partial<
    Omit<RouteMetadata, "path" | "label" | "group" | "rationale">
  > = {}
): RouteMetadata =>
  registryRoute({
    path,
    label,
    group: "settings",
    surface: "default_self_hosted",
    nav: "hidden",
    requiresBackend: true,
    rationale,
    ...overrides
  })

const adminRoute = (
  path: string,
  label: string,
  rationale: string,
  overrides: Partial<
    Omit<RouteMetadata, "path" | "label" | "group" | "rationale">
  > = {}
): RouteMetadata =>
  registryRoute({
    path,
    label,
    group: "operations",
    surface: "admin_operator",
    availability: webAndExtensionOptions,
    nav: "secondary",
    requiresAuth: true,
    requiresBackend: true,
    rationale,
    ...overrides
  })

const ROUTE_REGISTRY_METADATA: RouteMetadata[] = [
  settingsRoute(
    "/settings/tldw",
    "TLDW Settings",
    "Product-level settings are nested under the primary settings route."
  ),
  settingsRoute(
    "/settings/model",
    "Model Settings",
    "Provider and model defaults are nested under settings for setup continuity."
  ),
  settingsRoute(
    "/settings/mcp-hub",
    "MCP Hub Settings",
    "MCP settings configure the advanced MCP hub surface."
  ),
  settingsRoute(
    "/settings/prompt",
    "Prompt Settings",
    "Prompt workspace preferences are nested under settings."
  ),
  settingsRoute(
    "/settings/evaluations",
    "Evaluation Settings",
    "Evaluation defaults are nested under settings because they affect advanced workflows."
  ),
  settingsRoute(
    "/settings/chat",
    "Chat Settings",
    "Chat defaults are nested under settings and are reachable from chat contexts."
  ),
  settingsRoute(
    "/settings/ui",
    "UI Settings",
    "Interface preferences are nested under settings."
  ),
  settingsRoute(
    "/settings/splash",
    "Splash Settings",
    "Startup and splash preferences are nested under settings."
  ),
  settingsRoute(
    "/settings/quick-ingest",
    "Quick Ingest Settings",
    "Quick ingest preferences are nested under settings for capture workflow configuration."
  ),
  settingsRoute(
    "/settings/speech",
    "Speech Settings",
    "Speech defaults are nested under settings and support STT/TTS workflows."
  ),
  settingsRoute(
    "/settings/image-generation",
    "Image Generation Settings",
    "Image-generation provider defaults are nested under settings."
  ),
  settingsRoute(
    "/settings/image-gen",
    "Image Generation Settings Alias",
    "Legacy image-generation settings route redirects to the canonical settings route.",
    {
      canonicalPath: "/settings/image-generation",
      surface: "redirect",
      redirectsTo: "/settings/image-generation",
      smoke: "exclude",
      commandPalette: "alias_only"
    }
  ),
  settingsRoute(
    "/settings/share",
    "Share Settings",
    "Sharing preferences are nested under settings because they affect publish and recovery flows."
  ),
  settingsRoute(
    "/settings/processed",
    "Processed Content Settings",
    "Processed-content settings are nested under settings for library maintenance."
  ),
  settingsRoute(
    "/settings/health",
    "Health Settings",
    "Health and connection diagnostics are nested under settings.",
    {
      commandLabel: "Go to Health & Diagnostics"
    }
  ),
  settingsRoute(
    "/settings/prompt-studio",
    "Prompt Studio Settings",
    "Prompt Studio preferences remain under settings after the workspace moved to prompts."
  ),
  settingsRoute(
    "/settings/knowledge",
    "Knowledge Settings",
    "Knowledge and retrieval settings are nested under settings."
  ),
  settingsRoute(
    "/settings/chatbooks",
    "Chatbooks Settings",
    "Chatbook import/export preferences are nested under settings."
  ),
  settingsRoute(
    "/settings/characters",
    "Characters Settings",
    "Character workspace preferences are nested under settings."
  ),
  settingsRoute(
    "/settings/world-books",
    "World Books Settings",
    "World book preferences are nested under settings."
  ),
  settingsRoute(
    "/settings/chat-dictionaries",
    "Dictionary Settings",
    "Dictionary preferences are nested under settings."
  ),
  settingsRoute(
    "/settings/rag",
    "RAG Settings",
    "RAG defaults are nested under settings because they affect chat and knowledge answers."
  ),
  settingsRoute(
    "/settings/about",
    "About",
    "About and version details live under settings rather than primary navigation.",
    {
      requiresBackend: false
    }
  ),
  settingsRoute(
    "/settings/family-guardrails",
    "Family Guardrails",
    "Family guardrail configuration is capability-gated under settings.",
    {
      surface: "advanced_self_hosted",
      smoke: "manual"
    }
  ),
  settingsRoute(
    "/settings/guardian",
    "Guardian Settings",
    "Guardian settings are capability-gated and should remain nested under settings.",
    {
      surface: "advanced_self_hosted",
      smoke: "manual"
    }
  ),
  adminRoute(
    "/admin/integrations",
    "Admin Integrations",
    "Administrative integration controls are operator-only."
  ),
  adminRoute(
    "/admin/sources",
    "Admin Sources",
    "Administrative source controls are operator-only."
  ),
  adminRoute(
    "/admin/server",
    "Server Admin",
    "Server runtime controls are operator-only."
  ),
  adminRoute(
    "/admin/llamacpp",
    "Llama.cpp Admin",
    "Local llama.cpp server controls are operator-only."
  ),
  adminRoute(
    "/admin/mlx",
    "MLX Admin",
    "Local MLX server controls are operator-only."
  ),
  adminRoute(
    "/admin/runtime-config",
    "Runtime Config Admin",
    "Runtime configuration controls are operator-only.",
    {
      smoke: "manual"
    }
  ),
  adminRoute(
    "/admin/monitoring",
    "Monitoring Admin",
    "Monitoring is an operator route for service status and diagnostics."
  ),
  adminRoute(
    "/admin/api-keys",
    "API Keys Admin",
    "API key administration is a Next-only operator page outside normal user navigation.",
    {
      availability: webOnly
    }
  ),
  adminRoute(
    "/admin/billing",
    "Billing Admin",
    "Billing administration is an operator page and remains separate from hosted user billing.",
    {
      availability: webOnly
    }
  ),
  adminRoute(
    "/admin/data-ops",
    "Data Operations Admin",
    "Data operations administration is a Next-only operator workflow.",
    {
      availability: webOnly
    }
  ),
  adminRoute(
    "/admin/maintenance",
    "Maintenance Admin",
    "Maintenance controls are operator-only and should stay out of default user navigation.",
    {
      availability: webOnly
    }
  ),
  adminRoute(
    "/admin/orgs",
    "Organizations Admin",
    "Organization administration is operator-owned and deployment-mode dependent.",
    {
      availability: webOnly
    }
  ),
  adminRoute(
    "/admin/rate-limiting",
    "Rate Limiting Admin",
    "Rate-limit administration is an operator-only controls page.",
    {
      availability: webOnly
    }
  ),
  adminRoute(
    "/admin/rbac",
    "RBAC Admin",
    "RBAC administration is an operator-only access-control page.",
    {
      availability: webOnly
    }
  ),
  adminRoute(
    "/admin/usage",
    "Usage Admin",
    "Usage administration is an operator-only reporting page.",
    {
      availability: webOnly
    }
  ),
  adminRoute(
    "/admin/watchlists-items",
    "Watchlists Items Admin",
    "Watchlist item administration is an operator drill-down page.",
    {
      availability: webOnly
    }
  ),
  adminRoute(
    "/admin/watchlists-runs",
    "Watchlists Runs Admin",
    "Watchlist run administration is an operator placeholder and drill-down page.",
    {
      availability: webOnly
    }
  ),
  registryRoute({
    path: "/auth/magic-link",
    label: "Magic Link Sign-In",
    group: "account",
    surface: "hosted_only",
    availability: webOnly,
    commandPalette: "hide",
    nav: "hidden",
    rationale: "Hosted magic-link auth lives outside the self-hosted WebUI distribution."
  }),
  registryRoute({
    path: "/auth/reset-password",
    label: "Reset Password",
    group: "account",
    surface: "hosted_only",
    availability: webOnly,
    commandPalette: "hide",
    nav: "hidden",
    rationale: "Hosted password-reset auth lives outside the self-hosted WebUI distribution."
  }),
  registryRoute({
    path: "/auth/verify-email",
    label: "Verify Email",
    group: "account",
    surface: "hosted_only",
    availability: webOnly,
    commandPalette: "hide",
    nav: "hidden",
    rationale: "Hosted email verification lives outside the self-hosted WebUI distribution."
  }),
  registryRoute({
    path: "/billing/success",
    label: "Billing Success",
    group: "account",
    surface: "hosted_only",
    availability: webOnly,
    commandPalette: "hide",
    nav: "hidden",
    rationale: "Hosted billing completion is represented by a self-hosted placeholder."
  }),
  registryRoute({
    path: "/billing/cancel",
    label: "Billing Cancel",
    group: "account",
    surface: "hosted_only",
    availability: webOnly,
    commandPalette: "hide",
    nav: "hidden",
    rationale: "Hosted billing cancellation is represented by a self-hosted placeholder."
  }),
  registryRoute({
    path: "/chat/agent",
    canonicalPath: "/agents",
    label: "Agent Chat",
    group: "chat",
    surface: "advanced_self_hosted",
    availability: webOnly,
    smoke: "include",
    commandPalette: "hide",
    nav: "hidden",
    requiresBackend: true,
    rationale: "Legacy web wrapper for the sidepanel agent chat route remains in smoke inventory."
  }),
  registryRoute({
    path: "/media/:id/view",
    canonicalPath: "/media",
    label: "Media Item View Redirect",
    group: "media_library",
    surface: "redirect",
    availability: webOnly,
    aliases: ["/media/123/view"],
    redirectsTo: "/media",
    smoke: "exclude",
    commandPalette: "alias_only",
    nav: "hidden",
    requiresBackend: true,
    rationale: "Dynamic media view redirects into the canonical media library surface."
  }),
  registryRoute({
    path: "/connectors/browse",
    label: "Connector Browse",
    group: "operations",
    surface: "labs_beta",
    availability: webOnly,
    commandPalette: "hide",
    nav: "hidden",
    rationale: "Connector browse is a placeholder child route under the connectors hub."
  }),
  registryRoute({
    path: "/connectors/jobs",
    label: "Connector Jobs",
    group: "operations",
    surface: "labs_beta",
    availability: webOnly,
    commandPalette: "hide",
    nav: "hidden",
    rationale: "Connector jobs is a placeholder child route under the connectors hub."
  }),
  registryRoute({
    path: "/connectors/sources",
    label: "Connector Sources",
    group: "operations",
    surface: "labs_beta",
    availability: webOnly,
    commandPalette: "hide",
    nav: "hidden",
    rationale: "Connector sources is a placeholder child route under the connectors hub."
  }),
  registryRoute({
    path: "/for/journalists",
    label: "For Journalists",
    group: "marketing",
    surface: "default_self_hosted",
    availability: webOnly,
    commandPalette: "hide",
    nav: "hidden",
    rationale: "Audience landing page is public web content rather than application navigation."
  }),
  registryRoute({
    path: "/for/osint",
    label: "For OSINT",
    group: "marketing",
    surface: "default_self_hosted",
    availability: webOnly,
    commandPalette: "hide",
    nav: "hidden",
    rationale: "Audience landing page is public web content rather than application navigation."
  }),
  registryRoute({
    path: "/for/researchers",
    label: "For Researchers",
    group: "marketing",
    surface: "default_self_hosted",
    availability: webOnly,
    commandPalette: "hide",
    nav: "hidden",
    rationale: "Audience landing page is public web content rather than application navigation."
  }),
  registryRoute({
    path: "/sources/new",
    label: "New Source",
    group: "knowledge",
    surface: "default_self_hosted",
    nav: "hidden",
    requiresBackend: true,
    rationale: "Source creation is a task route owned by the sources workflow."
  }),
  registryRoute({
    path: "/sources/:sourceId",
    canonicalPath: "/sources",
    label: "Source Detail",
    group: "knowledge",
    surface: "default_self_hosted",
    smoke: "manual",
    commandPalette: "hide",
    nav: "hidden",
    requiresBackend: true,
    rationale: "Dynamic source detail route belongs to the canonical Sources workflow."
  }),
  registryRoute({
    path: "/share/:token",
    canonicalPath: "/shared",
    label: "Public Share",
    group: "knowledge",
    surface: "advanced_self_hosted",
    smoke: "manual",
    commandPalette: "hide",
    nav: "hidden",
    requiresBackend: true,
    rationale: "Tokenized public share route depends on share state and should not appear in navigation."
  }),
  registryRoute({
    path: "/knowledge/thread/:threadId",
    canonicalPath: "/knowledge",
    label: "Knowledge Thread",
    group: "knowledge",
    surface: "default_self_hosted",
    smoke: "manual",
    commandPalette: "hide",
    nav: "hidden",
    requiresBackend: true,
    rationale: "Dynamic knowledge thread routes resolve inside the canonical Knowledge workspace."
  }),
  registryRoute({
    path: "/knowledge/shared/:shareToken",
    canonicalPath: "/knowledge",
    label: "Shared Knowledge Thread",
    group: "knowledge",
    surface: "advanced_self_hosted",
    smoke: "manual",
    commandPalette: "hide",
    nav: "hidden",
    requiresBackend: true,
    rationale: "Shared knowledge thread links are tokenized stateful routes under the Knowledge workspace."
  }),
  registryRoute({
    path: "/companion/conversation",
    label: "Companion Conversation",
    group: "chat",
    surface: "labs_beta",
    availability: [...webAndExtensionOptions, "extension_sidepanel"],
    smoke: "manual",
    requiresBackend: true,
    rationale: "Companion conversation is a nested sidepanel-capable companion workflow."
  }),
  registryRoute({
    path: "/agent",
    canonicalPath: "/agents",
    label: "Sidepanel Agent",
    group: "chat",
    surface: "extension_sidepanel",
    availability: ["extension_sidepanel"],
    smoke: "manual",
    commandPalette: "hide",
    nav: "hidden",
    requiresBackend: true,
    rationale: "Singular sidepanel agent route is extension-only and maps conceptually to the web Agents surface."
  }),
  registryRoute({
    path: "/clipper",
    label: "Sidepanel Clipper",
    group: "knowledge",
    surface: "extension_sidepanel",
    availability: ["extension_sidepanel"],
    smoke: "manual",
    commandPalette: "hide",
    nav: "hidden",
    requiresBackend: true,
    rationale: "Clipper is a sidepanel-only capture workflow and should not be inferred from web route availability."
  }),
  registryRoute({
    path: "/error-boundary-test",
    label: "Sidepanel Error Boundary Test",
    group: "specialized",
    surface: "internal_qa_debug",
    availability: ["extension_sidepanel"],
    smoke: "manual",
    commandPalette: "hide",
    nav: "hidden",
    rationale: "Sidepanel error-boundary route is a development-only QA route."
  }),
  registryRoute({
    path: "/__debug__/sidepanel-chat",
    label: "Debug Sidepanel Chat",
    group: "specialized",
    surface: "internal_qa_debug",
    availability: webOnly,
    smoke: "manual",
    commandPalette: "hide",
    nav: "hidden",
    rationale: "Web debug page mounts the sidepanel chat route for QA without exposing it in navigation."
  }),
  registryRoute({
    path: "/__debug__/sidepanel-error-boundary",
    label: "Debug Sidepanel Error Boundary",
    group: "specialized",
    surface: "internal_qa_debug",
    availability: webOnly,
    smoke: "manual",
    commandPalette: "hide",
    nav: "hidden",
    rationale: "Web debug page mounts the sidepanel error-boundary route for QA without exposing it in navigation."
  }),
  registryRoute({
    path: "/presentation-studio/new",
    label: "New Presentation",
    group: "workspace",
    surface: "labs_beta",
    nav: "hidden",
    rationale: "Presentation creation is a nested task route owned by Presentation Studio."
  }),
  registryRoute({
    path: "/presentation-studio/start",
    label: "Presentation Studio Start",
    group: "workspace",
    surface: "labs_beta",
    smoke: "manual",
    nav: "hidden",
    rationale: "Presentation startup flow is a nested task route owned by Presentation Studio."
  }),
  registryRoute({
    path: "/presentation-studio/:projectId",
    canonicalPath: "/presentation-studio",
    label: "Presentation Detail",
    group: "workspace",
    surface: "labs_beta",
    smoke: "manual",
    commandPalette: "hide",
    nav: "hidden",
    rationale: "Dynamic presentation project route belongs to the canonical Presentation Studio workflow."
  }),
  registryRoute({
    path: "/moderation",
    label: "Moderation Review",
    group: "safety",
    surface: "advanced_self_hosted",
    availability: webAndExtensionOptions,
    requiresBackend: true,
    rationale: "Moderation review is the canonical review route for safety workflows."
  }),
  registryRoute({
    path: "/moderation/rules",
    label: "Moderation Rules",
    group: "safety",
    surface: "advanced_self_hosted",
    availability: webAndExtensionOptions,
    requiresBackend: true,
    rationale: "Moderation rules own policy, blocklist, override, and testing configuration."
  }),
  registryRoute({
    path: "/prototype-workspaces",
    label: "Prototype Workspaces",
    group: "workspace",
    surface: "labs_beta",
    smoke: "manual",
    requiresBackend: true,
    rationale: "Prototype workspaces are a labs route retained for compatibility."
  }),
  registryRoute({
    path: "/research-studio",
    label: "Research Studio",
    group: "workspace",
    surface: "advanced_self_hosted",
    availability: webAndExtensionOptions,
    hostedOptionVisibility: "visible",
    requiresBackend: true,
    rationale: "Research Studio is the canonical route for workspace playground behavior in the shared router."
  }),
  registryRoute({
    path: "/workspace-studio",
    canonicalPath: "/research-studio",
    label: "Workspace Studio",
    group: "workspace",
    surface: "redirect",
    availability: webAndExtensionOptions,
    redirectsTo: "/research-studio",
    smoke: "exclude",
    commandPalette: "alias_only",
    nav: "hidden",
    requiresBackend: true,
    rationale: "Workspace Studio is a compatibility alias to Research Studio in the shared router."
  })
]

export const ROUTE_METADATA: RouteMetadata[] = [
  ...AUDITED_ROUTE_METADATA,
  ...ROUTE_REGISTRY_METADATA
]

export const normalizeRoutePath = (path: string): string => {
  const trimmed = path.trim()
  if (!trimmed) {
    return "/"
  }

  const withoutHash = trimmed.split("#", 1)[0]
  const withoutQuery = withoutHash.split("?", 1)[0] || "/"
  if (withoutQuery === "/") {
    return withoutQuery
  }

  return withoutQuery.replace(/\/+$/, "")
}

const routeMetadataByPath = new Map<string, RouteMetadata>()
const routeMetadataByAlias = new Map<string, RouteMetadata>()

for (const metadata of ROUTE_METADATA) {
  routeMetadataByPath.set(normalizeRoutePath(metadata.path), metadata)

  for (const alias of metadata.aliases ?? []) {
    routeMetadataByAlias.set(normalizeRoutePath(alias), metadata)
  }
}

export const getRouteMetadata = (path: string): RouteMetadata | undefined => {
  const normalizedPath = normalizeRoutePath(path)

  return (
    routeMetadataByPath.get(normalizedPath) ??
    routeMetadataByAlias.get(normalizedPath)
  )
}

export const getCanonicalRoutePath = (path: string): string | undefined =>
  getRouteMetadata(path)?.canonicalPath

export const getRouteCommandPaletteLabel = (
  route: RouteMetadata | string
): string => {
  const metadata = typeof route === "string" ? getRouteMetadata(route) : route

  return metadata?.commandLabel ?? (metadata ? `Go to ${metadata.label}` : "")
}

const h1ExceptionSurfaceReasons: Partial<Record<RouteSurface, string>> = {
  hosted_only: "Hosted-only route can be gated outside self-hosted smoke runs.",
  extension_sidepanel:
    "Extension sidepanel route uses a constrained shell instead of the standard page heading contract.",
  internal_qa_debug:
    "Internal QA/debug route is not part of normal user-facing page inventory.",
  legacy_alias:
    "Legacy alias route delegates ownership to its canonical route.",
  redirect: "Redirect route delegates ownership to its destination route.",
  deprecated: "Deprecated route is retained for compatibility and not normal IA."
}

export const getRouteHeadingPolicy = (
  route: RouteMetadata | string
): RouteHeadingPolicy => {
  const metadata = typeof route === "string" ? getRouteMetadata(route) : route

  if (!metadata) {
    return {
      requiresH1: false,
      exceptionReason: "Route metadata is missing."
    }
  }

  if (metadata.requiresH1 === true) {
    return { requiresH1: true }
  }

  if (metadata.requiresH1 === false) {
    return {
      requiresH1: false,
      exceptionReason: metadata.h1ExceptionReason
    }
  }

  const exceptionReason = h1ExceptionSurfaceReasons[metadata.surface]
  if (exceptionReason) {
    return {
      requiresH1: false,
      exceptionReason: metadata.h1ExceptionReason ?? exceptionReason
    }
  }

  return { requiresH1: true }
}

export const isRouteVisibleForSurface = (
  path: string,
  availability: RouteAvailability
): boolean => getRouteMetadata(path)?.availability.includes(availability) ?? false

export const getRoutesForSmokeInventory = (): RouteMetadata[] =>
  ROUTE_METADATA.filter((metadata) => metadata.smoke === "include")
