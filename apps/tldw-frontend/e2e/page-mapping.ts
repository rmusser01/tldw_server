/**
 * Page mapping between WebUI and Extension routes
 *
 * This file defines which WebUI routes correspond to which Extension routes,
 * allowing parallel review of shared components.
 */

export type PageCategory =
  | "chat"
  | "settings"
  | "media"
  | "workspace"
  | "knowledge"
  | "audio"
  | "admin"
  | "connectors"
  | "other"

export type ReviewPriority = 1 | 2 | 3 | 4 | 5 | 6 | 7

export interface PageMapping {
  /** Human-readable name */
  name: string
  /** Category for grouping */
  category: PageCategory
  /** WebUI route path (null if extension-only) */
  webuiPath: string | null
  /** Extension options route path (hash-based) - null if WebUI-only */
  extensionOptionsPath: string | null
  /** Extension sidepanel route (if applicable) - null if not sidepanel */
  extensionSidepanelPath: string | null
  /** Shared component name (for reference) */
  sharedComponent: string | null
  /** Review session priority (1 = critical, 7 = extension-specific) */
  session: ReviewPriority
  /** Interaction checklist items */
  checklistItems: string[]
}

/**
 * Interaction checklist templates by category
 */
export const CHECKLIST_TEMPLATES: Record<PageCategory, string[]> = {
  chat: [
    "Send a message and receive response",
    "Streaming response displays correctly",
    "Message history loads",
    "New conversation button works",
    "Model selector changes model (if visible)"
  ],
  settings: [
    "Form fields accept input",
    "Dropdowns open and close",
    "Toggle switches respond to clicks",
    "Save/Apply button is present and clickable",
    "Changes persist after refresh (spot check)"
  ],
  media: [
    "Media list loads (or empty state shows)",
    "Search/filter works",
    "Can open media item detail",
    "Upload button is clickable"
  ],
  workspace: [
    "Primary workspace area renders",
    "Can create/add new item (if applicable)",
    "List/grid displays items",
    "Navigation between sections works"
  ],
  knowledge: [
    "Knowledge items load (or empty state)",
    "Can search/filter items",
    "Can open item detail",
    "Create/edit form works"
  ],
  audio: [
    "Voice/model selector loads options",
    "Play/record controls are present",
    "Audio preview plays (if content exists)"
  ],
  admin: [
    "Data loads from backend (or appropriate empty/error state)",
    "Tables render with data",
    "Action buttons are clickable"
  ],
  connectors: [
    "Connector list loads",
    "Can view connector details",
    "Status indicators show correctly"
  ],
  other: [
    "Page renders without errors",
    "Main content area displays"
  ]
}

/**
 * Complete page mapping for WebUI <-> Extension parallel review
 */
export const PAGE_MAPPINGS: PageMapping[] = [
  // ═══════════════════════════════════════════════════════════════════════════
  // Session 1: Critical Paths (must work)
  // ═══════════════════════════════════════════════════════════════════════════
  {
    name: "Chat",
    category: "chat",
    webuiPath: "/chat",
    extensionOptionsPath: null,
    extensionSidepanelPath: "/",
    sharedComponent: "SidepanelChat",
    session: 1,
    checklistItems: CHECKLIST_TEMPLATES.chat
  },
  {
    name: "Agent Chat",
    category: "chat",
    webuiPath: "/chat/agent",
    extensionOptionsPath: null,
    extensionSidepanelPath: "/agent",
    sharedComponent: "SidepanelAgent",
    session: 1,
    checklistItems: CHECKLIST_TEMPLATES.chat
  },
  {
    name: "Persona Chat",
    category: "chat",
    webuiPath: "/persona",
    extensionOptionsPath: null,
    extensionSidepanelPath: "/persona",
    sharedComponent: "SidepanelPersona",
    session: 1,
    checklistItems: CHECKLIST_TEMPLATES.chat
  },
  {
    name: "TLDW Settings (Server Connection)",
    category: "settings",
    webuiPath: "/settings/tldw",
    extensionOptionsPath: "/settings/tldw",
    extensionSidepanelPath: null,
    sharedComponent: "TldwSettings",
    session: 1,
    checklistItems: [
      "Server URL field accepts input",
      "API key field is present",
      "Connection test button works",
      "Connection status shows correctly",
      "Save persists configuration"
    ]
  },
  {
    name: "Model Settings",
    category: "settings",
    webuiPath: "/settings/model",
    extensionOptionsPath: "/settings/model",
    extensionSidepanelPath: null,
    sharedComponent: "ModelsBody",
    session: 1,
    checklistItems: [
      "Model list loads",
      "Can add/configure a model",
      "Provider dropdown works",
      "API key fields accept input",
      "Save persists settings"
    ]
  },
  {
    name: "Media Library",
    category: "media",
    webuiPath: "/media",
    extensionOptionsPath: "/media",
    extensionSidepanelPath: null,
    sharedComponent: "OptionMedia",
    session: 1,
    checklistItems: CHECKLIST_TEMPLATES.media
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // Session 2: Core Settings
  // ═══════════════════════════════════════════════════════════════════════════
  {
    name: "General Settings",
    category: "settings",
    webuiPath: "/settings",
    extensionOptionsPath: "/settings",
    extensionSidepanelPath: null,
    sharedComponent: "GeneralSettings",
    session: 2,
    checklistItems: CHECKLIST_TEMPLATES.settings
  },
  {
    name: "Chat Settings",
    category: "settings",
    webuiPath: "/settings/chat",
    extensionOptionsPath: "/settings/chat",
    extensionSidepanelPath: null,
    sharedComponent: "ChatSettings",
    session: 2,
    checklistItems: CHECKLIST_TEMPLATES.settings
  },
  {
    name: "Knowledge Settings",
    category: "settings",
    webuiPath: "/settings/knowledge",
    extensionOptionsPath: "/settings/knowledge",
    extensionSidepanelPath: null,
    sharedComponent: "KnowledgeSettings",
    session: 2,
    checklistItems: CHECKLIST_TEMPLATES.settings
  },
  {
    name: "RAG Settings",
    category: "settings",
    webuiPath: "/settings/rag",
    extensionOptionsPath: "/settings/rag",
    extensionSidepanelPath: null,
    sharedComponent: "RagSettings",
    session: 2,
    checklistItems: CHECKLIST_TEMPLATES.settings
  },
  {
    name: "Speech Settings",
    category: "settings",
    webuiPath: "/settings/speech",
    extensionOptionsPath: "/settings/speech",
    extensionSidepanelPath: null,
    sharedComponent: "SpeechSettings",
    session: 2,
    checklistItems: CHECKLIST_TEMPLATES.settings
  },
  {
    name: "Health Settings",
    category: "settings",
    webuiPath: "/settings/health",
    extensionOptionsPath: "/settings/health",
    extensionSidepanelPath: null,
    sharedComponent: "HealthSettings",
    session: 2,
    checklistItems: [
      "Health status indicator shows",
      "Backend connection status visible",
      "Refresh/check button works"
    ]
  },
  {
    name: "Family Guardrails Wizard",
    category: "settings",
    webuiPath: "/settings/family-guardrails",
    extensionOptionsPath: "/settings/family-guardrails",
    extensionSidepanelPath: null,
    sharedComponent: "FamilyGuardrailsWizard",
    session: 2,
    checklistItems: [
      "Wizard stepper renders all core setup steps",
      "Can progress through household + guardian + dependent setup",
      "Template apply controls are visible",
      "Acceptance tracker shows queued/active statuses",
      "Back/Save & Continue controls remain responsive"
    ]
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // Session 3: Knowledge & Content
  // ═══════════════════════════════════════════════════════════════════════════
  {
    name: "Knowledge QA Workspace",
    category: "knowledge",
    webuiPath: "/knowledge",
    extensionOptionsPath: "/knowledge",
    extensionSidepanelPath: null,
    sharedComponent: "KnowledgeQA",
    session: 3,
    checklistItems: CHECKLIST_TEMPLATES.knowledge
  },
  {
    name: "Characters Workspace",
    category: "knowledge",
    webuiPath: "/characters",
    extensionOptionsPath: "/characters",
    extensionSidepanelPath: null,
    sharedComponent: "OptionCharactersWorkspace",
    session: 3,
    checklistItems: CHECKLIST_TEMPLATES.knowledge
  },
  {
    name: "World Books Workspace",
    category: "knowledge",
    webuiPath: "/world-books",
    extensionOptionsPath: "/world-books",
    extensionSidepanelPath: null,
    sharedComponent: "OptionWorldBooksWorkspace",
    session: 3,
    checklistItems: CHECKLIST_TEMPLATES.knowledge
  },
  {
    name: "Dictionaries Workspace",
    category: "knowledge",
    webuiPath: "/dictionaries",
    extensionOptionsPath: "/dictionaries",
    extensionSidepanelPath: null,
    sharedComponent: "OptionDictionariesWorkspace",
    session: 3,
    checklistItems: CHECKLIST_TEMPLATES.knowledge
  },
  {
    name: "Prompts Workspace",
    category: "knowledge",
    webuiPath: "/prompts",
    extensionOptionsPath: "/prompts",
    extensionSidepanelPath: null,
    sharedComponent: "OptionPromptsWorkspace",
    session: 3,
    checklistItems: CHECKLIST_TEMPLATES.knowledge
  },
  {
    name: "Notes",
    category: "workspace",
    webuiPath: "/notes",
    extensionOptionsPath: "/notes",
    extensionSidepanelPath: null,
    sharedComponent: "OptionNotes",
    session: 3,
    checklistItems: [
      "Notes list loads (or empty state)",
      "Can create new note",
      "Can edit existing note",
      "Delete/archive works"
    ]
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // Session 4: Workspace Tools
  // ═══════════════════════════════════════════════════════════════════════════
  {
    name: "Flashcards",
    category: "workspace",
    webuiPath: "/flashcards",
    extensionOptionsPath: "/flashcards",
    extensionSidepanelPath: null,
    sharedComponent: "OptionFlashcards",
    session: 4,
    checklistItems: [
      "Flashcard list/deck loads",
      "Can create new flashcard",
      "Flip card interaction works",
      "Study mode functions"
    ]
  },
  {
    name: "Quiz",
    category: "workspace",
    webuiPath: "/quiz",
    extensionOptionsPath: "/quiz",
    extensionSidepanelPath: null,
    sharedComponent: "OptionQuiz",
    session: 4,
    checklistItems: [
      "Quiz list loads",
      "Can start a quiz",
      "Answer selection works",
      "Results display correctly"
    ]
  },
  {
    name: "Content Review",
    category: "workspace",
    webuiPath: "/content-review",
    extensionOptionsPath: "/content-review",
    extensionSidepanelPath: null,
    sharedComponent: "OptionContentReview",
    session: 4,
    checklistItems: CHECKLIST_TEMPLATES.workspace
  },
  {
    name: "Media Multi (Review)",
    category: "workspace",
    webuiPath: "/media-multi",
    extensionOptionsPath: "/media-multi",
    extensionSidepanelPath: null,
    sharedComponent: "OptionMediaMulti",
    session: 4,
    checklistItems: [
      "Multi-media view loads",
      "Can select multiple items",
      "Bulk actions available",
      "Filter/sort works"
    ]
  },
  {
    name: "Kanban",
    category: "workspace",
    webuiPath: "/kanban",
    extensionOptionsPath: "/kanban",
    extensionSidepanelPath: null,
    sharedComponent: "OptionKanbanPlayground",
    session: 4,
    checklistItems: [
      "Kanban board loads",
      "Columns display correctly",
      "Can drag cards between columns",
      "Can create new cards"
    ]
  },
  {
    name: "Collections",
    category: "workspace",
    webuiPath: "/collections",
    extensionOptionsPath: "/collections",
    extensionSidepanelPath: null,
    sharedComponent: "OptionCollections",
    session: 4,
    checklistItems: CHECKLIST_TEMPLATES.workspace
  },
  {
    name: "Data Tables",
    category: "workspace",
    webuiPath: "/data-tables",
    extensionOptionsPath: "/data-tables",
    extensionSidepanelPath: null,
    sharedComponent: "OptionDataTables",
    session: 4,
    checklistItems: [
      "Table loads with data",
      "Sorting works",
      "Filtering works",
      "Can edit cells (if editable)"
    ]
  },
  {
    name: "Media Trash",
    category: "media",
    webuiPath: "/media-trash",
    extensionOptionsPath: "/media-trash",
    extensionSidepanelPath: null,
    sharedComponent: "OptionMediaTrash",
    session: 4,
    checklistItems: [
      "Trash items load (or empty state)",
      "Can restore items",
      "Can permanently delete",
      "Empty trash works"
    ]
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // Session 5: Audio & Advanced
  // ═══════════════════════════════════════════════════════════════════════════
  {
    name: "TTS (Text-to-Speech)",
    category: "audio",
    webuiPath: "/tts",
    extensionOptionsPath: "/tts",
    extensionSidepanelPath: null,
    sharedComponent: "OptionTts",
    session: 5,
    checklistItems: [
      "Voice selector loads",
      "Text input accepts text",
      "Generate button works",
      "Audio playback works"
    ]
  },
  {
    name: "STT (Speech-to-Text)",
    category: "audio",
    webuiPath: "/stt",
    extensionOptionsPath: "/stt",
    extensionSidepanelPath: null,
    sharedComponent: "OptionStt",
    session: 5,
    checklistItems: [
      "Model/provider selector loads",
      "Record button is present",
      "File upload works",
      "Transcription result displays"
    ]
  },
  {
    name: "Speech",
    category: "audio",
    webuiPath: "/speech",
    extensionOptionsPath: "/speech",
    extensionSidepanelPath: null,
    sharedComponent: "OptionSpeech",
    session: 5,
    checklistItems: CHECKLIST_TEMPLATES.audio
  },
  {
    name: "Evaluations",
    category: "workspace",
    webuiPath: "/evaluations",
    extensionOptionsPath: "/evaluations",
    extensionSidepanelPath: null,
    sharedComponent: "OptionEvaluations",
    session: 5,
    checklistItems: [
      "Evaluation list loads",
      "Can create new evaluation",
      "Run evaluation button works",
      "Results display correctly"
    ]
  },
  {
    name: "Prompt Studio",
    category: "workspace",
    webuiPath: "/prompts?tab=studio",
    extensionOptionsPath: "/prompts?tab=studio",
    extensionSidepanelPath: null,
    sharedComponent: "OptionPromptsWorkspace",
    session: 5,
    checklistItems: [
      "Studio tab loads",
      "Project list loads from server",
      "Can pull prompts to local",
      "Can push local prompts to studio"
    ]
  },
  {
    name: "Chatbooks",
    category: "workspace",
    webuiPath: "/chatbooks",
    extensionOptionsPath: "/chatbooks",
    extensionSidepanelPath: null,
    sharedComponent: "OptionChatbooksPlayground",
    session: 5,
    checklistItems: [
      "Chatbooks list loads",
      "Can import chatbook",
      "Can export chatbook",
      "Preview works"
    ]
  },
  {
    name: "Watchlists",
    category: "workspace",
    webuiPath: "/watchlists",
    extensionOptionsPath: "/watchlists",
    extensionSidepanelPath: null,
    sharedComponent: "OptionWatchlists",
    session: 5,
    checklistItems: CHECKLIST_TEMPLATES.workspace
  },
  {
    name: "Documentation",
    category: "other",
    webuiPath: "/documentation",
    extensionOptionsPath: "/documentation",
    extensionSidepanelPath: null,
    sharedComponent: "OptionDocumentation",
    session: 7,
    checklistItems: ["Docs page loads", "Content renders"]
  },
  {
    name: "Quick Chat Popout",
    category: "other",
    webuiPath: "/quick-chat-popout",
    extensionOptionsPath: "/quick-chat-popout",
    extensionSidepanelPath: null,
    sharedComponent: "OptionQuickChatPopout",
    session: 7,
    checklistItems: ["Popout loads", "Chat works"]
  },
  {
    name: "Setup",
    category: "other",
    webuiPath: "/setup",
    extensionOptionsPath: "/setup",
    extensionSidepanelPath: null,
    sharedComponent: "OptionSetup",
    session: 7,
    checklistItems: ["Setup wizard loads", "Can proceed through steps"]
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // Session 6: Admin & Settings Details (WebUI-focused)
  // ═══════════════════════════════════════════════════════════════════════════
  {
    name: "Admin Server",
    category: "admin",
    webuiPath: "/admin/server",
    extensionOptionsPath: "/admin/server",
    extensionSidepanelPath: null,
    sharedComponent: "OptionAdminServer",
    session: 6,
    checklistItems: CHECKLIST_TEMPLATES.admin
  },
  {
    name: "Admin LlamaCpp",
    category: "admin",
    webuiPath: "/admin/llamacpp",
    extensionOptionsPath: "/admin/llamacpp",
    extensionSidepanelPath: null,
    sharedComponent: "OptionAdminLlamacpp",
    session: 6,
    checklistItems: [
      "Server status loads",
      "Model list displays",
      "Can start/stop server",
      "Configuration options work"
    ]
  },
  {
    name: "Admin MLX",
    category: "admin",
    webuiPath: "/admin/mlx",
    extensionOptionsPath: "/admin/mlx",
    extensionSidepanelPath: null,
    sharedComponent: "OptionAdminMlx",
    session: 6,
    checklistItems: CHECKLIST_TEMPLATES.admin
  },
  {
    name: "Evaluations Settings",
    category: "settings",
    webuiPath: "/settings/evaluations",
    extensionOptionsPath: "/settings/evaluations",
    extensionSidepanelPath: null,
    sharedComponent: "EvaluationsSettings",
    session: 6,
    checklistItems: CHECKLIST_TEMPLATES.settings
  },
  {
    name: "Chatbooks Settings",
    category: "settings",
    webuiPath: "/settings/chatbooks",
    extensionOptionsPath: "/settings/chatbooks",
    extensionSidepanelPath: null,
    sharedComponent: "ChatbooksSettings",
    session: 6,
    checklistItems: CHECKLIST_TEMPLATES.settings
  },
  {
    name: "Characters Settings",
    category: "settings",
    webuiPath: "/settings/characters",
    extensionOptionsPath: "/settings/characters",
    extensionSidepanelPath: null,
    sharedComponent: "CharactersWorkspaceSettings",
    session: 6,
    checklistItems: CHECKLIST_TEMPLATES.settings
  },
  {
    name: "World Books Settings",
    category: "settings",
    webuiPath: "/settings/world-books",
    extensionOptionsPath: "/settings/world-books",
    extensionSidepanelPath: null,
    sharedComponent: "WorldBooksWorkspaceSettings",
    session: 6,
    checklistItems: CHECKLIST_TEMPLATES.settings
  },
  {
    name: "Dictionaries Settings",
    category: "settings",
    webuiPath: "/settings/chat-dictionaries",
    extensionOptionsPath: "/settings/chat-dictionaries",
    extensionSidepanelPath: null,
    sharedComponent: "DictionariesWorkspaceSettings",
    session: 6,
    checklistItems: CHECKLIST_TEMPLATES.settings
  },
  {
    name: "Prompt Studio Settings",
    category: "settings",
    webuiPath: "/settings/prompt-studio",
    extensionOptionsPath: "/settings/prompt-studio",
    extensionSidepanelPath: null,
    sharedComponent: "PromptStudioSettings",
    session: 6,
    checklistItems: CHECKLIST_TEMPLATES.settings
  },
  {
    name: "About",
    category: "settings",
    webuiPath: "/settings/about",
    extensionOptionsPath: "/settings/about",
    extensionSidepanelPath: null,
    sharedComponent: "AboutApp",
    session: 6,
    checklistItems: [
      "Version info displays",
      "Links work",
      "About content renders"
    ]
  },
  {
    name: "Moderation Review",
    category: "workspace",
    webuiPath: "/moderation",
    extensionOptionsPath: "/moderation",
    extensionSidepanelPath: null,
    sharedComponent: "ModerationReviewShell",
    session: 6,
    checklistItems: [
      "Review shell renders",
      "Queue unavailable status is clear",
      "Review item detail shows decision history and redaction state",
      "Bulk selection reports partial failures",
      "Saved filter presets and scoped shortcuts work",
      "Content Rules link works"
    ]
  },
  {
    name: "Content Rules",
    category: "workspace",
    webuiPath: "/moderation/rules",
    extensionOptionsPath: "/moderation/rules",
    extensionSidepanelPath: null,
    sharedComponent: "ModerationPlaygroundShell",
    session: 6,
    checklistItems: [
      "Policy tabs render",
      "Blocklist and override controls are reachable",
      "Rule test sandbox works"
    ]
  },
  {
    name: "Chunking Playground",
    category: "workspace",
    webuiPath: "/chunking-playground",
    extensionOptionsPath: "/chunking-playground",
    extensionSidepanelPath: null,
    sharedComponent: "OptionChunkingPlayground",
    session: 6,
    checklistItems: [
      "Text input works",
      "Chunking options available",
      "Preview displays chunks"
    ]
  },
]

// ═══════════════════════════════════════════════════════════════════════════
// Extension-Only Pages (no WebUI equivalent)
// ═══════════════════════════════════════════════════════════════════════════
export const EXTENSION_ONLY_PAGES: PageMapping[] = [
  {
    name: "Sidepanel Settings",
    category: "settings",
    webuiPath: null,
    extensionOptionsPath: null,
    extensionSidepanelPath: "/settings",
    sharedComponent: "SidepanelSettings",
    session: 7,
    checklistItems: [
      "Settings panel opens",
      "Options are interactive",
      "Changes apply"
    ]
  },
  {
    name: "Sidepanel Error Boundary Test",
    category: "settings",
    webuiPath: null,
    extensionOptionsPath: null,
    extensionSidepanelPath: "/error-boundary-test",
    sharedComponent: "SidepanelErrorBoundaryTest",
    session: 7,
    checklistItems: ["Error boundary renders", "Recovery action available"]
  },
  {
    name: "UI Customization Settings",
    category: "settings",
    webuiPath: null,
    extensionOptionsPath: "/settings/ui",
    extensionSidepanelPath: null,
    sharedComponent: "UiCustomizationSettings",
    session: 7,
    checklistItems: [
      "Theme options available",
      "Color/font options work",
      "Preview updates"
    ]
  },
  {
    name: "Quick Ingest Settings",
    category: "settings",
    webuiPath: null,
    extensionOptionsPath: "/settings/quick-ingest",
    extensionSidepanelPath: null,
    sharedComponent: "QuickIngestSettings",
    session: 7,
    checklistItems: CHECKLIST_TEMPLATES.settings
  },
  {
    name: "Image Generation Settings",
    category: "settings",
    webuiPath: null,
    extensionOptionsPath: "/settings/image-generation",
    extensionSidepanelPath: null,
    sharedComponent: "ImageGenerationSettings",
    session: 7,
    checklistItems: CHECKLIST_TEMPLATES.settings
  },
  {
    name: "Writing Playground",
    category: "workspace",
    webuiPath: null,
    extensionOptionsPath: "/writing-playground",
    extensionSidepanelPath: null,
    sharedComponent: "OptionWritingPlayground",
    session: 7,
    checklistItems: [
      "Editor loads",
      "Can type text",
      "AI assist features work"
    ]
  },
  {
    name: "Model Playground",
    category: "workspace",
    webuiPath: null,
    extensionOptionsPath: "/model-playground",
    extensionSidepanelPath: null,
    sharedComponent: "OptionModelPlayground",
    session: 7,
    checklistItems: [
      "Model selector works",
      "Input accepts prompts",
      "Can run inference"
    ]
  },
  {
    name: "Audiobook Studio",
    category: "audio",
    webuiPath: null,
    extensionOptionsPath: "/audiobook-studio",
    extensionSidepanelPath: null,
    sharedComponent: "OptionAudiobookStudio",
    session: 7,
    checklistItems: [
      "Studio interface loads",
      "Can import content",
      "Voice options available",
      "Preview/generate works"
    ]
  },
  {
    name: "Workflow Editor",
    category: "workspace",
    webuiPath: null,
    extensionOptionsPath: "/workflow-editor",
    extensionSidepanelPath: null,
    sharedComponent: "OptionWorkflowEditor",
    session: 7,
    checklistItems: [
      "Editor canvas loads",
      "Can add nodes",
      "Connections work"
    ]
  },
  {
    name: "Research Workspace",
    category: "workspace",
    webuiPath: null,
    extensionOptionsPath: "/research-workspace",
    extensionSidepanelPath: null,
    sharedComponent: "OptionResearchWorkspace",
    session: 7,
    checklistItems: CHECKLIST_TEMPLATES.workspace
  }
]

// ═══════════════════════════════════════════════════════════════════════════
// WebUI-Only Pages (no extension equivalent)
// ═══════════════════════════════════════════════════════════════════════════
export const WEBUI_ONLY_PAGES: PageMapping[] = [
  {
    name: "Home",
    category: "other",
    webuiPath: "/",
    extensionOptionsPath: null,
    extensionSidepanelPath: null,
    sharedComponent: null,
    session: 6,
    checklistItems: ["Page loads", "Navigation works"]
  },
  {
    name: "Login",
    category: "other",
    webuiPath: "/login",
    extensionOptionsPath: null,
    extensionSidepanelPath: null,
    sharedComponent: null,
    session: 6,
    checklistItems: ["Login form displays", "Can submit credentials"]
  },
  {
    name: "Config",
    category: "other",
    webuiPath: "/config",
    extensionOptionsPath: null,
    extensionSidepanelPath: null,
    sharedComponent: null,
    session: 6,
    checklistItems: ["Config page loads", "Options are visible"]
  },
  {
    name: "Profile",
    category: "other",
    webuiPath: "/profile",
    extensionOptionsPath: null,
    extensionSidepanelPath: null,
    sharedComponent: null,
    session: 6,
    checklistItems: ["Profile loads", "User info displays"]
  },
  {
    name: "Privileges",
    category: "other",
    webuiPath: "/privileges",
    extensionOptionsPath: null,
    extensionSidepanelPath: null,
    sharedComponent: null,
    session: 6,
    checklistItems: ["Privileges page loads"]
  },
  {
    name: "Search",
    category: "workspace",
    webuiPath: "/search",
    extensionOptionsPath: null,
    extensionSidepanelPath: null,
    sharedComponent: null,
    session: 4,
    checklistItems: ["Search input works", "Results display"]
  },
  {
    name: "Review",
    category: "workspace",
    webuiPath: "/review",
    extensionOptionsPath: null,
    extensionSidepanelPath: null,
    sharedComponent: null,
    session: 4,
    checklistItems: CHECKLIST_TEMPLATES.workspace
  },
  {
    name: "Reading",
    category: "workspace",
    webuiPath: "/reading",
    extensionOptionsPath: null,
    extensionSidepanelPath: null,
    sharedComponent: null,
    session: 4,
    checklistItems: ["Reading view loads", "Content displays"]
  },
  {
    name: "Items",
    category: "workspace",
    webuiPath: "/items",
    extensionOptionsPath: null,
    extensionSidepanelPath: null,
    sharedComponent: null,
    session: 4,
    checklistItems: CHECKLIST_TEMPLATES.workspace
  },
  {
    name: "Admin",
    category: "admin",
    webuiPath: "/admin",
    extensionOptionsPath: null,
    extensionSidepanelPath: null,
    sharedComponent: null,
    session: 6,
    checklistItems: CHECKLIST_TEMPLATES.admin
  },
  {
    name: "Admin Orgs",
    category: "admin",
    webuiPath: "/admin/orgs",
    extensionOptionsPath: null,
    extensionSidepanelPath: null,
    sharedComponent: null,
    session: 6,
    checklistItems: CHECKLIST_TEMPLATES.admin
  },
  {
    name: "Admin Data Ops",
    category: "admin",
    webuiPath: "/admin/data-ops",
    extensionOptionsPath: null,
    extensionSidepanelPath: null,
    sharedComponent: null,
    session: 6,
    checklistItems: CHECKLIST_TEMPLATES.admin
  },
  {
    name: "Admin Watchlists Items",
    category: "admin",
    webuiPath: "/admin/watchlists-items",
    extensionOptionsPath: null,
    extensionSidepanelPath: null,
    sharedComponent: null,
    session: 6,
    checklistItems: CHECKLIST_TEMPLATES.admin
  },
  {
    name: "Admin Watchlists Runs",
    category: "admin",
    webuiPath: "/admin/watchlists-runs",
    extensionOptionsPath: null,
    extensionSidepanelPath: null,
    sharedComponent: null,
    session: 6,
    checklistItems: CHECKLIST_TEMPLATES.admin
  },
  {
    name: "Admin Maintenance",
    category: "admin",
    webuiPath: "/admin/maintenance",
    extensionOptionsPath: null,
    extensionSidepanelPath: null,
    sharedComponent: null,
    session: 6,
    checklistItems: CHECKLIST_TEMPLATES.admin
  },
  {
    name: "Connectors",
    category: "connectors",
    webuiPath: "/connectors",
    extensionOptionsPath: null,
    extensionSidepanelPath: null,
    sharedComponent: null,
    session: 6,
    checklistItems: CHECKLIST_TEMPLATES.connectors
  },
  {
    name: "Connectors Browse",
    category: "connectors",
    webuiPath: "/connectors/browse",
    extensionOptionsPath: null,
    extensionSidepanelPath: null,
    sharedComponent: null,
    session: 6,
    checklistItems: CHECKLIST_TEMPLATES.connectors
  },
  {
    name: "Connectors Jobs",
    category: "connectors",
    webuiPath: "/connectors/jobs",
    extensionOptionsPath: null,
    extensionSidepanelPath: null,
    sharedComponent: null,
    session: 6,
    checklistItems: CHECKLIST_TEMPLATES.connectors
  },
  {
    name: "Connectors Sources",
    category: "connectors",
    webuiPath: "/connectors/sources",
    extensionOptionsPath: null,
    extensionSidepanelPath: null,
    sharedComponent: null,
    session: 6,
    checklistItems: CHECKLIST_TEMPLATES.connectors
  },
  {
    name: "Claims Review",
    category: "workspace",
    webuiPath: "/claims-review",
    extensionOptionsPath: null,
    extensionSidepanelPath: null,
    sharedComponent: null,
    session: 4,
    checklistItems: CHECKLIST_TEMPLATES.workspace
  },
  {
    name: "Audio",
    category: "audio",
    webuiPath: "/audio",
    extensionOptionsPath: null,
    extensionSidepanelPath: null,
    sharedComponent: null,
    session: 5,
    checklistItems: CHECKLIST_TEMPLATES.audio
  }
]

/**
 * Get all pages for a specific session
 */
export function getPagesBySession(session: ReviewPriority): PageMapping[] {
  return [...PAGE_MAPPINGS, ...WEBUI_ONLY_PAGES, ...EXTENSION_ONLY_PAGES].filter(
    (p) => p.session === session
  )
}

/**
 * Get all shared pages (have both WebUI and Extension routes)
 */
export function getSharedPages(): PageMapping[] {
  return PAGE_MAPPINGS.filter(
    (p) => p.extensionOptionsPath !== null || p.extensionSidepanelPath !== null
  )
}

/**
 * Get WebUI-only pages
 */
export function getWebuiOnlyPages(): PageMapping[] {
  return [...WEBUI_ONLY_PAGES]
}

export function getExtensionOnlyPages(): PageMapping[] {
  return [...EXTENSION_ONLY_PAGES]
}

export const TOTAL_PAGE_COUNT =
  PAGE_MAPPINGS.length + WEBUI_ONLY_PAGES.length + EXTENSION_ONLY_PAGES.length

/**
 * Get pages by category
 */
export function getPagesByCategory(category: PageCategory): PageMapping[] {
  return [...PAGE_MAPPINGS, ...WEBUI_ONLY_PAGES, ...EXTENSION_ONLY_PAGES].filter(
    (p) => p.category === category
  )
}

export const SHARED_PAGE_COUNT = getSharedPages().length
