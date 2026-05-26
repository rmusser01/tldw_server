import type { LucideIcon } from "lucide-react"
import {
  Activity,
  BrainCircuit,
  BookMarked,
  BookOpen,
  BookText,
  Bot,
  ClipboardList,
  CogIcon,
  CombineIcon,
  Eye,
  FileSearch,
  FileText,
  FlaskConical,
  Gauge,
  GitCompare,
  Headphones,
  Kanban,
  Layers,
  LayoutGrid,
  Library,
  ListTodo,
  MessageSquare,
  Mic,
  Microscope,
  NotebookPen,
  Rss,
  Scissors,
  Server,
  ShieldCheck,
  SquareTerminal,
  SquarePen,
  StickyNote,
  UserCircle2,
  Users,
  Volume2,
  Table2,
  Workflow,
  Zap
} from "lucide-react"
import type { HeaderShortcutId } from "@/services/settings/ui-settings"
import { HEADER_SHORTCUT_IDS } from "@/services/settings/ui-settings"
export { HEADER_SHORTCUT_IDS }
import type { UserPersona } from "@/types/connection"
import {
  CHAT_WORKSPACE_PATH,
  DOCUMENT_WORKSPACE_PATH,
  MODERATION_REVIEW_PATH,
  MODERATION_RULES_PATH,
  REPO2TXT_PATH,
  SOURCES_PATH
} from "@/routes/route-paths"
import { isHostedTldwDeployment } from "@/services/tldw/deployment-mode"

export type HeaderShortcutItem = {
  id: HeaderShortcutId
  to: string
  icon: LucideIcon
  labelKey: string
  labelDefault: string
  /** Optional 1-9 index for ⌘+number shortcut when the launcher is open */
  shortcutIndex?: number
  /** Optional plain-language description for non-technical users */
  descriptionKey?: string
  descriptionDefault?: string
}

export type HeaderShortcutGroup = {
  id: string
  titleKey: string
  titleDefault: string
  items: HeaderShortcutItem[]
}

const BASE_HEADER_SHORTCUT_GROUPS: HeaderShortcutGroup[] = [
  {
    id: "chat-persona",
    titleKey: "option:header.groupChatPersona",
    titleDefault: "Chat & Persona",
    items: [
      {
        id: "chat",
        to: "/chat",
        icon: MessageSquare,
        labelKey: "option:header.modePlayground",
        labelDefault: "Chat",
        shortcutIndex: 1
      },
      {
        id: "chat-workspace",
        to: CHAT_WORKSPACE_PATH,
        icon: SquareTerminal,
        labelKey: "option:header.chatWorkspace",
        labelDefault: "Chat Workspace",
        shortcutIndex: 2,
        descriptionKey: "option:header.chatWorkspaceDesc",
        descriptionDefault:
          "Chat-first workspace with staged sources and runtime context"
      },
      {
        id: "prompts",
        to: "/prompts",
        icon: NotebookPen,
        labelKey: "option:header.modePromptsPlayground",
        labelDefault: "Prompts"
      },
      {
        id: "characters",
        to: "/characters",
        icon: UserCircle2,
        labelKey: "option:header.modeCharacters",
        labelDefault: "Characters",
        shortcutIndex: 4
      },
      {
        id: "chat-dictionaries",
        to: "/dictionaries",
        icon: BookMarked,
        labelKey: "option:header.modeDictionaries",
        labelDefault: "Chat Dictionaries",
        shortcutIndex: 5,
        descriptionKey: "option:header.modeDictionariesDesc",
        descriptionDefault: "Custom word lists for pronunciation and spelling"
      },
      {
        id: "world-books",
        to: "/world-books",
        icon: BookOpen,
        labelKey: "option:header.modeWorldBooks",
        labelDefault: "World Books",
        shortcutIndex: 6,
        descriptionKey: "option:header.modeWorldBooksDesc",
        descriptionDefault: "Shared lore and context injected into character chats"
      },
      {
        id: "model-playground",
        to: "/model-playground",
        icon: FlaskConical,
        labelKey: "settings:modelPlaygroundNav",
        labelDefault: "Model Playground",
        descriptionKey: "settings:modelPlaygroundDesc",
        descriptionDefault: "Compare model outputs side by side"
      }
    ]
  },
  {
    id: "research",
    titleKey: "option:header.groupResearch",
    titleDefault: "Research",
    items: [
      {
        id: "prompt-studio",
        to: "/prompt-studio",
        icon: NotebookPen,
        labelKey: "option:header.modePromptStudio",
        labelDefault: "Prompt Studio",
        shortcutIndex: 3,
        descriptionKey: "option:header.modePromptStudioDesc",
        descriptionDefault: "Design, test, and optimize prompts across models"
      },
      {
        id: "deep-research",
        to: "/research",
        icon: BrainCircuit,
        labelKey: "option:header.deepResearch",
        labelDefault: "Deep Research",
        descriptionKey: "option:header.deepResearchDesc",
        descriptionDefault: "Long-running research with citations and checkpoints"
      },
      {
        id: "research-workspace",
        to: "/research-workspace",
        icon: GitCompare,
        labelKey: "settings:researchWorkspaceNav",
        labelDefault: "Research Workspace",
        shortcutIndex: 7,
        descriptionKey: "settings:researchWorkspaceDesc",
        descriptionDefault: "Three-pane workspace: sources, chat, and generated outputs"
      },
      {
        id: "knowledge-qa",
        to: "/knowledge",
        icon: CombineIcon,
        labelKey: "option:header.modeKnowledge",
        labelDefault: "Knowledge QA",
        shortcutIndex: 8,
        descriptionKey: "option:header.modeKnowledgeDesc",
        descriptionDefault: "Search your ingested documents and get cited answers"
      },
      {
        id: "document-workspace",
        to: DOCUMENT_WORKSPACE_PATH,
        icon: FileSearch,
        labelKey: "option:header.documentWorkspace",
        labelDefault: "Document Workspace"
      },
      {
        id: "repo2txt",
        to: REPO2TXT_PATH,
        icon: FileText,
        labelKey: "option:repo2txt.nav",
        labelDefault: "Repo2Txt",
        descriptionKey: "option:repo2txt.desc",
        descriptionDefault: "Convert code repositories into text for ingestion"
      },
      {
        id: "evaluations",
        to: "/evaluations",
        icon: Microscope,
        labelKey: "option:header.evaluations",
        labelDefault: "Evaluations",
        descriptionKey: "option:header.evaluationsDesc",
        descriptionDefault: "Score and benchmark model quality with automated tests"
      }
    ]
  },
  {
    id: "library",
    titleKey: "option:header.groupLibrary",
    titleDefault: "Library",
    items: [
      {
        id: "media",
        to: "/media",
        icon: BookText,
        labelKey: "option:header.media",
        labelDefault: "Media",
        shortcutIndex: 9
      },
      {
        id: "sources",
        to: SOURCES_PATH,
        icon: Layers,
        labelKey: "option:header.sources",
        labelDefault: "Sources",
        descriptionKey: "option:header.sourcesDesc",
        descriptionDefault:
          "Manage server folders and archive snapshots that sync into notes or media"
      },
      {
        id: "multi-item-review",
        to: "/media-multi",
        icon: LayoutGrid,
        labelKey: "option:header.libraryView",
        labelDefault: "Multi-Item Review"
      },
      {
        id: "collections",
        to: "/collections",
        icon: Library,
        labelKey: "option:header.modeCollections",
        labelDefault: "Collections"
      },
      {
        id: "watchlists",
        to: "/watchlists",
        icon: Rss,
        labelKey: "option:header.modeWatchlists",
        labelDefault: "Watchlists"
      },
      {
        id: "notes",
        to: "/notes",
        icon: StickyNote,
        labelKey: "option:header.notes",
        labelDefault: "Notes"
      }
    ]
  },
  {
    id: "safety",
    titleKey: "option:header.groupSafety",
    titleDefault: "Safety",
    items: [
      {
        id: "family-guardrails",
        to: "/settings/family-guardrails",
        icon: Users,
        labelKey: "settings:familyGuardrailsWizardNav",
        labelDefault: "Family Guardrails",
        descriptionKey: "settings:familyGuardrailsWizardDesc",
        descriptionDefault: "Set up family profiles, safety templates, and invite guardians"
      },
      {
        id: "moderation-review",
        to: MODERATION_REVIEW_PATH,
        icon: ClipboardList,
        labelKey: "option:moderationReview.nav",
        labelDefault: "Moderation Review",
        descriptionKey: "option:moderationReview.desc",
        descriptionDefault: "Review flagged items, escalations, and decisions"
      },
      {
        id: "moderation-rules",
        to: MODERATION_RULES_PATH,
        icon: ShieldCheck,
        labelKey: "option:moderationRules.nav",
        labelDefault: "Content Rules",
        descriptionKey: "option:moderationRules.desc",
        descriptionDefault: "Configure moderation policies, blocklists, and testing"
      },
      {
        id: "guardian",
        to: "/settings/guardian",
        icon: Eye,
        labelKey: "settings:guardianNav",
        labelDefault: "Guardian",
        descriptionKey: "settings:guardianDesc",
        descriptionDefault: "Monitor and manage dependent account activity"
      }
    ]
  },
  {
    id: "creation",
    titleKey: "option:header.groupCreationWorkspace",
    titleDefault: "Creation",
    items: [
      {
        id: "writing-playground",
        to: "/writing-playground",
        icon: SquarePen,
        labelKey: "option:header.writingPlayground",
        labelDefault: "Writing Playground"
      },
      {
        id: "data-tables",
        to: "/data-tables",
        icon: Table2,
        labelKey: "option:header.dataTables",
        labelDefault: "Data Tables"
      },
      {
        id: "stt-playground",
        to: "/stt",
        icon: Mic,
        labelKey: "option:header.modeStt",
        labelDefault: "STT Playground",
        descriptionKey: "option:header.modeSttDesc",
        descriptionDefault: "Speech to Text \u2014 transcribe audio and video"
      },
      {
        id: "tts-playground",
        to: "/tts",
        icon: Volume2,
        labelKey: "option:tts.playground",
        labelDefault: "TTS Playground",
        descriptionKey: "option:header.modeTtsDesc",
        descriptionDefault: "Text to Speech \u2014 generate spoken audio from text"
      },
      {
        id: "audiobook-studio",
        to: "/audiobook-studio",
        icon: Headphones,
        labelKey: "option:header.audiobookStudio",
        labelDefault: "Audiobook Studio"
      },
      {
        id: "presentation-studio",
        to: "/presentation-studio",
        icon: FileText,
        labelKey: "option:header.presentationStudio",
        labelDefault: "Presentation Studio"
      }
    ]
  },
  {
    id: "planning-learning",
    titleKey: "option:header.groupPlanningLearning",
    titleDefault: "Planning & Learning",
    items: [
      {
        id: "kanban-playground",
        to: "/kanban",
        icon: Kanban,
        labelKey: "option:header.modeKanban",
        labelDefault: "Kanban Playground"
      },
      {
        id: "flashcards",
        to: "/flashcards",
        icon: Layers,
        labelKey: "option:header.flashcards",
        labelDefault: "Flashcards"
      },
      {
        id: "quizzes",
        to: "/quiz",
        icon: ClipboardList,
        labelKey: "option:header.quiz",
        labelDefault: "Quizzes"
      }
    ]
  },
  {
    id: "automation-agents",
    titleKey: "option:header.groupAutomationAgents",
    titleDefault: "Automation & Agents",
    items: [
      {
        id: "workflows",
        to: "/workflow-editor",
        icon: Workflow,
        labelKey: "option:header.workflows",
        labelDefault: "Workflows"
      },
      {
        id: "integrations",
        to: "/integrations",
        icon: Bot,
        labelKey: "option:header.integrations",
        labelDefault: "Integrations"
      },
      {
        id: "mcp-hub",
        to: "/mcp-hub",
        icon: Server,
        labelKey: "settings:mcpHubNav",
        labelDefault: "MCP Hub",
        descriptionDefault:
          "Manage MCP servers, tool catalogs, approvals, and ACP profiles"
      },
      {
        id: "scheduled-tasks",
        to: "/scheduled-tasks",
        icon: ListTodo,
        labelKey: "option:header.scheduledTasks",
        labelDefault: "Scheduled Tasks"
      },
      {
        id: "acp-playground",
        to: "/acp-playground",
        icon: Bot,
        labelKey: "option:header.acpPlayground",
        labelDefault: "ACP Playground",
        descriptionKey: "option:header.acpPlaygroundDesc",
        descriptionDefault: "Agent Client Protocol \u2014 run and manage AI agents"
      },
      {
        id: "skills",
        to: "/skills",
        icon: Zap,
        labelKey: "settings:skillsNav",
        labelDefault: "Skills"
      }
    ]
  },
  {
    id: "tools",
    titleKey: "option:header.groupTools",
    titleDefault: "Tools",
    items: [
      {
        id: "chatbooks-playground",
        to: "/chatbooks",
        icon: BookOpen,
        labelKey: "option:header.chatbooksPlayground",
        labelDefault: "Chatbooks Playground",
        descriptionKey: "option:header.chatbooksPlaygroundDesc",
        descriptionDefault: "Export and import chat sessions as portable bundles"
      },
      {
        id: "chunking-playground",
        to: "/chunking-playground",
        icon: Scissors,
        labelKey: "settings:chunkingPlayground.nav",
        labelDefault: "Chunking Playground",
        descriptionKey: "settings:chunkingPlayground.desc",
        descriptionDefault: "Split documents into searchable segments"
      }
    ]
  },
  {
    id: "admin",
    titleKey: "option:header.groupAdminHelp",
    titleDefault: "Admin & Help",
    items: [
      {
        id: "admin-server",
        to: "/admin/server",
        icon: CogIcon,
        labelKey: "option:header.adminServer",
        labelDefault: "Server Admin"
      },
      {
        id: "admin-integrations",
        to: "/admin/integrations",
        icon: CombineIcon,
        labelKey: "option:header.adminIntegrations",
        labelDefault: "Workspace Integrations"
      },
      {
        id: "admin-llamacpp",
        to: "/admin/llamacpp",
        icon: Microscope,
        labelKey: "option:header.adminLlamacpp",
        labelDefault: "Llama.cpp Admin"
      },
      {
        id: "admin-mlx",
        to: "/admin/mlx",
        icon: Gauge,
        labelKey: "option:header.adminMlx",
        labelDefault: "MLX LM Admin"
      },
      {
        id: "admin-monitoring",
        to: "/admin/monitoring",
        icon: Activity,
        labelKey: "option:header.adminMonitoring",
        labelDefault: "Monitoring"
      },
      {
        id: "settings",
        to: "/settings",
        icon: CogIcon,
        labelKey: "settings",
        labelDefault: "Settings"
      },
      {
        id: "documentation",
        to: "/documentation",
        icon: FileText,
        labelKey: "option:header.modeDocumentation",
        labelDefault: "Documentation"
      }
    ]
  }
]

const HOSTED_VISIBLE_SHORTCUT_PATHS = new Set([
  "/chat",
  CHAT_WORKSPACE_PATH,
  "/knowledge",
  "/media",
  "/collections",
  "/stt",
  "/tts"
])

const HOSTED_AUDIO_SHORTCUT_DESCRIPTIONS: Partial<
  Record<
    HeaderShortcutId,
    Pick<HeaderShortcutItem, "descriptionKey" | "descriptionDefault">
  >
> = {
  "stt-playground": {
    descriptionKey: "option:header.modeSttHostedDesc",
    descriptionDefault:
      "Requires a self-hosted tldw server for speech transcription."
  },
  "tts-playground": {
    descriptionKey: "option:header.modeTtsHostedDesc",
    descriptionDefault:
      "Requires a self-hosted tldw server for speech generation."
  }
}

const getHostedHeaderShortcutGroups = (): HeaderShortcutGroup[] => {
  const filteredGroups = BASE_HEADER_SHORTCUT_GROUPS.map((group) => ({
    ...group,
    items: group.items
      .filter((item) => HOSTED_VISIBLE_SHORTCUT_PATHS.has(item.to))
      .map((item) => {
        const hostedDescription = HOSTED_AUDIO_SHORTCUT_DESCRIPTIONS[item.id]
        return hostedDescription
          ? {
              ...item,
              ...hostedDescription
            }
          : item
      })
  })).filter((group) => group.items.length > 0)

  filteredGroups.push({
    id: "account-help",
    titleKey: "option:header.groupAccountHelp",
    titleDefault: "Account & Billing",
    items: [
      {
        id: "account",
        to: "/account",
        icon: UserCircle2,
        labelKey: "option:header.account",
        labelDefault: "Account"
      },
      {
        id: "billing",
        to: "/billing",
        icon: ClipboardList,
        labelKey: "option:header.billing",
        labelDefault: "Billing"
      }
    ]
  })

  return filteredGroups
}

export const getHeaderShortcutGroups = (): HeaderShortcutGroup[] =>
  isHostedTldwDeployment()
    ? getHostedHeaderShortcutGroups()
    : BASE_HEADER_SHORTCUT_GROUPS

export const getHeaderShortcutItems = (): HeaderShortcutItem[] =>
  getHeaderShortcutGroups().flatMap((group) => group.items)

export const normalizeHeaderShortcutSelection = (
  selection: HeaderShortcutId[]
): HeaderShortcutItem[] => {
  const selected = new Set(selection)
  return getHeaderShortcutItems().filter((item) => selected.has(item.id))
}

// ---------------------------------------------------------------------------
// Persona-specific shortcut defaults
// ---------------------------------------------------------------------------

/** Default shortcut selections per persona. Explorer/null = all items. */
export const PERSONA_SHORTCUT_DEFAULTS: Record<
  NonNullable<UserPersona> | "default",
  HeaderShortcutId[]
> = {
  family: [
    "chat",
    "media",
    "family-guardrails",
    "moderation-review",
    "moderation-rules",
    "guardian",
    "settings"
  ],
  researcher: [
    "chat",
    "prompts",
    "deep-research",
    "knowledge-qa",
    "media",
    "research-workspace",
    "collections",
    "notes",
    "evaluations",
    "flashcards",
    "quizzes",
    "settings",
    // required items
    "workflows",
    "acp-playground",
    "integrations",
    "scheduled-tasks",
    "admin-integrations"
  ],
  explorer: [...HEADER_SHORTCUT_IDS],
  default: [...HEADER_SHORTCUT_IDS]
}

/** Get default shortcut selection for a persona. */
export const getDefaultShortcutsForPersona = (
  persona: UserPersona
): HeaderShortcutId[] => {
  if (!persona || persona === "explorer") return [...HEADER_SHORTCUT_IDS]
  const defaults = PERSONA_SHORTCUT_DEFAULTS[persona] ?? HEADER_SHORTCUT_IDS
  return [...defaults]
}
