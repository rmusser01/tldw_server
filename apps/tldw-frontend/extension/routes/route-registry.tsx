import { lazy } from "react"
import type { ReactElement } from "react"
import type { LucideIcon } from "lucide-react"
import {
  ActivityIcon,
  BookIcon,
  BookMarked,
  BookOpen,
  BookText,
  Bot,
  BrainCircuitIcon,
  CombineIcon,
  CpuIcon,
  Gauge,
  InfoIcon,
  OrbitIcon,
  ServerIcon,
  ShareIcon,
  Layers,
  StickyNote,
  Microscope,
  FlaskConical,
  MessageSquare,
  ClipboardList,
  MicIcon,
  ImageIcon,
  SlidersHorizontal,
  Sparkles,
  Trash2,
  Table2,
  Library,
  ListTodo,
  PenLine,
  ShieldCheck,
  SquareTerminal
} from "lucide-react"
import { Navigate } from "react-router-dom"
import { ALL_TARGETS, type PlatformTarget } from "@/config/platform"
import { createSettingsRoute } from "./settings-route"
import {
  MODERATION_PLAYGROUND_LEGACY_PATH,
  MODERATION_REVIEW_PATH,
  MODERATION_RULES_PATH
} from "@/routes/route-paths"

export type RouteKind = "options" | "sidepanel"

export type NavGroupKey = "server" | "knowledge" | "workspace" | "about"

type RouteNav = {
  group: NavGroupKey
  labelToken: string
  icon: LucideIcon
  order: number
  beta?: boolean
}

export type RouteDefinition = {
  kind: RouteKind
  path: string
  element: ReactElement
  targets?: PlatformTarget[]
  nav?: RouteNav
}

const OptionIndex = lazy(() => import("./option-index"))
const OptionOnboardingTest = lazy(() => import("./option-onboarding-test"))
const OptionSettings = createSettingsRoute(
  () => import("~/components/Option/Settings/general-settings"),
  "GeneralSettings"
)
const OptionModal = createSettingsRoute(
  () => import("~/components/Option/Models"),
  "ModelsBody"
)
const OptionPrompt = createSettingsRoute(
  () => import("~/components/Option/Settings/WorkspaceLinks"),
  "PromptWorkspaceSettings"
)
const OptionShare = createSettingsRoute(
  () => import("~/components/Option/Share"),
  "OptionShareBody"
)
const OptionProcessed = lazy(() => import("./option-settings-processed"))
const OptionHealth = lazy(() => import("./option-settings-health"))
const OptionMediaTrash = lazy(() => import("./option-media-trash"))
const OptionKnowledgeBase = createSettingsRoute(
  () => import("~/components/Option/Knowledge"),
  "KnowledgeSettings"
)
const OptionAbout = createSettingsRoute(
  () => import("~/components/Option/Settings/about"),
  "AboutApp"
)
const OptionChatbooks = createSettingsRoute(
  () => import("~/components/Option/Settings/chatbooks"),
  "ChatbooksSettings"
)
const SidepanelChat = lazy(() => import("./sidepanel-chat"))
const SidepanelSettings = lazy(() => import("./sidepanel-settings"))
const SidepanelAgent = lazy(() => import("./sidepanel-agent"))
const SidepanelPersona = lazy(() => import("./sidepanel-persona"))
const SidepanelErrorBoundaryTest = lazy(() => import("./sidepanel-error-boundary-test"))
const OptionRagSettings = createSettingsRoute(
  () => import("~/components/Option/Settings/rag"),
  "RagSettings"
)
const OptionTldwSettings = createSettingsRoute(
  () => import("~/components/Option/Settings/tldw"),
  "TldwSettings"
)
const OptionProviderKeysSettings = createSettingsRoute(
  () => import("~/components/Option/Settings/ProviderKeysSettings"),
  "ProviderKeysSettings"
)
const OptionMedia = lazy(() => import("./option-media"))
const OptionMediaCollection = lazy(() => import("./option-media-collection"))
const OptionMediaMulti = lazy(() => import("./option-media-multi"))
const OptionNotes = lazy(() => import("./option-notes"))
const OptionWorldBooks = createSettingsRoute(
  () => import("~/components/Option/Settings/WorkspaceLinks"),
  "WorldBooksWorkspaceSettings"
)
const OptionDictionaries = createSettingsRoute(
  () => import("~/components/Option/Settings/WorkspaceLinks"),
  "DictionariesWorkspaceSettings"
)
const OptionCharacters = createSettingsRoute(
  () => import("~/components/Option/Settings/WorkspaceLinks"),
  "CharactersWorkspaceSettings"
)
const OptionWorldBooksWorkspace = lazy(() => import("./option-world-books"))
const OptionDictionariesWorkspace = lazy(() => import("./option-dictionaries"))
const OptionCharactersWorkspace = lazy(() => import("./option-characters"))
const OptionPromptsWorkspace = lazy(() => import("./option-prompts"))
const OptionKnowledgeWorkspace = lazy(() => import("./option-knowledge"))
const OptionFlashcards = lazy(() => import("./option-flashcards"))
const OptionTts = lazy(() => import("./option-tts"))
const OptionEvaluations = lazy(() => import("./option-evaluations"))
const OptionStt = lazy(() => import("./option-stt"))
const OptionSpeech = lazy(() => import("./option-speech"))
const OptionSettingsEvaluations = createSettingsRoute(
  () => import("~/components/Option/Settings/evaluations"),
  "EvaluationsSettings"
)
const OptionSpeechSettings = createSettingsRoute(
  () => import("@/components/Option/Settings/SpeechSettings"),
  "SpeechSettings"
)
const OptionImageGenerationSettings = createSettingsRoute(
  () => import("~/components/Option/Settings/ImageGenerationSettings"),
  "ImageGenerationSettings"
)
// Note: OptionPromptStudio has been unified with OptionPromptsWorkspace (/prompts)
// The /prompt-studio route now redirects to /prompts?tab=studio
const OptionSettingsPromptStudio = createSettingsRoute(
  () => import("~/components/Option/Settings/prompt-studio"),
  "PromptStudioSettings"
)
const OptionAdminServer = lazy(() => import("./option-admin-server"))
const OptionAdminLlamacpp = lazy(() => import("./option-admin-llamacpp"))
const OptionAdminMlx = lazy(() => import("./option-admin-mlx"))
const OptionAdminMonitoring = lazy(() => import("./option-admin-monitoring"))
const OptionChatSettings = createSettingsRoute(
  () => import("~/components/Option/Settings/ChatSettings"),
  "ChatSettings"
)
const OptionUiCustomization = createSettingsRoute(
  () => import("~/components/Option/Settings/ui-customization"),
  "UiCustomizationSettings"
)
const OptionSplashSettings = createSettingsRoute(
  () => import("~/components/Option/Settings/splash"),
  "SplashSettings"
)
const OptionQuickIngestSettings = createSettingsRoute(
  () => import("~/components/Option/Settings/QuickIngestSettings"),
  "QuickIngestSettings"
)
const OptionQuickChatPopout = lazy(() => import("./option-quick-chat-popout"))
const OptionContentReview = lazy(() => import("./option-content-review"))
const OptionACPPlayground = lazy(() => import("./option-acp-playground"))
const OptionMcpHub = lazy(() => import("./option-mcp-hub"))
const OptionSettingsMcpHub = lazy(() => import("./option-settings-mcp-hub"))
const OptionChunkingPlayground = lazy(() => import("./option-chunking-playground"))
const OptionDocumentation = lazy(() => import("./option-documentation"))
const OptionQuiz = lazy(() => import("./option-quiz"))
const OptionChatbooksPlayground = lazy(() => import("./option-chatbooks-playground"))
const OptionWatchlists = lazy(() => import("./option-watchlists"))
const OptionIntegrations = lazy(() => import("./option-integrations"))
const OptionAdminIntegrations = lazy(() => import("./option-admin-integrations"))
const OptionScheduledTasks = lazy(() => import("./option-scheduled-tasks"))
const OptionKanbanPlayground = lazy(() => import("./option-kanban-playground"))
const OptionDataTables = lazy(() => import("./option-data-tables"))
const OptionCollections = lazy(() => import("./option-collections"))
const OptionSources = lazy(() => import("./option-sources"))
const OptionSourcesNew = lazy(() => import("./option-sources-new"))
const OptionSourcesDetail = lazy(() => import("./option-sources-detail"))
const OptionWritingPlayground = lazy(() => import("./option-writing-playground"))
const OptionModerationReview = lazy(() => import("./option-moderation-review"))
const OptionModerationRules = lazy(() => import("./option-moderation-rules"))
const OptionModerationPlayground = lazy(() => import("./option-moderation-playground"))
const OptionFamilyGuardrailsWizard = lazy(
  () => import("./option-family-guardrails-wizard")
)
const OptionGuardianSettings = createSettingsRoute(
  () => import("~/components/Option/Settings/GuardianSettings"),
  "GuardianSettings"
)
const OptionResearchWorkspace = lazy(
  () => import("./option-research-workspace")
)
const OptionChatWorkspace = lazy(() => import("./option-chat-workspace"))
const OptionAdminSources = lazy(() => import("./option-admin-sources"))

const ERROR_BOUNDARY_TEST_ENABLED = process.env.NODE_ENV !== "production"

const errorBoundaryRoutes: RouteDefinition[] = ERROR_BOUNDARY_TEST_ENABLED
  ? [
      {
        kind: "sidepanel",
        path: "/error-boundary-test",
        element: <SidepanelErrorBoundaryTest />,
        targets: ALL_TARGETS
      }
    ]
  : []

export const ROUTE_DEFINITIONS: RouteDefinition[] = [
  { kind: "options", path: "/", element: <OptionIndex /> },
  {
    kind: "options",
    path: "/onboarding-test",
    element: <OptionOnboardingTest />,
    targets: ALL_TARGETS
  },
  {
    kind: "options",
    path: "/settings",
    element: <OptionSettings />,
    nav: {
      group: "server",
      labelToken: "settings:generalSettings.title",
      icon: OrbitIcon,
      order: 2
    }
  },
  {
    kind: "options",
    path: "/settings/tldw",
    element: <OptionTldwSettings />,
    nav: {
      group: "server",
      labelToken: "settings:tldw.serverNav",
      icon: ServerIcon,
      order: 1
    }
  },
  {
    kind: "options",
    path: "/settings/provider-keys",
    element: <OptionProviderKeysSettings />,
    nav: {
      group: "server",
      labelToken: "settings:providerKeys.navTitle",
      icon: ServerIcon,
      order: 1.5
    }
  },
  {
    kind: "options",
    path: "/settings/model",
    element: <OptionModal />,
    nav: {
      group: "server",
      labelToken: "settings:manageModels.title",
      icon: BrainCircuitIcon,
      order: 6
    }
  },
  {
    kind: "options",
    path: "/settings/prompt",
    element: <OptionPrompt />,
    nav: {
      group: "workspace",
      labelToken: "settings:managePrompts.title",
      icon: BookIcon,
      order: 6
    }
  },
  {
    kind: "options",
    path: "/settings/evaluations",
    element: <OptionSettingsEvaluations />,
    nav: {
      group: "server",
      labelToken: "settings:evaluationsSettings.title",
      icon: FlaskConical,
      order: 9,
      beta: true
    }
  },
  {
    kind: "options",
    path: "/settings/chat",
    element: <OptionChatSettings />,
    nav: {
      group: "server",
      labelToken: "settings:chatSettingsNav",
      icon: MessageSquare,
      order: 3
    }
  },
  {
    kind: "options",
    path: "/settings/ui",
    element: <OptionUiCustomization />,
    nav: {
      group: "server",
      labelToken: "settings:uiCustomizationNav",
      icon: SlidersHorizontal,
      order: 3.5
    }
  },
  {
    kind: "options",
    path: "/settings/splash",
    element: <OptionSplashSettings />,
    nav: {
      group: "server",
      labelToken: "settings:splashSettingsNav",
      icon: Sparkles,
      order: 3.6
    }
  },
  {
    kind: "options",
    path: "/settings/quick-ingest",
    element: <OptionQuickIngestSettings />,
    nav: {
      group: "server",
      labelToken: "settings:quickIngestSettingsNav",
      icon: ClipboardList,
      order: 4
    }
  },
  {
    kind: "options",
    path: "/settings/speech",
    element: <OptionSpeechSettings />,
    nav: {
      group: "server",
      labelToken: "settings:speechSettingsNav",
      icon: MicIcon,
      order: 5
    }
  },
  {
    kind: "options",
    path: "/settings/image-generation",
    element: <OptionImageGenerationSettings />,
    nav: {
      group: "server",
      labelToken: "settings:imageGenerationSettingsNav",
      icon: ImageIcon,
      order: 7
    }
  },
  {
    kind: "options",
    path: "/settings/image-gen",
    element: <Navigate to="/settings/image-generation" replace />
  },
  {
    kind: "options",
    path: "/settings/share",
    element: <OptionShare />,
    nav: {
      group: "workspace",
      labelToken: "settings:manageShare.title",
      icon: ShareIcon,
      order: 8
    }
  },
  { kind: "options", path: "/settings/processed", element: <OptionProcessed /> },
  {
    kind: "options",
    path: "/settings/health",
    element: <OptionHealth />,
    nav: {
      group: "server",
      labelToken: "settings:healthNav",
      icon: ActivityIcon,
      order: 11
    }
  },
  {
    kind: "options",
    path: "/settings/prompt-studio",
    element: <OptionSettingsPromptStudio />,
    nav: {
      group: "server",
      labelToken: "settings:promptStudio.nav",
      icon: Microscope,
      order: 10,
      beta: true
    }
  },
  {
    kind: "options",
    path: "/settings/mcp-hub",
    element: <OptionSettingsMcpHub />,
    nav: {
      group: "server",
      labelToken: "settings:mcpHubNav",
      icon: ServerIcon,
      order: 11.1,
      beta: true
    }
  },
  {
    kind: "options",
    path: "/settings/knowledge",
    element: <OptionKnowledgeBase />,
    nav: {
      group: "knowledge",
      labelToken: "settings:manageKnowledge.title",
      icon: BookText,
      order: 1
    }
  },
  {
    kind: "options",
    path: "/settings/chatbooks",
    element: <OptionChatbooks />,
    nav: {
      group: "knowledge",
      labelToken: "settings:chatbooksNav",
      icon: BookText,
      order: 4
    }
  },
  {
    kind: "options",
    path: "/settings/characters",
    element: <OptionCharacters />,
    nav: {
      group: "knowledge",
      labelToken: "settings:charactersNav",
      icon: BookIcon,
      order: 5
    }
  },
  {
    kind: "options",
    path: "/settings/world-books",
    element: <OptionWorldBooks />,
    nav: {
      group: "knowledge",
      labelToken: "settings:worldBooksNav",
      icon: BookOpen,
      order: 2
    }
  },
  {
    kind: "options",
    path: "/settings/chat-dictionaries",
    element: <OptionDictionaries />,
    nav: {
      group: "knowledge",
      labelToken: "settings:chatDictionariesNav",
      icon: BookMarked,
      order: 3
    }
  },
  {
    kind: "options",
    path: "/settings/rag",
    element: <OptionRagSettings />,
    nav: {
      group: "server",
      labelToken: "settings:rag.title",
      icon: CombineIcon,
      order: 4
    }
  },
  { kind: "options", path: "/chunking-playground", element: <OptionChunkingPlayground /> },
  { kind: "options", path: "/documentation", element: <OptionDocumentation /> },
  {
    kind: "options",
    path: "/settings/about",
    element: <OptionAbout />,
    nav: {
      group: "about",
      labelToken: "settings:about.title",
      icon: InfoIcon,
      order: 1
    }
  },
  { kind: "options", path: "/review", element: <OptionMediaMulti /> },
  {
    kind: "options",
    path: "/research-workspace",
    element: <OptionResearchWorkspace />,
    nav: {
      group: "workspace",
      labelToken: "settings:researchWorkspaceNav",
      icon: FlaskConical,
      order: 0,
      beta: true
    }
  },
  {
    kind: "options",
    path: "/chat-workspace",
    element: <OptionChatWorkspace />,
    nav: {
      group: "workspace",
      labelToken: "option:header.chatWorkspace",
      icon: SquareTerminal,
      order: 1
    }
  },
  {
    kind: "options",
    path: "/flashcards",
    element: <OptionFlashcards />,
    nav: {
      group: "workspace",
      labelToken: "option:header.flashcards",
      icon: Layers,
      order: 4
    }
  },
  {
    kind: "options",
    path: "/quiz",
    element: <OptionQuiz />,
    targets: ALL_TARGETS,
    nav: {
      group: "workspace",
      labelToken: "option:header.quiz",
      icon: ClipboardList,
      order: 5,
      beta: true
    }
  },
  {
    kind: "options",
    path: "/writing-playground",
    element: <OptionWritingPlayground />,
    nav: {
      group: "workspace",
      labelToken: "option:header.writingPlayground",
      icon: PenLine,
      order: 7
    }
  },
  {
    kind: "options",
    path: MODERATION_REVIEW_PATH,
    element: <OptionModerationReview />,
    nav: {
      group: "server",
      labelToken: "option:moderationReview.nav",
      icon: ShieldCheck,
      order: 12
    }
  },
  {
    kind: "options",
    path: MODERATION_RULES_PATH,
    element: <OptionModerationRules />,
    nav: {
      group: "server",
      labelToken: "option:moderationRules.nav",
      icon: ShieldCheck,
      order: 12.1
    }
  },
  {
    kind: "options",
    path: MODERATION_PLAYGROUND_LEGACY_PATH,
    element: <OptionModerationPlayground />
  },
  {
    kind: "options",
    path: "/settings/family-guardrails",
    element: <OptionFamilyGuardrailsWizard />,
    nav: {
      group: "server",
      labelToken: "settings:familyGuardrailsWizardNav",
      icon: ShieldCheck,
      order: 8,
      beta: true
    }
  },
  {
    kind: "options",
    path: "/settings/guardian",
    element: <OptionGuardianSettings />,
    nav: {
      group: "server",
      labelToken: "settings:guardianNav",
      icon: ShieldCheck,
      order: 9,
      beta: true
    }
  },
  { kind: "options", path: "/chatbooks", element: <OptionChatbooksPlayground /> },
  { kind: "options", path: "/watchlists", element: <OptionWatchlists /> },
  {
    kind: "options",
    path: "/integrations",
    element: <OptionIntegrations />,
    nav: {
      group: "workspace",
      labelToken: "option:header.integrations",
      icon: Bot,
      order: 3.2
    }
  },
  {
    kind: "options",
    path: "/admin/integrations",
    element: <OptionAdminIntegrations />
  },
  {
    kind: "options",
    path: "/scheduled-tasks",
    element: <OptionScheduledTasks />,
    nav: {
      group: "workspace",
      labelToken: "option:header.scheduledTasks",
      icon: ListTodo,
      order: 3.4
    }
  },
  { kind: "options", path: "/kanban", element: <OptionKanbanPlayground /> },
  {
    kind: "options",
    path: "/data-tables",
    element: <OptionDataTables />,
    nav: {
      group: "workspace",
      labelToken: "option:header.dataTables",
      icon: Table2,
      order: 9,
      beta: true
    }
  },
  {
    kind: "options",
    path: "/collections",
    element: <OptionCollections />,
    nav: {
      group: "workspace",
      labelToken: "option:header.collections",
      icon: Library,
      order: 10,
      beta: true
    }
  },
  {
    kind: "options",
    path: "/sources",
    element: <OptionSources />,
    nav: {
      group: "workspace",
      labelToken: "option:header.sources",
      icon: Layers,
      order: 9.5,
      beta: true
    }
  },
  { kind: "options", path: "/sources/new", element: <OptionSourcesNew /> },
  { kind: "options", path: "/sources/:sourceId", element: <OptionSourcesDetail /> },
  { kind: "options", path: "/admin/sources", element: <OptionAdminSources /> },
  {
    kind: "options",
    path: "/media",
    element: <OptionMedia />,
    nav: {
      group: "knowledge",
      labelToken: "settings:mediaNav",
      icon: BookText,
      order: 6
    }
  },
  {
    kind: "options",
    path: "/media-trash",
    element: <OptionMediaTrash />,
    nav: {
      group: "knowledge",
      labelToken: "settings:mediaTrashNav",
      icon: Trash2,
      order: 7
    }
  },
  {
    kind: "options",
    path: "/media-multi",
    element: <OptionMediaMulti />,
    nav: {
      group: "workspace",
      labelToken: "option:header.libraryView",
      icon: Microscope,
      order: 1
    }
  },
  {
    kind: "options",
    path: "/media-collections/:collectionId",
    element: <OptionMediaCollection />
  },
  {
    kind: "options",
    path: "/content-review",
    element: <OptionContentReview />,
    nav: {
      group: "workspace",
      labelToken: "option:header.contentReview",
      icon: BookText,
      order: 2
    }
  },
  {
    kind: "options",
    path: "/notes",
    element: <OptionNotes />,
    nav: {
      group: "workspace",
      labelToken: "option:header.notes",
      icon: StickyNote,
      order: 3
    }
  },
  { kind: "options", path: "/knowledge", element: <OptionKnowledgeWorkspace /> },
  { kind: "options", path: "/knowledge/thread/:threadId", element: <OptionKnowledgeWorkspace /> },
  { kind: "options", path: "/knowledge/shared/:shareToken", element: <OptionKnowledgeWorkspace /> },
  { kind: "options", path: "/world-books", element: <OptionWorldBooksWorkspace /> },
  { kind: "options", path: "/dictionaries", element: <OptionDictionariesWorkspace /> },
  { kind: "options", path: "/characters", element: <OptionCharactersWorkspace /> },
  { kind: "options", path: "/prompts", element: <OptionPromptsWorkspace /> },
  // Legacy route - redirect to unified Prompts page
  { kind: "options", path: "/prompt-studio", element: <Navigate to="/prompts?tab=studio" replace /> },
  { kind: "options", path: "/tts", element: <OptionTts /> },
  { kind: "options", path: "/stt", element: <OptionStt /> },
  { kind: "options", path: "/speech", element: <OptionSpeech /> },
  { kind: "options", path: "/evaluations", element: <OptionEvaluations /> },
  {
    kind: "options",
    path: "/acp-playground",
    element: <OptionACPPlayground />,
    nav: {
      group: "workspace",
      labelToken: "settings:acpPlaygroundNav",
      icon: Bot,
      order: 11,
      beta: true
    }
  },
  { kind: "options", path: "/mcp-hub", element: <OptionMcpHub /> },
  {
    kind: "options",
    path: "/admin/server",
    element: <OptionAdminServer />,
    targets: ALL_TARGETS
  },
  {
    kind: "options",
    path: "/admin/llamacpp",
    element: <OptionAdminLlamacpp />,
    targets: ALL_TARGETS,
    nav: {
      group: "server",
      labelToken: "option:header.adminLlamacpp",
      icon: CpuIcon,
      order: 7
    }
  },
  {
    kind: "options",
    path: "/admin/mlx",
    element: <OptionAdminMlx />,
    targets: ALL_TARGETS,
    nav: {
      group: "server",
      labelToken: "option:header.adminMlx",
      icon: Gauge,
      order: 8
    }
  },
  {
    kind: "options",
    path: "/admin/monitoring",
    element: <OptionAdminMonitoring />,
    targets: ALL_TARGETS,
    nav: {
      group: "server",
      labelToken: "option:header.adminMonitoring",
      icon: ActivityIcon,
      order: 9
    }
  },
  {
    kind: "options",
    path: "/quick-chat-popout",
    element: <OptionQuickChatPopout />,
    targets: ALL_TARGETS
  },
  { kind: "sidepanel", path: "/", element: <SidepanelChat /> },
  {
    kind: "sidepanel",
    path: "/chat",
    element: <SidepanelChat />,
    targets: ALL_TARGETS
  },
  {
    kind: "sidepanel",
    path: "/agent",
    element: <SidepanelAgent />,
    targets: ALL_TARGETS
  },
  {
    kind: "sidepanel",
    path: "/persona",
    element: <SidepanelPersona />,
    targets: ALL_TARGETS
  },
  { kind: "sidepanel", path: "/settings", element: <SidepanelSettings /> },
  ...errorBoundaryRoutes
]

export const optionRoutes = ROUTE_DEFINITIONS.filter(
  (route) => route.kind === "options"
)

export const sidepanelRoutes = ROUTE_DEFINITIONS.filter(
  (route) => route.kind === "sidepanel"
)
