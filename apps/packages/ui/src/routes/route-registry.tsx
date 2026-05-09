import { lazy } from "react"
import type { ReactElement } from "react"
import { ALL_TARGETS, type PlatformTarget } from "@/config/platform"
import { createSettingsRoute } from "./settings-route"
import { Navigate } from "react-router-dom"
import {
  CHAT_WORKSPACE_PATH,
  DOCUMENT_WORKSPACE_PATH,
  PROTOTYPE_WORKSPACES_PATH,
  REPO2TXT_PATH
} from "@/routes/route-paths"
import { isHostedTldwDeployment } from "@/services/tldw/deployment-mode"
import { isHostedVisibleOptionPath } from "./option-route-visibility"

import OptionIndex from "./option-index"

export type RouteKind = "options" | "sidepanel"

export type RouteDefinition = {
  kind: RouteKind
  path: string
  element: ReactElement
  targets?: PlatformTarget[]
}

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
const OptionRagSettings = createSettingsRoute(
  () => import("~/components/Option/Settings/rag"),
  "RagSettings"
)
const OptionTldwSettings = createSettingsRoute(
  () => import("~/components/Option/Settings/tldw"),
  "TldwSettings"
)
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
const OptionAdminRuntimeConfig = lazy(() => import("./option-admin-runtime-config"))
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
const OptionChunkingPlayground = lazy(() => import("./option-chunking-playground"))
const OptionDocumentation = lazy(() => import("./option-documentation"))
const OptionQuiz = lazy(() => import("./option-quiz"))
const OptionWritingPlayground = lazy(() => import("./option-writing-playground"))
const OptionDocumentWorkspace = lazy(() => import("./option-document-workspace"))
const OptionModelPlayground = lazy(() => import("./option-model-playground"))
const OptionModerationPlayground = lazy(() => import("./option-moderation-playground"))
const OptionFamilyGuardrailsWizard = lazy(
  () => import("./option-family-guardrails-wizard")
)
const OptionGuardianSettings = createSettingsRoute(
  () => import("~/components/Option/Settings/GuardianSettings"),
  "GuardianSettings"
)
const OptionChatbooksPlayground = lazy(() => import("./option-chatbooks-playground"))
const OptionWatchlists = lazy(() => import("./option-watchlists"))
const OptionIntegrations = lazy(() => import("./option-integrations"))
const OptionAdminIntegrations = lazy(() => import("./option-admin-integrations"))
const OptionScheduledTasks = lazy(() => import("./option-scheduled-tasks"))
const OptionCompanion = lazy(() => import("./option-companion"))
const OptionCompanionConversation = lazy(
  () => import("./option-companion-conversation")
)
const OptionKanbanPlayground = lazy(() => import("./option-kanban-playground"))
const OptionDataTables = lazy(() => import("./option-data-tables"))
const OptionCollections = lazy(() => import("./option-collections"))
const OptionSources = lazy(() => import("./option-sources"))
const OptionSourcesNew = lazy(() => import("./option-sources-new"))
const OptionSourcesDetail = lazy(() => import("./option-sources-detail"))
const OptionAdminSources = lazy(() => import("./option-admin-sources"))
const OptionAudiobookStudio = lazy(() => import("./option-audiobook-studio"))
const OptionPresentationStudio = lazy(() => import("./option-presentation-studio"))
const OptionPresentationStudioNew = lazy(() => import("./option-presentation-studio-new"))
const OptionPresentationStudioStart = lazy(() => import("./option-presentation-studio-start"))
const OptionPresentationStudioDetail = lazy(
  () => import("./option-presentation-studio-detail")
)
const OptionChatWorkflows = lazy(() => import("./option-chat-workflows"))
const OptionWorkflowEditor = lazy(() => import("./option-workflow-editor"))
const OptionACPPlayground = lazy(() => import("./option-acp-playground"))
const OptionAgents = lazy(() => import("./option-agents"))
const OptionAgentTasks = lazy(() => import("./option-agent-tasks"))
const OptionMcpHub = lazy(() => import("./option-mcp-hub"))
const OptionSettingsMcpHub = lazy(() => import("./option-settings-mcp-hub"))
const OptionSkills = lazy(() => import("./option-skills"))
const OptionRepo2Txt = lazy(() => import("./option-repo2txt"))
const OptionSetup = lazy(() => import("./option-setup"))
const OptionOnboardingTest = lazy(() => import("./option-onboarding-test"))
const OptionWorkspacePlayground = lazy(() => import("./option-workspace-playground"))
const OptionChatWorkspace = lazy(() => import("./option-chat-workspace"))
const OptionPrototypeWorkspaces = lazy(() => import("./option-prototype-workspaces"))
const OptionSharedWithMe = lazy(() => import("./option-shared-with-me"))
const OptionPublicShare = lazy(() => import("./option-public-share"))

export const ROUTE_DEFINITIONS: RouteDefinition[] = [
  { kind: "options", path: "/", element: <OptionIndex /> },
  { kind: "options", path: "/setup", element: <OptionSetup /> },
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
  },
  {
    kind: "options",
    path: "/settings/tldw",
    element: <OptionTldwSettings />,
  },
  {
    kind: "options",
    path: "/settings/model",
    element: <OptionModal />,
  },
  {
    kind: "options",
    path: "/settings/mcp-hub",
    element: <OptionSettingsMcpHub />,
  },
  {
    kind: "options",
    path: "/settings/prompt",
    element: <OptionPrompt />,
  },
  {
    kind: "options",
    path: "/settings/evaluations",
    element: <OptionSettingsEvaluations />,
  },
  {
    kind: "options",
    path: "/settings/chat",
    element: <OptionChatSettings />,
  },
  {
    kind: "options",
    path: "/settings/ui",
    element: <OptionUiCustomization />,
  },
  {
    kind: "options",
    path: "/settings/splash",
    element: <OptionSplashSettings />,
  },
  {
    kind: "options",
    path: "/settings/quick-ingest",
    element: <OptionQuickIngestSettings />,
  },
  {
    kind: "options",
    path: "/settings/speech",
    element: <OptionSpeechSettings />,
  },
  {
    kind: "options",
    path: "/settings/image-generation",
    element: <OptionImageGenerationSettings />,
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
  },
  { kind: "options", path: "/settings/processed", element: <OptionProcessed /> },
  {
    kind: "options",
    path: "/settings/health",
    element: <OptionHealth />,
  },
  {
    kind: "options",
    path: "/settings/prompt-studio",
    element: <OptionSettingsPromptStudio />,
  },
  {
    kind: "options",
    path: "/settings/knowledge",
    element: <OptionKnowledgeBase />,
  },
  {
    kind: "options",
    path: "/settings/chatbooks",
    element: <OptionChatbooks />,
  },
  {
    kind: "options",
    path: "/shared",
    element: <OptionSharedWithMe />,
  },
  {
    kind: "options",
    path: "/settings/characters",
    element: <OptionCharacters />,
  },
  {
    kind: "options",
    path: "/settings/world-books",
    element: <OptionWorldBooks />,
  },
  {
    kind: "options",
    path: "/settings/chat-dictionaries",
    element: <OptionDictionaries />,
  },
  {
    kind: "options",
    path: "/settings/rag",
    element: <OptionRagSettings />,
  },
  { kind: "options", path: "/chunking-playground", element: <OptionChunkingPlayground /> },
  { kind: "options", path: "/documentation", element: <OptionDocumentation /> },
  {
    kind: "options",
    path: "/settings/about",
    element: <OptionAbout />,
  },
  {
    kind: "options",
    path: "/flashcards",
    element: <OptionFlashcards />,
  },
  {
    kind: "options",
    path: "/quiz",
    element: <OptionQuiz />,
    targets: ALL_TARGETS,
  },
  {
    kind: "options",
    path: "/writing-playground",
    element: <OptionWritingPlayground />,
  },
  {
    kind: "options",
    path: REPO2TXT_PATH,
    element: <OptionRepo2Txt />,
  },
  {
    kind: "options",
    path: "/model-playground",
    element: <OptionModelPlayground />,
  },
  { kind: "options", path: "/chatbooks", element: <OptionChatbooksPlayground /> },
  { kind: "options", path: "/watchlists", element: <OptionWatchlists /> },
  {
    kind: "options",
    path: "/integrations",
    element: <OptionIntegrations />,
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
  },
  { kind: "options", path: "/kanban", element: <OptionKanbanPlayground /> },
  {
    kind: "options",
    path: "/data-tables",
    element: <OptionDataTables />,
  },
  {
    kind: "options",
    path: "/collections",
    element: <OptionCollections />,
  },
  {
    kind: "options",
    path: "/sources",
    element: <OptionSources />,
  },
  { kind: "options", path: "/sources/new", element: <OptionSourcesNew /> },
  { kind: "options", path: "/sources/:sourceId", element: <OptionSourcesDetail /> },
  { kind: "options", path: "/admin/sources", element: <OptionAdminSources /> },
  {
    kind: "options",
    path: "/companion",
    element: <OptionCompanion />,
  },
  {
    kind: "options",
    path: "/companion/conversation",
    element: <OptionCompanionConversation />,
    targets: ALL_TARGETS
  },
  {
    kind: "options",
    path: "/notes",
    element: <OptionNotes />,
  },
  { kind: "options", path: "/share/:token", element: <OptionPublicShare /> },
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
    path: "/audiobook-studio",
    element: <OptionAudiobookStudio />,
  },
  {
    kind: "options",
    path: "/presentation-studio",
    element: <OptionPresentationStudio />,
  },
  {
    kind: "options",
    path: "/presentation-studio/new",
    element: <OptionPresentationStudioNew />
  },
  {
    kind: "options",
    path: "/presentation-studio/start",
    element: <OptionPresentationStudioStart />
  },
  {
    kind: "options",
    path: "/presentation-studio/:projectId",
    element: <OptionPresentationStudioDetail />
  },
  {
    kind: "options",
    path: "/chat-workflows",
    element: <OptionChatWorkflows />,
  },
  {
    kind: "options",
    path: "/workflow-editor",
    element: <OptionWorkflowEditor />,
  },
  {
    kind: "options",
    path: "/acp-playground",
    element: <OptionACPPlayground />,
  },
  {
    kind: "options",
    path: "/agents",
    element: <OptionAgents />,
  },
  {
    kind: "options",
    path: "/agent-tasks",
    element: <OptionAgentTasks />,
  },
  {
    kind: "options",
    path: "/mcp-hub",
    element: <OptionMcpHub />,
  },
  {
    kind: "options",
    path: "/skills",
    element: <OptionSkills />,
  },
  {
    kind: "options",
    path: "/workspace-playground",
    element: <OptionWorkspacePlayground />,
  },
  {
    kind: "options",
    path: CHAT_WORKSPACE_PATH,
    element: <OptionChatWorkspace />,
  },
  {
    kind: "options",
    path: PROTOTYPE_WORKSPACES_PATH,
    element: <OptionPrototypeWorkspaces />,
  },
  {
    kind: "options",
    path: DOCUMENT_WORKSPACE_PATH,
    element: <OptionDocumentWorkspace />,
  },
  {
    kind: "options",
    path: "/moderation-playground",
    element: <OptionModerationPlayground />,
    targets: ALL_TARGETS,
  },
  {
    kind: "options",
    path: "/settings/family-guardrails",
    element: <OptionFamilyGuardrailsWizard />,
  },
  {
    kind: "options",
    path: "/settings/guardian",
    element: <OptionGuardianSettings />,
  },
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
  },
  {
    kind: "options",
    path: "/admin/mlx",
    element: <OptionAdminMlx />,
    targets: ALL_TARGETS,
  },
  {
    kind: "options",
    path: "/admin/runtime-config",
    element: <OptionAdminRuntimeConfig />,
    targets: ALL_TARGETS,
  },
  {
    kind: "options",
    path: "/admin/monitoring",
    element: <OptionAdminMonitoring />,
    targets: ALL_TARGETS,
  },
  {
    kind: "options",
    path: "/quick-chat-popout",
    element: <OptionQuickChatPopout />,
    targets: ALL_TARGETS
  }
]

export const optionRoutes = ROUTE_DEFINITIONS.filter(
  (route) =>
    route.kind === "options" &&
    (!isHostedTldwDeployment() || isHostedVisibleOptionPath(route.path))
)
