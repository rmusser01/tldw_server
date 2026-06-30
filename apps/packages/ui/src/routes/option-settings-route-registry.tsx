import { lazy } from "react"
import { Navigate } from "react-router-dom"

import type { RouteDefinition } from "./route-registry"
import { createSettingsRoute } from "./settings-route"

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
const OptionSettingsPromptStudio = createSettingsRoute(
  () => import("~/components/Option/Settings/prompt-studio"),
  "PromptStudioSettings"
)
const OptionSettingsMcpHub = lazy(() => import("./option-settings-mcp-hub"))
const OptionProviderKeysSettings = createSettingsRoute(
  () => import("~/components/Option/Settings/ProviderKeysSettings"),
  "ProviderKeysSettings"
)
const OptionDataManagementSettings = createSettingsRoute(
  () => import("~/components/Option/Settings/data-management"),
  "DataManagementSettings"
)
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
const OptionFamilyGuardrailsWizard = lazy(
  () => import("./option-family-guardrails-wizard")
)
const OptionGuardianSettings = createSettingsRoute(
  () => import("~/components/Option/Settings/GuardianSettings"),
  "GuardianSettings"
)

export const optionSettingsRoutes: RouteDefinition[] = [
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
    path: "/settings/provider-keys",
    element: <OptionProviderKeysSettings />,
  },
  {
    kind: "options",
    path: "/settings/data",
    element: <OptionDataManagementSettings />,
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
  {
    kind: "options",
    path: "/settings/about",
    element: <OptionAbout />,
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
  }
]
