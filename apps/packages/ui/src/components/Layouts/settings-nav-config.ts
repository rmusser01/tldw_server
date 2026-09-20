import type { LucideIcon } from "lucide-react"
import {
  ActivityIcon,
  BookIcon,
  BookMarked,
  BookOpen,
  BookText,
  BrainCircuitIcon,
  ClipboardList,
  CombineIcon,
  DatabaseIcon,
  Eye,
  FlaskConical,
  ImageIcon,
  InfoIcon,
  MessageSquare,
  MicIcon,
  Microscope,
  OrbitIcon,
  ServerIcon,
  ShareIcon,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Users,
} from "lucide-react"
import {
  MODERATION_REVIEW_PATH,
  MODERATION_RULES_PATH
} from "@/routes/route-paths"

export type NavGroupKey =
  | "setupRecovery"
  | "preferencesWorkflow"
  | "dataAdmin"

export type SettingsNavRouteMeta = {
  path: string
  group: NavGroupKey
  labelToken: string
  icon: LucideIcon
  order: number
  beta?: boolean
}

export const SETTINGS_ROUTE_NAV_ITEMS: SettingsNavRouteMeta[] = [
  {
    path: "/settings",
    group: "setupRecovery",
    labelToken: "settings:setupRecovery.title",
    icon: OrbitIcon,
    order: 1
  },
  {
    path: "/settings/preferences",
    group: "preferencesWorkflow",
    labelToken: "settings:preferencesSettings.navTitle",
    icon: SlidersHorizontal,
    order: 1
  },
  {
    path: "/settings/tldw",
    group: "setupRecovery",
    labelToken: "settings:tldw.serverNav",
    icon: ServerIcon,
    order: 2
  },
  {
    path: "/settings/provider-keys",
    group: "setupRecovery",
    labelToken: "settings:providerKeys.navTitle",
    icon: ServerIcon,
    order: 3
  },
  {
    path: "/settings/data",
    group: "dataAdmin",
    labelToken: "settings:dataManagement.navTitle",
    icon: DatabaseIcon,
    order: 1
  },
  {
    path: "/settings/model",
    group: "setupRecovery",
    labelToken: "settings:manageModels.title",
    icon: BrainCircuitIcon,
    order: 4
  },
  {
    path: "/settings/mcp-hub",
    group: "dataAdmin",
    labelToken: "settings:mcpHubNav",
    icon: ServerIcon,
    order: 8
  },
  {
    path: "/settings/prompt",
    group: "preferencesWorkflow",
    labelToken: "settings:servicePrompts.title",
    icon: BookIcon,
    order: 8
  },
  {
    path: "/settings/evaluations",
    group: "dataAdmin",
    labelToken: "settings:evaluationsSettings.title",
    icon: FlaskConical,
    order: 3,
    beta: true
  },
  {
    path: "/settings/chat",
    group: "preferencesWorkflow",
    labelToken: "settings:chatSettingsNav",
    icon: MessageSquare,
    order: 2
  },
  {
    path: "/settings/chat-macros",
    group: "preferencesWorkflow",
    labelToken: "settings:chatMacrosNav",
    icon: CombineIcon,
    order: 2.5
  },
  {
    path: "/settings/ui",
    group: "preferencesWorkflow",
    labelToken: "settings:uiCustomizationNav",
    icon: SlidersHorizontal,
    order: 3
  },
  {
    path: "/settings/splash",
    group: "preferencesWorkflow",
    labelToken: "settings:splashSettingsNav",
    icon: Sparkles,
    order: 3.5
  },
  {
    path: "/settings/quick-ingest",
    group: "preferencesWorkflow",
    labelToken: "settings:quickIngestSettingsNav",
    icon: ClipboardList,
    order: 4
  },
  {
    path: "/settings/speech",
    group: "preferencesWorkflow",
    labelToken: "settings:speechSettingsNav",
    icon: MicIcon,
    order: 5
  },
  {
    path: "/settings/image-generation",
    group: "preferencesWorkflow",
    labelToken: "settings:imageGenerationSettingsNav",
    icon: ImageIcon,
    order: 6
  },
  {
    path: "/settings/share",
    group: "dataAdmin",
    labelToken: "settings:manageShare.title",
    icon: ShareIcon,
    order: 2
  },
  {
    path: "/settings/health",
    group: "setupRecovery",
    labelToken: "settings:healthNav",
    icon: ActivityIcon,
    order: 5
  },
  {
    path: "/settings/prompt-studio",
    group: "preferencesWorkflow",
    labelToken: "settings:promptStudio.nav",
    icon: Microscope,
    order: 9,
    beta: true
  },
  {
    path: "/settings/knowledge",
    group: "preferencesWorkflow",
    labelToken: "settings:manageKnowledge.title",
    icon: BookText,
    order: 7
  },
  {
    path: "/settings/chatbooks",
    group: "preferencesWorkflow",
    labelToken: "settings:chatbooksNav",
    icon: BookText,
    order: 10
  },
  {
    path: "/settings/characters",
    group: "preferencesWorkflow",
    labelToken: "settings:charactersNav",
    icon: BookIcon,
    order: 11
  },
  {
    path: "/settings/world-books",
    group: "preferencesWorkflow",
    labelToken: "settings:worldBooksNav",
    icon: BookOpen,
    order: 12
  },
  {
    path: "/settings/chat-dictionaries",
    group: "preferencesWorkflow",
    labelToken: "settings:chatDictionariesNav",
    icon: BookMarked,
    order: 13
  },
  {
    path: "/settings/rag",
    group: "preferencesWorkflow",
    labelToken: "settings:rag.title",
    icon: CombineIcon,
    order: 7.5
  },
  {
    path: "/settings/about",
    group: "dataAdmin",
    labelToken: "settings:about.title",
    icon: InfoIcon,
    order: 99
  },
  {
    path: MODERATION_REVIEW_PATH,
    group: "dataAdmin",
    labelToken: "option:moderationReview.nav",
    icon: ClipboardList,
    order: 4
  },
  {
    path: MODERATION_RULES_PATH,
    group: "dataAdmin",
    labelToken: "option:moderationRules.nav",
    icon: ShieldCheck,
    order: 5
  },
  {
    path: "/settings/family-guardrails",
    group: "dataAdmin",
    labelToken: "settings:familyGuardrailsWizardNav",
    icon: Users,
    order: 7
  },
  {
    path: "/settings/guardian",
    group: "dataAdmin",
    labelToken: "settings:guardianNav",
    icon: Eye,
    order: 6
  }
]
