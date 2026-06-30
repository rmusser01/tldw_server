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
  | "connect"
  | "aiModels"
  | "experience"
  | "knowledgeWorkspace"
  | "safetyAdmin"
  | "dataManagement"
  | "about"

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
    group: "experience",
    labelToken: "settings:generalSettings.title",
    icon: OrbitIcon,
    order: 1
  },
  {
    path: "/settings/tldw",
    group: "connect",
    labelToken: "settings:tldw.serverNav",
    icon: ServerIcon,
    order: 1
  },
  {
    path: "/settings/provider-keys",
    group: "connect",
    labelToken: "settings:providerKeys.navTitle",
    icon: ServerIcon,
    order: 2
  },
  {
    path: "/settings/data",
    group: "dataManagement",
    labelToken: "settings:dataManagement.navTitle",
    icon: DatabaseIcon,
    order: 1
  },
  {
    path: "/settings/model",
    group: "aiModels",
    labelToken: "settings:manageModels.title",
    icon: BrainCircuitIcon,
    order: 1
  },
  {
    path: "/settings/mcp-hub",
    group: "aiModels",
    labelToken: "settings:mcpHubNav",
    icon: ServerIcon,
    order: 5
  },
  {
    path: "/settings/prompt",
    group: "knowledgeWorkspace",
    labelToken: "settings:managePrompts.title",
    icon: BookIcon,
    order: 2
  },
  {
    path: "/settings/evaluations",
    group: "safetyAdmin",
    labelToken: "settings:evaluationsSettings.title",
    icon: FlaskConical,
    order: 3,
    beta: true
  },
  {
    path: "/settings/chat",
    group: "experience",
    labelToken: "settings:chatSettingsNav",
    icon: MessageSquare,
    order: 2
  },
  {
    path: "/settings/ui",
    group: "experience",
    labelToken: "settings:uiCustomizationNav",
    icon: SlidersHorizontal,
    order: 3.5
  },
  {
    path: "/settings/splash",
    group: "experience",
    labelToken: "settings:splashSettingsNav",
    icon: Sparkles,
    order: 3.6
  },
  {
    path: "/settings/quick-ingest",
    group: "experience",
    labelToken: "settings:quickIngestSettingsNav",
    icon: ClipboardList,
    order: 4
  },
  {
    path: "/settings/speech",
    group: "aiModels",
    labelToken: "settings:speechSettingsNav",
    icon: MicIcon,
    order: 3
  },
  {
    path: "/settings/image-generation",
    group: "aiModels",
    labelToken: "settings:imageGenerationSettingsNav",
    icon: ImageIcon,
    order: 4
  },
  {
    path: "/settings/share",
    group: "knowledgeWorkspace",
    labelToken: "settings:manageShare.title",
    icon: ShareIcon,
    order: 7
  },
  {
    path: "/settings/health",
    group: "connect",
    labelToken: "settings:healthNav",
    icon: ActivityIcon,
    order: 3
  },
  {
    path: "/settings/prompt-studio",
    group: "knowledgeWorkspace",
    labelToken: "settings:promptStudio.nav",
    icon: Microscope,
    order: 3,
    beta: true
  },
  {
    path: "/settings/knowledge",
    group: "knowledgeWorkspace",
    labelToken: "settings:manageKnowledge.title",
    icon: BookText,
    order: 1
  },
  {
    path: "/settings/chatbooks",
    group: "knowledgeWorkspace",
    labelToken: "settings:chatbooksNav",
    icon: BookText,
    order: 4
  },
  {
    path: "/settings/characters",
    group: "knowledgeWorkspace",
    labelToken: "settings:charactersNav",
    icon: BookIcon,
    order: 5
  },
  {
    path: "/settings/world-books",
    group: "knowledgeWorkspace",
    labelToken: "settings:worldBooksNav",
    icon: BookOpen,
    order: 5.1
  },
  {
    path: "/settings/chat-dictionaries",
    group: "knowledgeWorkspace",
    labelToken: "settings:chatDictionariesNav",
    icon: BookMarked,
    order: 5.2
  },
  {
    path: "/settings/rag",
    group: "aiModels",
    labelToken: "settings:rag.title",
    icon: CombineIcon,
    order: 2
  },
  {
    path: "/settings/about",
    group: "about",
    labelToken: "settings:about.title",
    icon: InfoIcon,
    order: 1
  },
  {
    path: MODERATION_REVIEW_PATH,
    group: "safetyAdmin",
    labelToken: "option:moderationReview.nav",
    icon: ClipboardList,
    order: 4
  },
  {
    path: MODERATION_RULES_PATH,
    group: "safetyAdmin",
    labelToken: "option:moderationRules.nav",
    icon: ShieldCheck,
    order: 5
  },
  {
    path: "/settings/family-guardrails",
    group: "safetyAdmin",
    labelToken: "settings:familyGuardrailsWizardNav",
    icon: Users,
    order: 2
  },
  {
    path: "/settings/guardian",
    group: "safetyAdmin",
    labelToken: "settings:guardianNav",
    icon: Eye,
    order: 1
  }
]
