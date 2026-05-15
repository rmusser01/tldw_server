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

export type NavGroupKey = "server" | "knowledge" | "workspace" | "about"

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
    group: "server",
    labelToken: "settings:generalSettings.title",
    icon: OrbitIcon,
    order: 2
  },
  {
    path: "/settings/tldw",
    group: "server",
    labelToken: "settings:tldw.serverNav",
    icon: ServerIcon,
    order: 1
  },
  {
    path: "/settings/provider-keys",
    group: "server",
    labelToken: "settings:providerKeys.navTitle",
    icon: ServerIcon,
    order: 2
  },
  {
    path: "/settings/model",
    group: "server",
    labelToken: "settings:manageModels.title",
    icon: BrainCircuitIcon,
    order: 6
  },
  {
    path: "/settings/mcp-hub",
    group: "server",
    labelToken: "settings:mcpHubNav",
    icon: ServerIcon,
    order: 7
  },
  {
    path: "/settings/prompt",
    group: "workspace",
    labelToken: "settings:managePrompts.title",
    icon: BookIcon,
    order: 6
  },
  {
    path: "/settings/evaluations",
    group: "server",
    labelToken: "settings:evaluationsSettings.title",
    icon: FlaskConical,
    order: 9,
    beta: true
  },
  {
    path: "/settings/chat",
    group: "server",
    labelToken: "settings:chatSettingsNav",
    icon: MessageSquare,
    order: 3
  },
  {
    path: "/settings/ui",
    group: "server",
    labelToken: "settings:uiCustomizationNav",
    icon: SlidersHorizontal,
    order: 3.5
  },
  {
    path: "/settings/splash",
    group: "server",
    labelToken: "settings:splashSettingsNav",
    icon: Sparkles,
    order: 3.6
  },
  {
    path: "/settings/quick-ingest",
    group: "server",
    labelToken: "settings:quickIngestSettingsNav",
    icon: ClipboardList,
    order: 4
  },
  {
    path: "/settings/speech",
    group: "server",
    labelToken: "settings:speechSettingsNav",
    icon: MicIcon,
    order: 5
  },
  {
    path: "/settings/image-generation",
    group: "server",
    labelToken: "settings:imageGenerationSettingsNav",
    icon: ImageIcon,
    order: 7
  },
  {
    path: "/settings/share",
    group: "workspace",
    labelToken: "settings:manageShare.title",
    icon: ShareIcon,
    order: 7
  },
  {
    path: "/settings/health",
    group: "server",
    labelToken: "settings:healthNav",
    icon: ActivityIcon,
    order: 11
  },
  {
    path: "/settings/prompt-studio",
    group: "server",
    labelToken: "settings:promptStudio.nav",
    icon: Microscope,
    order: 10,
    beta: true
  },
  {
    path: "/settings/knowledge",
    group: "knowledge",
    labelToken: "settings:manageKnowledge.title",
    icon: BookText,
    order: 1
  },
  {
    path: "/settings/chatbooks",
    group: "knowledge",
    labelToken: "settings:chatbooksNav",
    icon: BookText,
    order: 4
  },
  {
    path: "/settings/characters",
    group: "knowledge",
    labelToken: "settings:charactersNav",
    icon: BookIcon,
    order: 5
  },
  {
    path: "/settings/world-books",
    group: "knowledge",
    labelToken: "settings:worldBooksNav",
    icon: BookOpen,
    order: 2
  },
  {
    path: "/settings/chat-dictionaries",
    group: "knowledge",
    labelToken: "settings:chatDictionariesNav",
    icon: BookMarked,
    order: 3
  },
  {
    path: "/settings/rag",
    group: "server",
    labelToken: "settings:rag.title",
    icon: CombineIcon,
    order: 4
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
    group: "server",
    labelToken: "option:moderationReview.nav",
    icon: ClipboardList,
    order: 10
  },
  {
    path: MODERATION_RULES_PATH,
    group: "server",
    labelToken: "option:moderationRules.nav",
    icon: ShieldCheck,
    order: 10.1
  },
  {
    path: "/settings/family-guardrails",
    group: "server",
    labelToken: "settings:familyGuardrailsWizardNav",
    icon: Users,
    order: 8
  },
  {
    path: "/settings/guardian",
    group: "server",
    labelToken: "settings:guardianNav",
    icon: Eye,
    order: 9
  }
]
