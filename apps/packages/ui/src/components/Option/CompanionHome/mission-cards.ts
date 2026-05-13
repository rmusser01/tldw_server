import type { LucideIcon } from "lucide-react"
import { Upload, BookOpen, MessageSquare, Shield, ShieldCheck, GraduationCap, Search } from "lucide-react"
import type { UserPersona } from "@/types/connection"
import type { MilestoneId } from "@/store/milestones"

export type MissionCardCategory = "getting-started" | "create" | "analyze"

export type MissionCard = {
  id: string
  title: string
  description: string
  icon: LucideIcon
  href: string
  persona: UserPersona[] | "all"
  prerequisiteMilestones: MilestoneId[]
  linkedMilestone?: MilestoneId  // the milestone this card represents completing
  category: MissionCardCategory
  priority: number
}

export const MISSION_CARDS: MissionCard[] = [
  // === FAMILY PERSONA ===
  {
    id: "family-guardrails",
    title: "Set up family profiles",
    description: "Create guardian and dependent profiles in the Family Guardrails wizard.",
    icon: Shield,
    href: "/settings/family-guardrails",
    persona: ["family"],
    prerequisiteMilestones: ["first_connection"],
    linkedMilestone: "family_profiles_created",
    category: "getting-started",
    priority: 1
  },
  {
    id: "family-content-rules",
    title: "Review content rules",
    description: "Configure content safety policies and blocklists.",
    icon: ShieldCheck,
    href: "/moderation/rules",
    persona: ["family"],
    prerequisiteMilestones: ["first_connection"],
    linkedMilestone: "content_rules_reviewed",
    category: "getting-started",
    priority: 2
  },
  {
    id: "family-test-rules",
    title: "Test your rules",
    description: "Try your content rules in the Test Sandbox to see them in action.",
    icon: MessageSquare,
    href: "/moderation/rules?tab=test",
    persona: ["family"],
    prerequisiteMilestones: ["first_connection"],
    linkedMilestone: "content_rules_tested",
    category: "getting-started",
    priority: 3
  },
  {
    id: "family-start-chatting",
    title: "Start chatting",
    description: "Try a conversation with your safety rules active.",
    icon: MessageSquare,
    href: "/chat",
    persona: ["family"],
    prerequisiteMilestones: ["first_connection"],
    linkedMilestone: "first_chat",
    category: "getting-started",
    priority: 4
  },

  // === RESEARCHER PERSONA ===
  {
    id: "researcher-ingest",
    title: "Import your first content",
    description: "Upload a video, article, or document to build your knowledge base.",
    icon: Upload,
    href: "/media",
    persona: ["researcher"],
    prerequisiteMilestones: ["first_connection"],
    linkedMilestone: "first_ingest",
    category: "getting-started",
    priority: 1
  },
  {
    id: "researcher-browse",
    title: "Browse your library",
    description: "Search, filter, and review your ingested content.",
    icon: BookOpen,
    href: "/media",
    persona: ["researcher"],
    prerequisiteMilestones: ["first_ingest"],
    category: "getting-started",
    priority: 2
  },
  {
    id: "researcher-ask",
    title: "Ask questions about your content",
    description: "Use Knowledge QA to chat with your documents.",
    icon: Search,
    href: "/chat",
    persona: ["researcher"],
    prerequisiteMilestones: ["first_ingest"],
    linkedMilestone: "first_chat",
    category: "getting-started",
    priority: 3
  },
  {
    id: "researcher-quiz",
    title: "Test your understanding",
    description: "Generate a quiz from your content to reinforce learning.",
    icon: GraduationCap,
    href: "/quiz",
    persona: ["researcher"],
    prerequisiteMilestones: ["first_ingest"],
    linkedMilestone: "first_quiz_taken",
    category: "create",
    priority: 4
  },

  // === EXPLORER PERSONA (and fallback for null persona) ===
  {
    id: "explorer-ingest",
    title: "Add your first source",
    description: "Upload a video, article, or document to get started.",
    icon: Upload,
    href: "/media",
    persona: ["explorer", null],
    prerequisiteMilestones: ["first_connection"],
    linkedMilestone: "first_ingest",
    category: "getting-started",
    priority: 1
  },
  {
    id: "explorer-chat",
    title: "Start a conversation",
    description: "Chat with the AI about anything, or ask about your content.",
    icon: MessageSquare,
    href: "/chat",
    persona: ["explorer", null],
    prerequisiteMilestones: ["first_connection"],
    linkedMilestone: "first_chat",
    category: "getting-started",
    priority: 2
  }
]
