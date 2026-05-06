import type { ArtifactType } from "@/types/workspace"
import type {
  WorkProductCitationPolicy,
  WorkProductTemplateId
} from "./types"

export interface WorkProductTemplate {
  id: WorkProductTemplateId
  label: string
  description: string
  outputArtifactType: ArtifactType
  minSelectedSources: number
  sections: string[]
  reviewChecklist: string[]
  citationPolicy: WorkProductCitationPolicy
}

export const DEFAULT_WORK_PRODUCT_TEMPLATE_ID: WorkProductTemplateId =
  "executive_brief"

export const WORK_PRODUCT_TEMPLATES: WorkProductTemplate[] = [
  {
    id: "executive_brief",
    label: "Executive Brief",
    description:
      "Decision-ready summary with context, evidence, risks, and next actions.",
    outputArtifactType: "report",
    minSelectedSources: 1,
    sections: [
      "Situation",
      "Key Findings",
      "Evidence",
      "Risks",
      "Recommended Actions"
    ],
    reviewChecklist: [
      "Every material claim has a source or explicit uncertainty.",
      "Recommendations are separated from evidence.",
      "Risks and open questions are visible before export."
    ],
    citationPolicy: "required"
  },
  {
    id: "research_dossier",
    label: "Research Dossier",
    description:
      "Structured evidence packet with source coverage, claims, gaps, and follow-up questions.",
    outputArtifactType: "report",
    minSelectedSources: 2,
    sections: [
      "Scope",
      "Source Inventory",
      "Evidence Map",
      "Findings",
      "Gaps",
      "Follow-up Questions"
    ],
    reviewChecklist: [
      "Source coverage is visible and balanced against the research scope.",
      "Findings distinguish direct evidence from synthesis.",
      "Evidence gaps and follow-up questions are explicit."
    ],
    citationPolicy: "required"
  },
  {
    id: "competitive_market_memo",
    label: "Competitive Market Memo",
    description:
      "Market-facing memo comparing competitors, positioning, risks, and strategic options.",
    outputArtifactType: "report",
    minSelectedSources: 2,
    sections: [
      "Market Context",
      "Competitor Snapshot",
      "Positioning",
      "Risks",
      "Strategic Options"
    ],
    reviewChecklist: [
      "Competitor claims are tied to cited source material.",
      "Market assumptions are separated from observed evidence.",
      "Strategic options include tradeoffs and uncertainty."
    ],
    citationPolicy: "required"
  },
  {
    id: "technical_project_spec",
    label: "Technical Project Spec",
    description:
      "Implementation-oriented spec with requirements, constraints, architecture notes, and acceptance criteria.",
    outputArtifactType: "report",
    minSelectedSources: 1,
    sections: [
      "Problem",
      "Goals",
      "Requirements",
      "Architecture Notes",
      "Acceptance Criteria",
      "Risks"
    ],
    reviewChecklist: [
      "Requirements are testable and separated from implementation notes.",
      "Dependencies, constraints, and unresolved decisions are visible.",
      "Acceptance criteria reflect the cited source requirements."
    ],
    citationPolicy: "recommended"
  }
]

export const getWorkProductTemplate = (
  templateId: WorkProductTemplateId
): WorkProductTemplate => {
  return (
    WORK_PRODUCT_TEMPLATES.find((template) => template.id === templateId) ||
    WORK_PRODUCT_TEMPLATES[0]
  )
}
