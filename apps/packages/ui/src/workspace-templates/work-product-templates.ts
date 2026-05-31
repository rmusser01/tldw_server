import type { ArtifactType } from "@/types/workspace"
import type {
  WorkProductGenerationStrategy,
  WorkProductTemplateAvailability,
  WorkProductTemplateCategory,
  WorkProductCitationPolicy,
  WorkProductTemplateId
} from "./types"

export interface WorkProductTemplate {
  id: WorkProductTemplateId
  label: string
  description: string
  category: WorkProductTemplateCategory
  availability: WorkProductTemplateAvailability
  generationStrategy: WorkProductGenerationStrategy
  outputArtifactType: ArtifactType
  minSelectedSources: number
  minUsableSources: number
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
    category: "general",
    availability: "actionable",
    generationStrategy: "executive_brief_markdown",
    outputArtifactType: "report",
    minSelectedSources: 1,
    minUsableSources: 1,
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
    category: "general",
    availability: "planned",
    generationStrategy: "planned",
    outputArtifactType: "report",
    minSelectedSources: 2,
    minUsableSources: 2,
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
    category: "general",
    availability: "planned",
    generationStrategy: "planned",
    outputArtifactType: "report",
    minSelectedSources: 2,
    minUsableSources: 2,
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
    category: "general",
    availability: "planned",
    generationStrategy: "planned",
    outputArtifactType: "report",
    minSelectedSources: 1,
    minUsableSources: 1,
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
  },
  {
    id: "literature_matrix",
    label: "Literature Matrix",
    description:
      "Structured comparison table across selected studies, methods, findings, limitations, and evidence status.",
    category: "literature_review",
    availability: "actionable",
    generationStrategy: "literature_matrix_json",
    outputArtifactType: "data_table",
    minSelectedSources: 2,
    minUsableSources: 2,
    sections: [
      "Source",
      "Methodology",
      "Sample Or Setting",
      "Primary Finding",
      "Limitations",
      "Contradictions"
    ],
    reviewChecklist: [
      "Every row maps to a usable source context included in generation.",
      "Unknown values are not filled with guesses.",
      "Contradictions name the involved source(s)."
    ],
    citationPolicy: "required"
  },
  {
    id: "corpus_gap_finder",
    label: "Corpus Gap Finder",
    description:
      "Gap table that separates source-stated gaps from inferred gaps, missing contexts, and follow-up questions.",
    category: "literature_review",
    availability: "actionable",
    generationStrategy: "corpus_gap_json",
    outputArtifactType: "data_table",
    minSelectedSources: 2,
    minUsableSources: 2,
    sections: [
      "Gap",
      "Gap Type",
      "Evidence Basis",
      "Sources",
      "Why It Matters",
      "Follow-up Question"
    ],
    reviewChecklist: [
      "Gaps distinguish source-stated gaps from inferred gaps.",
      "Each high-confidence gap has more than one evidence basis or a strong source.",
      "Missing population/context/method details are visible."
    ],
    citationPolicy: "required"
  },
  {
    id: "evidence_bound_hypotheses",
    label: "Evidence-Bound Hypotheses",
    description:
      "Testable hypotheses with supporting findings, source basis, predictions, methodology, and validity risks.",
    category: "literature_review",
    availability: "actionable",
    generationStrategy: "hypotheses_json",
    outputArtifactType: "report",
    minSelectedSources: 2,
    minUsableSources: 2,
    sections: [
      "Hypothesis",
      "Supporting Findings",
      "Prediction",
      "Suggested Methodology",
      "Threats To Validity",
      "Confidence"
    ],
    reviewChecklist: [
      "Hypotheses are testable.",
      "Predictions and methods are separated from existing findings.",
      "Confounders and falsification criteria are visible."
    ],
    citationPolicy: "required"
  },
  {
    id: "research_proposal_pack",
    label: "Research Proposal Pack",
    description:
      "Structured proposal draft with literature overview, gaps, hypothesis, methodology, risks, and source audit.",
    category: "literature_review",
    availability: "actionable",
    generationStrategy: "proposal_markdown",
    outputArtifactType: "report",
    minSelectedSources: 2,
    minUsableSources: 2,
    sections: [
      "Title",
      "Research Question",
      "Literature Overview",
      "Identified Gaps",
      "Proposed Hypothesis",
      "Methodology",
      "Risks And Limitations",
      "Source Audit"
    ],
    reviewChecklist: [
      "Literature claims are cited.",
      "Proposed work is not presented as established evidence.",
      "Risks and limitations are visible before export."
    ],
    citationPolicy: "required"
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
