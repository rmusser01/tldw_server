export type WorkProductTemplateId =
  | "executive_brief"
  | "research_dossier"
  | "competitive_market_memo"
  | "technical_project_spec"
  | "literature_matrix"
  | "corpus_gap_finder"
  | "evidence_bound_hypotheses"
  | "research_proposal_pack"

export type WorkProductCitationPolicy = "required" | "recommended"

export type WorkProductTemplateCategory = "general" | "literature_review"

export type WorkProductTemplateAvailability =
  | "actionable"
  | "planned"
  | "disabled"

export type WorkProductGenerationStrategy =
  | "executive_brief_markdown"
  | "literature_matrix_json"
  | "corpus_gap_json"
  | "hypotheses_json"
  | "proposal_markdown"
  | "planned"
