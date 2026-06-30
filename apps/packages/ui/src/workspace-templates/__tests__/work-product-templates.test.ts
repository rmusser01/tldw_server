import { describe, expect, it } from "vitest"
import {
  DEFAULT_WORK_PRODUCT_TEMPLATE_ID,
  getWorkProductTemplate,
  WORK_PRODUCT_TEMPLATES
} from "../work-product-templates"

describe("work product templates", () => {
  it("defines all roadmap flagship templates", () => {
    expect(WORK_PRODUCT_TEMPLATES.map((template) => template.id)).toEqual([
      "executive_brief",
      "research_dossier",
      "competitive_market_memo",
      "technical_project_spec",
      "literature_matrix",
      "corpus_gap_finder",
      "evidence_bound_hypotheses",
      "research_proposal_pack"
    ])
  })

  it("uses executive brief as the first golden path", () => {
    const template = getWorkProductTemplate(DEFAULT_WORK_PRODUCT_TEMPLATE_ID)
    expect(template.id).toBe("executive_brief")
    expect(template.outputArtifactType).toBe("report")
    expect(template.category).toBe("general")
    expect(template.availability).toBe("actionable")
    expect(template.generationStrategy).toBe("executive_brief_markdown")
    expect(template.reviewChecklist.length).toBeGreaterThanOrEqual(3)
    expect(template.citationPolicy).toBe("required")
  })

  it("declares literature review templates as actionable typed work products", () => {
    const templateById = new Map(
      WORK_PRODUCT_TEMPLATES.map((template) => [template.id, template])
    )

    expect(templateById.get("literature_matrix")).toMatchObject({
      category: "literature_review",
      availability: "actionable",
      generationStrategy: "literature_matrix_json",
      outputArtifactType: "data_table",
      minSelectedSources: 2,
      minUsableSources: 2
    })
    expect(templateById.get("corpus_gap_finder")).toMatchObject({
      category: "literature_review",
      availability: "actionable",
      generationStrategy: "corpus_gap_json",
      minSelectedSources: 2,
      minUsableSources: 2
    })
    expect(templateById.get("evidence_bound_hypotheses")).toMatchObject({
      category: "literature_review",
      availability: "actionable",
      generationStrategy: "hypotheses_json",
      outputArtifactType: "report",
      minSelectedSources: 2,
      minUsableSources: 2
    })
    expect(templateById.get("research_proposal_pack")).toMatchObject({
      category: "literature_review",
      availability: "actionable",
      generationStrategy: "proposal_markdown",
      outputArtifactType: "report",
      minSelectedSources: 2,
      minUsableSources: 2
    })

    for (const id of [
      "literature_matrix",
      "corpus_gap_finder",
      "evidence_bound_hypotheses",
      "research_proposal_pack"
    ] as const) {
      expect(templateById.get(id)?.reviewChecklist.length).toBeGreaterThanOrEqual(3)
    }
  })

  it("keeps roadmap templates planned until their generation strategies are implemented", () => {
    for (const id of [
      "research_dossier",
      "competitive_market_memo",
      "technical_project_spec"
    ] as const) {
      const template = getWorkProductTemplate(id)
      expect(template.availability).toBe("planned")
      expect(template.generationStrategy).toBe("planned")
    }
  })
})
