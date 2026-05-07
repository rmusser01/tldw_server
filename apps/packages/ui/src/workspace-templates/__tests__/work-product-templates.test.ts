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
      "technical_project_spec"
    ])
  })

  it("uses executive brief as the first golden path", () => {
    const template = getWorkProductTemplate(DEFAULT_WORK_PRODUCT_TEMPLATE_ID)
    expect(template.id).toBe("executive_brief")
    expect(template.outputArtifactType).toBe("report")
    expect(template.reviewChecklist.length).toBeGreaterThanOrEqual(3)
    expect(template.citationPolicy).toBe("required")
  })
})
