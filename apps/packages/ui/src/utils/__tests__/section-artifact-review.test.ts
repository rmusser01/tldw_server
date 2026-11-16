import { describe, expect, it } from "vitest"
import { buildSectionReviewPatch, getSectionReviewTarget, getSectionReviewEntry, renderSectionReviewExport } from "../section-artifact-review"
import type { GeneratedArtifact } from "@/types/workspace"

const artifact = (): GeneratedArtifact => ({
  id: "local", workspaceArtifactId: "saved", workspaceArtifactVersion: 2, type: "summary", title: "Policy", status: "completed",
  createdAt: new Date(), content: "RAW MODEL DRAFT [E1.1.1]",
  data: { sectionAnalysis: { schemaVersion: 1, fingerprint: "source-v1", synthesis: "The model claims agreement [E1.1.1].", sections: [
    { title: "First contribution", node_id: "node", navigation_version: "v1", content: "A position [E1.1.1].", batches: [{ evidence: [{ id: "E1.1.1", quote: "Exact source" }] }] }
  ] } }
})

describe("source-bound artifact review", () => {
  it("retains the raw output and evidence while saving a separately identified correction", () => {
    const original = artifact(); const before = structuredClone(original)
    const target = getSectionReviewTarget(original, "overview")!
    const patch = buildSectionReviewPatch(original, target, { status: "needs_revision", correction: "The authors disagree [E1.1.1].", reason: "The cited position is not agreement." })
    expect(original).toEqual(before)
    expect(Object.keys(patch).sort()).toEqual(["reviewMetadata", "reviewStatus"])
    const reviewed = { ...original, ...patch }
    expect(getSectionReviewEntry(reviewed, target)).toMatchObject({ correction: "The authors disagree [E1.1.1].", stale: false })
    expect(renderSectionReviewExport(reviewed)).toContain(before.content)
    expect(renderSectionReviewExport(reviewed)).toContain("Reviewer correction")
  })
  it("marks a retained correction stale when its generated target changes", () => {
    const original = artifact(); const target = getSectionReviewTarget(original, "overview")!
    const reviewed = { ...original, ...buildSectionReviewPatch(original, target, { status: "reviewing", correction: "Correction [E1.1.1]", reason: "Reviewed source" }) }
    const updated = structuredClone(reviewed)
    ;(updated.data!.sectionAnalysis as { synthesis: string }).synthesis = "A regenerated overview."
    expect(getSectionReviewEntry(updated, getSectionReviewTarget(updated, "overview")!)).toMatchObject({ stale: true })
    expect(renderSectionReviewExport(updated)).toContain("Original target changed")
  })
  it("rejects correction citations that are absent from the selected target evidence", () => {
    const original = artifact()
    expect(() => buildSectionReviewPatch(original, getSectionReviewTarget(original, "0")!, { status: "accepted", correction: "Claim [E99.1.1]", reason: "Review" })).toThrow("not available")
  })
  it("refuses to replace an unsupported saved review format", () => {
    const original = { ...artifact(), reviewMetadata: { sectionReview: { schemaVersion: 2, entries: [] } } }
    expect(() => buildSectionReviewPatch(original, getSectionReviewTarget(original, "overview")!, { status: "reviewing", correction: "Correction", reason: "Review" })).toThrow("unsupported")
  })
  it("keeps prior review metadata and does not infer acceptance from a correction", () => {
    const original = { ...artifact(), reviewMetadata: { retained: "value" } }
    const patch = buildSectionReviewPatch(original, getSectionReviewTarget(original, "overview")!, { status: "needs_revision", correction: "Correction", reason: "Attribution" })
    expect(patch.reviewStatus).toBe("needs_revision")
    expect(patch.reviewMetadata).toMatchObject({ retained: "value" })
  })
})
