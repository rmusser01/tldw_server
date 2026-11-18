import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import type { GeneratedArtifact } from "@/types/workspace"
import { getSectionReviewTarget } from "@/utils/section-artifact-review"
import { SectionArtifactReviewPanel } from "../SectionArtifactReviewPanel"

vi.mock("react-i18next", () => ({ useTranslation: () => ({ t: (_key: string, fallback: string) => fallback }) }))
vi.mock("@/utils/download-blob", () => ({ downloadBlob: vi.fn() }))
const artifact: GeneratedArtifact = { id: "a", workspaceArtifactId: "saved", type: "summary", title: "Draft", status: "completed", createdAt: new Date(), content: "Raw",
  data: { sectionAnalysis: { fingerprint: "source", synthesis: "Original overview [E1.1.1]", sections: [{ title: "Chapter", content: "Chapter text", batches: [{ evidence: [{ id: "E1.1.1" }] }] }] } } }

describe("section review controls", () => {
  it("keeps the correction draft and surfaces a failed save without changing the original", async () => {
    const save = vi.fn().mockRejectedValue(new Error("The saved version changed"))
    render(<SectionArtifactReviewPanel artifact={artifact} target={getSectionReviewTarget(artifact, "overview")!} onSave={save} onSaved={vi.fn()} />)
    fireEvent.change(screen.getByLabelText("Correction for this section"), { target: { value: "Corrected attribution [E1.1.1]" } })
    fireEvent.change(screen.getByLabelText("Reason or review note"), { target: { value: "Compared with the source" } })
    fireEvent.change(screen.getByLabelText("Review status for this output"), { target: { value: "needs_revision" } })
    fireEvent.click(screen.getByRole("button", { name: "Save review" }))
    await screen.findByRole("alert")
    expect(screen.getByLabelText("Correction for this section")).toHaveValue("Corrected attribution [E1.1.1]")
    expect(screen.getByRole("alert")).toHaveTextContent("The saved version changed")
    expect(artifact.content).toBe("Raw")
    expect(save.mock.calls[0][1]).toMatchObject({ reviewStatus: "needs_revision" })
  })
  it("retains an unsaved target draft while switching sections", () => {
    const props = { artifact, onSave: vi.fn(), onSaved: vi.fn() }
    const view = render(<SectionArtifactReviewPanel {...props} target={getSectionReviewTarget(artifact, "overview")!} />)
    fireEvent.change(screen.getByLabelText("Correction for this section"), { target: { value: "Unsaved correction" } })
    view.rerender(<SectionArtifactReviewPanel {...props} target={getSectionReviewTarget(artifact, "0")!} />)
    expect(screen.getByLabelText("Correction for this section")).toHaveValue("")
    view.rerender(<SectionArtifactReviewPanel {...props} target={getSectionReviewTarget(artifact, "overview")!} />)
    expect(screen.getByLabelText("Correction for this section")).toHaveValue("Unsaved correction")
  })
  it("publishes saved metadata only after persistence succeeds", async () => {
    const saved = vi.fn(); const persist = vi.fn(async (original, patch) => ({ ...original, ...patch }))
    render(<SectionArtifactReviewPanel artifact={artifact} target={getSectionReviewTarget(artifact, "overview")!} onSave={persist} onSaved={saved} />)
    fireEvent.change(screen.getByLabelText("Correction for this section"), { target: { value: "Correction" } })
    fireEvent.click(screen.getByRole("button", { name: "Save review" }))
    await waitFor(() => expect(saved).toHaveBeenCalledOnce())
    expect(saved.mock.calls[0][0].content).toBe("Raw")
    expect(saved.mock.calls[0][0].reviewMetadata.sectionReview.entries[0].correction).toBe("Correction")
  })
})
