import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { describe, expect, it, vi } from "vitest"
import type { GeneratedArtifact } from "@/types/workspace"
import { getSectionAnalysis, SectionAnalysisArtifactViewer } from "../SectionAnalysisArtifactViewer"
import { claimTextHash } from "@/utils/claim-verification"

vi.mock("react-i18next", () => ({ useTranslation: () => ({ t: (_key: string, fallback: string) => fallback }) }))

const checkpoint = (status: GeneratedArtifact["status"], version: number): GeneratedArtifact => ({
  id: "reading", workspaceArtifactId: "canonical-reading", workspaceArtifactVersion: version,
  title: "Saved chapters", type: "summary", status, createdAt: new Date("2026-11-16T00:00:00Z"),
  content: `Immutable draft ${version}`,
  data: { sectionAnalysis: { schemaVersion: 1, fingerprint: "fixed-source", synthesis: `Overview ${version}`,
    sections: [{ title: "Chapter", status: "completed", content: `Chapter checkpoint ${version}`,
      batches: [{ status: "completed", evidence: [{ id: `E${version}.1.1`, claim: "Source claim", quote: "Exact source." }] }] }] } }
})

describe("section reader checkpoint updates", () => {
  it("separates overview, chapter and extracted-evidence findings from the saved human review", () => {
    const artifact = checkpoint("completed", 1)
    artifact.reviewStatus = "accepted"
    const analysis = getSectionAnalysis(artifact)!
    const report = (text: string, reason: string, status: string) => ({ version: 1, text_sha256: claimTextHash(text), verdict: status === "supported" ? "supported" : "needs_revision",
      provider: "openai", model: "saved-model", coverage: { total: 1, evaluated: 1, supported: status === "supported" ? 1 : 0, uncovered: 0, complete: true },
      findings: [{ id: "C1", text, start: 0, end: text.length, evidence_ids: ["E1.1.1"], status, reason }], evidence: { "E1.1.1": { text_sha256: "a".repeat(64) } }, context: {}, limits: {} })
    analysis.synthesis = "Overview 1 [E1.1.1]"
    analysis.sections[0].content = "Chapter checkpoint 1 [E1.1.1]"
    analysis.synthesis_verification = report("Overview 1 [E1.1.1]", "Overview attribution is unsupported.", "unsupported")
    analysis.sections[0].claim_verification = report("Chapter checkpoint 1 [E1.1.1]", "Chapter evidence supports the text.", "supported")
    analysis.sections[0].batches = [{ verification_text: "Extracted evidence claim [E1.1.1]", claim_verification: report("Extracted evidence claim [E1.1.1]", "Earlier extraction needs correction.", "unsupported"), evidence: [] }]
    const original = JSON.stringify(artifact)
    render(<MemoryRouter><SectionAnalysisArtifactViewer artifact={artifact} analysis={analysis} sources={[]} /></MemoryRouter>)
    expect(screen.getByText("Needs review")).toBeInTheDocument()
    fireEvent.change(screen.getByRole("combobox", { name: "Chapter or section" }), { target: { value: "0" } })
    expect(screen.getByText("Supported by cited evidence")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: /Show evidence/ }))
    expect(screen.getByText("Needs review")).toBeInTheDocument()
    expect(screen.getByText(/Earlier extraction needs correction/)).toBeInTheDocument()
    expect(JSON.stringify(artifact)).toBe(original)
  })
  it("keeps generation read-only even if its caller supplies a save callback", () => {
    const artifact = checkpoint("generating", 1)
    const save = vi.fn()
    render(<MemoryRouter><SectionAnalysisArtifactViewer artifact={artifact} analysis={getSectionAnalysis(artifact)!} sources={[]} onSaveReview={save} /></MemoryRouter>)
    expect(screen.getByRole("button", { name: "Save review" })).toBeDisabled()
    expect(screen.getByLabelText("Correction for this section")).toBeDisabled()
    fireEvent.click(screen.getByRole("button", { name: "Save review" }))
    expect(save).not.toHaveBeenCalled()
    expect(screen.getByText(/saved overview may change while generation continues/i)).toBeInTheDocument()
  })

  it.each(["completed", "failed"] as const)("reviews the latest %s checkpoint without resetting the selected chapter", async status => {
    const initial = checkpoint("generating", 1)
    const latest = checkpoint(status, 9)
    const before = JSON.stringify(latest)
    const save = vi.fn(async (original, patch) => ({ ...original, ...patch, workspaceArtifactVersion: 10 }))
    const view = render(<MemoryRouter><SectionAnalysisArtifactViewer artifact={initial} analysis={getSectionAnalysis(initial)!} sources={[]} onSaveReview={save} /></MemoryRouter>)
    fireEvent.change(screen.getByRole("combobox", { name: "Chapter or section" }), { target: { value: "0" } })
    fireEvent.click(screen.getByRole("button", { name: /Show evidence/ }))
    view.rerender(<MemoryRouter><SectionAnalysisArtifactViewer artifact={latest} analysis={getSectionAnalysis(latest)!} sources={[]} onSaveReview={save} /></MemoryRouter>)
    expect(screen.getByRole("combobox", { name: "Chapter or section" })).toHaveValue("0")
    expect(screen.getByText("Exact source.")).toBeInTheDocument()
    expect(screen.getByText("Chapter checkpoint 9")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Save review" })).toBeEnabled()
    fireEvent.change(screen.getByLabelText("Correction for this section"), { target: { value: "Reviewed latest evidence [E9.1.1]." } })
    fireEvent.click(screen.getByRole("button", { name: "Save review" }))
    await waitFor(() => expect(save).toHaveBeenCalledOnce())
    expect(save.mock.calls[0][0]).toMatchObject({ workspaceArtifactVersion: 9, content: "Immutable draft 9" })
    expect(screen.getByText("Review saved. The original draft is retained.")).toBeInTheDocument()
    expect(JSON.stringify(latest)).toBe(before)
  })
})
