import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { describe, expect, it, vi } from "vitest"
import { ClaimVerificationPanel } from "../ClaimVerificationPanel"
import { claimTextHash } from "@/utils/claim-verification"
vi.mock("react-i18next", () => ({ useTranslation: () => ({ t: (_key: string, fallback: string) => fallback }) }))
const text = "The named author proposed this [E1.1.1]."
const report = { version: 1, text_sha256: claimTextHash(text), verdict: "needs_revision", provider: "openai", model: "test",
  coverage: { total: 1, evaluated: 1, supported: 0, uncovered: 0, complete: true },
  findings: [{ id: "C1", text, start: 0, end: text.length, evidence_ids: ["E1.1.1"], status: "unsupported", reason: "The source attributes the proposal to the chapter, not this person." }],
  evidence: { "E1.1.1": { text_sha256: "a".repeat(64), media_id: 7, page_spans: [{ page_number: 352 }] } }, limits: {}, context: {} }
describe("citation-support review", () => {
  it("shows reasons and exact evidence links separately from human review", () => {
    render(<MemoryRouter><ClaimVerificationPanel report={report} text={text} imported /></MemoryRouter>)
    expect(screen.getByText("Needs review")).toBeInTheDocument()
    fireEvent.click(screen.getByText(/Findings and cited evidence/))
    expect(screen.getByText(`unsupported: ${report.findings[0].reason}`)).toBeInTheDocument()
    expect(screen.getByRole("link", { name: "[E1.1.1]" })).toHaveAttribute("href", "/document-workspace?open=7&page=352")
    expect(screen.getByText(/Imported historical check/)).toBeInTheDocument()
    expect(screen.getByText(/do not establish real-world truth/)).toBeInTheDocument()
  })
  it.each([undefined, report])("shows checks pending while verification runs", value => {
    render(<MemoryRouter><ClaimVerificationPanel report={value} text={text} checking /></MemoryRouter>)
    expect(screen.getByRole("status")).toHaveTextContent("Checking citation support")
    expect(screen.queryByText("Needs review")).not.toBeInTheDocument()
  })
  it("labels legacy or edited text as not checked", () => {
    render(<MemoryRouter><ClaimVerificationPanel report={report} text="An edited draft" /></MemoryRouter>)
    expect(screen.getByText("Not checked")).toBeInTheDocument()
    expect(screen.getByText(/draft changed/)).toBeInTheDocument()
  })
})
