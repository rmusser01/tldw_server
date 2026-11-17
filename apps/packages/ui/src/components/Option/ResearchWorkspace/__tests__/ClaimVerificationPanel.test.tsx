import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { describe, expect, it, vi } from "vitest"
import { ClaimVerificationPanel } from "../ClaimVerificationPanel"
import { claimTextHash, renderClaimVerification } from "@/utils/claim-verification"
import { portableFixture } from "@/utils/__tests__/fixtures/portable-workspace"
import { prepareLegacyWorkspaceImport } from "@/store/workspace-import-authority"
import { remapPortableWorkspace } from "@/utils/workspace-reference-remap"
vi.mock("react-i18next", () => ({ useTranslation: () => ({ t: (_key: string, fallback: string) => fallback }) }))
const text = "The named author proposed this [E1.1.1]."
const report = { version: 1, text_sha256: claimTextHash(text), verdict: "needs_revision", provider: "openai", model: "test",
  coverage: { total: 1, evaluated: 1, supported: 0, uncovered: 0, complete: true },
  findings: [{ id: "C1", text, start: 0, end: text.length, evidence_ids: ["E1.1.1"], status: "unsupported", reason: "The source attributes the proposal to the chapter, not this person." }],
  evidence: { "E1.1.1": { text_sha256: "a".repeat(64), media_id: 7, page_spans: [{ page_number: 352 }] } }, limits: {}, context: {} }
describe("citation-support review", () => {
  it.each([["text_or_unit_limit", false], ["report_size_limit", false], ["report_size_limit", true]] as const)("shows %s outside the collapsed list with context fallback=%s and retains the full unchecked text", (reason, contextFallback) => {
    const parts = reason === "text_or_unit_limit" ? Array.from({ length: 256 }, (_, index) => `Claim ${index} [1].`) : []
    const tail = "Unchecked remaining text [1]. 🌍"
    parts.push(tail)
    const draft = parts.join("\n\n")
    let cursor = 0
    const findings = parts.map((part, index) => {
      const start = cursor, end = start + Array.from(part).length
      cursor = end + 2
      return { id: `C${index + 1}`, text: part, start, end, evidence_ids: index === parts.length - 1 ? [] : ["1"],
        status: index === parts.length - 1 ? "not_checked" : "supported", reason: index === parts.length - 1 ? reason : "Cited evidence supports this claim." }
    })
    const receipt = { ...report, text_sha256: claimTextHash(draft), verdict: "unverified", findings,
      evidence: reason === "text_or_unit_limit" ? { "1": { text_sha256: "a".repeat(64) } } : {},
      context: contextFallback ? { unavailable_reason: "report_size_limit" } : {},
      coverage: { total: parts.length, evaluated: parts.length - 1, supported: parts.length - 1, uncovered: 1, complete: false } }
    const before = JSON.stringify(receipt)
    render(<MemoryRouter><ClaimVerificationPanel report={receipt} text={draft} /></MemoryRouter>)
    const explanation = reason === "text_or_unit_limit" ? "This claim exceeded the check's text or claim limit and was not checked."
      : "The citation report exceeded its size limit. The draft is retained, but its citation support has not been verified."
    expect(screen.getByText(explanation, { exact: true })).toBeVisible()
    expect(screen.queryByText("report_size_limit", { exact: true })).not.toBeInTheDocument()
    expect(screen.getByText("Not fully checked")).toBeVisible()
    expect(screen.queryByText("Supported by cited evidence")).not.toBeInTheDocument()
    expect(renderClaimVerification(receipt, draft)).toContain(tail)
    expect(renderClaimVerification(receipt, draft)).toContain(explanation)
    expect(JSON.stringify(receipt)).toBe(before)
  })
  it("shows reasons and exact evidence links separately from human review", () => {
    render(<MemoryRouter><ClaimVerificationPanel report={report} text={text} /></MemoryRouter>)
    expect(screen.getByText("Needs review")).toBeInTheDocument()
    fireEvent.click(screen.getByText(/Findings and cited evidence/))
    expect(screen.getByText(`unsupported: ${report.findings[0].reason}`)).toBeInTheDocument()
    expect(screen.getByRole("link", { name: "[E1.1.1]" })).toHaveAttribute("href", "/document-workspace?open=7&page=352")
    expect(screen.getByText(/do not establish real-world truth/)).toBeInTheDocument()
  })
  it("keeps numeric citation links and source page locations with comma annotations", () => {
    const draft = "The passage belongs to another contributor [2, PDF pp. 846–847, printed pp. 813–814]."
    const receipt = { ...report, text_sha256: claimTextHash(draft),
      findings: [{ ...report.findings[0], text: draft, end: draft.length, evidence_ids: ["2"] }],
      evidence: { "2": { text_sha256: "a".repeat(64), media_id: 1,
        page_spans: [{ page_number: 846, printed_page_label: "813" }, { page_number: 847, printed_page_label: "814" }] } } }
    render(<MemoryRouter><ClaimVerificationPanel report={receipt} text={draft} /></MemoryRouter>)
    fireEvent.click(screen.getByText(/Findings and cited evidence/))
    expect(screen.getByText("Needs review")).toBeVisible()
    expect(screen.getByRole("link", { name: "[2]" })).toHaveAttribute("href", "/document-workspace?open=1&page=846")
    expect(screen.getByText(draft)).toBeVisible()
    expect(screen.getAllByRole("link")).toHaveLength(1)
  })
  it.each(["snapshot", "complete pack"])("retains %s report evidence without navigating to historical server IDs", mode => {
    const before = JSON.stringify(report)
    const bundle = portableFixture()
    bundle.workspace.chatSession!.messages[0].message = text
    bundle.workspace.chatSession!.history = [{ role: "assistant", content: text }]
    bundle.workspace.chatSession!.messages[0].metadataExtra = { claim_verification: report }
    const identity = { id: "destination", name: "Imported", tag: "workspace:destination" }
    const result = mode === "snapshot" ? prepareLegacyWorkspaceImport(bundle, identity)
      : remapPortableWorkspace(bundle, { media: { "1": 2, "7": 2 }, notes: { "note-old": "note-new" } }, identity)
    const message = result.workspace.chatSession!.messages[0]
    render(<MemoryRouter><ClaimVerificationPanel report={message.metadataExtra?.claim_verification} text={message.message} imported={message.metadataExtra?.claim_verification_imported === true} /></MemoryRouter>)
    fireEvent.click(screen.getByText(/Findings and cited evidence/))
    expect(screen.getByText("[E1.1.1]")).toBeVisible()
    expect(screen.queryByRole("link", { name: "[E1.1.1]" })).not.toBeInTheDocument()
    expect(screen.getByText(/Use the restored source citations to open evidence/)).toBeVisible()
    expect(JSON.stringify(report)).toBe(before)
    expect(JSON.stringify(message.metadataExtra?.claim_verification)).toBe(before)
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
