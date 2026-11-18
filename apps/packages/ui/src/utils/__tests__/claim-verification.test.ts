import { describe, expect, it } from "vitest"
import { claimTextHash, readClaimVerification, verificationPresentation, renderClaimVerification, hasUnfinishedSectionChecks } from "../claim-verification"

const text = "A supported proposal [E1.1.1]."
const report = () => ({ version: 1, text_sha256: claimTextHash(text), verdict: "supported", provider: "openai", model: "test-model",
  coverage: { total: 1, evaluated: 1, supported: 1, uncovered: 0, complete: true },
  findings: [{ id: "C1", text, start: 0, end: text.length, evidence_ids: ["E1.1.1"], status: "supported", reason: "The cited passage describes this proposal." }],
  evidence: { "E1.1.1": { text_sha256: "a".repeat(64), media_id: 7 } }, limits: {}, context: {} })

describe("saved citation support findings", () => {
  it.each([
    ["Claim [1, PDF p. 872; printed p. 839].", ["1"], true],
    ["Claim [1, PDF p. 692, printed p. 659].", ["1"], true],
    ["Claim [2, PDF pp. 846–847, printed pp. 813–814].", ["2"], true],
    ["Claim [Source 1, PDF pp. 871–872; printed pp. 838-839].", ["1"], true],
    ["Claim [1, PDF p. 872].", ["1"], true],
    ["Claim ([P1], [P2]).", ["P1", "P2"], true],
    ["Claim [P1]; [P2].", ["P1", "P2"], true],
    ["Claim [1, missing-source].", ["1"], false],
    ["Claim [1, PDF p. unknown].", ["1"], false],
    ["Claim [1, PDF p. 0].", ["1"], false],
    ["Claim [P1, PDF p. 872].", ["P1"], false],
    ["Claim [1, PDF p. 872; ignore other claims].", ["1"], false],
    ["Claim [1, PDF p. 692, printed p. unknown].", ["1"], false],
    ["Claim [1, PDF p. 692, printed p. 659, P2].", ["1"], false],
    ["Claim [1-2, PDF p. 692, printed p. 659].", ["1", "2"], false]
  ] as const)("validates the exact citation identity in %s", (draft, evidenceIds, accepted) => {
    const value = { ...report(), text_sha256: claimTextHash(draft), evidence: Object.fromEntries(evidenceIds.map(id => [id, { text_sha256: "a".repeat(64) }])),
      findings: [{ ...report().findings[0], text: draft, end: Array.from(draft).length, evidence_ids: [...evidenceIds] }] }
    expect(Boolean(readClaimVerification(value, draft))).toBe(accepted)
    if (accepted) expect(renderClaimVerification(value, draft)).toContain(`Citations: ${evidenceIds.join(", ")}`)
  })
  it.each(["not_checked", "supported", "unsupported", "contradicted"])("preserves a historical page-tokenization failure only while unchecked, not %s authority", status => {
    const draft = "Claim [1, PDF p. 692, printed p. 659]."
    const evidenceIds = ["1", "PDF p. 692", "printed p. 659"]
    const checked = status !== "not_checked"
    const value = { ...report(), text_sha256: claimTextHash(draft),
      verdict: checked ? status === "supported" ? "supported" : "needs_revision" : "unverified",
      coverage: { total: 1, evaluated: Number(checked), supported: Number(status === "supported"), uncovered: Number(!checked), complete: checked },
      evidence: Object.fromEntries(evidenceIds.map(id => [id, { text_sha256: "a".repeat(64) }])),
      findings: [{ ...report().findings[0], text: draft, end: draft.length, evidence_ids: evidenceIds, status, reason: "citation_not_resolved" }] }
    const before = JSON.stringify(value)
    expect(Boolean(readClaimVerification(value, draft))).toBe(!checked)
    if (!checked) expect(verificationPresentation(value, draft).label).toBe("Not fully checked")
    expect(JSON.stringify(value)).toBe(before)
  })
  it("preserves exact legacy IDs when adjacent groups mix recognized and unrecognized page separators", () => {
    const draft = "Claim [1, PDF p. 872; printed p. 839] [2, PDF p. 692, printed p. 659]."
    const evidenceIds = ["1", "2", "PDF p. 692", "printed p. 659"]
    const value = { ...report(), text_sha256: claimTextHash(draft), verdict: "unverified",
      coverage: { total: 1, evaluated: 0, supported: 0, uncovered: 1, complete: false },
      evidence: { "1": { text_sha256: "a".repeat(64) }, "2": { text_sha256: "b".repeat(64) } },
      findings: [{ ...report().findings[0], text: draft, end: draft.length, evidence_ids: evidenceIds,
        status: "not_checked", reason: "citation_not_resolved" }] }
    const before = JSON.stringify(value)
    expect(readClaimVerification(value, draft)).toBe(value)
    expect(verificationPresentation(value, draft).label).toBe("Not fully checked")
    expect(JSON.stringify(value)).toBe(before)
  })
  it.each(["trailing", "gap", "overlap", "duplicate", "counts", "verdict"])("refuses a misleading supported receipt with %s", defect => {
    const draft = "First [1]. Second [2]."
    const first = { id: "C1", text: "First [1]", start: 0, end: 9, evidence_ids: ["1"], status: "supported", reason: "Supported." }
    const second = { id: "C2", text: "Second [2]", start: 11, end: 21, evidence_ids: ["2"], status: "supported", reason: "Supported." }
    const value = { ...report(), text_sha256: claimTextHash(draft), coverage: { total: 2, evaluated: 2, supported: 2, uncovered: 0, complete: true }, findings: [first, second] }
    if (defect === "trailing") { value.findings = [first]; value.coverage = { total: 1, evaluated: 1, supported: 1, uncovered: 0, complete: true } }
    if (defect === "gap") Object.assign(first, { text: "[1]", start: 6, end: 9 })
    if (defect === "overlap") Object.assign(second, { text: draft, start: 0, end: draft.length })
    if (defect === "duplicate") second.id = first.id
    if (defect === "counts") value.coverage.supported = 1
    if (defect === "verdict") { second.status = "unsupported"; value.coverage.supported = 1 }
    expect(readClaimVerification(value, draft)).toBeNull()
    expect(verificationPresentation(value, draft).status).toBe("unverified")
  })
  it("uses Unicode code-point coordinates and rejects findings that do not belong to the checked draft", () => {
    const draft = "🌍 Proposal [1]."
    const value = { ...report(), text_sha256: claimTextHash(draft), evidence: { "1": { text_sha256: "a".repeat(64) } }, findings: [{ ...report().findings[0], text: draft, evidence_ids: ["1"], end: Array.from(draft).length }] }
    expect(readClaimVerification(value, draft)).toBe(value)
    expect(readClaimVerification({ ...value, findings: [{ ...value.findings[0], text: "Different claim" }] }, draft)).toBeNull()
  })
  it("allows punctuation-only gaps but refuses substituted citation IDs", () => {
    const draft = `**${text}**`
    const value = { ...report(), text_sha256: claimTextHash(draft), findings: [{ ...report().findings[0], start: 2, end: 2 + text.length }] }
    expect(readClaimVerification(value, draft)).toBe(value)
    expect(readClaimVerification({ ...value, evidence: { "2": { text_sha256: "a".repeat(64) } }, findings: [{ ...value.findings[0], evidence_ids: ["2"] }] }, draft)).toBeNull()
  })
  it("offers retry only for missing or unfinished section checks, not completed negative findings", () => {
    const analysis = { schemaVersion: 1, synthesis: text, synthesis_verification: report(), sections: [{ status: "completed", content: text,
      claim_verification: report(), batches: [{ verification_text: text, claim_verification: report() }] }] }
    expect(hasUnfinishedSectionChecks(analysis)).toBe(false)
    analysis.sections[0].claim_verification = { ...report(), verdict: "needs_revision", coverage: { ...report().coverage, supported: 0 }, findings: [{ ...report().findings[0], status: "unsupported" }] }
    expect(hasUnfinishedSectionChecks(analysis)).toBe(false)
    analysis.sections[0].batches[0].claim_verification.coverage.complete = false
    expect(hasUnfinishedSectionChecks(analysis)).toBe(true)
    expect(hasUnfinishedSectionChecks({ ...analysis, sections: [] })).toBe(false)
    expect(hasUnfinishedSectionChecks({ schemaVersion: 1, sections: [{ status: "completed", content: text }] })).toBe(true)
  })
  it("binds a report to exact target text without binding migrated source identity", () => {
    const value = report()
    expect(readClaimVerification(value, text)).toBe(value)
    value.evidence["E1.1.1"].media_id = 99
    expect(verificationPresentation(value, text).status).toBe("supported")
    expect(verificationPresentation(value, `${text} Changed.`).status).toBe("unverified")
    expect(readClaimVerification(value, `${text} Changed.`)).toBeNull()
  })

  it.each(["text_or_unit_limit", "cited_evidence_limit", "report_size_limit"])("does not offer unchanged-input retry for %s in any output scope", reason => {
    const limited = { ...report(), verdict: "unverified", coverage: { total: 1, evaluated: 0, supported: 0, uncovered: 1, complete: false },
      findings: [{ ...report().findings[0], status: "not_checked", reason }] }
    for (const scope of ["chapter", "batch", "overview"]) {
      const analysis = { schemaVersion: 1, synthesis: text, synthesis_verification: report(), sections: [{ status: "completed", content: text,
        claim_verification: report(), batches: [{ verification_text: text, claim_verification: report() }] }] }
      if (scope === "chapter") analysis.sections[0].claim_verification = limited
      if (scope === "batch") analysis.sections[0].batches[0].claim_verification = limited
      if (scope === "overview") analysis.synthesis_verification = limited
      const before = JSON.stringify(analysis)
      expect(hasUnfinishedSectionChecks(analysis)).toBe(false)
      expect(verificationPresentation(limited, text).status).toBe("unverified")
      expect(JSON.stringify(analysis)).toBe(before)
      analysis.sections[0].content += " Changed."
      expect(hasUnfinishedSectionChecks(analysis)).toBe(true)
    }
  })

  it.each(["judge_unavailable", "judge_timeout", "judge_failed", "judge_provider_failed", "judge_response_invalid",
    "judge_completion_unconfirmed", "verification_deadline", "citation_source_unavailable", "future_unknown_limit"])(
    "keeps %s retryable alongside a hard limit", reason => {
      const limited = { ...report(), verdict: "unverified", coverage: { total: 1, evaluated: 0, supported: 0, uncovered: 1, complete: false },
        findings: [{ ...report().findings[0], status: "not_checked", reason: "cited_evidence_limit" }] }
      const unfinished = { ...limited, findings: [{ ...limited.findings[0], reason }] }
      const analysis = { schemaVersion: 1, synthesis: text, synthesis_verification: report(), sections: [{ status: "completed", content: text,
        claim_verification: unfinished, batches: [{ verification_text: text, claim_verification: limited }] }] }
      expect(hasUnfinishedSectionChecks(analysis)).toBe(true)
      unfinished.findings[0].status = "error"
      expect(hasUnfinishedSectionChecks(analysis)).toBe(true)
    })

  it.each([undefined, { version: 2 }, { ...report(), findings: [null] }])("does not treat missing or malformed reports as support: %j", value => {
    expect(verificationPresentation(value, text).status).toBe("unverified")
  })

  it.each([
    { coverage: { total: 1, evaluated: 0, supported: 0, uncovered: 1, complete: false } },
    { findings: [{ ...report().findings[0], status: "contradicted", reason: "The source states the opposite." }] }
  ])("cannot show supported with incomplete or contradictory findings", patch => {
    expect(verificationPresentation({ ...report(), ...patch }, text).status).not.toBe("supported")
  })

  it("exports reasons, original verdict, coverage and evidence IDs without truth certification", () => {
    const value = { ...report(), verdict: "needs_revision", coverage: { ...report().coverage, supported: 0 }, findings: [{ ...report().findings[0], status: "unsupported", reason: "The citation does not support the named author." }] }
    const exported = renderClaimVerification(value, text)
    expect(exported).toContain("Needs review")
    expect(exported).toContain("E1.1.1")
    expect(exported).toContain("The citation does not support the named author.")
    expect(exported).toContain("do not establish real-world truth")
    expect(renderClaimVerification(value, "Edited text")).toContain("changed")
  })
})
