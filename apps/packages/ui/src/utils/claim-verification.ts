import { sha256 } from "@noble/hashes/sha2.js"
import { bytesToHex, utf8ToBytes } from "@noble/hashes/utils.js"

export const CLAIM_VERIFICATION_KEY = "claim_verification"
export const CLAIM_VERIFICATION_IMPORTED_KEY = "claim_verification_imported"
export type ClaimStatus = "supported" | "unsupported" | "contradicted" | "uncited" | "not_checked" | "error"
export type ClaimVerification = {
  version: 1; text_sha256: string; verdict: "supported" | "needs_revision" | "unverified"
  provider: string | null; model: string | null
  coverage: { total: number; evaluated: number; supported: number; uncovered: number; complete: boolean }
  findings: Array<{ id: string; text: string; start: number; end: number; evidence_ids: string[]; status: ClaimStatus; reason: string }>
  evidence: Record<string, Record<string, unknown>>
  limits: Record<string, unknown>; context: Record<string, unknown>
}
const record = (value: unknown): value is Record<string, unknown> => value !== null && typeof value === "object" && !Array.isArray(value)
const statuses = new Set(["supported", "unsupported", "contradicted", "uncited", "not_checked", "error"])
// Match the server's bounded numeric location annotation; unknown suffixes remain citation IDs.
const PDF_LOCATION_CITATION = /^(?:Source )?([1-9][0-9]*)[ \t]*,[ \t]*PDF pp?\.[ \t]*[1-9][0-9]*(?:[ \t]*[-–][ \t]*[1-9][0-9]*)?(?:[ \t]*;[ \t]*printed pp?\.[ \t]*[1-9][0-9]*(?:[ \t]*[-–][ \t]*[1-9][0-9]*)?)?$/
const count = (value: unknown): value is number => typeof value === "number" && Number.isSafeInteger(value) && value >= 0
export const claimTextHash = (text: string) => bytesToHex(sha256(utf8ToBytes(text)))

/** A saved assessment is bound to its exact draft, never authority to access its sources. */
export function readClaimVerification(value: unknown, text?: string): ClaimVerification | null {
  if (!record(value) || value.version !== 1 || typeof value.text_sha256 !== "string" || !/^[a-f0-9]{64}$/.test(value.text_sha256) ||
    (text !== undefined && value.text_sha256 !== claimTextHash(text)) ||
    !["supported", "needs_revision", "unverified"].includes(String(value.verdict)) ||
    (value.provider !== null && typeof value.provider !== "string") || (value.model !== null && typeof value.model !== "string") ||
    !record(value.coverage) || !["total", "evaluated", "supported", "uncovered"].every(key => count(value.coverage[key])) ||
    typeof value.coverage.complete !== "boolean" || !record(value.evidence) || !record(value.limits) || !record(value.context) ||
    Object.values(value.evidence).some(item => !record(item)) ||
    !Array.isArray(value.findings) || value.findings.some(item => !record(item) ||
      typeof item.id !== "string" || typeof item.text !== "string" || !count(item.start) || !count(item.end) || item.end < item.start ||
      !Array.isArray(item.evidence_ids) || item.evidence_ids.some(id => typeof id !== "string") ||
      !statuses.has(String(item.status)) || typeof item.reason !== "string")) return null
  if (value.findings.some(item => ["supported", "unsupported", "contradicted"].includes(item.status) &&
    (!item.evidence_ids.length || item.evidence_ids.some((id: string) => {
      const receipt = value.evidence[id]
      return !record(receipt) || typeof receipt.text_sha256 !== "string" || !/^[a-f0-9]{64}$/.test(receipt.text_sha256)
    })))) return null
  if (text !== undefined) {
    const characters = Array.from(text)
    let cursor = 0
    const ids = new Set<string>()
    for (const item of value.findings) {
      if (ids.has(item.id) || item.start < cursor || item.end > characters.length ||
        /[\p{L}\p{N}]/u.test(characters.slice(cursor, item.start).join("")) ||
        characters.slice(item.start, item.end).join("") !== item.text) return null
      ids.add(item.id)
      cursor = item.end
      const cited = [...new Set(Array.from(item.text.matchAll(/\[([^\]\n]+)\]/g), (match: RegExpMatchArray) => {
        const group = match[1].trim()
        const located = PDF_LOCATION_CITATION.exec(group)
        return located ? [located[1]] : group.split(",").map(id => id.trim().replace(/^Source /, ""))
      }).flat())]
      const unavailable = ["not_checked", "error"].includes(item.status) && item.evidence_ids.length === 0
      if (!unavailable && JSON.stringify(cited) !== JSON.stringify(item.evidence_ids)) return null
    }
    const unavailable = value.findings.length === 0 && value.verdict === "unverified" && typeof value.context.unavailable_reason === "string"
    if (!unavailable && /[\p{L}\p{N}]/u.test(characters.slice(cursor).join(""))) return null
  }
  const evaluated = value.findings.filter(item => ["supported", "unsupported", "contradicted"].includes(item.status)).length
  const supported = value.findings.filter(item => item.status === "supported").length
  const total = value.findings.length
  const complete = total > 0 && evaluated === total
  const verdict = value.findings.some(item => ["unsupported", "contradicted", "uncited"].includes(item.status))
    ? "needs_revision" : complete && supported === total ? "supported" : "unverified"
  if (value.coverage.total !== total || value.coverage.evaluated !== evaluated || value.coverage.supported !== supported ||
    value.coverage.uncovered !== total - evaluated || value.coverage.complete !== complete || value.verdict !== verdict) return null
  try { JSON.stringify(value) } catch { return null }
  return value as ClaimVerification
}

export function verificationPresentation(value: unknown, text: string) {
  const report = readClaimVerification(value, text)
  if (!report) return { report: null, status: "unverified" as const, label: "Not checked", reason: value
    ? "The saved check is unavailable or its draft changed. Review the current text and evidence."
    : "No citation-support check is recorded for this text." }
  const completeSupport = report.verdict === "supported" && report.coverage.complete && report.coverage.total > 0 &&
    report.coverage.evaluated === report.coverage.total && report.coverage.supported === report.coverage.total &&
    report.coverage.uncovered === 0 && report.findings.length === report.coverage.total && report.findings.every(item => item.status === "supported")
  const needsReview = report.verdict === "needs_revision" || report.findings.some(item => ["unsupported", "contradicted", "uncited"].includes(item.status))
  return { report, status: completeSupport ? "supported" as const : needsReview ? "needs_revision" as const : "unverified" as const,
    label: completeSupport ? "Supported by cited evidence" : needsReview ? "Needs review" : "Not fully checked",
    reason: completeSupport ? "All recorded claims were checked against their cited evidence." : typeof report.context.unavailable_reason === "string"
      ? claimCheckReason(report.context.unavailable_reason) : "Read the findings and check the cited evidence before relying on this draft." }
}

export const CLAIM_VERIFICATION_LIMITATION = "These checks compare the draft with its cited evidence. They do not establish real-world truth or replace your review."

export const CLAIM_CHECK_REASONS: Record<string, string> = {
  missing_citation: "This claim has no citation. Review it and add supporting evidence.",
  citation_not_resolved: "The cited evidence could not be resolved.",
  citation_source_unavailable: "The cited source is unavailable in this workspace.",
  citation_source_changed_or_unresolved: "The citation could not be matched exactly to the saved source.",
  text_or_unit_limit: "This claim exceeded the check's text or claim limit and was not checked.",
  report_size_limit: "The citation report exceeded its size limit. The draft is retained, but its citation support has not been verified.",
  cited_evidence_limit: "The cited evidence exceeds the check's context limit.",
  citation_without_claim: "This citation has no accompanying claim to check.",
  judge_unavailable: "The citation checker is unavailable.",
  judge_timeout: "The citation check timed out.",
  verification_deadline: "The check's total time limit was reached before this claim was checked.",
  judge_failed: "The citation check could not finish.",
  judge_provider_failed: "The citation checking provider could not complete the request.",
  judge_response_invalid: "The checker returned an unusable response; this claim is not verified.",
  judge_completion_unconfirmed: "The checker's response was incomplete; this claim is not verified.",
  semantic_judgment_unavailable: "No usable citation-support judgment was returned."
}
export const claimCheckReason = (reason: string): string => CLAIM_CHECK_REASONS[reason] || reason

const INPUT_LIMIT_REASONS = new Set(["text_or_unit_limit", "cited_evidence_limit", "report_size_limit"])

/** Retry recoverable checks; unchanged input limits and completed judgments require review. */
export function hasUnfinishedSectionChecks(value: unknown): boolean {
  if (!record(value) || value.schemaVersion !== 1 || !Array.isArray(value.sections) || !value.sections.length ||
    !value.sections.every(section => record(section) && section.status === "completed" && typeof section.content === "string" && section.content.trim())) return false
  const unfinished = (reportValue: unknown, text: string) => {
    const report = readClaimVerification(reportValue, text)
    return !report || report.findings.length === 0 || report.findings.length !== report.coverage.total ||
      report.findings.some(finding => finding.status === "error" ||
        (finding.status === "not_checked" && !INPUT_LIMIT_REASONS.has(finding.reason)))
  }
  if (typeof value.synthesis === "string" && value.synthesis.trim() && unfinished(value.synthesis_verification, value.synthesis)) return true
  return value.sections.some(section => unfinished(section.claim_verification, section.content) ||
    (Array.isArray(section.batches) && section.batches.some((batch: unknown) => record(batch) &&
      (!batch.claim_verification || (typeof batch.verification_text === "string" && unfinished(batch.claim_verification, batch.verification_text))))))
}

/** Plain-text projection shared by readable exports; structured reports remain unchanged. */
export function renderClaimVerification(value: unknown, text: string, imported = false): string {
  const view = verificationPresentation(value, text)
  const lines = [`Citation support: ${view.label}`, view.reason, CLAIM_VERIFICATION_LIMITATION]
  if (imported) lines.push("Imported historical check; sources have not been checked again in this workspace. Use the restored source citations to open evidence.")
  if (view.report) {
    const { coverage } = view.report
    lines.push(coverage.total ? `${coverage.evaluated} of ${coverage.total} claims checked; ${coverage.supported} supported; ${coverage.uncovered} uncovered.` : "Check coverage is unavailable.")
    for (const finding of view.report.findings) lines.push(`[${finding.id}] ${finding.status}: ${finding.text}\n${claimCheckReason(finding.reason)}\nCitations: ${finding.evidence_ids.join(", ") || "None"}`)
  }
  return lines.join("\n\n")
}
