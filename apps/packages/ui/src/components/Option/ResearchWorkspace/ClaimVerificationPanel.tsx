import React from "react"
import { Link } from "react-router-dom"
import { useTranslation } from "react-i18next"
import { buildDocumentPagePath } from "@/routes/route-paths"
import { CLAIM_VERIFICATION_LIMITATION, CLAIM_CHECK_REASONS, claimCheckReason, verificationPresentation } from "@/utils/claim-verification"

/** Automated citation support remains separate from a person's review decision. */
export const ClaimVerificationPanel: React.FC<{ report?: unknown; text: string; checking?: boolean; imported?: boolean }> = ({ report, text, checking, imported }) => {
  const { t } = useTranslation("playground")
  const view = verificationPresentation(report, text)
  const limitReasons = [...new Set((view.report?.findings || []).filter(finding => finding.status === "not_checked" &&
    ["text_or_unit_limit", "report_size_limit"].includes(finding.reason)).map(finding => finding.reason))]
    .filter(reason => claimCheckReason(reason) !== view.reason)
  if (checking) return <p role="status" className="text-sm text-text-muted">{t("claims.checking", "Checking citation support… The answer is retained while checks finish.")}</p>
  return <section className="space-y-2 border-t border-border pt-3 text-sm" aria-label={t("claims.heading", "Citation support")}>
    <p className={view.status === "supported" ? "font-medium text-success" : view.status === "needs_revision" ? "font-medium text-warning" : "font-medium text-text-muted"}>
      {t(`claims.status.${view.status}.${view.report ? "recorded" : "missing"}`, view.label)}
    </p>
    <p>{view.reason}</p>
    {limitReasons.map(reason => <p key={reason} className="text-warning">{t(`claims.reason.${reason}`, claimCheckReason(reason))}</p>)}
    {imported && <p className="text-text-muted">{t("claims.imported", "Imported historical check; sources have not been checked again in this workspace.")}</p>}
    <p className="text-xs text-text-muted">{t("claims.limitation", CLAIM_VERIFICATION_LIMITATION)}</p>
    {view.report && <>
      <p className="text-xs text-text-muted">{view.report.coverage.total
        ? t("claims.coverage", "{{evaluated}} of {{total}} claims checked; {{supported}} supported; {{uncovered}} uncovered.", view.report.coverage)
        : t("claims.coverageUnavailable", "Check coverage is unavailable.")}</p>
      {view.report.findings.length > 0 && <details>
        <summary className="min-h-10 cursor-pointer py-2 font-medium text-primary focus-visible:outline focus-visible:outline-2 focus-visible:outline-primary">{t("claims.findings", "Findings and cited evidence")} ({view.report.findings.length})</summary>
        <ol className="space-y-4">
          {view.report.findings.slice(0, 256).map((finding, index) => <li key={`${finding.id}:${index}`} className="space-y-1">
            <p className="font-medium">{finding.text}</p>
            <p>{t(`claims.finding.${finding.status}`, finding.status.replaceAll("_", " "))}: {CLAIM_CHECK_REASONS[finding.reason]
              ? t(`claims.reason.${finding.reason}`, claimCheckReason(finding.reason)) : finding.reason}</p>
            <p className="flex flex-wrap gap-x-3 gap-y-1">{finding.evidence_ids.map(id => {
              const evidence = view.report!.evidence[id]
              const spans = evidence?.page_spans
              const first = Array.isArray(spans) ? spans[0] : undefined
              const page = first && typeof first === "object" ? first.page_number : undefined
              const path = buildDocumentPagePath(evidence?.media_id, page)
              return path ? <Link key={id} to={path} target="_blank" rel="noreferrer" className="text-primary underline underline-offset-2">[{id}]</Link> : <span key={id}>[{id}]</span>
            })}{finding.evidence_ids.length === 0 && t("claims.uncited", "No cited evidence")}</p>
          </li>)}
        </ol>
        {view.report.findings.length > 256 && <p className="text-text-muted">{t("claims.moreFindings", "Showing the first 256 findings. All findings are retained in the complete research pack.")}</p>}
      </details>}
    </>}
  </section>
}
