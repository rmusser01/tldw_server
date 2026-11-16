import React from "react"
import { ArtifactRunIdentity } from "./ArtifactRunIdentity"
import { getArtifactRunIdentity } from "./output-identity"
import { ClaimVerificationPanel } from "../ClaimVerificationPanel"
import { SectionArtifactReviewPanel, type SaveSectionArtifactReview } from "./SectionArtifactReviewPanel"
import { getSectionReviewTarget } from "@/utils/section-artifact-review"
import { Link } from "react-router-dom"
import { useTranslation } from "react-i18next"
import { MarkdownPreview } from "@/components/Common/MarkdownPreview"
import { buildDocumentPagePath } from "@/routes/route-paths"
import type { GeneratedArtifact, WorkspaceSource } from "@/types/workspace"
import { useWorkspaceStore } from "@/store/workspace"
import { TraceableArtifactDetail, hasTraceableArtifactMetadata } from "./TraceableArtifactDetail"

type Section = Record<string, unknown> & { title: string; status: string }
type SectionAnalysis = Record<string, unknown> & { schemaVersion: 1; sections: Section[] }

const record = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === "object" && !Array.isArray(value)
const text = (value: unknown): string => typeof value === "string" ? value : ""

/** Read the structured display payload without changing the canonical artifact. */
export const getSectionAnalysis = (artifact: GeneratedArtifact): SectionAnalysis | null => {
  const client = artifact.versionMetadata?.workspaceClient
  const restoredData = record(client) && record(client.data) ? client.data : undefined
  const candidate = artifact.data?.sectionAnalysis ?? restoredData?.sectionAnalysis
  if (!record(candidate) || candidate.schemaVersion !== 1 ||
      !Array.isArray(candidate.sections) || candidate.sections.length === 0 ||
      !candidate.sections.every(section => record(section) &&
        typeof section.title === "string" && typeof section.status === "string")) return null
  return candidate as SectionAnalysis
}

/** Follow the existing saved-checkpoint stream without adding requests or authority. */
export const CurrentSectionAnalysisArtifactViewer: React.FC<{
  artifactId: string
  workspaceId: string | null
  isCurrent: () => boolean
  onSaveReview: SaveSectionArtifactReview
}> = ({ artifactId, workspaceId, isCurrent, onSaveReview }) => {
  const { t } = useTranslation("playground")
  const artifact = useWorkspaceStore(state => state.workspaceId === workspaceId
    ? state.generatedArtifacts.find(item => item.id === artifactId) : undefined)
  const sources = useWorkspaceStore(state => state.sources)
  const analysis = artifact && getSectionAnalysis(artifact)
  if (!isCurrent() || !artifact || !analysis)
    return <p role="status">{t("studio.sectionReader.unavailable", "This output is no longer available in the current workspace.")}</p>
  return <SectionAnalysisArtifactViewer key={`${workspaceId}:${artifactId}`} artifact={artifact} analysis={analysis} sources={sources}
    onSaveReview={artifact.workspaceArtifactId && artifact.status !== "generating" ? onSaveReview : undefined} />
}

const PageLinks: React.FC<{ mediaId: unknown; spans: unknown; range?: boolean }> = ({ mediaId, spans, range }) => {
  const { t } = useTranslation("playground")
  const pages = Array.isArray(spans) ? spans.filter(record).flatMap(span => {
    const path = buildDocumentPagePath(mediaId, span.page_number)
    if (!path) return []
    const printed = text(span.printed_page_label)
    return [{ path, page: span.page_number, printed }]
  }) : []
  const visible = range && pages.length > 2 ? [pages[0], pages[pages.length - 1]] : pages
  if (!visible.length) return <span>{t("studio.sectionReader.pageUnavailable", "Page reference unavailable")}</span>
  return <span className="inline-flex flex-wrap gap-x-2 gap-y-1">
    {visible.map((page, index) => <React.Fragment key={`${page.path}:${index}`}>
      {index > 0 && <span aria-hidden="true">{range ? "–" : ";"}</span>}
      <Link to={page.path} className="text-primary underline underline-offset-2 hover:no-underline focus-visible:outline focus-visible:outline-2 focus-visible:outline-primary">
        {page.printed ? `${t("studio.sectionReader.printed", "Printed")} ${page.printed} · ` : ""}
        {`PDF ${page.page}`}
      </Link>
    </React.Fragment>)}
  </span>
}

/** A bounded reading projection; full summaries and evidence remain in exports. */
export const SectionAnalysisArtifactViewer: React.FC<{
  artifact: GeneratedArtifact
  analysis: SectionAnalysis
  sources: WorkspaceSource[]
  onSaveReview?: SaveSectionArtifactReview
}> = ({ artifact, analysis, sources, onSaveReview }) => {
  const { t } = useTranslation("playground")
  const [savedReview, setSavedReview] = React.useState<{ original: GeneratedArtifact; saved: GeneratedArtifact }>()
  const reviewArtifact = savedReview?.original === artifact ? savedReview.saved : artifact
  const generating = artifact.status === "generating"
  const selectorId = React.useId()
  const [selection, setSelection] = React.useState("overview")
  const [evidenceOpen, setEvidenceOpen] = React.useState(false)
  const [technicalOpen, setTechnicalOpen] = React.useState(false)
  const section = selection === "overview" ? undefined : analysis.sections[Number(selection)]
  const completed = analysis.sections.filter(item => item.status === "completed").length
  const evidence = React.useMemo(() => !section || !Array.isArray(section.batches) ? [] :
    section.batches.filter(record).flatMap(batch => Array.isArray(batch.evidence)
      ? batch.evidence.filter(record) : []), [section])
  const sourceTitle = section ? sources.find(source => source.id === section.source_id ||
    source.mediaId === section.media_id)?.title : undefined
  const summary = section ? text(section.content) : text(analysis.synthesis)
  const reviewTarget = getSectionReviewTarget(reviewArtifact, selection)

  return <div className="max-h-[70vh] space-y-4 overflow-y-auto text-text">
    <div className="space-y-2">
      <ArtifactRunIdentity artifact={artifact} />
      <p className="text-sm font-medium">{`${completed} ${t("studio.sectionReader.of", "of")} ${analysis.sections.length} ${t("studio.sectionReader.completed", "sections completed")}`}</p>
      <p className="text-xs text-text-muted">{t("studio.sectionReader.scope", "Only the selected sections are covered. Document proposals are not evidence of enacted policy.")}</p>
      {generating && <p role="status" className="text-sm text-text-muted">{t("studio.sectionReader.generating", "Generation continues. Completed chapters update here. Review changes can be saved after generation finishes.")}</p>}
      {!generating && completed !== analysis.sections.length && <p className="text-sm text-warning">{t("studio.sectionReader.partial", "Partial coverage: failed or unread sections are excluded from the overview.")}</p>}
      {text(analysis.stopped_error) && <p className="text-sm text-error">{text(analysis.stopped_error)}</p>}
      {!section && text(analysis.synthesis_error) && <p className="text-sm text-error">{text(analysis.synthesis_error)}</p>}
      {!section && summary && text(analysis.synthesis_error) && <p className="text-sm text-text-muted">{t("studio.sectionReader.retainedOverview", "The overview below is from an earlier attempt. Completed chapters and evidence remain available.")}</p>}
      {!section && summary && generating && <p className="text-sm text-text-muted">{t("studio.sectionReader.runningOverview", "This saved overview may change while generation continues.")}</p>}
      <label htmlFor={selectorId} className="block text-sm font-medium">{t("studio.sectionReader.selector", "Chapter or section")}</label>
      <select id={selectorId} value={selection} onChange={event => {
        setSelection(event.target.value)
        setEvidenceOpen(false)
      }} className="min-h-10 w-full rounded border border-border bg-surface px-3 py-2 text-sm text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-primary">
        <option value="overview">{t("studio.sectionReader.overview", "Overview")}</option>
        {analysis.sections.map((item, index) => <option key={index} value={String(index)}>
          {item.title}{item.status !== "completed" ? ` — ${item.status}` : ""}
        </option>)}
      </select>
    </div>

    <section key={selection} className="space-y-3" aria-label={section?.title || t("studio.sectionReader.overview", "Overview")}>
      <h3 className="text-lg font-semibold">{section?.title || t("studio.sectionReader.overview", "Overview")}</h3>
      {section && <div className="space-y-1 text-sm text-text-muted">
        {sourceTitle && <p>{sourceTitle}</p>}
        <PageLinks mediaId={section.media_id} spans={section.page_spans} range />
        {section.status !== "completed" && <p>{t("studio.sectionReader.status", "Status")}: {section.status}</p>}
        {text(section.error) && <p className="text-error">{text(section.error)}</p>}
      </div>}
      {summary ? <MarkdownPreview content={summary} size="base" /> : <p className="text-sm text-text-muted">
        {section ? section.status === "unread"
          ? t("studio.sectionReader.unreadSection", "This section has not been processed yet.")
          : section.status === "generating"
            ? t("studio.sectionReader.generatingSection", "This section is still being processed. Saved evidence remains available below.")
            : t("studio.sectionReader.noSummary", "This section has no completed summary yet. Saved evidence remains available below.")
          : generating
            ? t("studio.sectionReader.pendingOverview", "The overview is pending while generation continues. Choose a completed chapter to read now.")
            : t("studio.sectionReader.noOverview", "Overview is not available yet.")}
      </p>}
      {summary && <ClaimVerificationPanel text={summary} report={section ? section.claim_verification : analysis.synthesis_verification}
        imported={(getArtifactRunIdentity(artifact).imported || artifact.producerMetadata?.origin === "research_pack_import")} />}
      {section && <div className="border-t border-border pt-3">
        <button type="button" aria-expanded={evidenceOpen} onClick={() => setEvidenceOpen(open => !open)}
          className="min-h-10 text-sm font-medium text-primary underline underline-offset-2 focus-visible:outline focus-visible:outline-2 focus-visible:outline-primary">
          {evidenceOpen ? t("studio.sectionReader.hideEvidence", "Hide evidence") : t("studio.sectionReader.showEvidence", "Show evidence")} ({evidence.length})
        </button>
        {evidenceOpen && <div className="mt-3 space-y-5">
          {Array.isArray(section.batches) && section.batches.filter(record).map((batch, index) => batch.claim_verification && <details key={index}>
            <summary className="min-h-10 cursor-pointer py-2 font-medium">{t("claims.extraction", "Saved evidence checks")} {index + 1}</summary>
            <ClaimVerificationPanel text={text(batch.verification_text)} report={batch.claim_verification} imported={(getArtifactRunIdentity(artifact).imported || artifact.producerMetadata?.origin === "research_pack_import")} />
          </details>)}
          {evidence.length === 0 && <p className="text-sm text-text-muted">{t("studio.sectionReader.noEvidence", "No saved evidence for this section.")}</p>}
          {evidence.map((item, index) => <section key={`${text(item.id)}:${index}`} className="space-y-2 border-b border-border pb-4 text-sm">
            <p className="font-medium">[{text(item.id) || index + 1}] {text(item.claim)}</p>
            {text(item.author) && <p>{text(item.author)}</p>}
            <PageLinks mediaId={section.media_id} spans={item.page_spans} />
            <blockquote className="whitespace-pre-wrap rounded bg-surface2 p-3">{text(item.quote)}</blockquote>
            {text(item.interpretation) && <p><strong>{t("studio.sectionReader.interpretation", "Interpretation")}: </strong>{text(item.interpretation)}</p>}
            {text(item.limitations) && <p><strong>{t("studio.sectionReader.limitations", "Limitations")}: </strong>{text(item.limitations)}</p>}
          </section>)}
        </div>}
      </div>}
    </section>

    {reviewTarget && <SectionArtifactReviewPanel artifact={reviewArtifact} target={reviewTarget}
      onSave={generating ? undefined : onSaveReview} onSaved={saved => setSavedReview({ original: artifact, saved })} />}

    {hasTraceableArtifactMetadata(reviewArtifact) && <div className="border-t border-border pt-3">
      <button type="button" aria-expanded={technicalOpen} onClick={() => setTechnicalOpen(open => !open)}
        className="min-h-10 text-sm text-text-muted underline underline-offset-2 focus-visible:outline focus-visible:outline-2 focus-visible:outline-primary">
        {t("studio.sectionReader.technical", "Technical details")}
      </button>
      {technicalOpen && <TraceableArtifactDetail artifact={reviewArtifact} />}
    </div>}
  </div>
}
