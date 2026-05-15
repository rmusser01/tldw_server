import React from "react"
import { Link } from "react-router-dom"
import { useTranslation } from "react-i18next"
import {
  Archive,
  CheckCircle,
  ExternalLink,
  FileOutput,
  GitBranch,
  History,
  ListChecks,
  Send,
  ShieldAlert,
  ShieldCheck,
  XCircle
} from "lucide-react"
import { Badge, type BadgeVariant } from "@/components/ui/primitives/Badge"
import { cn } from "@/libs/utils"
import type {
  ArtifactReviewStatus,
  GeneratedArtifact,
  TraceableArtifactExportRef
} from "@/types/workspace"

type Translate = ReturnType<typeof useTranslation>["t"]

const REVIEW_STATE_ORDER: ArtifactReviewStatus[] = [
  "draft",
  "reviewing",
  "accepted",
  "needs_revision",
  "rejected",
  "exported",
  "assigned",
  "archived"
]

const REVIEW_STATE_VARIANTS: Record<ArtifactReviewStatus, BadgeVariant> = {
  draft: "secondary",
  reviewing: "info",
  accepted: "success",
  needs_revision: "warning",
  rejected: "danger",
  exported: "primary",
  assigned: "info",
  archived: "secondary"
}

const REVIEW_STATE_ICONS: Partial<
  Record<ArtifactReviewStatus, React.ElementType>
> = {
  accepted: CheckCircle,
  needs_revision: ListChecks,
  rejected: XCircle,
  assigned: Send,
  archived: Archive
}

const REVIEW_STATE_COPY: Record<
  ArtifactReviewStatus,
  { key: string; defaultValue: string }
> = {
  draft: {
    key: "playground:studio.traceableArtifact.reviewStates.draft",
    defaultValue: "Draft"
  },
  reviewing: {
    key: "playground:studio.traceableArtifact.reviewStates.reviewing",
    defaultValue: "Reviewing"
  },
  accepted: {
    key: "playground:studio.traceableArtifact.reviewStates.accepted",
    defaultValue: "Accepted"
  },
  needs_revision: {
    key: "playground:studio.traceableArtifact.reviewStates.needsRevision",
    defaultValue: "Needs Revision"
  },
  rejected: {
    key: "playground:studio.traceableArtifact.reviewStates.rejected",
    defaultValue: "Rejected"
  },
  exported: {
    key: "playground:studio.traceableArtifact.reviewStates.exported",
    defaultValue: "Exported"
  },
  assigned: {
    key: "playground:studio.traceableArtifact.reviewStates.assigned",
    defaultValue: "Assigned"
  },
  archived: {
    key: "playground:studio.traceableArtifact.reviewStates.archived",
    defaultValue: "Archived"
  }
}

const EXPORT_FORMAT_COPY: Record<string, { key: string; defaultValue: string }> = {
  md: {
    key: "playground:studio.traceableArtifact.exportFormats.markdown",
    defaultValue: "Markdown"
  },
  markdown: {
    key: "playground:studio.traceableArtifact.exportFormats.markdown",
    defaultValue: "Markdown"
  },
  docx: {
    key: "playground:studio.traceableArtifact.exportFormats.docx",
    defaultValue: "DOCX"
  },
  pdf: {
    key: "playground:studio.traceableArtifact.exportFormats.pdf",
    defaultValue: "PDF"
  },
  slides: {
    key: "playground:studio.traceableArtifact.exportFormats.slides",
    defaultValue: "Slides"
  },
  chatbook: {
    key: "playground:studio.traceableArtifact.exportFormats.chatbook",
    defaultValue: "Chatbook"
  }
}

const getArtifactReviewStateLabel = (
  t: Translate,
  status: ArtifactReviewStatus | undefined
): string => {
  const copy = REVIEW_STATE_COPY[status || "draft"]
  return t(copy.key, copy.defaultValue)
}

const getExportFormatLabel = (t: Translate, format: string): string => {
  const copy = EXPORT_FORMAT_COPY[format.toLowerCase()]
  if (copy) return t(copy.key, copy.defaultValue)
  return t("playground:studio.traceableArtifact.exportFormats.unknown", {
    defaultValue: "{{format}}",
    format
  })
}

const buildAcpSessionRoute = (sessionId: string, view?: string): string => {
  const params = new URLSearchParams({ session: sessionId })
  if (view) params.set("view", view)
  return `/acp-playground?${params.toString()}`
}

const TraceValue: React.FC<{
  label: string
  value?: React.ReactNode
}> = ({ label, value }) => {
  if (value === null || value === undefined || value === "") return null
  return (
    <div className="min-w-0">
      <dt className="text-[10px] font-semibold uppercase tracking-wide text-text-muted">
        {label}
      </dt>
      <dd className="mt-0.5 min-w-0 break-words text-xs text-text">{value}</dd>
    </div>
  )
}

const DetailSection: React.FC<{
  title: string
  icon: React.ElementType
  children: React.ReactNode
}> = ({ title, icon: Icon, children }) => (
  <section className="rounded border border-border bg-surface2/40 p-3">
    <div className="mb-2 flex items-center gap-2">
      <Icon className="h-4 w-4 text-text-muted" />
      <h4 className="text-xs font-semibold uppercase tracking-wide text-text-muted">
        {title}
      </h4>
    </div>
    {children}
  </section>
)

type TraceableArtifactSummaryProps = {
  artifact: GeneratedArtifact
  className?: string
}

export const hasTraceableArtifactMetadata = (
  artifact: GeneratedArtifact
): boolean =>
  Boolean(
    artifact.reviewStatus ||
      artifact.producerMetadata ||
      artifact.sourceLineage?.length ||
      artifact.reviewMetadata ||
      artifact.versionMetadata ||
      artifact.exportRefs?.length ||
      artifact.redaction ||
      artifact.rootArtifactId ||
      artifact.artifactVersionId ||
      artifact.previousVersionId ||
      artifact.schemaVersion !== undefined
  )

export const TraceableArtifactSummary: React.FC<
  TraceableArtifactSummaryProps
> = ({ artifact, className }) => {
  const { t } = useTranslation(["playground", "common"])
  const reviewStatus = artifact.reviewStatus || "draft"
  const redactionRestricted =
    artifact.redaction?.supportSafe === false || artifact.redaction?.redacted === true
  const versionLabel =
    artifact.version !== undefined ? `v${artifact.version}` : artifact.artifactVersionId
  const redactionLabel =
    artifact.redaction?.supportSafe === false
      ? t("playground:studio.traceableArtifact.restricted", "Restricted")
      : artifact.redaction?.redacted
        ? t("playground:studio.traceableArtifact.redacted", "Redacted")
        : t("playground:studio.traceableArtifact.supportSafe", "Support safe")

  return (
    <div
      data-testid="traceable-artifact-summary"
      className={cn("flex flex-wrap items-center gap-1.5", className)}
    >
      <Badge
        size="sm"
        variant={REVIEW_STATE_VARIANTS[reviewStatus]}
        dot
        data-testid="traceable-artifact-review-state"
      >
        {getArtifactReviewStateLabel(t, reviewStatus)}
      </Badge>
      {versionLabel && (
        <Badge size="sm" variant="secondary" outline>
          {versionLabel}
        </Badge>
      )}
      {!redactionRestricted &&
        (artifact.producerMetadata?.producerType || artifact.producerMetadata?.runId) && (
          <Badge size="sm" variant="info" outline>
            {artifact.producerMetadata.producerType?.toUpperCase() ||
              t("playground:studio.traceableArtifact.run", "Run")}
          </Badge>
        )}
      {artifact.redaction && (
        <Badge
          size="sm"
          variant={artifact.redaction.supportSafe === false ? "warning" : "secondary"}
          outline
        >
          {redactionLabel}
        </Badge>
      )}
    </div>
  )
}

type TraceableArtifactDetailProps = {
  artifact: GeneratedArtifact
  onReviewStateChange?: (reviewStatus: ArtifactReviewStatus) => void
  reviewStateControlsDisabled?: boolean
}

export const TraceableArtifactDetail: React.FC<
  TraceableArtifactDetailProps
> = ({ artifact, onReviewStateChange, reviewStateControlsDisabled = false }) => {
  const { t } = useTranslation(["playground", "common"])
  const reviewStatus = artifact.reviewStatus || "draft"
  const producer = artifact.producerMetadata
  const redaction = artifact.redaction
  const redactionRestricted =
    redaction?.supportSafe === false || redaction?.redacted === true
  const sessionId = producer?.sessionId
  const versionLabel =
    artifact.version !== undefined ? `v${artifact.version}` : undefined
  const sourceLineage = artifact.sourceLineage || []
  const exportRefs = artifact.exportRefs || []

  return (
    <div className="space-y-3 text-sm text-text">
      <TraceableArtifactSummary artifact={artifact} />

      <DetailSection
        title={t("playground:studio.traceableArtifact.reviewState", "Review state")}
        icon={ListChecks}
      >
        <div
          role="group"
          aria-label={t(
            "playground:studio.traceableArtifact.reviewStateControls",
            "Review state controls"
          )}
          className="flex flex-wrap gap-1.5"
        >
          {REVIEW_STATE_ORDER.map((state) => {
            const active = state === reviewStatus
            const Icon = REVIEW_STATE_ICONS[state]
            const disabled =
              reviewStateControlsDisabled || !onReviewStateChange || active
            return (
              <button
                key={state}
                type="button"
                disabled={disabled}
                aria-pressed={active}
                onClick={() => onReviewStateChange?.(state)}
                className={cn(
                  "inline-flex items-center gap-1 rounded border px-2 py-1 text-[11px] font-medium transition",
                  active
                    ? "border-primary/40 bg-primary/10 text-primary"
                    : "border-border bg-surface text-text-muted hover:border-primary/40 hover:text-text",
                  disabled && !active
                    ? "cursor-not-allowed opacity-60 hover:border-border hover:text-text-muted"
                    : ""
                )}
              >
                {Icon && <Icon className="h-3 w-3" aria-hidden="true" />}
                {getArtifactReviewStateLabel(t, state)}
              </button>
            )
          })}
        </div>
      </DetailSection>

      <DetailSection
        title={t("playground:studio.traceableArtifact.acpProvenance", "ACP provenance")}
        icon={GitBranch}
      >
        {redactionRestricted ? (
          <p className="text-xs text-text-muted">
            {t(
              "playground:studio.traceableArtifact.provenanceRedacted",
              "Provenance hidden by redaction posture"
            )}
          </p>
        ) : producer ? (
          <div className="space-y-2">
            <dl className="grid gap-2 sm:grid-cols-2">
              <TraceValue
                label={t("playground:studio.traceableArtifact.producer", "Producer")}
                value={producer.producerType}
              />
              <TraceValue
                label={t("playground:studio.traceableArtifact.task", "Task")}
                value={producer.producerId || producer.taskId}
              />
              <TraceValue
                label={t("playground:studio.traceableArtifact.run", "Run")}
                value={producer.runId}
              />
              <TraceValue
                label={t("playground:studio.traceableArtifact.session", "Session")}
                value={producer.sessionId}
              />
              <TraceValue
                label={t("playground:studio.traceableArtifact.model", "Model")}
                value={producer.model}
              />
              <TraceValue
                label={t("playground:studio.traceableArtifact.provider", "Provider")}
                value={producer.provider}
              />
            </dl>
            {sessionId && (
              <div className="flex flex-wrap gap-2 pt-1">
                <Link
                  to={buildAcpSessionRoute(sessionId)}
                  className="inline-flex items-center gap-1 rounded border border-border px-2 py-1 text-xs text-primary hover:bg-primary/10"
                >
                  {t("playground:studio.traceableArtifact.openSession", "Open session")}
                  <ExternalLink className="h-3 w-3" aria-hidden="true" />
                </Link>
                <Link
                  to={buildAcpSessionRoute(sessionId, "diagnostics")}
                  className="inline-flex items-center gap-1 rounded border border-border px-2 py-1 text-xs text-primary hover:bg-primary/10"
                >
                  {t("playground:studio.traceableArtifact.diagnostics", "Diagnostics")}
                  <ExternalLink className="h-3 w-3" aria-hidden="true" />
                </Link>
              </div>
            )}
          </div>
        ) : (
          <p className="text-xs text-text-muted">
            {t(
              "playground:studio.traceableArtifact.noAcpProvenance",
              "No ACP provenance recorded"
            )}
          </p>
        )}
      </DetailSection>

      <DetailSection
        title={t("playground:studio.traceableArtifact.sourceLineage", "Source lineage")}
        icon={GitBranch}
      >
        {redactionRestricted ? (
          <p className="text-xs text-text-muted">
            {t(
              "playground:studio.traceableArtifact.lineageRedacted",
              "Source lineage hidden by redaction posture"
            )}
          </p>
        ) : sourceLineage.length > 0 ? (
          <ul className="space-y-2">
            {sourceLineage.map((source) => (
              <li
                key={`${source.sourceId}:${source.mediaId ?? source.title ?? ""}`}
                className="rounded border border-border/70 bg-surface/60 p-2"
              >
                <div className="flex flex-wrap items-start justify-between gap-2">
                  <div className="min-w-0">
                    <p className="break-words text-xs font-medium text-text">
                      {source.title || source.label || source.sourceId}
                    </p>
                    <p className="mt-0.5 break-words text-[11px] text-text-muted">
                      {source.sourceId}
                    </p>
                  </div>
                  {source.citationCount !== undefined && (
                    <Badge size="sm" variant="secondary" outline>
                      {t("playground:studio.traceableArtifact.citationCount", {
                        count: source.citationCount,
                        defaultValue: "{{count}} citation",
                        defaultValue_plural: "{{count}} citations"
                      })}
                    </Badge>
                  )}
                </div>
                {(source.sourceType || source.mediaId !== undefined) && (
                  <dl className="mt-2 grid gap-2 sm:grid-cols-2">
                    <TraceValue
                      label={t("playground:studio.traceableArtifact.type", "Type")}
                      value={source.sourceType}
                    />
                    <TraceValue
                      label={t("playground:studio.traceableArtifact.media", "Media")}
                      value={source.mediaId}
                    />
                  </dl>
                )}
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-xs text-text-muted">
            {t(
              "playground:studio.traceableArtifact.noSourceLineage",
              "No source lineage recorded"
            )}
          </p>
        )}
      </DetailSection>

      <DetailSection
        title={t("playground:studio.traceableArtifact.version", "Version")}
        icon={History}
      >
        <dl className="grid gap-2 sm:grid-cols-2">
          <TraceValue
            label={t("playground:studio.traceableArtifact.version", "Version")}
            value={versionLabel}
          />
          <TraceValue
            label={t("playground:studio.traceableArtifact.versionId", "Version ID")}
            value={artifact.artifactVersionId}
          />
          <TraceValue
            label={t("playground:studio.traceableArtifact.root", "Root")}
            value={artifact.rootArtifactId}
          />
          <TraceValue
            label={t("playground:studio.traceableArtifact.previous", "Previous")}
            value={artifact.previousVersionId}
          />
          <TraceValue
            label={t("playground:studio.traceableArtifact.reason", "Reason")}
            value={artifact.versionMetadata?.revisionReason}
          />
          <TraceValue
            label={t("playground:studio.traceableArtifact.schema", "Schema")}
            value={artifact.schemaVersion}
          />
        </dl>
      </DetailSection>

      <DetailSection
        title={t("playground:studio.traceableArtifact.redaction", "Redaction")}
        icon={
          redaction?.supportSafe === false || redaction?.redacted
            ? ShieldAlert
            : ShieldCheck
        }
      >
        {redaction ? (
          <div className="flex flex-wrap gap-1.5">
            <Badge
              size="sm"
              variant={redaction.supportSafe === false ? "warning" : "success"}
              outline
            >
              {redaction.supportSafe === false
                ? t("playground:studio.traceableArtifact.restricted", "Restricted")
                : t("playground:studio.traceableArtifact.supportSafe", "Support safe")}
            </Badge>
            <Badge
              size="sm"
              variant={redaction.redacted ? "warning" : "secondary"}
              outline
            >
              {redaction.redacted
                ? t("playground:studio.traceableArtifact.redacted", "Redacted")
                : t("playground:studio.traceableArtifact.notRedacted", "Not redacted")}
            </Badge>
            {redaction.retentionClass && (
              <Badge size="sm" variant="secondary" outline>
                {redaction.retentionClass}
              </Badge>
            )}
          </div>
        ) : (
          <p className="text-xs text-text-muted">
            {t(
              "playground:studio.traceableArtifact.noRedactionPosture",
              "No redaction posture recorded"
            )}
          </p>
        )}
      </DetailSection>

      <DetailSection
        title={t("playground:studio.traceableArtifact.exports", "Exports")}
        icon={FileOutput}
      >
        {exportRefs.length > 0 ? (
          <ul className="space-y-1.5">
            {exportRefs.map((exportRef, index) => (
              <li
                key={getExportRefKey(exportRef, index)}
                className="flex flex-wrap items-center gap-2 rounded border border-border/70 bg-surface/60 px-2 py-1.5"
              >
                <Badge size="sm" variant="secondary" outline>
                  {getExportFormatLabel(t, exportRef.format)}
                </Badge>
                {exportRef.fileId !== undefined && (
                  <span className="text-xs text-text-muted">
                    {t("playground:studio.traceableArtifact.fileRef", {
                      id: String(exportRef.fileId),
                      defaultValue: "file #{{id}}"
                    })}
                  </span>
                )}
                {exportRef.jobId !== undefined && (
                  <span className="text-xs text-text-muted">
                    {t("playground:studio.traceableArtifact.jobRef", {
                      id: String(exportRef.jobId),
                      defaultValue: "job #{{id}}"
                    })}
                  </span>
                )}
                {exportRef.status && (
                  <span className="text-xs text-text-muted">
                    {t("playground:studio.traceableArtifact.exportStatus", {
                      defaultValue: "{{status}}",
                      status: exportRef.status
                    })}
                  </span>
                )}
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-xs text-text-muted">
            {t("playground:studio.traceableArtifact.noExports", "No exports recorded")}
          </p>
        )}
      </DetailSection>
    </div>
  )
}

const getExportRefKey = (
  exportRef: TraceableArtifactExportRef,
  index: number
): string =>
  [
    exportRef.id,
    exportRef.format,
    exportRef.fileId,
    exportRef.jobId,
    exportRef.artifactVersionId,
    exportRef.url,
    index
  ]
    .filter((value) => value !== undefined && value !== null && value !== "")
    .map(String)
    .join(":")
