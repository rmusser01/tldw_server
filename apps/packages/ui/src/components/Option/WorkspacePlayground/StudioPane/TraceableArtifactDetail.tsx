import React from "react"
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

export const formatArtifactReviewStateLabel = (
  status: ArtifactReviewStatus | string | undefined
): string => {
  if (!status) return "Draft"
  return status
    .split("_")
    .filter(Boolean)
    .map((part) => part[0].toUpperCase() + part.slice(1))
    .join(" ")
}

const formatExportFormatLabel = (format: string): string => {
  if (format.toLowerCase() === "md") return "Markdown"
  return formatArtifactReviewStateLabel(format)
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
      artifact.schemaVersion
  )

export const TraceableArtifactSummary: React.FC<
  TraceableArtifactSummaryProps
> = ({ artifact, className }) => {
  const reviewStatus = artifact.reviewStatus || "draft"
  const versionLabel =
    artifact.version !== undefined ? `v${artifact.version}` : artifact.artifactVersionId
  const redactionLabel =
    artifact.redaction?.supportSafe === false
      ? "Restricted"
      : artifact.redaction?.redacted
        ? "Redacted"
        : "Support safe"

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
        {formatArtifactReviewStateLabel(reviewStatus)}
      </Badge>
      {versionLabel && (
        <Badge size="sm" variant="secondary" outline>
          {versionLabel}
        </Badge>
      )}
      {(artifact.producerMetadata?.producerType || artifact.producerMetadata?.runId) && (
        <Badge size="sm" variant="info" outline>
          {artifact.producerMetadata.producerType?.toUpperCase() || "Run"}
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
  const reviewStatus = artifact.reviewStatus || "draft"
  const producer = artifact.producerMetadata
  const redaction = artifact.redaction
  const sessionId = producer?.sessionId
  const versionLabel =
    artifact.version !== undefined ? `v${artifact.version}` : undefined
  const sourceLineage = artifact.sourceLineage || []
  const exportRefs = artifact.exportRefs || []

  return (
    <div className="space-y-3 text-sm text-text">
      <TraceableArtifactSummary artifact={artifact} />

      <DetailSection title="Review state" icon={ListChecks}>
        <div
          role="group"
          aria-label="Review state controls"
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
                {formatArtifactReviewStateLabel(state)}
              </button>
            )
          })}
        </div>
      </DetailSection>

      <DetailSection title="ACP provenance" icon={GitBranch}>
        {producer ? (
          <div className="space-y-2">
            <dl className="grid gap-2 sm:grid-cols-2">
              <TraceValue label="Producer" value={producer.producerType} />
              <TraceValue label="Task" value={producer.producerId || producer.taskId} />
              <TraceValue label="Run" value={producer.runId} />
              <TraceValue label="Session" value={producer.sessionId} />
              <TraceValue label="Model" value={producer.model} />
              <TraceValue label="Provider" value={producer.provider} />
            </dl>
            {sessionId && (
              <div className="flex flex-wrap gap-2 pt-1">
                <a
                  href={buildAcpSessionRoute(sessionId)}
                  className="inline-flex items-center gap-1 rounded border border-border px-2 py-1 text-xs text-primary hover:bg-primary/10"
                >
                  Open session
                  <ExternalLink className="h-3 w-3" aria-hidden="true" />
                </a>
                <a
                  href={buildAcpSessionRoute(sessionId, "diagnostics")}
                  className="inline-flex items-center gap-1 rounded border border-border px-2 py-1 text-xs text-primary hover:bg-primary/10"
                >
                  Diagnostics
                  <ExternalLink className="h-3 w-3" aria-hidden="true" />
                </a>
              </div>
            )}
          </div>
        ) : (
          <p className="text-xs text-text-muted">No ACP provenance recorded</p>
        )}
      </DetailSection>

      <DetailSection title="Source lineage" icon={GitBranch}>
        {sourceLineage.length > 0 ? (
          <ul className="space-y-2">
            {sourceLineage.map((source) => (
              <li
                key={`${source.sourceId}:${source.mediaId || source.title || ""}`}
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
                      {source.citationCount}{" "}
                      {source.citationCount === 1 ? "citation" : "citations"}
                    </Badge>
                  )}
                </div>
                {(source.sourceType || source.mediaId !== undefined) && (
                  <dl className="mt-2 grid gap-2 sm:grid-cols-2">
                    <TraceValue label="Type" value={source.sourceType} />
                    <TraceValue label="Media" value={source.mediaId} />
                  </dl>
                )}
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-xs text-text-muted">No source lineage recorded</p>
        )}
      </DetailSection>

      <DetailSection title="Version" icon={History}>
        <dl className="grid gap-2 sm:grid-cols-2">
          <TraceValue label="Version" value={versionLabel} />
          <TraceValue label="Version ID" value={artifact.artifactVersionId} />
          <TraceValue label="Root" value={artifact.rootArtifactId} />
          <TraceValue label="Previous" value={artifact.previousVersionId} />
          <TraceValue
            label="Reason"
            value={artifact.versionMetadata?.revisionReason}
          />
          <TraceValue label="Schema" value={artifact.schemaVersion} />
        </dl>
      </DetailSection>

      <DetailSection
        title="Redaction"
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
              {redaction.supportSafe === false ? "Restricted" : "Support safe"}
            </Badge>
            <Badge
              size="sm"
              variant={redaction.redacted ? "warning" : "secondary"}
              outline
            >
              {redaction.redacted ? "Redacted" : "Not redacted"}
            </Badge>
            {redaction.retentionClass && (
              <Badge size="sm" variant="secondary" outline>
                {redaction.retentionClass}
              </Badge>
            )}
          </div>
        ) : (
          <p className="text-xs text-text-muted">No redaction posture recorded</p>
        )}
      </DetailSection>

      <DetailSection title="Exports" icon={FileOutput}>
        {exportRefs.length > 0 ? (
          <ul className="space-y-1.5">
            {exportRefs.map((exportRef) => (
              <li
                key={getExportRefKey(exportRef)}
                className="flex flex-wrap items-center gap-2 rounded border border-border/70 bg-surface/60 px-2 py-1.5"
              >
                <Badge size="sm" variant="secondary" outline>
                  {formatExportFormatLabel(exportRef.format)}
                </Badge>
                {exportRef.fileId !== undefined && (
                  <span className="text-xs text-text-muted">
                    file #{String(exportRef.fileId)}
                  </span>
                )}
                {exportRef.jobId !== undefined && (
                  <span className="text-xs text-text-muted">
                    job #{String(exportRef.jobId)}
                  </span>
                )}
                {exportRef.status && (
                  <span className="text-xs text-text-muted">{exportRef.status}</span>
                )}
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-xs text-text-muted">No exports recorded</p>
        )}
      </DetailSection>
    </div>
  )
}

const getExportRefKey = (exportRef: TraceableArtifactExportRef): string =>
  [
    exportRef.format,
    exportRef.fileId,
    exportRef.jobId,
    exportRef.artifactVersionId,
    exportRef.url
  ]
    .filter((value) => value !== undefined && value !== null && value !== "")
    .map(String)
    .join(":")
