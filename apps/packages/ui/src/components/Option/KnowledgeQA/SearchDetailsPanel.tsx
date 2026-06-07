/**
 * SearchDetailsPanel - Optional runtime details for retrieval quality inspection
 */

import React from "react"
import { BarChart3 } from "lucide-react"
import { useKnowledgeQA } from "./KnowledgeQAProvider"
import { cn } from "@/libs/utils"
import type { RagResult } from "./types"
import {
  getEvidenceOrigin,
  getResultSourceStatus,
  getResultUnavailableReason,
} from "./sourceListUtils"

type SearchDetailsPanelProps = {
  className?: string
}

type CandidateReasonCategory = {
  label: string
  className: string
}

function isReportedNumber(value: number | null | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value)
}

function formatPercent(value: number | null | undefined): string {
  if (!isReportedNumber(value)) return "Not reported"
  return `${Math.round(value * 100)}%`
}

function formatInteger(value: number | null | undefined): string {
  if (!isReportedNumber(value)) return "Not reported"
  return value.toLocaleString()
}

function formatScorePercent(value: number | null | undefined): string {
  if (!isReportedNumber(value)) return "Not reported"
  return `${Math.round(value * 100)}%`
}

function getWeakestIncludedScore(results: RagResult[]): number | null {
  const scores = results
    .map((result) => result.score)
    .filter((score): score is number => typeof score === "number" && Number.isFinite(score))
  if (scores.length === 0) return null
  return Math.min(...scores)
}

function buildEvidenceOriginParts(results: RagResult[]): string[] {
  if (results.length === 0) return []

  let localCount = 0
  let webCount = 0
  let mixedCount = 0
  let unknownCount = 0
  let unavailableCount = 0

  for (const result of results) {
    const origin = getEvidenceOrigin(result)
    if (origin === "local_library") localCount += 1
    else if (origin === "web_fallback") webCount += 1
    else if (origin === "mixed") mixedCount += 1
    else unknownCount += 1

    const status = getResultSourceStatus(result)
    if (getResultUnavailableReason(result) || (status && status !== "searched")) {
      unavailableCount += 1
    }
  }

  return [
    localCount > 0 ? `local library ${localCount}` : null,
    webCount > 0 ? `web fallback ${webCount}` : null,
    mixedCount > 0 ? `mixed ${mixedCount}` : null,
    unknownCount > 0 ? `unknown origin ${unknownCount}` : null,
    unavailableCount > 0 ? `unavailable ${unavailableCount}` : null,
  ].filter((value): value is string => Boolean(value))
}

function formatCandidateScoreContext(
  score: number | null,
  weakestIncludedScore: number | null
): string {
  if (score == null) return ""

  const scorePercent = formatScorePercent(score)
  if (weakestIncludedScore == null) {
    return ` (${scorePercent})`
  }

  const scoreDelta = score - weakestIncludedScore
  const deltaLabel = `${scoreDelta >= 0 ? "+" : ""}${Math.round(scoreDelta * 100)} pts vs weakest included`
  return ` (${scorePercent} • ${deltaLabel})`
}

function getCandidateReasonCategory(reason: string | null): CandidateReasonCategory | null {
  if (!reason) return null
  const normalized = reason.trim().toLowerCase()
  if (!normalized) return null

  if (/threshold|below|cutoff|minimum/.test(normalized)) {
    return {
      label: "Threshold",
      className: "border-warn/30 bg-warn/10 text-warn",
    }
  }
  if (/relevance|semantic|similarity|match/.test(normalized)) {
    return {
      label: "Low relevance",
      className: "border-warn/30 bg-warn/10 text-warn",
    }
  }
  if (/diversity|redundant|overlap|similar/.test(normalized)) {
    return {
      label: "Diversity",
      className: "border-primary/30 bg-primary/10 text-primary",
    }
  }
  if (/fresh|stale|recency|outdated|age|date/.test(normalized)) {
    return {
      label: "Freshness",
      className: "border-success/30 bg-success/10 text-success",
    }
  }
  if (/duplicate|dedup|already included/.test(normalized)) {
    return {
      label: "Deduplicated",
      className: "border-border bg-surface text-text-muted",
    }
  }
  if (/policy|safety|blocked|restricted|permission/.test(normalized)) {
    return {
      label: "Policy filter",
      className: "border-danger/30 bg-danger/10 text-danger",
    }
  }
  return {
    label: "Other",
    className: "border-border bg-surface text-text-muted",
  }
}

export function SearchDetailsPanel({ className }: SearchDetailsPanelProps) {
  const { searchDetails, isSearching, results = [] } = useKnowledgeQA()

  if (!searchDetails) {
    return null
  }

  const expandedQueries = Array.isArray(searchDetails.expandedQueries)
    ? searchDetails.expandedQueries
    : []
  const expansionTerms =
    expandedQueries.length > 0 ? expandedQueries.join(", ") : null
  const alsoConsidered = Array.isArray(searchDetails.alsoConsidered)
    ? searchDetails.alsoConsidered
    : []
  const weakestIncludedScore = getWeakestIncludedScore(results)
  const consideredCount = searchDetails.candidatesConsidered
  const returnedCount = searchDetails.candidatesReturned
  const consideredDocuments =
    searchDetails.documentsConsidered ?? searchDetails.candidatesConsidered
  const retainedPercent =
    isReportedNumber(consideredCount) &&
    consideredCount > 0 &&
    isReportedNumber(returnedCount)
      ? Math.round((returnedCount / consideredCount) * 100)
      : null
  const claimCount =
    searchDetails.faithfulnessTotalClaims ?? searchDetails.verificationTotalClaims
  const candidateParts: string[] = []
  if (isReportedNumber(consideredCount)) {
    candidateParts.push(formatInteger(consideredCount))
  }
  if (isReportedNumber(returnedCount)) {
    candidateParts.push(`returned ${formatInteger(returnedCount)}`)
  }
  if (isReportedNumber(searchDetails.candidatesRejected)) {
    candidateParts.push(`rejected ${formatInteger(searchDetails.candidatesRejected)}`)
  }
  if (retainedPercent != null) {
    candidateParts.push(`retained ${retainedPercent}%`)
  }
  const coverageParts: string[] = []
  if (isReportedNumber(consideredDocuments)) {
    coverageParts.push(`considered ${formatInteger(consideredDocuments)} documents`)
  }
  if (isReportedNumber(searchDetails.chunksConsidered)) {
    coverageParts.push(`${formatInteger(searchDetails.chunksConsidered)} chunks scanned`)
  }
  if (isReportedNumber(searchDetails.documentsReturned)) {
    coverageParts.push(`returned ${formatInteger(searchDetails.documentsReturned)} sources`)
  }
  const hasRetrievalCounts = candidateParts.length > 0 || coverageParts.length > 0
  const evidenceOriginParts = buildEvidenceOriginParts(results)

  return (
    <details
      className={cn("rounded-xl border border-border bg-muted/20", className)}
      aria-label="Search details"
    >
      <summary className="flex cursor-pointer list-none items-center gap-2 px-4 py-3 text-sm font-medium">
        <BarChart3 className="w-4 h-4 text-text-muted" />
        Search details
        {isSearching && (
          <span className="ml-auto text-xs text-text-muted">Updating...</span>
        )}
      </summary>

      <div className="grid gap-2 border-t border-border px-4 py-3 text-sm">
        {expansionTerms ? (
          <div>
            <span className="font-medium">Query expansion:</span> {expansionTerms}
          </div>
        ) : (
          <p className="text-text-muted">No query expansion terms were reported.</p>
        )}
        <div>
          <span className="font-medium">Reranking:</span>{" "}
          {searchDetails.rerankingEnabled
            ? searchDetails.rerankingStrategy
              ? `Enabled (${searchDetails.rerankingStrategy})`
              : "Enabled"
            : "Disabled"}
        </div>
        {isReportedNumber(searchDetails.averageRelevance) ? (
          <div>
            <span className="font-medium">Average relevance:</span>{" "}
            {formatPercent(searchDetails.averageRelevance)}
          </div>
        ) : null}
        <div>
          <span className="font-medium">Web fallback:</span>{" "}
          {searchDetails.webFallbackEnabled
            ? searchDetails.webFallbackTriggered
              ? `Triggered${searchDetails.webFallbackEngine ? ` (${searchDetails.webFallbackEngine})` : ""}`
              : "Enabled (not triggered)"
            : "Disabled"}
        </div>
        {evidenceOriginParts.length > 0 ? (
          <div>
            <span className="font-medium">Evidence origins:</span>{" "}
            {evidenceOriginParts.join(" • ")}
          </div>
        ) : null}
        {searchDetails.whyTheseSources && (
          <div>
            <span className="font-medium">Why these sources:</span>{" "}
            topicality {formatPercent(searchDetails.whyTheseSources.topicality)}, diversity{" "}
            {formatPercent(searchDetails.whyTheseSources.diversity)}, freshness{" "}
            {formatPercent(searchDetails.whyTheseSources.freshness)}
          </div>
        )}
        {(searchDetails.faithfulnessScore != null ||
          searchDetails.verificationReportAvailable) && (
          <div>
            <span className="font-medium">Verification:</span>{" "}
            faithfulness {formatPercent(searchDetails.faithfulnessScore)}
            {searchDetails.verificationRate != null
              ? `, verification rate ${formatPercent(searchDetails.verificationRate)}`
              : ""}
            {searchDetails.verificationCoverage != null
              ? `, claim coverage ${formatPercent(searchDetails.verificationCoverage)}`
              : ""}
            {claimCount != null
              ? `, claims ${formatInteger(claimCount)}`
              : ""}
            {searchDetails.faithfulnessSupportedClaims != null
              ? `, supported ${formatInteger(searchDetails.faithfulnessSupportedClaims)}`
              : ""}
            {searchDetails.faithfulnessUnsupportedClaims != null
              ? `, unsupported ${formatInteger(searchDetails.faithfulnessUnsupportedClaims)}`
              : ""}
          </div>
        )}
        {hasRetrievalCounts ? (
          <>
            {candidateParts.length > 0 ? (
              <div>
                <span className="font-medium">Candidates considered:</span>{" "}
                {candidateParts.join(" • ")}
              </div>
            ) : null}
            {coverageParts.length > 0 ? (
              <div>
                <span className="font-medium">Search coverage:</span>{" "}
                {coverageParts.join(" • ")}
              </div>
            ) : null}
          </>
        ) : (
          <p className="text-text-muted">
            No retrieval counts were reported for this search.
          </p>
        )}
        {searchDetails.retrievalLatencyMs != null ? (
          <div>
            <span className="font-medium">Retrieval latency:</span>{" "}
            {formatInteger(searchDetails.retrievalLatencyMs)} ms
          </div>
        ) : null}
        {alsoConsidered.length > 0 ? (
          <div>
            <p className="font-medium">Also considered (closest misses):</p>
            <ul className="mt-1 space-y-1 text-xs text-text-muted">
              {alsoConsidered.slice(0, 5).map((candidate) => {
                const reasonCategory = getCandidateReasonCategory(candidate.reason)
                return (
                  <li key={candidate.id}>
                    <span className="font-medium text-text">{candidate.title}</span>
                    {reasonCategory ? (
                      <span
                        className={cn(
                          "ml-1.5 inline-flex items-center rounded border px-1.5 py-0.5 text-[10px] font-medium",
                          reasonCategory.className
                        )}
                      >
                        {reasonCategory.label}
                      </span>
                    ) : null}
                    {formatCandidateScoreContext(candidate.score, weakestIncludedScore)}
                    {candidate.reason ? ` — ${candidate.reason}` : ""}
                  </li>
                )
              })}
            </ul>
          </div>
        ) : (
          <p className="text-text-muted">
            No closest-miss candidates were reported.
          </p>
        )}
      </div>
    </details>
  )
}
