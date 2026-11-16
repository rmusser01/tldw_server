import { sha256 } from "@noble/hashes/sha2.js"
import { bytesToHex, utf8ToBytes } from "@noble/hashes/utils.js"
import type { ArtifactReviewStatus, GeneratedArtifact } from "@/types/workspace"

const record = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === "object" && !Array.isArray(value)
const text = (value: unknown) => typeof value === "string" ? value : ""
const reviewStates = new Set<ArtifactReviewStatus>(["draft", "reviewing", "needs_revision", "accepted", "rejected"])

export interface SectionReviewTarget {
  id: string
  title: string
  originalText: string
  fingerprint: string
  evidenceIds: string[]
}
export interface SectionReviewEntry {
  targetId: string
  title: string
  targetFingerprint: string
  correction: string
  reason: string
  evidenceIds: string[]
  savedAt: string
  provenance: "user_review"
}
export interface SectionReviewDraft { status: ArtifactReviewStatus; correction: string; reason: string }

/** Bind an editorial overlay to the original target without rewriting generated evidence. */
export const getSectionReviewTarget = (artifact: GeneratedArtifact, selection: string): SectionReviewTarget | null => {
  const client = artifact.versionMetadata?.workspaceClient
  const restored = record(client) && record(client.data) ? client.data : undefined
  const analysis = artifact.data?.sectionAnalysis ?? restored?.sectionAnalysis
  if (!record(analysis) || !Array.isArray(analysis.sections)) return null
  const sections = analysis.sections.filter(record)
  const index = Number(selection)
  const section = selection === "overview" ? undefined : Number.isSafeInteger(index) && index >= 0 ? sections[index] : undefined
  if (selection !== "overview" && !section) return null
  const originalText = text(section ? section.content : analysis.synthesis)
  if (!originalText) return null
  const title = section ? text(section.title) : "Overview"
  const id = section ? `section:${index}` : "overview"
  const evidenceIds = (section ? [section] : sections).flatMap(item => Array.isArray(item.batches)
    ? item.batches.filter(record).flatMap(batch => Array.isArray(batch.evidence)
      ? batch.evidence.filter(record).map(evidence => text(evidence.id)).filter(Boolean) : []) : [])
  const fingerprint = bytesToHex(sha256(utf8ToBytes(JSON.stringify([analysis.fingerprint, id, section?.node_id, section?.navigation_version, originalText]))))
  return { id, title, originalText, fingerprint, evidenceIds: [...new Set(evidenceIds)] }
}

const entries = (artifact: GeneratedArtifact): SectionReviewEntry[] => {
  const review = artifact.reviewMetadata?.sectionReview
  if (!record(review) || review.schemaVersion !== 1 || !Array.isArray(review.entries)) return []
  return review.entries.filter((entry): entry is unknown & SectionReviewEntry => record(entry) &&
    typeof entry.targetId === "string" && typeof entry.title === "string" &&
    typeof entry.targetFingerprint === "string" && typeof entry.correction === "string" &&
    typeof entry.reason === "string" && typeof entry.savedAt === "string" && entry.provenance === "user_review" &&
    Array.isArray(entry.evidenceIds) && entry.evidenceIds.every(id => typeof id === "string"))
}

export const getSectionReviewEntry = (artifact: GeneratedArtifact, target: SectionReviewTarget): (SectionReviewEntry & { stale: boolean }) | undefined => {
  const entry = entries(artifact).find(item => item.targetId === target.id)
  return entry ? { ...entry, stale: entry.targetFingerprint !== target.fingerprint } : undefined
}

export const buildSectionReviewPatch = (artifact: GeneratedArtifact, target: SectionReviewTarget, draft: SectionReviewDraft): Pick<GeneratedArtifact, "reviewStatus" | "reviewMetadata"> => {
  const savedReview = artifact.reviewMetadata?.sectionReview
  if (savedReview !== undefined && (!record(savedReview) || savedReview.schemaVersion !== 1 ||
      !Array.isArray(savedReview.entries) || entries(artifact).length !== savedReview.entries.length))
    throw new Error("The saved review format is unsupported. Export the workspace before editing this review.")
  if (!reviewStates.has(draft.status)) throw new Error("Choose a valid review status.")
  if (draft.correction.length > 20000 || draft.reason.length > 4000) throw new Error("Keep the correction under 20,000 characters and the reason under 4,000 characters.")
  const cited = [...new Set(draft.correction.match(/\bE\d+\.\d+\.\d+\b/g) ?? [])]
  if (cited.some(id => !target.evidenceIds.includes(id))) throw new Error("A cited evidence ID is not available in this selected section.")
  const prior = entries(artifact).filter(entry => entry.targetId !== target.id)
  if (prior.length >= 100) throw new Error("This output already has 100 reviewed targets.")
  const entry: SectionReviewEntry = { targetId: target.id, title: target.title, targetFingerprint: target.fingerprint,
    correction: draft.correction.trim(), reason: draft.reason.trim(), evidenceIds: cited,
    savedAt: new Date().toISOString(), provenance: "user_review" }
  return { reviewStatus: draft.status, reviewMetadata: { ...artifact.reviewMetadata,
    sectionReview: { schemaVersion: 1, entries: [...prior, entry] } } }
}

/** Export the unchanged draft plus clearly separated reviewer amendments. */
export const renderSectionReviewExport = (artifact: GeneratedArtifact): string => {
  const lines = [artifact.content ?? "", "\n# Reviewer corrections", `Review status: ${artifact.reviewStatus ?? "draft"}.`,
    "The model draft and source evidence above are unchanged. Review status records a reviewer decision, not automatic verification."]
  for (const entry of entries(artifact)) {
    const selection = entry.targetId === "overview" ? "overview" : entry.targetId.replace(/^section:/, "")
    const target = getSectionReviewTarget(artifact, selection)
    lines.push(`## ${entry.title}`, `Saved: ${entry.savedAt}; original target fingerprint: ${entry.targetFingerprint}.`)
    if (!target || target.fingerprint !== entry.targetFingerprint) lines.push("**Original target changed: this retained correction needs review.**")
    if (entry.reason) lines.push(`Review note: ${entry.reason}`)
    if (entry.correction) lines.push("### Reviewer correction", entry.correction)
  }
  return lines.join("\n\n")
}
