import React from "react"
import { useTranslation } from "react-i18next"
import type { GeneratedArtifact } from "@/types/workspace"
import { downloadBlob } from "@/utils/download-blob"
import { buildSectionReviewPatch, getSectionReviewEntry, renderSectionReviewExport, type SectionReviewDraft, type SectionReviewTarget } from "@/utils/section-artifact-review"

export type SaveSectionArtifactReview = (artifact: GeneratedArtifact, patch: Pick<GeneratedArtifact, "reviewStatus" | "reviewMetadata">) => Promise<GeneratedArtifact>

/** Editorial corrections remain separate from the original generated artifact. */
export const SectionArtifactReviewPanel: React.FC<{
  artifact: GeneratedArtifact
  target: SectionReviewTarget
  onSave?: SaveSectionArtifactReview
  onSaved: (artifact: GeneratedArtifact) => void
}> = ({ artifact, target, onSave, onSaved }) => {
  const { t } = useTranslation("playground")
  const fieldId = React.useId()
  const [drafts, setDrafts] = React.useState<Record<string, SectionReviewDraft>>({})
  const [saving, setSaving] = React.useState(false)
  const [error, setError] = React.useState<{ targetId: string; message: string } | null>(null)
  const [savedTarget, setSavedTarget] = React.useState<string | null>(null)
  const entry = getSectionReviewEntry(artifact, target)
  const draft = drafts[target.id] ?? { status: artifact.reviewStatus ?? "draft", correction: entry?.correction ?? "", reason: entry?.reason ?? "" }
  const change = (patch: Partial<SectionReviewDraft>) => {
    setDrafts(previous => ({ ...previous, [target.id]: { ...draft, ...patch } }))
    setSavedTarget(null)
  }
  const save = async () => {
    if (!onSave || saving) return
    setSaving(true); setError(null); setSavedTarget(null)
    try {
      const patch = buildSectionReviewPatch(artifact, target, draft)
      const saved = await onSave(artifact, patch)
      onSaved(saved)
      setDrafts(previous => { const next = { ...previous }; delete next[target.id]; return next })
      setSavedTarget(target.id)
    } catch (failure) {
      setError({ targetId: target.id, message: failure instanceof Error ? failure.message : "The review could not be saved. Your draft is retained." })
    } finally { setSaving(false) }
  }
  const fieldClass = "min-h-10 w-full rounded border border-border bg-surface px-3 py-2 text-sm text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-primary"

  return <section className="space-y-3 rounded border border-border p-3" aria-label={t("studio.sectionReview.heading", "Review and corrections")}>
    <h3 className="font-semibold">{t("studio.sectionReview.heading", "Review and corrections")}</h3>
    <p className="text-xs text-text-muted">{t("studio.sectionReview.scope", "The model draft and exact source evidence stay unchanged. Corrections below are separate reviewer notes for the selected overview or section.")}</p>
    {entry?.stale && <p role="status" className="text-warning">{t("studio.sectionReview.stale", "The original target changed. Check this retained correction against the latest draft before saving.")}</p>}
    {entry?.correction && <div className="rounded bg-surface2 p-3 whitespace-pre-wrap text-sm"><strong>{t("studio.sectionReview.savedCorrection", "Saved reviewer correction")}</strong><p>{entry.correction}</p></div>}
    <label htmlFor={`${fieldId}-status`} className="block text-sm font-medium">{t("studio.sectionReview.status", "Review status for this output")}</label>
    <select id={`${fieldId}-status`} value={draft.status} disabled={saving || !onSave} onChange={event => change({ status: event.target.value as SectionReviewDraft["status"] })} className={fieldClass}>
      <option value="draft">{t("studio.sectionReview.draft", "Draft")}</option>
      <option value="reviewing">{t("studio.sectionReview.reviewing", "Reviewing")}</option>
      <option value="needs_revision">{t("studio.sectionReview.needsRevision", "Needs revision")}</option>
      <option value="accepted">{t("studio.sectionReview.accepted", "Accepted by reviewer")}</option>
      <option value="rejected">{t("studio.sectionReview.rejected", "Rejected")}</option>
    </select>
    <label htmlFor={`${fieldId}-correction`} className="block text-sm font-medium">{t("studio.sectionReview.correction", "Correction for this section")}</label>
    <textarea id={`${fieldId}-correction`} value={draft.correction} rows={5} maxLength={20000} disabled={saving || !onSave} onChange={event => change({ correction: event.target.value })} className={fieldClass} />
    <p className="text-xs text-text-muted">{t("studio.sectionReview.citations", "Keep supporting evidence IDs, such as [E1.1.1], when correcting a claim. This form checks IDs, not whether the claim is supported.")}</p>
    <label htmlFor={`${fieldId}-reason`} className="block text-sm font-medium">{t("studio.sectionReview.reason", "Reason or review note")}</label>
    <textarea id={`${fieldId}-reason`} value={draft.reason} rows={2} maxLength={4000} disabled={saving || !onSave} onChange={event => change({ reason: event.target.value })} className={fieldClass} />
    {error?.targetId === target.id && <p role="alert" className="text-sm text-error">{error.message}</p>}
    {savedTarget === target.id && <p role="status" className="text-sm">{t("studio.sectionReview.saved", "Review saved. The original draft is retained.")}</p>}
    <div className="flex flex-wrap gap-3">
      <button type="button" disabled={saving || !onSave} onClick={() => void save()} className="min-h-10 rounded border border-primary px-3 py-2 text-sm text-primary disabled:opacity-50">{saving ? t("studio.sectionReview.saving", "Saving review…") : t("studio.sectionReview.save", "Save review")}</button>
      <button type="button" onClick={() => downloadBlob(new Blob([renderSectionReviewExport(artifact)], { type: "text/markdown;charset=utf-8" }), `${artifact.title}-review.md`)} className="min-h-10 text-sm text-primary underline">{t("studio.sectionReview.download", "Download draft with saved review")}</button>
    </div>
  </section>
}
