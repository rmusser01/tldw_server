import React from "react"
import { useTranslation } from "react-i18next"
import type { GeneratedArtifact } from "@/types/workspace"
import { getArtifactRunIdentity } from "./output-identity"

/** Compact saved identity shared by an output card and its reader. */
export function ArtifactRunIdentity({ artifact }: { artifact: GeneratedArtifact }) {
  const { t } = useTranslation("playground")
  const identity = getArtifactRunIdentity(artifact)
  return <div className="space-y-1 break-words text-xs text-text-muted">
    {identity.imported && <p>{t("studio.outputIdentity.imported", "Imported output")}</p>}
    <p>
      {identity.model ? t("studio.outputIdentity.model", { defaultValue: "Recorded model: {{model}}", model: identity.model })
        : t("studio.outputIdentity.modelMissing", "Model not recorded")}
      {" · "}
      {identity.provider ? t("studio.outputIdentity.provider", { defaultValue: "Provider: {{provider}}", provider: identity.provider })
        : t("studio.outputIdentity.providerMissing", "Provider not recorded")}
    </p>
    <p>{identity.createdAt ? <>
      {t("studio.outputIdentity.created", "Created")}{" "}
      <time dateTime={identity.createdAt.toISOString()}>{identity.createdAt.toLocaleString()}</time>
    </> : t("studio.outputIdentity.createdMissing", "Creation time not recorded")}</p>
    <p>{identity.sectionCount === undefined
      ? t("studio.outputIdentity.scopeMissing", "Section scope not recorded")
      : <>{t("studio.outputIdentity.sections", {
        defaultValue_one: "{{count}} selected section", defaultValue_other: "{{count}} selected sections", count: identity.sectionCount
      })}{identity.sectionTitles.length > 0 && <>: {identity.sectionTitles.join("; ")}
        {identity.sectionCount > identity.sectionTitles.length && "…"}</>}</>}
    </p>
  </div>
}

/** A measured example, not a prediction for the current job. */
export function LongDocumentWaitingGuidance() {
  const { t } = useTranslation("playground")
  return <div className="space-y-1 text-xs text-text-muted">
    <p>{t("studio.longDocumentWaiting.example", "In observed 922-page, 30-chapter runs before citation checks were added, generation took 52–62 minutes.")}</p>
    <p>{t("studio.longDocumentWaiting.variation", "This is not an estimate for this run. Duration varies with document length, selected sections, model and settings. Progress is saved after each portion.")}</p>
    <p>{t("studio.longDocumentWaiting.readCompleted", "Read completed chapters while the remaining sections are being generated.")}</p>
    <p>{t("studio.longDocumentWaiting.usage", "Token usage and cost are unavailable for these saved jobs.")}</p>
  </div>
}
