import React, { useEffect, useId, useRef, useState } from "react"
import { Button } from "antd"
import { useTranslation } from "react-i18next"
import { getQuickIngestAnalysisProviderWarning } from "@/services/tldw/quick-ingest-batch"
import { AddContentStep } from "./AddContentStep"
import { WizardConfigureStep } from "./WizardConfigureStep"
import { useIngestWizard } from "./IngestWizardContext"
import type { WizardQueueItem } from "./types"

type WebIngestDraftProps = {
  replacementItems: WizardQueueItem[]
  processingItems: WizardQueueItem[]
  isOnlineForIngest: boolean
  isCheckingConnection: boolean
  connectionRecoveryMessage: string
  onRetryConnection: () => void
  onProcess: () => void
  visible: boolean
}

export function WebIngestDraft({
  processingItems, replacementItems, isOnlineForIngest, isCheckingConnection, connectionRecoveryMessage,
  onRetryConnection, onProcess, visible,
}: WebIngestDraftProps) {
  const { state } = useIngestWizard()
  const { presetConfig, queueItems } = state
  const { t } = useTranslation("option", { keyPrefix: "quickIngest.web" })
  const [expanded, setExpanded] = useState(state.currentStep === 2)
  const [reviewing, setReviewing] = useState(false)
  const requiresReplacementReview = replacementItems.length > 0
  const reviewHeading = useRef<HTMLHeadingElement>(null)
  const reviewTrigger = useRef<HTMLElement | null>(null)
  const showReview = reviewing && requiresReplacementReview
  const wasReviewing = useRef(false)
  useEffect(() => {
    if (showReview) reviewHeading.current?.focus()
    else if (wasReviewing.current) reviewTrigger.current?.focus({ preventScroll: true })
    wasReviewing.current = showReview
  }, [showReview])
  const settingsId = useId()
  const count = processingItems.length
  const providerMissing = Boolean(getQuickIngestAnalysisProviderWarning(presetConfig))
  const disabled = count === 0 || providerMissing || !isOnlineForIngest || isCheckingConnection
  const destination = presetConfig.reviewBeforeStorage
    ? t("draftDestination", "Review drafts in this browser")
    : presetConfig.storeRemote ? t("saveDestination", "Save to Media") : t("extractDestination", "Extract without saving")
  const action = presetConfig.reviewBeforeStorage
    ? (count === 1 ? t("createDraft", "Create 1 review draft") : t("createDrafts", { defaultValue: "Create {{count}} review drafts", count }))
    : presetConfig.storeRemote
      ? (count === 1 ? t("saveSource", "Save 1 source") : t("saveSources", { defaultValue: "Save {{count}} sources", count }))
      : (count === 1 ? t("extractSource", "Extract 1 source") : t("extractSources", { defaultValue: "Extract {{count}} sources", count }))
  const provider = String(presetConfig.advancedValues?.api_name || "").trim()
  const analysis = !presetConfig.common.perform_analysis
    ? t("analysisOff", "AI analysis off")
    : provider ? t("analysisWithProvider", { defaultValue: "AI analysis with {{provider}}", provider }) : t("providerNeeded", "AI analysis needs a provider")

  return <>
    <div className="qi-web-scroll" hidden={showReview}>
      <AddContentStep oneScreen processingItems={processingItems}
        isOnlineForIngest={isOnlineForIngest} isCheckingConnection={isCheckingConnection}
        connectionRecoveryMessage={connectionRecoveryMessage} onRetryConnection={onRetryConnection} />
      {queueItems.length > 0 && <section className="border-t border-border py-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <h3 className="text-sm font-semibold text-text">{destination}</h3>
            <p className="text-sm text-text-muted">{analysis}</p>
          </div>
          <Button type="link" aria-expanded={expanded} aria-controls={settingsId} onClick={() => setExpanded(value => !value)}>
            {expanded ? t("hideSettings", "Hide settings") : t("changeSettings", "Change settings")}
          </Button>
        </div>
        <div id={settingsId}>
          <WizardConfigureStep inline expanded={expanded} processingItems={processingItems} isStepVisible={visible && !showReview} />
        </div>
      </section>}
    </div>
    {showReview ? <>
      <section className="qi-web-scroll space-y-3" aria-label={t("replaceTitle", "Review replacements")}>
        <h3 ref={reviewHeading} tabIndex={-1} className="text-base font-semibold">{t("replaceTitle", "Review replacements")}</h3>
        <p className="text-sm text-text-muted">{t("replaceHelp", "Existing Media items that match these sources may be replaced.")}</p>
        <ul className="space-y-1 text-sm">{replacementItems.map(item =>
          <li className="break-words" key={item.id}>{item.fileName || item.url}</li>
        )}</ul>
        <p className="text-sm">{destination} · {analysis}</p>
      </section>
      <div className="qi-web-footer">
        <Button onClick={() => setReviewing(false)}>{t("editSources", "Back to sources")}</Button>
        <Button type="primary" disabled={disabled} onClick={onProcess}>
          {count === 1 ? t("confirmSave", "Confirm and save 1 source") : t("confirmSaves", { defaultValue: "Confirm and save {{count}} sources", count })}
        </Button>
      </div>
    </> : null}
    <div hidden={showReview} className={showReview ? undefined : "qi-web-footer qi-draft-footer"}>
      {count > 0 && providerMissing ? (
        <Button type="link" className="justify-self-start" onClick={event =>
          event.currentTarget.closest("[role=dialog]")?.querySelector<HTMLInputElement>("#quick-ingest-analysis-provider")?.focus()}>
          {t("chooseProvider", "Choose analysis provider")}
        </Button>
      ) : <span className="min-w-0 text-sm text-text-muted">
        {count === 0 ? t("addSourceHint", "Add a source to continue.")
          : queueItems.length > count ? t("readyOnly", "Only ready sources will run.")
            : presetConfig.reviewBeforeStorage ? t("reviewHint", "Review before saving to Media.") : null}
      </span>}
      <Button type="primary" disabled={disabled} onClick={requiresReplacementReview ? event => {
        reviewTrigger.current = event.currentTarget
        setReviewing(true)
      } : onProcess}>
        {requiresReplacementReview ? t("replaceTitle", "Review replacements") : action}
      </Button>
    </div>
  </>
}
