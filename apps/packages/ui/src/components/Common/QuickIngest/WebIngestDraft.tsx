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
  const { state, setCustomOptions } = useIngestWizard()
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
  const connectionBlocked = !isOnlineForIngest || isCheckingConnection
  const disabled = count === 0 || providerMissing || connectionBlocked
  const submitButton = useRef<HTMLButtonElement>(null)
  const focusAfterSkip = useRef(false)
  useEffect(() => {
    if (focusAfterSkip.current) {
      focusAfterSkip.current = false
      if (!disabled) submitButton.current?.focus()
    }
  }, [disabled])
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

  const blockerId = settingsId + "-blocker"
  const actionLabel = t("importActions", "Import actions")
  const remedies = connectionBlocked ? <div className="flex min-w-0 flex-wrap items-center gap-x-3">
    <span id={blockerId} role="status" className="text-sm text-text">
      {isCheckingConnection
        ? t("checkingServer", "Checking server connection…")
        : t("title", { keyPrefix: "quickIngest.wizard.offline", defaultValue: "Server offline" })}
    </span>
    <Button type="link" className="!px-0" disabled={isCheckingConnection} onClick={onRetryConnection}>
      {t("retry", { keyPrefix: "quickIngest.wizard.offline", defaultValue: "Retry connection" })}
    </Button>
  </div> : count > 0 && providerMissing ? <div className="min-w-0">
    <p id={blockerId} className="text-sm text-text">{analysis}</p>
    <div className="flex flex-wrap items-center gap-x-4">
      <Button type="link" className="!px-0" onClick={event => {
        event.currentTarget.closest("[role=dialog]")?.querySelector<HTMLInputElement>("#quick-ingest-analysis-provider")?.focus()
      }}>{t("chooseProvider", "Choose provider")}</Button>
      <Button type="link" className="!px-0" onClick={() => {
        focusAfterSkip.current = true
        setCustomOptions({ common: { ...presetConfig.common, perform_analysis: false } })
      }}>{t("skipAnalysis", { keyPrefix: "quickIngest.oneScreen", defaultValue: "Skip analysis for this import" })}</Button>
    </div>
  </div> : null

  return <>
    <div className="qi-web-scroll" hidden={showReview}>
      <AddContentStep oneScreen processingItems={processingItems}
        isOnlineForIngest={isOnlineForIngest} isCheckingConnection={isCheckingConnection}
        connectionRecoveryMessage={connectionRecoveryMessage} />
      {queueItems.length > 0 && <section className="border-t border-border py-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <h3 className="text-sm font-semibold text-text">{destination}</h3>
            {!providerMissing && <p className="text-sm text-text-muted">{analysis}</p>}
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
      <section className="qi-web-scroll" aria-labelledby={settingsId + "-review"}>
        <h3 id={settingsId + "-review"} ref={reviewHeading} tabIndex={-1} className="text-xl font-semibold text-text">
          {t("replaceTitle", "Review replacements")}
        </h3>
        <p className="mt-2 text-base font-medium text-text">
          {count === 1 ? t("oneSourceWillRun", "1 source will run") : t("sourcesWillRun", { defaultValue: "{{count}} sources will run", count })}
        </p>
        <p className="mt-1 text-sm text-text-muted">
          {replacementItems.length === 1
            ? t("onePossibleReplacement", "1 existing Media item may be replaced.")
            : t("possibleReplacements", { defaultValue: "Up to {{count}} existing Media items may be replaced.", count: replacementItems.length })}
        </p>
        <ul className="my-4 divide-y divide-border border-y border-border">
          {replacementItems.map(item => {
            const source = item.url || item.fileName
            const title = item.conferenceOverride?.title || item.playlist?.sourceTitle
            return <li key={item.id} className="space-y-1 py-3">
              <p className="break-words font-medium text-text">{title || source}</p>
              {title && source && title !== source && <p className="break-words text-sm text-text-muted">{source}</p>}
              <p className="text-xs text-text-muted">{item.playlist?.duplicateStatus === "duplicate_existing"
                ? t("existingMatch", "Match found in Media")
                : t("replaceIfMatched", "Replaces an item only if a match is found")}</p>
            </li>
          })}
        </ul>
        {count > replacementItems.length && <p className="mb-3 text-sm text-text">
          {count - replacementItems.length === 1
            ? t("oneOtherSource", "The other source will run without replacing an item.")
            : t("otherSources", { defaultValue: "The other {{count}} sources will run without replacing an item.", count: count - replacementItems.length })}
        </p>}
        <p className="text-sm text-text-muted">{destination} · {analysis}</p>
      </section>
      <div className="qi-web-footer qi-replacement-footer" role="group" aria-label={actionLabel}>
        {connectionBlocked && remedies}
        <Button onClick={() => setReviewing(false)}>{t("editSources", "Back to sources")}</Button>
        <Button type="primary" disabled={disabled} aria-describedby={connectionBlocked ? blockerId : undefined} onClick={onProcess}>
          {count === 1 ? t("confirmSave", "Confirm and save 1 source") : t("confirmSaves", { defaultValue: "Confirm and save {{count}} sources", count })}
        </Button>
      </div>
    </> : null}
    <div hidden={showReview} className={showReview ? undefined : "qi-web-footer qi-draft-footer"} role="group" aria-label={actionLabel}>
      {(!showReview && remedies) || <span className="min-w-0 text-sm text-text-muted">
        {count === 0 && queueItems.some(item => !item.validation.valid)
          ? t("repairSourceHint", "Repair a queued source to continue.")
          : count === 0 ? t("addSourceHint", "Add a source to continue.")
          : queueItems.length > count ? t("readyOnly", "Only ready sources will run.")
            : presetConfig.reviewBeforeStorage ? t("reviewHint", "Review before saving to Media.") : null}
      </span>}
      <Button ref={submitButton} type="primary" disabled={disabled} aria-describedby={remedies && !showReview ? blockerId : undefined} onClick={requiresReplacementReview ? event => {
        reviewTrigger.current = event.currentTarget
        setReviewing(true)
      } : onProcess}>
        {requiresReplacementReview ? t("replaceTitle", "Review replacements") : action}
      </Button>
    </div>
  </>
}
