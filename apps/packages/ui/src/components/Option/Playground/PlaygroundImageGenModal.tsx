import React from "react"
import {
  Button,
  Input,
  InputNumber,
  Modal,
  Radio,
  Select
} from "antd"
import { WandSparkles } from "lucide-react"
import { ModalFooter } from "@/components/ui/layout"
import { getProviderDisplayName } from "@/utils/provider-registry"
import type { ReferenceImageCandidate } from "@/services/tldw/TldwApiClient"
import {
  normalizeImageGenerationEventSyncPolicy,
  normalizeImageGenerationEventSyncMode,
  type ImageGenerationEventSyncPolicy,
  type ImageGenerationEventSyncMode,
  type ImageGenerationPromptMode
} from "@/utils/image-generation-chat"
import { toText } from "./hooks"

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type ImagePromptStrategy = {
  id: string
  label: string
}

export type ImagePromptContextEntry = {
  id: string
  label: string
  text: string
  score: number
}

export type ImagePromptRefineDiff = {
  overlapRatio: number
  addedHighlights: string[]
  removedHighlights: string[]
}

export type ImageBackendOption = {
  value: string
  label: string
  provider?: string
}

export interface PlaygroundImageGenModalProps {
  open: boolean
  onClose: () => void
  busy: boolean

  // Backend
  backend: string
  backendOptions: ImageBackendOption[]
  onBackendChange: (value: string) => void
  onHydrateSettings: (backend: string) => void

  // Prompt mode
  promptMode: ImageGenerationPromptMode
  onPromptModeChange: (mode: ImageGenerationPromptMode) => void
  promptStrategies: ImagePromptStrategy[]

  // Sync policy
  syncPolicy: ImageGenerationEventSyncPolicy
  onSyncPolicyChange: (policy: ImageGenerationEventSyncPolicy) => void
  syncChatMode: ImageGenerationEventSyncMode
  onSyncChatModeChange: (mode: ImageGenerationEventSyncMode) => void
  syncGlobalDefault: ImageGenerationEventSyncMode
  onSyncGlobalDefaultChange: (mode: ImageGenerationEventSyncMode) => void
  resolvedSyncMode: ImageGenerationEventSyncMode

  // Prompt
  prompt: string
  onPromptChange: (value: string) => void
  contextBreakdown: ImagePromptContextEntry[]
  onClearRefineState: () => void

  // Refine
  refineSubmitting: boolean
  refineBaseline: string
  refineCandidate: string | null
  refineModel: string | null
  refineLatencyMs: number | null
  refineDiff: ImagePromptRefineDiff | null
  onCreateDraft: () => void
  onRefine: () => void
  onApplyRefined: () => void
  onRejectRefined: () => void

  // Generation settings
  format: "png" | "jpg" | "webp"
  onFormatChange: (value: "png" | "jpg" | "webp") => void
  width: number | undefined
  onWidthChange: (value: number | undefined) => void
  height: number | undefined
  onHeightChange: (value: number | undefined) => void
  steps: number | undefined
  onStepsChange: (value: number | undefined) => void
  cfgScale: number | undefined
  onCfgScaleChange: (value: number | undefined) => void
  seed: number | undefined
  onSeedChange: (value: number | undefined) => void
  sampler: string
  onSamplerChange: (value: string) => void
  model: string
  onModelChange: (value: string) => void
  negativePrompt: string
  onNegativePromptChange: (value: string) => void
  extraParams: string
  onExtraParamsChange: (value: string) => void
  referenceFileId?: number
  onReferenceFileIdChange: (value: number | undefined) => void
  referenceImageCandidates: ReferenceImageCandidate[]
  referenceImageCandidatesLoading: boolean

  // Submit
  submitting: boolean
  onSubmit: () => void

  t: (key: string, defaultValue?: string, options?: any) => any
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const toNum = (value: number | null | undefined): number | undefined =>
  typeof value === "number" && Number.isFinite(value) ? value : undefined

const formatReferenceImageLabel = (
  candidate: ReferenceImageCandidate
): string => {
  const parts = [candidate.title]
  if (
    typeof candidate.width === "number" &&
    Number.isFinite(candidate.width) &&
    typeof candidate.height === "number" &&
    Number.isFinite(candidate.height)
  ) {
    parts.push(`${candidate.width}x${candidate.height}`)
  }
  if (candidate.mime_type) {
    parts.push(candidate.mime_type)
  }
  return parts.join(" · ")
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export const PlaygroundImageGenModal: React.FC<PlaygroundImageGenModalProps> =
  React.memo(function PlaygroundImageGenModal(props) {
    const {
      open,
      onClose,
      busy,
      backend,
      backendOptions,
      onBackendChange,
      onHydrateSettings,
      promptMode,
      onPromptModeChange,
      promptStrategies,
      syncPolicy,
      onSyncPolicyChange,
      syncChatMode,
      onSyncChatModeChange,
      syncGlobalDefault,
      onSyncGlobalDefaultChange,
      resolvedSyncMode,
      prompt,
      onPromptChange,
      contextBreakdown,
      onClearRefineState,
      refineSubmitting,
      refineBaseline,
      refineCandidate,
      refineModel,
      refineLatencyMs,
      refineDiff,
      onCreateDraft,
      onRefine,
      onApplyRefined,
      onRejectRefined,
      format,
      onFormatChange,
      width,
      onWidthChange,
      height,
      onHeightChange,
      steps,
      onStepsChange,
      cfgScale,
      onCfgScaleChange,
      seed,
      onSeedChange,
      sampler,
      onSamplerChange,
      model,
      onModelChange,
      negativePrompt,
      onNegativePromptChange,
      extraParams,
      onExtraParamsChange,
      referenceFileId,
      onReferenceFileIdChange,
      referenceImageCandidates,
      referenceImageCandidatesLoading,
      submitting,
      onSubmit,
      t
    } = props

    return (
      <Modal
        open={open}
        onCancel={() => {
          if (busy) return
          onClose()
        }}
        title={t("playground:imageGeneration.modalTitle", "Generate image")}
        width={720}
        destroyOnHidden
        footer={
          <ModalFooter
            align="between"
            data-testid="image-generation-modal-footer"
            leftActions={[
              {
                label: t(
                  "playground:imageGeneration.createPrompt",
                  "Create prompt"
                ),
                onClick: onCreateDraft,
                icon: <WandSparkles className="h-4 w-4" />,
                disabled: busy
              },
              {
                label: t(
                  "playground:imageGeneration.refineWithLlm",
                  "Refine with LLM"
                ),
                onClick: () => {
                  void onRefine()
                },
                loading: busy || refineSubmitting,
                disabled: busy || submitting,
                "data-testid": "image-refine-with-llm"
              }
            ]}
            hideCancel
            actions={[
              {
                label: t("common:cancel", "Cancel"),
                onClick: onClose,
                disabled: busy,
                variant: "ghost"
              }
            ]}
            primaryAction={{
              label: t(
                "playground:imageGeneration.generateNow",
                "Generate image"
              ),
              onClick: () => {
                void onSubmit()
              },
              loading: submitting,
              disabled: busy || refineSubmitting
            }}
          />
        }
      >
        <div className="space-y-4">
          <div className="grid gap-3 sm:grid-cols-3">
            <div className="space-y-1">
              <label className="text-xs font-medium text-text-muted">
                {t("playground:imageGeneration.backendLabel", "Backend")}
              </label>
              <Select
                value={backend || undefined}
                data-testid="image-generate-backend-select"
                options={backendOptions.map((option) => ({
                  value: option.value,
                  label: option.provider
                    ? `${getProviderDisplayName(option.provider)} · ${option.label}`
                    : option.label
                }))}
                onChange={(value) => {
                  const next = String(value || "")
                  onBackendChange(next)
                  void onHydrateSettings(next)
                }}
                placeholder={t(
                  "playground:imageGeneration.backendPlaceholder",
                  "Select backend"
                )}
                disabled={busy}
              />
            </div>
            <div className="space-y-1">
              <label className="text-xs font-medium text-text-muted">
                {t(
                  "playground:imageGeneration.promptModeLabel",
                  "Prompt mode"
                )}
              </label>
              <Radio.Group
                optionType="button"
                value={promptMode}
                onChange={(event) =>
                  onPromptModeChange(
                    event.target.value as ImageGenerationPromptMode
                  )
                }
                disabled={busy}
              >
                {promptStrategies.map((strategy) => (
                  <Radio.Button key={strategy.id} value={strategy.id}>
                    {strategy.label}
                  </Radio.Button>
                ))}
              </Radio.Group>
            </div>
            <div className="space-y-1">
              <label className="text-xs font-medium text-text-muted">
                {t(
                  "playground:imageGeneration.syncPolicyLabel",
                  "Server sync"
                )}
              </label>
              <Select
                value={syncPolicy}
                data-testid="image-generate-sync-policy-select"
                options={[
                  {
                    value: "inherit",
                    label: t(
                      "playground:imageGeneration.syncPolicyInherit",
                      "Inherit defaults"
                    )
                  },
                  {
                    value: "on",
                    label: t(
                      "playground:imageGeneration.syncPolicyOn",
                      "Mirror event"
                    )
                  },
                  {
                    value: "off",
                    label: t(
                      "playground:imageGeneration.syncPolicyOff",
                      "Local only"
                    )
                  }
                ]}
                onChange={(value) =>
                  onSyncPolicyChange(
                    normalizeImageGenerationEventSyncPolicy(value, "inherit")
                  )
                }
                disabled={busy}
              />
            </div>
          </div>
          <div className="space-y-1">
            <label className="text-xs font-medium text-text-muted">
              {t(
                "playground:imageGeneration.referenceImageLabel",
                "Reference image (optional)"
              )}
            </label>
            <Select
              value={referenceFileId ?? undefined}
              data-testid="image-generate-reference-image-select"
              allowClear
              loading={referenceImageCandidatesLoading}
              placeholder={t(
                "playground:imageGeneration.referenceImagePlaceholder",
                "Select a managed reference image"
              )}
              options={referenceImageCandidates.map((candidate) => ({
                value: candidate.file_id,
                label: formatReferenceImageLabel(candidate)
              }))}
              onChange={(value) =>
                onReferenceFileIdChange(
                  typeof value === "number" ? value : undefined
                )
              }
              disabled={busy}
            />
            <p className="text-[11px] text-text-muted">
              {t(
                "playground:imageGeneration.referenceImageHint",
                "The list is provided by managed reference images on this server."
              )}
            </p>
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="space-y-1">
              <label className="text-xs font-medium text-text-muted">
                {t(
                  "playground:imageGeneration.syncChatDefault",
                  "Chat default"
                )}
              </label>
              <Select
                value={syncChatMode}
                data-testid="image-generate-chat-default-select"
                options={[
                  {
                    value: "off",
                    label: t(
                      "playground:imageGeneration.syncChatDefaultOff",
                      "Off (local only)"
                    )
                  },
                  {
                    value: "on",
                    label: t(
                      "playground:imageGeneration.syncChatDefaultOn",
                      "On (mirror events)"
                    )
                  }
                ]}
                onChange={(value) => {
                  const next = normalizeImageGenerationEventSyncMode(
                    value,
                    "off"
                  )
                  void onSyncChatModeChange(next)
                }}
                disabled={busy}
              />
            </div>
            <div className="space-y-1">
              <label className="text-xs font-medium text-text-muted">
                {t(
                  "playground:imageGeneration.syncGlobalDefault",
                  "Global default"
                )}
              </label>
              <Select
                value={normalizeImageGenerationEventSyncMode(
                  syncGlobalDefault,
                  "off"
                )}
                data-testid="image-generate-global-default-select"
                options={[
                  {
                    value: "off",
                    label: t(
                      "playground:imageGeneration.syncGlobalDefaultOff",
                      "Off (local only)"
                    )
                  },
                  {
                    value: "on",
                    label: t(
                      "playground:imageGeneration.syncGlobalDefaultOn",
                      "On (mirror events)"
                    )
                  }
                ]}
                onChange={(value) => {
                  const next = normalizeImageGenerationEventSyncMode(
                    value,
                    "off"
                  )
                  void onSyncGlobalDefaultChange(next)
                }}
                disabled={busy}
              />
            </div>
          </div>
          <p className="text-[11px] text-text-muted">
            {resolvedSyncMode === "on"
              ? t(
                  "playground:imageGeneration.syncEffectiveOn",
                  "Effective policy: this generation event will also be mirrored to server chat history."
                )
              : t(
                  "playground:imageGeneration.syncEffectiveOff",
                  "Effective policy: this generation event stays local-only and does not mirror to server chat history."
                )}
          </p>
          <div className="space-y-1">
            <label className="text-xs font-medium text-text-muted">
              {t("playground:imageGeneration.promptLabel", "Prompt")}
            </label>
            <Input.TextArea
              value={prompt}
              onChange={(event) => {
                onPromptChange(event.target.value)
                onClearRefineState()
              }}
              autoSize={{ minRows: 4, maxRows: 8 }}
              disabled={busy}
              placeholder={t(
                "playground:imageGeneration.promptPlaceholder",
                "Describe the image you want to generate."
              )}
            />
            <p className="text-[11px] text-text-muted">
              {t(
                "playground:imageGeneration.promptHint",
                "Create prompt drafts from current chat context, then edit before generating."
              )}
            </p>
            {contextBreakdown.length > 0 && (
              <div
                className="rounded-md border border-border/70 bg-surface2/60 px-2 py-2 text-[11px] text-text-muted"
                data-testid="image-prompt-context-breakdown"
              >
                <div className="mb-1 font-medium text-text">
                  {t(
                    "playground:imageGeneration.contextBlendLabel",
                    "Weighted context blend"
                  )}
                </div>
                <div className="flex flex-wrap gap-1">
                  {contextBreakdown.map((entry) => (
                    <span
                      key={`${entry.id}-${entry.score}`}
                      className="inline-flex items-center rounded-full border border-border px-2 py-0.5"
                      title={entry.text}
                    >
                      {entry.label} {Math.round(entry.score * 100)}%
                    </span>
                  ))}
                </div>
              </div>
            )}
            {refineCandidate && (
              <div
                className="rounded-md border border-primary/30 bg-primary/10 px-3 py-3"
                data-testid="image-prompt-refine-diff"
              >
                <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                  <div className="text-xs font-semibold uppercase tracking-wide text-primaryStrong">
                    {t(
                      "playground:imageGeneration.refineCandidateTitle",
                      "Refined prompt candidate"
                    )}
                  </div>
                  <div className="flex flex-wrap items-center gap-1 text-[11px] text-primaryStrong">
                    {refineModel ? (
                      <span className="rounded-full border border-primary/30 bg-surface px-2 py-0.5">
                        {refineModel}
                      </span>
                    ) : null}
                    {refineLatencyMs != null ? (
                      <span className="rounded-full border border-primary/30 bg-surface px-2 py-0.5">
                        {toText(
                          t(
                            "playground:imageGeneration.refineLatency",
                            "{{ms}} ms",
                            { ms: refineLatencyMs } as any
                          )
                        )}
                      </span>
                    ) : null}
                    {refineDiff ? (
                      <span className="rounded-full border border-primary/30 bg-surface px-2 py-0.5">
                        {toText(
                          t(
                            "playground:imageGeneration.refineOverlap",
                            "{{percent}}% overlap",
                            {
                              percent: Math.round(
                                refineDiff.overlapRatio * 100
                              )
                            } as any
                          )
                        )}
                      </span>
                    ) : null}
                  </div>
                </div>
                <div className="grid gap-2 sm:grid-cols-2">
                  <div className="space-y-1">
                    <div className="text-[11px] font-medium text-text-muted">
                      {t(
                        "playground:imageGeneration.refineOriginalLabel",
                        "Original draft"
                      )}
                    </div>
                    <Input.TextArea
                      value={refineBaseline}
                      autoSize={{ minRows: 3, maxRows: 6 }}
                      readOnly
                    />
                  </div>
                  <div className="space-y-1">
                    <div className="text-[11px] font-medium text-text-muted">
                      {t(
                        "playground:imageGeneration.refineCandidateLabel",
                        "Refined prompt"
                      )}
                    </div>
                    <Input.TextArea
                      value={refineCandidate}
                      autoSize={{ minRows: 3, maxRows: 6 }}
                      readOnly
                    />
                  </div>
                </div>
                {refineDiff && (
                  <div className="mt-2 grid gap-2 sm:grid-cols-2">
                    <div className="space-y-1">
                      <div className="text-[11px] font-medium text-success">
                        {t(
                          "playground:imageGeneration.refineAdded",
                          "Added"
                        )}
                      </div>
                      <div className="space-y-1 text-[11px] text-text-muted">
                        {refineDiff.addedHighlights.length > 0 ? (
                          refineDiff.addedHighlights.map((entry, index) => (
                            <div
                              key={`image-refine-added-${index}`}
                              className="rounded border border-success/40 bg-success/10 px-2 py-1"
                            >
                              {entry}
                            </div>
                          ))
                        ) : (
                          <div className="rounded border border-border/70 bg-surface2/50 px-2 py-1">
                            {t(
                              "playground:imageGeneration.refineNoAdded",
                              "No added segments"
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                    <div className="space-y-1">
                      <div className="text-[11px] font-medium text-danger">
                        {t(
                          "playground:imageGeneration.refineRemoved",
                          "Removed"
                        )}
                      </div>
                      <div className="space-y-1 text-[11px] text-text-muted">
                        {refineDiff.removedHighlights.length > 0 ? (
                          refineDiff.removedHighlights.map(
                            (entry, index) => (
                              <div
                                key={`image-refine-removed-${index}`}
                                className="rounded border border-danger/40 bg-danger/10 px-2 py-1"
                              >
                                {entry}
                              </div>
                            )
                          )
                        ) : (
                          <div className="rounded border border-border/70 bg-surface2/50 px-2 py-1">
                            {t(
                              "playground:imageGeneration.refineNoRemoved",
                              "No removed segments"
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                )}
                <div className="mt-3 flex flex-wrap justify-end gap-2">
                  <Button onClick={onRejectRefined}>
                    {t(
                      "playground:imageGeneration.refineKeepOriginal",
                      "Keep original"
                    )}
                  </Button>
                  <Button
                    type="primary"
                    onClick={onApplyRefined}
                    data-testid="image-refine-accept"
                  >
                    {t(
                      "playground:imageGeneration.refineAccept",
                      "Apply refined prompt"
                    )}
                  </Button>
                </div>
              </div>
            )}
          </div>
          <div className="grid gap-3 sm:grid-cols-3">
            <div className="space-y-1">
              <label className="text-xs font-medium text-text-muted">
                {t("playground:imageGeneration.formatLabel", "Format")}
              </label>
              <Select
                value={format}
                options={[
                  { value: "png", label: "PNG" },
                  { value: "jpg", label: "JPG" },
                  { value: "webp", label: "WEBP" }
                ]}
                onChange={(value) =>
                  onFormatChange(value as "png" | "jpg" | "webp")
                }
                disabled={busy}
              />
            </div>
            <div className="space-y-1">
              <label className="text-xs font-medium text-text-muted">
                {t("playground:imageGeneration.widthLabel", "Width")}
              </label>
              <InputNumber
                value={width}
                min={64}
                step={64}
                style={{ width: "100%" }}
                onChange={(value) => onWidthChange(toNum(value))}
                disabled={busy}
              />
            </div>
            <div className="space-y-1">
              <label className="text-xs font-medium text-text-muted">
                {t("playground:imageGeneration.heightLabel", "Height")}
              </label>
              <InputNumber
                value={height}
                min={64}
                step={64}
                style={{ width: "100%" }}
                onChange={(value) => onHeightChange(toNum(value))}
                disabled={busy}
              />
            </div>
            <div className="space-y-1">
              <label className="text-xs font-medium text-text-muted">
                {t("playground:imageGeneration.stepsLabel", "Steps")}
              </label>
              <InputNumber
                value={steps}
                min={1}
                style={{ width: "100%" }}
                onChange={(value) => onStepsChange(toNum(value))}
                disabled={busy}
              />
            </div>
            <div className="space-y-1">
              <label className="text-xs font-medium text-text-muted">
                {t("playground:imageGeneration.cfgScaleLabel", "CFG scale")}
              </label>
              <InputNumber
                value={cfgScale}
                min={0}
                step={0.5}
                style={{ width: "100%" }}
                onChange={(value) => onCfgScaleChange(toNum(value))}
                disabled={busy}
              />
            </div>
            <div className="space-y-1">
              <label className="text-xs font-medium text-text-muted">
                {t("playground:imageGeneration.seedLabel", "Seed")}
              </label>
              <InputNumber
                value={seed}
                style={{ width: "100%" }}
                onChange={(value) => onSeedChange(toNum(value))}
                disabled={busy}
              />
            </div>
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="space-y-1">
              <label className="text-xs font-medium text-text-muted">
                {t("playground:imageGeneration.samplerLabel", "Sampler")}
              </label>
              <Input
                value={sampler}
                onChange={(event) => onSamplerChange(event.target.value)}
                disabled={busy}
              />
            </div>
            <div className="space-y-1">
              <label className="text-xs font-medium text-text-muted">
                {t("playground:imageGeneration.modelLabel", "Image model")}
              </label>
              <Input
                value={model}
                onChange={(event) => onModelChange(event.target.value)}
                disabled={busy}
              />
            </div>
          </div>
          <div className="space-y-1">
            <label className="text-xs font-medium text-text-muted">
              {t(
                "playground:imageGeneration.negativePromptLabel",
                "Negative prompt"
              )}
            </label>
            <Input.TextArea
              value={negativePrompt}
              onChange={(event) =>
                onNegativePromptChange(event.target.value)
              }
              autoSize={{ minRows: 2, maxRows: 4 }}
              disabled={busy}
            />
          </div>
          <div className="space-y-1">
            <label className="text-xs font-medium text-text-muted">
              {t(
                "playground:imageGeneration.extraParamsLabel",
                "Extra params (JSON object)"
              )}
            </label>
            <Input.TextArea
              value={extraParams}
              onChange={(event) => onExtraParamsChange(event.target.value)}
              autoSize={{ minRows: 3, maxRows: 6 }}
              disabled={busy}
              placeholder='{"tiling": false}'
            />
          </div>
        </div>
      </Modal>
    )
  })
