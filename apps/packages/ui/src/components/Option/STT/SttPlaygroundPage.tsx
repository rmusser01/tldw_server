import React, { useCallback, useEffect, useMemo, useState } from "react"
import { useStorage } from "@plasmohq/storage/hook"
import { Trans, useTranslation } from "react-i18next"
import { Link } from "react-router-dom"
import { Alert, Button, Typography } from "antd"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { PageShell } from "@/components/Common/PageShell"
import { useAntdNotification } from "@/hooks/useAntdNotification"
import { useTranscriptionModelsCatalog } from "@/hooks/useTranscriptionModelsCatalog"
import { RecordingStrip } from "./RecordingStrip"
import { InlineSettingsPanel } from "./InlineSettingsPanel"
import type { SttLocalSettings } from "./InlineSettingsPanel"
import { ComparisonPanel } from "./ComparisonPanel"
import { HistoryPanel } from "./HistoryPanel"
import type { SttHistoryEntry, SttHistoryResult } from "./HistoryPanel"
import type { ComparisonResult } from "@/hooks/useComparisonTranscribe"
import { AudioReadinessStrip } from "@/components/Option/Audio/AudioReadinessStrip"
import { AudioPresetControls } from "@/components/Option/Audio/AudioPresetControls"
import { buildSttReadinessItems } from "@/components/Option/Audio/audio-readiness"
import {
  saveSttRecording,
  getSttRecording,
  deleteSttRecording
} from "@/db/dexie/stt-recordings"

const { Text, Title } = Typography

export const SttPlaygroundPage: React.FC = () => {
  const { t } = useTranslation(["playground"])
  const notification = useAntdNotification()
  const [globalModel] = useStorage("sttModel", "whisper-1")

  // ── Server models (fetched on mount) ──────────────────────────────
  const {
    serverModels,
    modelOptions,
    serverModelsLoading,
    serverModelsError,
    retryServerModels
  } = useTranscriptionModelsCatalog({
    defaultModel: globalModel,
    warnLabel: "STT Playground"
  })

  // ── Current blob from RecordingStrip ──────────────────────────────
  const [currentBlob, setCurrentBlob] = useState<Blob | null>(null)
  const [currentDurationMs, setCurrentDurationMs] = useState<number>(0)

  const handleBlobReady = useCallback((blob: Blob, durationMs: number) => {
    setCurrentBlob(blob)
    setCurrentDurationMs(durationMs)
  }, [])

  // ── Settings ──────────────────────────────────────────────────────
  const [globalLanguage] = useStorage("speechToTextLanguage", "en-US")
  const [globalTask] = useStorage("sttTask", "transcribe")
  const [globalFormat] = useStorage("sttResponseFormat", "json")
  const [globalTemperature] = useStorage("sttTemperature", 0)
  const [globalPrompt] = useStorage("sttPrompt", "")
  const [globalUseSegmentation] = useStorage("sttUseSegmentation", false)
  const [globalSegK] = useStorage("sttSegK", 6)
  const [globalSegMinSegmentSize] = useStorage("sttSegMinSegmentSize", 5)
  const [globalSegLambdaBalance] = useStorage("sttSegLambdaBalance", 0.01)
  const [globalSegUtteranceExpansionWidth] = useStorage(
    "sttSegUtteranceExpansionWidth",
    2
  )
  const [globalSegEmbeddingsProvider] = useStorage(
    "sttSegEmbeddingsProvider",
    ""
  )
  const [globalSegEmbeddingsModel] = useStorage("sttSegEmbeddingsModel", "")

  const defaultSttSettings = useMemo<SttLocalSettings>(
    () => ({
      language: globalLanguage,
      task: globalTask,
      responseFormat: globalFormat,
      temperature: globalTemperature,
      prompt: globalPrompt,
      useSegmentation: globalUseSegmentation,
      segK: globalSegK,
      segMinSegmentSize: globalSegMinSegmentSize,
      segLambdaBalance: globalSegLambdaBalance,
      segUtteranceExpansionWidth: globalSegUtteranceExpansionWidth,
      segEmbeddingsProvider: globalSegEmbeddingsProvider,
      segEmbeddingsModel: globalSegEmbeddingsModel
    }),
    [
      globalFormat,
      globalLanguage,
      globalPrompt,
      globalSegEmbeddingsModel,
      globalSegEmbeddingsProvider,
      globalSegK,
      globalSegLambdaBalance,
      globalSegMinSegmentSize,
      globalSegUtteranceExpansionWidth,
      globalTask,
      globalTemperature,
      globalUseSegmentation
    ]
  )

  const [sttSettings, setSttSettings] = useState<SttLocalSettings>(
    defaultSttSettings
  )
  const [showSettings, setShowSettings] = useState(false)
  const [selectedSttModels, setSelectedSttModels] = useState<string[]>([])

  useEffect(() => {
    if (!showSettings) {
      setSttSettings(defaultSttSettings)
    }
  }, [defaultSttSettings, showSettings])

  const toggleSettings = useCallback(() => {
    setShowSettings((prev) => !prev)
  }, [])

  const sttOptions = useMemo(() => {
    const opts: Record<string, unknown> = {}
    if (sttSettings.language) opts.language = sttSettings.language
    if (sttSettings.task) opts.task = sttSettings.task
    if (sttSettings.responseFormat)
      opts.response_format = sttSettings.responseFormat
    if (typeof sttSettings.temperature === "number")
      opts.temperature = sttSettings.temperature
    if (sttSettings.prompt) opts.prompt = sttSettings.prompt
    if (sttSettings.useSegmentation) {
      opts.segment = true
      if (typeof sttSettings.segK === "number") opts.seg_K = sttSettings.segK
      if (typeof sttSettings.segMinSegmentSize === "number")
        opts.seg_min_segment_size = sttSettings.segMinSegmentSize
      if (typeof sttSettings.segLambdaBalance === "number")
        opts.seg_lambda_balance = sttSettings.segLambdaBalance
      if (typeof sttSettings.segUtteranceExpansionWidth === "number")
        opts.seg_utterance_expansion_width =
          sttSettings.segUtteranceExpansionWidth
      if (sttSettings.segEmbeddingsProvider)
        opts.seg_embeddings_provider = sttSettings.segEmbeddingsProvider
      if (sttSettings.segEmbeddingsModel)
        opts.seg_embeddings_model = sttSettings.segEmbeddingsModel
    }
    return opts
  }, [sttSettings])

  const sttPresetConfig = useMemo(
    () => ({
      models: selectedSttModels,
      ...sttOptions
    }),
    [selectedSttModels, sttOptions]
  )

  const handleApplySttPreset = useCallback(
    (config: Record<string, unknown>) => {
      const nextSettings: SttLocalSettings = { ...sttSettings }
      if (typeof config.language === "string") nextSettings.language = config.language
      if (typeof config.task === "string") nextSettings.task = config.task
      if (typeof config.response_format === "string") {
        nextSettings.responseFormat = config.response_format
      }
      if (typeof config.temperature === "number") {
        nextSettings.temperature = config.temperature
      }
      if (typeof config.prompt === "string") nextSettings.prompt = config.prompt
      if (typeof config.segment === "boolean") nextSettings.useSegmentation = config.segment
      if (typeof config.seg_K === "number") nextSettings.segK = config.seg_K
      if (typeof config.seg_min_segment_size === "number") {
        nextSettings.segMinSegmentSize = config.seg_min_segment_size
      }
      if (typeof config.seg_lambda_balance === "number") {
        nextSettings.segLambdaBalance = config.seg_lambda_balance
      }
      if (typeof config.seg_utterance_expansion_width === "number") {
        nextSettings.segUtteranceExpansionWidth = config.seg_utterance_expansion_width
      }
      if (typeof config.seg_embeddings_provider === "string") {
        nextSettings.segEmbeddingsProvider = config.seg_embeddings_provider
      }
      if (typeof config.seg_embeddings_model === "string") {
        nextSettings.segEmbeddingsModel = config.seg_embeddings_model
      }
      const rawModels = config.models
      const hasExplicitModelList = Array.isArray(rawModels)
      const models = hasExplicitModelList
        ? rawModels
            .map((model) => (typeof model === "string" ? model.trim() : ""))
            .filter((model) => model.length > 0)
        : typeof config.model === "string" && config.model.trim()
          ? [config.model.trim()]
          : []
      if (hasExplicitModelList || models.length > 0) setSelectedSttModels(models)
      setSttSettings(nextSettings)
      setShowSettings(true)
    },
    [sttSettings]
  )

  const readinessItems = useMemo(
    () =>
      buildSttReadinessItems({
        modelOptions,
        loading: serverModelsLoading,
        error: serverModelsError
      }),
    [modelOptions, serverModelsError, serverModelsLoading]
  )

  // ── History (persisted via Plasmo storage) ────────────────────────
  const [history, setHistory] = useStorage<SttHistoryEntry[]>(
    "sttComparisonHistory",
    []
  )

  // ── Callbacks ─────────────────────────────────────────────────────

  const handleComparisonComplete = useCallback(
    async (compResults: ComparisonResult[]) => {
      if (!currentBlob || compResults.length === 0) return
      try {
        const recordingId = await saveSttRecording({
          blob: currentBlob,
          durationMs: currentDurationMs,
          mimeType: currentBlob.type || "audio/webm"
        })
        const historyResults: SttHistoryResult[] = compResults
          .filter((r) => r.status === "done")
          .map((r) => ({
            model: r.model,
            text: r.text,
            latencyMs: r.latencyMs,
            wordCount: r.wordCount,
            config: r.config,
            metadata: r.metadata
          }))
        if (historyResults.length === 0) return
        const entry: SttHistoryEntry = {
          id: `${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
          recordingId,
          createdAt: new Date().toISOString(),
          durationMs: currentDurationMs,
          results: historyResults
        }
        setHistory((prev) => [entry, ...(prev ?? [])].slice(0, 20))
      } catch (e) {
        console.error("Failed to save comparison to history", e)
      }
    },
    [currentBlob, currentDurationMs, setHistory]
  )

  const handleSaveToNotes = useCallback(
    async (text: string, model: string) => {
      const title = `STT Comparison: ${model} - ${new Date().toLocaleString()}`
      try {
        await tldwClient.createNote(text, {
          title,
          metadata: {
            origin: "stt-playground",
            stt_model: model
          }
        })
        notification.success({
          message: t("playground:stt.savedToNotes", "Saved to Notes"),
          description: t("playground:stt.savedToNotesDesc", "Transcription saved as a note.")
        })
      } catch (e: unknown) {
        notification.error({
          message: t("error", "Error"),
          description: e instanceof Error ? e.message : t("somethingWentWrong", "Something went wrong")
        })
      }
    },
    [notification, t]
  )

  const handleRecompare = useCallback(
    async (entry: SttHistoryEntry) => {
      try {
        const recording = await getSttRecording(entry.recordingId)
        if (recording) {
          setCurrentBlob(recording.blob)
          setCurrentDurationMs(recording.durationMs ?? entry.durationMs ?? 0)
        } else {
          notification.error({
            message: t("playground:stt.recordingNotFound", "Recording not found"),
            description: t("playground:stt.recordingNotFoundDesc",
              "The audio recording was not found in local storage. It may have been deleted.")
          })
        }
      } catch (e: unknown) {
        notification.error({
          message: t("error", "Error"),
          description: e instanceof Error ? e.message : t("playground:stt.loadFailed", "Failed to load recording")
        })
      }
    },
    [notification, t]
  )

  const handleExport = useCallback(
    async (entry: SttHistoryEntry) => {
      const lines = [
        `# STT Comparison - ${new Date(entry.createdAt).toLocaleString()}`,
        "",
        `Duration: ${entry.durationMs ? (entry.durationMs / 1000).toFixed(1) + "s" : "unknown"}`,
        ""
      ]
      if (entry.results) {
        for (const result of entry.results) {
          lines.push(`## ${result.model}`)
          lines.push("")
          lines.push(result.text || "(no text)")
          lines.push("")
        }
      }
      const markdown = lines.join("\n")
      try {
        await navigator.clipboard.writeText(markdown)
        notification.success({
          message: t("playground:stt.exported", "Copied"),
          description: t("playground:stt.exportedDesc", "Comparison results copied to clipboard as Markdown.")
        })
      } catch {
        notification.error({
          message: t("playground:stt.exportFailed", "Copy failed"),
          description: t("playground:stt.exportFailedDesc", "Unable to copy to clipboard.")
        })
      }
    },
    [notification, t]
  )

  const handleDeleteEntry = useCallback(
    async (id: string) => {
      const entry = (history ?? []).find((e) => e.id === id)
      if (entry) {
        try {
          await deleteSttRecording(entry.recordingId)
        } catch {
          // Dexie record may already be gone; proceed with removal
        }
      }
      setHistory((prev) => (prev ?? []).filter((e) => e.id !== id))
      notification.info({
        message: t("playground:stt.deleted", "Deleted"),
        description: t("playground:stt.deletedDesc", "History entry removed.")
      })
    },
    [history, setHistory, notification, t]
  )

  const handleClearAll = useCallback(async () => {
    const entries = history ?? []
    for (const entry of entries) {
      try {
        await deleteSttRecording(entry.recordingId)
      } catch {
        // best-effort cleanup
      }
    }
    setHistory([])
  }, [history, setHistory])

  // ── Keyboard shortcuts ────────────────────────────────────────────
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.code !== "Space") return
      const tag = (e.target as HTMLElement)?.tagName?.toLowerCase()
      const isEditable = (e.target as HTMLElement)?.isContentEditable
      if (
        tag === "input" ||
        tag === "textarea" ||
        tag === "select" ||
        tag === "button" ||
        isEditable
      ) {
        return
      }
      e.preventDefault()
      window.dispatchEvent(new CustomEvent("stt-toggle-record"))
    }
    window.addEventListener("keydown", handleKeyDown)
    return () => window.removeEventListener("keydown", handleKeyDown)
  }, [])

  // ── Render ────────────────────────────────────────────────────────
  return (
    <PageShell maxWidthClassName="max-w-5xl" className="py-6">
      <Title level={1} className="!text-2xl">
        {t("playground:speechToText", "Speech to Text")}
      </Title>
      <Text type="secondary">
        {t("playground:stt.subtitle", "Record audio and compare transcription results across multiple models.")}
      </Text>
      <p className="text-[11px] text-text-subtle mt-1">
        {t("playground:stt.serverRequirement", "Transcription requires a configured STT engine on your tldw server.")}
      </p>
      <p className="text-[11px] text-text-subtle">
        <Trans
          i18nKey="playground:stt.combinedWorkflowHint"
          defaults="For combined TTS + STT workflows, try the <speechLink>Speech Playground</speechLink>."
          components={{ speechLink: <Link to="/speech" className="underline" /> }}
        />
      </p>
      <div className="mt-1">
        <a
          href="https://github.com/rmusser01/tldw_server/blob/main/Docs/Getting_Started/First_Time_Audio_Setup_CPU.md"
          target="_blank"
          rel="noopener noreferrer"
          className="text-xs text-text-muted underline hover:text-text"
        >
          {t("playground:stt.audioSetupGuide", "Audio Setup Guide")}
        </a>
      </div>

      <div className="mt-4 space-y-4">
        <AudioReadinessStrip items={readinessItems} label="STT readiness" />
        <AudioPresetControls
          kind="stt"
          currentConfig={sttPresetConfig}
          capabilityAssumptions={{
            models: modelOptions
              .filter((model) => selectedSttModels.includes(model.id))
              .map((model) => ({
                id: model.id,
                availability: model.availability,
                capabilities: model.capabilities,
                sources: model.sources
              }))
          }}
          onApply={handleApplySttPreset}
        />
        {!currentBlob && (
          <p className="text-center text-sm text-text-muted mb-4">
            {t("playground:stt.firstUseHint", "Press the record button or upload an audio file to get started with transcription.")}
          </p>
        )}
        {serverModelsError && (
          <Alert
            type="warning"
            showIcon
            title={t("playground:stt.modelsLoadError", "Model load failed")}
            description={serverModelsError}
            action={
              <Button
                size="small"
                onClick={() => {
                  retryServerModels()
                }}
                disabled={serverModelsLoading}
              >
                {t("common:retry", "Retry")}
              </Button>
            }
          />
        )}
        {!serverModelsLoading && !serverModelsError && serverModels.length === 0 && (
          <Alert
            type="warning"
            showIcon
            className="mb-4"
            message={t("playground:stt.noModelsTitle", "No transcription models available")}
            description={t("playground:stt.noModelsBody", "Configure STT models in your server settings. Check the Audio Setup Guide for instructions.")}
          />
        )}
        <div data-testid="stt-record-strip">
          <RecordingStrip
            onBlobReady={handleBlobReady}
            onSettingsToggle={toggleSettings}
          />
        </div>
        <p className="text-[11px] text-text-subtle text-center mt-1">
          <Trans
            i18nKey="playground:stt.spaceKeyHint"
            defaults="Press <kbd>Space</kbd> to toggle recording"
            components={{ kbd: <kbd className="rounded border border-border px-1 py-0.5 text-[10px]" /> }}
          />
        </p>
        {showSettings && <div data-testid="stt-settings-panel"><InlineSettingsPanel onChange={setSttSettings} /></div>}
        <div data-testid="stt-transcription-output">
          <ComparisonPanel
            blob={currentBlob}
            availableModels={serverModels}
            availableModelOptions={modelOptions}
            selectedModels={selectedSttModels}
            onSelectedModelsChange={setSelectedSttModels}
            sttOptions={sttOptions}
            onSaveToNotes={handleSaveToNotes}
            onComparisonComplete={handleComparisonComplete}
          />
        </div>
        <HistoryPanel
          entries={history ?? []}
          onRecompare={handleRecompare}
          onExport={handleExport}
          onDelete={handleDeleteEntry}
          onClearAll={handleClearAll}
        />
      </div>
    </PageShell>
  )
}

export default SttPlaygroundPage
