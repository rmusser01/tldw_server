import { useState, useEffect, useMemo, useRef, useCallback } from 'react'
import { Modal, Button, Select, Input, Spin } from 'antd'
import { useStorage } from '@plasmohq/storage/hook'
import { useTranslation } from 'react-i18next'
import { bgRequest, bgStream } from '@/services/background-proxy'
import { tldwModels } from '@/services/tldw'
import { ANALYSIS_PRESETS } from "@/components/Media/analysisPresets"
import { useAntdMessage } from "@/hooks/useAntdMessage"
import { createSafeStorage } from "@/utils/safe-storage"
import { resolveApiProviderForModel } from "@/utils/resolve-api-provider"
import { DEFAULT_ANALYSIS_SUMMARY_PROMPT } from "@/utils/default-prompts"
import { extractMediaDetailContent } from "@/utils/media-detail-content"

interface AnalysisTimeoutConfig {
  chatRequestTimeoutMs?: number
  requestTimeoutMs?: number
  chatStreamIdleTimeoutMs?: number
  streamIdleTimeoutMs?: number
}

interface SavedAnalysisPrompts {
  systemPrompt?: string
  userPrefix?: string
}

interface AnalysisModalProps {
  open: boolean
  onClose: () => void
  mediaId: string | number
  mediaContent: string
  onAnalysisGenerated?: (analysisText: string, prompt?: string) => void
}

const MIN_ANALYSIS_TIMEOUT_MS = 120_000
const MIN_ANALYSIS_STREAM_IDLE_TIMEOUT_MS = 120_000
const LOCAL_TIMEOUT_STORAGE = createSafeStorage({ area: 'local' })

const toPositiveNumber = (value: unknown): number => {
  const num = Number(value)
  return Number.isFinite(num) && num > 0 ? num : 0
}

const firstNonEmptyString = (...vals: unknown[]): string => {
  for (const v of vals) {
    if (typeof v === 'string' && v.trim().length > 0) return v
  }
  return ''
}

const normalizePersistedModelSelection = (value: unknown): string => {
  if (typeof value !== 'string') return ''
  const trimmed = value.trim()
  if (!trimmed) return ''
  try {
    const parsed = JSON.parse(trimmed)
    return typeof parsed === 'string' ? parsed.trim() : trimmed
  } catch {
    return trimmed
  }
}

const extractStreamDelta = (chunk: string): string | null => {
  if (!chunk) return null
  let payload = chunk.trim()
  if (!payload) return null
  if (payload.startsWith("data:")) payload = payload.slice(5).trim()
  if (!payload || payload === "[DONE]") return null
  try {
    const parsed = JSON.parse(payload)
    const delta =
      parsed?.choices?.[0]?.delta?.content ??
      parsed?.choices?.[0]?.message?.content ??
      parsed?.choices?.[0]?.text ??
      parsed?.content
    return typeof delta === "string" ? delta : null
  } catch {
    return null
  }
}

export function AnalysisModal({
  open,
  onClose,
  mediaId,
  mediaContent,
  onAnalysisGenerated
}: AnalysisModalProps) {
  const { t } = useTranslation(['review', 'common'])
  const messageApi = useAntdMessage()
  const [selectedModel, setSelectedModel] = useStorage<string | undefined>('selectedModel')
  const [models, setModels] = useState<Array<{ id: string; name?: string }>>([])
  const [systemPrompt, setSystemPrompt] = useState(DEFAULT_ANALYSIS_SUMMARY_PROMPT)
  const [userPrefix, setUserPrefix] = useState('')
  const [generating, setGenerating] = useState(false)
  const [showPresets, setShowPresets] = useState(false)
  const [elapsedSeconds, setElapsedSeconds] = useState(0)
  const [analysisPreview, setAnalysisPreview] = useState("")
  const [recoveredMediaContent, setRecoveredMediaContent] = useState("")
  const [cancelledGeneration, setCancelledGeneration] = useState(false)
  const activeAbortControllerRef = useRef<AbortController | null>(null)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const cancelledByUserRef = useRef(false)

  const buildTimeoutMessage = () => {
    const summary = t('common:error.friendlyTimeoutSummary', 'Your chat timed out.')
    const hint = t(
      'common:error.friendlyTimeoutHint',
      'The server stopped streaming responses. Try again, or open Health & diagnostics to check server status.'
    )
    return `${summary} ${hint}`.trim()
  }

  const isTimeoutError = (err: unknown) => {
    const msg =
      err instanceof Error
        ? err.message
        : typeof err === "string"
          ? err
          : ""
    const lowered = msg.toLowerCase()
    return (
      lowered.includes("timeout") ||
      lowered.includes("timed out")
    )
  }

  const isAbortError = (err: unknown) => {
    if (err instanceof Error && err.name === 'AbortError') return true
    const msg =
      err instanceof Error
        ? err.message
        : typeof err === 'string'
          ? err
          : ''
    return msg.toLowerCase().includes('abort')
  }

  const presets = useMemo(
    () =>
      ANALYSIS_PRESETS.map((preset) => ({
        name: t(preset.nameKey, preset.nameDefault),
        system: preset.systemPrompt,
        user: preset.userPrefix ?? ''
      })),
    [t]
  )
  const normalizedSelectedModel = useMemo(
    () => normalizePersistedModelSelection(selectedModel),
    [selectedModel]
  )
  const selectedModelKey = useMemo(() => {
    if (!normalizedSelectedModel) return undefined
    return normalizedSelectedModel.startsWith("tldw:")
      ? normalizedSelectedModel
      : `tldw:${normalizedSelectedModel}`
  }, [normalizedSelectedModel])
  const effectiveMediaContent = mediaContent.trim()
    ? mediaContent
    : recoveredMediaContent

  useEffect(() => {
    if (!open) return
    if (mediaContent.trim()) {
      setRecoveredMediaContent(mediaContent)
      return
    }

    let cancelled = false
    setRecoveredMediaContent("")
    ;(async () => {
      try {
        const detail = await bgRequest<any>({
          path: `/api/v1/media/${mediaId}?include_content=true&include_versions=false&cache_bust=${Date.now()}` as any,
          method: 'GET' as any
        })
        if (!cancelled) {
          setRecoveredMediaContent(extractMediaDetailContent(detail))
        }
      } catch {
        // Keep generation disabled when the selected media body is unavailable.
      }
    })()

    return () => {
      cancelled = true
    }
  }, [mediaContent, mediaId, open])

  useEffect(() => {
    if (!open) {
      setAnalysisPreview("")
      setCancelledGeneration(false)
    }
  }, [open])

  const clearGenerationTimer = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }
  }, [])

  const handleCancelGeneration = useCallback((showMessage = true) => {
    cancelledByUserRef.current = true
    try {
      activeAbortControllerRef.current?.abort()
    } catch {}
    clearGenerationTimer()
    setGenerating(false)
    setElapsedSeconds(0)
    setAnalysisPreview('')
    setCancelledGeneration(true)
    if (showMessage) {
      messageApi.info(t('mediaPage.analysisGenerationCancelled', 'Analysis generation cancelled'))
    }
  }, [clearGenerationTimer, messageApi, t])

  useEffect(() => {
    return () => {
      try {
        activeAbortControllerRef.current?.abort()
      } catch {}
      clearGenerationTimer()
    }
  }, [clearGenerationTimer])

  const handleModalClose = useCallback(() => {
    if (generating) {
      handleCancelGeneration(false)
    }
    onClose()
  }, [generating, handleCancelGeneration, onClose])

  const getAnalysisTimeouts = async () => {
    try {
      const cfg = await LOCAL_TIMEOUT_STORAGE
        .get<AnalysisTimeoutConfig>('tldwConfig')
        .catch(() => null)
      const configuredRequest =
        toPositiveNumber(cfg?.chatRequestTimeoutMs) ||
        toPositiveNumber(cfg?.requestTimeoutMs)
      const configuredStreamIdle =
        toPositiveNumber(cfg?.chatStreamIdleTimeoutMs) ||
        toPositiveNumber(cfg?.streamIdleTimeoutMs)
      return {
        requestTimeoutMs: Math.max(configuredRequest, MIN_ANALYSIS_TIMEOUT_MS),
        streamIdleTimeoutMs: Math.max(configuredStreamIdle, MIN_ANALYSIS_STREAM_IDLE_TIMEOUT_MS)
      }
    } catch {
      return {
        requestTimeoutMs: MIN_ANALYSIS_TIMEOUT_MS,
        streamIdleTimeoutMs: MIN_ANALYSIS_STREAM_IDLE_TIMEOUT_MS
      }
    }
  }

  // Load models from tldw_server
  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const chatModels = await tldwModels.getChatModels()
        const allModels = chatModels.map((m) => ({
          id: m.id.startsWith("tldw:") ? m.id : `tldw:${m.id}`,
          name: m.name || m.id
        }))
        if (!cancelled) {
          setModels(allModels || [])
        }
      } catch (err) {
        console.warn('Failed to load models:', err)
        if (!cancelled) {
          setModels([])
        }
      }
    })()

    return () => {
      cancelled = true
    }
  }, [])

  // Load saved prompts from storage
  useEffect(() => {
    if (open) {
      let cancelled = false
      ;(async () => {
        try {
          const storage = createSafeStorage({ area: 'local' })
          const data = await storage.get<SavedAnalysisPrompts>('media:analysisPrompts').catch(() => null)
          if (!cancelled && data && typeof data === 'object') {
            if (typeof data.systemPrompt === 'string') setSystemPrompt(data.systemPrompt)
            if (typeof data.userPrefix === 'string') setUserPrefix(data.userPrefix)
          }
        } catch (err) {
          console.warn('Failed to load saved prompts:', err)
        }
      })()

      return () => {
        cancelled = true
      }
    }
  }, [open])

  const handleSaveAsDefault = async () => {
    try {
      const storage = createSafeStorage({ area: 'local' })
      await storage.set('media:analysisPrompts', { systemPrompt, userPrefix })
      messageApi.success(t('mediaPage.savedAsDefault', 'Saved as default prompts'))
    } catch {
      messageApi.error(t('mediaPage.savePromptsFailed', 'Failed to save prompts'))
    }
  }

  const handleGenerate = async () => {
    if (!effectiveMediaContent || !effectiveMediaContent.trim()) {
      messageApi.warning(t('mediaPage.noContentForAnalysis', 'No content available for analysis'))
      return
    }

    const effectiveModel = selectedModelKey || models[0]?.id
    if (!effectiveModel) {
      messageApi.warning(
        t(
          'mediaPage.noModelSelected',
          'Select a model before generating analysis'
        )
      )
      return
    }
    if (generating) return
    const normalizedModel = effectiveModel.replace(/^tldw:/, "").trim()
    const resolvedApiProvider = await resolveApiProviderForModel({
      modelId: effectiveModel
    })

    cancelledByUserRef.current = false
    setCancelledGeneration(false)
    const abortController = new AbortController()
    activeAbortControllerRef.current = abortController
    setGenerating(true)
    setElapsedSeconds(0)
    setAnalysisPreview("")
    const startTime = Date.now()
    timerRef.current = setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - startTime) / 1000))
    }, 1000)
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- untyped API response with deeply nested optional fields
    const extractPersistedAnalysis = (detail: Record<string, any>): string => {
      if (!detail || typeof detail !== 'object') return ''
      const fromProcessing = firstNonEmptyString(detail?.processing?.analysis)
      if (fromProcessing) return fromProcessing
      const fromRoot = firstNonEmptyString(
        detail?.analysis,
        detail?.analysis_content,
        detail?.analysisContent
      )
      if (fromRoot) return fromRoot
      if (Array.isArray(detail?.analyses)) {
        for (const entry of detail.analyses) {
          const text = typeof entry === 'string'
            ? entry
            : (entry?.content || entry?.text || entry?.summary || entry?.analysis_content || '')
          const resolved = firstNonEmptyString(text)
          if (resolved) return resolved
        }
      }
      return ''
    }

    const saveAsVersion = async (analysisText: string) => {
      if (!mediaId) return false
      if (!effectiveMediaContent || !effectiveMediaContent.trim()) return false
      try {
        const savedDetail = await bgRequest<any>({
          path: `/api/v1/media/${mediaId}/versions`,
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: {
            content: String(effectiveMediaContent || ''),
            analysis_content: analysisText,
            prompt: systemPrompt
          }
        })
        let persistedAnalysis = extractPersistedAnalysis(savedDetail)
        if (persistedAnalysis.trim() !== analysisText.trim()) {
          const refreshedDetail = await bgRequest<any>({
            path: `/api/v1/media/${mediaId}`,
            method: 'GET'
          })
          persistedAnalysis = extractPersistedAnalysis(refreshedDetail)
        }
        return persistedAnalysis.trim() === analysisText.trim()
      } catch (err) {
        console.error('Failed to save analysis as version:', err)
        return false
      }
    }

    try {
      const { requestTimeoutMs, streamIdleTimeoutMs } = await getAnalysisTimeouts()
      const requestBody = {
        model: normalizedModel || effectiveModel,
        ...(resolvedApiProvider ? { api_provider: resolvedApiProvider } : {}),
        messages: [
          { role: 'system', content: systemPrompt },
          {
            role: 'user',
            content: `${userPrefix ? userPrefix + '\n\n' : ''}${effectiveMediaContent}`
          }
        ]
      }

      let analysisText = ''
      let streamError: unknown = null

      try {
        let lastPreviewAt = 0
        let streamedText = ""
        for await (const chunk of bgStream({
          path: '/api/v1/chat/completions',
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: { ...requestBody, stream: true },
          streamIdleTimeoutMs,
          abortSignal: abortController.signal
        })) {
          if (cancelledByUserRef.current || abortController.signal.aborted) {
            return
          }
          const delta = extractStreamDelta(chunk)
          if (delta) {
            streamedText += delta
            analysisText = streamedText
            const now = Date.now()
            if (now - lastPreviewAt > 120) {
              setAnalysisPreview(streamedText)
              lastPreviewAt = now
            }
          }
        }
        if (analysisText) {
          setAnalysisPreview(analysisText)
        }
      } catch (err) {
        if (cancelledByUserRef.current || abortController.signal.aborted || isAbortError(err)) {
          return
        }
        streamError = err
      }

      if (cancelledByUserRef.current || abortController.signal.aborted) return
      if (!analysisText) {
        const resp = await bgRequest<any>({
          path: '/api/v1/chat/completions',
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: { ...requestBody, stream: false },
          timeoutMs: requestTimeoutMs
        })
        analysisText =
          resp?.choices?.[0]?.message?.content || resp?.content || ''
        if (analysisText) setAnalysisPreview(analysisText)
      } else if (streamError) {
        console.warn('Analysis stream completed with warnings:', streamError)
      }
      if (cancelledByUserRef.current || abortController.signal.aborted) return

      if (!analysisText) {
        messageApi.error(t('mediaPage.noAnalysisReturned', 'No analysis returned from API'))
        return
      }

      const persisted = await saveAsVersion(analysisText)
      if (persisted) {
        onAnalysisGenerated?.(analysisText, systemPrompt)
        messageApi.success(t('mediaPage.analysisGeneratedAndSaved', 'Analysis generated and saved'))
        onClose()
      } else {
        messageApi.error(t('mediaPage.analysisSaveFailed', 'Failed to save analysis to media item'))
      }
    } catch (err) {
      if (cancelledByUserRef.current || abortController.signal.aborted || isAbortError(err)) {
        return
      }
      if (isTimeoutError(err)) {
        messageApi.error(buildTimeoutMessage())
      } else {
        const errorText = String((err as Error)?.message || err || '').toLowerCase()
        messageApi.error(
          errorText.includes('analysis api provider is required')
            ? t(
                'mediaPage.analysisProviderRequired',
                'Choose an analysis provider, then retry analysis.'
              )
            : t('mediaPage.analysisGenerateFailed', 'Failed to generate analysis')
        )
      }
      console.error('Generation error:', err)
    } finally {
      clearGenerationTimer()
      if (activeAbortControllerRef.current === abortController) {
        activeAbortControllerRef.current = null
      }
      setGenerating(false)
      setElapsedSeconds(0)
    }
  }

  return (
    <Modal
      title={t('mediaPage.generateAnalysis', 'Generate Analysis')}
      open={open}
      onCancel={handleModalClose}
      width={700}
      footer={[
        <Button key="save" onClick={handleSaveAsDefault}>
          {t('mediaPage.saveAsDefault', 'Save as default')}
        </Button>,
        <Button key="cancel" onClick={handleModalClose}>
          {t('common:cancel', 'Cancel')}
        </Button>,
        generating ? (
          <Button key="cancel-generation" danger onClick={() => handleCancelGeneration(true)}>
            {t('mediaPage.cancelGeneration', 'Cancel generation')}
          </Button>
        ) : null,
        <Button
          key="generate"
          type="primary"
          loading={generating}
          onClick={handleGenerate}
          disabled={
            !effectiveMediaContent ||
            !effectiveMediaContent.trim() ||
            (!selectedModelKey && models.length === 0)
          }
        >
          {t('mediaPage.generateAnalysis', 'Generate Analysis')}
        </Button>
      ].filter(Boolean)}
    >
      <div className="space-y-4">
        {cancelledGeneration && !generating && (
          <div className="rounded-md border border-warning/40 bg-warning/10 p-2 text-xs text-warning">
            {t('mediaPage.analysisGenerationCancelled', 'Analysis generation cancelled')}
          </div>
        )}
        {/* M9: Show indeterminate spinner with elapsed time instead of misleading progress bar */}
        {generating && (
          <div className="rounded-md border border-primary bg-surface2 p-3">
            <div className="flex items-center gap-3">
              <Spin size="small" />
              <div className="flex-1">
                <div className="text-sm font-medium text-primaryStrong">
                  {t('mediaPage.generatingAnalysis', 'Generating analysis...')}
                </div>
                <div className="text-xs text-primary">
                  {t('mediaPage.elapsedTime', 'Elapsed: {{seconds}}s', { seconds: elapsedSeconds })}
                </div>
              </div>
            </div>
          </div>
        )}
        {(generating || analysisPreview) && (
          <div aria-live="polite">
            <label className="block text-sm font-medium text-text mb-2">
              {t('mediaPage.analysis', 'Analysis')}
            </label>
            <Input.TextArea
              value={analysisPreview}
              placeholder={t('mediaPage.generatingAnalysis', 'Generating analysis...')}
              rows={6}
              readOnly
              className="text-sm"
            />
          </div>
        )}
        <div>
          <label
            htmlFor="media-analysis-model"
            className="block text-sm font-medium text-text mb-2">
            {t('mediaPage.model', 'Model')}
          </label>
          <Select
            id="media-analysis-model"
            aria-label={t('mediaPage.model', 'Model')}
            value={selectedModelKey}
            onChange={setSelectedModel}
            className="w-full"
            placeholder={t('mediaPage.selectModel', 'Select a model')}
            notFoundContent={
              models.length === 0
                ? t('mediaPage.noModelsAvailable', 'No models available')
                : undefined
            }
          >
            {models.map((model) => (
              <Select.Option key={model.id} value={model.id}>
                {model.name || model.id}
              </Select.Option>
            ))}
          </Select>
        </div>

        <div>
          <div className="flex items-center justify-between mb-2">
            <label className="block text-sm font-medium text-text">
              {t('mediaPage.promptPresets', 'Prompt Presets')}
            </label>
            <Button
              size="small"
              type="link"
              onClick={() => setShowPresets(!showPresets)}
            >
              {showPresets
                ? t('mediaPage.hidePresets', 'Hide Presets')
                : t('mediaPage.showPresets', 'Show Presets')}
            </Button>
          </div>
          {showPresets && (
            <div className="flex flex-wrap gap-2 mb-3">
              {presets.map((preset, idx) => (
                <Button
                  key={idx}
                  size="small"
                  onClick={() => {
                    setSystemPrompt(preset.system)
                    setUserPrefix(preset.user)
                  }}
                >
                  {preset.name}
                </Button>
              ))}
            </div>
          )}
        </div>

        {/* System Prompt */}
        <div>
          <label
            htmlFor="systemPrompt"
            className="block text-sm font-medium text-text mb-2">
            {t('mediaPage.systemPromptLabel', 'System Prompt')}
          </label>
          <Input.TextArea
            id="systemPrompt"
            aria-label={t('mediaPage.systemPromptLabel', 'System Prompt')}
            value={systemPrompt}
            onChange={(e) => setSystemPrompt(e.target.value)}
            rows={4}
            placeholder={t(
              'mediaPage.systemPromptPlaceholder',
              'Enter system prompt...'
            )}
            className="text-sm"
          />
        </div>

        {/* User Prefix */}
        <div>
          <label
            htmlFor="userPromptPrefix"
            className="block text-sm font-medium text-text mb-2">
            {t('mediaPage.userPromptPrefixLabel', 'User Prompt Prefix')}
            <span className="text-xs text-text-muted ml-2">
              {t(
                'mediaPage.prependedBeforeContent',
                '(prepended before content)'
              )}
            </span>
          </label>
          <Input.TextArea
            id="userPromptPrefix"
            aria-label={t('mediaPage.userPromptPrefixLabel', 'User Prompt Prefix')}
            value={userPrefix}
            onChange={(e) => setUserPrefix(e.target.value)}
            rows={3}
            placeholder={t(
              'mediaPage.userPrefixPlaceholder',
              'Optional: Enter text to prepend before the media content...'
            )}
            className="text-sm"
          />
        </div>
      </div>
    </Modal>
  )
}
