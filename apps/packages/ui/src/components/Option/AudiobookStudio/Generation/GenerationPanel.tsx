import React from "react"
import {
  Card,
  Typography,
  Button,
  Space,
  Progress,
  Alert,
  Empty,
  Select,
  Switch
} from "antd"
import { useTranslation } from "react-i18next"
import { useQuery } from "@tanstack/react-query"
import { Play, Square, AlertCircle, Check, Loader2 } from "lucide-react"
import { useAudiobookStudioStore } from "@/store/audiobook-studio"
import { useAudiobookGeneration } from "@/hooks/useAudiobookGeneration"
import { getTTSSettings } from "@/services/tts"
import { getTtsProviderLabel } from "@/services/tts-providers"
import {
  useTtsProviderData,
  OPENAI_TTS_VOICES,
  OPENAI_TTS_MODELS
} from "@/hooks/useTtsProviderData"
import { inferTldwProviderFromModel } from "@/services/tts-provider"

const { Text, Title } = Typography

export const GenerationPanel: React.FC = () => {
  const { t } = useTranslation(["audiobook", "playground", "common"])

  const chapters = useAudiobookStudioStore((s) => s.chapters)
  const isGenerating = useAudiobookStudioStore((s) => s.isGenerating)
  const currentGeneratingId = useAudiobookStudioStore(
    (s) => s.currentGeneratingId
  )
  const defaultVoiceConfig = useAudiobookStudioStore((s) => s.defaultVoiceConfig)
  const setDefaultVoiceConfig = useAudiobookStudioStore(
    (s) => s.setDefaultVoiceConfig
  )

  const { generateAllChapters, cancelGeneration } = useAudiobookGeneration()

  const { data: ttsSettings } = useQuery({
    queryKey: ["fetchTTSSettings"],
    queryFn: getTTSSettings
  })

  const provider = defaultVoiceConfig.provider || ttsSettings?.ttsProvider || "browser"
  const isTldw = provider === "tldw"
  const hasLocalBackendChoice = defaultVoiceConfig.tldwBackend !== undefined
  const configuredBackend =
    defaultVoiceConfig.tldwBackend ?? ttsSettings?.tldwTtsBackend ?? ""
  const configuredTldwModel =
    defaultVoiceConfig.tldwModel ||
    (hasLocalBackendChoice ? undefined : ttsSettings?.tldwTtsModel)
  const configuredTldwVoice =
    defaultVoiceConfig.tldwVoice ||
    (hasLocalBackendChoice ? undefined : ttsSettings?.tldwTtsVoice)
  const inferredProviderKey = React.useMemo(() => {
    if (!isTldw) return null
    return inferTldwProviderFromModel(configuredTldwModel)
  }, [configuredTldwModel, isTldw])

  const {
    providersInfo,
    tldwTtsModels,
    tldwVoiceCatalog,
    elevenLabsData
  } = useTtsProviderData({
    provider,
    backend: configuredBackend || undefined,
    model: configuredTldwModel,
    elevenLabsApiKey: ttsSettings?.elevenLabsApiKey,
    inferredProviderKey
  })

  const completedCount = chapters.filter((ch) => ch.status === "completed").length
  const errorCount = chapters.filter((ch) => ch.status === "error").length
  const pendingCount = chapters.filter(
    (ch) => ch.status === "pending" || ch.status === "error"
  ).length

  const currentIndex = chapters.findIndex((ch) => ch.id === currentGeneratingId)
  const progress =
    chapters.length > 0
      ? Math.round((completedCount / chapters.length) * 100)
      : 0

  const handleGenerateAll = async () => {
    await generateAllChapters({ chapters })
  }

  const providerLabel = getTtsProviderLabel(provider)
  const isBrowserTts = provider === "browser"

  const providerVoices = React.useMemo(() => {
    if (tldwVoiceCatalog && tldwVoiceCatalog.length > 0) {
      return tldwVoiceCatalog
    }
    return []
  }, [tldwVoiceCatalog])

  const explicitBackendSupported =
    providersInfo?.supports_explicit_backend === true
  const handleBackendChange = (backend: string) => {
    const capabilities = providersInfo?.providers?.[backend]
    const model =
      capabilities?.default_model || capabilities?.models?.[0] || undefined
    const modelCapabilities = model
      ? capabilities?.model_capabilities?.[model]
      : undefined
    const firstVoice = capabilities?.voices?.[0]
    const voice =
      modelCapabilities?.default_voice ||
      modelCapabilities?.voices?.[0] ||
      firstVoice?.voice_id ||
      firstVoice?.id ||
      firstVoice?.name ||
      undefined

    setDefaultVoiceConfig({
      ...defaultVoiceConfig,
      provider: "tldw",
      tldwBackend: backend,
      tldwAllowFallback: defaultVoiceConfig.tldwAllowFallback ?? true,
      tldwModel: model,
      tldwVoice: voice
    })
  }
  const backendOptions = React.useMemo(
    () => [
      {
        label: "Automatic (legacy model inference)",
        value: ""
      },
      ...Object.entries(providersInfo?.providers || {})
        .filter(
          ([backend, caps]) =>
            typeof caps.display_name === "string" ||
            backend === "openrouter" ||
            backend.startsWith("gateway:")
        )
        .map(([backend, caps]) => ({
          label: caps.display_name || backend,
          value: backend
        }))
    ],
    [providersInfo]
  )

  const openAiVoiceOptions = React.useMemo(() => {
    const model = defaultVoiceConfig.openAiModel || ttsSettings?.openAITTSModel
    if (!model) {
      const seen = new Set<string>()
      const all: { label: string; value: string }[] = []
      Object.values(OPENAI_TTS_VOICES).forEach((list) => {
        list.forEach((v) => {
          if (!seen.has(v.value)) {
            seen.add(v.value)
            all.push(v)
          }
        })
      })
      return all
    }
    return OPENAI_TTS_VOICES[model] || []
  }, [defaultVoiceConfig.openAiModel, ttsSettings?.openAITTSModel])

  if (chapters.length === 0) {
    return (
      <Card>
        <Empty
          description={
            <Text type="secondary">
              {t(
                "audiobook:generation.noChapters",
                "Add chapters first to generate audio."
              )}
            </Text>
          }
        />
      </Card>
    )
  }

  return (
    <div className="space-y-4">
      {isBrowserTts && (
        <Alert
          type="warning"
          showIcon
          icon={<AlertCircle className="h-4 w-4" />}
          title={t(
            "audiobook:generation.browserWarningTitle",
            "Browser TTS cannot export audio"
          )}
          description={t(
            "audiobook:generation.browserWarningDesc",
            "The browser TTS provider streams audio directly and cannot be saved to files. Please switch to a different TTS provider (tldw, ElevenLabs, or OpenAI) in your settings to generate downloadable audiobook files."
          )}
        />
      )}

      <Card>
        <div className="space-y-4">
          <div>
            <Title level={5} className="!mb-1">
              {t("audiobook:generation.voiceSettings", "Voice Settings")}
            </Title>
            <Text type="secondary" className="text-sm">
              {t(
                "audiobook:generation.voiceSettingsDesc",
                "Configure the default voice for new chapters. Existing chapters keep their current settings. Using provider: {{provider}}",
                { provider: providerLabel }
              )}
            </Text>
          </div>

          {isTldw && explicitBackendSupported && (
            <div className="flex flex-wrap items-center gap-4">
              <div aria-label="Audiobook backend">
                <label className="block text-xs mb-1">Backend</label>
                <Select
                  style={{ minWidth: 220 }}
                  value={configuredBackend}
                  options={backendOptions}
                  onChange={handleBackendChange}
                />
              </div>
              <label className="flex items-center gap-2 text-xs">
                <Switch
                  size="small"
                  aria-label="Allow configured fallback"
                  checked={defaultVoiceConfig.tldwAllowFallback !== false}
                  onChange={(checked) =>
                    setDefaultVoiceConfig({
                      ...defaultVoiceConfig,
                      tldwAllowFallback: checked
                    })
                  }
                />
                Allow configured fallback
              </label>
            </div>
          )}

          {isTldw && (
            <div className="flex flex-wrap gap-4">
              <div>
                <label className="block text-xs mb-1">
                  {t("audiobook:generation.model", "Model")}
                </label>
                {tldwTtsModels && tldwTtsModels.length > 0 ? (
                  <Select
                    style={{ minWidth: 180 }}
                    placeholder={t("audiobook:generation.selectModel", "Select model")}
                    showSearch
                    optionFilterProp="label"
                    options={tldwTtsModels.map((m) => ({
                      label: m.label,
                      value: m.id
                    }))}
                    value={configuredTldwModel}
                    onChange={(val) =>
                      setDefaultVoiceConfig({ ...defaultVoiceConfig, tldwModel: val })
                    }
                  />
                ) : (
                  <Text type="secondary" className="text-xs">
                    {t("audiobook:generation.noModels", "No models available")}
                  </Text>
                )}
              </div>
              {providerVoices.length > 0 && (
                <div>
                  <label className="block text-xs mb-1">
                    {t("audiobook:generation.voice", "Voice")}
                  </label>
                  <Select
                    style={{ minWidth: 200 }}
                    placeholder={t("audiobook:generation.selectVoice", "Select voice")}
                    options={providerVoices.map((v, idx) => ({
                      label: `${v.name || v.voice_id || v.id || `Voice ${idx + 1}`}${
                        v.language ? ` (${v.language})` : ""
                      }`,
                      value: v.voice_id || v.id || v.name || ""
                    }))}
                    value={configuredTldwVoice}
                    onChange={(val) =>
                      setDefaultVoiceConfig({ ...defaultVoiceConfig, tldwVoice: val })
                    }
                  />
                </div>
              )}
            </div>
          )}

          {provider === "elevenlabs" && elevenLabsData && (
            <div className="flex flex-wrap gap-4">
              <div>
                <label className="block text-xs mb-1">
                  {t("audiobook:generation.model", "Model")}
                </label>
                <Select
                  style={{ minWidth: 180 }}
                  placeholder={t("audiobook:generation.selectModel", "Select model")}
                  options={elevenLabsData.models.map((m: any) => ({
                    label: m.name,
                    value: m.model_id
                  }))}
                  value={
                    defaultVoiceConfig.elevenLabsModel ||
                    ttsSettings?.elevenLabsModel
                  }
                  onChange={(val) =>
                    setDefaultVoiceConfig({
                      ...defaultVoiceConfig,
                      elevenLabsModel: val
                    })
                  }
                />
              </div>
              <div>
                <label className="block text-xs mb-1">
                  {t("audiobook:generation.voice", "Voice")}
                </label>
                <Select
                  style={{ minWidth: 200 }}
                  placeholder={t("audiobook:generation.selectVoice", "Select voice")}
                  options={elevenLabsData.voices.map((v: any) => ({
                    label: v.name,
                    value: v.voice_id
                  }))}
                  value={
                    defaultVoiceConfig.elevenLabsVoiceId ||
                    ttsSettings?.elevenLabsVoiceId
                  }
                  onChange={(val) =>
                    setDefaultVoiceConfig({
                      ...defaultVoiceConfig,
                      elevenLabsVoiceId: val
                    })
                  }
                />
              </div>
            </div>
          )}

          {provider === "openai" && (
            <div className="flex flex-wrap gap-4">
              <div>
                <label className="block text-xs mb-1">
                  {t("audiobook:generation.model", "Model")}
                </label>
                <Select
                  style={{ minWidth: 150 }}
                  placeholder={t("audiobook:generation.selectModel", "Select model")}
                  options={OPENAI_TTS_MODELS}
                  value={
                    defaultVoiceConfig.openAiModel || ttsSettings?.openAITTSModel
                  }
                  onChange={(val) =>
                    setDefaultVoiceConfig({
                      ...defaultVoiceConfig,
                      openAiModel: val
                    })
                  }
                />
              </div>
              <div>
                <label className="block text-xs mb-1">
                  {t("audiobook:generation.voice", "Voice")}
                </label>
                <Select
                  style={{ minWidth: 150 }}
                  placeholder={t("audiobook:generation.selectVoice", "Select voice")}
                  options={openAiVoiceOptions}
                  value={
                    defaultVoiceConfig.openAiVoice || ttsSettings?.openAITTSVoice
                  }
                  onChange={(val) =>
                    setDefaultVoiceConfig({
                      ...defaultVoiceConfig,
                      openAiVoice: val
                    })
                  }
                />
              </div>
            </div>
          )}
        </div>
      </Card>

      <Card>
        <div className="space-y-4">
          <div>
            <Title level={5} className="!mb-1">
              {t("audiobook:generation.title", "Generate Audio")}
            </Title>
            <Text type="secondary" className="text-sm">
              {t(
                "audiobook:generation.description",
                "Generate audio for all pending chapters sequentially."
              )}
            </Text>
          </div>

          <div className="flex flex-wrap items-center gap-4">
            <div className="flex items-center gap-2">
              <Text>
                {t("audiobook:generation.completed", "Completed:")}
              </Text>
              <Text strong className="text-success">
                {completedCount}/{chapters.length}
              </Text>
            </div>
            {errorCount > 0 && (
              <div className="flex items-center gap-2">
                <Text>{t("audiobook:generation.errors", "Errors:")}</Text>
                <Text strong className="text-danger">
                  {errorCount}
                </Text>
              </div>
            )}
            {pendingCount > 0 && (
              <div className="flex items-center gap-2">
                <Text>{t("audiobook:generation.pending", "Pending:")}</Text>
                <Text strong>{pendingCount}</Text>
              </div>
            )}
          </div>

          <Progress
            percent={progress}
            status={isGenerating ? "active" : errorCount > 0 ? "exception" : "normal"}
            strokeColor={errorCount > 0 ? undefined : { from: "rgb(var(--color-primary))", to: "rgb(var(--color-success))" }}
          />

          {isGenerating && currentIndex >= 0 && (
            <div className="flex items-center gap-2 text-sm">
              <Loader2 className="h-4 w-4 animate-spin text-primary" />
              <Text>
                {t(
                  "audiobook:generation.currentlyGenerating",
                  "Generating chapter {{current}} of {{total}}: {{title}}",
                  {
                    current: currentIndex + 1,
                    total: chapters.length,
                    title: chapters[currentIndex]?.title
                  }
                )}
              </Text>
            </div>
          )}

          <Space>
            {!isGenerating ? (
              <Button
                type="primary"
                icon={<Play className="h-4 w-4" />}
                onClick={handleGenerateAll}
                disabled={pendingCount === 0 || isBrowserTts}
              >
                {pendingCount === chapters.length
                  ? t("audiobook:generation.generateAll", "Generate all")
                  : t("audiobook:generation.generateRemaining", "Generate remaining ({{count}})", {
                      count: pendingCount
                    })}
              </Button>
            ) : (
              <Button
                danger
                icon={<Square className="h-4 w-4" />}
                onClick={cancelGeneration}
              >
                {t("audiobook:generation.cancel", "Cancel")}
              </Button>
            )}
          </Space>

          {completedCount === chapters.length && chapters.length > 0 && (
            <Alert
              type="success"
              showIcon
              icon={<Check className="h-4 w-4" />}
              title={t(
                "audiobook:generation.allComplete",
                "All chapters generated successfully!"
              )}
              description={t(
                "audiobook:generation.allCompleteDesc",
                "Go to the Output tab to download your audiobook files."
              )}
            />
          )}
        </div>
      </Card>
    </div>
  )
}

export default GenerationPanel
