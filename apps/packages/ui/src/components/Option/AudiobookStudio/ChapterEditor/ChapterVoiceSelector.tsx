import React, { useMemo } from "react"
import { Select, Space, Switch, Typography } from "antd"
import { useTranslation } from "react-i18next"
import { Mic } from "lucide-react"
import { useStorage } from "@plasmohq/storage/hook"
import { useQuery } from "@tanstack/react-query"
import {
  useTtsProviderData,
  OPENAI_TTS_MODELS,
  OPENAI_TTS_VOICES
} from "@/hooks/useTtsProviderData"
import {
  inferTldwProviderFromModel,
  type TtsProviderOverrides
} from "@/services/tts-provider"
import { getTTSSettings } from "@/services/tts"

const { Text } = Typography

type ChapterVoiceSelectorProps = {
  voiceConfig: TtsProviderOverrides & { speed?: number }
  onChange: (config: TtsProviderOverrides & { speed?: number }) => void
  compact?: boolean
}

export const ChapterVoiceSelector: React.FC<ChapterVoiceSelectorProps> = ({
  voiceConfig,
  onChange,
  compact = true
}) => {
  const { t } = useTranslation(["audiobook", "playground"])
  const [elevenLabsApiKey] = useStorage<string | null>("elevenLabsApiKey", null)
  const { data: ttsSettings } = useQuery({
    queryKey: ["fetchTTSSettings"],
    queryFn: getTTSSettings
  })

  const provider = voiceConfig.provider || ttsSettings?.ttsProvider || "browser"
  const hasLocalBackendChoice = voiceConfig.tldwBackend !== undefined
  const configuredBackend =
    voiceConfig.tldwBackend ?? ttsSettings?.tldwTtsBackend ?? ""
  const configuredTldwModel =
    voiceConfig.tldwModel ||
    (hasLocalBackendChoice ? undefined : ttsSettings?.tldwTtsModel)
  // Infer tldw provider key from the selected model
  const inferredProviderKey = inferTldwProviderFromModel(voiceConfig.tldwModel)

  const {
    hasAudio,
    providersInfo,
    tldwTtsModels,
    tldwVoiceCatalog,
    elevenLabsData,
    elevenLabsLoading
  } = useTtsProviderData({
    provider,
    backend: configuredBackend || undefined,
    model: configuredTldwModel,
    elevenLabsApiKey,
    inferredProviderKey
  })

  // Provider options
  const providerOptions = useMemo(() => {
    const options = [
      { value: "browser", label: t("playground:tts.provider.browser", "Browser") }
    ]
    if (hasAudio) {
      options.unshift({
        value: "tldw",
        label: t("playground:tts.provider.tldw", "Server TTS")
      })
    }
    if (elevenLabsApiKey) {
      options.push({
        value: "elevenlabs",
        label: t("playground:tts.provider.elevenlabs", "ElevenLabs")
      })
    }
    options.push({
      value: "openai",
      label: t("playground:tts.provider.openai", "OpenAI")
    })
    return options
  }, [hasAudio, elevenLabsApiKey, t])

  // Model options based on provider
  const modelOptions = useMemo(() => {
    if (provider === "openai") {
      return OPENAI_TTS_MODELS
    }
    if (provider === "tldw" && tldwTtsModels) {
      return tldwTtsModels.map((m) => ({
        value: m.id,
        label: m.label || m.id
      }))
    }
    if (provider === "elevenlabs" && elevenLabsData?.models) {
      return elevenLabsData.models.map((m) => ({
        value: m.model_id,
        label: m.name
      }))
    }
    return []
  }, [provider, tldwTtsModels, elevenLabsData])

  // Voice options based on provider
  const voiceOptions = useMemo(() => {
    if (provider === "openai") {
      const model = voiceConfig.openAiModel || "tts-1"
      return OPENAI_TTS_VOICES[model] || OPENAI_TTS_VOICES["tts-1"]
    }
    if (provider === "tldw" && tldwVoiceCatalog) {
      return tldwVoiceCatalog.map((v) => ({
        value: v.voice_id || v.id || v.name,
        label: v.name || v.voice_id || v.id
      }))
    }
    if (provider === "elevenlabs" && elevenLabsData?.voices) {
      return elevenLabsData.voices.map((v) => ({
        value: v.voice_id,
        label: v.name
      }))
    }
    return []
  }, [provider, voiceConfig.openAiModel, tldwVoiceCatalog, elevenLabsData])

  // TLDW provider key options (from providers object)
  const explicitBackendSupported =
    providersInfo?.supports_explicit_backend === true
  const tldwProviderOptions = useMemo(() => {
    if (!providersInfo?.providers) return []
    return [
      { value: "", label: "Automatic (legacy model inference)" },
      ...Object.entries(providersInfo.providers)
        .filter(
          ([backend, info]) =>
            typeof info.display_name === "string" ||
            backend === "openrouter" ||
            backend.startsWith("gateway:")
        )
        .map(([key, info]) => ({
          value: key,
          label: info.display_name || key
        }))
    ]
  }, [providersInfo])

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

    onChange({
      ...voiceConfig,
      provider: "tldw",
      tldwBackend: backend,
      tldwAllowFallback: voiceConfig.tldwAllowFallback ?? true,
      tldwModel: model,
      tldwVoice: voice
    })
  }

  const gatewayControls =
    provider === "tldw" && explicitBackendSupported ? (
      <>
        <div aria-label="Chapter TTS backend">
          <Select
            size={compact ? "small" : "middle"}
            value={configuredBackend}
            onChange={handleBackendChange}
            options={tldwProviderOptions}
            style={compact ? { minWidth: 160 } : undefined}
            className={compact ? undefined : "w-full"}
          />
        </div>
        <Switch
          size="small"
          aria-label="Chapter allow configured fallback"
          checked={voiceConfig.tldwAllowFallback !== false}
          onChange={(checked) =>
            onChange({ ...voiceConfig, tldwAllowFallback: checked })
          }
        />
      </>
    ) : null

  const handleProviderChange = (value: string) => {
    // Clear all provider-specific settings when switching
    onChange({
      provider: value,
      speed: voiceConfig.speed
    })
  }

  const handleModelChange = (value: string) => {
    if (provider === "openai") {
      onChange({
        ...voiceConfig,
        openAiModel: value,
        openAiVoice: undefined
      })
    } else if (provider === "elevenlabs") {
      onChange({
        ...voiceConfig,
        elevenLabsModel: value,
        elevenLabsVoiceId: undefined
      })
    } else {
      // tldw
      onChange({
        ...voiceConfig,
        tldwModel: value,
        tldwVoice: undefined
      })
    }
  }

  const handleVoiceChange = (value: string) => {
    if (provider === "openai") {
      onChange({
        ...voiceConfig,
        openAiVoice: value
      })
    } else if (provider === "elevenlabs") {
      onChange({
        ...voiceConfig,
        elevenLabsVoiceId: value
      })
    } else {
      // tldw
      onChange({
        ...voiceConfig,
        tldwVoice: value
      })
    }
  }

  // Get current model value based on provider
  const getCurrentModel = () => {
    if (provider === "openai") return voiceConfig.openAiModel
    if (provider === "elevenlabs") return voiceConfig.elevenLabsModel
    return voiceConfig.tldwModel
  }

  // Get current voice value based on provider
  const getCurrentVoice = () => {
    if (provider === "openai") return voiceConfig.openAiVoice
    if (provider === "elevenlabs") return voiceConfig.elevenLabsVoiceId
    return voiceConfig.tldwVoice
  }

  const currentModel = getCurrentModel()
  const currentVoice = getCurrentVoice()

  if (compact) {
    return (
      <div className="flex items-center gap-2 flex-wrap">
        <Mic className="h-4 w-4 text-text-muted" />
        <Select
          size="small"
          value={provider}
          onChange={handleProviderChange}
          options={providerOptions}
          style={{ minWidth: 100 }}
          dropdownStyle={{ minWidth: 120 }}
        />
        {gatewayControls}
        {modelOptions.length > 0 && (
          <Select
            size="small"
            value={currentModel}
            onChange={handleModelChange}
            options={modelOptions}
            placeholder={t("audiobook:voice.selectModel", "Model")}
            style={{ minWidth: 100 }}
            loading={elevenLabsLoading}
            allowClear
          />
        )}
        {voiceOptions.length > 0 && (
          <Select
            size="small"
            value={currentVoice}
            onChange={handleVoiceChange}
            options={voiceOptions}
            placeholder={t("audiobook:voice.selectVoice", "Voice")}
            style={{ minWidth: 120 }}
            showSearch
            filterOption={(input, option) =>
              (option?.label as string)?.toLowerCase().includes(input.toLowerCase()) ?? false
            }
            allowClear
          />
        )}
      </div>
    )
  }

  return (
    <Space orientation="vertical" className="w-full" size="small">
      <div>
        <Text type="secondary" className="text-xs block mb-1">
          {t("audiobook:voice.providerLabel", "Provider")}
        </Text>
        <Select
          value={provider}
          onChange={handleProviderChange}
          options={providerOptions}
          className="w-full"
        />
      </div>

      {gatewayControls && (
        <div>
          <Text type="secondary" className="text-xs block mb-1">
            Backend
          </Text>
          <div className="flex items-center gap-2">{gatewayControls}</div>
        </div>
      )}

      {modelOptions.length > 0 && (
        <div>
          <Text type="secondary" className="text-xs block mb-1">
            {t("audiobook:voice.modelLabel", "Model")}
          </Text>
          <Select
            value={currentModel}
            onChange={handleModelChange}
            options={modelOptions}
            placeholder={t("audiobook:voice.selectModel", "Select model...")}
            className="w-full"
            loading={elevenLabsLoading}
            allowClear
          />
        </div>
      )}

      {voiceOptions.length > 0 && (
        <div>
          <Text type="secondary" className="text-xs block mb-1">
            {t("audiobook:voice.voiceLabel", "Voice")}
          </Text>
          <Select
            value={currentVoice}
            onChange={handleVoiceChange}
            options={voiceOptions}
            placeholder={t("audiobook:voice.selectVoice", "Select voice...")}
            className="w-full"
            showSearch
            filterOption={(input, option) =>
              (option?.label as string)?.toLowerCase().includes(input.toLowerCase()) ?? false
            }
            allowClear
          />
        </div>
      )}
    </Space>
  )
}

export default ChapterVoiceSelector
