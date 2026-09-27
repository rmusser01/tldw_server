import React from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { useTranslation } from "react-i18next"
import {
  Button,
  Input,
  List,
  Popconfirm,
  Select,
  Space,
  Switch,
  Tag,
  Typography,
  notification
} from "antd"
import { Trash2, Play, UploadCloud, Wand2 } from "lucide-react"
import { Alert } from "@/components/ui/primitives/Alert"
import type { TldwTtsProvidersInfo } from "@/services/tldw/audio-providers"
import { getProviderLabel } from "@/utils/provider-registry"
import {
  deleteCustomVoice,
  encodeCustomVoice,
  listCustomVoices,
  uploadCustomVoice,
  VOICE_PROVIDER_REQUIREMENTS,
  type TldwCustomVoice
} from "@/services/tldw/voice-cloning"
import { normalizeTtsProviderKey, toServerTtsProviderKey } from "@/services/tldw/tts-provider-keys"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { sanitizeServerErrorMessage } from "@/utils/server-error-message"

const { Text, Paragraph } = Typography

type ProviderOption = {
  value: string
  label: string
  available: boolean
  supportsCloning: boolean
}

type VoiceCloningManagerProps = {
  providersInfo?: TldwTtsProvidersInfo | null
  onSelectVoice?: (value: string, provider?: string) => void
  onSelectVoices?: (voices: Array<{ role: string; voiceId: string }>) => void
}

const formatSeconds = (value?: number | null) => {
  if (value == null || Number.isNaN(value)) return "n/a"
  if (value < 1) return `${Math.round(value * 1000)} ms`
  return `${value.toFixed(value < 10 ? 1 : 0)} s`
}

const formatBytes = (value?: number | null) => {
  if (!value || value <= 0) return "n/a"
  if (value < 1024) return `${value} B`
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`
  return `${(value / (1024 * 1024)).toFixed(1)} MB`
}

const VOICE_ROLE_OPTIONS = [
  { label: "Narrator", value: "narrator" },
  { label: "Speaker A", value: "speaker_a" },
  { label: "Speaker B", value: "speaker_b" },
  { label: "Speaker C", value: "speaker_c" }
]
const DEFAULT_VOICE_ROLE = "narrator"

const resolveCaps = (providersInfo: TldwTtsProvidersInfo | null | undefined, key: string) => {
  if (!providersInfo?.providers) return null
  const target = normalizeTtsProviderKey(key)
  const matchKey = Object.keys(providersInfo.providers).find(
    (candidate) => normalizeTtsProviderKey(candidate) === target
  )
  return matchKey ? providersInfo.providers[matchKey] : null
}

const getVoiceProviderRequirement = (key: string) => {
  const direct = VOICE_PROVIDER_REQUIREMENTS[key]
  if (direct) return direct
  const serverKey = toServerTtsProviderKey(key)
  return VOICE_PROVIDER_REQUIREMENTS[serverKey]
}

export const VoiceCloningManager: React.FC<VoiceCloningManagerProps> = ({
  providersInfo,
  onSelectVoice,
  onSelectVoices
}) => {
  const { t } = useTranslation(["playground"])
  const queryClient = useQueryClient()
  const { data: customVoices = [], isLoading: voicesLoading } = useQuery<TldwCustomVoice[]>(
    {
      queryKey: ["tts-custom-voices"],
      queryFn: listCustomVoices
    }
  )

  const providerOptions = React.useMemo<ProviderOption[]>(() => {
    const candidates = new Set<string>(Object.keys(VOICE_PROVIDER_REQUIREMENTS))
    if (providersInfo?.providers) {
      Object.keys(providersInfo.providers).forEach((key) => candidates.add(key))
    }
    const entries: ProviderOption[] = []
    for (const key of Array.from(candidates)) {
      const caps = resolveCaps(providersInfo, key)
      const supportsCloning =
        caps?.supports_voice_cloning ?? Boolean(getVoiceProviderRequirement(key))
      if (!supportsCloning) continue
      entries.push({
        value: key,
        label: getProviderLabel(key, "tts-engine") || key,
        available: Boolean(caps),
        supportsCloning
      })
    }
    return entries.sort((a, b) => a.label.localeCompare(b.label))
  }, [providersInfo])

  const [uploadProvider, setUploadProvider] = React.useState<string>(
    providerOptions.find((opt) => opt.available)?.value || providerOptions[0]?.value || ""
  )
  const [uploadName, setUploadName] = React.useState("")
  const [uploadDescription, setUploadDescription] = React.useState("")
  const [referenceText, setReferenceText] = React.useState("")
  const [uploadFile, setUploadFile] = React.useState<File | null>(null)
  const [previewText, setPreviewText] = React.useState(
    "Hello, this is a preview of your custom voice."
  )
  const [previewUrl, setPreviewUrl] = React.useState<string | null>(null)
  const [previewingId, setPreviewingId] = React.useState<string | null>(null)
  const [useVoiceRoles, setUseVoiceRoles] = React.useState(false)
  const [voiceCards, setVoiceCards] = React.useState<
    Array<{ id: string; role: string; voiceId: string }>
  >([])

  React.useEffect(() => {
    if (!uploadProvider && providerOptions.length > 0) {
      const next = providerOptions.find((opt) => opt.available) || providerOptions[0]
      setUploadProvider(next?.value || "")
    }
  }, [providerOptions, uploadProvider])

  React.useEffect(() => {
    return () => {
      if (previewUrl) {
        try {
          URL.revokeObjectURL(previewUrl)
        } catch {}
      }
    }
  }, [previewUrl])

  React.useEffect(() => {
    if (!useVoiceRoles) return
    setVoiceCards((prev) => {
      if (prev.length > 0) return prev
      const fallbackVoice = customVoices[0]?.voice_id
      return [
        {
          id: `voice-${Date.now()}`,
          role: DEFAULT_VOICE_ROLE,
          voiceId: fallbackVoice ? `custom:${fallbackVoice}` : ""
        }
      ]
    })
  }, [useVoiceRoles, customVoices])

  const uploadMutation = useMutation({
    mutationFn: async () => {
      if (!uploadFile) throw new Error("Please select a voice sample file.")
      if (!uploadProvider) throw new Error("Please select a target provider.")
      if (!uploadName.trim()) throw new Error("Please enter a name for the voice.")
      return uploadCustomVoice({
        file: uploadFile,
        name: uploadName.trim(),
        description: uploadDescription.trim() || undefined,
        provider: toServerTtsProviderKey(uploadProvider),
        referenceText: referenceText.trim() || undefined
      })
    },
    onSuccess: (data) => {
      notification.success({
        message: "Voice uploaded",
        description: data?.info || "Your custom voice is ready to use."
      })
      setUploadFile(null)
      setUploadName("")
      setUploadDescription("")
      setReferenceText("")
      queryClient.invalidateQueries({ queryKey: ["tts-custom-voices"] })
    },
    onError: (error: unknown) => {
      notification.error({
        message: "Voice upload failed",
        description: sanitizeServerErrorMessage(error, "Unable to upload voice sample.")
      })
    }
  })

  const encodeMutation = useMutation({
    mutationFn: async ({ voiceId, provider }: { voiceId: string; provider: string }) => {
      return encodeCustomVoice({
        voice_id: voiceId,
        provider: toServerTtsProviderKey(provider)
      })
    },
    onSuccess: () => {
      notification.success({
        message: "Voice prepared",
        description: "Provider-specific artifacts were generated."
      })
      queryClient.invalidateQueries({ queryKey: ["tts-custom-voices"] })
    },
    onError: (error: unknown) => {
      notification.error({
        message: "Voice encoding failed",
        description: sanitizeServerErrorMessage(error, "Unable to encode voice.")
      })
    }
  })

  const deleteMutation = useMutation({
    mutationFn: async (voiceId: string) => {
      await deleteCustomVoice(voiceId)
    },
    onSuccess: () => {
      notification.success({
        message: "Voice deleted"
      })
      queryClient.invalidateQueries({ queryKey: ["tts-custom-voices"] })
    },
    onError: (error: unknown) => {
      notification.error({
        message: "Delete failed",
        description: sanitizeServerErrorMessage(error, "Unable to delete voice.")
      })
    }
  })

  const handlePreview = async (voice: TldwCustomVoice) => {
    const voiceId = voice.voice_id
    if (!voiceId) return
    if (!voice.provider) {
      notification.error({
        message: "Missing provider",
        description: "This voice is missing a provider identifier."
      })
      return
    }
    setPreviewingId(voiceId)
    try {
      const text = previewText.trim() || "Hello, this is a preview of your custom voice."
      const result = await tldwClient.synthesizeSpeechDetailed(text, {
        model: voice.provider,
        voice: `custom:${voiceId}`,
        responseFormat: "mp3"
      })
      const blob = new Blob([result.buffer], { type: "audio/mpeg" })
      const url = URL.createObjectURL(blob)
      if (previewUrl) {
        try {
          URL.revokeObjectURL(previewUrl)
        } catch {}
      }
      setPreviewUrl(url)
    } catch (error: unknown) {
      notification.error({
        message: "Preview failed",
        description: sanitizeServerErrorMessage(error, "Unable to generate preview.")
      })
    } finally {
      setPreviewingId(null)
    }
  }

  const handleAddVoiceCard = () => {
    setVoiceCards((prev) => {
      if (prev.length >= 4) return prev

      const usedRoles = new Set(prev.map((card) => card.role))
      const nextRole =
        VOICE_ROLE_OPTIONS.find((option) => !usedRoles.has(option.value))?.value ??
        VOICE_ROLE_OPTIONS[0].value

      return [
        ...prev,
        {
          id: `voice-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
          role: nextRole,
          voiceId: ""
        }
      ]
    })
  }

  const handleRemoveVoiceCard = (id: string) => {
    setVoiceCards((prev) => prev.filter((card) => card.id !== id))
  }

  const handleUpdateVoiceCard = (
    id: string,
    updates: Partial<{ role: string; voiceId: string }>
  ) => {
    setVoiceCards((prev) =>
      prev.map((card) => (card.id === id ? { ...card, ...updates } : card))
    )
  }

  const voiceRoleError = React.useMemo(() => {
    if (!useVoiceRoles) return null
    if (voiceCards.length < 1) {
      return t(
        "playground:tts.cloning.voiceRoleSelectAtLeastOne",
        "Select at least one voice."
      )
    }
    if (voiceCards.length > 4) {
      return t(
        "playground:tts.cloning.voiceRoleSelectUpToFour",
        "Select up to four voices."
      )
    }
    const roles = new Set<string>()
    const voices = new Set<string>()
    for (const card of voiceCards) {
      if (!card.voiceId) {
        return t(
          "playground:tts.cloning.voiceRoleNeedsVoice",
          "Each role needs a voice."
        )
      }
      if (roles.has(card.role)) {
        return t(
          "playground:tts.cloning.voiceRoleUniqueRoles",
          "Roles must be unique."
        )
      }
      if (voices.has(card.voiceId)) {
        return t(
          "playground:tts.cloning.voiceRoleUniqueVoices",
          "Voices must be unique."
        )
      }
      roles.add(card.role)
      voices.add(card.voiceId)
    }
    return null
  }, [t, useVoiceRoles, voiceCards])

  const roleVoiceOptions = React.useMemo(() => {
    return customVoices.map((voice) => ({
      label: `Custom: ${voice.name || voice.voice_id}`,
      value: `custom:${voice.voice_id}`
    }))
  }, [customVoices])

  const resolveVoiceForId = (voiceId: string) => {
    const key = voiceId.replace("custom:", "")
    return customVoices.find((voice) => voice.voice_id === key) || null
  }

  const activeProvider = providerOptions.find((opt) => opt.value === uploadProvider)
  const requirement = uploadProvider ? getVoiceProviderRequirement(uploadProvider) : null
  const providerUnavailable = activeProvider ? !activeProvider.available : false

  return (
    <div className="space-y-4">
      <div>
        <Text strong>Upload a custom voice</Text>
        <Paragraph className="!mb-1 text-xs text-text-subtle">
          Upload a short voice sample to enable cloning on supported providers. Use the
          generated voice in TTS requests as <Text code>custom:&lt;voice_id&gt;</Text>.
        </Paragraph>
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        <div>
          <label className="block text-xs mb-1 text-text">Provider</label>
          <Select
            className="w-full focus-ring"
            placeholder="Select provider"
            value={uploadProvider || undefined}
            onChange={(value) => setUploadProvider(value)}
            options={providerOptions.map((opt) => ({
              value: opt.value,
              label: (
                <Space size="small">
                  <span>{opt.label}</span>
                  <Tag color={opt.available ? "green" : "default"} bordered>
                    {opt.available ? "Enabled" : "Disabled"}
                  </Tag>
                </Space>
              )
            }))}
          />
        </div>
        <div>
          <label className="block text-xs mb-1 text-text">Voice name</label>
          <Input
            placeholder="Researcher voice"
            value={uploadName}
            onChange={(e) => setUploadName(e.target.value)}
          />
        </div>
        <div>
          <label className="block text-xs mb-1 text-text">Description (optional)</label>
          <Input
            placeholder="Short note about this voice"
            value={uploadDescription}
            onChange={(e) => setUploadDescription(e.target.value)}
          />
        </div>
        <div>
          <label className="block text-xs mb-1 text-text">Reference transcript (optional)</label>
          <Input
            placeholder="Transcript of the sample audio"
            value={referenceText}
            onChange={(e) => setReferenceText(e.target.value)}
          />
        </div>
        <div className="sm:col-span-2">
          <label className="block text-xs mb-1 text-text">Voice sample file</label>
          <Input
            type="file"
            accept="audio/*"
            onChange={(e) => {
              const file = e.target.files?.[0] || null
              setUploadFile(file)
            }}
          />
        </div>
      </div>

      {requirement && (
        <div className="text-xs text-text-subtle">
          <Space size="small" wrap>
            <span>Formats: {requirement.formats.join(" ")}</span>
            {requirement.duration && (
              <span>
                Duration: {requirement.duration.min}-{requirement.duration.max}s
              </span>
            )}
            {requirement.sample_rate && (
              <span>Sample rate: {requirement.sample_rate} Hz</span>
            )}
          </Space>
        </div>
      )}

      {providerUnavailable && (
        <Alert
          title={t(
            "playground:tts.cloning.providerDisabledTitle",
            "Provider disabled on server"
          )}
          variant="warning"
        >
          {t(
            "playground:tts.cloning.providerDisabledBody",
            "Enable this provider in Config_Files/tts_providers_config.yaml to upload voices."
          )}
        </Alert>
      )}

      <div className="flex flex-wrap items-center gap-2">
        <Button
          type="primary"
          icon={<UploadCloud className="h-4 w-4" />}
          loading={uploadMutation.isPending}
          disabled={providerUnavailable || !uploadProvider || !uploadFile || !uploadName.trim()}
          onClick={() => uploadMutation.mutate()}
        >
          Upload voice
        </Button>
        <Text type="secondary" className="text-xs">
          Uploaded voices appear below. Use them with the TTS provider that supports cloning.
        </Text>
      </div>

      <div className="border-t border-border pt-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <Text strong>Custom voices</Text>
          <Button
            size="small"
            onClick={() => queryClient.invalidateQueries({ queryKey: ["tts-custom-voices"] })}
          >
            Refresh
          </Button>
        </div>
        <div className="mt-2 space-y-2">
          <label className="block text-xs text-text">Preview text</label>
          <Input
            value={previewText}
            onChange={(e) => setPreviewText(e.target.value)}
            placeholder="Preview text"
          />
          {previewUrl && (
            <audio className="w-full" controls src={previewUrl} />
          )}
        </div>

        <div className="mt-4 rounded-md border border-border p-3 space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <Text strong>Voice roles</Text>
              <div className="text-xs text-text-subtle">
                Assign roles to 1–4 custom voices for multi-speaker TTS.
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Switch size="small" checked={useVoiceRoles} onChange={setUseVoiceRoles} />
              <Text className="text-xs">Enable</Text>
            </div>
          </div>

          {useVoiceRoles && (
            <>
              <div className="space-y-2">
                {voiceCards.map((card) => (
                  <div
                    key={card.id}
                    className="flex flex-wrap items-center gap-2 rounded-md border border-border/60 p-2"
                  >
                    <Select
                      aria-label="Voice role"
                      className="min-w-[140px]"
                      options={VOICE_ROLE_OPTIONS}
                      value={card.role}
                      onChange={(val) => handleUpdateVoiceCard(card.id, { role: val })}
                    />
                    <Select
                      aria-label="Voice selection"
                      className="min-w-[220px] flex-1"
                      options={roleVoiceOptions}
                      value={card.voiceId}
                      onChange={(val) => handleUpdateVoiceCard(card.id, { voiceId: val })}
                    />
                    <Button
                      size="small"
                      icon={<Play className="h-3 w-3" />}
                      loading={previewingId === card.voiceId.replace("custom:", "")}
                      onClick={() => {
                        const voice = resolveVoiceForId(card.voiceId)
                        if (voice) {
                          handlePreview(voice)
                        }
                      }}
                      disabled={!card.voiceId}
                    >
                      Preview
                    </Button>
                    <Button
                      size="small"
                      type="text"
                      danger
                      icon={<Trash2 className="h-3 w-3" />}
                      onClick={() => handleRemoveVoiceCard(card.id)}
                      disabled={voiceCards.length <= 1}
                    />
                  </div>
                ))}
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <Button size="small" onClick={handleAddVoiceCard} disabled={voiceCards.length >= 4}>
                  Add voice
                </Button>
                <Text type="secondary" className="text-xs">
                  {voiceCards.length}/4 voices selected
                </Text>
                {onSelectVoices && (
                  <Button
                    size="small"
                    type="primary"
                    disabled={Boolean(voiceRoleError)}
                    onClick={() => {
                      if (voiceRoleError) return
                      onSelectVoices(
                        voiceCards.map((card) => ({
                          role: card.role,
                          voiceId: card.voiceId
                        }))
                      )
                    }}
                  >
                    Use roles
                  </Button>
                )}
              </div>
              {voiceRoleError && (
                <Alert
                  title={t(
                    "playground:tts.cloning.voiceRolesAttentionTitle",
                    "Voice roles need attention"
                  )}
                  variant="warning"
                >
                  {voiceRoleError}
                </Alert>
              )}
            </>
          )}
        </div>

        <List
          className="mt-3"
          loading={voicesLoading}
          dataSource={customVoices}
          locale={{ emptyText: "No custom voices yet" }}
          renderItem={(voice) => (
            <List.Item
              key={voice.voice_id}
              actions={[
                <Button
                  key="use"
                  size="small"
                  onClick={() =>
                    onSelectVoice?.(`custom:${voice.voice_id}`, voice.provider)
                  }
                >
                  Use
                </Button>,
                <Button
                  key="preview"
                  size="small"
                  icon={<Play className="h-3 w-3" />}
                  loading={previewingId === voice.voice_id}
                  onClick={() => handlePreview(voice)}
                >
                  Preview
                </Button>,
                <Button
                  key="encode"
                  size="small"
                  icon={<Wand2 className="h-3 w-3" />}
                  loading={
                    encodeMutation.isPending &&
                    encodeMutation.variables?.voiceId === voice.voice_id
                  }
                  onClick={() => {
                    if (!voice.provider) {
                      notification.error({
                        message: "Missing provider",
                        description: "This voice is missing a provider identifier."
                      })
                      return
                    }
                    encodeMutation.mutate({
                      voiceId: voice.voice_id,
                      provider: voice.provider
                    })
                  }}
                >
                  Encode
                </Button>,
                <Popconfirm
                  key="delete"
                  title="Delete this voice?"
                  onConfirm={() => deleteMutation.mutate(voice.voice_id)}
                >
                  <Button
                    size="small"
                    type="text"
                    danger
                    icon={<Trash2 className="h-3 w-3" />}
                  />
                </Popconfirm>
              ]}
            >
              <List.Item.Meta
                title={
                  <div className="flex flex-wrap items-center gap-2">
                    <Text strong>{voice.name || voice.voice_id}</Text>
                    {voice.provider && (
                      <Tag bordered>{getProviderLabel(voice.provider, "tts-engine")}</Tag>
                    )}
                  </div>
                }
                description={
                  <div className="text-xs text-text-subtle space-y-1">
                    <div className="flex flex-wrap gap-3">
                      <span>Duration: {formatSeconds(voice.duration)}</span>
                      <span>Format: {voice.format || "n/a"}</span>
                      <span>Size: {formatBytes(voice.size_bytes)}</span>
                      {voice.sample_rate && <span>Sample rate: {voice.sample_rate} Hz</span>}
                    </div>
                    {voice.description && <div>{voice.description}</div>}
                  </div>
                }
              />
            </List.Item>
          )}
        />
      </div>
    </div>
  )
}
