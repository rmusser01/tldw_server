import React from "react"
import { Button, Input, Modal, Tag, Tooltip, Typography, Collapse, Empty, Spin } from "antd"
import { Play, Search, Volume2 } from "lucide-react"
import { useQuery } from "@tanstack/react-query"
import {
  fetchTtsProviders,
  type TldwTtsProvidersInfo,
  type TldwTtsVoiceInfo,
  type TldwTtsProviderCapabilities
} from "@/services/tldw/audio-providers"
import { PROVIDER_REGISTRY } from "@/utils/provider-registry"
import { OPENAI_TTS_VOICES } from "@/hooks/useTtsProviderData"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { normalizeTtsProviderKey } from "@/services/tldw/tts-provider-keys"

const { Text } = Typography

export type VoiceSelection = {
  provider: string
  backend?: string
  voice: string
  model?: string
}

type VoicePickerModalProps = {
  open: boolean
  onClose: () => void
  onSelect: (selection: VoiceSelection) => void
  /** Current server provider info (passed to avoid duplicate queries) */
  providersInfo?: TldwTtsProvidersInfo | null
  allowFallback?: boolean
}

const RECENT_VOICES_KEY = "tts-recent-voices"
const MAX_RECENT = 5

const getRecentVoices = (): VoiceSelection[] => {
  try {
    const stored = localStorage.getItem(RECENT_VOICES_KEY)
    return stored ? JSON.parse(stored) : []
  } catch {
    return []
  }
}

const saveRecentVoice = (selection: VoiceSelection) => {
  try {
    const existing = getRecentVoices()
    const filtered = existing.filter(
      (v) =>
        !(
          v.provider === selection.provider &&
          v.backend === selection.backend &&
          v.voice === selection.voice
        )
    )
    const next = [selection, ...filtered].slice(0, MAX_RECENT)
    localStorage.setItem(RECENT_VOICES_KEY, JSON.stringify(next))
  } catch {}
}

type ProviderGroup = {
  key: string
  providerKey: string
  backend?: string
  model?: string
  label: string
  category: "local" | "cloud"
  caps?: TldwTtsProviderCapabilities
  voices: TldwTtsVoiceInfo[]
}

const categorizeProvider = (key: string): "local" | "cloud" => {
  const cloudProviders = new Set(["elevenlabs", "openai", "openrouter"])
  if (cloudProviders.has(key)) return "cloud"
  return "local"
}

const getProviderLabel = (key: string): string => {
  const meta = PROVIDER_REGISTRY[key]
  return meta?.ttsLabel || meta?.label || key
}

export const VoicePickerModal: React.FC<VoicePickerModalProps> = ({
  open,
  onClose,
  onSelect,
  providersInfo: externalProvidersInfo,
  allowFallback = true
}) => {
  const [searchQuery, setSearchQuery] = React.useState("")
  const [previewingVoice, setPreviewingVoice] = React.useState<string | null>(null)
  const previewAudioRef = React.useRef<HTMLAudioElement | null>(null)
  const previewBlobUrlRef = React.useRef<string | null>(null)

  // Fetch providers if not passed externally
  const { data: fetchedProvidersInfo, isLoading } = useQuery<TldwTtsProvidersInfo | null>({
    queryKey: ["tldw-tts-providers-picker"],
    queryFn: () => fetchTtsProviders(),
    enabled: open && !externalProvidersInfo
  })

  const providersInfo = externalProvidersInfo || fetchedProvidersInfo

  const recentVoices = open ? getRecentVoices() : []

  // Build provider groups
  const providerGroups = React.useMemo((): ProviderGroup[] => {
    const groups: ProviderGroup[] = []

    // Add browser TTS
    groups.push({
      key: "browser",
      providerKey: "browser",
      label: "Browser TTS",
      category: "local",
      voices: [{ id: "default", name: "System Default" }]
    })

    // Add OpenAI voices
    const allOpenAiVoices = new Set<string>()
    Object.values(OPENAI_TTS_VOICES).forEach((list) =>
      list.forEach((v) => allOpenAiVoices.add(v.value))
    )
    groups.push({
      key: "openai",
      providerKey: "openai",
      label: "OpenAI",
      category: "cloud",
      voices: Array.from(allOpenAiVoices).map((v) => ({
        id: v,
        name: v
      }))
    })

    // Add server providers
    if (providersInfo) {
      const { providers, voices } = providersInfo
      for (const [provKey, caps] of Object.entries(providers)) {
        const normalizedKey = normalizeTtsProviderKey(provKey)
        // Skip if already handled
        if (normalizedKey === "browser" || normalizedKey === "openai") continue

        const provVoices = voices[provKey] || voices[normalizedKey] || []
        groups.push({
          key: provKey,
          providerKey: normalizedKey,
          backend:
            providersInfo.supports_explicit_backend === true
              ? provKey
              : undefined,
          model:
            caps.default_model ||
            caps.models?.[0] ||
            (providersInfo.supports_explicit_backend === true
              ? undefined
              : normalizedKey),
          label:
            caps.display_name ||
            caps.provider_name ||
            getProviderLabel(normalizedKey) ||
            provKey,
          category: categorizeProvider(normalizedKey),
          caps,
          voices: provVoices
        })
      }
    }

    return groups
  }, [providersInfo])

  // Filter groups by search
  const filteredGroups = React.useMemo(() => {
    if (!searchQuery.trim()) return providerGroups
    const q = searchQuery.toLowerCase()
    return providerGroups
        .map((group) => ({
        ...group,
        voices: group.voices.filter(
          (v) =>
            (v.name || "").toLowerCase().includes(q) ||
            (v.voice_id || v.id || "").toLowerCase().includes(q) ||
            group.label.toLowerCase().includes(q) ||
            (v.language || "").toLowerCase().includes(q)
        )
      }))
      .filter((group) => group.voices.length > 0)
  }, [providerGroups, searchQuery])

  const localGroups = filteredGroups.filter((g) => g.category === "local")
  const cloudGroups = filteredGroups.filter((g) => g.category === "cloud")

  const handleSelect = (group: ProviderGroup, voice: TldwTtsVoiceInfo) => {
    const voiceId = voice.voice_id || voice.id || voice.name || ""
    const providerKey = group.providerKey
    const isDirectProvider =
      providerKey === "browser" ||
      providerKey === "openai" ||
      providerKey === "elevenlabs"
    const selection: VoiceSelection = {
      provider: isDirectProvider ? providerKey : "tldw",
      backend: group.backend,
      voice: voiceId,
      model: isDirectProvider ? undefined : group.model
    }
    saveRecentVoice(selection)
    onSelect(selection)
    onClose()
  }

  const handlePreview = async (group: ProviderGroup, voice: TldwTtsVoiceInfo) => {
    const voiceId = voice.voice_id || voice.id || voice.name || ""
    const providerKey = group.providerKey
    if (previewingVoice === voiceId) {
      // Stop preview
      previewAudioRef.current?.pause()
      setPreviewingVoice(null)
      return
    }

    setPreviewingVoice(voiceId)
    try {
      // Use preview_url if available
      if (voice.preview_url) {
        const audio = new Audio(voice.preview_url)
        previewAudioRef.current = audio
        audio.onended = () => setPreviewingVoice(null)
        audio.onerror = () => setPreviewingVoice(null)
        await audio.play()
        return
      }

      // Generate a quick preview via the server
      const model = group.backend
        ? group.model
        : providerKey !== "browser" &&
            providerKey !== "openai" &&
            providerKey !== "elevenlabs"
          ? providerKey
          : undefined
      const provider = providerKey === "browser" || providerKey === "openai" || providerKey === "elevenlabs"
        ? providerKey
        : "tldw"

      if (provider === "browser") {
        // Use browser speech synthesis
        const utterance = new SpeechSynthesisUtterance("Hello, this is a voice preview.")
        speechSynthesis.speak(utterance)
        utterance.onend = () => setPreviewingVoice(null)
        return
      }

      const result = await tldwClient.synthesizeSpeechDetailed(
        "Hello, this is a voice preview.",
        {
          model: model || "tts-1",
          voice: voiceId,
          responseFormat: "mp3",
          backend: group.backend,
          allowFallback: group.backend ? allowFallback : undefined
        }
      )
      const blob = new Blob([result.buffer], { type: "audio/mpeg" })
      const url = URL.createObjectURL(blob)
      // Track blob URL so cleanup effect can revoke it if modal closes during playback
      previewBlobUrlRef.current = url
      const audio = new Audio(url)
      previewAudioRef.current = audio
      audio.onended = () => {
        setPreviewingVoice(null)
        URL.revokeObjectURL(url)
        previewBlobUrlRef.current = null
      }
      audio.onerror = () => {
        setPreviewingVoice(null)
        URL.revokeObjectURL(url)
        previewBlobUrlRef.current = null
      }
      await audio.play()
    } catch {
      setPreviewingVoice(null)
    }
  }

  // Cleanup preview audio on close
  React.useEffect(() => {
    if (!open) {
      previewAudioRef.current?.pause()
      setPreviewingVoice(null)
      setSearchQuery("")
      // Revoke any outstanding preview blob URL to prevent memory leak
      if (previewBlobUrlRef.current) {
        URL.revokeObjectURL(previewBlobUrlRef.current)
        previewBlobUrlRef.current = null
      }
    }
  }, [open])

  const renderVoiceList = (group: ProviderGroup) => (
    <div className="grid grid-cols-1 gap-1 sm:grid-cols-2">
      {group.voices.map((voice, idx) => {
        const voiceId = voice.voice_id || voice.id || voice.name || `voice-${idx}`
        const isActive = previewingVoice === voiceId
        return (
          <div
            key={voiceId}
            className="flex items-center gap-2 rounded-md border border-border px-3 py-2 transition-colors hover:bg-surface-hover cursor-pointer"
            onClick={() => handleSelect(group, voice)}
            role="option"
            aria-selected={false}
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === "Enter") handleSelect(group, voice)
            }}
          >
            <Volume2 className="h-3.5 w-3.5 shrink-0 text-text-muted" />
            <div className="flex-1 min-w-0">
              <Text className="block truncate text-sm">
                {voice.name || voice.voice_id || voice.id || `Voice ${idx + 1}`}
              </Text>
              {voice.language && (
                <Text className="text-xs text-text-muted">{voice.language}</Text>
              )}
            </div>
            {/* Capability badges */}
            {group.caps?.supports_streaming && (
              <Tag className="text-[10px]" variant="filled">
                Stream
              </Tag>
            )}
            {group.caps?.supports_voice_cloning && (
              <Tag className="text-[10px]" variant="filled">
                Clone
              </Tag>
            )}
            {/* Preview button */}
            <Tooltip title={isActive ? "Stop preview" : "Preview voice"}>
              <Button
                type="text"
                size="small"
                loading={isActive}
                icon={<Play className="h-3 w-3" />}
                onClick={(e) => {
                  e.stopPropagation()
                  handlePreview(group, voice)
                }}
                aria-label={`Preview ${voice.name || voice.voice_id || voice.id}`}
              />
            </Tooltip>
          </div>
        )
      })}
    </div>
  )

  const renderGroupSection = (title: string, groups: ProviderGroup[]) => {
    if (groups.length === 0) return null
    return (
      <div className="space-y-2">
        <Text strong className="text-xs uppercase tracking-wide text-text-muted">
          {title}
        </Text>
        <Collapse
          ghost
          defaultActiveKey={groups.map((g) => g.key)}
          items={groups.map((group) => ({
            key: group.key,
            label: (
              <div className="flex items-center gap-2">
                <span>{group.label}</span>
                <Tag className="text-[10px]" variant="filled">
                  {group.voices.length}
                </Tag>
                {group.caps && (
                  <span
                    className={`h-2 w-2 rounded-full ${
                      group.caps.provider_name ? "bg-green-500" : "bg-gray-400"
                    }`}
                    title={group.caps.provider_name ? "Available" : "Unknown status"}
                  />
                )}
              </div>
            ),
            children: renderVoiceList(group)
          }))}
        />
      </div>
    )
  }

  return (
    <Modal
      title={
        <div className="flex items-center gap-2">
          <Volume2 className="h-5 w-5" />
          <span>Choose a Voice</span>
        </div>
      }
      open={open}
      onCancel={onClose}
      footer={null}
      width={640}
      styles={{ body: { maxHeight: "70vh", overflowY: "auto" } }}
    >
      {/* Search */}
      <Input
        prefix={<Search className="h-4 w-4 text-text-muted" />}
        placeholder="Search voices, providers..."
        value={searchQuery}
        onChange={(e) => setSearchQuery(e.target.value)}
        allowClear
        className="mb-4"
        aria-label="Search voices"
        autoFocus
      />

      {/* Recent voices */}
      {!searchQuery.trim() && recentVoices.length > 0 && (
        <div className="mb-4">
          <Text strong className="mb-1.5 block text-xs uppercase tracking-wide text-text-muted">
            Recent
          </Text>
          <div className="flex flex-wrap gap-1.5">
            {recentVoices.map((rv, i) => (
              <Tag
                key={`recent-${i}`}
                className="cursor-pointer"
                onClick={() => {
                  onSelect(rv)
                  onClose()
                }}
              >
                {rv.backend
                  ? `${rv.backend}/${rv.voice}`
                  : rv.model
                    ? `${rv.model}/${rv.voice}`
                    : `${rv.provider}/${rv.voice}`}
              </Tag>
            ))}
          </div>
        </div>
      )}

      {isLoading && (
        <div className="flex justify-center py-8">
          <Spin tip="Loading voices..." />
        </div>
      )}

      {!isLoading && filteredGroups.length === 0 && (
        <Empty description="No voices found" />
      )}

      {!isLoading && (
        <div className="space-y-4">
          {renderGroupSection("Local Engines", localGroups)}
          {renderGroupSection("Cloud Providers", cloudGroups)}
        </div>
      )}
    </Modal>
  )
}

export default VoicePickerModal
