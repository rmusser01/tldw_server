import * as React from "react"
import { Button, Input, Select, Tag, Tooltip } from "antd"
import { CopyPlus, Pencil, Play, Save, Star, Trash2 } from "lucide-react"

import { useAntdNotification } from "@/hooks/useAntdNotification"
import { useAudioPresets } from "@/hooks/useAudioPresets"
import type { AudioPreset, AudioPresetKind } from "@/types/audio-presets"

type AudioPresetControlsProps = {
  kind: Extract<AudioPresetKind, "tts" | "stt">
  currentConfig: Record<string, unknown>
  capabilityAssumptions?: Record<string, unknown>
  onApply: (config: Record<string, unknown>, preset: AudioPreset) => Promise<void> | void
  className?: string
}

const kindLabel = {
  tts: "TTS",
  stt: "STT"
} as const

const buildDuplicatePresetName = (
  sourceName: string,
  existingPresets: AudioPreset[]
): string => {
  const baseName = `${sourceName.trim() || "Preset"} copy`
  const existingNames = new Set(
    existingPresets.map((preset) => preset.name.trim().toLowerCase())
  )
  if (!existingNames.has(baseName.toLowerCase())) return baseName
  let index = 2
  while (existingNames.has(`${baseName} ${index}`.toLowerCase())) {
    index += 1
  }
  return `${baseName} ${index}`
}

export const AudioPresetControls: React.FC<AudioPresetControlsProps> = ({
  kind,
  currentConfig,
  capabilityAssumptions,
  onApply,
  className
}) => {
  const notification = useAntdNotification()
  const {
    presets,
    loading,
    createPreset,
    updatePreset,
    deletePreset,
    validatePreset,
    creating,
    updating,
    deleting,
    validating
  } = useAudioPresets({ kind })
  const [selectedId, setSelectedId] = React.useState<string | undefined>()
  const [name, setName] = React.useState("")
  const syncedPresetIdRef = React.useRef<string | undefined>()

  const selectedPreset = React.useMemo(
    () => presets.find((preset) => preset.id === selectedId),
    [presets, selectedId]
  )

  React.useEffect(() => {
    if (selectedPreset && selectedPreset.id !== syncedPresetIdRef.current) {
      syncedPresetIdRef.current = selectedPreset.id
      setName(selectedPreset.name)
      return
    }
    if (!selectedId && presets.length > 0) {
      const defaultPreset = presets.find((preset) => preset.is_default)
      const next = defaultPreset || presets[0]
      syncedPresetIdRef.current = next.id
      setSelectedId(next.id)
      setName(next.name)
      return
    }
    if (selectedId && !selectedPreset) {
      syncedPresetIdRef.current = undefined
      setSelectedId(undefined)
      setName("")
    }
  }, [presets, selectedId, selectedPreset])

  const requireName = React.useCallback(() => {
    const trimmed = name.trim()
    if (!trimmed) {
      notification.warning({
        message: "Preset name required"
      })
      return null
    }
    return trimmed
  }, [name, notification])

  const notifyFailure = React.useCallback(
    (message: string, error: unknown) => {
      notification.error({
        message,
        description:
          error instanceof Error ? error.message : "Preset update failed. Try again."
      })
    },
    [notification]
  )

  const handleSave = React.useCallback(async () => {
    const trimmed = requireName()
    if (!trimmed) return
    try {
      const created = await createPreset({
        kind,
        name: trimmed,
        config: currentConfig,
        capability_assumptions: capabilityAssumptions || {},
        is_default: presets.length === 0
      })
      setSelectedId(created.id)
      notification.success({ message: "Preset saved" })
    } catch (error: unknown) {
      notifyFailure("Preset save failed", error)
    }
  }, [
    capabilityAssumptions,
    createPreset,
    currentConfig,
    kind,
    notifyFailure,
    notification,
    presets.length,
    requireName
  ])

  const handleApply = React.useCallback(async () => {
    if (!selectedPreset) return
    try {
      const validation = await validatePreset(selectedPreset.id)
      if (validation.warnings.length > 0) {
        notification.warning({
          message: "Preset needs attention",
          description: validation.warnings.map((warning) => warning.message).join(" ")
        })
      }
      await onApply(selectedPreset.config || {}, selectedPreset)
    } catch (error: unknown) {
      notifyFailure("Preset validation failed", error)
    }
  }, [notification, notifyFailure, onApply, selectedPreset, validatePreset])

  const handleDuplicate = React.useCallback(async () => {
    if (!selectedPreset) return
    try {
      const created = await createPreset({
        kind,
        name: buildDuplicatePresetName(selectedPreset.name, presets),
        description: selectedPreset.description,
        favorite: selectedPreset.favorite,
        config: selectedPreset.config,
        capability_assumptions: selectedPreset.capability_assumptions
      })
      setSelectedId(created.id)
      notification.success({ message: "Preset duplicated" })
    } catch (error: unknown) {
      notifyFailure("Preset duplicate failed", error)
    }
  }, [createPreset, kind, notification, notifyFailure, presets, selectedPreset])

  const handleRename = React.useCallback(async () => {
    if (!selectedPreset) return
    const trimmed = requireName()
    if (!trimmed) return
    try {
      const updated = await updatePreset(selectedPreset.id, { name: trimmed })
      setSelectedId(updated.id)
      notification.success({ message: "Preset renamed" })
    } catch (error: unknown) {
      notifyFailure("Preset rename failed", error)
    }
  }, [notification, notifyFailure, requireName, selectedPreset, updatePreset])

  const handleFavorite = React.useCallback(async () => {
    if (!selectedPreset) return
    try {
      await updatePreset(selectedPreset.id, { favorite: !selectedPreset.favorite })
    } catch (error: unknown) {
      notifyFailure("Preset favorite update failed", error)
    }
  }, [notifyFailure, selectedPreset, updatePreset])

  const handleDefault = React.useCallback(async () => {
    if (!selectedPreset || selectedPreset.is_default) return
    try {
      await updatePreset(selectedPreset.id, { is_default: true })
    } catch (error: unknown) {
      notifyFailure("Preset default update failed", error)
    }
  }, [notifyFailure, selectedPreset, updatePreset])

  const handleDelete = React.useCallback(async () => {
    if (!selectedPreset) return
    try {
      await deletePreset(selectedPreset.id)
      setSelectedId(undefined)
      setName("")
      notification.info({ message: "Preset deleted" })
    } catch (error: unknown) {
      notifyFailure("Preset delete failed", error)
    }
  }, [deletePreset, notification, notifyFailure, selectedPreset])

  const busy = creating || updating || deleting || validating

  return (
    <div
      className={`flex flex-wrap items-center gap-2 rounded-md border border-border bg-surface px-3 py-2 ${className || ""}`}
      data-testid={`${kind}-preset-controls`}
    >
      <Tag bordered>{kindLabel[kind]} presets</Tag>
      <Select
        size="small"
        loading={loading}
        value={selectedId}
        placeholder="Saved preset"
        className="min-w-[190px]"
        onChange={(value) => setSelectedId(value)}
        options={presets.map((preset) => ({
          value: preset.id,
          label: `${preset.name}${preset.is_default ? " (default)" : ""}${preset.favorite ? " ★" : ""}`
        }))}
      />
      <Input
        size="small"
        value={name}
        onChange={(event) => setName(event.target.value)}
        placeholder="Preset name"
        className="max-w-[220px]"
      />
      <Tooltip title="Apply preset">
        <Button
          size="small"
          aria-label="Apply preset"
          icon={<Play className="h-3.5 w-3.5" />}
          disabled={!selectedPreset}
          loading={validating}
          onClick={() => void handleApply()}
        />
      </Tooltip>
      <Tooltip title="Save current settings">
        <Button
          size="small"
          aria-label="Save current settings"
          icon={<Save className="h-3.5 w-3.5" />}
          loading={creating}
          onClick={() => void handleSave()}
        />
      </Tooltip>
      <Tooltip title="Duplicate selected preset">
        <Button
          size="small"
          aria-label="Duplicate selected preset"
          icon={<CopyPlus className="h-3.5 w-3.5" />}
          disabled={!selectedPreset}
          loading={creating}
          onClick={() => void handleDuplicate()}
        />
      </Tooltip>
      <Tooltip title="Rename selected preset">
        <Button
          size="small"
          aria-label="Rename selected preset"
          icon={<Pencil className="h-3.5 w-3.5" />}
          disabled={!selectedPreset}
          loading={updating}
          onClick={() => void handleRename()}
        />
      </Tooltip>
      <Tooltip title={selectedPreset?.favorite ? "Remove favorite" : "Favorite"}>
        <Button
          size="small"
          aria-label={selectedPreset?.favorite ? "Remove favorite" : "Favorite"}
          type={selectedPreset?.favorite ? "primary" : "default"}
          icon={<Star className="h-3.5 w-3.5" />}
          disabled={!selectedPreset}
          loading={updating}
          onClick={() => void handleFavorite()}
        />
      </Tooltip>
      <Tooltip title="Set as default">
        <Button
          size="small"
          aria-label="Set as default"
          disabled={!selectedPreset || selectedPreset.is_default}
          loading={updating}
          onClick={() => void handleDefault()}
        >
          Default
        </Button>
      </Tooltip>
      <Tooltip title="Delete selected preset">
        <Button
          size="small"
          aria-label="Delete selected preset"
          danger
          icon={<Trash2 className="h-3.5 w-3.5" />}
          disabled={!selectedPreset || busy}
          onClick={() => void handleDelete()}
        />
      </Tooltip>
    </div>
  )
}

export default AudioPresetControls
