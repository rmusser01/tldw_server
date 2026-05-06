import React from 'react'
import { useStorage } from '@plasmohq/storage/hook'
import type { CommonOptions, IngestPreset, TypeDefaults } from "../QuickIngest/types"
import {
  DEFAULT_PRESET,
  detectPreset,
  getPresetConfig,
  resolvePresetMap,
  type PresetMap
} from "../QuickIngest/presets"

// ---------------------------------------------------------------------------
// Deps interface
// ---------------------------------------------------------------------------

export interface UseIngestPresetsDeps {
  open: boolean
  /** Common ingest options from the options hook */
  common: CommonOptions
  setCommon: (v: CommonOptions) => void
  storeRemote: boolean
  setStoreRemote: (v: boolean) => void
  reviewBeforeStorage: boolean
  setReviewBeforeStorage: (v: boolean) => void
  normalizedTypeDefaults: TypeDefaults
  setTypeDefaults: (v: TypeDefaults) => void
  advancedValues: Record<string, any>
  setAdvancedValues: React.Dispatch<React.SetStateAction<Record<string, any>>>
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useIngestPresets(deps: UseIngestPresetsDeps) {
  const {
    open,
    common,
    setCommon,
    storeRemote,
    setStoreRemote,
    reviewBeforeStorage,
    setReviewBeforeStorage,
    normalizedTypeDefaults,
    setTypeDefaults,
    advancedValues,
    setAdvancedValues,
  } = deps

  // ---- persisted preset state ----
  const [activePreset, setActivePreset] = useStorage<IngestPreset>(
    "quickIngestPreset",
    DEFAULT_PRESET
  )
  const [presetConfigs] = useStorage<PresetMap>(
    "quickIngestPresetConfigs",
    resolvePresetMap()
  )
  const resolvedPresets = React.useMemo(
    () => resolvePresetMap(presetConfigs),
    [presetConfigs]
  )

  // Track if we're currently applying a preset to avoid auto-switch to Custom
  const applyingPresetRef = React.useRef(false)

  // Initialize options from preset when modal opens
  React.useEffect(() => {
    if (!open) return
    applyingPresetRef.current = true

    if (activePreset && activePreset !== "custom") {
      const presetConfig = getPresetConfig(activePreset, resolvedPresets)
      if (presetConfig) {
        setCommon(presetConfig.common)
        setStoreRemote(presetConfig.storeRemote)
        setReviewBeforeStorage(presetConfig.reviewBeforeStorage)
        setTypeDefaults(presetConfig.typeDefaults)
        setAdvancedValues(presetConfig.advancedValues ?? {})
      }
    }

    setTimeout(() => {
      applyingPresetRef.current = false
    }, 100)
  }, [open]) // Only run on modal open

  // Detect option changes and auto-switch to Custom preset
  React.useEffect(() => {
    if (!open || applyingPresetRef.current) return
    if (activePreset === "custom") return

    const currentConfig = {
      common,
      storeRemote,
      reviewBeforeStorage: reviewBeforeStorage ?? false,
      typeDefaults: normalizedTypeDefaults,
      advancedValues
    }

    const detectedPreset = detectPreset(currentConfig, resolvedPresets)
    if (detectedPreset !== activePreset) {
      setActivePreset("custom")
    }
  }, [
    common,
    storeRemote,
    reviewBeforeStorage,
    normalizedTypeDefaults,
    activePreset,
    open,
    advancedValues,
    resolvedPresets
  ])

  // Handler for preset selection
  const handlePresetChange = React.useCallback(
    (preset: IngestPreset) => {
      applyingPresetRef.current = true
      setActivePreset(preset)

      if (preset !== "custom") {
        const presetConfig = getPresetConfig(preset, resolvedPresets)
        if (presetConfig) {
          setCommon(presetConfig.common)
          setStoreRemote(presetConfig.storeRemote)
          setReviewBeforeStorage(presetConfig.reviewBeforeStorage)
          setTypeDefaults(presetConfig.typeDefaults)
          setAdvancedValues(presetConfig.advancedValues ?? {})
        }
      }

      setTimeout(() => {
        applyingPresetRef.current = false
      }, 100)
    },
    [
      setActivePreset,
      setCommon,
      setStoreRemote,
      setReviewBeforeStorage,
      setTypeDefaults,
      setAdvancedValues,
      resolvedPresets
    ]
  )

  // Handler for reset to defaults
  const handlePresetReset = React.useCallback(() => {
    handlePresetChange(DEFAULT_PRESET)
  }, [handlePresetChange])

  return {
    activePreset: activePreset ?? DEFAULT_PRESET,
    setActivePreset,
    presetConfigs,
    resolvedPresets,
    handlePresetChange,
    handlePresetReset,
  }
}

export { DEFAULT_PRESET }
