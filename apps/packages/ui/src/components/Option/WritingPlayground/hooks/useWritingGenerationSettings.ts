/**
 * Hook: useWritingGenerationSettings
 *
 * Manages generation parameters (advanced extra_body), logit bias,
 * capabilities queries, and provider-specific feature detection.
 */

import React from "react"
import { useQuery } from "@tanstack/react-query"
import type { TFunction } from "i18next"
import {
  getWritingCapabilities,
  type WritingExtraBodyCompat
} from "@/services/writing-playground"
import { resolveApiProviderForModel } from "@/utils/resolve-api-provider"
import {
  formatLogitBiasValue,
  normalizeLogitBiasValue,
  parseLogitBiasInput,
  withLogitBiasEntry,
  withTokenIdsPresetLogitBias,
  withTokenIdPresetLogitBias,
  withoutLogitBiasEntry
} from "../writing-logit-bias-utils"
import {
  parseExtraBodyJsonObject
} from "../extra-body-utils"
import {
  isRecord,
  normalizeStringArrayValue,
  type WritingSessionSettings
} from "./utils"

export interface UseWritingGenerationSettingsDeps {
  isOnline: boolean
  hasWriting: boolean
  selectedModel: string | undefined
  apiProviderOverride: string | undefined
  settings: WritingSessionSettings
  advancedExtraBody: Record<string, unknown>
  updateSetting: (partial: Partial<WritingSessionSettings>, nextStopInput?: string) => void
  settingsDisabled: boolean
  logitBiasInput: string
  setLogitBiasInput: React.Dispatch<React.SetStateAction<string>>
  logitBiasError: string | null
  setLogitBiasError: React.Dispatch<React.SetStateAction<string | null>>
  logitBiasTokenInput: string
  setLogitBiasTokenInput: React.Dispatch<React.SetStateAction<string>>
  logitBiasValueInput: number | null
  setLogitBiasValueInput: React.Dispatch<React.SetStateAction<number | null>>
  bannedTokensInput: string
  setBannedTokensInput: React.Dispatch<React.SetStateAction<string>>
  drySequenceBreakersInput: string
  setDrySequenceBreakersInput: React.Dispatch<React.SetStateAction<string>>
  extraBodyJsonModalOpen: boolean
  setExtraBodyJsonModalOpen: React.Dispatch<React.SetStateAction<boolean>>
  extraBodyJsonDraft: string
  setExtraBodyJsonDraft: React.Dispatch<React.SetStateAction<string>>
  extraBodyJsonError: string | null
  setExtraBodyJsonError: React.Dispatch<React.SetStateAction<string | null>>
  t: TFunction
}

export function useWritingGenerationSettings(deps: UseWritingGenerationSettingsDeps) {
  const {
    isOnline,
    hasWriting,
    selectedModel,
    apiProviderOverride,
    settings,
    advancedExtraBody,
    updateSetting,
    settingsDisabled,
    logitBiasInput,
    setLogitBiasInput,
    logitBiasError,
    setLogitBiasError,
    logitBiasTokenInput,
    setLogitBiasTokenInput,
    logitBiasValueInput,
    setLogitBiasValueInput,
    bannedTokensInput,
    setBannedTokensInput,
    drySequenceBreakersInput,
    setDrySequenceBreakersInput,
    extraBodyJsonModalOpen,
    setExtraBodyJsonModalOpen,
    extraBodyJsonDraft,
    setExtraBodyJsonDraft,
    extraBodyJsonError,
    setExtraBodyJsonError,
    t
  } = deps

  // --- Capabilities query ---
  const { data: requestedCaps, isLoading: requestedCapsLoading } = useQuery({
    queryKey: [
      "writing-capabilities",
      "requested",
      selectedModel || "",
      apiProviderOverride || ""
    ],
    queryFn: async () => {
      const provider = await resolveApiProviderForModel({
        modelId: selectedModel,
        explicitProvider: apiProviderOverride
      })
      return await getWritingCapabilities({
        provider,
        model: selectedModel || undefined,
        includeProviders: false
      })
    },
    enabled: isOnline && hasWriting && Boolean(selectedModel),
    staleTime: 60 * 1000
  })

  const extraBodyCompat: WritingExtraBodyCompat | null =
    requestedCaps?.requested?.extra_body_compat ?? null
  const requestedFeatures = isRecord(requestedCaps?.requested?.features)
    ? requestedCaps.requested.features
    : {}
  const requestedSupportedFields = Array.isArray(
    requestedCaps?.requested?.supported_fields
  )
    ? requestedCaps.requested.supported_fields
        .map((field) => String(field || "").trim())
        .filter(Boolean)
    : []
  const requestedLogprobsSupported = requestedFeatures.logprobs === true
  const requestedLogprobsExplicitlyUnsupported = requestedFeatures.logprobs === false
  const requestedTopLogprobsSupported = requestedSupportedFields.includes(
    "top_logprobs"
  )

  // --- Extra body derived ---
  const knownExtraBodyParams = React.useMemo(
    () =>
      Array.isArray(extraBodyCompat?.known_params)
        ? extraBodyCompat.known_params
            .map((entry) => String(entry || "").trim())
            .filter(Boolean)
        : [],
    [extraBodyCompat?.known_params]
  )
  const knownExtraBodyParamSet = React.useMemo(
    () => new Set(knownExtraBodyParams),
    [knownExtraBodyParams]
  )
  const supportsAdvancedCompat = extraBodyCompat?.supported !== false

  const shouldShowAdvancedParam = React.useCallback(
    (key: string) => {
      if (Object.prototype.hasOwnProperty.call(advancedExtraBody, key)) {
        return true
      }
      if (knownExtraBodyParamSet.size === 0) return false
      return knownExtraBodyParamSet.has(key)
    },
    [advancedExtraBody, knownExtraBodyParamSet]
  )

  const tokenInspectorSupportsLogitBias =
    supportsAdvancedCompat && shouldShowAdvancedParam("logit_bias")

  // --- Advanced extra body field helpers ---
  const updateAdvancedExtraBodyField = React.useCallback(
    (key: string, value: unknown) => {
      const nextAdvanced = { ...advancedExtraBody }
      const shouldRemove =
        value == null ||
        (typeof value === "string" && value.trim().length === 0) ||
        (Array.isArray(value) && value.length === 0)
      if (shouldRemove) {
        delete nextAdvanced[key]
      } else {
        nextAdvanced[key] = value
      }
      updateSetting({ advanced_extra_body: nextAdvanced })
    },
    [advancedExtraBody, updateSetting]
  )

  const getAdvancedNumberValue = React.useCallback(
    (key: string): number | null => {
      const raw = advancedExtraBody[key]
      if (typeof raw === "number" && Number.isFinite(raw)) {
        return raw
      }
      if (typeof raw === "string" && raw.trim()) {
        const parsed = Number(raw)
        return Number.isFinite(parsed) ? parsed : null
      }
      return null
    },
    [advancedExtraBody]
  )

  // --- Logit bias ---
  const applyLogitBiasObject = React.useCallback(
    (nextLogitBias: Record<string, number>) => {
      updateAdvancedExtraBodyField(
        "logit_bias",
        Object.keys(nextLogitBias).length > 0 ? nextLogitBias : null
      )
      setLogitBiasInput(formatLogitBiasValue(nextLogitBias))
      setLogitBiasError(null)
    },
    [updateAdvancedExtraBodyField, setLogitBiasInput, setLogitBiasError]
  )

  const applyTokenInspectorLogitBiasPreset = React.useCallback(
    (tokenId: number, preset: "ban" | "favor") => {
      const next = withTokenIdPresetLogitBias(
        advancedExtraBody.logit_bias,
        tokenId,
        preset
      )
      applyLogitBiasObject(next)
    },
    [advancedExtraBody.logit_bias, applyLogitBiasObject]
  )

  const applyTokenInspectorLogitBiasPresetBatch = React.useCallback(
    (tokenIds: number[], preset: "ban" | "favor") => {
      const next = withTokenIdsPresetLogitBias(
        advancedExtraBody.logit_bias,
        tokenIds,
        preset
      )
      applyLogitBiasObject(next)
    },
    [advancedExtraBody.logit_bias, applyLogitBiasObject]
  )

  const logitBiasEntries = React.useMemo(
    () =>
      Object.entries(normalizeLogitBiasValue(advancedExtraBody.logit_bias)).sort(
        ([left], [right]) => left.localeCompare(right)
      ),
    [advancedExtraBody.logit_bias]
  )

  // --- Extra body JSON editor ---
  const openExtraBodyJsonEditor = React.useCallback(() => {
    setExtraBodyJsonDraft(
      JSON.stringify(advancedExtraBody || {}, null, 2) || "{}"
    )
    setExtraBodyJsonError(null)
    setExtraBodyJsonModalOpen(true)
  }, [advancedExtraBody, setExtraBodyJsonDraft, setExtraBodyJsonError, setExtraBodyJsonModalOpen])

  const applyExtraBodyJsonDraft = React.useCallback(() => {
    const parsed = parseExtraBodyJsonObject(extraBodyJsonDraft)
    if (parsed.error) {
      setExtraBodyJsonError(parsed.error)
      return
    }
    if (Object.prototype.hasOwnProperty.call(parsed.value, "logit_bias")) {
      const normalized = parseLogitBiasInput(
        JSON.stringify(parsed.value.logit_bias)
      )
      if (normalized.error) {
        setExtraBodyJsonError(`Invalid logit_bias: ${normalized.error}`)
        return
      }
      if (normalized.value) {
        parsed.value.logit_bias = normalized.value
      } else {
        delete parsed.value.logit_bias
      }
    }
    updateSetting({
      advanced_extra_body: parsed.value
    })
    setBannedTokensInput(
      normalizeStringArrayValue(parsed.value.banned_tokens).join("\n")
    )
    setDrySequenceBreakersInput(
      normalizeStringArrayValue(parsed.value.dry_sequence_breakers).join("\n")
    )
    setLogitBiasInput(formatLogitBiasValue(parsed.value.logit_bias))
    setLogitBiasError(null)
    setLogitBiasTokenInput("")
    setLogitBiasValueInput(null)
    setExtraBodyJsonError(null)
    setExtraBodyJsonModalOpen(false)
  }, [
    extraBodyJsonDraft,
    updateSetting,
    setBannedTokensInput,
    setDrySequenceBreakersInput,
    setLogitBiasInput,
    setLogitBiasError,
    setLogitBiasTokenInput,
    setLogitBiasValueInput,
    setExtraBodyJsonError,
    setExtraBodyJsonModalOpen
  ])

  // --- Derived flags ---
  const advancedExtraBodyUnknownKeys = React.useMemo(
    () =>
      Object.keys(advancedExtraBody || {}).filter(
        (key) =>
          knownExtraBodyParamSet.size > 0 && !knownExtraBodyParamSet.has(key)
      ),
    [advancedExtraBody, knownExtraBodyParamSet]
  )
  const hasAdvancedSettingsValues =
    Object.keys(advancedExtraBody || {}).length > 0
  const showAdvancedSamplerControls =
    knownExtraBodyParams.length > 0 || hasAdvancedSettingsValues

  const logprobsControlsDisabled =
    settingsDisabled || requestedLogprobsExplicitlyUnsupported
  const topLogprobsControlsDisabled =
    logprobsControlsDisabled || !settings.logprobs

  return {
    // capabilities
    requestedCaps,
    requestedCapsLoading,
    extraBodyCompat,
    requestedLogprobsSupported,
    requestedLogprobsExplicitlyUnsupported,
    requestedTopLogprobsSupported,
    requestedSupportedFields,
    // extra body
    knownExtraBodyParams,
    knownExtraBodyParamSet,
    supportsAdvancedCompat,
    shouldShowAdvancedParam,
    tokenInspectorSupportsLogitBias,
    updateAdvancedExtraBodyField,
    getAdvancedNumberValue,
    // logit bias
    applyLogitBiasObject,
    applyTokenInspectorLogitBiasPreset,
    applyTokenInspectorLogitBiasPresetBatch,
    logitBiasEntries,
    // json editor
    openExtraBodyJsonEditor,
    applyExtraBodyJsonDraft,
    // derived
    advancedExtraBodyUnknownKeys,
    hasAdvancedSettingsValues,
    showAdvancedSamplerControls,
    logprobsControlsDisabled,
    topLogprobsControlsDisabled
  }
}
