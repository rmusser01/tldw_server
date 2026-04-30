import React from "react"

import { bgRequest } from "@/services/background-proxy"
import { toAllowedPath } from "@/services/tldw/path-utils"
import {
  deriveAdminGuardFromError,
  sanitizeAdminErrorMessage,
  type AdminGuardState
} from "@/components/Option/Admin/admin-error-utils"

const RECOMMENDATIONS_PATH = toAllowedPath("/api/v1/setup/admin/audio/recommendations")
const INSTALL_STATUS_PATH = toAllowedPath("/api/v1/setup/admin/install-status")
const PROVISION_PATH = toAllowedPath("/api/v1/setup/admin/audio/provision")
const VERIFY_PATH = toAllowedPath("/api/v1/setup/admin/audio/verify")

const POLL_INTERVAL_MS = 3000
const PROVISION_TIMEOUT_MS = 30 * 60 * 1000
const ACTIVE_INSTALL_STATUSES = new Set(["queued", "running", "in_progress"])

type MachineProfile = {
  platform?: string
  arch?: string
  apple_silicon?: boolean
  cuda_available?: boolean
  free_disk_gb?: number
  network_available_for_downloads?: boolean
}

type AudioResourceProfile = {
  profile_id: string
  label: string
  description?: string | null
  estimated_disk_gb?: number | null
  resource_class?: string | null
  stt_plan?: Array<Record<string, unknown>>
  tts_plan?: Array<Record<string, unknown>>
  tts_choices?: Array<{
    choice_id: string
    label: string
    description?: string | null
  }>
  default_tts_choice?: string | null
}

type AudioBundle = {
  bundle_id: string
  label: string
  description: string
  default_resource_profile?: string
  resource_profiles?: Record<string, AudioResourceProfile>
}

type AudioRecommendation = {
  bundle_id: string
  resource_profile?: string
  selection_key?: string
  label?: string
  reasons?: string[]
  bundle?: AudioBundle
  profile?: AudioResourceProfile
}

type InstallStep = {
  name?: string
  label?: string
  status?: string
}

type InstallStatusSnapshot = {
  status?: string
  steps?: InstallStep[]
  errors?: unknown[]
  bundle_id?: string
  resource_profile?: string
  tts_choice?: string | null
  safe_rerun?: boolean
}

type VerificationResult = {
  status?: string
  bundle_id?: string
  selected_resource_profile?: string
  tts_choice?: string | null
  targets_checked?: string[]
  remediation_items?: Array<{ code?: string; action?: string; message?: string } | string>
}

type RecommendationsResponse = {
  machine_profile?: MachineProfile
  recommendations?: AudioRecommendation[]
  excluded?: AudioRecommendation[]
  catalog?: AudioBundle[]
}

const toError = async (
  response: Awaited<ReturnType<typeof bgRequest>>
): Promise<Error> => {
  const detail =
    typeof response?.error === "string"
      ? response.error
      : typeof response?.data?.detail === "string"
        ? response.data.detail
        : typeof response?.data?.error === "string"
          ? response.data.error
          : ""
  const status = response?.status ?? 500
  const suffix = detail ? ` ${detail}` : ""
  return new Error(`Request failed: ${status}${suffix}`)
}

const requestJson = async <T,>(
  path: ReturnType<typeof toAllowedPath>,
  init?: {
    method?: string
    headers?: Record<string, string>
    body?: string
    timeoutMs?: number
    signal?: AbortSignal
    responseType?: "json" | "text" | "arrayBuffer"
  }
): Promise<T> => {
  const response = await bgRequest<any>({
    path,
    method: init?.method || "GET",
    headers: init?.headers,
    body: init?.body,
    timeoutMs: init?.timeoutMs,
    abortSignal: init?.signal,
    responseType: init?.responseType,
    returnResponse: true
  })
  if (!response.ok) {
    throw await toError(response)
  }
  return response.data as T
}

const deriveSelection = (
  recommendations: AudioRecommendation[],
  catalog: AudioBundle[],
  currentBundleId: string | null,
  currentResourceProfile: string | null
) => {
  const recommendedBundle = recommendations[0]?.bundle
  const fallbackBundle = catalog[0]
  const nextBundleId = currentBundleId || recommendedBundle?.bundle_id || fallbackBundle?.bundle_id || null
  const currentBundle =
    recommendations.find((entry) => entry.bundle_id === nextBundleId)?.bundle ||
    catalog.find((entry) => entry.bundle_id === nextBundleId) ||
    recommendedBundle ||
    fallbackBundle ||
    null

  const recommendedProfile = recommendations.find((entry) => entry.bundle_id === nextBundleId)?.resource_profile
  const defaultProfile =
    currentResourceProfile ||
    recommendedProfile ||
    currentBundle?.default_resource_profile ||
    Object.keys(currentBundle?.resource_profiles || {})[0] ||
    null

  return {
    bundleId: nextBundleId,
    resourceProfile: defaultProfile
  }
}

const deriveDefaultTtsChoice = (profile: AudioResourceProfile | null | undefined): string | null => {
  if (!profile) {
    return null
  }
  if (profile.default_tts_choice && String(profile.default_tts_choice).trim().length > 0) {
    return profile.default_tts_choice
  }
  const firstChoice = Array.isArray(profile.tts_choices) ? profile.tts_choices[0] : null
  return firstChoice?.choice_id || null
}

const resolveSelectedTtsChoice = (
  profile: AudioResourceProfile | null | undefined,
  currentTtsChoice: string | null
): string | null => {
  const availableChoices = Array.isArray(profile?.tts_choices) ? profile.tts_choices : []
  if (
    currentTtsChoice &&
    availableChoices.some((choice) => choice.choice_id === currentTtsChoice)
  ) {
    return currentTtsChoice
  }
  return deriveDefaultTtsChoice(profile)
}

const logNonFatalRefreshError = (context: string, err: unknown) => {
  console.warn(`Audio installer status refresh failed during ${context}.`, err)
}

const buildBundleRequestBody = (
  bundleId: string,
  resourceProfile: string,
  ttsChoice: string | null,
  extra: Record<string, unknown> = {}
) => ({
  bundle_id: bundleId,
  resource_profile: resourceProfile,
  ...(ttsChoice ? { tts_choice: ttsChoice } : {}),
  ...extra
})

export const useAudioInstaller = () => {
  const selectionRef = React.useRef<{
    bundleId: string | null
    resourceProfile: string | null
    ttsChoice: string | null
  }>({ bundleId: null, resourceProfile: null, ttsChoice: null })
  const [loading, setLoading] = React.useState(true)
  const [error, setError] = React.useState<string | null>(null)
  const [adminGuard, setAdminGuard] = React.useState<AdminGuardState>(null)
  const [machineProfile, setMachineProfile] = React.useState<MachineProfile | null>(null)
  const [recommendations, setRecommendations] = React.useState<AudioRecommendation[]>([])
  const [catalog, setCatalog] = React.useState<AudioBundle[]>([])
  const [selectedBundleId, setSelectedBundleId] = React.useState<string | null>(null)
  const [selectedResourceProfile, setSelectedResourceProfile] = React.useState<string | null>(null)
  const [selectedTtsChoice, setSelectedTtsChoice] = React.useState<string | null>(null)
  const [installStatus, setInstallStatus] = React.useState<InstallStatusSnapshot | null>(null)
  const [verification, setVerification] = React.useState<VerificationResult | null>(null)
  const [provisioning, setProvisioning] = React.useState(false)
  const [verifying, setVerifying] = React.useState(false)

  React.useEffect(() => {
    selectionRef.current = {
      bundleId: selectedBundleId,
      resourceProfile: selectedResourceProfile,
      ttsChoice: selectedTtsChoice
    }
  }, [selectedBundleId, selectedResourceProfile, selectedTtsChoice])

  const refreshInstallStatus = React.useCallback(async () => {
    const snapshot = await requestJson<InstallStatusSnapshot>(INSTALL_STATUS_PATH)
    setInstallStatus(snapshot)
    return snapshot
  }, [])

  const load = React.useCallback(async () => {
    try {
      setLoading(true)
      const response = await requestJson<RecommendationsResponse>(RECOMMENDATIONS_PATH)
      const nextRecommendations = Array.isArray(response.recommendations) ? response.recommendations : []
      const nextCatalog = Array.isArray(response.catalog) ? response.catalog : []
      const nextSelection = deriveSelection(
        nextRecommendations,
        nextCatalog,
        selectionRef.current.bundleId,
        selectionRef.current.resourceProfile
      )

      setMachineProfile(response.machine_profile || null)
      setRecommendations(nextRecommendations)
      setCatalog(nextCatalog)
      setSelectedBundleId(nextSelection.bundleId)
      setSelectedResourceProfile(nextSelection.resourceProfile)
      const nextBundle =
        nextRecommendations.find((entry) => entry.bundle_id === nextSelection.bundleId)?.bundle ||
        nextCatalog.find((entry) => entry.bundle_id === nextSelection.bundleId) ||
        null
      const nextProfile =
        nextRecommendations.find(
          (entry) =>
            entry.bundle_id === nextSelection.bundleId &&
            entry.resource_profile === nextSelection.resourceProfile
        )?.profile ||
        (nextBundle?.resource_profiles || {})[nextSelection.resourceProfile || ""] ||
        null
      setSelectedTtsChoice(resolveSelectedTtsChoice(nextProfile, selectionRef.current.ttsChoice))
      setAdminGuard(null)
      setError(null)

      await refreshInstallStatus()
    } catch (err) {
      const guardState = deriveAdminGuardFromError(err)
      setAdminGuard(guardState)
      setError(
        sanitizeAdminErrorMessage(
          err,
          "Unable to load the admin audio installer for this server."
        )
      )
    } finally {
      setLoading(false)
    }
  }, [refreshInstallStatus])

  React.useEffect(() => {
    void load()
  }, [load])

  React.useEffect(() => {
    if (!installStatus?.status || !ACTIVE_INSTALL_STATUSES.has(installStatus.status)) {
      return
    }

    const timeout = window.setTimeout(() => {
      void refreshInstallStatus().catch((error) => {
        logNonFatalRefreshError("polling", error)
      })
    }, POLL_INTERVAL_MS)

    return () => {
      window.clearTimeout(timeout)
    }
  }, [installStatus, refreshInstallStatus])

  const selectedBundle = React.useMemo(
    () =>
      recommendations.find((entry) => entry.bundle_id === selectedBundleId)?.bundle ||
      catalog.find((entry) => entry.bundle_id === selectedBundleId) ||
      null,
    [catalog, recommendations, selectedBundleId]
  )

  const selectedProfile = React.useMemo(() => {
    const recommendedProfile = recommendations.find(
      (entry) =>
        entry.bundle_id === selectedBundleId &&
        entry.resource_profile === selectedResourceProfile
    )?.profile
    return (
      recommendedProfile ||
      (selectedBundle?.resource_profiles || {})[selectedResourceProfile || ""] ||
      null
    )
  }, [recommendations, selectedBundle, selectedBundleId, selectedResourceProfile])

  const bundleOptions = React.useMemo(
    () => {
      if (recommendations.length > 0) {
        const seenBundleIds = new Set<string>()
        return recommendations.flatMap((entry) => {
          if (seenBundleIds.has(entry.bundle_id)) {
            return []
          }
          seenBundleIds.add(entry.bundle_id)
          return [
            {
              value: entry.bundle_id,
              label: entry.bundle?.label || entry.label || entry.bundle_id
            }
          ]
        })
      }
      return catalog.map((entry) => ({ value: entry.bundle_id, label: entry.label }))
    },
    [catalog, recommendations]
  )

  const profileOptions = React.useMemo(
    () =>
      Object.values(selectedBundle?.resource_profiles || {}).map((profile) => ({
        value: profile.profile_id,
        label: profile.label
      })),
    [selectedBundle]
  )

  const ttsChoiceOptions = React.useMemo(
    () =>
      Array.isArray(selectedProfile?.tts_choices)
        ? selectedProfile.tts_choices.map((choice) => ({
            value: choice.choice_id,
            label: choice.label
          }))
        : [],
    [selectedProfile]
  )

  const handleBundleChange = React.useCallback(
    (bundleId: string) => {
      const bundle =
        recommendations.find((entry) => entry.bundle_id === bundleId)?.bundle ||
        catalog.find((entry) => entry.bundle_id === bundleId) ||
        null
      const recommendedProfile = recommendations.find(
        (entry) => entry.bundle_id === bundleId
      )?.resource_profile
      setSelectedBundleId(bundleId)
      const nextProfileId =
        recommendedProfile ||
        bundle?.default_resource_profile ||
        Object.keys(bundle?.resource_profiles || {})[0] ||
        null
      const nextProfile = (bundle?.resource_profiles || {})[nextProfileId || ""]
      setSelectedResourceProfile(nextProfileId)
      setSelectedTtsChoice(resolveSelectedTtsChoice(nextProfile, null))
      setVerification(null)
    },
    [catalog, recommendations]
  )

  const handleResourceProfileChange = React.useCallback((profileId: string) => {
    const nextProfile = (selectedBundle?.resource_profiles || {})[profileId]
    setSelectedResourceProfile(profileId)
    setSelectedTtsChoice(resolveSelectedTtsChoice(nextProfile, null))
    setVerification(null)
  }, [selectedBundle])

  const handleTtsChoiceChange = React.useCallback((ttsChoice: string) => {
    setSelectedTtsChoice(ttsChoice)
    setVerification(null)
  }, [])

  const handleProvision = React.useCallback(
    async (safeRerun = false) => {
      if (!selectedBundleId || !selectedResourceProfile) return
      setProvisioning(true)
      try {
        const result = await requestJson<InstallStatusSnapshot>(PROVISION_PATH, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          timeoutMs: PROVISION_TIMEOUT_MS,
          body: JSON.stringify(
            buildBundleRequestBody(selectedBundleId, selectedResourceProfile, selectedTtsChoice, {
              safe_rerun: safeRerun
            })
          )
        })
        setInstallStatus(result)
        setVerification(null)
        setAdminGuard(null)
        setError(null)
        if (result?.status && ACTIVE_INSTALL_STATUSES.has(result.status)) {
          void refreshInstallStatus().catch((error) => {
            logNonFatalRefreshError("post-provision refresh", error)
          })
        }
      } catch (err) {
        setAdminGuard(deriveAdminGuardFromError(err))
        setError(
          sanitizeAdminErrorMessage(err, "Unable to provision the selected audio bundle.")
        )
      } finally {
        setProvisioning(false)
      }
    },
    [refreshInstallStatus, selectedBundleId, selectedResourceProfile, selectedTtsChoice]
  )

  const handleVerify = React.useCallback(async () => {
    if (!selectedBundleId || !selectedResourceProfile) return
    setVerifying(true)
    try {
      const result = await requestJson<VerificationResult>(VERIFY_PATH, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(
          buildBundleRequestBody(selectedBundleId, selectedResourceProfile, selectedTtsChoice)
        )
      })
      setVerification(result)
      setAdminGuard(null)
      setError(null)
    } catch (err) {
      setAdminGuard(deriveAdminGuardFromError(err))
      setError(
        sanitizeAdminErrorMessage(err, "Unable to verify the selected audio bundle.")
      )
    } finally {
      setVerifying(false)
    }
  }, [selectedBundleId, selectedResourceProfile, selectedTtsChoice])

  return {
    adminGuard,
    bundleOptions,
    catalog,
    error,
    installStatus,
    loading,
    machineProfile,
    profileOptions,
    provisioning,
    recommendations,
    selectedBundle,
    selectedBundleId,
    selectedProfile,
    selectedResourceProfile,
    selectedTtsChoice,
    setSelectedResourceProfile,
    handleBundleChange,
    handleResourceProfileChange,
    handleTtsChoiceChange,
    handleProvision,
    handleVerify,
    refresh: load,
    ttsChoiceOptions,
    verification,
    verifying
  }
}

export type UseAudioInstallerResult = ReturnType<typeof useAudioInstaller>
