import React from "react"
import { Space, Typography } from "antd"
import { useTranslation } from "react-i18next"
import { PageShell } from "@/components/Common/PageShell"
import { Alert as DesignSystemAlert } from "@/components/ui/primitives"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import type {
  LlamacppAcquisitionJobListResponse,
  LlamacppAssetDownloadRequest,
  LlamacppAssetImportPreviewResponse,
  LlamacppAssetsResponse,
  LlamacppConfigResponse,
  LlamacppHardwareSnapshotResponse,
  LlamacppInventoryResponse,
  LlamacppProfile,
  LlamacppProfileCreateRequest,
  LlamacppProfileUpdateRequest,
  LlamacppRuntime
} from "@/types/llamacpp-admin"
import {
  buildLlamacppServerArgs,
  type LlamacppServerArgsInput
} from "@/utils/build-llamacpp-server-args"
import { downloadBlob } from "@/utils/download-blob"
import { StatusBanner } from "./StatusBanner"
import {
  deriveAdminGuardFromError,
  sanitizeAdminErrorMessage,
  type AdminGuardState
} from "./admin-error-utils"
import { LlamacppAssetsPanel } from "./LlamacppAssetsPanel"
import { LlamacppInventoryPanel } from "./LlamacppInventoryPanel"
import { LlamacppLaunchPanel } from "./LlamacppLaunchPanel"
import { LlamacppProfilesPanel } from "./LlamacppProfilesPanel"
import { LlamacppReadinessPanel } from "./LlamacppReadinessPanel"
import { LlamacppRuntimePanel } from "./LlamacppRuntimePanel"

const { Title, Text } = Typography
const passiveAlertProps = {
  role: "status",
  "aria-live": "polite"
} as const

type LlamacppStatus = {
  backend?: string
  model?: string | null
  state?: string
  status?: string
  port?: number | null
  [key: string]: unknown
}

const DEFAULT_LLAMACPP_SETTINGS: LlamacppServerArgsInput = {
  contextSize: 4096,
  gpuLayers: 0,
  cacheType: "f16",
  splitMode: "layer",
  rowSplit: false,
  mlock: false,
  noMmap: false,
  noKvOffload: false,
  streamingLlm: false,
  cpuMoe: false,
  mmprojAuto: true,
  mmprojOffload: true,
  flashAttn: "auto",
  customArgs: {}
}

const LLAMACPP_PRESET_FORMAT_VERSION = 1
const LLAMACPP_PRESET_TYPE = "tldw_llamacpp_settings_preset"

interface LlamacppSettingsPresetV1 {
  type: typeof LLAMACPP_PRESET_TYPE
  version: typeof LLAMACPP_PRESET_FORMAT_VERSION
  createdAt: string
  settings: LlamacppServerArgsInput
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value)

const coerceImportedSettings = (input: unknown): LlamacppServerArgsInput | null => {
  if (!isRecord(input)) return null

  const maybePreset = input as Partial<LlamacppSettingsPresetV1>
  const source = isRecord(maybePreset.settings) ? maybePreset.settings : input
  if (!isRecord(source)) return null

  const merged = {
    ...DEFAULT_LLAMACPP_SETTINGS,
    ...source
  } as LlamacppServerArgsInput

  if (typeof merged.contextSize !== "number" || !Number.isFinite(merged.contextSize)) {
    return null
  }
  if (typeof merged.gpuLayers !== "number" || !Number.isFinite(merged.gpuLayers)) {
    return null
  }
  if (merged.splitMode && !["none", "layer", "row"].includes(merged.splitMode)) {
    merged.splitMode = "layer"
  }
  if (merged.flashAttn && !["auto", "on", "off"].includes(merged.flashAttn)) {
    merged.flashAttn = "auto"
  }
  if (
    merged.numa !== undefined &&
    merged.numa !== true &&
    merged.numa !== false &&
    !["distribute", "isolate", "numactl"].includes(String(merged.numa))
  ) {
    merged.numa = undefined
  }
  if (!isRecord(merged.customArgs)) {
    merged.customArgs = {}
  }

  return merged
}

export const LlamacppAdminPage: React.FC = () => {
  const { t } = useTranslation(["option", "settings", "common"])
  const initialLoadRef = React.useRef(false)
  const presetFileInputRef = React.useRef<HTMLInputElement | null>(null)
  const refreshedDownloadJobIdsRef = React.useRef<Set<string>>(new Set())
  const downloadsInitializedRef = React.useRef(false)

  const [config, setConfig] = React.useState<LlamacppConfigResponse | null>(null)
  const [status, setStatus] = React.useState<LlamacppStatus | null>(null)
  const [inventory, setInventory] = React.useState<LlamacppInventoryResponse | null>(null)
  const [assets, setAssets] = React.useState<LlamacppAssetsResponse | null>(null)
  const [assetImportPreview, setAssetImportPreview] =
    React.useState<LlamacppAssetImportPreviewResponse | null>(null)
  const [assetDownloads, setAssetDownloads] =
    React.useState<LlamacppAcquisitionJobListResponse | null>(null)
  const [hardware, setHardware] = React.useState<LlamacppHardwareSnapshotResponse | null>(null)
  const [runtimeProfiles, setRuntimeProfiles] = React.useState<LlamacppProfile[]>([])
  const [runtimeInstances, setRuntimeInstances] = React.useState<LlamacppRuntime[]>([])

  const [loadingConfig, setLoadingConfig] = React.useState(false)
  const [loadingStatus, setLoadingStatus] = React.useState(false)
  const [loadingInventory, setLoadingInventory] = React.useState(true)
  const [loadingAssets, setLoadingAssets] = React.useState(true)
  const [loadingRuntimes, setLoadingRuntimes] = React.useState(true)
  const [registeringPath, setRegisteringPath] = React.useState(false)
  const [registeringAssetPath, setRegisteringAssetPath] = React.useState(false)
  const [previewingAssetFolder, setPreviewingAssetFolder] = React.useState(false)
  const [importingAssetFolder, setImportingAssetFolder] = React.useState(false)
  const [loadingAssetDownloads, setLoadingAssetDownloads] = React.useState(false)
  const [startingAssetDownload, setStartingAssetDownload] = React.useState(false)
  const [cancelingAssetDownloadId, setCancelingAssetDownloadId] = React.useState<string | null>(null)

  const [statusError, setStatusError] = React.useState<string | null>(null)
  const [inventoryError, setInventoryError] = React.useState<string | null>(null)
  const [assetError, setAssetError] = React.useState<string | null>(null)
  const [runtimeError, setRuntimeError] = React.useState<string | null>(null)
  const [runtimeUnsupported, setRuntimeUnsupported] = React.useState(false)
  const [adminGuard, setAdminGuard] = React.useState<AdminGuardState>(null)

  const [selectedModelId, setSelectedModelId] = React.useState<string | undefined>()
  const [settings, setSettings] = React.useState<LlamacppServerArgsInput>(DEFAULT_LLAMACPP_SETTINGS)
  const [presetNotice, setPresetNotice] = React.useState<string | null>(null)
  const [actionLoading, setActionLoading] = React.useState(false)
  const [chatActionVisible, setChatActionVisible] = React.useState(false)
  const [chatActionLoading, setChatActionLoading] = React.useState(false)
  const [chatNotice, setChatNotice] = React.useState<string | null>(null)
  const [chatWarnings, setChatWarnings] = React.useState<string[]>([])
  const [profileActionId, setProfileActionId] = React.useState<string | null>(null)
  const [profileError, setProfileError] = React.useState<string | null>(null)
  const [runtimeActionProfileId, setRuntimeActionProfileId] = React.useState<string | null>(null)

  const markAdminGuardFromError = React.useCallback((error: unknown) => {
    const guardState = deriveAdminGuardFromError(error)
    if (guardState) {
      setAdminGuard(guardState)
    }
  }, [])

  const loadConfig = React.useCallback(async () => {
    try {
      setLoadingConfig(true)
      const data = await tldwClient.getLlamacppConfig()
      setConfig(data)
    } catch (error: unknown) {
      setStatusError(
        sanitizeAdminErrorMessage(error, "Failed to load Llama.cpp configuration.")
      )
      markAdminGuardFromError(error)
    } finally {
      setLoadingConfig(false)
    }
  }, [markAdminGuardFromError])

  const loadStatus = React.useCallback(async () => {
    try {
      setLoadingStatus(true)
      setStatusError(null)
      const data = await tldwClient.getLlamacppStatus()
      setStatus(data as LlamacppStatus)
    } catch (error: unknown) {
      setStatus(null)
      setStatusError(
        sanitizeAdminErrorMessage(error, "Failed to load Llama.cpp status.")
      )
      markAdminGuardFromError(error)
    } finally {
      setLoadingStatus(false)
    }
  }, [markAdminGuardFromError])

  const loadInventory = React.useCallback(async () => {
    try {
      setLoadingInventory(true)
      setInventoryError(null)
      const data = await tldwClient.getLlamacppInventory()
      setInventory(data)
      setSelectedModelId((current) => {
        if (current && data.models.some((item) => item.model_id === current)) {
          return current
        }
        return data.models[0]?.model_id
      })
    } catch (error: unknown) {
      setInventory(null)
      setSelectedModelId(undefined)
      setInventoryError(
        sanitizeAdminErrorMessage(error, "Failed to load Llama.cpp inventory.")
      )
      markAdminGuardFromError(error)
    } finally {
      setLoadingInventory(false)
    }
  }, [markAdminGuardFromError])

  const loadAssets = React.useCallback(async (): Promise<boolean> => {
    try {
      setLoadingAssets(true)
      setAssetError(null)
      const data = await tldwClient.getLlamacppAssets()
      setAssets(data)
      return true
    } catch (error: unknown) {
      setAssets(null)
      setAssetError(
        sanitizeAdminErrorMessage(error, "Failed to load Llama.cpp assets.")
      )
      return false
    } finally {
      setLoadingAssets(false)
    }
  }, [])

  const loadHardware = React.useCallback(async () => {
    try {
      const data = await tldwClient.getLlamacppHardware()
      setHardware(data)
    } catch (error: unknown) {
      setHardware({
        gpus: [],
        warnings: [
          sanitizeAdminErrorMessage(error, "Hardware snapshot is unavailable.")
        ]
      })
    }
  }, [])

  const loadAssetDownloads = React.useCallback(async () => {
    try {
      setLoadingAssetDownloads(true)
      setAssetError(null)
      const data = await tldwClient.listLlamacppAssetDownloads()
      setAssetDownloads(data)
      const completedJobs = (data.jobs || []).filter((job) =>
        ["completed", "succeeded", "done"].includes(job.status.toLowerCase())
      )
      const newlyCompleted = completedJobs.filter(
        (job) => !refreshedDownloadJobIdsRef.current.has(job.job_id)
      )
      if (newlyCompleted.length > 0) {
        if (downloadsInitializedRef.current) {
          const refreshed = await loadAssets()
          if (refreshed) {
            newlyCompleted.forEach((job) => {
              refreshedDownloadJobIdsRef.current.add(job.job_id)
            })
          }
        } else {
          newlyCompleted.forEach((job) => {
            refreshedDownloadJobIdsRef.current.add(job.job_id)
          })
        }
      }
      downloadsInitializedRef.current = true
    } catch (error: unknown) {
      downloadsInitializedRef.current = true
      setAssetError(
        sanitizeAdminErrorMessage(error, "Failed to load Llama.cpp asset downloads.")
      )
    } finally {
      setLoadingAssetDownloads(false)
    }
  }, [loadAssets])

  const loadRuntimePlane = React.useCallback(async () => {
    try {
      setLoadingRuntimes(true)
      setRuntimeError(null)
      const [profileData, instanceData] = await Promise.all([
        tldwClient.listLlamacppProfiles(),
        tldwClient.listLlamacppInstances()
      ])
      setRuntimeProfiles(profileData.profiles || [])
      setRuntimeInstances(instanceData.runtimes || [])
      setRuntimeUnsupported(false)
    } catch (error: unknown) {
      const guardState = deriveAdminGuardFromError(error)
      if (guardState === "forbidden") {
        setAdminGuard(guardState)
        return
      }
      setRuntimeProfiles([])
      setRuntimeInstances([])
      if (guardState === "notFound") {
        setRuntimeUnsupported(true)
        setRuntimeError(null)
        return
      }
      setRuntimeUnsupported(false)
      setRuntimeError(
        sanitizeAdminErrorMessage(error, "Failed to load Llama.cpp runtime instances.")
      )
    } finally {
      setLoadingRuntimes(false)
    }
  }, [])

  React.useEffect(() => {
    if (initialLoadRef.current) return
    initialLoadRef.current = true

    void Promise.all([
      loadConfig(),
      loadStatus(),
      loadInventory(),
      loadAssets(),
      loadAssetDownloads(),
      loadHardware(),
      loadRuntimePlane()
    ])
  }, [
    loadAssets,
    loadAssetDownloads,
    loadConfig,
    loadHardware,
    loadInventory,
    loadRuntimePlane,
    loadStatus
  ])

  const effectiveState =
    status?.state || status?.status || status?.backend || "unknown"
  const isRunning = effectiveState === "running" || effectiveState === "online"
  const selectedModel = React.useMemo(
    () => inventory?.models.find((item) => item.model_id === selectedModelId),
    [inventory?.models, selectedModelId]
  )
  const selectedModelLabel =
    selectedModel?.display_name || selectedModel?.basename || selectedModelId
  const hardwareWarnings = hardware?.warnings || []
  const inventoryUnavailable = Boolean(inventoryError) || (!loadingInventory && !inventory)
  const inventoryLoadedOrUnavailable = Boolean(inventory) || inventoryUnavailable

  React.useEffect(() => {
    if (isRunning) {
      setChatActionVisible(true)
      return
    }

    if (status && !loadingStatus && !actionLoading) {
      setChatActionVisible(false)
      setChatNotice(null)
      setChatWarnings([])
    }
  }, [actionLoading, isRunning, loadingStatus, status])

  const handleRegisterPath = async (path: string): Promise<boolean> => {
    try {
      setRegisteringPath(true)
      setInventoryError(null)
      await tldwClient.registerLlamacppModelPath(path)
      await loadInventory()
      return true
    } catch (error: unknown) {
      setInventoryError(
        sanitizeAdminErrorMessage(error, "Failed to register Llama.cpp model path.")
      )
      markAdminGuardFromError(error)
      return false
    } finally {
      setRegisteringPath(false)
    }
  }

  const handleRegisterAssetPath = async (path: string): Promise<boolean> => {
    try {
      setRegisteringAssetPath(true)
      setAssetError(null)
      const asset = await tldwClient.registerLlamacppAssetPath(path)
      await loadAssets()
      if (asset.kind === "gguf") {
        await loadInventory()
      }
      return true
    } catch (error: unknown) {
      setAssetError(
        sanitizeAdminErrorMessage(error, "Failed to register Llama.cpp asset path.")
      )
      return false
    } finally {
      setRegisteringAssetPath(false)
    }
  }

  const handlePreviewAssetFolder = async (path: string): Promise<boolean> => {
    try {
      setPreviewingAssetFolder(true)
      setAssetError(null)
      const preview = await tldwClient.previewLlamacppAssetFolder(path)
      setAssetImportPreview(preview)
      return true
    } catch (error: unknown) {
      setAssetImportPreview(null)
      setAssetError(
        sanitizeAdminErrorMessage(error, "Failed to preview Llama.cpp asset folder.")
      )
      return false
    } finally {
      setPreviewingAssetFolder(false)
    }
  }

  const handleImportAssetFolder = async (path: string): Promise<boolean> => {
    try {
      setImportingAssetFolder(true)
      setAssetError(null)
      await tldwClient.importLlamacppAssetFolder(path)
      setAssetImportPreview(null)
      await loadAssets()
      return true
    } catch (error: unknown) {
      setAssetError(
        sanitizeAdminErrorMessage(error, "Failed to import Llama.cpp asset folder.")
      )
      return false
    } finally {
      setImportingAssetFolder(false)
    }
  }

  const handleStartAssetDownload = async (
    payload: LlamacppAssetDownloadRequest
  ): Promise<boolean> => {
    try {
      setStartingAssetDownload(true)
      setAssetError(null)
      await tldwClient.startLlamacppAssetDownload(payload)
      await loadAssetDownloads()
      return true
    } catch (error: unknown) {
      setAssetError(
        sanitizeAdminErrorMessage(error, "Failed to queue Llama.cpp asset download.")
      )
      return false
    } finally {
      setStartingAssetDownload(false)
    }
  }

  const handleCancelAssetDownload = async (jobId: string): Promise<boolean> => {
    try {
      setCancelingAssetDownloadId(jobId)
      setAssetError(null)
      await tldwClient.cancelLlamacppAssetDownload(jobId)
      await loadAssetDownloads()
      return true
    } catch (error: unknown) {
      setAssetError(
        sanitizeAdminErrorMessage(error, "Failed to cancel Llama.cpp asset download.")
      )
      return false
    } finally {
      setCancelingAssetDownloadId(null)
    }
  }

  const handleCreateProfile = async (
    payload: LlamacppProfileCreateRequest
  ): Promise<boolean> => {
    try {
      setProfileActionId("__create__")
      setProfileError(null)
      await tldwClient.createLlamacppProfile(payload)
      await loadRuntimePlane()
      return true
    } catch (error: unknown) {
      setProfileError(
        sanitizeAdminErrorMessage(error, "Failed to create llama.cpp profile.")
      )
      markAdminGuardFromError(error)
      return false
    } finally {
      setProfileActionId(null)
    }
  }

  const handleUpdateProfile = async (
    profileId: string,
    payload: LlamacppProfileUpdateRequest
  ): Promise<boolean> => {
    try {
      setProfileActionId(profileId)
      setProfileError(null)
      await tldwClient.updateLlamacppProfile(profileId, payload)
      await loadRuntimePlane()
      return true
    } catch (error: unknown) {
      setProfileError(
        sanitizeAdminErrorMessage(error, "Failed to update llama.cpp profile.")
      )
      markAdminGuardFromError(error)
      return false
    } finally {
      setProfileActionId(null)
    }
  }

  const handleDeleteProfile = async (profileId: string): Promise<boolean> => {
    try {
      setProfileActionId(profileId)
      setProfileError(null)
      await tldwClient.deleteLlamacppProfile(profileId)
      await loadRuntimePlane()
      return true
    } catch (error: unknown) {
      setProfileError(
        sanitizeAdminErrorMessage(error, "Failed to delete llama.cpp profile.")
      )
      markAdminGuardFromError(error)
      return false
    } finally {
      setProfileActionId(null)
    }
  }

  const handleStart = async () => {
    if (!selectedModelId) return
    try {
      setActionLoading(true)
      setStatusError(null)
      const serverArgs = buildLlamacppServerArgs(settings)
      await tldwClient.startLlamacppModel(selectedModelId, serverArgs)
      setChatActionVisible(true)
      await Promise.all([loadStatus(), loadRuntimePlane()])
    } catch (error: unknown) {
      setStatusError(
        sanitizeAdminErrorMessage(error, "Failed to start Llama.cpp server.")
      )
      markAdminGuardFromError(error)
    } finally {
      setActionLoading(false)
    }
  }

  const handleStartWithDefaults = async () => {
    if (!selectedModelId) return
    try {
      setActionLoading(true)
      setStatusError(null)
      await tldwClient.startLlamacppModel(selectedModelId)
      setChatActionVisible(true)
      await Promise.all([loadStatus(), loadRuntimePlane()])
    } catch (error: unknown) {
      setStatusError(
        sanitizeAdminErrorMessage(error, "Failed to start Llama.cpp server.")
      )
      markAdminGuardFromError(error)
    } finally {
      setActionLoading(false)
    }
  }

  const handleStop = async () => {
    try {
      setActionLoading(true)
      await tldwClient.stopLlamacppServer()
      setChatActionVisible(false)
      await Promise.all([loadStatus(), loadRuntimePlane()])
    } catch (error: unknown) {
      setStatusError(
        sanitizeAdminErrorMessage(error, "Failed to stop Llama.cpp server.")
      )
      markAdminGuardFromError(error)
    } finally {
      setActionLoading(false)
    }
  }

  const handleUseInChat = async () => {
    try {
      setChatActionLoading(true)
      setChatNotice(null)
      setChatWarnings([])
      const response = await tldwClient.useLlamacppInChat()
      setChatNotice(
        response.effective
          ? "Chat provider updated."
          : "Chat provider setting saved, but an override may still be active."
      )
      setChatWarnings(response.warnings || [])
    } catch (error: unknown) {
      setChatNotice(null)
      setChatWarnings([
        sanitizeAdminErrorMessage(error, "Failed to wire llama.cpp into Chat.")
      ])
      markAdminGuardFromError(error)
    } finally {
      setChatActionLoading(false)
    }
  }

  const runRuntimeAction = async (
    profileId: string,
    action: () => Promise<unknown>,
    fallbackMessage: string
  ) => {
    try {
      setRuntimeActionProfileId(profileId)
      setRuntimeError(null)
      await action()
      await Promise.all([loadStatus(), loadRuntimePlane()])
    } catch (error: unknown) {
      setRuntimeError(sanitizeAdminErrorMessage(error, fallbackMessage))
      markAdminGuardFromError(error)
    } finally {
      setRuntimeActionProfileId(null)
    }
  }

  const handleStartProfile = (profileId: string) => {
    void runRuntimeAction(
      profileId,
      () => tldwClient.startLlamacppProfile(profileId),
      "Failed to start Llama.cpp runtime profile."
    )
  }

  const handleStopProfile = (profileId: string) => {
    void runRuntimeAction(
      profileId,
      () => tldwClient.stopLlamacppProfile(profileId),
      "Failed to stop Llama.cpp runtime profile."
    )
  }

  const handlePauseProfile = (profileId: string) => {
    void runRuntimeAction(
      profileId,
      () => tldwClient.pauseLlamacppProfile(profileId),
      "Failed to pause Llama.cpp runtime profile."
    )
  }

  const handleResumeProfile = (profileId: string) => {
    void runRuntimeAction(
      profileId,
      () => tldwClient.resumeLlamacppProfile(profileId),
      "Failed to resume Llama.cpp runtime profile."
    )
  }

  const handleUseProfileInChat = async (profileId: string) => {
    try {
      setRuntimeActionProfileId(profileId)
      setChatNotice(null)
      setChatWarnings([])
      const response = await tldwClient.useLlamacppProfileInChat(profileId)
      setChatNotice(
        response.effective
          ? "Chat provider updated."
          : "Chat provider setting saved, but an override may still be active."
      )
      setChatWarnings(response.warnings || [])
    } catch (error: unknown) {
      setChatNotice(null)
      setChatWarnings([
        sanitizeAdminErrorMessage(error, "Failed to wire llama.cpp into Chat.")
      ])
      markAdminGuardFromError(error)
    } finally {
      setRuntimeActionProfileId(null)
    }
  }

  const handleExportPreset = () => {
    const payload: LlamacppSettingsPresetV1 = {
      type: LLAMACPP_PRESET_TYPE,
      version: LLAMACPP_PRESET_FORMAT_VERSION,
      createdAt: new Date().toISOString(),
      settings
    }
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: "application/json"
    })
    const date = new Date().toISOString().slice(0, 10)
    downloadBlob(blob, `llamacpp-settings-preset-${date}.json`)
    setPresetNotice(
      t("settings:admin.llamacppPresetExported", "Exported Llama.cpp settings preset.")
    )
  }

  const handleOpenImportPreset = () => {
    presetFileInputRef.current?.click()
  }

  const handleImportPreset = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0]
    event.target.value = ""
    if (!file) return

    try {
      const text = await file.text()
      const parsed = JSON.parse(text)
      const importedSettings = coerceImportedSettings(parsed)
      if (!importedSettings) {
        throw new Error("Invalid Llama.cpp preset format.")
      }
      setSettings(importedSettings)
      setStatusError(null)
      setPresetNotice(
        t(
          "settings:admin.llamacppPresetImported",
          `Imported preset from ${file.name}.`
        )
      )
    } catch (error: unknown) {
      setPresetNotice(null)
      setStatusError(
        sanitizeAdminErrorMessage(
          error,
          t(
            "settings:admin.llamacppPresetImportFailed",
            "Failed to import preset."
          )
        )
      )
    }
  }

  const importPresetInput = (
    <input
      ref={presetFileInputRef}
      type="file"
      accept=".json,application/json"
      onChange={handleImportPreset}
      className="hidden"
      aria-label={t("settings:admin.llamacppImportPreset", "Import preset")}
    />
  )

  return (
    <PageShell>
      <Space orientation="vertical" size="large" className="w-full py-6">
        {adminGuard && (
          <DesignSystemAlert
            variant="warning"
            {...passiveAlertProps}
            title={
              adminGuard === "forbidden"
                ? t("settings:admin.adminGuardForbiddenTitle", "Admin access required")
                : t("settings:admin.adminGuardNotFoundTitle", "Admin APIs not available")
            }
          >
            <span>
              {adminGuard === "forbidden"
                ? t(
                    "settings:admin.adminGuardForbiddenBody",
                    "Sign in as an admin user on your tldw server to access these controls."
                  )
                : t(
                    "settings:admin.adminGuardNotFoundBody",
                    "This tldw server does not expose the admin endpoints."
                  )}{" "}
              <a
                href="https://github.com/rmusser01/tldw_server#documentation--resources"
                target="_blank"
                rel="noreferrer"
              >
                {t("settings:admin.adminGuardLearnMore", "Learn more")}
              </a>
            </span>
          </DesignSystemAlert>
        )}

        <div>
          <Title level={2}>
            {t("option:header.adminLlamacpp", "Llama.cpp Admin")}
          </Title>
          <Text type="secondary">
            {t(
              "settings:admin.llamacppIntro",
              "Configure, launch, inspect, and explicitly wire the managed llama.cpp server into chat."
            )}
          </Text>
        </div>

        {!adminGuard && (
          <>
            <StatusBanner
              state={effectiveState}
              loading={loadingStatus}
              error={statusError}
              items={[
                { label: t("settings:admin.llamacppActiveModel", "Model"), value: status?.model, code: true },
                { label: t("settings:admin.llamacppPort", "Port"), value: status?.port }
              ]}
              onRefresh={loadStatus}
              quickAction={
                isRunning
                  ? {
                      label: t("settings:admin.llamacppStop", "Stop"),
                      onClick: handleStop,
                      loading: actionLoading,
                      danger: true
                    }
                  : undefined
              }
            />

            <LlamacppReadinessPanel
              config={config}
              loading={loadingConfig}
            />

            <LlamacppAssetsPanel
              assets={assets}
              loading={loadingAssets}
              registeringPath={registeringAssetPath}
              previewingFolder={previewingAssetFolder}
              importingFolder={importingAssetFolder}
              importPreview={assetImportPreview}
              downloads={assetDownloads}
              loadingDownloads={loadingAssetDownloads}
              startingDownload={startingAssetDownload}
              cancelingDownloadId={cancelingAssetDownloadId}
              error={assetError}
              onRegisterPath={handleRegisterAssetPath}
              onPreviewImportFolder={handlePreviewAssetFolder}
              onClearImportPreview={() => setAssetImportPreview(null)}
              onImportFolder={handleImportAssetFolder}
              onStartDownload={handleStartAssetDownload}
              onCancelDownload={handleCancelAssetDownload}
              onReloadDownloads={loadAssetDownloads}
              onReload={loadAssets}
            />

            {!runtimeUnsupported && (
              <>
                <LlamacppProfilesPanel
                  profiles={runtimeProfiles}
                  assets={assets}
                  loading={loadingRuntimes}
                  savingProfileId={profileActionId}
                  error={profileError}
                  onRefresh={loadRuntimePlane}
                  onCreate={handleCreateProfile}
                  onUpdate={handleUpdateProfile}
                  onDelete={handleDeleteProfile}
                />

                <LlamacppRuntimePanel
                  profiles={runtimeProfiles}
                  runtimes={runtimeInstances}
                  loading={loadingRuntimes}
                  error={runtimeError}
                  actionProfileId={runtimeActionProfileId}
                  onRefresh={loadRuntimePlane}
                  onStart={handleStartProfile}
                  onStop={handleStopProfile}
                  onPause={handlePauseProfile}
                  onResume={handleResumeProfile}
                  onUseInChat={(profileId) => {
                    void handleUseProfileInChat(profileId)
                  }}
                />
              </>
            )}

            <LlamacppInventoryPanel
              inventory={inventory}
              selectedModelId={selectedModelId}
              activeModel={status?.model || config?.active_config.active_model}
              loading={loadingInventory}
              registering={registeringPath}
              error={inventoryError}
              onSelectModel={setSelectedModelId}
              onRegisterPath={handleRegisterPath}
              onReload={loadInventory}
            />

            {inventoryLoadedOrUnavailable && (
              <LlamacppLaunchPanel
                settings={settings}
                onSettingsChange={setSettings}
                selectedModelId={selectedModelId}
                selectedModelLabel={selectedModelLabel}
                isRunning={isRunning}
                actionLoading={actionLoading}
                inventoryUnavailable={inventoryUnavailable}
                adminUnavailable={Boolean(adminGuard)}
                hardwareWarnings={hardwareWarnings}
                presetNotice={presetNotice}
                onStart={handleStart}
                onStartWithDefaults={handleStartWithDefaults}
                onExportPreset={handleExportPreset}
                onOpenImportPreset={handleOpenImportPreset}
                importPresetInput={importPresetInput}
                chatAction={{
                  visible: chatActionVisible || isRunning,
                  loading: chatActionLoading,
                  notice: chatNotice,
                  warnings: chatWarnings,
                  onUse: handleUseInChat
                }}
              />
            )}
          </>
        )}
      </Space>
    </PageShell>
  )
}

export default LlamacppAdminPage
