import React from "react"
import { Space, Typography } from "antd"
import { useTranslation } from "react-i18next"
import { PageShell } from "@/components/Common/PageShell"
import { Alert as DesignSystemAlert } from "@/components/ui/primitives"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import type {
  LlamacppAssetsResponse,
  LlamacppConfigResponse,
  LlamacppHardwareSnapshotResponse,
  LlamacppInventoryResponse,
  LlamacppProfile,
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

  const [config, setConfig] = React.useState<LlamacppConfigResponse | null>(null)
  const [status, setStatus] = React.useState<LlamacppStatus | null>(null)
  const [inventory, setInventory] = React.useState<LlamacppInventoryResponse | null>(null)
  const [assets, setAssets] = React.useState<LlamacppAssetsResponse | null>(null)
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
  const [importingAssetFolder, setImportingAssetFolder] = React.useState(false)

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

  const loadAssets = React.useCallback(async () => {
    try {
      setLoadingAssets(true)
      setAssetError(null)
      const data = await tldwClient.getLlamacppAssets()
      setAssets(data)
    } catch (error: unknown) {
      setAssets(null)
      setAssetError(
        sanitizeAdminErrorMessage(error, "Failed to load Llama.cpp assets.")
      )
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
      loadHardware(),
      loadRuntimePlane()
    ])
  }, [
    loadAssets,
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

  const handleImportAssetFolder = async (path: string): Promise<boolean> => {
    try {
      setImportingAssetFolder(true)
      setAssetError(null)
      await tldwClient.importLlamacppAssetFolder(path)
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

            {!runtimeUnsupported && (
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
            )}

            <LlamacppAssetsPanel
              assets={assets}
              loading={loadingAssets}
              registeringPath={registeringAssetPath}
              importingFolder={importingAssetFolder}
              error={assetError}
              onRegisterPath={handleRegisterAssetPath}
              onImportFolder={handleImportAssetFolder}
              onReload={loadAssets}
            />

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
