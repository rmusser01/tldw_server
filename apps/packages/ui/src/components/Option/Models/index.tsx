import { Select, Spin } from "antd"
import { useCallback, useEffect, useMemo, useState } from "react"
import { useTranslation } from "react-i18next"
import { useQuery, useQueryClient } from "@tanstack/react-query"
import { Link } from "react-router-dom"
import { browser } from "wxt/browser"
import { AvailableModelsList } from "./AvailableModelsList"
import {
  buildConfiguredFirstModelOptions,
  formatModelsLastRefreshedTime,
  summarizeProviderReadiness,
  type ModelDisplayEntry
} from "./modelsDisplayUtils"
import { useAntdNotification } from "@/hooks/useAntdNotification"
import { tldwClient, tldwModels } from "@/services/tldw"
import { useStorage } from "@plasmohq/storage/hook"
import { fetchChatModels } from "@/services/tldw-server"
import {
  getProviderDisplayName,
  normalizeProviderKey
} from "@/utils/provider-registry"
import { isAutoModelId } from "@/utils/resolve-api-provider"
import { isConfiguredUsableModel } from "@/hooks/playground/modelSelectorUtils"

interface RefreshResponse {
  ok: boolean
}

const isRefreshResponse = (res: unknown): res is RefreshResponse =>
  typeof res === "object" &&
  res !== null &&
  typeof (res as { ok?: unknown }).ok === "boolean"

type OpenAIOAuthAction =
  | "connect"
  | "refresh"
  | "disconnect"
  | "switch_api_key"
  | "reload_status"

const getStatusCode = (error: unknown): number | null => {
  if (!error || typeof error !== "object") return null
  const maybeStatus = (error as { status?: unknown }).status
  if (typeof maybeStatus !== "number" || !Number.isFinite(maybeStatus)) {
    return null
  }
  return maybeStatus
}

const getRecord = (value: unknown): Record<string, unknown> | null =>
  typeof value === "object" && value !== null
    ? (value as Record<string, unknown>)
    : null

const getBooleanFromModelPaths = (
  model: unknown,
  keys: string[]
): boolean | null => {
  const root = getRecord(model)
  if (!root) return null
  const details = getRecord(root.details)
  const metadata = getRecord(root.metadata)

  for (const key of keys) {
    const value = root[key] ?? details?.[key] ?? metadata?.[key]
    if (typeof value === "boolean") return value
  }

  return null
}

const getModelConfiguredSignal = (model: unknown): boolean | null => {
  const configuredKeys = [
    "is_configured",
    "isConfigured",
    "configured",
    "provider_is_configured",
    "providerIsConfigured",
    "provider_configured",
    "providerConfigured"
  ]
  let sawConfigured = false

  for (const key of configuredKeys) {
    const value = getBooleanFromModelPaths(model, [key])
    if (value === false) return false
    if (value === true) sawConfigured = true
  }

  return sawConfigured ? true : null
}

export const ModelsBody = () => {
  // Custom provider models have been removed; we only show
  // tldw_server models discovered from the server.
  const [refreshing, setRefreshing] = useState(false)
  const [lastRefreshedAt, setLastRefreshedAt] = useState<number | null>(null)
  const [openaiOauthAction, setOpenaiOauthAction] =
    useState<OpenAIOAuthAction | null>(null)

  const { t } = useTranslation(["settings", "common"])
  const notification = useAntdNotification()
  const queryClient = useQueryClient()
  const [selectedModel, setSelectedModel] = useStorage<string | null>(
    "selectedModel",
    null
  )
  const [defaultApiProvider, setDefaultApiProvider] = useStorage<
    string | null
  >("defaultApiProvider", null)

  const {
    data: availableModels = [],
    isLoading: modelsLoading
  } = useQuery({
    queryKey: ["tldw-chat-models"],
    queryFn: async () => fetchChatModels({ returnEmpty: true }),
    staleTime: 5 * 60 * 1000
  })

  const {
    data: openaiOauthStatus,
    isLoading: openaiOauthStatusLoading,
    error: openaiOauthStatusError,
    refetch: refetchOpenaiOauthStatus
  } = useQuery({
    queryKey: ["openai-oauth-status"],
    queryFn: async () => tldwClient.getOpenAIOAuthStatus(),
    staleTime: 30 * 1000,
    retry: false
  })

  const {
    data: providerKeysStatus,
    isLoading: providerKeysLoading,
    error: providerKeysError
  } = useQuery({
    queryKey: ["tldw-provider-keys-summary"],
    queryFn: async () => tldwClient.listUserProviderKeys(),
    staleTime: 30 * 1000,
    retry: false
  })

  const openaiOauthStatusCode = useMemo(
    () => getStatusCode(openaiOauthStatusError),
    [openaiOauthStatusError]
  )
  const providerKeysStatusCode = useMemo(
    () => getStatusCode(providerKeysError),
    [providerKeysError]
  )

  const openaiOauthUnavailable = useMemo(
    () =>
      openaiOauthStatusCode === 403 ||
      openaiOauthStatusCode === 404 ||
      openaiOauthStatusCode === 501,
    [openaiOauthStatusCode]
  )

  const openaiAuthSource = openaiOauthStatus?.auth_source ?? "none"
  const openaiOauthConnected = Boolean(openaiOauthStatus?.connected)
  const openaiOauthActive = openaiAuthSource === "oauth"
  const providerKeysUnavailable =
    providerKeysStatusCode === 403 ||
    providerKeysStatusCode === 404 ||
    providerKeysStatusCode === 501
  const providerKeysLoadFailed =
    Boolean(providerKeysError) && !providerKeysUnavailable

  const configuredProviderKeys = useMemo(() => {
    const keys = new Set<string>()
    for (const item of providerKeysStatus?.items ?? []) {
      if (!item.has_key) continue
      const normalized = normalizeProviderKey(item.provider)
      if (normalized && normalized !== "unknown") {
        keys.add(normalized)
      }
    }
    if (
      openaiOauthConnected ||
      openaiAuthSource === "oauth" ||
      openaiAuthSource === "api_key"
    ) {
      keys.add("openai")
    }
    return keys
  }, [openaiAuthSource, openaiOauthConnected, providerKeysStatus])

  const openaiOauthChip = useMemo(() => {
    if (openaiOauthActive && openaiOauthConnected) {
      return {
        label: t(
          "settings:models.openaiOauth.connectedOauth",
          "Connected (OAuth)"
        ),
        className:
          "border-emerald-300/70 bg-emerald-50 text-emerald-700 dark:border-emerald-700/50 dark:bg-emerald-950/30 dark:text-emerald-300"
      }
    }
    if (openaiAuthSource === "api_key") {
      return {
        label: t("settings:models.openaiOauth.apiKey", "API Key"),
        className:
          "border-sky-300/70 bg-sky-50 text-sky-700 dark:border-sky-700/50 dark:bg-sky-950/30 dark:text-sky-300"
      }
    }
    return {
      label: t("settings:models.openaiOauth.notConnected", "Not connected"),
      className:
        "border-border/80 bg-surface2 text-text-subtle"
    }
  }, [openaiAuthSource, openaiOauthActive, openaiOauthConnected, t])

  const normalizedDefaultProvider = useMemo(() => {
    if (!defaultApiProvider) return ""
    const normalized = normalizeProviderKey(defaultApiProvider)
    return normalized === "unknown" ? "" : normalized
  }, [defaultApiProvider])

  const providerSelectValue = useMemo(
    () => normalizedDefaultProvider || "auto",
    [normalizedDefaultProvider]
  )

  const providerOptions = useMemo(() => {
    const providers = new Map<string, string>()
    for (const model of availableModels) {
      const rawProvider = model.details?.provider ?? model.provider
      if (!rawProvider) continue
      const key = normalizeProviderKey(rawProvider)
      if (!key || key === "unknown") continue
      if (!providers.has(key)) {
        providers.set(key, getProviderDisplayName(rawProvider))
      }
    }
    return Array.from(providers.entries())
      .map(([value, label]) => ({ value, label }))
      .sort((a, b) => a.label.localeCompare(b.label))
  }, [availableModels])

  const toModelDisplayEntry = useCallback(
    (model: (typeof availableModels)[number]): ModelDisplayEntry | null => {
      const rawProvider = model.details?.provider ?? model.provider
      if (!rawProvider) return null
      const normalizedProvider = normalizeProviderKey(rawProvider)
      if (!normalizedProvider || normalizedProvider === "unknown") return null
      const configuredSignal = getModelConfiguredSignal(model)
      const configured =
        configuredSignal === false
          ? false
          : configuredProviderKeys.has(normalizedProvider) ||
            configuredSignal === true
      return {
        id: model.model,
        provider: getProviderDisplayName(rawProvider),
        nickname: model.nickname,
        configured,
        usable: configured && isConfiguredUsableModel(model),
        selected: selectedModel === model.model
      }
    },
    [configuredProviderKeys, selectedModel]
  )

  const modelDisplayEntries = useMemo(
    () =>
      availableModels
        .map(toModelDisplayEntry)
        .filter((model): model is ModelDisplayEntry => model !== null),
    [availableModels, toModelDisplayEntry]
  )

  const providerReadiness = useMemo(
    () => summarizeProviderReadiness(modelDisplayEntries),
    [modelDisplayEntries]
  )

  const defaultProviderLabel = normalizedDefaultProvider
    ? getProviderDisplayName(normalizedDefaultProvider)
    : t("settings:onboarding.defaults.providerAuto", "Auto (from model)")
  const defaultModelLabel =
    selectedModel && !isAutoModelId(selectedModel)
      ? selectedModel
      : t("settings:onboarding.defaults.modelAuto", "Auto (route on server)")

  const modelOptions = useMemo(() => {
    const modelEntries = availableModels
      .filter((model) => {
        if (!normalizedDefaultProvider) return true
        const rawProvider = model.details?.provider ?? model.provider
        if (!rawProvider) return false
        return normalizeProviderKey(rawProvider) === normalizedDefaultProvider
      })
      .map(toModelDisplayEntry)
      .filter((model): model is ModelDisplayEntry => model !== null)
    return buildConfiguredFirstModelOptions(modelEntries, {
      autoLabel: t(
        "settings:onboarding.defaults.modelAuto",
        "Auto (route on server)"
      )
    })
  }, [availableModels, normalizedDefaultProvider, t, toModelDisplayEntry])

  const handleProviderChange = useCallback(
    (value: string) => {
      if (value === "auto") {
        setDefaultApiProvider(null)
        return
      }
      const normalized = normalizeProviderKey(value)
      setDefaultApiProvider(
        normalized && normalized !== "unknown" ? normalized : null
      )
    },
    [setDefaultApiProvider]
  )

  const handleModelChange = useCallback(
    (value: string) => {
      setSelectedModel(value || null)
    },
    [setSelectedModel]
  )

  useEffect(() => {
    if (!normalizedDefaultProvider || !selectedModel) return
    if (isAutoModelId(selectedModel)) return
    if (availableModels.length === 0) return
    const selectedEntry = availableModels.find(
      (model) => model.model === selectedModel
    )
    if (!selectedEntry) return
    const rawProvider = selectedEntry.details?.provider ?? selectedEntry.provider
    if (!rawProvider) return
    if (normalizeProviderKey(rawProvider) !== normalizedDefaultProvider) {
      setSelectedModel(null)
    }
  }, [
    availableModels,
    normalizedDefaultProvider,
    selectedModel,
    setSelectedModel
  ])

  const handleRefresh = async () => {
    setRefreshing(true)
    try {
      const res = await browser.runtime
        .sendMessage({ type: "tldw:models:refresh" })
        .catch(() => null)
      if (!isRefreshResponse(res) || !res.ok) {
        // Fallback to local warm-up if background message failed
        await tldwModels.warmCache(true)
      }
      await Promise.all([
        queryClient.refetchQueries({ queryKey: ["tldw-providers-models"] }),
        queryClient.refetchQueries({ queryKey: ["tldw-models"] }),
        queryClient.refetchQueries({ queryKey: ["tldw-chat-models"] })
      ])
      const providers = queryClient.getQueryData<Record<string, unknown[]>>([
        "tldw-providers-models"
      ])
      setLastRefreshedAt(Date.now())
      if (!providers || Object.keys(providers).length === 0) {
        notification.error({
          message: t("settings:models.refreshEmpty", {
            defaultValue: "No providers available after refresh"
          }),
          description: t("settings:models.refreshEmptyHint", {
            defaultValue:
              "Check your server URL and API key, ensure your tldw_server is running, then try refreshing again."
          })
        })
      } else {
        notification.success({
          message: t("settings:models.refreshSuccess", {
            defaultValue: "Model list refreshed"
          })
        })
      }
    } catch (e: unknown) {
      console.error("[tldw] Failed to refresh models", e)
      const rawMessage = e instanceof Error ? e.message : String(e)
      const message =
        rawMessage.length > 200 ? `${rawMessage.slice(0, 197)}...` : rawMessage
      notification.error({
        message: t("settings:models.refreshFailed", { defaultValue: "Failed to refresh models" }),
        description: message
      })
    } finally {
      setRefreshing(false)
    }
  }

  const _formatOauthError = (error: unknown) => {
    const message = error instanceof Error ? error.message : String(error)
    return message.length > 220 ? `${message.slice(0, 217)}...` : message
  }

  const handleOpenAIOauthConnect = useCallback(async () => {
    setOpenaiOauthAction("connect")
    try {
      const authorize = await tldwClient.startOpenAIOAuthAuthorize({
        return_path: "/settings/model"
      })
      if (!authorize?.auth_url) {
        throw new Error("OpenAI OAuth authorize URL was not returned.")
      }
      if (typeof window !== "undefined") {
        const popup = window.open(
          authorize.auth_url,
          "_blank",
          "noopener,noreferrer"
        )
        if (!popup) {
          window.location.assign(authorize.auth_url)
        }
      }
      notification.info({
        message: t(
          "settings:models.openaiOauth.connectStarted",
          "OpenAI OAuth started"
        ),
        description: t(
          "settings:models.openaiOauth.connectHint",
          "Finish sign-in in the opened tab, then come back here and check status."
        )
      })
    } catch (error) {
      notification.error({
        message: t(
          "settings:models.openaiOauth.connectFailed",
          "Failed to start OpenAI OAuth"
        ),
        description: _formatOauthError(error)
      })
    } finally {
      setOpenaiOauthAction(null)
    }
  }, [notification, t])

  const handleOpenAIOauthRefreshStatus = useCallback(async () => {
    setOpenaiOauthAction("reload_status")
    try {
      await refetchOpenaiOauthStatus()
    } finally {
      setOpenaiOauthAction(null)
    }
  }, [refetchOpenaiOauthStatus])

  const handleOpenAIOauthRefreshToken = useCallback(async () => {
    setOpenaiOauthAction("refresh")
    try {
      await tldwClient.refreshOpenAIOAuth()
      await refetchOpenaiOauthStatus()
      notification.success({
        message: t(
          "settings:models.openaiOauth.refreshSuccess",
          "OpenAI OAuth token refreshed"
        )
      })
    } catch (error) {
      notification.error({
        message: t(
          "settings:models.openaiOauth.refreshFailed",
          "Failed to refresh OpenAI OAuth token"
        ),
        description: _formatOauthError(error)
      })
    } finally {
      setOpenaiOauthAction(null)
    }
  }, [notification, refetchOpenaiOauthStatus, t])

  const handleOpenAIOauthDisconnect = useCallback(async () => {
    setOpenaiOauthAction("disconnect")
    try {
      await tldwClient.disconnectOpenAIOAuth()
      await refetchOpenaiOauthStatus()
      notification.success({
        message: t(
          "settings:models.openaiOauth.disconnectSuccess",
          "OpenAI OAuth disconnected"
        )
      })
    } catch (error) {
      notification.error({
        message: t(
          "settings:models.openaiOauth.disconnectFailed",
          "Failed to disconnect OpenAI OAuth"
        ),
        description: _formatOauthError(error)
      })
    } finally {
      setOpenaiOauthAction(null)
    }
  }, [notification, refetchOpenaiOauthStatus, t])

  const handleOpenAIOauthUseApiKey = useCallback(async () => {
    setOpenaiOauthAction("switch_api_key")
    try {
      await tldwClient.switchOpenAICredentialSource("api_key")
      await refetchOpenaiOauthStatus()
      notification.success({
        message: t(
          "settings:models.openaiOauth.switchApiKeySuccess",
          "Switched to API key credential"
        )
      })
    } catch (error) {
      notification.error({
        message: t(
          "settings:models.openaiOauth.switchApiKeyFailed",
          "Failed to switch to API key credential"
        ),
        description: _formatOauthError(error)
      })
    } finally {
      setOpenaiOauthAction(null)
    }
  }, [notification, refetchOpenaiOauthStatus, t])

  return (
    <div>
      <div>
        <div className="mb-6">
          <div className="-ml-4 -mt-2 flex flex-wrap items-center justify-between sm:flex-nowrap">
            <div className="ml-4 mt-2 flex flex-wrap items-center gap-3">
              <button
                onClick={() => void handleRefresh()}
                disabled={refreshing}
                className="inline-flex items-center rounded-md border border-border bg-surface px-3 py-2 text-sm font-medium text-text shadow-sm hover:bg-surface2 focus:outline-none focus:ring-2 focus:ring-focus focus:ring-offset-2 disabled:opacity-60">
                {refreshing ? (
                  <>
                    <Spin size="small" className="mr-2" />
                    {t("common:loading.title", {
                      defaultValue: "Loading..."
                    })}
                  </>
                ) : (
                  t("common:refresh", { defaultValue: "Refresh" })
                )}
              </button>
              {lastRefreshedAt && (
                <span className="text-xs text-text-subtle">
                  {t("settings:models.lastRefreshedAt", {
                    defaultValue: "Last checked at {{time}}",
                    time: formatModelsLastRefreshedTime(lastRefreshedAt)
                  })}
                </span>
              )}
            </div>
          </div>
          <div className="mt-4 rounded-2xl border border-border/70 bg-surface p-4">
            <div className="mb-3">
              <div className="text-sm font-semibold text-text">
                {t("settings:onboarding.defaults.title", "Set your defaults")}
              </div>
              <p className="text-xs text-text-subtle">
                {t(
                  "settings:onboarding.defaults.subtitle",
                  "Pick a default provider and model for new chats."
                )}
              </p>
            </div>
            {modelsLoading ? (
              <div className="flex items-center gap-2 text-xs text-text-subtle">
                <Spin size="small" />
                {t("settings:onboarding.defaults.loading", "Loading models...")}
              </div>
            ) : availableModels.length === 0 ? (
              <div className="text-xs text-text-subtle">
                <div className="font-medium text-text">
                  {t(
                    "settings:models.noProvidersTitle",
                    "No providers available."
                  )}
                </div>
                <div className="mt-1">
                  {t(
                    "settings:models.noProvidersBody",
                    "The extension could not load providers from your tldw_server. Check your server URL and API key in Settings, ensure the server is running, then use Retry (or Refresh) to try again."
                  )}
                </div>
              </div>
            ) : (
              <div className="grid gap-3 sm:grid-cols-2">
                <div>
                  <label className="mb-1 block text-xs font-medium text-text">
                    {t(
                      "settings:onboarding.defaults.providerLabel",
                      "Default API provider"
                    )}
                  </label>
                  <Select
                    size="large"
                    value={providerSelectValue}
                    onChange={handleProviderChange}
                    options={[
                      {
                        value: "auto",
                        label: t(
                          "settings:onboarding.defaults.providerAuto",
                          "Auto (from model)"
                        )
                      },
                      ...providerOptions
                    ]}
                    className="w-full"
                  />
                  <p className="mt-1 text-[11px] text-text-subtle">
                    {t(
                      "settings:onboarding.defaults.providerHelp",
                      "Leave on Auto to use the provider attached to each model."
                    )}
                  </p>
                </div>
                <div>
                  <label className="mb-1 block text-xs font-medium text-text">
                    {t(
                      "settings:onboarding.defaults.modelLabel",
                      "Default model"
                    )}
                  </label>
                  <Select
                    showSearch
                    size="large"
                    value={selectedModel || undefined}
                    onChange={handleModelChange}
                    placeholder={t(
                      "settings:onboarding.defaults.modelPlaceholder",
                      "Select a model"
                    )}
                    options={modelOptions}
                    optionFilterProp="label"
                    className="w-full"
                    allowClear
                  />
                  <p className="mt-1 text-[11px] text-text-subtle">
                    {t(
                      "settings:onboarding.defaults.modelHelp",
                      "Choose Auto to let the server route each request, or pick a concrete model as the default."
                    )}
                  </p>
                </div>
              </div>
            )}
          </div>
          <div className="mt-4 rounded-2xl border border-border/70 bg-surface p-4">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <div className="text-sm font-semibold text-text">
                  {t("settings:models.readiness.title", "Provider readiness")}
                </div>
                <p className="mt-1 text-xs text-text-subtle">
                  {t(
                    "settings:models.readiness.subtitle",
                    "Review usable providers and saved defaults before browsing the full catalog."
                  )}
                </p>
              </div>
              <span className="inline-flex items-center rounded-full border border-border/80 bg-surface2 px-2.5 py-1 text-[11px] font-medium text-text-subtle">
                {modelsLoading
                  ? t("common:loading.title", "Loading...")
                  : t("settings:models.readiness.usableSummary", {
                      defaultValue: "{{count}} usable",
                      count: providerReadiness.usableProviders
                    })}
              </span>
            </div>
            <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <div className="rounded-md border border-border/70 bg-surface2 p-3">
                <div className="text-[11px] font-medium uppercase tracking-wide text-text-subtle">
                  {t(
                    "settings:models.readiness.defaultProvider",
                    "Default provider"
                  )}
                </div>
                <div className="mt-1 text-sm font-medium text-text">
                  {defaultProviderLabel}
                </div>
              </div>
              <div className="rounded-md border border-border/70 bg-surface2 p-3">
                <div className="text-[11px] font-medium uppercase tracking-wide text-text-subtle">
                  {t("settings:models.readiness.defaultModel", "Default model")}
                </div>
                <div className="mt-1 text-sm font-medium text-text">
                  {defaultModelLabel}
                </div>
              </div>
              <div className="rounded-md border border-border/70 bg-surface2 p-3">
                <div className="text-[11px] font-medium uppercase tracking-wide text-text-subtle">
                  {t(
                    "settings:models.readiness.configuredProviders",
                    "Configured providers"
                  )}
                </div>
                <div className="mt-1 text-sm font-medium text-text">
                  {providerKeysLoading
                    ? t("common:loading.title", "Loading...")
                    : providerKeysUnavailable
                      ? t(
                          "settings:models.readiness.keysUnavailable",
                          "Account keys unavailable"
                        )
                      : providerKeysLoadFailed
                        ? t(
                            "settings:models.readiness.keysLoadFailed",
                            "Unable to load account keys"
                          )
                      : t("settings:models.readiness.configuredCount", {
                          defaultValue: "{{count}} configured",
                          count: providerReadiness.configuredProviders
                        })}
                </div>
              </div>
              <div className="rounded-md border border-border/70 bg-surface2 p-3">
                <div className="text-[11px] font-medium uppercase tracking-wide text-text-subtle">
                  {t("settings:models.readiness.usableProviders", "Usable providers")}
                </div>
                <div className="mt-1 text-sm font-medium text-text">
                  {t("settings:models.readiness.usableCount", {
                    defaultValue: "{{count}} usable",
                    count: providerReadiness.usableProviders
                  })}
                </div>
              </div>
            </div>
            {availableModels.length === 0 && !modelsLoading && (
              <div className="mt-4 flex flex-wrap gap-2 text-xs">
                <Link className="text-primary hover:underline" to="/settings/tldw">
                  {t("settings:models.readiness.configureServer", "Configure server")}
                </Link>
                <Link
                  className="text-primary hover:underline"
                  to="/settings/provider-keys">
                  {t("settings:models.readiness.configureKeys", "Manage provider keys")}
                </Link>
              </div>
            )}
          </div>
          <div className="mt-4 rounded-2xl border border-border/70 bg-surface p-4">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <div className="text-sm font-semibold text-text">
                  {t(
                    "settings:models.openaiOauth.title",
                    "OpenAI Account Linking"
                  )}
                </div>
                <p className="mt-1 text-xs text-text-subtle">
                  {t(
                    "settings:models.openaiOauth.subtitle",
                    "Connect OpenAI with OAuth, refresh tokens, or fall back to your API key."
                  )}
                </p>
              </div>
              <span
                className={`inline-flex items-center rounded-full border px-2.5 py-1 text-[11px] font-medium ${openaiOauthChip.className}`}>
                {openaiOauthChip.label}
              </span>
            </div>

            {openaiOauthStatusLoading ? (
              <div className="mt-4 flex items-center gap-2 text-xs text-text-subtle">
                <Spin size="small" />
                {t(
                  "settings:models.openaiOauth.loading",
                  "Loading OpenAI OAuth status..."
                )}
              </div>
            ) : openaiOauthUnavailable ? (
              <div className="mt-4 text-xs text-text-subtle">
                {t(
                  "settings:models.openaiOauth.unavailable",
                  "OpenAI OAuth is unavailable on this server. Configure and enable it to use account linking."
                )}
              </div>
            ) : (
              <>
                <div className="mt-4 flex flex-wrap gap-2">
                  <button
                    onClick={() => void handleOpenAIOauthConnect()}
                    disabled={openaiOauthAction !== null}
                    className="inline-flex items-center rounded-md border border-border bg-surface2 px-3 py-2 text-xs font-medium text-text hover:bg-surface3 disabled:opacity-60">
                    {openaiOauthAction === "connect" ? (
                      <>
                        <Spin size="small" className="mr-2" />
                        {t("common:loading.title", {
                          defaultValue: "Loading..."
                        })}
                      </>
                    ) : openaiOauthConnected ? (
                      t("settings:models.openaiOauth.reconnect", "Reconnect OpenAI")
                    ) : (
                      t("settings:models.openaiOauth.connect", "Connect OpenAI")
                    )}
                  </button>
                  <button
                    onClick={() => void handleOpenAIOauthRefreshStatus()}
                    disabled={openaiOauthAction !== null}
                    className="inline-flex items-center rounded-md border border-border bg-surface2 px-3 py-2 text-xs font-medium text-text hover:bg-surface3 disabled:opacity-60">
                    {openaiOauthAction === "reload_status" ? (
                      <>
                        <Spin size="small" className="mr-2" />
                        {t("common:loading.title", {
                          defaultValue: "Loading..."
                        })}
                      </>
                    ) : (
                      t("settings:models.openaiOauth.checkStatus", "Check status")
                    )}
                  </button>
                  {openaiOauthConnected && (
                    <button
                      onClick={() => void handleOpenAIOauthRefreshToken()}
                      disabled={openaiOauthAction !== null}
                      className="inline-flex items-center rounded-md border border-border bg-surface2 px-3 py-2 text-xs font-medium text-text hover:bg-surface3 disabled:opacity-60">
                      {openaiOauthAction === "refresh" ? (
                        <>
                          <Spin size="small" className="mr-2" />
                          {t("common:loading.title", {
                            defaultValue: "Loading..."
                          })}
                        </>
                      ) : (
                        t("settings:models.openaiOauth.refresh", "Refresh")
                      )}
                    </button>
                  )}
                  {openaiOauthConnected && (
                    <button
                      onClick={() => void handleOpenAIOauthDisconnect()}
                      disabled={openaiOauthAction !== null}
                      className="inline-flex items-center rounded-md border border-border bg-surface2 px-3 py-2 text-xs font-medium text-text hover:bg-surface3 disabled:opacity-60">
                      {openaiOauthAction === "disconnect" ? (
                        <>
                          <Spin size="small" className="mr-2" />
                          {t("common:loading.title", {
                            defaultValue: "Loading..."
                          })}
                        </>
                      ) : (
                        t("settings:models.openaiOauth.disconnect", "Disconnect")
                      )}
                    </button>
                  )}
                  {openaiOauthActive && (
                    <button
                      onClick={() => void handleOpenAIOauthUseApiKey()}
                      disabled={openaiOauthAction !== null}
                      className="inline-flex items-center rounded-md border border-border bg-surface2 px-3 py-2 text-xs font-medium text-text hover:bg-surface3 disabled:opacity-60">
                      {openaiOauthAction === "switch_api_key" ? (
                        <>
                          <Spin size="small" className="mr-2" />
                          {t("common:loading.title", {
                            defaultValue: "Loading..."
                          })}
                        </>
                      ) : (
                        t(
                          "settings:models.openaiOauth.useApiKeyInstead",
                          "Use API Key Instead"
                        )
                      )}
                    </button>
                  )}
                </div>
                {openaiOauthStatusCode && openaiOauthStatusCode >= 400 && (
                  <p className="mt-2 text-xs text-red-600">
                    {t(
                      "settings:models.openaiOauth.statusError",
                      "Failed to load OpenAI OAuth status."
                    )}
                  </p>
                )}
              </>
            )}
          </div>
          <AvailableModelsList />
        </div>
      </div>
    </div>
  )
}
