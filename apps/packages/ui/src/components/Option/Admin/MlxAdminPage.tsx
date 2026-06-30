import React from "react"
import {
  Typography,
  Card,
  List,
  Space,
  Button,
  Input,
  Select,
  Switch,
  InputNumber,
  AutoComplete
} from "antd"
import { useTranslation } from "react-i18next"
import { AlertTriangle } from "lucide-react"
import { tldwClient, type MlxStatus, type MlxLoadRequest } from "@/services/tldw/TldwApiClient"
import { PageShell } from "@/components/Common/PageShell"
import { buildMlxLoadRequest } from "@/utils/build-mlx-load-request"
import { StatusBanner } from "./StatusBanner"
import { CollapsibleSection } from "./CollapsibleSection"
import { Alert, Badge } from "@/components/ui/primitives"
import {
  deriveAdminGuardFromError,
  sanitizeAdminErrorMessage
} from "./admin-error-utils"

const { Title, Text } = Typography
const { TextArea } = Input

type ProviderConfig = {
  name?: string
  models_info?: Array<Record<string, any>>
  [key: string]: any
}

const DTYPE_OPTIONS = [
  { label: "auto", value: "auto" },
  { label: "float16", value: "float16" },
  { label: "bfloat16", value: "bfloat16" },
  { label: "float32", value: "float32" }
]

const DEVICE_OPTIONS = [
  { label: "auto", value: "auto" },
  { label: "mps (Metal)", value: "mps" },
  { label: "cpu", value: "cpu" }
]

const QUANTIZATION_OPTIONS = [
  { label: "None", value: "" },
  { label: "4bit", value: "4bit" },
  { label: "8bit", value: "8bit" }
]

const coerceModelLabel = (value: unknown): string => {
  if (typeof value === "string" || typeof value === "number") return String(value)
  if (value && typeof value === "object") {
    const record = value as Record<string, unknown>
    const candidate = record.name ?? record.id ?? record.title
    if (typeof candidate === "string" || typeof candidate === "number") {
      return String(candidate)
    }
    try {
      return JSON.stringify(value)
    } catch {
      return "[object]"
    }
  }
  return ""
}

const coerceModelNotes = (value: unknown): string | null => {
  if (typeof value === "string") return value
  if (value && typeof value === "object") {
    const record = value as Record<string, unknown>
    const parts = [record.title, record.description, record.content]
      .filter((entry) => typeof entry === "string" && entry.trim().length > 0) as string[]
    if (parts.length > 0) return parts.join(" - ")
  }
  return null
}

export const MlxAdminPage: React.FC = () => {
  const { t } = useTranslation(["option", "settings", "common"])

  // Provider state
  const [mlxProvider, setMlxProvider] = React.useState<ProviderConfig | null>(null)
  const [loadingProvider, setLoadingProvider] = React.useState(false)

  // Status state
  const [status, setStatus] = React.useState<MlxStatus | null>(null)
  const [statusLoading, setStatusLoading] = React.useState(false)
  const [statusError, setStatusError] = React.useState<string | null>(null)

  // Action state
  const [actionLoading, setActionLoading] = React.useState(false)

  // Basic settings (always visible)
  const [modelPath, setModelPath] = React.useState<string>("")
  const [device, setDevice] = React.useState<string>("auto")
  const [compileFlag, setCompileFlag] = React.useState<boolean>(true)
  const [maxConcurrent, setMaxConcurrent] = React.useState<number>(1)

  // Performance settings
  const [maxSeqLen, setMaxSeqLen] = React.useState<number | undefined>()
  const [maxBatchSize, setMaxBatchSize] = React.useState<number | undefined>()
  const [dtype, setDtype] = React.useState<string>("auto")
  const [maxKvCacheSize, setMaxKvCacheSize] = React.useState<number | undefined>()

  // Advanced settings
  const [quantization, setQuantization] = React.useState<string>("")
  const [warmupFlag, setWarmupFlag] = React.useState<boolean>(true)
  const [trustRemoteCode, setTrustRemoteCode] = React.useState<boolean>(false)
  const [revision, setRevision] = React.useState<string>("")
  const [tokenizer, setTokenizer] = React.useState<string>("")
  const [promptTemplate, setPromptTemplate] = React.useState<string>("")

  // LoRA settings
  const [adapter, setAdapter] = React.useState<string>("")
  const [adapterWeights, setAdapterWeights] = React.useState<string>("")

  // Admin guard
  const [adminGuard, setAdminGuard] = React.useState<"forbidden" | "notFound" | null>(null)

  const markAdminGuardFromError = (err: any) => {
    const guardState = deriveAdminGuardFromError(err)
    if (guardState) {
      setAdminGuard(guardState)
    }
  }

  const loadStatus = React.useCallback(async () => {
    try {
      setStatusLoading(true)
      setStatusError(null)
      const data = await tldwClient.getMlxStatus()
      setStatus(data)
      // Sync form state from loaded model config
      if (data?.model) {
        setModelPath((current) => (current ? current : String(data.model)))
      }
      if (data?.config) {
        if (data.config.device) setDevice(data.config.device)
        if (typeof data.config.compile === "boolean") setCompileFlag(data.config.compile)
        if (typeof data.config.warmup === "boolean") setWarmupFlag(data.config.warmup)
        if (data.config.dtype) setDtype(data.config.dtype)
      }
      if (typeof data?.max_concurrent === "number") {
        setMaxConcurrent(data.max_concurrent)
      }
    } catch (e: any) {
      setStatusError(sanitizeAdminErrorMessage(e, "Failed to load MLX status."))
      markAdminGuardFromError(e)
    } finally {
      setStatusLoading(false)
    }
  }, [])

  const loadProviders = React.useCallback(async () => {
    try {
      setLoadingProvider(true)
      const data = await tldwClient.getLlmProviders()
      const providers: ProviderConfig[] = Array.isArray(data?.providers)
        ? (data.providers as ProviderConfig[])
        : []
      const match =
        providers.find(
          (p) =>
            p.name?.toLowerCase() === "mlx" ||
            p.name?.toLowerCase() === "mlx_lm"
        ) || null
      setMlxProvider(match)
    } catch (e: any) {
      markAdminGuardFromError(e)
    } finally {
      setLoadingProvider(false)
    }
  }, [])

  React.useEffect(() => {
    let cancelled = false
    const init = async () => {
      await Promise.all([loadStatus(), loadProviders()])
      if (cancelled) return
    }
    void init()
    return () => {
      cancelled = true
    }
  }, [loadProviders, loadStatus])

  const handleLoadModel = async () => {
    const path = modelPath.trim()
    if (!path) return

    try {
      setActionLoading(true)
      const payload: MlxLoadRequest = buildMlxLoadRequest({
        modelPath,
        compile: compileFlag,
        warmup: warmupFlag,
        maxConcurrent,
        device,
        maxSeqLen,
        maxBatchSize,
        dtype,
        maxKvCacheSize,
        quantization,
        revision,
        trustRemoteCode,
        tokenizer,
        promptTemplate,
        adapter,
        adapterWeights
      })
      const data = await tldwClient.loadMlxModel(payload)
      setStatus(data)
      setStatusError(null)
    } catch (e: any) {
      setStatusError(sanitizeAdminErrorMessage(e, "Failed to load MLX model."))
      markAdminGuardFromError(e)
    } finally {
      setActionLoading(false)
    }
  }

  const handleUnloadModel = async () => {
    try {
      setActionLoading(true)
      await tldwClient.unloadMlxModel()
      await loadStatus()
    } catch (e: any) {
      setStatusError(sanitizeAdminErrorMessage(e, "Failed to unload MLX model."))
      markAdminGuardFromError(e)
    } finally {
      setActionLoading(false)
    }
  }

  // Build autocomplete options from provider models
  const modelOptions = React.useMemo(() => {
    const models = (mlxProvider?.models_info || []) as Array<Record<string, any>>
    return models
      .map((m) => {
        const label = coerceModelLabel(m.id ?? m.name ?? m.model_id ?? "")
        if (!label) return null
        return { value: label, label }
      })
      .filter((option): option is { value: string; label: string } => option !== null)
  }, [mlxProvider])

  const effectiveState = status?.active ? "active" : "inactive"
  const providerModels = (mlxProvider?.models_info || []) as Array<Record<string, any>>
  const statusUnavailable = !status && Boolean(statusError)
  const concurrencyLabel = status?.active
    ? t("settings:admin.mlxConcurrency", "Concurrent")
    : t(
        "settings:admin.mlxConcurrencyWhenInactive",
        "Configured concurrency (inactive)"
      )

  return (
    <PageShell>
      <Space orientation="vertical" size="large" className="w-full py-6">
        {/* Admin Guard Alert */}
        {adminGuard && (
          <Alert
            variant="warning"
            title={
              adminGuard === "forbidden"
                ? t("settings:admin.adminGuardForbiddenTitle", "Admin access required")
                : t("settings:admin.adminGuardNotFoundTitle", "Admin APIs not available")
            }>
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
          </Alert>
        )}

        {/* Page Header */}
        <div>
          <Title level={2}>
            {t("option:header.adminMlx", "MLX LM Admin")}
          </Title>
          <Text type="secondary">
            {t(
              "settings:admin.mlxIntro",
              "Manage MLX language models: load models with custom configuration and monitor status."
            )}
          </Text>
        </div>

        {!adminGuard && (
          <>
            {/* Status Banner */}
            <StatusBanner
              state={effectiveState}
              loading={statusLoading}
              error={statusError}
              items={[
                {
                  label: t("settings:admin.mlxCurrentModel", "Model"),
                  value: status?.model,
                  code: true
                },
                {
                  label: concurrencyLabel,
                  value: status?.max_concurrent
                }
              ]}
              onRefresh={loadStatus}
              quickAction={
                status?.active
                  ? {
                      label: t("settings:admin.mlxUnloadCta", "Unload"),
                      onClick: handleUnloadModel,
                      loading: actionLoading,
                      danger: true
                    }
                  : undefined
              }
              stateLabel={(state) =>
                state === "active"
                  ? t("settings:admin.mlxActive", "Active")
                  : t("settings:admin.mlxInactive", "Inactive")
              }
            />
            {!status?.active && (
              <Text type="secondary" className="text-xs">
                {t(
                  "settings:admin.mlxInactiveConcurrencyHint",
                  "Concurrency is a configured limit and applies once a model is active."
                )}
              </Text>
            )}
            {statusUnavailable && (
              <Alert
                variant="warning"
                title={t(
                  "settings:admin.mlxUnavailableHint",
                  "MLX controls are temporarily unavailable until status checks succeed."
                )}
              />
            )}

            {/* Model Load Card */}
            <Card title={t("settings:admin.mlxLoadTitle", "Load Model")}>
              <Space orientation="vertical" size="middle" className="w-full">
                {/* Model Path Input */}
                <div>
                  <Text strong className="mb-2 block">
                    {t("settings:admin.mlxModelPathLabel", "Model path or HuggingFace repo")}
                  </Text>
                  <AutoComplete
                    value={modelPath}
                    onChange={setModelPath}
                    options={modelOptions}
                    placeholder={t(
                      "settings:admin.mlxModelPathPlaceholder",
                      "e.g., mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
                    )}
                    className="w-full"
                    filterOption={(input, option) =>
                      (option?.value ?? "").toLowerCase().includes(input.toLowerCase())
                    }
                  />
                  <Text type="secondary" className="mt-1 block text-xs">
                    {t("settings:admin.mlxModelPathHint", "Enter a HuggingFace repo ID or local path to an MLX model")}
                  </Text>
                </div>

                {/* Basic Settings - Always Visible */}
                <div className="rounded-lg border border-border p-4">
                  <Text strong className="mb-3 block">
                    {t("settings:admin.mlxBasicSettings", "Basic Settings")}
                  </Text>
                  <Space orientation="vertical" size="small" className="w-full">
                    {/* Device */}
                    <div className="flex flex-wrap items-center gap-3">
                      <Text className="w-32">
                        {t("settings:admin.mlxDeviceLabel", "Device")}:
                      </Text>
                      <Select
                        size="small"
                        value={device}
                        onChange={setDevice}
                        options={DEVICE_OPTIONS}
                        style={{ width: 160 }}
                      />
                    </div>

                    {/* Compile */}
                    <div className="flex items-center gap-3">
                      <Text className="w-32">
                        {t("settings:admin.mlxCompileLabel", "Compile at load")}:
                      </Text>
                      <Switch
                        size="small"
                        checked={compileFlag}
                        onChange={setCompileFlag}
                      />
                      <Text type="secondary" className="text-xs">
                        {t("settings:admin.mlxCompileHint", "Compile model for faster inference")}
                      </Text>
                    </div>

                    {/* Max Concurrent */}
                    <div className="flex items-center gap-3">
                      <Text className="w-32">
                        {t("settings:admin.mlxMaxConcurrentLabel", "Max concurrent")}:
                      </Text>
                      <InputNumber
                        size="small"
                        value={maxConcurrent}
                        onChange={(val) => setMaxConcurrent(val ?? 1)}
                        min={1}
                        max={32}
                        style={{ width: 80 }}
                      />
                    </div>
                  </Space>
                </div>

                {/* Performance Settings - Collapsible */}
                <CollapsibleSection
                  title={t("settings:admin.mlxPerformanceSettings", "Performance Settings")}
                  description={t("settings:admin.mlxPerformanceSettingsDesc", "Sequence length, batch size, data type, cache")}
                >
                  <Space orientation="vertical" size="small" className="w-full">
                    {/* Max Sequence Length */}
                    <div className="flex items-center gap-3">
                      <Text className="w-36">
                        {t("settings:admin.mlxMaxSeqLen", "Max sequence length")}:
                      </Text>
                      <InputNumber
                        size="small"
                        value={maxSeqLen}
                        onChange={(val) => setMaxSeqLen(val ?? undefined)}
                        min={128}
                        max={131072}
                        placeholder="auto"
                        style={{ width: 100 }}
                      />
                    </div>

                    {/* Max Batch Size */}
                    <div className="flex items-center gap-3">
                      <Text className="w-36">
                        {t("settings:admin.mlxMaxBatchSize", "Max batch size")}:
                      </Text>
                      <InputNumber
                        size="small"
                        value={maxBatchSize}
                        onChange={(val) => setMaxBatchSize(val ?? undefined)}
                        min={1}
                        max={256}
                        placeholder="auto"
                        style={{ width: 100 }}
                      />
                    </div>

                    {/* Data Type */}
                    <div className="flex items-center gap-3">
                      <Text className="w-36">
                        {t("settings:admin.mlxDtype", "Data type")}:
                      </Text>
                      <Select
                        size="small"
                        value={dtype}
                        onChange={setDtype}
                        options={DTYPE_OPTIONS}
                        style={{ width: 120 }}
                      />
                    </div>

                    {/* KV Cache Size */}
                    <div className="flex items-center gap-3">
                      <Text className="w-36">
                        {t("settings:admin.mlxKvCacheSize", "KV cache size")}:
                      </Text>
                      <InputNumber
                        size="small"
                        value={maxKvCacheSize}
                        onChange={(val) => setMaxKvCacheSize(val ?? undefined)}
                        min={0}
                        placeholder="auto"
                        style={{ width: 100 }}
                      />
                      <Text type="secondary" className="text-xs">
                        {t("settings:admin.mlxKvCacheSizeHint", "Max KV cache entries (0 = unlimited)")}
                      </Text>
                    </div>
                  </Space>
                </CollapsibleSection>

                {/* Advanced Settings - Collapsible */}
                <CollapsibleSection
                  title={t("settings:admin.mlxAdvancedSettings", "Advanced Settings")}
                  description={t("settings:admin.mlxAdvancedSettingsDesc", "Quantization, tokenizer, prompt template")}
                >
                  <Space orientation="vertical" size="small" className="w-full">
                    {/* Quantization */}
                    <div className="flex items-center gap-3">
                      <Text className="w-36">
                        {t("settings:admin.mlxQuantization", "Quantization")}:
                      </Text>
                      <Select
                        size="small"
                        value={quantization}
                        onChange={setQuantization}
                        options={QUANTIZATION_OPTIONS}
                        style={{ width: 120 }}
                      />
                    </div>

                    {/* Warmup */}
                    <div className="flex items-center gap-3">
                      <Text className="w-36">
                        {t("settings:admin.mlxWarmupLabel", "Warmup after load")}:
                      </Text>
                      <Switch
                        size="small"
                        checked={warmupFlag}
                        onChange={setWarmupFlag}
                      />
                    </div>

                    {/* Trust Remote Code */}
                    <div className="flex items-center gap-3">
                      <Text className="w-36">
                        {t("settings:admin.mlxTrustRemoteCode", "Trust remote code")}:
                      </Text>
                      <Switch
                        size="small"
                        checked={trustRemoteCode}
                        onChange={setTrustRemoteCode}
                      />
                      {trustRemoteCode && (
                        <Badge variant="warning" size="sm" outline>
                          <AlertTriangle size={12} aria-hidden="true" />
                          {t("settings:admin.mlxTrustRemoteCodeWarning", "Security risk")}
                        </Badge>
                      )}
                    </div>

                    {/* Revision */}
                    <div className="flex items-center gap-3">
                      <Text className="w-36">
                        {t("settings:admin.mlxRevision", "HF revision")}:
                      </Text>
                      <Input
                        size="small"
                        value={revision}
                        onChange={(e) => setRevision(e.target.value)}
                        placeholder="main"
                        style={{ width: 160 }}
                      />
                    </div>

                    {/* Tokenizer */}
                    <div className="flex items-center gap-3">
                      <Text className="w-36">
                        {t("settings:admin.mlxTokenizer", "Custom tokenizer")}:
                      </Text>
                      <Input
                        size="small"
                        value={tokenizer}
                        onChange={(e) => setTokenizer(e.target.value)}
                        placeholder={t("settings:admin.mlxTokenizerPlaceholder", "Path or repo ID")}
                        style={{ width: 240 }}
                      />
                    </div>

                    {/* Prompt Template */}
                    <div>
                      <Text className="mb-1 block">
                        {t("settings:admin.mlxPromptTemplate", "Prompt template")}:
                      </Text>
                      <TextArea
                        size="small"
                        value={promptTemplate}
                        onChange={(e) => setPromptTemplate(e.target.value)}
                        placeholder={t("settings:admin.mlxPromptTemplatePlaceholder", "Custom prompt template (optional)")}
                        autoSize={{ minRows: 2, maxRows: 6 }}
                        className="font-mono text-xs"
                      />
                    </div>
                  </Space>
                </CollapsibleSection>

                {/* LoRA Settings - Collapsible */}
                <CollapsibleSection
                  title={t("settings:admin.mlxLoraSettings", "LoRA / Adapter Settings")}
                  description={t("settings:admin.mlxLoraSettingsDesc", "Load fine-tuned adapters")}
                >
                  <Space orientation="vertical" size="small" className="w-full">
                    {/* Adapter Path */}
                    <div className="flex items-center gap-3">
                      <Text className="w-36">
                        {t("settings:admin.mlxAdapter", "Adapter path")}:
                      </Text>
                      <Input
                        size="small"
                        value={adapter}
                        onChange={(e) => setAdapter(e.target.value)}
                        placeholder={t("settings:admin.mlxAdapterPlaceholder", "Path to LoRA adapter")}
                        style={{ width: 280 }}
                      />
                    </div>

                    {/* Adapter Weights */}
                    <div className="flex items-center gap-3">
                      <Text className="w-36">
                        {t("settings:admin.mlxAdapterWeights", "Adapter weights")}:
                      </Text>
                      <Input
                        size="small"
                        value={adapterWeights}
                        onChange={(e) => setAdapterWeights(e.target.value)}
                        placeholder={t("settings:admin.mlxAdapterWeightsPlaceholder", "Path to adapter weights")}
                        style={{ width: 280 }}
                      />
                    </div>
                  </Space>
                </CollapsibleSection>

                {/* Action Buttons */}
                <Space className="mt-4">
                  <Button
                    type="primary"
                    onClick={handleLoadModel}
                    loading={actionLoading}
                    disabled={!modelPath.trim() || statusLoading || statusUnavailable}
                  >
                    {t("settings:admin.mlxLoadCta", "Load Model")}
                  </Button>
                  <Button
                    danger
                    onClick={handleUnloadModel}
                    loading={actionLoading}
                    disabled={!status?.active || statusLoading || statusUnavailable}
                  >
                    {t("settings:admin.mlxUnloadCta", "Unload Model")}
                  </Button>
                </Space>

                {status?.active && (
                  <Alert
                    variant="info"
                    title={t(
                      "settings:admin.mlxAlreadyLoaded",
                      "A model is currently loaded. Unload it first or load a different model to replace it."
                    )}
                  />
                )}
              </Space>
            </Card>

            {/* Provider Models - Collapsible */}
            <CollapsibleSection
              title={t("settings:admin.mlxProviderTitle", "Available Provider Models")}
              description={t("settings:admin.mlxProviderDesc", `${providerModels.length} model(s) configured`)}
            >
              {loadingProvider ? (
                <Text type="secondary">
                  {t("common:loading.title", "Loading...")}
                </Text>
              ) : providerModels.length > 0 ? (
                <List
                  size="small"
                  bordered
                  dataSource={providerModels}
                  renderItem={(m) => {
                    const idLabel = coerceModelLabel(m.id ?? m.name ?? m.model_id ?? "") || "model"
                    const notes = coerceModelNotes(m.notes)
                    const capabilities = m.capabilities as Record<string, boolean> | undefined
                    return (
                      <List.Item
                        actions={[
                          <Button
                            key="use"
                            size="small"
                            type="link"
                            onClick={() => setModelPath(idLabel)}
                          >
                            {t("common:use", "Use")}
                          </Button>
                        ]}
                      >
                        <div className="flex flex-col gap-1">
                          <div className="flex flex-wrap items-center gap-2">
                            <Text code>{idLabel}</Text>
                            {capabilities?.vision && (
                              <Badge variant="primary" size="sm">
                                {t("settings:admin.mlxVision", "Vision")}
                              </Badge>
                            )}
                            {capabilities?.tool_use && (
                              <Badge variant="info" size="sm">
                                {t("settings:admin.mlxTools", "Tools")}
                              </Badge>
                            )}
                            {capabilities?.audio_input && (
                              <Badge variant="warning" size="sm">
                                {t("settings:admin.mlxAudio", "Audio")}
                              </Badge>
                            )}
                          </div>
                          {notes && (
                            <Text type="secondary" className="text-xs">
                              {notes}
                            </Text>
                          )}
                        </div>
                      </List.Item>
                    )
                  }}
                />
              ) : (
                <Text type="secondary">
                  {t(
                    "settings:admin.mlxNoModels",
                    "No MLX models configured. Enable MLX provider on the server to see models here."
                  )}
                </Text>
              )}
            </CollapsibleSection>
          </>
        )}
      </Space>
    </PageShell>
  )
}

export default MlxAdminPage
