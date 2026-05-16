import React from "react"
import {
  Alert,
  Button,
  Card,
  Input,
  InputNumber,
  Segmented,
  Select,
  Space,
  Switch,
  Typography
} from "antd"
import type { LlamacppServerArgsInput } from "@/utils/build-llamacpp-server-args"
import { CollapsibleSection } from "./CollapsibleSection"
import { ServerArgsEditor } from "./ServerArgsEditor"

const { Text } = Typography
const { TextArea } = Input

const CONTEXT_PRESETS = [
  { label: "2K", value: 2048 },
  { label: "4K", value: 4096 },
  { label: "8K", value: 8192 },
  { label: "16K", value: 16384 },
  { label: "32K", value: 32768 },
  { label: "64K", value: 65536 },
  { label: "128K", value: 131072 }
]

const CACHE_TYPE_OPTIONS = [
  "f16",
  "f32",
  "bf16",
  "q8_0",
  "q4_0",
  "q4_1",
  "iq4_nl",
  "q5_0",
  "q5_1"
]

const NUMA_OPTIONS = [
  { label: "Off", value: "off" },
  { label: "Auto", value: "on" },
  { label: "distribute", value: "distribute" },
  { label: "isolate", value: "isolate" },
  { label: "numactl", value: "numactl" }
] as const

type NumaSelectValue = (typeof NUMA_OPTIONS)[number]["value"]

interface ChatActionState {
  visible: boolean
  loading?: boolean
  notice?: string | null
  warnings?: string[]
  onUse: () => void
}

interface LlamacppLaunchPanelProps {
  settings: LlamacppServerArgsInput
  onSettingsChange: (settings: LlamacppServerArgsInput) => void
  selectedModelId?: string
  selectedModelLabel?: string
  isRunning: boolean
  actionLoading: boolean
  inventoryUnavailable: boolean
  adminUnavailable: boolean
  hardwareWarnings: string[]
  presetNotice: string | null
  onStart: () => void
  onStartWithDefaults: () => void
  onExportPreset: () => void
  onOpenImportPreset: () => void
  importPresetInput: React.ReactNode
  chatAction: ChatActionState | null
}

export const LlamacppLaunchPanel: React.FC<LlamacppLaunchPanelProps> = ({
  settings,
  onSettingsChange,
  selectedModelId,
  selectedModelLabel,
  isRunning,
  actionLoading,
  inventoryUnavailable,
  adminUnavailable,
  hardwareWarnings,
  presetNotice,
  onStart,
  onStartWithDefaults,
  onExportPreset,
  onOpenImportPreset,
  importPresetInput,
  chatAction
}) => {
  function updateSetting<K extends keyof LlamacppServerArgsInput>(
    key: K,
    value: LlamacppServerArgsInput[K]
  ) {
    onSettingsChange({ ...settings, [key]: value })
  }

  const numaValue: NumaSelectValue =
    settings.numa === undefined || settings.numa === false
      ? "off"
      : settings.numa === true
        ? "on"
        : settings.numa

  const startDisabled =
    !selectedModelId ||
    isRunning ||
    inventoryUnavailable ||
    adminUnavailable ||
    actionLoading

  return (
    <Card title="Launch">
      <Space orientation="vertical" size="middle" className="w-full">
        <Space wrap>
          <Button onClick={onExportPreset} disabled={actionLoading}>
            Export preset
          </Button>
          <Button onClick={onOpenImportPreset} disabled={actionLoading}>
            Import preset
          </Button>
          {importPresetInput}
        </Space>

        {selectedModelLabel && (
          <Text type="secondary">
            Selected model: <Text code>{selectedModelLabel}</Text>
          </Text>
        )}

        {presetNotice && <Alert type="success" showIcon title={presetNotice} />}

        {hardwareWarnings.length > 0 && (
          <Alert
            type="warning"
            showIcon
            title="Hardware guidance"
            description={
              <Space orientation="vertical" size={2}>
                {hardwareWarnings.map((warning) => (
                  <Text key={warning}>{warning}</Text>
                ))}
              </Space>
            }
          />
        )}

        <div className="rounded-lg border border-border p-4">
          <Text strong className="mb-3 block">
            Main Options
          </Text>
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            <div>
              <Text>Context size</Text>
              <div className="mt-1 flex flex-wrap items-center gap-2">
                <Segmented
                  size="small"
                  options={CONTEXT_PRESETS}
                  value={settings.contextSize}
                  onChange={(value) => updateSetting("contextSize", value as number)}
                />
                <InputNumber
                  size="small"
                  value={settings.contextSize}
                  onChange={(value) => updateSetting("contextSize", value ?? 4096)}
                  min={256}
                  max={131072}
                  step={256}
                  style={{ width: 120 }}
                />
              </div>
            </div>

            <div>
              <Text>GPU layers</Text>
              <InputNumber
                size="small"
                value={settings.gpuLayers}
                onChange={(value) => updateSetting("gpuLayers", value ?? 0)}
                min={-1}
                max={300}
                style={{ width: "100%", marginTop: 4 }}
              />
              <Text type="secondary" className="text-xs">
                0 = CPU only, -1 = all layers
              </Text>
            </div>

            <div>
              <Text>Cache type (K/V)</Text>
              <Select
                size="small"
                value={settings.cacheType}
                onChange={(value) => updateSetting("cacheType", value)}
                options={CACHE_TYPE_OPTIONS.map((value) => ({ label: value, value }))}
                style={{ width: "100%", marginTop: 4 }}
              />
            </div>

            <div>
              <Text>Split mode</Text>
              <Select
                size="small"
                value={settings.splitMode ?? "layer"}
                onChange={(value) => updateSetting("splitMode", value)}
                options={[
                  { label: "none", value: "none" },
                  { label: "layer", value: "layer" },
                  { label: "row", value: "row" }
                ]}
                style={{ width: "100%", marginTop: 4 }}
              />
              <div className="mt-2 flex items-center justify-between">
                <Text type="secondary" className="text-xs">
                  Force row split
                </Text>
                <Switch
                  size="small"
                  checked={Boolean(settings.rowSplit)}
                  onChange={(checked) => updateSetting("rowSplit", checked)}
                />
              </div>
            </div>

            <div>
              <Text>RoPE freq base</Text>
              <InputNumber
                size="small"
                value={settings.ropeFreqBase}
                onChange={(value) => updateSetting("ropeFreqBase", value ?? undefined)}
                min={0}
                step={1000}
                placeholder="auto"
                style={{ width: "100%", marginTop: 4 }}
              />
            </div>

            <div>
              <Text>RoPE freq scale</Text>
              <InputNumber
                size="small"
                value={settings.ropeFreqScale}
                onChange={(value) => updateSetting("ropeFreqScale", value ?? undefined)}
                min={0}
                step={0.01}
                placeholder="auto"
                style={{ width: "100%", marginTop: 4 }}
              />
            </div>

            <div>
              <Text>compress_pos_emb</Text>
              <InputNumber
                size="small"
                value={settings.compressPosEmb}
                onChange={(value) => updateSetting("compressPosEmb", value ?? undefined)}
                min={0.001}
                step={0.01}
                placeholder="optional"
                style={{ width: "100%", marginTop: 4 }}
              />
            </div>

            <div>
              <Text>Flash attention</Text>
              <Select
                size="small"
                value={settings.flashAttn ?? "auto"}
                onChange={(value) => updateSetting("flashAttn", value)}
                options={[
                  { label: "auto", value: "auto" },
                  { label: "on", value: "on" },
                  { label: "off", value: "off" }
                ]}
                style={{ width: "100%", marginTop: 4 }}
              />
            </div>

            <div className="flex items-center justify-between">
              <Text>cpu-moe</Text>
              <Switch
                size="small"
                checked={Boolean(settings.cpuMoe)}
                onChange={(checked) => updateSetting("cpuMoe", checked)}
              />
            </div>

            <div>
              <Text>n-cpu-moe</Text>
              <InputNumber
                size="small"
                value={settings.nCpuMoe}
                onChange={(value) => updateSetting("nCpuMoe", value ?? undefined)}
                min={0}
                style={{ width: "100%", marginTop: 4 }}
              />
            </div>

            <div className="flex items-center justify-between">
              <Text>streaming-llm</Text>
              <Switch
                size="small"
                checked={Boolean(settings.streamingLlm)}
                onChange={(checked) => updateSetting("streamingLlm", checked)}
              />
            </div>

            <div className="flex items-center justify-between">
              <Text>no-kv-offload</Text>
              <Switch
                size="small"
                checked={Boolean(settings.noKvOffload)}
                onChange={(checked) => updateSetting("noKvOffload", checked)}
              />
            </div>
          </div>
        </div>

        <CollapsibleSection
          title="Other Options"
          description="CPU, batching, memory, and extra flags"
        >
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            <div>
              <Text>Threads</Text>
              <InputNumber
                size="small"
                value={settings.threads}
                onChange={(value) => updateSetting("threads", value ?? undefined)}
                min={1}
                max={256}
                placeholder="auto"
                style={{ width: "100%", marginTop: 4 }}
              />
            </div>
            <div>
              <Text>threads_batch</Text>
              <InputNumber
                size="small"
                value={settings.threadsBatch}
                onChange={(value) => updateSetting("threadsBatch", value ?? undefined)}
                min={1}
                max={256}
                placeholder="auto"
                style={{ width: "100%", marginTop: 4 }}
              />
            </div>
            <div>
              <Text>batch_size</Text>
              <InputNumber
                size="small"
                value={settings.batchSize}
                onChange={(value) => updateSetting("batchSize", value ?? undefined)}
                min={1}
                max={8192}
                style={{ width: "100%", marginTop: 4 }}
              />
            </div>
            <div>
              <Text>ubatch_size</Text>
              <InputNumber
                size="small"
                value={settings.ubatchSize}
                onChange={(value) => updateSetting("ubatchSize", value ?? undefined)}
                min={1}
                max={8192}
                style={{ width: "100%", marginTop: 4 }}
              />
            </div>
            <div>
              <Text>main-gpu</Text>
              <InputNumber
                size="small"
                value={settings.mainGpu}
                onChange={(value) => updateSetting("mainGpu", value ?? undefined)}
                min={0}
                style={{ width: "100%", marginTop: 4 }}
              />
            </div>
            <div>
              <Text>tensor_split</Text>
              <Input
                size="small"
                value={settings.tensorSplit}
                onChange={(event) => updateSetting("tensorSplit", event.target.value || undefined)}
                placeholder="e.g. 38,62"
                style={{ marginTop: 4 }}
              />
            </div>
            <div>
              <Text>numa</Text>
              <Select
                size="small"
                value={numaValue}
                onChange={(value: NumaSelectValue) => {
                  if (value === "off") {
                    updateSetting("numa", undefined)
                  } else if (value === "on") {
                    updateSetting("numa", true)
                  } else {
                    updateSetting("numa", value)
                  }
                }}
                options={NUMA_OPTIONS.map((entry) => ({
                  label: entry.label,
                  value: entry.value
                }))}
                style={{ width: "100%", marginTop: 4 }}
              />
            </div>
            <div className="flex items-center justify-between">
              <Text>no-mmap</Text>
              <Switch
                size="small"
                checked={Boolean(settings.noMmap)}
                onChange={(checked) => updateSetting("noMmap", checked)}
              />
            </div>
            <div className="flex items-center justify-between">
              <Text>mlock</Text>
              <Switch
                size="small"
                checked={Boolean(settings.mlock)}
                onChange={(checked) => updateSetting("mlock", checked)}
              />
            </div>
            <div className="md:col-span-2">
              <Text>extra-flags</Text>
              <TextArea
                value={settings.extraFlags}
                onChange={(event) => updateSetting("extraFlags", event.target.value || undefined)}
                placeholder="flag, key=value, n-cpu-moe=27"
                autoSize={{ minRows: 2, maxRows: 6 }}
                style={{ marginTop: 4 }}
              />
              <Text type="secondary" className="text-xs">
                Comma or newline separated; these are parsed and merged without requiring JSON.
              </Text>
            </div>
          </div>
        </CollapsibleSection>

        <CollapsibleSection
          title="Multimodal (vision)"
          description="mmproj and image token controls"
        >
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            <div>
              <Text>mmproj file</Text>
              <Input
                size="small"
                value={settings.mmproj}
                onChange={(event) => updateSetting("mmproj", event.target.value || undefined)}
                placeholder="/absolute/path/to/mmproj.gguf"
                style={{ marginTop: 4 }}
              />
            </div>
            <div>
              <Text>mmproj URL</Text>
              <Input
                size="small"
                value={settings.mmprojUrl}
                onChange={(event) => updateSetting("mmprojUrl", event.target.value || undefined)}
                placeholder="https://..."
                style={{ marginTop: 4 }}
              />
            </div>
            <div className="flex items-center justify-between">
              <Text>mmproj auto</Text>
              <Switch
                size="small"
                checked={settings.mmprojAuto !== false}
                onChange={(checked) => updateSetting("mmprojAuto", checked)}
              />
            </div>
            <div className="flex items-center justify-between">
              <Text>mmproj offload</Text>
              <Switch
                size="small"
                checked={settings.mmprojOffload !== false}
                onChange={(checked) => updateSetting("mmprojOffload", checked)}
              />
            </div>
            <div>
              <Text>image-min-tokens</Text>
              <InputNumber
                size="small"
                value={settings.imageMinTokens}
                onChange={(value) => updateSetting("imageMinTokens", value ?? undefined)}
                min={0}
                style={{ width: "100%", marginTop: 4 }}
              />
            </div>
            <div>
              <Text>image-max-tokens</Text>
              <InputNumber
                size="small"
                value={settings.imageMaxTokens}
                onChange={(value) => updateSetting("imageMaxTokens", value ?? undefined)}
                min={0}
                style={{ width: "100%", marginTop: 4 }}
              />
            </div>
          </div>
        </CollapsibleSection>

        <CollapsibleSection
          title="Speculative decoding"
          description="Draft model and draft token controls"
        >
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            <div className="md:col-span-2">
              <Text>model-draft</Text>
              <Input
                size="small"
                value={settings.draftModel}
                onChange={(event) => updateSetting("draftModel", event.target.value || undefined)}
                placeholder="/absolute/path/to/draft-model.gguf"
                style={{ marginTop: 4 }}
              />
            </div>
            <div>
              <Text>draft-max</Text>
              <InputNumber
                size="small"
                value={settings.draftMax}
                onChange={(value) => updateSetting("draftMax", value ?? undefined)}
                min={0}
                style={{ width: "100%", marginTop: 4 }}
              />
            </div>
            <div>
              <Text>draft-min</Text>
              <InputNumber
                size="small"
                value={settings.draftMin}
                onChange={(value) => updateSetting("draftMin", value ?? undefined)}
                min={0}
                style={{ width: "100%", marginTop: 4 }}
              />
            </div>
            <div>
              <Text>draft-p-min</Text>
              <InputNumber
                size="small"
                value={settings.draftPMin}
                onChange={(value) => updateSetting("draftPMin", value ?? undefined)}
                min={0}
                max={1}
                step={0.01}
                style={{ width: "100%", marginTop: 4 }}
              />
            </div>
            <div>
              <Text>ctx-size-draft</Text>
              <InputNumber
                size="small"
                value={settings.ctxSizeDraft}
                onChange={(value) => updateSetting("ctxSizeDraft", value ?? undefined)}
                min={0}
                style={{ width: "100%", marginTop: 4 }}
              />
            </div>
            <div>
              <Text>gpu-layers-draft</Text>
              <InputNumber
                size="small"
                value={settings.gpuLayersDraft}
                onChange={(value) => updateSetting("gpuLayersDraft", value ?? undefined)}
                min={-1}
                style={{ width: "100%", marginTop: 4 }}
              />
            </div>
            <div className="flex items-center justify-between">
              <Text>cpu-moe-draft</Text>
              <Switch
                size="small"
                checked={Boolean(settings.cpuMoeDraft)}
                onChange={(checked) => updateSetting("cpuMoeDraft", checked)}
              />
            </div>
            <div>
              <Text>n-cpu-moe-draft</Text>
              <InputNumber
                size="small"
                value={settings.nCpuMoeDraft}
                onChange={(value) => updateSetting("nCpuMoeDraft", value ?? undefined)}
                min={0}
                style={{ width: "100%", marginTop: 4 }}
              />
            </div>
          </div>
        </CollapsibleSection>

        <CollapsibleSection
          title="Network & Runtime"
          description="Host/port and runtime overrides"
        >
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            <div>
              <Text>host</Text>
              <Input
                size="small"
                value={settings.host}
                onChange={(event) => updateSetting("host", event.target.value || undefined)}
                placeholder="127.0.0.1"
                style={{ marginTop: 4 }}
              />
            </div>
            <div>
              <Text>port</Text>
              <InputNumber
                size="small"
                value={settings.port}
                onChange={(value) => updateSetting("port", value ?? undefined)}
                min={1}
                max={65535}
                style={{ width: "100%", marginTop: 4 }}
              />
            </div>
          </div>
        </CollapsibleSection>

        <CollapsibleSection
          title="Raw argument overrides"
          description="Optional key-value overrides on top of structured controls"
        >
          <ServerArgsEditor
            value={settings.customArgs || {}}
            onChange={(value) => updateSetting("customArgs", value)}
            placeholder="No overrides. Structured controls already cover common options."
          />
        </CollapsibleSection>

        <Space className="mt-2" wrap>
          <Button
            type="primary"
            onClick={onStart}
            loading={actionLoading}
            disabled={startDisabled}
          >
            Start Server
          </Button>
          <Button
            onClick={onStartWithDefaults}
            loading={actionLoading}
            disabled={startDisabled}
          >
            Start with Defaults
          </Button>
          {chatAction?.visible && (
            <Button onClick={chatAction.onUse} loading={chatAction.loading}>
              Use this in Chat
            </Button>
          )}
        </Space>

        {!selectedModelId && (
          <Alert type="info" showIcon title="Select a model from inventory before launching." />
        )}
        {inventoryUnavailable && (
          <Alert
            type="warning"
            showIcon
            title="Inventory is unavailable, launch is disabled until models load."
          />
        )}
        {isRunning && (
          <Alert
            type="info"
            showIcon
            title="Server is already running. Stop it first to start with new settings."
          />
        )}
        {chatAction?.notice && (
          <Alert type="success" showIcon title={chatAction.notice} />
        )}
        {chatAction?.warnings?.map((warning) => (
          <Alert key={warning} type="warning" showIcon title={warning} />
        ))}
      </Space>
    </Card>
  )
}

export default LlamacppLaunchPanel
