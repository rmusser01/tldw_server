import React from "react"
import { useVirtualizer, type VirtualItem } from "@tanstack/react-virtual"
import { Button, Card, Input, Space, Tag, Typography } from "antd"
import { RefreshCw } from "lucide-react"
import { Alert as DesignSystemAlert } from "@/components/ui/primitives"
import type {
  LlamacppInventoryItem,
  LlamacppInventoryResponse
} from "@/types/llamacpp-admin"

const { Text } = Typography
const passiveAlertProps = {
  role: "status",
  "aria-live": "polite"
} as const

// ── Virtualization (admin perf C-S4 / F17) ──
// Filesystem-derived inventories can reach hundreds of models; past the
// threshold the list renders a windowed subset inside a bounded scroll
// container instead of mounting every row.
export const INVENTORY_VIRTUALIZE_THRESHOLD = 30
/** Estimated `<li>` height from the current row layout (tags + paths + padding). */
export const INVENTORY_ROW_ESTIMATE = 88
export const INVENTORY_OVERSCAN = 8

interface LlamacppInventoryPanelProps {
  inventory: LlamacppInventoryResponse | null
  selectedModelId?: string
  activeModel?: string | null
  loading?: boolean
  registering?: boolean
  error?: string | null
  onSelectModel: (modelId: string) => void
  onRegisterPath: (path: string) => boolean | Promise<boolean>
  onReload: () => void
}

const formatBytes = (value?: number | null) => {
  if (!value || value <= 0) return null
  const units = ["B", "KB", "MB", "GB", "TB"]
  let size = value
  let unitIndex = 0
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024
    unitIndex += 1
  }
  return `${size.toFixed(unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`
}

const isActiveModel = (item: LlamacppInventoryItem, activeModel?: string | null) => {
  if (!activeModel) return false
  return [item.model_id, item.basename, item.display_name, item.path].includes(activeModel)
}

const LlamacppInventoryPanelImpl: React.FC<LlamacppInventoryPanelProps> = ({
  inventory,
  selectedModelId,
  activeModel,
  loading = false,
  registering = false,
  error,
  onSelectModel,
  onRegisterPath,
  onReload
}) => {
  const [path, setPath] = React.useState("")
  const models = inventory?.models || []
  const shouldVirtualize = models.length >= INVENTORY_VIRTUALIZE_THRESHOLD
  const scrollContainerRef = React.useRef<HTMLDivElement | null>(null)
  const virtualizer = useVirtualizer({
    count: shouldVirtualize ? models.length : 0,
    getScrollElement: () => scrollContainerRef.current,
    estimateSize: () => INVENTORY_ROW_ESTIMATE,
    overscan: INVENTORY_OVERSCAN,
    getItemKey: (index) => models[index]?.model_id ?? index,
    // jsdom reports zero-height rows; fall back to the estimate so window
    // math stays deterministic in tests (same idiom as Sidepanel/Chat body).
    measureElement: (el) => el?.getBoundingClientRect().height || INVENTORY_ROW_ESTIMATE
  })

  const handleRegister = async () => {
    const trimmed = path.trim()
    if (!trimmed) return
    try {
      const registered = await onRegisterPath(trimmed)
      if (registered) {
        setPath("")
      }
    } catch {
      // Keep the path available for correction/retry. Parent owns the error display.
    }
  }

  const renderModelRow = (item: LlamacppInventoryItem, virtualRow?: VirtualItem) => {
    const selected = selectedModelId === item.model_id
    const active = isActiveModel(item, activeModel)
    const size = formatBytes(item.size_bytes)

    return (
      <li
        key={virtualRow ? virtualRow.key : item.model_id}
        data-index={virtualRow?.index}
        data-model-id={virtualRow ? item.model_id : undefined}
        ref={virtualRow ? virtualizer.measureElement : undefined}
        className="flex flex-col gap-3 px-4 py-2 sm:flex-row sm:items-center sm:justify-between"
        style={
          virtualRow
            ? {
                position: "absolute",
                top: 0,
                left: 0,
                width: "100%",
                transform: `translateY(${virtualRow.start}px)`
              }
            : undefined
        }
      >
        <Space orientation="vertical" size={4} className="min-w-0 flex-1">
          <Space wrap size="small">
            <Text strong>{item.display_name}</Text>
            {active && <Tag color="green">Active</Tag>}
            <Tag>{item.source}</Tag>
            {size && <Tag>{size}</Tag>}
            {item.metadata.parameter_hint && (
              <Tag color="geekblue">{item.metadata.parameter_hint}</Tag>
            )}
            {item.metadata.quantization && (
              <Tag color="purple">{item.metadata.quantization}</Tag>
            )}
            {item.metadata.context_hint && (
              <Tag>{item.metadata.context_hint} ctx</Tag>
            )}
          </Space>
          <Space wrap size="small">
            <Text code>{item.basename}</Text>
            <Text type="secondary" className="break-all">
              {item.path}
            </Text>
          </Space>
          {item.warnings.length > 0 && (
            <Space wrap size="small">
              {item.warnings.map((warning) => (
                <Tag key={warning} color="orange">
                  {warning}
                </Tag>
              ))}
            </Space>
          )}
        </Space>
        <div className="flex shrink-0 flex-wrap gap-2">
          <Button
            size="small"
            type={selected ? "default" : "link"}
            onClick={() => onSelectModel(item.model_id)}
            disabled={selected}
          >
            {selected ? "Selected" : "Select"}
          </Button>
        </div>
      </li>
    )
  }

  return (
    <Card
      title="Inventory"
      loading={loading}
      extra={
        <Button size="small" icon={<RefreshCw size={14} />} onClick={onReload}>
          Rescan
        </Button>
      }
    >
      <Space orientation="vertical" size="middle" className="w-full">
        {error && <DesignSystemAlert variant="error" title={error} />}

        <Space.Compact className="w-full">
          <Input
            aria-label="Register local GGUF path"
            value={path}
            onChange={(event) => setPath(event.target.value)}
            placeholder="/absolute/path/to/model.gguf"
            disabled={registering}
          />
          <Button
            onClick={handleRegister}
            loading={registering}
            disabled={!path.trim()}
          >
            Register path
          </Button>
        </Space.Compact>

        {inventory?.warnings.map((warning) => (
          <DesignSystemAlert
            key={warning}
            variant="warning"
            {...passiveAlertProps}
            title={warning}
          />
        ))}

        {inventory?.scan_limited && (
          <DesignSystemAlert
            variant="warning"
            {...passiveAlertProps}
            title="Inventory scan limit reached"
          />
        )}

        {models.length > 0 ? (
          shouldVirtualize ? (
            <div
              ref={scrollContainerRef}
              className="h-96 overflow-y-auto rounded-lg border border-border"
            >
              <ul
                role="list"
                aria-label="Local GGUF models"
                className="m-0 list-none divide-y divide-border p-0"
                style={{
                  height: virtualizer.getTotalSize(),
                  position: "relative",
                  width: "100%"
                }}
              >
                {virtualizer.getVirtualItems().map((virtualRow) =>
                  renderModelRow(models[virtualRow.index]!, virtualRow)
                )}
              </ul>
            </div>
          ) : (
            <ul
              role="list"
              aria-label="Local GGUF models"
              className="m-0 list-none divide-y divide-border rounded-lg border border-border p-0"
            >
              {models.map((item) => renderModelRow(item))}
            </ul>
          )
        ) : (
          <Text type="secondary">
            No local GGUF models detected. Rescan or register a local GGUF path.
          </Text>
        )}
      </Space>
    </Card>
  )
}

export const LlamacppInventoryPanel = React.memo(LlamacppInventoryPanelImpl)

export default LlamacppInventoryPanel
