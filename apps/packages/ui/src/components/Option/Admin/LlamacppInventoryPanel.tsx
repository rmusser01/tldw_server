import React from "react"
import { Button, Card, Input, List, Space, Tag, Typography } from "antd"
import { RefreshCw } from "lucide-react"
import { Alert as DesignSystemAlert } from "@/components/ui/primitives"
import type {
  LlamacppInventoryItem,
  LlamacppInventoryResponse
} from "@/types/llamacpp-admin"

const { Text } = Typography

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

export const LlamacppInventoryPanel: React.FC<LlamacppInventoryPanelProps> = ({
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
          <DesignSystemAlert key={warning} variant="warning" title={warning} />
        ))}

        {inventory?.scan_limited && (
          <DesignSystemAlert
            variant="warning"
            title="Inventory scan limit reached"
          />
        )}

        {models.length > 0 ? (
          <List
            size="small"
            bordered
            dataSource={models}
            renderItem={(item) => {
              const selected = selectedModelId === item.model_id
              const active = isActiveModel(item, activeModel)
              const size = formatBytes(item.size_bytes)

              return (
                <List.Item
                  actions={[
                    <Button
                      key="select"
                      size="small"
                      type={selected ? "default" : "link"}
                      onClick={() => onSelectModel(item.model_id)}
                      disabled={selected}
                    >
                      {selected ? "Selected" : "Select"}
                    </Button>
                  ]}
                >
                  <Space orientation="vertical" size={4} className="w-full">
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
                </List.Item>
              )
            }}
          />
        ) : (
          <Text type="secondary">
            No local GGUF models detected. Rescan or register a local GGUF path.
          </Text>
        )}
      </Space>
    </Card>
  )
}

export default LlamacppInventoryPanel
