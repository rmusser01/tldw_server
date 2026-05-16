import React from "react"
import { Button, Card, Input, List, Space, Tag, Typography } from "antd"
import { RefreshCw } from "lucide-react"
import { Alert as DesignSystemAlert } from "@/components/ui/primitives"
import type {
  LlamacppAsset,
  LlamacppAssetKind,
  LlamacppAssetsResponse
} from "@/types/llamacpp-admin"

const { Text, Title } = Typography
const passiveAlertProps = {
  role: "status",
  "aria-live": "polite"
} as const

interface LlamacppAssetsPanelProps {
  assets: LlamacppAssetsResponse | null
  loading?: boolean
  registeringPath?: boolean
  importingFolder?: boolean
  error?: string | null
  onRegisterPath: (path: string) => boolean | Promise<boolean>
  onImportFolder: (path: string) => boolean | Promise<boolean>
  onReload: () => void
}

const groupLabels: Record<LlamacppAssetKind, string> = {
  gguf: "GGUF models",
  mmproj: "mmproj projectors",
  folder: "Imported folders",
  unknown: "Other assets"
}

const groupOrder: LlamacppAssetKind[] = ["gguf", "mmproj", "folder", "unknown"]

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

const toAssetGroups = (assetList: LlamacppAsset[]) =>
  groupOrder
    .map((kind) => ({
      kind,
      label: groupLabels[kind],
      items: assetList.filter((asset) => asset.kind === kind)
    }))
    .filter((group) => group.items.length > 0)

const renderMetadataTags = (asset: LlamacppAsset) => (
  <>
    {asset.metadata.parameter_hint && (
      <Tag color="geekblue">{asset.metadata.parameter_hint}</Tag>
    )}
    {asset.metadata.quantization && (
      <Tag color="purple">{asset.metadata.quantization}</Tag>
    )}
    {asset.metadata.context_hint && <Tag>{asset.metadata.context_hint} ctx</Tag>}
    {asset.metadata.family_hint && <Tag>{asset.metadata.family_hint}</Tag>}
  </>
)

const CandidateLabels: React.FC<{ asset: LlamacppAsset }> = ({ asset }) => (
  <Space orientation="vertical" size={2}>
    {asset.mmproj_asset_ids.length > 0 && (
      <Text type="secondary">
        Projector candidates: {asset.mmproj_asset_ids.join(", ")}
      </Text>
    )}
    {asset.base_model_asset_ids.length > 0 && (
      <Text type="secondary">
        Base model candidates: {asset.base_model_asset_ids.join(", ")}
      </Text>
    )}
  </Space>
)

export const LlamacppAssetsPanel: React.FC<LlamacppAssetsPanelProps> = ({
  assets,
  loading = false,
  registeringPath = false,
  importingFolder = false,
  error,
  onRegisterPath,
  onImportFolder,
  onReload
}) => {
  const [assetPath, setAssetPath] = React.useState("")
  const [folderPath, setFolderPath] = React.useState("")
  const assetList = assets?.assets || []
  const assetGroups = toAssetGroups(assetList)

  const handleRegisterPath = async () => {
    const trimmed = assetPath.trim()
    if (!trimmed) return
    const registered = await onRegisterPath(trimmed)
    if (registered) {
      setAssetPath("")
    }
  }

  const handleImportFolder = async () => {
    const trimmed = folderPath.trim()
    if (!trimmed) return
    const imported = await onImportFolder(trimmed)
    if (imported) {
      setFolderPath("")
    }
  }

  return (
    <Card
      title="Assets"
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
            aria-label="Register local asset path"
            value={assetPath}
            onChange={(event) => setAssetPath(event.target.value)}
            placeholder="/absolute/path/to/model-or-mmproj.gguf"
            disabled={registeringPath}
          />
          <Button
            onClick={handleRegisterPath}
            loading={registeringPath}
            disabled={!assetPath.trim()}
          >
            Register asset
          </Button>
        </Space.Compact>

        <Space.Compact className="w-full">
          <Input
            aria-label="Import local asset folder"
            value={folderPath}
            onChange={(event) => setFolderPath(event.target.value)}
            placeholder="/absolute/path/to/model-folder"
            disabled={importingFolder}
          />
          <Button
            onClick={handleImportFolder}
            loading={importingFolder}
            disabled={!folderPath.trim()}
          >
            Import folder
          </Button>
        </Space.Compact>

        {assets?.warnings.map((warning) => (
          <DesignSystemAlert
            key={warning}
            variant="warning"
            {...passiveAlertProps}
            title={warning}
          />
        ))}

        {assets?.scan_limited && (
          <DesignSystemAlert
            variant="warning"
            {...passiveAlertProps}
            title="Asset scan limit reached"
          />
        )}

        {assetGroups.length > 0 ? (
          <Space orientation="vertical" size="middle" className="w-full">
            {assetGroups.map((group) => (
              <section key={group.kind} aria-label={group.label}>
                <Title level={5}>{group.label}</Title>
                <List
                  size="small"
                  bordered
                  rowKey="asset_id"
                  dataSource={group.items}
                  renderItem={(asset) => {
                    const size = formatBytes(asset.size_bytes)
                    return (
                      <List.Item>
                        <Space orientation="vertical" size={4} className="w-full">
                          <Space wrap size="small">
                            <Text strong>{asset.display_name}</Text>
                            <Tag>{asset.source}</Tag>
                            {size && <Tag>{size}</Tag>}
                            {renderMetadataTags(asset)}
                            {asset.capabilities.map((capability) => (
                              <Tag key={capability}>{capability}</Tag>
                            ))}
                          </Space>
                          <Space wrap size="small">
                            <Text code>{asset.asset_id}</Text>
                            <Text type="secondary" className="break-all">
                              {asset.path}
                            </Text>
                          </Space>
                          <CandidateLabels asset={asset} />
                          {asset.warnings.length > 0 && (
                            <Space wrap size="small">
                              {asset.warnings.map((warning) => (
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
              </section>
            ))}
          </Space>
        ) : (
          <Text type="secondary">
            No llama.cpp assets detected. Rescan or register a local asset path.
          </Text>
        )}
      </Space>
    </Card>
  )
}

export default LlamacppAssetsPanel
