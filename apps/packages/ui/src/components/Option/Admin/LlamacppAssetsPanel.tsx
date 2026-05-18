import React from "react"
import { Button, Card, Input, List, Space, Tag, Typography } from "antd"
import { RefreshCw } from "lucide-react"
import { Alert as DesignSystemAlert } from "@/components/ui/primitives"
import type {
  LlamacppAcquisitionJobListResponse,
  LlamacppAcquisitionJobResponse,
  LlamacppAsset,
  LlamacppAssetDownloadRequest,
  LlamacppAssetImportPreviewResponse,
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
  previewingFolder?: boolean
  importPreview?: LlamacppAssetImportPreviewResponse | null
  downloads?: LlamacppAcquisitionJobListResponse | null
  loadingDownloads?: boolean
  startingDownload?: boolean
  cancelingDownloadId?: string | null
  error?: string | null
  onRegisterPath: (path: string) => boolean | Promise<boolean>
  onImportFolder: (path: string) => boolean | Promise<boolean>
  onPreviewImportFolder?: (path: string) => boolean | Promise<boolean>
  onStartDownload?: (payload: LlamacppAssetDownloadRequest) => boolean | Promise<boolean>
  onCancelDownload?: (jobId: string) => boolean | Promise<boolean>
  onReloadDownloads?: () => void
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

const acquisitionStatusColors: Record<string, string> = {
  queued: "blue",
  pending: "blue",
  running: "processing",
  processing: "processing",
  completed: "green",
  succeeded: "green",
  done: "green",
  failed: "red",
  canceled: "default",
  cancelled: "default"
}

const cancelableDownloadStatuses = new Set([
  "queued",
  "pending",
  "running",
  "processing",
  "downloading"
])

const previewCountLabel = (kind: string) => (kind === "gguf" ? "GGUF" : kind)

const progressPercent = (job: LlamacppAcquisitionJobResponse): number | null => {
  const raw = job.progress?.progress_percent
  const value =
    typeof raw === "number"
      ? raw
      : typeof raw === "string"
        ? Number(raw)
        : NaN
  if (!Number.isFinite(value)) return null
  return Math.max(0, Math.min(100, Math.round(value)))
}

const downloadLabel = (job: LlamacppAcquisitionJobResponse) =>
  job.source_label || job.destination_path || `Download ${job.job_id}`

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
  previewingFolder = false,
  importPreview = null,
  downloads = null,
  loadingDownloads = false,
  startingDownload = false,
  cancelingDownloadId = null,
  error,
  onRegisterPath,
  onImportFolder,
  onPreviewImportFolder,
  onStartDownload,
  onCancelDownload,
  onReloadDownloads,
  onReload
}) => {
  const [assetPath, setAssetPath] = React.useState("")
  const [folderPath, setFolderPath] = React.useState("")
  const [previewedFolderPath, setPreviewedFolderPath] = React.useState("")
  const [downloadUrl, setDownloadUrl] = React.useState("")
  const [downloadDestinationDir, setDownloadDestinationDir] = React.useState("")
  const [downloadFilename, setDownloadFilename] = React.useState("")
  const assetList = assets?.assets || []
  const assetGroups = toAssetGroups(assetList)
  const downloadJobs = downloads?.jobs || []
  const showDownloadWorkflow = Boolean(onStartDownload) || downloadJobs.length > 0
  const importUsesPreview = Boolean(onPreviewImportFolder)

  const handleRegisterPath = async () => {
    const trimmed = assetPath.trim()
    if (!trimmed) return
    const registered = await onRegisterPath(trimmed)
    if (registered) {
      setAssetPath("")
    }
  }

  const handlePreviewFolder = async () => {
    const trimmed = folderPath.trim()
    if (!trimmed || !onPreviewImportFolder) return
    const previewed = await onPreviewImportFolder(trimmed)
    if (previewed) {
      setPreviewedFolderPath(trimmed)
    }
  }

  const handleImportFolder = async () => {
    const trimmed = importUsesPreview
      ? previewedFolderPath || folderPath.trim()
      : folderPath.trim()
    if (!trimmed) return
    const imported = await onImportFolder(trimmed)
    if (imported) {
      setFolderPath("")
      setPreviewedFolderPath("")
    }
  }

  const handleStartDownload = async () => {
    if (!onStartDownload) return
    const trimmedUrl = downloadUrl.trim()
    if (!trimmedUrl) return
    const payload: LlamacppAssetDownloadRequest = {
      url: trimmedUrl
    }
    const trimmedDestinationDir = downloadDestinationDir.trim()
    const trimmedFilename = downloadFilename.trim()
    if (trimmedDestinationDir) {
      payload.destination_dir = trimmedDestinationDir
    }
    if (trimmedFilename) {
      payload.filename = trimmedFilename
    }

    const started = await onStartDownload(payload)
    if (started) {
      setDownloadUrl("")
      setDownloadDestinationDir("")
      setDownloadFilename("")
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
            onChange={(event) => {
              setFolderPath(event.target.value)
              setPreviewedFolderPath("")
            }}
            placeholder="/absolute/path/to/model-folder"
            disabled={importingFolder || previewingFolder}
          />
          {importUsesPreview ? (
            <Button
              onClick={handlePreviewFolder}
              loading={previewingFolder}
              disabled={!folderPath.trim()}
            >
              Preview folder
            </Button>
          ) : (
            <Button
              onClick={handleImportFolder}
              loading={importingFolder}
              disabled={!folderPath.trim()}
            >
              Import folder
            </Button>
          )}
        </Space.Compact>

        {importUsesPreview && importPreview && (
          <section aria-label="Import preview">
            <Space orientation="vertical" size="small" className="w-full">
              <Space wrap size="small">
                <Text strong>Import preview</Text>
                <Text type="secondary" className="break-all">
                  {importPreview.folder.path}
                </Text>
                {Object.entries(importPreview.asset_counts).map(([kind, count]) => (
                  <Tag key={kind}>
                    {previewCountLabel(kind)}: {count}
                  </Tag>
                ))}
                {importPreview.scan_limited && <Tag color="orange">scan limited</Tag>}
              </Space>
              {importPreview.warnings.map((warning) => (
                <DesignSystemAlert
                  key={warning}
                  variant="warning"
                  {...passiveAlertProps}
                  title={warning}
                />
              ))}
              <Button
                size="small"
                onClick={handleImportFolder}
                loading={importingFolder}
                disabled={!previewedFolderPath && !folderPath.trim()}
              >
                Confirm import
              </Button>
            </Space>
          </section>
        )}

        {showDownloadWorkflow && (
          <section aria-label="Asset downloads">
            <Space orientation="vertical" size="small" className="w-full">
              <Space wrap className="w-full" size="small">
                <Input
                  aria-label="Download source URL"
                  value={downloadUrl}
                  onChange={(event) => setDownloadUrl(event.target.value)}
                  placeholder="https://example.com/model.gguf"
                  disabled={startingDownload}
                  className="min-w-[260px] flex-1"
                />
                <Input
                  aria-label="Download destination directory"
                  value={downloadDestinationDir}
                  onChange={(event) => setDownloadDestinationDir(event.target.value)}
                  placeholder="/absolute/path/to/models"
                  disabled={startingDownload}
                  className="min-w-[220px] flex-1"
                />
                <Input
                  aria-label="Download filename"
                  value={downloadFilename}
                  onChange={(event) => setDownloadFilename(event.target.value)}
                  placeholder="model.gguf"
                  disabled={startingDownload}
                  className="min-w-[160px] flex-1"
                />
                <Button
                  onClick={handleStartDownload}
                  loading={startingDownload}
                  disabled={!downloadUrl.trim() || !onStartDownload}
                >
                  Queue download
                </Button>
                {onReloadDownloads && (
                  <Button size="small" onClick={onReloadDownloads} loading={loadingDownloads}>
                    Refresh downloads
                  </Button>
                )}
              </Space>

              {downloadJobs.length > 0 && (
                <List
                  size="small"
                  bordered
                  loading={loadingDownloads}
                  rowKey="job_id"
                  dataSource={downloadJobs}
                  renderItem={(job) => {
                    const status = job.status.toLowerCase()
                    const percent = progressPercent(job)
                    const canCancel = cancelableDownloadStatuses.has(status)
                    return (
                      <List.Item
                        actions={
                          canCancel && onCancelDownload
                            ? [
                                <Button
                                  key="cancel"
                                  size="small"
                                  danger
                                  loading={cancelingDownloadId === job.job_id}
                                  onClick={() => {
                                    void onCancelDownload(job.job_id)
                                  }}
                                >
                                  Cancel download {job.job_id}
                                </Button>
                              ]
                            : undefined
                        }
                      >
                        <Space orientation="vertical" size={4} className="w-full">
                          <Space wrap size="small">
                            <Text strong>{downloadLabel(job)}</Text>
                            <Tag color={acquisitionStatusColors[status] || "default"}>
                              {job.status}
                            </Tag>
                            {percent !== null && <Tag>{percent}%</Tag>}
                            {job.asset_id && <Tag>{job.asset_id}</Tag>}
                          </Space>
                          {job.destination_path && (
                            <Text type="secondary" className="break-all">
                              {job.destination_path}
                            </Text>
                          )}
                          {job.error_message && (
                            <DesignSystemAlert
                              variant="error"
                              {...passiveAlertProps}
                              title={job.error_message}
                            />
                          )}
                          {job.warnings.map((warning) => (
                            <DesignSystemAlert
                              key={warning}
                              variant="warning"
                              {...passiveAlertProps}
                              title={warning}
                            />
                          ))}
                        </Space>
                      </List.Item>
                    )
                  }}
                />
              )}
            </Space>
          </section>
        )}

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
