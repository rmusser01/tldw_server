import React from "react"
import {
  Alert,
  Button,
  Card,
  Descriptions,
  Empty,
  Radio,
  Space,
  Tag,
  Typography
} from "antd"
import {
  CheckCircle2,
  ExternalLink,
  RefreshCw,
  SearchCheck,
  ServerCog
} from "lucide-react"

import { useSetupReadiness } from "./hooks/useSetupReadiness"
import type {
  SetupReadinessClientMode,
  SetupReadinessLane,
  SetupReadinessProfile
} from "@/services/tldw/setup-readiness"

const { Paragraph, Text, Title } = Typography

type ReadinessSetupScreenProps = {
  mode?: SetupReadinessClientMode
  onComplete?: () => void
}

const statusColor = (status?: string) => {
  switch (status) {
    case "ready":
      return "green"
    case "ready_with_warnings":
    case "previewed":
      return "gold"
    case "failed":
    case "blocked":
      return "red"
    case "provisioning":
      return "blue"
    case "skipped":
      return "default"
    default:
      return "default"
  }
}

const formatStatus = (status?: string) =>
  String(status || "not_configured").replace(/_/g, " ")

const displayValue = (value: unknown): string => {
  if (value == null || value === "") return "Not set"
  if (Array.isArray(value)) return value.map(displayValue).join(", ")
  if (typeof value === "object") return JSON.stringify(value)
  return String(value)
}

const selectionEntries = (lane: SetupReadinessLane) =>
  Object.entries(lane.selection || {}).filter(([, value]) => value != null && value !== "")

const ttsSelection = (lane: SetupReadinessLane): string | null => {
  const value = lane.selection?.tts_choice || lane.selection?.tts_model || lane.selection?.tts_provider
  return value ? String(value) : null
}

const renderList = (items: string[]) => {
  if (items.length === 0) return null
  return (
    <ul className="mb-0 mt-2 space-y-1 pl-5 text-xs text-text-muted">
      {items.map((item, index) => (
        <li key={`${item}:${index}`}>{item}</li>
      ))}
    </ul>
  )
}

const ReadinessLaneCard = ({ lane }: { lane: SetupReadinessLane }) => {
  const entries = selectionEntries(lane)
  const tts = lane.lane_id === "speech" ? ttsSelection(lane) : null

  return (
    <Card size="small" className="h-full" styles={{ body: { minHeight: 180 } }}>
      <Space orientation="vertical" size="small" className="w-full">
        <div className="flex items-start justify-between gap-3">
          <div>
            <Text strong>{lane.label || lane.lane_id}</Text>
            {lane.primary_capability && (
              <div className="mt-1 text-xs text-text-muted">
                {lane.primary_capability.replace(/_/g, " ")}
              </div>
            )}
          </div>
          <Tag color={statusColor(lane.status)}>{formatStatus(lane.status)}</Tag>
        </div>

        {entries.length > 0 ? (
          <Descriptions size="small" column={1}>
            {entries.map(([key, value]) => (
              <Descriptions.Item key={key} label={key.replace(/_/g, " ")}>
                {displayValue(value)}
              </Descriptions.Item>
            ))}
          </Descriptions>
        ) : (
          <Text type="secondary">No selection yet.</Text>
        )}

        {tts && <Text className="secondary text-xs text-text-muted">TTS: {tts}</Text>}
        {renderList([...(lane.warnings || []), ...(lane.blockers || [])])}
      </Space>
    </Card>
  )
}

export const ReadinessSetupScreen: React.FC<ReadinessSetupScreenProps> = ({
  mode = "first-run",
  onComplete
}) => {
  const {
    error,
    fallbackUrl,
    guard,
    loading,
    preview,
    previewSelection,
    previewing,
    profiles,
    provision,
    provisionResult,
    provisioning,
    refresh,
    status,
    verification,
    verify,
    verifying
  } = useSetupReadiness({ mode })
  const profileOptions = profiles?.profiles || []
  const recommendedProfileId = profiles?.recommended_profile_id || profileOptions[0]?.profile_id || null
  const [selectedProfileId, setSelectedProfileId] = React.useState<string | null>(null)

  React.useEffect(() => {
    if (!recommendedProfileId) return
    const selectedExists = profileOptions.some((profile) => profile.profile_id === selectedProfileId)
    if (!selectedProfileId || !selectedExists) {
      setSelectedProfileId(recommendedProfileId)
    }
  }, [profileOptions, recommendedProfileId, selectedProfileId])

  const selectedProfile = React.useMemo<SetupReadinessProfile | null>(
    () => profileOptions.find((profile) => profile.profile_id === selectedProfileId) || null,
    [profileOptions, selectedProfileId]
  )

  const lanes = React.useMemo(
    () => status?.lanes || profiles?.lanes || [],
    [profiles, status]
  )
  const consequences = React.useMemo(
    () =>
      lanes.flatMap((lane) =>
        (lane.consequences || []).map((item) => `${lane.label || lane.lane_id}: ${item}`)
      ),
    [lanes]
  )
  const overlays = preview?.overlays || status?.active_overlays || profiles?.active_overlays || []

  const handlePreview = React.useCallback(
    async (profileId = selectedProfileId) => {
      if (!profileId) return
      await previewSelection({ profile_id: profileId })
    },
    [previewSelection, selectedProfileId]
  )

  const handleProfileChange = React.useCallback(
    (event: any) => {
      const nextProfileId = String(event.target.value)
      setSelectedProfileId(nextProfileId)
      void handlePreview(nextProfileId)
    },
    [handlePreview]
  )

  const handleProvision = React.useCallback(async () => {
    if (preview?.preview_id) {
      await provision({ preview_id: preview.preview_id })
      return
    }
    if (selectedProfileId) {
      await provision({ selection: { profile_id: selectedProfileId } })
    }
  }, [preview, provision, selectedProfileId])

  const handleVerify = React.useCallback(async () => {
    if (preview?.preview_id) {
      await verify({ preview_id: preview.preview_id })
      return
    }
    if (selectedProfileId) {
      await verify({ selection: { profile_id: selectedProfileId } })
    }
  }, [preview, selectedProfileId, verify])

  if (guard === "remote_setup_blocked") {
    return (
      <section className="mx-auto w-full max-w-5xl px-4 py-6">
        <Alert
          type="warning"
          showIcon
          title="Local setup required"
          description={
            <Space orientation="vertical" size="small">
              <span>{error || "First-run setup is restricted to local requests."}</span>
              <Button href={fallbackUrl} icon={<ExternalLink className="h-4 w-4" />}>
                Open backend setup
              </Button>
            </Space>
          }
        />
      </section>
    )
  }

  return (
    <section className="mx-auto w-full max-w-6xl px-4 py-6">
      <Space orientation="vertical" size="large" className="w-full">
        <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
          <div className="max-w-3xl">
            <Title level={2} className="mb-1">
              Setup readiness
            </Title>
            <Paragraph type="secondary" className="mb-0">
              Choose the defaults that make chat, search, ingestion, and speech ready before the first large import.
            </Paragraph>
          </div>
          <Space wrap>
            <Button href={fallbackUrl} icon={<ExternalLink className="h-4 w-4" />}>
              Open backend setup
            </Button>
            <Button onClick={() => void refresh()} icon={<RefreshCw className="h-4 w-4" />}>
              Refresh
            </Button>
          </Space>
        </div>

        {error && (
          <Alert
            type={guard === "admin_required" ? "warning" : "error"}
            showIcon
            title={guard === "admin_required" ? "Admin access required" : "Readiness request failed"}
            description={error}
          />
        )}

        <Card loading={loading} size="small">
          <Space orientation="vertical" size="middle" className="w-full">
            <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
              <div>
                <Text strong>Readiness profile</Text>
                <Paragraph type="secondary" className="mb-0">
                  Profiles set defaults only. Provisioning is a separate action.
                </Paragraph>
              </div>
              {profiles?.machine_profile && (
                <Space wrap size="small">
                  {profiles.machine_profile.apple_silicon && <Tag>Apple Silicon</Tag>}
                  {profiles.machine_profile.platform && (
                    <Tag>{displayValue(profiles.machine_profile.platform)}</Tag>
                  )}
                  {typeof profiles.machine_profile.free_disk_gb === "number" && (
                    <Tag>{`${profiles.machine_profile.free_disk_gb} GB free`}</Tag>
                  )}
                </Space>
              )}
            </div>

            {profileOptions.length > 0 ? (
              <Radio.Group value={selectedProfileId || undefined} onChange={handleProfileChange}>
                <Space wrap>
                  {profileOptions.map((profile) => (
                    <Radio.Button key={profile.profile_id} value={profile.profile_id}>
                      {profile.label}
                    </Radio.Button>
                  ))}
                </Space>
              </Radio.Group>
            ) : (
              <Empty description="No setup readiness profiles are available." />
            )}

            {selectedProfile && (
              <div className="rounded-md border border-border bg-surface2/30 p-3">
                <Text strong>{selectedProfile.label}</Text>
                {selectedProfile.description && (
                  <Paragraph type="secondary" className="mb-0 mt-1">
                    {selectedProfile.description}
                  </Paragraph>
                )}
                {selectedProfile.advanced && (
                  <details className="mt-3 text-sm text-text-muted">
                    <summary className="cursor-pointer text-text">Advanced controls</summary>
                    <pre className="mt-2 max-h-48 overflow-auto rounded-md bg-surface p-3 text-xs">
                      {JSON.stringify(selectedProfile.lanes || {}, null, 2)}
                    </pre>
                  </details>
                )}
              </div>
            )}
          </Space>
        </Card>

        <div className="grid gap-3 lg:grid-cols-3">
          {lanes.map((lane) => (
            <ReadinessLaneCard key={lane.lane_id} lane={lane} />
          ))}
        </div>

        <Card size="small" title="Preview and provision">
          <Space orientation="vertical" size="middle" className="w-full">
            {preview ? (
              <div className="grid gap-3 lg:grid-cols-3">
                <div>
                  <Text strong>Config updates</Text>
                  <Paragraph type="secondary" className="mb-0">
                    {Object.keys(preview.config_updates || {}).length} section(s)
                  </Paragraph>
                </div>
                <div>
                  <Text strong>Install plan</Text>
                  <Paragraph type="secondary" className="mb-0">
                    {Object.keys(preview.install_plan || {}).length > 0 ? "Provisioning work planned" : "No downloads needed"}
                  </Paragraph>
                </div>
                <div>
                  <Text strong>Overlays</Text>
                  <Paragraph type="secondary" className="mb-0">
                    {overlays.length > 0 ? overlays.map(formatStatus).join(", ") : "None"}
                  </Paragraph>
                </div>
              </div>
            ) : (
              <Alert
                type="info"
                showIcon
                title="Review before provisioning"
                description="Select a profile or preview the current selection before running Provision now."
              />
            )}

            {consequences.length > 0 && (
              <Alert
                type="warning"
                showIcon
                title="Skipped-lane consequences"
                description={renderList(consequences)}
              />
            )}

            {provisionResult && (
              <Alert
                type="success"
                showIcon
                title={`Provision status: ${formatStatus(provisionResult.operation_status || provisionResult.status)}`}
                description={provisionResult.status_url}
              />
            )}

            {verification && (
              <Alert
                type="info"
                showIcon
                title={`Verification: ${formatStatus(verification.status)}`}
                description={verification.verified_at || "Verification completed."}
              />
            )}

            <Space wrap>
              <Button
                type="primary"
                icon={<SearchCheck className="h-4 w-4" />}
                loading={previewing}
                disabled={!selectedProfileId}
                onClick={() => void handlePreview()}
              >
                Preview selection
              </Button>
              <Button
                icon={<ServerCog className="h-4 w-4" />}
                loading={provisioning}
                disabled={!selectedProfileId}
                onClick={() => void handleProvision()}
              >
                Provision now
              </Button>
              <Button
                icon={<CheckCircle2 className="h-4 w-4" />}
                loading={verifying}
                disabled={!selectedProfileId}
                onClick={() => void handleVerify()}
              >
                Verify readiness
              </Button>
              {onComplete && (
                <Button type="link" onClick={onComplete}>
                  Continue
                </Button>
              )}
            </Space>
          </Space>
        </Card>
      </Space>
    </section>
  )
}

export default ReadinessSetupScreen
