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
import type { RadioChangeEvent } from "antd"
import {
  CheckCircle2,
  ExternalLink,
  RefreshCw,
  SearchCheck,
  ServerCog
} from "lucide-react"
import { useTranslation } from "react-i18next"

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
  onUnavailable?: () => void
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

const displayValue = (value: unknown, emptyValue: string): string => {
  if (value == null || value === "") return emptyValue
  if (Array.isArray(value)) return value.map((item) => displayValue(item, emptyValue)).join(", ")
  if (typeof value === "object") return JSON.stringify(value)
  return String(value)
}

const selectionEntries = (lane: SetupReadinessLane) =>
  Object.entries(lane.selection || {}).filter(([, value]) => value != null && value !== "")

const ttsSelection = (lane: SetupReadinessLane): string | null => {
  const value = lane.selection?.tts_choice || lane.selection?.tts_model || lane.selection?.tts_provider
  return value ? String(value) : null
}

const installPlanHasWork = (value: unknown): boolean => {
  if (!value) return false
  if (Array.isArray(value)) return value.length > 0
  if (typeof value === "object") return Object.values(value).some(installPlanHasWork)
  return Boolean(value)
}

const statusKey = (status?: string) => `setupReadiness.statuses.${status || "not_configured"}`

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
  const { t } = useTranslation("option")
  const entries = selectionEntries(lane)
  const tts = lane.lane_id === "speech" ? ttsSelection(lane) : null
  const emptyValue = t("setupReadiness.valueNotSet", "Not set")

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
          <Tag color={statusColor(lane.status)}>
            {t(statusKey(lane.status), formatStatus(lane.status))}
          </Tag>
        </div>

        {entries.length > 0 ? (
          <Descriptions size="small" column={1}>
            {entries.map(([key, value]) => (
              <Descriptions.Item key={key} label={key.replace(/_/g, " ")}>
                {displayValue(value, emptyValue)}
              </Descriptions.Item>
            ))}
          </Descriptions>
        ) : (
          <Text type="secondary">{t("setupReadiness.lanes.noSelection", "No selection yet.")}</Text>
        )}

        {tts && (
          <Text className="secondary text-xs text-text-muted">
            {t("setupReadiness.lanes.ttsSelection", "TTS: {{value}}", { value: tts })}
          </Text>
        )}
        {renderList([...(lane.warnings || []), ...(lane.blockers || [])])}
      </Space>
    </Card>
  )
}

export const ReadinessSetupScreen: React.FC<ReadinessSetupScreenProps> = ({
  mode = "first-run",
  onComplete,
  onUnavailable
}) => {
  const { t } = useTranslation("option")
  const {
    error,
    errorKey,
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
    if (mode === "first-run" && guard === "not_found") {
      onUnavailable?.()
    }
  }, [guard, mode, onUnavailable])

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

  const previewLanes = React.useMemo(
    () =>
      Object.values(preview?.lanes || {}).filter(
        (lane): lane is SetupReadinessLane => Boolean(lane && typeof lane === "object")
      ),
    [preview]
  )
  const lanes = React.useMemo(
    () => (previewLanes.length > 0 ? previewLanes : status?.lanes || profiles?.lanes || []),
    [previewLanes, profiles, status]
  )
  const consequences = React.useMemo(
    () =>
      lanes.flatMap((lane) =>
        (lane.consequences || []).map((item) => `${lane.label || lane.lane_id}: ${item}`)
      ),
    [lanes]
  )
  const overlays = preview?.overlays || status?.active_overlays || profiles?.active_overlays || []
  const emptyValue = t("setupReadiness.valueNotSet", "Not set")
  const errorDescription = errorKey
    ? t(errorKey, error || "Readiness request failed.")
    : error

  const handlePreview = React.useCallback(
    async (profileId = selectedProfileId) => {
      if (!profileId) return
      await previewSelection({ profile_id: profileId })
    },
    [previewSelection, selectedProfileId]
  )

  const handleProfileChange = React.useCallback(
    (event: RadioChangeEvent) => {
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
          title={t("setupReadiness.errors.localSetupRequiredTitle", "Local setup required")}
          description={
            <Space orientation="vertical" size="small">
              <span>
                {error ||
                  t(
                    "setupReadiness.errors.localSetupRequiredDescription",
                    "First-run setup is restricted to local requests."
                  )}
              </span>
              <Button href={fallbackUrl} icon={<ExternalLink className="h-4 w-4" />}>
                {t("setupReadiness.actions.openBackendSetup", "Open backend setup")}
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
            <Title level={1} className="mb-1">
              {t("setupReadiness.title", "Setup readiness")}
            </Title>
            <Paragraph type="secondary" className="mb-0">
              {t(
                "setupReadiness.description",
                "Choose the defaults that make chat, search, ingestion, and speech ready before the first large import."
              )}
            </Paragraph>
          </div>
          <Space wrap>
            <Button href={fallbackUrl} icon={<ExternalLink className="h-4 w-4" />}>
              {t("setupReadiness.actions.openBackendSetup", "Open backend setup")}
            </Button>
            <Button onClick={() => void refresh()} icon={<RefreshCw className="h-4 w-4" />}>
              {t("setupReadiness.actions.refresh", "Refresh")}
            </Button>
          </Space>
        </div>

        {error && (
          <Alert
            type={guard === "admin_required" ? "warning" : "error"}
            showIcon
            title={
              guard === "admin_required"
                ? t("setupReadiness.errors.adminRequiredTitle", "Admin access required")
                : t("setupReadiness.errors.requestFailedTitle", "Readiness request failed")
            }
            description={errorDescription}
          />
        )}

        <Card loading={loading} size="small">
          <Space orientation="vertical" size="middle" className="w-full">
            <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
              <div>
                <Text strong>{t("setupReadiness.profile.title", "Readiness profile")}</Text>
                <Paragraph type="secondary" className="mb-0">
                  {t(
                    "setupReadiness.profile.description",
                    "Profiles set defaults only. Provisioning is a separate action."
                  )}
                </Paragraph>
              </div>
              {profiles?.machine_profile && (
                <Space wrap size="small">
                  {profiles.machine_profile.apple_silicon && (
                    <Tag>{t("setupReadiness.machine.appleSilicon", "Apple Silicon")}</Tag>
                  )}
                  {profiles.machine_profile.platform && (
                    <Tag>{displayValue(profiles.machine_profile.platform, emptyValue)}</Tag>
                  )}
                  {typeof profiles.machine_profile.free_disk_gb === "number" && (
                    <Tag>
                      {t("setupReadiness.machine.freeDiskGb", "{{value}} GB free", {
                        value: profiles.machine_profile.free_disk_gb
                      })}
                    </Tag>
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
              <Empty
                description={t(
                  "setupReadiness.profile.empty",
                  "No setup readiness profiles are available."
                )}
              />
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
                    <summary className="cursor-pointer text-text">
                      {t("setupReadiness.profile.advancedControls", "Advanced controls")}
                    </summary>
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

        <Card size="small" title={t("setupReadiness.preview.title", "Preview and provision")}>
          <Space orientation="vertical" size="middle" className="w-full">
            {preview ? (
              <div className="grid gap-3 lg:grid-cols-3">
                <div>
                  <Text strong>{t("setupReadiness.preview.configUpdates", "Config updates")}</Text>
                  <Paragraph type="secondary" className="mb-0">
                    {t("setupReadiness.preview.configUpdateCount", "{{count}} section(s)", {
                      count: Object.keys(preview.config_updates || {}).length
                    })}
                  </Paragraph>
                </div>
                <div>
                  <Text strong>{t("setupReadiness.preview.installPlan", "Install plan")}</Text>
                  <Paragraph type="secondary" className="mb-0">
                    {installPlanHasWork(preview.install_plan)
                      ? t("setupReadiness.preview.installPlanWork", "Provisioning work planned")
                      : t("setupReadiness.preview.noDownloads", "No downloads needed")}
                  </Paragraph>
                </div>
                <div>
                  <Text strong>{t("setupReadiness.preview.overlays", "Overlays")}</Text>
                  <Paragraph type="secondary" className="mb-0">
                    {overlays.length > 0
                      ? overlays
                          .map((overlay) => t(statusKey(overlay), formatStatus(overlay)))
                          .join(", ")
                      : t("setupReadiness.preview.none", "None")}
                  </Paragraph>
                </div>
              </div>
            ) : (
              <Alert
                type="info"
                showIcon
                title={t("setupReadiness.preview.reviewTitle", "Review before provisioning")}
                description={t(
                  "setupReadiness.preview.reviewDescription",
                  "Select a profile or preview the current selection before running Provision now."
                )}
              />
            )}

            {consequences.length > 0 && (
              <Alert
                type="warning"
                showIcon
                title={t("setupReadiness.consequences.title", "Skipped-lane consequences")}
                description={renderList(consequences)}
              />
            )}

            {provisionResult && (
              <Alert
                type="success"
                showIcon
                title={t("setupReadiness.provision.statusTitle", "Provision status: {{status}}", {
                  status: t(
                    statusKey(provisionResult.operation_status || provisionResult.status),
                    formatStatus(provisionResult.operation_status || provisionResult.status)
                  )
                })}
                description={t(
                  "setupReadiness.provision.description",
                  "Readiness provisioning has started. Refresh or watch the status cards for progress."
                )}
              />
            )}

            {verification && (
              <Alert
                type="info"
                showIcon
                title={t("setupReadiness.verification.statusTitle", "Verification: {{status}}", {
                  status: t(statusKey(verification.status), formatStatus(verification.status))
                })}
                description={
                  verification.verified_at ||
                  t("setupReadiness.verification.description", "Verification completed.")
                }
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
                {t("setupReadiness.actions.preview", "Preview selection")}
              </Button>
              <Button
                icon={<ServerCog className="h-4 w-4" />}
                loading={provisioning}
                disabled={!selectedProfileId}
                onClick={() => void handleProvision()}
              >
                {t("setupReadiness.actions.provision", "Provision now")}
              </Button>
              <Button
                icon={<CheckCircle2 className="h-4 w-4" />}
                loading={verifying}
                disabled={!selectedProfileId}
                onClick={() => void handleVerify()}
              >
                {t("setupReadiness.actions.verify", "Verify readiness")}
              </Button>
              {onComplete && (
                <Button type="link" onClick={onComplete}>
                  {t("setupReadiness.actions.continue", "Continue")}
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
