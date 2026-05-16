import React from "react"
import {
  Alert,
  Button,
  Card,
  Empty,
  List,
  Space,
  Tag,
  Tooltip,
  Typography
} from "antd"
import { Pause, Play, RefreshCw, RotateCcw, Square, Unplug } from "lucide-react"
import type {
  LlamacppProfile,
  LlamacppRuntime,
  LlamacppRuntimeState
} from "@/types/llamacpp-admin"

const { Text } = Typography

interface LlamacppRuntimePanelProps {
  profiles: LlamacppProfile[]
  runtimes: LlamacppRuntime[]
  loading?: boolean
  error?: string | null
  actionProfileId?: string | null
  onRefresh: () => void
  onStart: (profileId: string) => void
  onStop: (profileId: string) => void
  onPause: (profileId: string) => void
  onResume: (profileId: string) => void
  onUseInChat: (profileId: string) => void
}

type RuntimeRow = {
  profileId: string
  profile?: LlamacppProfile
  runtime?: LlamacppRuntime
}

const stateColor = (state?: LlamacppRuntimeState) => {
  switch (state) {
    case "running":
      return "green"
    case "starting":
      return "blue"
    case "paused":
      return "gold"
    case "failed":
      return "red"
    case "stopped":
      return "default"
    case "defined":
    default:
      return "default"
  }
}

const formatEndpoint = (profile?: LlamacppProfile, runtime?: LlamacppRuntime) => {
  if (runtime?.endpoint) return runtime.endpoint
  const host = runtime?.host || profile?.host
  const port = runtime?.port ?? profile?.port
  if (host && port) return `${host}:${port}`
  return "Not running"
}

const profileLabel = (row: RuntimeRow) =>
  row.profile?.name || row.runtime?.profile_id || row.profileId

const modelLabel = (row: RuntimeRow) =>
  row.runtime?.model_path ||
  row.runtime?.model_id ||
  row.profile?.model_path ||
  row.profile?.model_id ||
  "No model selected"

export const LlamacppRuntimePanel: React.FC<LlamacppRuntimePanelProps> = ({
  profiles,
  runtimes,
  loading = false,
  error,
  actionProfileId = null,
  onRefresh,
  onStart,
  onStop,
  onPause,
  onResume,
  onUseInChat
}) => {
  const rows = React.useMemo<RuntimeRow[]>(() => {
    const runtimeByProfile = new Map(runtimes.map((runtime) => [runtime.profile_id, runtime]))
    const profileRows = profiles.map((profile) => ({
      profileId: profile.profile_id,
      profile,
      runtime: runtimeByProfile.get(profile.profile_id)
    }))
    const profileIds = new Set(profiles.map((profile) => profile.profile_id))
    const runtimeOnlyRows = runtimes
      .filter((runtime) => !profileIds.has(runtime.profile_id))
      .map((runtime) => ({
        profileId: runtime.profile_id,
        runtime
      }))
    return [...profileRows, ...runtimeOnlyRows]
  }, [profiles, runtimes])

  return (
    <Card
      title="Runtime instances"
      loading={loading}
      extra={
        <Button size="small" icon={<RefreshCw size={14} />} onClick={onRefresh}>
          Refresh
        </Button>
      }
      aria-label="Runtime instances"
    >
      <Space orientation="vertical" size="middle" className="w-full">
        {error && <Alert type="warning" showIcon title={error} />}

        {rows.length === 0 ? (
          <Empty
            image={Empty.PRESENTED_IMAGE_SIMPLE}
            description="No llama.cpp runtime profiles are available."
          />
        ) : (
          <List
            size="small"
            bordered
            dataSource={rows}
            renderItem={(row) => {
              const label = profileLabel(row)
              const state = row.runtime?.state || "defined"
              const endpoint = formatEndpoint(row.profile, row.runtime)
              const port = row.runtime?.port ?? row.profile?.port
              const isRunning = state === "running"
              const isPaused = state === "paused"
              const isBusy = actionProfileId === row.profileId
              const warnings = row.runtime?.warnings || []
              const actions: React.ReactNode[] = []

              if (isRunning) {
                actions.push(
                  <Tooltip key="use" title="Use this runtime in Chat">
                    <Button
                      size="small"
                      icon={<Unplug size={14} />}
                      onClick={() => onUseInChat(row.profileId)}
                      loading={isBusy}
                      aria-label={`Use ${label} in Chat`}
                    >
                      Use in Chat
                    </Button>
                  </Tooltip>,
                  <Button
                    key="pause"
                    size="small"
                    icon={<Pause size={14} />}
                    onClick={() => onPause(row.profileId)}
                    loading={isBusy}
                    aria-label={`Pause ${label}`}
                  >
                    Pause
                  </Button>,
                  <Button
                    key="stop"
                    size="small"
                    danger
                    icon={<Square size={14} />}
                    onClick={() => onStop(row.profileId)}
                    loading={isBusy}
                    aria-label={`Stop ${label}`}
                  >
                    Stop
                  </Button>
                )
              } else if (isPaused) {
                actions.push(
                  <Button
                    key="resume"
                    size="small"
                    icon={<RotateCcw size={14} />}
                    onClick={() => onResume(row.profileId)}
                    loading={isBusy}
                    aria-label={`Resume ${label}`}
                  >
                    Resume
                  </Button>
                )
              } else {
                actions.push(
                  <Button
                    key="start"
                    size="small"
                    type="primary"
                    icon={<Play size={14} />}
                    onClick={() => onStart(row.profileId)}
                    loading={isBusy}
                    aria-label={`Start ${label}`}
                  >
                    Start
                  </Button>
                )
              }

              return (
                <List.Item actions={actions}>
                  <Space orientation="vertical" size={4} className="w-full">
                    <Space wrap size="small">
                      <Text strong>{label}</Text>
                      <Tag color={stateColor(state)}>{state}</Tag>
                      {row.profile?.mode && <Tag>{row.profile.mode}</Tag>}
                      {row.profile?.enabled === false && <Tag color="orange">disabled</Tag>}
                      {row.runtime?.pid && <Tag>pid {row.runtime.pid}</Tag>}
                      {port && <Tag>{port}</Tag>}
                      {row.runtime?.restart_count ? (
                        <Tag>{row.runtime.restart_count} restarts</Tag>
                      ) : null}
                    </Space>

                    <Space wrap size="small">
                      <Text code className="break-all">
                        {endpoint}
                      </Text>
                      <Text type="secondary" className="break-all">
                        {modelLabel(row)}
                      </Text>
                    </Space>

                    {warnings.length > 0 && (
                      <Space wrap size="small">
                        {warnings.map((warning) => (
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
        )}
      </Space>
    </Card>
  )
}

export default LlamacppRuntimePanel
