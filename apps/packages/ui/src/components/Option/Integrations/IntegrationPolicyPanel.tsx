import React, { useEffect, useMemo, useState } from "react"
import {
  Alert,
  Button,
  Card,
  Descriptions,
  Form,
  Input,
  InputNumber,
  List,
  Select,
  Space,
  Switch,
  Tag,
  Typography,
  message
} from "antd"
import type {
  DiscordWorkspacePolicyResponse,
  DiscordWorkspacePolicyUpdate,
  SlackWorkspacePolicyResponse,
  SlackWorkspacePolicyUpdate,
  TelegramBotConfigResponse,
  TelegramBotConfigUpdate,
  TelegramLinkedActorItem,
  TelegramPairingCodeResponse
} from "@/services/integrations-control-plane"
import {
  RecoveryCallout,
  type CapabilityStateDescriptor
} from "@/components/ui/state"

type SharedPolicyPanelProps = {
  errorMessage?: string
  errorState?: CapabilityStateDescriptor
  loading?: boolean
  onRefresh?: () => void
}

type SlackPolicyPanelProps = SharedPolicyPanelProps & {
  provider: "slack"
  policy?: SlackWorkspacePolicyResponse
  onSave: (payload: SlackWorkspacePolicyUpdate) => Promise<unknown>
}

type DiscordPolicyPanelProps = SharedPolicyPanelProps & {
  provider: "discord"
  policy?: DiscordWorkspacePolicyResponse
  onSave: (payload: DiscordWorkspacePolicyUpdate) => Promise<unknown>
}

type TelegramPolicyPanelProps = SharedPolicyPanelProps & {
  provider: "telegram"
  bot?: TelegramBotConfigResponse
  linkedActors?: TelegramLinkedActorItem[]
  pairingCode?: TelegramPairingCodeResponse | null
  onSave: (payload: TelegramBotConfigUpdate) => Promise<unknown>
  onGeneratePairingCode: () => Promise<TelegramPairingCodeResponse>
  onRevokeActor: (actorId: number) => Promise<unknown>
}

export type IntegrationPolicyPanelProps =
  | SlackPolicyPanelProps
  | DiscordPolicyPanelProps
  | TelegramPolicyPanelProps

type WorkspacePolicyFormValues = {
  default_response_mode?: "ephemeral" | "thread" | "channel" | "guild" | "workspace_and_user" | "guild_and_user"
  service_user_id?: string
  strict_user_mapping?: boolean
  quota_per_minute?: number
  user_quota_per_minute?: number
  status_scope?: "workspace" | "workspace_and_user" | "guild" | "guild_and_user"
}

const toCommaList = (value: string[] | undefined): string => (value ?? []).join(", ")

const splitCommaList = (value?: string): string[] | undefined => {
  if (value == null) return undefined
  const trimmed = value.trim()
  if (!trimmed) return []
  return trimmed.split(",").map((entry) => entry.trim()).filter(Boolean)
}

const WorkspacePolicyEditor: React.FC<
  {
    title: string
    panelKind: "slack" | "discord"
    summary?: string
    installationIds?: string[]
    policy?: SlackWorkspacePolicyResponse["policy"] | DiscordWorkspacePolicyResponse["policy"]
    errorMessage?: string
    errorState?: CapabilityStateDescriptor
    loading?: boolean
    onSave: (payload: SlackWorkspacePolicyUpdate | DiscordWorkspacePolicyUpdate) => Promise<unknown>
    onRefresh?: () => void
  }
> = ({ title, panelKind, summary, installationIds = [], policy, errorMessage, errorState, loading, onSave, onRefresh }) => {
  const [form] = Form.useForm<WorkspacePolicyFormValues>()
  const [saving, setSaving] = useState(false)
  const isUnavailable = Boolean(errorMessage || errorState) || (!loading && !policy)

  useEffect(() => {
    const quotaPerMinute =
      panelKind === "slack"
        ? (policy as SlackWorkspacePolicyResponse["policy"] | undefined)?.workspace_quota_per_minute
        : (policy as DiscordWorkspacePolicyResponse["policy"] | undefined)?.guild_quota_per_minute

    form.setFieldsValue({
      default_response_mode: policy?.default_response_mode,
      service_user_id: policy?.service_user_id ?? undefined,
      strict_user_mapping: policy?.strict_user_mapping ?? false,
      quota_per_minute: quotaPerMinute,
      user_quota_per_minute: policy?.user_quota_per_minute,
      status_scope: policy?.status_scope
    })
  }, [form, policy])

  const handleSave = async () => {
    if (isUnavailable || !policy) {
      return
    }

    const values = await form.validateFields()
    setSaving(true)
    try {
      if (values.quota_per_minute != null && values.quota_per_minute <= 0) {
        throw new Error(`${panelKind === "slack" ? "Workspace" : "Guild"} quota must be greater than 0`)
      }
      if (values.user_quota_per_minute != null && values.user_quota_per_minute <= 0) {
        throw new Error("User quota must be greater than 0")
      }

      const payload =
        panelKind === "slack"
          ? {
              allowed_commands: policy.allowed_commands,
              channel_allowlist: policy.channel_allowlist,
              channel_denylist: policy.channel_denylist,
              default_response_mode: values.default_response_mode as "ephemeral" | "thread" | "channel" | undefined,
              service_user_id: values.service_user_id?.trim() || undefined,
              strict_user_mapping: values.strict_user_mapping,
              user_mappings: policy.user_mappings,
              workspace_quota_per_minute: values.quota_per_minute,
              user_quota_per_minute: values.user_quota_per_minute,
              status_scope: values.status_scope as "workspace" | "workspace_and_user" | undefined
            }
          : {
              allowed_commands: policy.allowed_commands,
              channel_allowlist: policy.channel_allowlist,
              channel_denylist: policy.channel_denylist,
              default_response_mode: values.default_response_mode as "ephemeral" | "channel" | undefined,
              service_user_id: values.service_user_id?.trim() || undefined,
              strict_user_mapping: values.strict_user_mapping,
              user_mappings: policy.user_mappings,
              guild_quota_per_minute: values.quota_per_minute,
              user_quota_per_minute: values.user_quota_per_minute,
              status_scope: values.status_scope as "guild" | "guild_and_user" | undefined
            }
      await onSave(payload as SlackWorkspacePolicyUpdate | DiscordWorkspacePolicyUpdate)
      message.success(`${title} saved`)
      onRefresh?.()
    } catch (error: any) {
      message.error(error?.message || `Unable to save ${title.toLowerCase()}`)
    } finally {
      setSaving(false)
    }
  }

  const policyDescription = useMemo(
    () => [
      { label: "Installations", value: installationIds.length ? installationIds.join(", ") : "—" },
      { label: "Allowed commands", value: policy?.allowed_commands?.join(", ") || "—" },
      { label: "Channel allowlist", value: toCommaList(policy?.channel_allowlist) || "—" },
      { label: "Channel denylist", value: toCommaList(policy?.channel_denylist) || "—" }
    ],
    [installationIds, policy]
  )

  return (
    <Card title={title} loading={loading}>
      {summary ? <Typography.Paragraph type="secondary">{summary}</Typography.Paragraph> : null}
      {errorState ? (
        <RecoveryCallout
          state={errorState.state}
          title={errorState.title}
          message={errorState.message}
          diagnostics={errorState.diagnostics}
          primaryAction={
            onRefresh
              ? {
                  label: "Try again",
                  onClick: onRefresh
                }
              : undefined
          }
          className="mb-4"
        />
      ) : errorMessage ? (
        <Alert type="error" showIcon style={{ marginBottom: 16 }} title={errorMessage} />
      ) : null}
      {!errorMessage && !errorState && isUnavailable ? (
        <Alert type="warning" showIcon style={{ marginBottom: 16 }} title={`${title} is unavailable`} />
      ) : null}
      <Descriptions size="small" bordered column={1} style={{ marginBottom: 16 }}>
        {policyDescription.map((item) => (
          <Descriptions.Item key={item.label} label={item.label}>
            {item.value}
          </Descriptions.Item>
        ))}
      </Descriptions>
      <Form form={form} layout="vertical">
        <Form.Item label="Default response mode" name="default_response_mode">
          <Select
            disabled={isUnavailable}
            options={[
              { value: "ephemeral", label: "Ephemeral" },
              ...(panelKind === "slack"
                ? [
                    { value: "thread", label: "Thread" },
                    { value: "channel", label: "Channel" }
                  ]
                : [{ value: "channel", label: "Channel" }])
            ]}
          />
        </Form.Item>
        <Form.Item label="Service user ID" name="service_user_id">
          <Input placeholder="Optional service user ID" disabled={isUnavailable} />
        </Form.Item>
        <Form.Item label="Strict user mapping" name="strict_user_mapping" valuePropName="checked">
          <Switch disabled={isUnavailable} />
        </Form.Item>
        <Space wrap style={{ width: "100%" }} align="start">
          <Form.Item
            label={panelKind === "slack" ? "Workspace quota / min" : "Guild quota / min"}
            name="quota_per_minute"
          >
          <InputNumber
              aria-label={panelKind === "slack" ? "Workspace quota / min" : "Guild quota / min"}
              min={0}
              style={{ width: 180 }}
              disabled={isUnavailable}
            />
          </Form.Item>
          <Form.Item label="User quota / min" name="user_quota_per_minute">
            <InputNumber aria-label="User quota / min" min={0} style={{ width: 180 }} disabled={isUnavailable} />
          </Form.Item>
          <Form.Item label="Status scope" name="status_scope">
            <Select
              style={{ width: 220 }}
              disabled={isUnavailable}
              options={
                panelKind === "slack"
                  ? [
                      { value: "workspace", label: "Workspace" },
                      { value: "workspace_and_user", label: "Workspace and user" }
                    ]
                  : [
                      { value: "guild", label: "Guild" },
                      { value: "guild_and_user", label: "Guild and user" }
                    ]
              }
            />
          </Form.Item>
        </Space>
        <Button type="primary" onClick={() => void handleSave()} loading={saving} disabled={isUnavailable}>
          {`Save ${title}`}
        </Button>
      </Form>
    </Card>
  )
}

const TelegramPolicyEditor: React.FC<
  {
    bot?: TelegramBotConfigResponse
    linkedActors?: TelegramLinkedActorItem[]
    pairingCode?: TelegramPairingCodeResponse | null
    errorMessage?: string
    errorState?: CapabilityStateDescriptor
    loading?: boolean
    onSave: (payload: TelegramBotConfigUpdate) => Promise<unknown>
    onGeneratePairingCode: () => Promise<TelegramPairingCodeResponse>
    onRevokeActor: (actorId: number) => Promise<unknown>
    onRefresh?: () => void
  }
> = ({ bot, linkedActors = [], pairingCode, errorMessage, errorState, loading, onSave, onGeneratePairingCode, onRevokeActor, onRefresh }) => {
  const [form] = Form.useForm<TelegramBotConfigUpdate>()
  const [saving, setSaving] = useState(false)
  const [generating, setGenerating] = useState(false)
  const [localPairingCode, setLocalPairingCode] = useState<TelegramPairingCodeResponse | null>(pairingCode ?? null)
  const isUnavailable = Boolean(errorMessage || errorState) || (!loading && !bot)

  useEffect(() => {
    setLocalPairingCode(pairingCode ?? null)
  }, [pairingCode])

  useEffect(() => {
    form.setFieldsValue({
      bot_token: "",
      webhook_secret: "",
      bot_username: bot?.bot_username,
      enabled: bot?.enabled ?? false
    })
  }, [bot, form])

  const handleSave = async () => {
    if (isUnavailable) {
      return
    }
    const values = await form.validateFields()
    setSaving(true)
    try {
      await onSave({
        bot_token: values.bot_token,
        webhook_secret: values.webhook_secret,
        bot_username: values.bot_username?.trim() || undefined,
        enabled: Boolean(values.enabled)
      })
      message.success("Telegram bot saved")
      onRefresh?.()
    } catch (error: any) {
      message.error(error?.message || "Unable to save Telegram bot")
    } finally {
      setSaving(false)
    }
  }

  const handleGeneratePairingCode = async () => {
    if (isUnavailable) {
      return
    }
    setGenerating(true)
    try {
      const result = await onGeneratePairingCode()
      setLocalPairingCode(result)
      message.success("Pairing code generated")
      onRefresh?.()
    } catch (error: any) {
      message.error(error?.message || "Unable to generate pairing code")
    } finally {
      setGenerating(false)
    }
  }

  const handleRevoke = async (actorId: number) => {
    try {
      await onRevokeActor(actorId)
      message.success("Telegram actor revoked")
      onRefresh?.()
    } catch (error: any) {
      message.error(error?.message || "Unable to revoke Telegram actor")
    }
  }

  return (
    <Card title="Telegram bot" loading={loading}>
      {errorState ? (
        <RecoveryCallout
          state={errorState.state}
          title={errorState.title}
          message={errorState.message}
          diagnostics={errorState.diagnostics}
          primaryAction={
            onRefresh
              ? {
                  label: "Try again",
                  onClick: onRefresh
                }
              : undefined
          }
          className="mb-4"
        />
      ) : errorMessage ? (
        <Alert type="error" showIcon style={{ marginBottom: 16 }} title={errorMessage} />
      ) : null}
      <Descriptions size="small" bordered column={1} style={{ marginBottom: 16 }}>
        <Descriptions.Item label="Bot username">{bot?.bot_username ?? "—"}</Descriptions.Item>
        <Descriptions.Item label="Enabled">
          <Tag color={bot?.enabled ? "green" : "default"}>{bot?.enabled ? "enabled" : "disabled"}</Tag>
        </Descriptions.Item>
        <Descriptions.Item label="Linked actors">{linkedActors.length}</Descriptions.Item>
      </Descriptions>

      <Form form={form} layout="vertical">
        <Form.Item label="Bot token" name="bot_token" rules={[{ required: true, message: "Bot token is required" }]}>
          <Input.Password placeholder="Enter bot token" autoComplete="off" disabled={isUnavailable} />
        </Form.Item>
        <Form.Item label="Webhook secret" name="webhook_secret" rules={[{ required: true, message: "Webhook secret is required" }]}>
          <Input.Password placeholder="Enter webhook secret" autoComplete="off" disabled={isUnavailable} />
        </Form.Item>
        <Form.Item label="Bot username" name="bot_username">
          <Input placeholder="@ExampleBot" disabled={isUnavailable} />
        </Form.Item>
        <Form.Item label="Enabled" name="enabled" valuePropName="checked">
          <Switch disabled={isUnavailable} />
        </Form.Item>
        <Space wrap>
          <Button type="primary" onClick={() => void handleSave()} loading={saving} disabled={isUnavailable}>
            Save bot config
          </Button>
          <Button onClick={() => void handleGeneratePairingCode()} loading={generating} disabled={isUnavailable}>
            Generate pairing code
          </Button>
        </Space>
      </Form>

      {localPairingCode ? (
        <Alert
          type="success"
          showIcon
          style={{ marginTop: 16 }}
          title="Pairing code generated"
          description={
            <div style={{ display: "flex", flexDirection: "column" }}>
              <span>
                <strong>{localPairingCode.pairing_code}</strong>
              </span>
              <span>Expires at {new Date(localPairingCode.expires_at).toLocaleString()}</span>
            </div>
          }
        />
      ) : null}

      <div style={{ marginTop: 16 }}>
        <Typography.Title level={5}>Linked actors</Typography.Title>
        {linkedActors.length > 0 ? (
          <List
            dataSource={linkedActors}
            renderItem={(item) => (
              <List.Item
                actions={[
                  <Button key="revoke" size="small" danger onClick={() => void handleRevoke(item.id)}>
                    Revoke
                  </Button>
                ]}
              >
                <List.Item.Meta
                  title={item.telegram_username || `Telegram user ${item.telegram_user_id}`}
                  description={`Auth user ${item.auth_user_id}`}
                />
              </List.Item>
            )}
          />
        ) : (
          <Typography.Text type="secondary">No linked actors found.</Typography.Text>
        )}
      </div>
    </Card>
  )
}

export const IntegrationPolicyPanel: React.FC<IntegrationPolicyPanelProps> = (props) => {
  if (props.provider === "telegram") {
    return (
      <TelegramPolicyEditor
        bot={props.bot}
        linkedActors={props.linkedActors}
        pairingCode={props.pairingCode}
        loading={props.loading}
        onSave={props.onSave}
        onGeneratePairingCode={props.onGeneratePairingCode}
        onRevokeActor={props.onRevokeActor}
        errorMessage={props.errorMessage}
        errorState={props.errorState}
        onRefresh={props.onRefresh}
      />
    )
  }

  const title = props.provider === "slack" ? "Slack policy" : "Discord policy"
  const summary =
    props.provider === "slack"
      ? "Workspace Slack policy, quotas, and ownership metadata."
      : "Workspace Discord policy, quotas, and ownership metadata."
  const policy = props.policy

  return (
    <WorkspacePolicyEditor
      title={title}
      panelKind={props.provider}
      summary={summary}
      installationIds={policy?.installation_ids}
      policy={policy?.policy}
      errorMessage={props.errorMessage}
      errorState={props.errorState}
      loading={props.loading}
      onSave={props.onSave}
      onRefresh={props.onRefresh}
    />
  )
}

export default IntegrationPolicyPanel
