import React, { useMemo, useState } from "react"
import { Button, Card, Col, Row, Skeleton, Space, Tag, Typography, message } from "antd"
import { useQuery } from "@tanstack/react-query"
import {
  RecoveryCallout,
  buildCapabilityState,
  getCapabilityErrorStatus
} from "@/components/ui/state"
import { useCanonicalConnectionConfig } from "@/hooks/useCanonicalConnectionConfig"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import {
  connectPersonalIntegration,
  deletePersonalIntegration,
  createWorkspaceTelegramPairingCode,
  getWorkspaceDiscordPolicy,
  getWorkspaceSlackPolicy,
  getWorkspaceTelegramBot,
  listPersonalIntegrations,
  listWorkspaceIntegrations,
  listWorkspaceTelegramLinkedActors,
  revokeWorkspaceTelegramLinkedActor,
  updatePersonalIntegration,
  updateWorkspaceDiscordPolicy,
  updateWorkspaceSlackPolicy,
  updateWorkspaceTelegramBot,
  type IntegrationConnection,
  type IntegrationProvider,
  type PersonalIntegrationProvider,
  type IntegrationScope
} from "@/services/integrations-control-plane"
import { IntegrationConnectionDrawer } from "./IntegrationConnectionDrawer"
import { IntegrationPolicyPanel } from "./IntegrationPolicyPanel"
import { IntegrationProviderCard } from "./IntegrationProviderCard"

type IntegrationManagementPageProps = {
  scope: IntegrationScope
}

const PERSONAL_PROVIDERS: IntegrationProvider[] = ["slack", "discord"]
const WORKSPACE_PROVIDERS: IntegrationProvider[] = ["slack", "discord", "telegram"]

const providerLabel: Record<IntegrationProvider, string> = {
  slack: "Slack",
  discord: "Discord",
  telegram: "Telegram"
}

const sortConnections = (connections: IntegrationConnection[]): IntegrationConnection[] =>
  [...connections].sort((left, right) => left.display_name.localeCompare(right.display_name))

const isPersonalProvider = (provider: IntegrationProvider): provider is PersonalIntegrationProvider =>
  provider === "slack" || provider === "discord"

const PERSONAL_INTEGRATIONS_PATH = "/api/v1/integrations/personal"
const WORKSPACE_INTEGRATIONS_PATH = "/api/v1/integrations/workspace"
const WORKSPACE_SLACK_POLICY_PATH = "/api/v1/integrations/workspace/slack/policy"
const WORKSPACE_DISCORD_POLICY_PATH = "/api/v1/integrations/workspace/discord/policy"
const WORKSPACE_TELEGRAM_BOT_PATH = "/api/v1/integrations/workspace/telegram/bot"
const WORKSPACE_TELEGRAM_LINKED_ACTORS_PATH =
  "/api/v1/integrations/workspace/telegram/linked-actors"

const isUnsupportedOverviewError = (scope: IntegrationScope, error: unknown): boolean => {
  if (scope !== "personal" || !error || typeof error !== "object") {
    return false
  }
  const maybeError = error as {
    status?: number
    message?: string
  }
  const status = getCapabilityErrorStatus(error)
  if (status === 404 || status === 405 || status === 422) {
    return true
  }
  return (
    typeof maybeError.message === "string" &&
    maybeError.message.includes("/api/v1/integrations/personal") &&
    /\b(404|405|422|not found|unsupported|not expose)/i.test(maybeError.message)
  )
}

export const buildIntegrationQueryKey = (
  scope: IntegrationScope,
  orgId: number | null | undefined,
  resource: "overview" | "slack-policy" | "discord-policy" | "telegram-bot" | "telegram-linked-actors"
) => {
  if (scope === "workspace") {
    return ["integrations", scope, orgId ?? "unscoped", resource] as const
  }

  return ["integrations", scope, resource] as const
}

export const IntegrationManagementPage: React.FC<IntegrationManagementPageProps> = ({ scope }) => {
  const navigate = useNavigate()
  const { t } = useTranslation(["integrations", "common"])
  const { config: connectionConfig, loading: connectionConfigLoading } = useCanonicalConnectionConfig()
  const [activeOrgId, setActiveOrgId] = useState<number | null>(connectionConfig?.orgId ?? null)
  const [selectedConnection, setSelectedConnection] = useState<IntegrationConnection | null>(null)
  const [activePersonalActionKey, setActivePersonalActionKey] = useState<string | null>(null)
  const [personalIntegrationsSupported, setPersonalIntegrationsSupported] = useState<boolean | null>(
    scope === "workspace" ? true : null
  )

  React.useEffect(() => {
    setActiveOrgId(typeof connectionConfig?.orgId === "number" ? connectionConfig.orgId : null)
  }, [connectionConfig?.orgId])

  React.useEffect(() => {
    if (scope !== "workspace" || typeof window === "undefined") {
      return
    }

    let cancelled = false

    const syncActiveOrgId = async () => {
      try {
        const refreshedConfig = await tldwClient.getConfig()
        if (cancelled) {
          return
        }
        setActiveOrgId(typeof refreshedConfig?.orgId === "number" ? refreshedConfig.orgId : null)
      } catch {
        if (!cancelled) {
          setActiveOrgId(typeof connectionConfig?.orgId === "number" ? connectionConfig.orgId : null)
        }
      }
    }

    const handleConfigUpdated = () => {
      void syncActiveOrgId()
    }

    window.addEventListener("tldw:config-updated", handleConfigUpdated)

    return () => {
      cancelled = true
      window.removeEventListener("tldw:config-updated", handleConfigUpdated)
    }
  }, [connectionConfig?.orgId, scope])

  React.useEffect(() => {
    if (scope !== "personal") {
      setPersonalIntegrationsSupported(true)
      return
    }

    let cancelled = false

    const checkPersonalIntegrationsSupport = async () => {
      if (connectionConfigLoading) {
        return
      }

      const serverUrl = connectionConfig?.serverUrl?.trim()
      if (!serverUrl) {
        if (!cancelled) {
          setPersonalIntegrationsSupported(true)
        }
        return
      }

      setPersonalIntegrationsSupported(null)

      try {
        const response = await fetch(`${serverUrl}/openapi.json`)
        if (!response.ok) {
          if (!cancelled) {
            setPersonalIntegrationsSupported(true)
          }
          return
        }

        const spec = await response.json()
        const paths =
          spec && typeof spec === "object" && spec.paths && typeof spec.paths === "object"
            ? (spec.paths as Record<string, unknown>)
            : null

        if (!cancelled) {
          setPersonalIntegrationsSupported(Boolean(paths && PERSONAL_INTEGRATIONS_PATH in paths))
        }
      } catch {
        if (!cancelled) {
          setPersonalIntegrationsSupported(true)
        }
      }
    }

    void checkPersonalIntegrationsSupport()

    return () => {
      cancelled = true
    }
  }, [connectionConfig?.serverUrl, connectionConfigLoading, scope])

  const overviewQuery = useQuery({
    queryKey: buildIntegrationQueryKey(scope, activeOrgId, "overview"),
    queryFn: scope === "workspace" ? listWorkspaceIntegrations : listPersonalIntegrations,
    enabled: scope === "workspace" || personalIntegrationsSupported === true,
    retry: (failureCount, error) => {
      const status = getCapabilityErrorStatus(error)
      if (status === 401 || status === 403) {
        return false
      }
      return !isUnsupportedOverviewError(scope, error) && failureCount < 3
    }
  })

  const slackPolicyQuery = useQuery({
    queryKey: buildIntegrationQueryKey("workspace", activeOrgId, "slack-policy"),
    queryFn: getWorkspaceSlackPolicy,
    enabled: scope === "workspace"
  })

  const discordPolicyQuery = useQuery({
    queryKey: buildIntegrationQueryKey("workspace", activeOrgId, "discord-policy"),
    queryFn: getWorkspaceDiscordPolicy,
    enabled: scope === "workspace"
  })

  const telegramBotQuery = useQuery({
    queryKey: buildIntegrationQueryKey("workspace", activeOrgId, "telegram-bot"),
    queryFn: getWorkspaceTelegramBot,
    enabled: scope === "workspace"
  })

  const telegramActorsQuery = useQuery({
    queryKey: buildIntegrationQueryKey("workspace", activeOrgId, "telegram-linked-actors"),
    queryFn: listWorkspaceTelegramLinkedActors,
    enabled: scope === "workspace"
  })

  const connectionsByProvider = useMemo(() => {
    const supportedProviders = scope === "workspace" ? WORKSPACE_PROVIDERS : PERSONAL_PROVIDERS
    const items = overviewQuery.data?.items ?? []
    return supportedProviders.map((provider) => ({
      provider,
      connections: sortConnections(
        items.filter(
          (item) => item.provider === provider && item.scope === scope
        )
      )
    }))
  }, [overviewQuery.data?.items, scope])

  const refreshAll = async () => {
    await Promise.all([
      overviewQuery.refetch(),
      scope === "workspace" ? slackPolicyQuery.refetch() : Promise.resolve(),
      scope === "workspace" ? discordPolicyQuery.refetch() : Promise.resolve(),
      scope === "workspace" ? telegramBotQuery.refetch() : Promise.resolve(),
      scope === "workspace" ? telegramActorsQuery.refetch() : Promise.resolve()
    ])
  }

  const isWorkspace = scope === "workspace"
  const personalIntegrationsUnsupported =
    scope === "personal" &&
    (personalIntegrationsSupported === false || isUnsupportedOverviewError(scope, overviewQuery.error))
  const personalIntegrationsCheckingSupport =
    scope === "personal" && (connectionConfigLoading || personalIntegrationsSupported === null)
  const overviewEndpoint = scope === "workspace" ? WORKSPACE_INTEGRATIONS_PATH : PERSONAL_INTEGRATIONS_PATH
  const overviewCapabilityName = scope === "workspace"
    ? "workspace integration management"
    : "personal integration management"
  const personalUnsupportedState = personalIntegrationsUnsupported
    ? buildCapabilityState({
        featureName: "Personal integrations",
        capabilityName: "personal integration management",
        endpoint: PERSONAL_INTEGRATIONS_PATH,
        method: "GET",
        serverUrl: connectionConfig?.serverUrl,
        reason: "unsupported",
        title: "Personal integrations are unavailable on this server"
      })
    : null
  const overviewErrorState =
    overviewQuery.isError && !overviewQuery.data && !personalIntegrationsUnsupported
      ? buildCapabilityState({
          featureName: isWorkspace ? "Workspace integrations" : "Personal integrations",
          capabilityName: overviewCapabilityName,
          endpoint: overviewEndpoint,
          method: "GET",
          serverUrl: connectionConfig?.serverUrl,
          error: overviewQuery.error
        })
      : null
  const slackPolicyState = slackPolicyQuery.isError
    ? buildCapabilityState({
        featureName: "Slack policy",
        capabilityName: "workspace Slack policy",
        endpoint: WORKSPACE_SLACK_POLICY_PATH,
        method: "GET",
        serverUrl: connectionConfig?.serverUrl,
        error: slackPolicyQuery.error,
        title: "Unable to load Slack policy",
        message: "The workspace Slack policy could not be loaded."
      })
    : null
  const discordPolicyState = discordPolicyQuery.isError
    ? buildCapabilityState({
        featureName: "Discord policy",
        capabilityName: "workspace Discord policy",
        endpoint: WORKSPACE_DISCORD_POLICY_PATH,
        method: "GET",
        serverUrl: connectionConfig?.serverUrl,
        error: discordPolicyQuery.error,
        title: "Unable to load Discord policy",
        message: "The workspace Discord policy could not be loaded."
      })
    : null
  const telegramBotState = telegramBotQuery.isError
    ? buildCapabilityState({
        featureName: "Telegram bot settings",
        capabilityName: "workspace Telegram bot settings",
        endpoint: WORKSPACE_TELEGRAM_BOT_PATH,
        method: "GET",
        serverUrl: connectionConfig?.serverUrl,
        error: telegramBotQuery.error,
        title: "Unable to load Telegram bot settings",
        message: "The workspace Telegram bot settings could not be loaded."
      })
    : null
  const telegramActorsState = telegramActorsQuery.isError
    ? buildCapabilityState({
        featureName: "Telegram linked actors",
        capabilityName: "workspace Telegram linked actors",
        endpoint: WORKSPACE_TELEGRAM_LINKED_ACTORS_PATH,
        method: "GET",
        serverUrl: connectionConfig?.serverUrl,
        error: telegramActorsQuery.error,
        title: "Unable to load Telegram linked actors",
        message: "Telegram linked actors could not be loaded."
      })
    : null

  const handlePersonalAction = async (connection: IntegrationConnection, action: string) => {
    if (scope !== "personal" || !isPersonalProvider(connection.provider)) {
      return
    }

    const actionKey = `${connection.id}:${action}`
    setActivePersonalActionKey(actionKey)

    try {
      if (action === "connect" || action === "reconnect") {
        const response = await connectPersonalIntegration(connection.provider)
        if (typeof window !== "undefined") {
          window.open(response.auth_url, "_blank", "noopener,noreferrer")
        }
        message.success(`${providerLabel[connection.provider]} authorization opened`)
      } else if (action === "enable" || action === "disable") {
        const updated = await updatePersonalIntegration(connection.provider, connection.id, {
          enabled: action === "enable"
        })
        setSelectedConnection(updated)
        message.success(`${providerLabel[connection.provider]} ${action}d`)
      } else if (action === "remove") {
        await deletePersonalIntegration(connection.provider, connection.id)
        setSelectedConnection(null)
        message.success(`${providerLabel[connection.provider]} removed`)
      } else {
        return
      }

      await overviewQuery.refetch()
    } catch (error: any) {
      message.error(error?.message || `Unable to ${action} ${providerLabel[connection.provider]}`)
    } finally {
      setActivePersonalActionKey(null)
    }
  }

  return (
    <div className="mx-auto flex w-full max-w-6xl flex-col gap-6 p-6">
      <Card>
        <div style={{ display: "flex", flexDirection: "column", gap: 8, width: "100%" }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 16 }}>
            <div>
              <Typography.Title level={2} style={{ marginBottom: 0 }}>
                {featureName}
              </Typography.Title>
              <Typography.Paragraph type="secondary" style={{ marginBottom: 0 }}>
                {isWorkspace
                  ? "Manage workspace policies, installation inventory, and Telegram bot settings."
                  : "Review your Slack and Discord connections from one shared surface."}
              </Typography.Paragraph>
            </div>
            <Button onClick={() => void refreshAll()}>Refresh all</Button>
          </div>
          <Space wrap>
            {connectionsByProvider.map((group) => (
              <Tag key={group.provider} color={group.connections.length > 0 ? "green" : "default"}>
                {providerLabel[group.provider]}: {group.connections.length}
              </Tag>
            ))}
          </Space>
        </div>
      </Card>

      {((overviewQuery.isLoading && !overviewQuery.data) || personalIntegrationsCheckingSupport) ? (
        <Skeleton active paragraph={{ rows: 6 }} />
      ) : null}
      {personalIntegrationsUnsupported ? (
        <RecoveryCallout
          state={personalUnsupportedState?.state ?? "unavailable"}
          title={personalUnsupportedState?.title ?? PERSONAL_INTEGRATIONS_UNSUPPORTED_TITLE}
          message={
            personalUnsupportedState?.message ??
            PERSONAL_INTEGRATIONS_UNSUPPORTED_DESCRIPTION
          }
          diagnostics={personalUnsupportedState?.diagnostics}
          primaryAction={{
            label: "Refresh all",
            onClick: () => void refreshAll()
          }}
        />
      ) : null}
      {overviewQuery.isError && !overviewQuery.data && !personalIntegrationsUnsupported ? (
        <RecoveryCallout
          state={overviewErrorState?.state ?? "error"}
          title={overviewErrorState?.title ?? "Unable to load integrations"}
          message={
            overviewErrorState?.message ??
            "The integrations overview could not be loaded."
          }
          diagnostics={overviewErrorState?.diagnostics}
          primaryAction={{
            label: "Try again",
            onClick: () => void overviewQuery.refetch()
          }}
        />
      ) : null}

      {!personalIntegrationsUnsupported ? (
        <Row gutter={[16, 16]}>
          {connectionsByProvider.map((group) => (
            <Col key={group.provider} xs={24} lg={8}>
              <IntegrationProviderCard
                title={providerLabel[group.provider]}
                provider={group.provider}
                scope={scope}
                connections={group.connections}
                onInspect={(connection) => setSelectedConnection(connection)}
              />
            </Col>
          ))}
        </Row>
      ) : null}

      {isWorkspace ? (
        <>
          {telegramActorsState ? (
            <RecoveryCallout
              state={telegramActorsState.state}
              title={telegramActorsState.title}
              message={telegramActorsState.message}
              diagnostics={telegramActorsState.diagnostics}
              primaryAction={{
                label: "Try again",
                onClick: () => void telegramActorsQuery.refetch()
              }}
            />
          ) : null}

          <Typography.Title level={4} style={{ marginBottom: 0 }}>
            Workspace policy
          </Typography.Title>
          <Row gutter={[16, 16]}>
            <Col xs={24} lg={12}>
              <IntegrationPolicyPanel
                provider="slack"
                policy={slackPolicyQuery.data}
                errorState={slackPolicyState ?? undefined}
                loading={slackPolicyQuery.isLoading}
                onSave={updateWorkspaceSlackPolicy}
                onRefresh={() => void slackPolicyQuery.refetch()}
              />
            </Col>
            <Col xs={24} lg={12}>
              <IntegrationPolicyPanel
                provider="discord"
                policy={discordPolicyQuery.data}
                errorState={discordPolicyState ?? undefined}
                loading={discordPolicyQuery.isLoading}
                onSave={updateWorkspaceDiscordPolicy}
                onRefresh={() => void discordPolicyQuery.refetch()}
              />
            </Col>
          </Row>

          <Typography.Title level={4} style={{ marginBottom: 0 }}>
            Telegram workspace bot
          </Typography.Title>
          <IntegrationPolicyPanel
            provider="telegram"
            bot={telegramBotQuery.data}
            linkedActors={telegramActorsQuery.data?.items ?? []}
            errorState={telegramBotState ?? undefined}
            loading={telegramBotQuery.isLoading || telegramActorsQuery.isLoading}
            onSave={updateWorkspaceTelegramBot}
            onGeneratePairingCode={createWorkspaceTelegramPairingCode}
            onRevokeActor={revokeWorkspaceTelegramLinkedActor}
            onRefresh={() => {
              void telegramBotQuery.refetch()
              void telegramActorsQuery.refetch()
            }}
          />
        </>
      ) : null}

      <IntegrationConnectionDrawer
        open={selectedConnection !== null}
        connection={selectedConnection}
        activeActionKey={activePersonalActionKey}
        onClose={() => setSelectedConnection(null)}
        onRunAction={scope === "personal" ? handlePersonalAction : undefined}
      />
    </div>
  )
}

export default IntegrationManagementPage
