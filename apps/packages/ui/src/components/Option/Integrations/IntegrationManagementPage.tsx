import React, { useMemo, useState } from "react"
import { Button, Card, Col, Row, Skeleton, Space, Tag, Typography, message } from "antd"
import { useQuery } from "@tanstack/react-query"
import { useNavigate } from "react-router-dom"
import {
  StatePanel,
  buildCapabilityState,
  classifyCapabilityError
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
const WORKSPACE_TELEGRAM_LINKED_ACTORS_PATH =
  "/api/v1/integrations/workspace/telegram/linked-actors"

const errorStatus = (error: unknown): number | undefined => {
  if (!error || typeof error !== "object") {
    return undefined
  }

  const status = (error as { status?: unknown; response?: { status?: unknown } }).status ??
    (error as { response?: { status?: unknown } }).response?.status

  return typeof status === "number" ? status : undefined
}

const errorMessage = (error: unknown, fallback: string): string => {
  if (error instanceof Error) {
    return error.message
  }

  if (error && typeof error === "object" && "message" in error) {
    const messageValue = (error as { message?: unknown }).message
    if (typeof messageValue === "string" && messageValue.trim()) {
      return messageValue
    }
  }

  return typeof error === "string" && error.trim() ? error : fallback
}

const isUnsupportedOverviewError = (scope: IntegrationScope, error: unknown): boolean => {
  if (scope !== "personal" || !error || typeof error !== "object") {
    return false
  }
  const maybeError = error as {
    status?: number
    message?: string
  }
  if (maybeError.status === 404) {
    return true
  }
  return typeof maybeError.message === "string" && maybeError.message.includes("/api/v1/integrations/personal")
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
    retry: false
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
  const slackPolicyError =
    slackPolicyQuery.isError && slackPolicyQuery.error instanceof Error
      ? slackPolicyQuery.error.message
      : slackPolicyQuery.isError
        ? "Slack policy could not be loaded."
        : null
  const discordPolicyError =
    discordPolicyQuery.isError && discordPolicyQuery.error instanceof Error
      ? discordPolicyQuery.error.message
      : discordPolicyQuery.isError
        ? "Discord policy could not be loaded."
        : null
  const telegramBotError =
    telegramBotQuery.isError && telegramBotQuery.error instanceof Error
      ? telegramBotQuery.error.message
      : telegramBotQuery.isError
        ? "Telegram bot settings could not be loaded."
        : null
  const personalIntegrationsUnsupported =
    scope === "personal" &&
    (personalIntegrationsSupported === false || isUnsupportedOverviewError(scope, overviewQuery.error))
  const personalIntegrationsCheckingSupport =
    scope === "personal" && (connectionConfigLoading || personalIntegrationsSupported === null)
  const overviewPath = scope === "workspace" ? WORKSPACE_INTEGRATIONS_PATH : PERSONAL_INTEGRATIONS_PATH
  const featureName = isWorkspace ? "Workspace integrations" : "Personal integrations"
  const capabilityName = isWorkspace ? "workspace integrations" : "personal integrations"
  const serverUrl = connectionConfig?.serverUrl?.trim()
  const personalUnsupportedState = buildCapabilityState({
    kind: "unavailable",
    featureName: "Personal integrations",
    capabilityName: "personal integrations",
    method: "GET",
    endpoint: PERSONAL_INTEGRATIONS_PATH,
    serverUrl,
    primaryAction: {
      label: "Check server setup",
      onClick: () => {
        navigate("/settings/health")
      }
    }
  })
  const overviewErrorState =
    overviewQuery.isError && !overviewQuery.data && !personalIntegrationsUnsupported
      ? buildCapabilityState({
          kind: classifyCapabilityError(overviewQuery.error),
          featureName,
          capabilityName,
          method: "GET",
          endpoint: overviewPath,
          status: errorStatus(overviewQuery.error),
          rawMessage: errorMessage(
            overviewQuery.error,
            "The integrations overview could not be loaded."
          ),
          primaryAction: {
            label: "Try again",
            onClick: () => {
              void overviewQuery.refetch()
            }
          }
      })
      : null
  const telegramActorsErrorState =
    telegramActorsQuery.isError
      ? buildCapabilityState({
          kind: "degraded",
          featureName: "Workspace integrations",
          method: "GET",
          endpoint: WORKSPACE_TELEGRAM_LINKED_ACTORS_PATH,
          status: errorStatus(telegramActorsQuery.error),
          rawMessage: errorMessage(
            telegramActorsQuery.error,
            "Telegram linked actors could not be loaded."
          ),
          primaryAction: {
            label: "Refresh Telegram actors",
            onClick: () => {
              void telegramActorsQuery.refetch()
            }
          }
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
                {isWorkspace ? "Workspace integrations" : "Personal integrations"}
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
        <StatePanel
          state={personalUnsupportedState.state}
          title={personalUnsupportedState.title}
          message={personalUnsupportedState.message}
          diagnostics={personalUnsupportedState.diagnostics}
          primaryAction={personalUnsupportedState.primaryAction}
        />
      ) : null}
      {overviewErrorState ? (
        <StatePanel
          state={overviewErrorState.state}
          title={overviewErrorState.title}
          message={overviewErrorState.message}
          diagnostics={overviewErrorState.diagnostics}
          primaryAction={overviewErrorState.primaryAction}
          role="alert"
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
          {telegramActorsErrorState ? (
            <StatePanel
              state={telegramActorsErrorState.state}
              title={telegramActorsErrorState.title}
              message={telegramActorsErrorState.message}
              diagnostics={telegramActorsErrorState.diagnostics}
              primaryAction={telegramActorsErrorState.primaryAction}
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
                errorMessage={slackPolicyError ? "Unable to load Slack policy" : undefined}
                loading={slackPolicyQuery.isLoading}
                onSave={updateWorkspaceSlackPolicy}
                onRefresh={() => void slackPolicyQuery.refetch()}
              />
            </Col>
            <Col xs={24} lg={12}>
              <IntegrationPolicyPanel
                provider="discord"
                policy={discordPolicyQuery.data}
                errorMessage={discordPolicyError ? "Unable to load Discord policy" : undefined}
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
            errorMessage={telegramBotError ? "Unable to load Telegram bot settings" : undefined}
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
