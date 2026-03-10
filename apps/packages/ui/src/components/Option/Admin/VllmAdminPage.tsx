import React from "react"
import {
  Alert,
  Button,
  Card,
  Input,
  InputNumber,
  List,
  Select,
  Space,
  Switch,
  Tag,
  Typography
} from "antd"
import { useTranslation } from "react-i18next"
import {
  tldwClient,
  type VllmExecutionMode,
  type VllmInstanceCreateRequest,
  type VllmInstanceJobResponse,
  type VllmInstanceRecord
} from "@/services/tldw/TldwApiClient"
import { PageShell } from "@/components/Common/PageShell"
import { StatusBanner } from "./StatusBanner"
import { CollapsibleSection } from "./CollapsibleSection"
import {
  deriveAdminGuardFromError,
  sanitizeAdminErrorMessage
} from "./admin-error-utils"

const { Title, Text } = Typography

const CAPABILITY_KEYS = ["chat", "embeddings", "vision", "audio", "multimodal"] as const

type CapabilityKey = (typeof CAPABILITY_KEYS)[number]

type CreateDraft = {
  name: string
  executionMode: VllmExecutionMode
  model: string
  servedModelName: string
  localHost: string
  servicePort: number
  sshHost: string
  sshPort: number
  sshUser: string
  apiKey: string
  capabilities: Record<CapabilityKey, boolean>
}

const INITIAL_DRAFT: CreateDraft = {
  name: "",
  executionMode: "local",
  model: "",
  servedModelName: "",
  localHost: "127.0.0.1",
  servicePort: 8000,
  sshHost: "",
  sshPort: 22,
  sshUser: "",
  apiKey: "",
  capabilities: {
    chat: true,
    embeddings: false,
    vision: false,
    audio: false,
    multimodal: false
  }
}

const resolveModelName = (instance: VllmInstanceRecord): string | null => {
  const model =
    instance.launch_spec?.served_model_name ??
    instance.launch_spec?.model ??
    instance.executor_handle?.model
  if (typeof model === "string" && model.trim().length > 0) {
    return model.trim()
  }
  return null
}

const resolveEndpoint = (instance: VllmInstanceRecord): string | null => {
  if (typeof instance.last_known_base_url === "string" && instance.last_known_base_url.trim()) {
    return instance.last_known_base_url.trim()
  }
  const host =
    instance.launch_spec?.host ??
    instance.transport_config?.base_url ??
    instance.transport_config?.host
  const port =
    instance.launch_spec?.port ??
    instance.transport_config?.service_port ??
    instance.transport_config?.port
  if (typeof host === "string" && host.trim()) {
    const normalizedHost = host.startsWith("http://") || host.startsWith("https://")
      ? host.trim()
      : `http://${host.trim()}`
    if (port != null && !normalizedHost.match(/:\d+$/)) {
      return `${normalizedHost}:${port}/v1`
    }
    return normalizedHost.endsWith("/v1") ? normalizedHost : `${normalizedHost}/v1`
  }
  return null
}

const buildCreatePayload = (draft: CreateDraft): VllmInstanceCreateRequest => {
  const launch_spec: Record<string, any> = {
    model: draft.model.trim(),
    port: draft.servicePort
  }
  if (draft.servedModelName.trim()) {
    launch_spec.served_model_name = draft.servedModelName.trim()
  }
  if (draft.apiKey.trim()) {
    launch_spec.api_key = draft.apiKey.trim()
  }

  const transport_config: Record<string, any> = {}
  if (draft.executionMode === "local") {
    launch_spec.host = draft.localHost.trim() || "127.0.0.1"
  } else if (draft.executionMode === "ssh") {
    transport_config.host = draft.sshHost.trim()
    transport_config.port = draft.sshPort
    if (draft.sshUser.trim()) {
      transport_config.user = draft.sshUser.trim()
    }
  }

  return {
    name: draft.name.trim(),
    execution_mode: draft.executionMode,
    transport_config,
    launch_spec,
    routing_policy: {},
    declared_capabilities: { ...draft.capabilities }
  }
}

export const VllmAdminPage: React.FC = () => {
  const { t } = useTranslation(["option", "settings", "common"])
  const [instances, setInstances] = React.useState<VllmInstanceRecord[]>([])
  const [defaultInstanceId, setDefaultInstanceId] = React.useState<string | null>(null)
  const [loading, setLoading] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)
  const [actionMessage, setActionMessage] = React.useState<string | null>(null)
  const [actionLoadingById, setActionLoadingById] = React.useState<Record<string, string | null>>({})
  const [createLoading, setCreateLoading] = React.useState(false)
  const [draft, setDraft] = React.useState<CreateDraft>(INITIAL_DRAFT)
  const [adminGuard, setAdminGuard] = React.useState<"forbidden" | "notFound" | null>(null)

  const markAdminGuardFromError = (err: unknown) => {
    const guardState = deriveAdminGuardFromError(err)
    if (guardState) {
      setAdminGuard(guardState)
    }
  }

  const loadInstances = React.useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      const response = await tldwClient.listVllmInstances()
      setInstances(Array.isArray(response.instances) ? response.instances : [])
      setDefaultInstanceId(response.default_instance_id || null)
    } catch (err) {
      setError(
        sanitizeAdminErrorMessage(err, "Failed to load managed vLLM instances.")
      )
      markAdminGuardFromError(err)
    } finally {
      setLoading(false)
    }
  }, [])

  React.useEffect(() => {
    void loadInstances()
  }, [loadInstances])

  const setActionState = (instanceId: string, action: string | null) => {
    setActionLoadingById((current) => ({
      ...current,
      [instanceId]: action
    }))
  }

  const handleLifecycleAction = async (
    instance: VllmInstanceRecord,
    action: "start" | "stop" | "restart" | "probe"
  ) => {
    try {
      setActionState(instance.instance_id, action)
      setActionMessage(null)
      let response: VllmInstanceJobResponse
      if (action === "start") {
        response = await tldwClient.startVllmInstance(instance.instance_id)
      } else if (action === "stop") {
        response = await tldwClient.stopVllmInstance(instance.instance_id)
      } else if (action === "restart") {
        response = await tldwClient.restartVllmInstance(instance.instance_id)
      } else {
        response = await tldwClient.probeVllmInstance(instance.instance_id)
      }
      setActionMessage(
        `${instance.name}: queued ${response.requested_action} job #${response.job_id}.`
      )
      await loadInstances()
    } catch (err) {
      setError(
        sanitizeAdminErrorMessage(err, `Failed to ${action} ${instance.name}.`)
      )
      markAdminGuardFromError(err)
    } finally {
      setActionState(instance.instance_id, null)
    }
  }

  const handleSetDefault = async (instanceId: string | null) => {
    try {
      setActionMessage(null)
      const response = await tldwClient.setDefaultVllmInstance(instanceId)
      setDefaultInstanceId(response.default_instance_id || null)
      setActionMessage(
        response.default_instance_id
          ? `Default route set to ${response.default_instance_id}.`
          : "Default managed vLLM route cleared."
      )
      await loadInstances()
    } catch (err) {
      setError(
        sanitizeAdminErrorMessage(err, "Failed to update the default managed vLLM route.")
      )
      markAdminGuardFromError(err)
    }
  }

  const handleDelete = async (instance: VllmInstanceRecord) => {
    try {
      setActionState(instance.instance_id, "delete")
      setActionMessage(null)
      await tldwClient.deleteVllmInstance(instance.instance_id)
      setActionMessage(`${instance.name} deleted.`)
      await loadInstances()
    } catch (err) {
      setError(
        sanitizeAdminErrorMessage(err, `Failed to delete ${instance.name}.`)
      )
      markAdminGuardFromError(err)
    } finally {
      setActionState(instance.instance_id, null)
    }
  }

  const handleCreateInstance = async () => {
    if (!draft.name.trim() || !draft.model.trim()) {
      setError("Name and model are required.")
      return
    }
    if (draft.executionMode === "ssh" && !draft.sshHost.trim()) {
      setError("SSH host is required for SSH-managed instances.")
      return
    }
    try {
      setCreateLoading(true)
      setError(null)
      setActionMessage(null)
      await tldwClient.createVllmInstance(buildCreatePayload(draft))
      setDraft(INITIAL_DRAFT)
      setActionMessage("Managed vLLM instance created.")
      await loadInstances()
    } catch (err) {
      setError(
        sanitizeAdminErrorMessage(err, "Failed to create the managed vLLM instance.")
      )
      markAdminGuardFromError(err)
    } finally {
      setCreateLoading(false)
    }
  }

  const statusItemsForInstance = (instance: VllmInstanceRecord) => [
    { label: t("settings:admin.vllmModel", "Model"), value: resolveModelName(instance), code: true },
    { label: t("settings:admin.vllmEndpoint", "Endpoint"), value: resolveEndpoint(instance), code: true }
  ]

  return (
    <PageShell maxWidthClassName="max-w-6xl" className="py-6">
      <Space orientation="vertical" size="large" className="w-full">
        <div>
          <Title level={2} className="mb-1">
            {t("option:header.adminVllm", "vLLM Admin")}
          </Title>
          <Text type="secondary">
            {t(
              "settings:admin.vllmIntro",
              "Inspect managed instances, queue lifecycle jobs, and pick the default route used by request-scoped vLLM traffic."
            )}
          </Text>
        </div>

        {adminGuard && (
          <Alert
            type="warning"
            showIcon
            message={
              adminGuard === "forbidden"
                ? t("settings:admin.adminGuardForbiddenTitle", "Admin access required")
                : t("settings:admin.adminGuardNotFoundTitle", "Admin APIs not available")
            }
            description={
              adminGuard === "forbidden"
                ? t(
                    "settings:admin.adminGuardForbiddenBody",
                    "Sign in as an admin user on your tldw server to access these controls."
                  )
                : t(
                    "settings:admin.adminGuardNotFoundBody",
                    "This tldw server does not expose the admin endpoints."
                  )
            }
          />
        )}

        {error && (
          <Alert type="error" showIcon message={error} />
        )}
        {actionMessage && (
          <Alert type="success" showIcon message={actionMessage} />
        )}

        <StatusBanner
          state={instances.length > 0 ? "active" : "inactive"}
          loading={loading}
          error={null}
          items={[
            { label: t("settings:admin.vllmInstances", "Instances"), value: instances.length },
            { label: t("settings:admin.vllmDefaultRoute", "Default route"), value: defaultInstanceId }
          ]}
          onRefresh={() => {
            void loadInstances()
          }}
        />

        {!adminGuard && (
          <>
            <CollapsibleSection
              defaultOpen
              title={t("settings:admin.vllmCreateTitle", "Create Managed Instance")}
              description={t(
                "settings:admin.vllmCreateDesc",
                "Add a local or SSH-backed vLLM instance with structured launch settings."
              )}>
              <Space orientation="vertical" size="middle" className="w-full">
                <Space wrap size="middle">
                  <div className="min-w-[220px]">
                    <Text strong>{t("settings:admin.vllmName", "Name")}</Text>
                    <Input
                      value={draft.name}
                      onChange={(event) => setDraft((current) => ({ ...current, name: event.target.value }))}
                      placeholder={t("settings:admin.vllmNamePlaceholder", "vision-a100")}
                    />
                  </div>
                  <div className="min-w-[180px]">
                    <Text strong>{t("settings:admin.vllmExecutionMode", "Execution mode")}</Text>
                    <Select
                      className="w-full"
                      value={draft.executionMode}
                      options={[
                        { label: "local", value: "local" },
                        { label: "ssh", value: "ssh" }
                      ]}
                      onChange={(value: VllmExecutionMode) =>
                        setDraft((current) => ({ ...current, executionMode: value }))
                      }
                    />
                  </div>
                  <div className="min-w-[280px]">
                    <Text strong>{t("settings:admin.vllmModel", "Model")}</Text>
                    <Input
                      value={draft.model}
                      onChange={(event) => setDraft((current) => ({ ...current, model: event.target.value }))}
                      placeholder={t(
                        "settings:admin.vllmModelPlaceholder",
                        "Qwen/Qwen2.5-VL-7B-Instruct"
                      )}
                    />
                  </div>
                </Space>

                <Space wrap size="middle">
                  <div className="min-w-[280px]">
                    <Text strong>{t("settings:admin.vllmServedModelName", "Served model name")}</Text>
                    <Input
                      value={draft.servedModelName}
                      onChange={(event) =>
                        setDraft((current) => ({ ...current, servedModelName: event.target.value }))
                      }
                      placeholder={t("settings:admin.vllmServedModelPlaceholder", "Optional alias")}
                    />
                  </div>
                  <div className="min-w-[180px]">
                    <Text strong>{t("settings:admin.vllmPort", "Service port")}</Text>
                    <InputNumber
                      className="w-full"
                      min={1}
                      max={65535}
                      value={draft.servicePort}
                      onChange={(value) =>
                        setDraft((current) => ({ ...current, servicePort: Number(value || 8000) }))
                      }
                    />
                  </div>
                  <div className="min-w-[240px]">
                    <Text strong>{t("settings:admin.vllmApiKey", "API key")}</Text>
                    <Input
                      value={draft.apiKey}
                      onChange={(event) => setDraft((current) => ({ ...current, apiKey: event.target.value }))}
                      placeholder={t("settings:admin.vllmApiKeyPlaceholder", "Optional managed key")}
                    />
                  </div>
                </Space>

                {draft.executionMode === "local" ? (
                  <div className="min-w-[220px] max-w-[280px]">
                    <Text strong>{t("settings:admin.vllmHost", "Host")}</Text>
                    <Input
                      value={draft.localHost}
                      onChange={(event) => setDraft((current) => ({ ...current, localHost: event.target.value }))}
                      placeholder="127.0.0.1"
                    />
                  </div>
                ) : (
                  <Space wrap size="middle">
                    <div className="min-w-[220px]">
                      <Text strong>{t("settings:admin.vllmSshHost", "SSH host")}</Text>
                      <Input
                        value={draft.sshHost}
                        onChange={(event) => setDraft((current) => ({ ...current, sshHost: event.target.value }))}
                        placeholder="gpu-a100-01.internal"
                      />
                    </div>
                    <div className="min-w-[180px]">
                      <Text strong>{t("settings:admin.vllmSshPort", "SSH port")}</Text>
                      <InputNumber
                        className="w-full"
                        min={1}
                        max={65535}
                        value={draft.sshPort}
                        onChange={(value) =>
                          setDraft((current) => ({ ...current, sshPort: Number(value || 22) }))
                        }
                      />
                    </div>
                    <div className="min-w-[220px]">
                      <Text strong>{t("settings:admin.vllmSshUser", "SSH user")}</Text>
                      <Input
                        value={draft.sshUser}
                        onChange={(event) => setDraft((current) => ({ ...current, sshUser: event.target.value }))}
                        placeholder="ubuntu"
                      />
                    </div>
                  </Space>
                )}

                <Card size="small" title={t("settings:admin.vllmCapabilities", "Capabilities")}>
                  <Space wrap size="large">
                    {CAPABILITY_KEYS.map((capability) => (
                      <Space key={capability} align="center">
                        <Switch
                          checked={draft.capabilities[capability]}
                          onChange={(checked) =>
                            setDraft((current) => ({
                              ...current,
                              capabilities: {
                                ...current.capabilities,
                                [capability]: checked
                              }
                            }))
                          }
                        />
                        <Text>{capability}</Text>
                      </Space>
                    ))}
                  </Space>
                </Card>

                <Space>
                  <Button type="primary" loading={createLoading} onClick={handleCreateInstance}>
                    {t("settings:admin.vllmCreateInstance", "Create instance")}
                  </Button>
                  <Button onClick={() => setDraft(INITIAL_DRAFT)}>
                    {t("common:reset", "Reset")}
                  </Button>
                </Space>
              </Space>
            </CollapsibleSection>

            <Card
              title={t("settings:admin.vllmManagedInstances", "Managed Instances")}
              extra={
                defaultInstanceId ? (
                  <Button size="small" onClick={() => void handleSetDefault(null)}>
                    {t("settings:admin.vllmClearDefault", "Clear default")}
                  </Button>
                ) : null
              }>
              <List
                loading={loading}
                locale={{
                  emptyText: t(
                    "settings:admin.vllmNoInstances",
                    "No managed vLLM instances have been created yet."
                  )
                }}
                dataSource={instances}
                renderItem={(instance) => {
                  const activeAction = actionLoadingById[instance.instance_id]
                  const isDefault = defaultInstanceId === instance.instance_id
                  const effectiveModel = resolveModelName(instance)
                  return (
                    <List.Item key={instance.instance_id}>
                      <Card className="w-full" size="small">
                        <Space orientation="vertical" size="middle" className="w-full">
                          <div className="flex flex-wrap items-center gap-2">
                            <Title level={5} className="!mb-0">
                              {instance.name}
                            </Title>
                            <Tag>{instance.execution_mode}</Tag>
                            {isDefault && <Tag color="blue">default</Tag>}
                            {effectiveModel && <Tag color="purple">{effectiveModel}</Tag>}
                          </div>

                          <StatusBanner
                            state={instance.observed_state || "unknown"}
                            items={statusItemsForInstance(instance)}
                            onRefresh={() => {
                              void loadInstances()
                            }}
                          />

                          {instance.last_error ? (
                            <Alert type="warning" showIcon message={instance.last_error} />
                          ) : null}

                          <Space wrap>
                            <Button
                              type="primary"
                              loading={activeAction === "start"}
                              onClick={() => void handleLifecycleAction(instance, "start")}>
                              {t("settings:admin.vllmStart", "Start")}
                            </Button>
                            <Button
                              loading={activeAction === "stop"}
                              onClick={() => void handleLifecycleAction(instance, "stop")}>
                              {t("settings:admin.vllmStop", "Stop")}
                            </Button>
                            <Button
                              loading={activeAction === "restart"}
                              onClick={() => void handleLifecycleAction(instance, "restart")}>
                              {t("settings:admin.vllmRestart", "Restart")}
                            </Button>
                            <Button
                              loading={activeAction === "probe"}
                              onClick={() => void handleLifecycleAction(instance, "probe")}>
                              {t("settings:admin.vllmProbe", "Probe")}
                            </Button>
                            <Button
                              disabled={isDefault}
                              onClick={() => void handleSetDefault(instance.instance_id)}>
                              {t("settings:admin.vllmSetDefault", "Set default")}
                            </Button>
                            <Button
                              danger
                              loading={activeAction === "delete"}
                              onClick={() => void handleDelete(instance)}>
                              {t("settings:admin.vllmDelete", "Delete")}
                            </Button>
                          </Space>
                        </Space>
                      </Card>
                    </List.Item>
                  )
                }}
              />
            </Card>
          </>
        )}
      </Space>
    </PageShell>
  )
}

export default VllmAdminPage
