import React, { useState, useCallback } from "react"
import {
  Badge,
  Button,
  Card,
  Checkbox,
  Collapse,
  Drawer,
  Empty,
  Form,
  Input,
  InputNumber,
  Modal,
  Radio,
  Select,
  Skeleton,
  Space,
  Switch,
  Table,
  Tabs,
  Tag,
  Tooltip,
  Typography,
  message
} from "antd"
import {
  DeleteOutlined,
  EditOutlined,
  ExclamationCircleOutlined,
  PauseCircleOutlined,
  PlayCircleOutlined,
  PlusOutlined,
  ReloadOutlined,
  StopOutlined
} from "@ant-design/icons"
import type { ColumnsType } from "antd/es/table"
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { useTranslation } from "react-i18next"
import { useNavigate } from "react-router-dom"

import { Alert as DsAlert } from "@/components/ui/primitives"
import { useServerOnline } from "@/hooks/useServerOnline"
import { useConnectionUxState } from "@/hooks/useConnectionState"
import { useServerCapabilities } from "@/hooks/useServerCapabilities"
import { useCanonicalConnectionConfig } from "@/hooks/useCanonicalConnectionConfig"
import {
  listRules,
  createRule,
  updateRule,
  deleteRule,
  deactivateRule,
  listAlerts,
  markAlertsRead,
  getUnreadCount,
  listGovernancePolicies,
  createGovernancePolicy,
  deleteGovernancePolicy,
  getCrisisResources,
  listRelationships,
  createRelationship,
  acceptRelationship,
  suspendRelationship,
  reactivateRelationship,
  dissolveRelationship,
  listPolicies,
  createPolicy,
  updatePolicy,
  deletePolicy,
  getAuditLog,
  type SelfMonitoringRule,
  type SelfMonitoringRuleCreate,
  type SelfMonitoringRuleUpdate,
  type SelfMonitoringAlert,
  type GovernancePolicy,
  type GovernancePolicyCreate,
  type GuardianRelationship,
  type GuardianRelationshipCreate,
  type SupervisedPolicy,
  type SupervisedPolicyCreate,
  type SupervisedPolicyUpdate,
  type AuditLogEntry,
  type CrisisResource
} from "@/services/guardian"

const { Title, Text, Paragraph } = Typography
const { TextArea } = Input

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const ACTION_COLORS: Record<string, string> = {
  notify: "blue",
  warn: "gold",
  redact: "purple",
  block: "red"
}

const SEVERITY_COLORS: Record<string, string> = {
  info: "default",
  warning: "gold",
  critical: "red"
}

const STATUS_COLORS: Record<string, string> = {
  pending: "orange",
  active: "green",
  suspended: "gold",
  dissolved: "default"
}

const normalizeRelationshipStatus = (status: string): string => {
  if (status === "pending_consent") return "pending"
  return status
}

const UNSUPPORTED_GUARDIAN_STATUS_CODES = new Set([404, 405, 410, 501])
const SELF_MONITORING_REQUIRED_PATHS = [
  "/api/v1/self-monitoring/rules",
  "/api/v1/self-monitoring/alerts",
  "/api/v1/self-monitoring/alerts/unread-count",
  "/api/v1/self-monitoring/governance-policies"
] as const

const getGuardianErrorStatus = (error: unknown): number | null => {
  const status = (error as { status?: unknown } | null)?.status
  if (typeof status === "number" && Number.isFinite(status)) return status
  const message =
    error instanceof Error ? error.message : String(error || "")
  const matched = message.match(/\b([45]\d{2})\b/)
  if (!matched) return null
  const parsed = Number.parseInt(matched[1], 10)
  return Number.isNaN(parsed) ? null : parsed
}

const isGuardianEndpointUnsupported = (error: unknown): boolean => {
  const status = getGuardianErrorStatus(error)
  if (status == null) return false
  return UNSUPPORTED_GUARDIAN_STATUS_CODES.has(status)
}

// ---------------------------------------------------------------------------
// Self-Monitoring Tab
// ---------------------------------------------------------------------------

function SelfMonitoringTab({ online }: { online: boolean }) {
  const { t } = useTranslation("settings")
  const qc = useQueryClient()
  const { config: connectionConfig, loading: connectionConfigLoading } =
    useCanonicalConnectionConfig()
  const [ruleDrawerOpen, setRuleDrawerOpen] = useState(false)
  const [governanceModalOpen, setGovernanceModalOpen] = useState(false)
  const [editingRule, setEditingRule] = useState<SelfMonitoringRule | null>(null)
  const [selectedAlertIds, setSelectedAlertIds] = useState<string[]>([])
  const [apiUnavailable, setApiUnavailable] = useState(false)
  const [selfMonitoringSupported, setSelfMonitoringSupported] = useState<
    boolean | null
  >(online ? null : true)
  const [form] = Form.useForm()
  const [governanceForm] = Form.useForm()
  const selfMonitoringQueriesEnabled =
    online && selfMonitoringSupported === true && !apiUnavailable

  React.useEffect(() => {
    if (!online) {
      setSelfMonitoringSupported(true)
      return
    }
    if (connectionConfigLoading) return

    const serverUrl = connectionConfig?.serverUrl?.trim()
    if (!serverUrl) {
      setSelfMonitoringSupported(true)
      return
    }

    let cancelled = false

    const probeSelfMonitoringSupport = async () => {
      try {
        const response = await fetch(`${serverUrl}/openapi.json`)
        if (!response.ok) {
          if (!cancelled) {
            setSelfMonitoringSupported(true)
          }
          return
        }
        const spec = await response.json()
        const paths =
          spec && typeof spec === "object" && spec.paths && typeof spec.paths === "object"
            ? (spec.paths as Record<string, unknown>)
            : null
        const supported = SELF_MONITORING_REQUIRED_PATHS.every(
          (path) => Boolean(paths && path in paths)
        )
        if (!cancelled) {
          setSelfMonitoringSupported(supported)
        }
      } catch {
        if (!cancelled) {
          setSelfMonitoringSupported(true)
        }
      }
    }

    void probeSelfMonitoringSupport()

    return () => {
      cancelled = true
    }
  }, [
    connectionConfig?.serverUrl,
    connectionConfigLoading,
    online
  ])

  const rulesQuery = useQuery({
    queryKey: ["guardian", "rules"],
    queryFn: () => listRules(),
    enabled: selfMonitoringQueriesEnabled,
    retry: false
  })

  const alertsQuery = useQuery({
    queryKey: ["guardian", "alerts"],
    queryFn: () => listAlerts({ limit: 50 }),
    enabled: selfMonitoringQueriesEnabled,
    retry: false
  })

  const unreadQuery = useQuery({
    queryKey: ["guardian", "unread"],
    queryFn: getUnreadCount,
    refetchInterval: 30_000,
    enabled: selfMonitoringQueriesEnabled,
    retry: false
  })

  const governanceQuery = useQuery({
    queryKey: ["guardian", "governance"],
    queryFn: () => listGovernancePolicies(),
    enabled: selfMonitoringQueriesEnabled,
    retry: false
  })

  React.useEffect(() => {
    if (apiUnavailable) return
    if (
      isGuardianEndpointUnsupported(rulesQuery.error) ||
      isGuardianEndpointUnsupported(alertsQuery.error) ||
      isGuardianEndpointUnsupported(unreadQuery.error) ||
      isGuardianEndpointUnsupported(governanceQuery.error)
    ) {
      setApiUnavailable(true)
    }
  }, [
    alertsQuery.error,
    apiUnavailable,
    governanceQuery.error,
    rulesQuery.error,
    unreadQuery.error
  ])

  const createMutation = useMutation({
    mutationFn: createRule,
    onSuccess: () => {
      message.success(t("guardian.rules.created", "Rule created"))
      qc.invalidateQueries({ queryKey: ["guardian", "rules"] })
      closeDrawer()
    },
    onError: () => message.error(t("guardian.rules.createFailed", "Failed to create rule"))
  })

  const updateMutation = useMutation({
    mutationFn: ({ id, body }: { id: string; body: SelfMonitoringRuleUpdate }) => updateRule(id, body),
    onSuccess: () => {
      message.success(t("guardian.rules.updated", "Rule updated"))
      qc.invalidateQueries({ queryKey: ["guardian", "rules"] })
      closeDrawer()
    },
    onError: () => message.error(t("guardian.rules.updateFailed", "Failed to update rule"))
  })

  const deleteMutation = useMutation({
    mutationFn: deleteRule,
    onSuccess: () => {
      message.success(t("guardian.rules.deleted", "Rule deleted"))
      qc.invalidateQueries({ queryKey: ["guardian", "rules"] })
    },
    onError: () => message.error(t("guardian.rules.deleteFailed", "Failed to delete rule"))
  })

  const deactivateMutation = useMutation({
    mutationFn: deactivateRule,
    onSuccess: () => {
      message.success(t("guardian.rules.deactivated", "Deactivation requested"))
      qc.invalidateQueries({ queryKey: ["guardian", "rules"] })
    },
    onError: () => message.error(t("guardian.rules.deactivateFailed", "Failed to deactivate rule"))
  })

  const markReadMutation = useMutation({
    mutationFn: markAlertsRead,
    onSuccess: () => {
      message.success(t("guardian.alerts.marked", "Alerts marked as read"))
      setSelectedAlertIds([])
      qc.invalidateQueries({ queryKey: ["guardian", "alerts"] })
      qc.invalidateQueries({ queryKey: ["guardian", "unread"] })
    },
    onError: () => message.error(t("guardian.alerts.markFailed", "Failed to mark alerts"))
  })

  const toggleEnabledMutation = useMutation({
    mutationFn: ({ id, enabled }: { id: string; enabled: boolean }) => updateRule(id, { enabled }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["guardian", "rules"] })
    },
    onError: () => message.error(t("guardian.rules.toggleFailed", "Failed to toggle rule"))
  })

  const createGovernanceMutation = useMutation({
    mutationFn: createGovernancePolicy,
    onSuccess: () => {
      message.success(t("guardian.governance.created", "Governance policy created"))
      setGovernanceModalOpen(false)
      governanceForm.resetFields()
      qc.invalidateQueries({ queryKey: ["guardian", "governance"] })
    },
    onError: () => message.error(t("guardian.governance.createFailed", "Failed to create governance policy"))
  })

  const deleteGovernanceMutation = useMutation({
    mutationFn: deleteGovernancePolicy,
    onSuccess: () => {
      message.success(t("guardian.governance.deleted", "Governance policy deleted"))
      qc.invalidateQueries({ queryKey: ["guardian", "governance"] })
    },
    onError: () => message.error(t("guardian.governance.deleteFailed", "Failed to delete governance policy"))
  })

  const openCreate = useCallback(() => {
    setEditingRule(null)
    form.resetFields()
    setRuleDrawerOpen(true)
  }, [form])

  const openEdit = useCallback(
    (rule: SelfMonitoringRule) => {
      setEditingRule(rule)
      form.setFieldsValue({
        ...rule,
        patterns: rule.patterns.join("\n"),
        except_patterns: (rule.except_patterns || []).join("\n")
      })
      setRuleDrawerOpen(true)
    },
    [form]
  )

  const closeDrawer = useCallback(() => {
    setRuleDrawerOpen(false)
    setEditingRule(null)
    form.resetFields()
  }, [form])

  const handleSubmit = useCallback(async () => {
    try {
      const values = await form.validateFields()
      const body: SelfMonitoringRuleCreate = {
        ...values,
        patterns: (values.patterns as string)
          .split("\n")
          .map((s: string) => s.trim())
          .filter(Boolean),
        except_patterns: values.except_patterns
          ? (values.except_patterns as string)
              .split("\n")
              .map((s: string) => s.trim())
              .filter(Boolean)
          : []
      }
      if (editingRule) {
        updateMutation.mutate({ id: editingRule.id, body })
      } else {
        createMutation.mutate(body)
      }
    } catch {
      // validation error
    }
  }, [form, editingRule, createMutation, updateMutation])

  const handleGovernanceSubmit = useCallback(async () => {
    try {
      const values = await governanceForm.validateFields()
      createGovernanceMutation.mutate(values as GovernancePolicyCreate)
    } catch {
      // validation error
    }
  }, [governanceForm, createGovernanceMutation])

  const confirmDelete = useCallback(
    (id: string) => {
      Modal.confirm({
        title: t("guardian.rules.deleteConfirm", "Delete rule?"),
        icon: <ExclamationCircleOutlined />,
        content: t("guardian.rules.deleteConfirmContent", "This action cannot be undone."),
        okText: t("guardian.common.delete", "Delete"),
        okType: "danger",
        onOk: () => deleteMutation.mutate(id)
      })
    },
    [deleteMutation, t]
  )

  const confirmDeactivate = useCallback(
    (id: string) => {
      Modal.confirm({
        title: t("guardian.rules.deactivateConfirm", "Deactivate rule?"),
        icon: <ExclamationCircleOutlined />,
        content: t(
          "guardian.rules.deactivateConfirmContent",
          "Deactivation may be subject to a cooldown period before taking effect."
        ),
        okText: t("guardian.common.deactivate", "Deactivate"),
        onOk: () => deactivateMutation.mutate(id)
      })
    },
    [deactivateMutation, t]
  )

  const confirmDeleteGovernance = useCallback(
    (id: string) => {
      Modal.confirm({
        title: t("guardian.governance.deleteConfirm", "Delete governance policy?"),
        icon: <ExclamationCircleOutlined />,
        content: t(
          "guardian.governance.deleteConfirmContent",
          "This action cannot be undone and may affect linked rules."
        ),
        okText: t("guardian.common.delete", "Delete"),
        okType: "danger",
        onOk: () => deleteGovernanceMutation.mutate(id)
      })
    },
    [deleteGovernanceMutation, t]
  )

  const ruleColumns: ColumnsType<SelfMonitoringRule> = [
    {
      title: t("guardian.rules.fields.name", "Name"),
      dataIndex: "name",
      key: "name",
      ellipsis: true,
      width: 180
    },
    {
      title: t("guardian.rules.fields.category", "Category"),
      dataIndex: "category",
      key: "category",
      width: 120,
      render: (cat: string) => <Tag>{cat}</Tag>
    },
    {
      title: t("guardian.rules.fields.patterns", "Patterns"),
      dataIndex: "patterns",
      key: "patterns",
      ellipsis: true,
      render: (patterns: string[]) => (
        <Text type="secondary" style={{ fontSize: 12 }}>
          {patterns.slice(0, 2).join(", ")}
          {patterns.length > 2 && ` +${patterns.length - 2} more`}
        </Text>
      )
    },
    {
      title: t("guardian.rules.fields.action", "Action"),
      dataIndex: "action",
      key: "action",
      width: 90,
      render: (action: string) => (
        <Tag color={ACTION_COLORS[action] ?? "default"}>{action}</Tag>
      )
    },
    {
      title: t("guardian.rules.fields.severity", "Severity"),
      dataIndex: "severity",
      key: "severity",
      width: 90,
      render: (sev: string) => (
        <Tag color={SEVERITY_COLORS[sev] ?? "default"}>{sev}</Tag>
      )
    },
    {
      title: t("guardian.rules.fields.enabled", "Enabled"),
      dataIndex: "enabled",
      key: "enabled",
      width: 80,
      render: (enabled: boolean, record) => (
        <Switch
          size="small"
          checked={enabled}
          onChange={(checked) =>
            toggleEnabledMutation.mutate({ id: record.id, enabled: checked })
          }
        />
      )
    },
    {
      title: t("guardian.common.actions", "Actions"),
      key: "actions",
      width: 130,
      render: (_, record) => (
        <Space size="small">
          <Tooltip title={t("guardian.common.edit", "Edit")}>
            <Button
              type="text"
              size="small"
              icon={<EditOutlined />}
              onClick={() => openEdit(record)}
            />
          </Tooltip>
          {record.enabled && record.can_disable && (
            <Tooltip title={t("guardian.common.deactivate", "Deactivate")}>
              <Button
                type="text"
                size="small"
                icon={<PauseCircleOutlined />}
                onClick={() => confirmDeactivate(record.id)}
              />
            </Tooltip>
          )}
          <Tooltip title={t("guardian.common.delete", "Delete")}>
            <Button
              type="text"
              size="small"
              danger
              icon={<DeleteOutlined />}
              onClick={() => confirmDelete(record.id)}
            />
          </Tooltip>
        </Space>
      )
    }
  ]

  const alertColumns: ColumnsType<SelfMonitoringAlert> = [
    {
      title: t("guardian.alerts.columns.rule", "Rule"),
      dataIndex: "rule_name",
      key: "rule_name",
      ellipsis: true,
      width: 160
    },
    {
      title: t("guardian.alerts.columns.severity", "Severity"),
      dataIndex: "severity",
      key: "severity",
      width: 90,
      render: (sev: string) => (
        <Badge
          status={sev === "critical" ? "error" : sev === "warning" ? "warning" : "default"}
          text={sev}
        />
      )
    },
    {
      title: t("guardian.alerts.columns.matchedPattern", "Matched pattern"),
      dataIndex: "matched_pattern",
      key: "matched_pattern",
      ellipsis: true
    },
    {
      title: t("guardian.alerts.columns.time", "Time"),
      dataIndex: "created_at",
      key: "created_at",
      width: 170,
      render: (val: string) => new Date(val).toLocaleString()
    },
    {
      title: t("guardian.alerts.columns.read", "Read"),
      dataIndex: "is_read",
      key: "is_read",
      width: 60,
      render: (read: boolean) => (read ? t("guardian.common.yes", "Yes") : <Tag color="blue">{t("guardian.common.new", "New")}</Tag>)
    }
  ]

  const governanceColumns: ColumnsType<GovernancePolicy> = [
    {
      title: t("guardian.governance.fields.name", "Name"),
      dataIndex: "name",
      key: "name",
      ellipsis: true
    },
    {
      title: t("guardian.governance.fields.policyMode", "Mode"),
      dataIndex: "policy_mode",
      key: "policy_mode",
      width: 110,
      render: (mode: string) => (
        <Tag color={mode === "guardian" ? "blue" : "default"}>
          {t(`guardian.governance.modes.${mode}`, mode)}
        </Tag>
      )
    },
    {
      title: t("guardian.governance.fields.scopeChatTypes", "Scope"),
      dataIndex: "scope_chat_types",
      key: "scope_chat_types",
      ellipsis: true
    },
    {
      title: t("guardian.governance.fields.enabled", "Enabled"),
      dataIndex: "enabled",
      key: "enabled",
      width: 90,
      render: (enabled: boolean) =>
        enabled ? <Tag color="green">{t("guardian.common.yes", "Yes")}</Tag> : <Tag>{t("guardian.common.no", "No")}</Tag>
    },
    {
      title: t("guardian.governance.fields.updatedAt", "Updated"),
      dataIndex: "updated_at",
      key: "updated_at",
      width: 170,
      render: (value: string) => new Date(value).toLocaleString()
    },
    {
      title: t("guardian.common.actions", "Actions"),
      key: "actions",
      width: 100,
      render: (_, record) => (
        <Button type="link" danger onClick={() => confirmDeleteGovernance(record.id)}>
          {t("guardian.common.delete", "Delete")}
        </Button>
      )
    }
  ]

  if (online && (connectionConfigLoading || selfMonitoringSupported === null)) {
    return <Card loading size="small" />
  }

  if (selfMonitoringSupported === false || apiUnavailable) {
    return (
      <DsAlert
        variant="info"
        title={t(
          "guardian.selfMonitoring.unavailableTitle",
          "Self-Monitoring endpoints unavailable"
        )}
      >
        {t(
          "guardian.selfMonitoring.unavailableDescription",
          "Your connected server does not expose self-monitoring endpoints yet. Upgrade the server to use Guardian monitoring rules."
        )}
      </DsAlert>
    )
  }

  const unreadCount = unreadQuery.data?.unread_count ?? 0

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {/* Rules section */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <Title level={5} style={{ margin: 0 }}>
          {t("guardian.rules.title", "Monitoring Rules")}
        </Title>
        <Space>
          <Button
            icon={<ReloadOutlined />}
            onClick={() => rulesQuery.refetch()}
            loading={rulesQuery.isRefetching}
          >
            {t("guardian.rules.refresh", "Refresh")}
          </Button>
          <Button type="primary" icon={<PlusOutlined />} onClick={openCreate}>
            {t("guardian.rules.create", "Create Rule")}
          </Button>
        </Space>
      </div>

      <Table
        dataSource={rulesQuery.data?.items ?? []}
        columns={ruleColumns}
        rowKey="id"
        loading={rulesQuery.isLoading}
        size="small"
        pagination={{ pageSize: 10 }}
        locale={{
          emptyText: <Empty description={t("guardian.rules.empty", "No monitoring rules yet")} />
        }}
      />

      {/* Governance policies section */}
      <Card
        size="small"
        title={t("guardian.governance.title", "Governance Policies")}
        extra={
          <Space>
            <Button
              size="small"
              icon={<ReloadOutlined />}
              onClick={() => governanceQuery.refetch()}
              loading={governanceQuery.isRefetching}
            >
              {t("guardian.governance.refresh", "Refresh")}
            </Button>
            <Button
              size="small"
              type="primary"
              icon={<PlusOutlined />}
              onClick={() => setGovernanceModalOpen(true)}
            >
              {t("guardian.governance.create", "Create Policy")}
            </Button>
          </Space>
        }
      >
        <Table
          dataSource={governanceQuery.data?.items ?? []}
          columns={governanceColumns}
          rowKey="id"
          loading={governanceQuery.isLoading}
          size="small"
          pagination={{ pageSize: 5 }}
          locale={{
            emptyText: <Empty description={t("guardian.governance.empty", "No governance policies yet")} />
          }}
        />
      </Card>

      <Modal
        title={t("guardian.governance.create", "Create Governance Policy")}
        open={governanceModalOpen}
        onCancel={() => {
          setGovernanceModalOpen(false)
          governanceForm.resetFields()
        }}
        onOk={handleGovernanceSubmit}
        confirmLoading={createGovernanceMutation.isPending}
      >
        <Form
          form={governanceForm}
          layout="vertical"
          initialValues={{
            policy_mode: "self",
            scope_chat_types: "all",
            schedule_timezone: "UTC",
            enabled: true,
            transparent: false
          }}
        >
          <Form.Item
            name="name"
            label={t("guardian.governance.fields.name", "Name")}
            rules={[{ required: true, message: t("guardian.governance.fields.nameRequired", "Name is required") }]}
          >
            <Input placeholder={t("guardian.governance.fields.namePlaceholder", "e.g. Evening usage policy")} />
          </Form.Item>
          <Form.Item name="description" label={t("guardian.governance.fields.description", "Description")}>
            <TextArea
              rows={2}
              placeholder={t("guardian.governance.fields.descriptionPlaceholder", "Optional policy description")}
            />
          </Form.Item>
          <Form.Item name="policy_mode" label={t("guardian.governance.fields.policyMode", "Mode")}>
            <Select>
              <Select.Option value="self">{t("guardian.governance.modes.self", "Self")}</Select.Option>
              <Select.Option value="guardian">{t("guardian.governance.modes.guardian", "Guardian")}</Select.Option>
            </Select>
          </Form.Item>
          <Form.Item name="scope_chat_types" label={t("guardian.governance.fields.scopeChatTypes", "Scope")}>
            <Input placeholder={t("guardian.governance.fields.scopePlaceholder", "all")} />
          </Form.Item>
          <Form.Item name="schedule_timezone" label={t("guardian.governance.fields.scheduleTimezone", "Timezone")}>
            <Input placeholder={t("guardian.governance.fields.scheduleTimezonePlaceholder", "UTC")} />
          </Form.Item>
          <Form.Item name="enabled" label={t("guardian.governance.fields.enabled", "Enabled")} valuePropName="checked">
            <Switch />
          </Form.Item>
          <Form.Item name="transparent" label={t("guardian.governance.fields.transparent", "Transparent to dependent")} valuePropName="checked">
            <Switch />
          </Form.Item>
        </Form>
      </Modal>

      {/* Rule drawer */}
      <Drawer
        title={editingRule ? t("guardian.rules.edit", "Edit Rule") : t("guardian.rules.create", "Create Rule")}
        open={ruleDrawerOpen}
        onClose={closeDrawer}
        size={520}
        extra={
          <Button
            type="primary"
            onClick={handleSubmit}
            loading={createMutation.isPending || updateMutation.isPending}
          >
            {editingRule ? t("guardian.common.update", "Update") : t("guardian.common.create", "Create")}
          </Button>
        }
      >
        <Form form={form} layout="vertical" initialValues={{
          pattern_type: "literal",
          rule_type: "notify",
          action: "notify",
          phase: "both",
          severity: "warning",
          display_mode: "inline_banner",
          notification_frequency: "once_per_conversation",
          notification_channels: ["in_app"],
          crisis_resources_enabled: false,
          cooldown_minutes: 60,
          bypass_protection: "none",
          enabled: true
        }}>
          <Form.Item name="name" label={t("guardian.rules.fields.name", "Name")} rules={[{ required: true }]}>
            <Input placeholder={t("guardian.rules.fields.namePlaceholder", "e.g. Harmful content filter")} />
          </Form.Item>

          <Form.Item name="category" label={t("guardian.rules.fields.category", "Category")} rules={[{ required: true }]}>
            <Input placeholder={t("guardian.rules.fields.categoryPlaceholder", "e.g. self-harm, addiction, anxiety")} />
          </Form.Item>

          <Form.Item
            name="governance_policy_id"
            label={t("guardian.rules.fields.governancePolicy", "Governance policy")}
          >
            <Select
              allowClear
              loading={governanceQuery.isLoading}
              placeholder={t("guardian.rules.fields.governancePolicyPlaceholder", "Optional policy assignment")}
              options={(governanceQuery.data?.items ?? []).map((policy) => ({
                label: policy.name,
                value: policy.id
              }))}
            />
          </Form.Item>

          <Form.Item
            name="patterns"
            label={t("guardian.rules.fields.patterns", "Patterns (one per line)")}
            rules={[{ required: true, message: t("guardian.rules.fields.patternsRequired", "At least one pattern required") }]}
          >
            <TextArea rows={4} placeholder={t("guardian.rules.fields.patternsPlaceholder", "pattern one\npattern two")} />
          </Form.Item>

          <Form.Item name="pattern_type" label={t("guardian.rules.fields.patternType", "Pattern type")}>
            <Radio.Group>
              <Radio value="literal">{t("guardian.common.literal", "Literal")}</Radio>
              <Radio value="regex">{t("guardian.common.regex", "Regex")}</Radio>
            </Radio.Group>
          </Form.Item>

          <Form.Item name="except_patterns" label={t("guardian.rules.fields.exceptPatterns", "Exception patterns (one per line)")}>
            <TextArea rows={2} placeholder={t("guardian.rules.fields.exceptPatternsPlaceholder", "Patterns to exclude")} />
          </Form.Item>

          <Form.Item name="action" label={t("guardian.rules.fields.action", "Action")}>
            <Select>
              <Select.Option value="notify">{t("guardian.rules.options.action.notify", "Notify")}</Select.Option>
              <Select.Option value="redact">{t("guardian.rules.options.action.redact", "Redact")}</Select.Option>
              <Select.Option value="block">{t("guardian.rules.options.action.block", "Block")}</Select.Option>
            </Select>
          </Form.Item>

          <Form.Item name="phase" label={t("guardian.rules.fields.phase", "Phase")}>
            <Select>
              <Select.Option value="input">{t("guardian.rules.options.phase.input", "Input")}</Select.Option>
              <Select.Option value="output">{t("guardian.rules.options.phase.output", "Output")}</Select.Option>
              <Select.Option value="both">{t("guardian.rules.options.phase.both", "Both")}</Select.Option>
            </Select>
          </Form.Item>

          <Form.Item name="severity" label={t("guardian.rules.fields.severity", "Severity")}>
            <Select>
              <Select.Option value="info">{t("guardian.rules.options.severity.info", "Info")}</Select.Option>
              <Select.Option value="warning">{t("guardian.rules.options.severity.warning", "Warning")}</Select.Option>
              <Select.Option value="critical">{t("guardian.rules.options.severity.critical", "Critical")}</Select.Option>
            </Select>
          </Form.Item>

          <Form.Item name="display_mode" label={t("guardian.rules.fields.displayMode", "Display mode")}>
            <Select>
              <Select.Option value="inline_banner">{t("guardian.rules.options.displayMode.inlineBanner", "Inline banner")}</Select.Option>
              <Select.Option value="sidebar_note">{t("guardian.rules.options.displayMode.sidebarNote", "Sidebar note")}</Select.Option>
              <Select.Option value="post_session_summary">{t("guardian.rules.options.displayMode.postSessionSummary", "Post-session summary")}</Select.Option>
              <Select.Option value="silent_log">{t("guardian.rules.options.displayMode.silentLog", "Silent log")}</Select.Option>
            </Select>
          </Form.Item>

          <Form.Item name="notification_frequency" label={t("guardian.rules.fields.notificationFrequency", "Notification frequency")}>
            <Select>
              <Select.Option value="every_message">{t("guardian.rules.options.notificationFrequency.everyMessage", "Every message")}</Select.Option>
              <Select.Option value="once_per_conversation">{t("guardian.rules.options.notificationFrequency.oncePerConversation", "Once per conversation")}</Select.Option>
              <Select.Option value="once_per_day">{t("guardian.rules.options.notificationFrequency.oncePerDay", "Once per day")}</Select.Option>
              <Select.Option value="once_per_session">{t("guardian.rules.options.notificationFrequency.oncePerSession", "Once per session")}</Select.Option>
            </Select>
          </Form.Item>

          <Form.Item name="notification_channels" label={t("guardian.rules.fields.notificationChannels", "Notification channels")}>
            <Checkbox.Group
              options={[
                { label: t("guardian.rules.options.notificationChannels.inApp", "In-app"), value: "in_app" },
                { label: t("guardian.rules.options.notificationChannels.email", "Email"), value: "email" },
                { label: t("guardian.rules.options.notificationChannels.webhook", "Webhook"), value: "webhook" }
              ]}
            />
          </Form.Item>

          <Form.Item name="webhook_url" label={t("guardian.rules.fields.webhookUrl", "Webhook URL")}>
            <Input placeholder="https://..." />
          </Form.Item>

          <Form.Item name="trusted_contact_email" label={t("guardian.rules.fields.trustedContactEmail", "Trusted contact email")}>
            <Input placeholder={t("guardian.rules.fields.trustedContactEmailPlaceholder", "contact@example.com")} />
          </Form.Item>

          <Form.Item name="crisis_resources_enabled" label={t("guardian.rules.fields.crisisResourcesEnabled", "Show crisis resources")} valuePropName="checked">
            <Switch />
          </Form.Item>

          <Form.Item name="cooldown_minutes" label={t("guardian.rules.fields.cooldownMinutes", "Cooldown (minutes)")}>
            <InputNumber min={0} style={{ width: "100%" }} />
          </Form.Item>

          <Form.Item name="bypass_protection" label={t("guardian.rules.fields.bypassProtection", "Bypass protection")}>
            <Select>
              <Select.Option value="none">{t("guardian.rules.options.bypassProtection.none", "None")}</Select.Option>
              <Select.Option value="cooldown">{t("guardian.rules.options.bypassProtection.cooldown", "Cooldown")}</Select.Option>
              <Select.Option value="confirmation">{t("guardian.rules.options.bypassProtection.confirmation", "Confirmation")}</Select.Option>
              <Select.Option value="partner_approval">{t("guardian.rules.options.bypassProtection.partnerApproval", "Partner approval")}</Select.Option>
            </Select>
          </Form.Item>

          <Form.Item name="enabled" label={t("guardian.rules.fields.enabled", "Enabled")} valuePropName="checked">
            <Switch />
          </Form.Item>

          <Collapse
            ghost
            items={[
              {
                key: "escalation",
                label: t("guardian.rules.fields.escalation", "Escalation settings"),
                children: (
                  <>
                    <Form.Item name="escalation_session_threshold" label={t("guardian.rules.fields.escalationSessionThreshold", "Session threshold")}>
                      <InputNumber min={0} style={{ width: "100%" }} />
                    </Form.Item>
                    <Form.Item name="escalation_session_action" label={t("guardian.rules.fields.escalationSessionAction", "Session escalation action")}>
                      <Input placeholder={t("guardian.rules.fields.escalationActionPlaceholder", "e.g. block, notify_contact")} />
                    </Form.Item>
                    <Form.Item name="escalation_window_days" label={t("guardian.rules.fields.escalationWindowDays", "Window (days)")}>
                      <InputNumber min={1} style={{ width: "100%" }} />
                    </Form.Item>
                    <Form.Item name="escalation_window_threshold" label={t("guardian.rules.fields.escalationWindowThreshold", "Window threshold")}>
                      <InputNumber min={0} style={{ width: "100%" }} />
                    </Form.Item>
                    <Form.Item name="escalation_window_action" label={t("guardian.rules.fields.escalationWindowAction", "Window escalation action")}>
                      <Input placeholder={t("guardian.rules.fields.escalationActionPlaceholder", "e.g. block, notify_contact")} />
                    </Form.Item>
                  </>
                )
              }
            ]}
          />
        </Form>
      </Drawer>

      {/* Alerts section */}
      <Collapse
        ghost
        items={[
          {
            key: "alerts",
            label: (
              <Space>
                <span>{t("guardian.alerts.title", "Alerts")}</span>
                {unreadCount > 0 && <Badge count={unreadCount} size="small" />}
              </Space>
            ),
            children: (
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {selectedAlertIds.length > 0 && (
                  <Button
                    size="small"
                    onClick={() => markReadMutation.mutate(selectedAlertIds)}
                    loading={markReadMutation.isPending}
                  >
                    {t("guardian.alerts.markRead", "Mark selected read")} ({selectedAlertIds.length})
                  </Button>
                )}
                <Table
                  dataSource={alertsQuery.data?.items ?? []}
                  columns={alertColumns}
                  rowKey="id"
                  loading={alertsQuery.isLoading}
                  size="small"
                  pagination={{ pageSize: 10 }}
                  rowSelection={{
                    selectedRowKeys: selectedAlertIds,
                    onChange: (keys) => setSelectedAlertIds(keys as string[])
                  }}
                  locale={{
                    emptyText: <Empty description={t("guardian.alerts.empty", "No alerts yet")} />
                  }}
                />
              </div>
            )
          }
        ]}
      />
    </div>
  )
}

// ---------------------------------------------------------------------------
// Guardian Controls Tab
// ---------------------------------------------------------------------------

function GuardianControlsTab({ online }: { online: boolean }) {
  const { t } = useTranslation("settings")
  const qc = useQueryClient()
  const [createModalOpen, setCreateModalOpen] = useState(false)
  const [relationshipRole, setRelationshipRole] = useState<"guardian" | "dependent">("guardian")
  const [selectedRelationshipId, setSelectedRelationshipId] = useState<string | null>(null)
  const [policyDrawerOpen, setPolicyDrawerOpen] = useState(false)
  const [editingPolicy, setEditingPolicy] = useState<SupervisedPolicy | null>(null)
  const [apiUnavailable, setApiUnavailable] = useState(false)
  const [createForm] = Form.useForm()
  const [policyForm] = Form.useForm()

  const relQuery = useQuery({
    queryKey: ["guardian", "relationships", relationshipRole],
    queryFn: () => listRelationships({ role: relationshipRole }),
    enabled: online && !apiUnavailable,
    retry: false
  })

  const selectedRelationship =
    relQuery.data?.items?.find((relationship) => relationship.id === selectedRelationshipId) ?? null

  const policiesQuery = useQuery({
    queryKey: ["guardian", "policies", selectedRelationship?.id],
    queryFn: () =>
      selectedRelationship
        ? listPolicies({ relationship_id: selectedRelationship.id })
        : Promise.resolve({ items: [], total: 0 }),
    enabled: online && !apiUnavailable && !!selectedRelationship,
    retry: false
  })

  const auditQuery = useQuery({
    queryKey: ["guardian", "audit", selectedRelationship?.id],
    queryFn: () =>
      selectedRelationship
        ? getAuditLog({ relationship_id: selectedRelationship.id, limit: 50 })
        : Promise.resolve({ items: [], total: 0 }),
    enabled:
      online &&
      !apiUnavailable &&
      relationshipRole === "guardian" &&
      !!selectedRelationship,
    retry: false
  })

  React.useEffect(() => {
    if (apiUnavailable) return
    if (
      isGuardianEndpointUnsupported(relQuery.error) ||
      isGuardianEndpointUnsupported(policiesQuery.error) ||
      isGuardianEndpointUnsupported(auditQuery.error)
    ) {
      setApiUnavailable(true)
    }
  }, [apiUnavailable, auditQuery.error, policiesQuery.error, relQuery.error])

  const createRelMutation = useMutation({
    mutationFn: createRelationship,
    onSuccess: () => {
      message.success(t("guardian.relationships.created", "Relationship created"))
      qc.invalidateQueries({ queryKey: ["guardian", "relationships"] })
      setCreateModalOpen(false)
      createForm.resetFields()
    },
    onError: () => message.error(t("guardian.relationships.createFailed", "Failed to create relationship"))
  })

  const acceptMutation = useMutation({
    mutationFn: acceptRelationship,
    onSuccess: () => {
      message.success(t("guardian.relationships.accepted", "Relationship accepted"))
      qc.invalidateQueries({ queryKey: ["guardian", "relationships"] })
      qc.invalidateQueries({ queryKey: ["guardian", "audit"] })
    },
    onError: () => message.error(t("guardian.relationships.acceptFailed", "Failed to accept"))
  })

  const suspendMutation = useMutation({
    mutationFn: suspendRelationship,
    onSuccess: () => {
      message.success(t("guardian.relationships.suspended", "Relationship suspended"))
      qc.invalidateQueries({ queryKey: ["guardian", "relationships"] })
      qc.invalidateQueries({ queryKey: ["guardian", "audit"] })
    },
    onError: () => message.error(t("guardian.relationships.suspendFailed", "Failed to suspend"))
  })

  const reactivateMutation = useMutation({
    mutationFn: reactivateRelationship,
    onSuccess: () => {
      message.success(t("guardian.relationships.reactivated", "Relationship reactivated"))
      qc.invalidateQueries({ queryKey: ["guardian", "relationships"] })
      qc.invalidateQueries({ queryKey: ["guardian", "audit"] })
    },
    onError: () => message.error(t("guardian.relationships.reactivateFailed", "Failed to reactivate"))
  })

  const dissolveMutation = useMutation({
    mutationFn: (id: string) => dissolveRelationship(id, "User requested dissolution"),
    onSuccess: () => {
      message.success(t("guardian.relationships.dissolved", "Relationship dissolved"))
      setSelectedRelationshipId(null)
      qc.invalidateQueries({ queryKey: ["guardian", "relationships"] })
      qc.invalidateQueries({ queryKey: ["guardian", "audit"] })
    },
    onError: () => message.error(t("guardian.relationships.dissolveFailed", "Failed to dissolve"))
  })

  const createPolicyMutation = useMutation({
    mutationFn: createPolicy,
    onSuccess: () => {
      message.success(t("guardian.policies.created", "Policy created"))
      qc.invalidateQueries({ queryKey: ["guardian", "policies", selectedRelationship?.id] })
      qc.invalidateQueries({ queryKey: ["guardian", "audit"] })
      closePolicyDrawer()
    },
    onError: () => message.error(t("guardian.policies.createFailed", "Failed to create policy"))
  })

  const updatePolicyMutation = useMutation({
    mutationFn: ({ id, body }: { id: string; body: SupervisedPolicyUpdate }) => updatePolicy(id, body),
    onSuccess: () => {
      message.success(t("guardian.policies.updated", "Policy updated"))
      qc.invalidateQueries({ queryKey: ["guardian", "policies", selectedRelationship?.id] })
      qc.invalidateQueries({ queryKey: ["guardian", "audit"] })
      closePolicyDrawer()
    },
    onError: () => message.error(t("guardian.policies.updateFailed", "Failed to update policy"))
  })

  const deletePolicyMutation = useMutation({
    mutationFn: deletePolicy,
    onSuccess: () => {
      message.success(t("guardian.policies.deleted", "Policy deleted"))
      qc.invalidateQueries({ queryKey: ["guardian", "policies", selectedRelationship?.id] })
      qc.invalidateQueries({ queryKey: ["guardian", "audit"] })
    },
    onError: () => message.error(t("guardian.policies.deleteFailed", "Failed to delete policy"))
  })

  const openCreatePolicy = useCallback(() => {
    if (relationshipRole !== "guardian") return
    setEditingPolicy(null)
    policyForm.resetFields()
    if (selectedRelationship) {
      policyForm.setFieldValue("relationship_id", selectedRelationship.id)
    }
    setPolicyDrawerOpen(true)
  }, [policyForm, relationshipRole, selectedRelationship])

  const openEditPolicy = useCallback(
    (policy: SupervisedPolicy) => {
      setEditingPolicy(policy)
      policyForm.setFieldsValue(policy)
      setPolicyDrawerOpen(true)
    },
    [policyForm]
  )

  const closePolicyDrawer = useCallback(() => {
    setPolicyDrawerOpen(false)
    setEditingPolicy(null)
    policyForm.resetFields()
  }, [policyForm])

  const handlePolicySubmit = useCallback(async () => {
    if (relationshipRole !== "guardian") return
    try {
      const values = await policyForm.validateFields()
      if (editingPolicy) {
        const { relationship_id: _, ...body } = values
        updatePolicyMutation.mutate({ id: editingPolicy.id, body })
      } else {
        createPolicyMutation.mutate(values as SupervisedPolicyCreate)
      }
    } catch {
      // validation error
    }
  }, [policyForm, editingPolicy, relationshipRole, createPolicyMutation, updatePolicyMutation])

  const confirmDissolve = useCallback(
    (id: string) => {
      Modal.confirm({
        title: t("guardian.relationships.dissolveConfirm", "Dissolve relationship?"),
        icon: <ExclamationCircleOutlined />,
        content: t(
          "guardian.relationships.dissolveConfirmContent",
          "This action cannot be undone. All associated policies will be removed."
        ),
        okText: t("guardian.common.dissolve", "Dissolve"),
        okType: "danger",
        onOk: () => dissolveMutation.mutate(id)
      })
    },
    [dissolveMutation, t]
  )

  const confirmDeletePolicy = useCallback(
    (id: string) => {
      Modal.confirm({
        title: t("guardian.policies.deleteConfirm", "Delete policy?"),
        icon: <ExclamationCircleOutlined />,
        content: t("guardian.policies.deleteConfirmContent", "This action cannot be undone."),
        okText: t("guardian.common.delete", "Delete"),
        okType: "danger",
        onOk: () => deletePolicyMutation.mutate(id)
      })
    },
    [deletePolicyMutation, t]
  )

  const relColumns: ColumnsType<GuardianRelationship> = [
    {
      title: t("guardian.relationships.columns.id", "ID"),
      dataIndex: "id",
      key: "id",
      width: 100,
      ellipsis: true,
      render: (id: string) => (
        <Text copyable style={{ fontSize: 12 }}>
          {id.slice(0, 8)}
        </Text>
      )
    },
    {
      title: t("guardian.relationships.columns.guardian", "Guardian"),
      dataIndex: "guardian_user_id",
      key: "guardian_user_id",
      ellipsis: true
    },
    {
      title: t("guardian.relationships.columns.dependent", "Dependent"),
      dataIndex: "dependent_user_id",
      key: "dependent_user_id",
      ellipsis: true
    },
    {
      title: t("guardian.relationships.columns.type", "Type"),
      dataIndex: "relationship_type",
      key: "relationship_type",
      width: 120,
      render: (val: string) => <Tag>{val}</Tag>
    },
    {
      title: t("guardian.relationships.columns.status", "Status"),
      dataIndex: "status",
      key: "status",
      width: 120,
      render: (status: string) => {
        const normalizedStatus = normalizeRelationshipStatus(status)
        return (
          <Tag color={STATUS_COLORS[normalizedStatus] ?? "default"}>
            {normalizedStatus}
          </Tag>
        )
      }
    },
    {
      title: t("guardian.relationships.columns.created", "Created"),
      dataIndex: "created_at",
      key: "created_at",
      width: 150,
      render: (val: string) => new Date(val).toLocaleDateString()
    },
    {
      title: t("guardian.common.actions", "Actions"),
      key: "actions",
      width: 200,
      render: (_, record) => {
        const normalizedStatus = normalizeRelationshipStatus(record.status)
        return (
          <Space size="small" wrap>
            {relationshipRole === "dependent" && normalizedStatus === "pending" && (
              <Button
                size="small"
                type="link"
                onClick={() => acceptMutation.mutate(record.id)}
              >
                {t("guardian.relationships.accept", "Accept")}
              </Button>
            )}
            {relationshipRole === "guardian" && normalizedStatus === "active" && (
              <Button
                size="small"
                type="link"
                icon={<PauseCircleOutlined />}
                onClick={() => suspendMutation.mutate(record.id)}
              >
                {t("guardian.relationships.suspend", "Suspend")}
              </Button>
            )}
            {relationshipRole === "guardian" && normalizedStatus === "suspended" && (
              <Button
                size="small"
                type="link"
                icon={<PlayCircleOutlined />}
                onClick={() => reactivateMutation.mutate(record.id)}
              >
                {t("guardian.relationships.reactivate", "Reactivate")}
              </Button>
            )}
            {normalizedStatus !== "dissolved" && (
              <Button
                size="small"
                type="link"
                danger
                icon={<StopOutlined />}
                onClick={() => confirmDissolve(record.id)}
              >
                {t("guardian.common.dissolve", "Dissolve")}
              </Button>
            )}
          </Space>
        )
      }
    }
  ]

  const policyColumns: ColumnsType<SupervisedPolicy> = [
    {
      title: t("guardian.policies.fields.category", "Category"),
      dataIndex: "category",
      key: "category",
      width: 120,
      render: (cat: string) => <Tag>{cat}</Tag>
    },
    {
      title: t("guardian.policies.fields.pattern", "Pattern"),
      dataIndex: "pattern",
      key: "pattern",
      ellipsis: true
    },
    {
      title: t("guardian.policies.fields.action", "Action"),
      dataIndex: "action",
      key: "action",
      width: 90,
      render: (action: string) => (
        <Tag color={ACTION_COLORS[action] ?? "default"}>{action}</Tag>
      )
    },
    {
      title: t("guardian.policies.fields.phase", "Phase"),
      dataIndex: "phase",
      key: "phase",
      width: 80
    },
    {
      title: t("guardian.policies.fields.enabled", "Enabled"),
      dataIndex: "enabled",
      key: "enabled",
      width: 70,
      render: (val: boolean) =>
        val ? <Tag color="green">{t("guardian.common.yes", "Yes")}</Tag> : <Tag>{t("guardian.common.no", "No")}</Tag>
    },
    {
      title: t("guardian.common.actions", "Actions"),
      key: "actions",
      width: 100,
      render: (_, record) =>
        relationshipRole === "guardian" ? (
          <Space size="small">
            <Button
              type="text"
              size="small"
              icon={<EditOutlined />}
              onClick={() => openEditPolicy(record)}
            />
            <Button
              type="text"
              size="small"
              danger
              icon={<DeleteOutlined />}
              onClick={() => confirmDeletePolicy(record.id)}
            />
          </Space>
        ) : null
    }
  ]

  const auditColumns: ColumnsType<AuditLogEntry> = [
    {
      title: t("guardian.auditLog.columns.action", "Action"),
      dataIndex: "action",
      key: "action",
      width: 160
    },
    {
      title: t("guardian.auditLog.columns.actor", "Actor"),
      dataIndex: "actor_user_id",
      key: "actor_user_id",
      ellipsis: true,
      width: 140
    },
    {
      title: t("guardian.auditLog.columns.details", "Details"),
      dataIndex: "detail",
      key: "detail",
      ellipsis: true
    },
    {
      title: t("guardian.auditLog.columns.time", "Time"),
      dataIndex: "created_at",
      key: "created_at",
      width: 170,
      render: (val: string) => new Date(val).toLocaleString()
    }
  ]

  if (apiUnavailable) {
    return (
      <DsAlert
        variant="info"
        title={t(
          "guardian.controls.unavailableTitle",
          "Guardian controls endpoints unavailable"
        )}
      >
        {t(
          "guardian.controls.unavailableDescription",
          "This server does not expose guardian relationship APIs yet. Upgrade the server to manage guardian controls."
        )}
      </DsAlert>
    )
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {/* Relationships */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <Title level={5} style={{ margin: 0 }}>
          {t("guardian.relationships.title", "Guardian Relationships")}
        </Title>
        <Space>
          <Radio.Group
            size="small"
            optionType="button"
            value={relationshipRole}
            onChange={(event) => {
              setRelationshipRole(event.target.value)
              setSelectedRelationshipId(null)
            }}
            options={[
              {
                value: "guardian",
                label: t("guardian.relationships.roleGuardian", "Guardian View")
              },
              {
                value: "dependent",
                label: t("guardian.relationships.roleDependent", "Dependent View")
              }
            ]}
          />
          <Button
            icon={<ReloadOutlined />}
            onClick={() => relQuery.refetch()}
            loading={relQuery.isRefetching}
          >
            {t("guardian.relationships.refresh", "Refresh")}
          </Button>
          {relationshipRole === "guardian" && (
            <Button
              type="primary"
              icon={<PlusOutlined />}
              onClick={() => setCreateModalOpen(true)}
            >
              {t("guardian.relationships.create", "Create Relationship")}
            </Button>
          )}
        </Space>
      </div>

      <Table
        dataSource={relQuery.data?.items ?? []}
        columns={relColumns}
        rowKey="id"
        loading={relQuery.isLoading}
        size="small"
        pagination={{ pageSize: 10 }}
        onRow={(record) => ({
          onClick: () => setSelectedRelationshipId(record.id),
          style: {
            cursor: "pointer",
              background:
                selectedRelationshipId === record.id
                ? "var(--ant-primary-1, rgb(230 244 255))"
                : undefined
          }
        })}
        locale={{
          emptyText: <Empty description={t("guardian.relationships.empty", "No guardian relationships")} />
        }}
      />

      {/* Create relationship modal */}
      <Modal
        title={t("guardian.relationships.create", "Create Guardian Relationship")}
        open={createModalOpen}
        onCancel={() => {
          setCreateModalOpen(false)
          createForm.resetFields()
        }}
        onOk={async () => {
          try {
            const values = await createForm.validateFields()
            createRelMutation.mutate(values as GuardianRelationshipCreate)
          } catch {
            // validation
          }
        }}
        confirmLoading={createRelMutation.isPending}
      >
        <Form form={createForm} layout="vertical" initialValues={{ relationship_type: "parent", dependent_visible: true }}>
          <Form.Item
            name="dependent_user_id"
            label={t("guardian.relationships.fields.dependentUserId", "Dependent user ID")}
            rules={[{ required: true, message: t("guardian.relationships.fields.dependentUserIdRequired", "User ID is required") }]}
          >
            <Input placeholder={t("guardian.relationships.fields.dependentUserIdPlaceholder", "Enter dependent user ID")} />
          </Form.Item>
          <Form.Item name="relationship_type" label={t("guardian.relationships.fields.relationshipType", "Relationship type")}>
            <Select>
              <Select.Option value="parent">{t("guardian.relationships.options.relationshipType.parent", "Parent")}</Select.Option>
              <Select.Option value="legal_guardian">{t("guardian.relationships.options.relationshipType.legalGuardian", "Legal guardian")}</Select.Option>
              <Select.Option value="institutional">{t("guardian.relationships.options.relationshipType.institutional", "Institutional")}</Select.Option>
            </Select>
          </Form.Item>
          <Form.Item name="dependent_visible" label={t("guardian.relationships.fields.dependentVisible", "Visible to dependent")} valuePropName="checked">
            <Switch />
          </Form.Item>
        </Form>
      </Modal>

      {/* Policies section (shown when relationship selected) */}
      {selectedRelationship && (
        <Card
          size="small"
          title={
            <Space>
              <span>{t("guardian.policies.title", "Supervised Policies")}</span>
              <Tag>{selectedRelationship.id.slice(0, 8)}</Tag>
            </Space>
          }
          extra={
            relationshipRole === "guardian" ? (
              <Button
                type="primary"
                size="small"
                icon={<PlusOutlined />}
                onClick={openCreatePolicy}
                disabled={
                  normalizeRelationshipStatus(selectedRelationship.status) !==
                  "active"
                }
              >
                {t("guardian.policies.create", "Add Policy")}
              </Button>
            ) : null
          }
        >
          <Table
            dataSource={policiesQuery.data?.items ?? []}
            columns={policyColumns}
            rowKey="id"
            loading={policiesQuery.isLoading}
            size="small"
            pagination={{ pageSize: 5 }}
            locale={{
              emptyText: <Empty description={t("guardian.policies.empty", "No policies for this relationship")} />
            }}
          />
        </Card>
      )}

      {/* Policy drawer */}
      <Drawer
        title={editingPolicy ? t("guardian.policies.edit", "Edit Supervised Policy") : t("guardian.policies.create", "Create Supervised Policy")}
        open={policyDrawerOpen}
        onClose={closePolicyDrawer}
        size={480}
        extra={
          <Button
            type="primary"
            onClick={handlePolicySubmit}
            loading={createPolicyMutation.isPending || updatePolicyMutation.isPending}
          >
            {editingPolicy ? t("guardian.common.update", "Update") : t("guardian.common.create", "Create")}
          </Button>
        }
      >
        <Form form={policyForm} layout="vertical" initialValues={{
          policy_type: "notify",
          pattern_type: "literal",
          action: "notify",
          phase: "both",
          severity: "warning",
          notify_guardian: true,
          notify_context: "topic_only",
          enabled: true
        }}>
          <Form.Item name="relationship_id" label={t("guardian.policies.fields.relationshipId", "Relationship ID")} rules={[{ required: true }]}>
            <Input disabled={!!editingPolicy} />
          </Form.Item>
          <Form.Item name="category" label={t("guardian.policies.fields.category", "Category")} rules={[{ required: true }]}>
            <Input placeholder={t("guardian.policies.fields.categoryPlaceholder", "e.g. violence, self-harm")} />
          </Form.Item>
          <Form.Item name="pattern" label={t("guardian.policies.fields.pattern", "Pattern")} rules={[{ required: true }]}>
            <Input placeholder={t("guardian.policies.fields.patternPlaceholder", "Pattern to match")} />
          </Form.Item>
          <Form.Item name="pattern_type" label={t("guardian.policies.fields.patternType", "Pattern type")}>
            <Radio.Group>
              <Radio value="literal">{t("guardian.common.literal", "Literal")}</Radio>
              <Radio value="regex">{t("guardian.common.regex", "Regex")}</Radio>
            </Radio.Group>
          </Form.Item>
          <Form.Item name="action" label={t("guardian.policies.fields.action", "Action")}>
            <Select>
              <Select.Option value="notify">{t("guardian.policies.options.action.notify", "Notify")}</Select.Option>
              <Select.Option value="warn">{t("guardian.policies.options.action.warn", "Warn")}</Select.Option>
              <Select.Option value="redact">{t("guardian.policies.options.action.redact", "Redact")}</Select.Option>
              <Select.Option value="block">{t("guardian.policies.options.action.block", "Block")}</Select.Option>
            </Select>
          </Form.Item>
          <Form.Item name="phase" label={t("guardian.policies.fields.phase", "Phase")}>
            <Select>
              <Select.Option value="input">{t("guardian.policies.options.phase.input", "Input")}</Select.Option>
              <Select.Option value="output">{t("guardian.policies.options.phase.output", "Output")}</Select.Option>
              <Select.Option value="both">{t("guardian.policies.options.phase.both", "Both")}</Select.Option>
            </Select>
          </Form.Item>
          <Form.Item name="severity" label={t("guardian.policies.fields.severity", "Severity")}>
            <Select>
              <Select.Option value="info">{t("guardian.policies.options.severity.info", "Info")}</Select.Option>
              <Select.Option value="warning">{t("guardian.policies.options.severity.warning", "Warning")}</Select.Option>
              <Select.Option value="critical">{t("guardian.policies.options.severity.critical", "Critical")}</Select.Option>
            </Select>
          </Form.Item>
          <Form.Item name="notify_guardian" label={t("guardian.policies.fields.notifyGuardian", "Notify guardian")} valuePropName="checked">
            <Switch />
          </Form.Item>
          <Form.Item name="notify_context" label={t("guardian.policies.fields.notifyContext", "Notification context")}>
            <Select>
              <Select.Option value="topic_only">{t("guardian.policies.options.notifyContext.topicOnly", "Topic only")}</Select.Option>
              <Select.Option value="snippet">{t("guardian.policies.options.notifyContext.snippet", "Snippet")}</Select.Option>
              <Select.Option value="full_message">{t("guardian.policies.options.notifyContext.fullMessage", "Full message")}</Select.Option>
            </Select>
          </Form.Item>
          <Form.Item name="message_to_dependent" label={t("guardian.policies.fields.messageToDependent", "Message to dependent")}>
            <TextArea rows={2} placeholder={t("guardian.policies.fields.messageToDependentPlaceholder", "Optional message shown to dependent")} />
          </Form.Item>
          <Form.Item name="enabled" label={t("guardian.policies.fields.enabled", "Enabled")} valuePropName="checked">
            <Switch />
          </Form.Item>
        </Form>
      </Drawer>

      {/* Audit log */}
      {relationshipRole === "guardian" && selectedRelationship && (
        <Collapse
          ghost
          items={[
            {
              key: "audit",
              label: t("guardian.auditLog.title", "Audit Log"),
              children: (
                <Table
                  dataSource={auditQuery.data?.items ?? []}
                  columns={auditColumns}
                  rowKey="id"
                  loading={auditQuery.isLoading}
                  size="small"
                  pagination={{ pageSize: 10 }}
                  locale={{
                    emptyText: <Empty description={t("guardian.auditLog.empty", "No audit entries")} />
                  }}
                />
              )
            }
          ]}
        />
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Crisis Resources Tab
// ---------------------------------------------------------------------------

function CrisisResourcesTab({ online }: { online: boolean }) {
  const { t } = useTranslation("settings")
  const [apiUnavailable, setApiUnavailable] = useState(false)

  const crisisQuery = useQuery({
    queryKey: ["guardian", "crisis"],
    queryFn: getCrisisResources,
    enabled: online && !apiUnavailable,
    retry: false
  })

  React.useEffect(() => {
    if (apiUnavailable) return
    if (isGuardianEndpointUnsupported(crisisQuery.error)) {
      setApiUnavailable(true)
    }
  }, [apiUnavailable, crisisQuery.error])

  if (apiUnavailable) {
    return (
      <DsAlert
        variant="info"
        title={t(
          "guardian.crisis.unavailableTitle",
          "Crisis resources endpoint unavailable"
        )}
      >
        {t(
          "guardian.crisis.unavailableDescription",
          "Your connected server does not expose crisis resources yet. Upgrade the server to access this support panel."
        )}
      </DsAlert>
    )
  }

  if (crisisQuery.isLoading) return <Skeleton active />

  const data = crisisQuery.data

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {data?.disclaimer && (
        <DsAlert
          variant="info"
          title={t("guardian.crisis.disclaimer", "Disclaimer")}
        >
          {data.disclaimer}
        </DsAlert>
      )}

      {(!data?.resources || data.resources.length === 0) ? (
        <Empty description={t("guardian.crisis.empty", "No crisis resources available")} />
      ) : (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(300px, 1fr))",
            gap: 16
          }}
        >
          {data.resources.map((resource: CrisisResource, idx: number) => (
            <Card key={idx} size="small" title={resource.name}>
              <Paragraph type="secondary">{resource.description}</Paragraph>
              <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                <Text strong>{t("guardian.crisis.contact", "Contact")}: {resource.contact}</Text>
                {resource.url && (
                  <a href={resource.url} target="_blank" rel="noopener noreferrer">
                    {resource.url}
                  </a>
                )}
                {resource.available_24_7 && (
                  <Tag color="green">{t("guardian.crisis.available247", "Available 24/7")}</Tag>
                )}
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

export function GuardianSettings() {
  const { t } = useTranslation("settings")
  const online = useServerOnline()
  const navigate = useNavigate()
  const { uxState } = useConnectionUxState()
  const { capabilities, loading: capabilitiesLoading } = useServerCapabilities()
  const guardianRoutesAvailable = Boolean(
    capabilities?.hasGuardian && capabilities?.hasSelfMonitoring
  )

  if (!capabilitiesLoading && !guardianRoutesAvailable) {
    return (
      <DsAlert
        variant="info"
        title={t("guardian.unavailableTitle", "Guardian settings unavailable")}
      >
        {t(
          "guardian.unavailableDescription",
          "Your current server does not expose Guardian or Self-Monitoring endpoints."
        )}
      </DsAlert>
    )
  }

  const offlineWarning =
    !online && uxState !== "testing"
      ? uxState === "error_auth" || uxState === "configuring_auth"
        ? (
          <DsAlert
            variant="warning"
            title={t("guardian.authRequiredTitle", "Add your credentials to manage Guardian settings.")}
            action={{
              label: t("guardian.openSettings", "Open Settings"),
              onClick: () => navigate("/settings/tldw")
            }}
            className="mb-4"
          >
            {t(
              "guardian.authRequiredDescription",
              "Your server is reachable, but Guardian settings require valid credentials before they can be managed."
            )}
          </DsAlert>
          )
        : uxState === "unconfigured" || uxState === "configuring_url"
          ? (
            <DsAlert
              variant="warning"
              title={t("guardian.setupRequiredTitle", "Finish setup to manage Guardian settings.")}
              action={{
                label: t("guardian.finishSetup", "Finish Setup"),
                onClick: () => navigate("/")
              }}
              className="mb-4"
            >
              {t(
                "guardian.setupRequiredDescription",
                "Complete the tldw server setup flow, then return here to manage Guardian and Self-Monitoring settings."
              )}
            </DsAlert>
            )
          : uxState === "error_unreachable"
            ? (
              <DsAlert
                variant="warning"
                title={t("guardian.unreachableTitle", "Can't reach your tldw server right now.")}
                action={{
                  label: t("guardian.diagnostics", "Health & diagnostics"),
                  onClick: () => navigate("/settings/health")
                }}
                secondaryAction={{
                  label: t("guardian.openSettings", "Open Settings"),
                  onClick: () => navigate("/settings/tldw")
                }}
                className="mb-4"
              >
                {t(
                  "guardian.unreachableDescription",
                  "Your server settings are saved, but Guardian settings cannot reach the tldw server right now."
                )}
              </DsAlert>
              )
            : (
              <DsAlert
                variant="warning"
                title={t("guardian.serverOfflineTitle", "Server offline")}
                className="mb-4"
              >
                {t(
                  "guardian.serverOffline",
                  "Connect to your tldw server to manage guardian and monitoring settings."
                )}
              </DsAlert>
              )
      : null

  return (
    <div style={{ maxWidth: 960, margin: "0 auto", padding: "16px 0" }}>
      <Title level={4}>{t("guardian.title", "Guardian & Self-Monitoring")}</Title>
      <Paragraph type="secondary">
        {t(
          "guardian.subtitle",
          "Configure personal content monitoring rules, guardian supervision relationships, and access crisis resources."
        )}
      </Paragraph>

      {offlineWarning}

      <Tabs
        defaultActiveKey="self-monitoring"
        items={[
          {
            key: "self-monitoring",
            label: t("guardian.tabs.selfMonitoring", "Self-Monitoring"),
            children: <SelfMonitoringTab online={online} />
          },
          {
            key: "guardian",
            label: t("guardian.tabs.guardianControls", "Guardian Controls"),
            children: <GuardianControlsTab online={online} />
          },
          {
            key: "crisis",
            label: t("guardian.tabs.crisisResources", "Crisis Resources"),
            children: <CrisisResourcesTab online={online} />
          }
        ]}
      />
    </div>
  )
}
