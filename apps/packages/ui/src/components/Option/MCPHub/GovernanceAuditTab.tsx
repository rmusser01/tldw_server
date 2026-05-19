import { useCallback, useEffect, useMemo, useState } from "react"
import { Button, Card, Empty, List, Space, Tag, Typography } from "antd"
import { ProductStateAlert as Alert } from "@/components/Option/productStatePrimitives"

import {
  listGovernanceAuditFindings,
  updatePermissionProfile,
  updatePolicyAssignment,
  updateExternalServer,
  type McpHubGovernanceAuditFinding,
  type McpHubGovernanceAuditNavigateTarget
} from "@/services/tldw/mcp-hub"
import { copyToClipboard } from "@/utils/clipboard"
import { downloadBlob } from "@/utils/download-blob"
import {
  buildAuditCounts,
  buildAuditInlineAction,
  buildAuditJsonExport,
  buildAuditMarkdownReport,
  buildAuditRemediationSteps,
  dedupeAuditValues,
  FINDING_TYPE_LABELS,
  FINDING_TYPE_ORDER,
  groupAuditFindings,
  OBJECT_KIND_LABELS,
  summarizeRelatedObjects,
  type GovernanceAuditFilterState,
  type GovernanceAuditRelatedObjectFocus
} from "./governanceAuditHelpers"

type GovernanceAuditTabProps = {
  onOpen?: (target: McpHubGovernanceAuditNavigateTarget) => void
}

const severityColor = (severity: "error" | "warning"): "red" | "gold" =>
  severity === "error" ? "red" : "gold"

const parseInlineActionObjectId = (objectId: string): number | null => {
  const parsed = Number(objectId)
  return Number.isInteger(parsed) && parsed >= 0 ? parsed : null
}

export const GovernanceAuditTab = ({ onOpen }: GovernanceAuditTabProps) => {
  const [allItems, setAllItems] = useState<McpHubGovernanceAuditFinding[]>([])
  const [loading, setLoading] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [severityFilter, setSeverityFilter] = useState<"all" | "error" | "warning">("all")
  const [findingTypeFilter, setFindingTypeFilter] = useState<
    "all" | McpHubGovernanceAuditFinding["finding_type"]
  >("all")
  const [objectKindFilter, setObjectKindFilter] = useState<"all" | string>("all")
  const [scopeTypeFilter, setScopeTypeFilter] = useState<"all" | string>("all")
  const [hasRelatedObjectOnly, setHasRelatedObjectOnly] = useState(false)
  const [relatedObjectFocus, setRelatedObjectFocus] =
    useState<GovernanceAuditRelatedObjectFocus | null>(null)
  const [actionStatus, setActionStatus] = useState<{
    type: "success" | "error"
    message: string
  } | null>(null)
  const [pendingActionObjectId, setPendingActionObjectId] = useState<string | null>(null)

  const loadAuditFindings = useCallback(async () => {
    setLoading(true)
    setErrorMessage(null)
    try {
      const result = await listGovernanceAuditFindings()
      setAllItems(Array.isArray(result?.items) ? result.items : [])
    } catch {
      setAllItems([])
      setErrorMessage("Failed to load governance audit findings.")
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void loadAuditFindings()
  }, [loadAuditFindings])

  const availableFindingTypes = useMemo(
    () => dedupeAuditValues(allItems.map((item) => String(item.finding_type || ""))),
    [allItems]
  )
  const availableObjectKinds = useMemo(
    () => dedupeAuditValues(allItems.map((item) => String(item.object_kind || ""))),
    [allItems]
  )
  const availableScopeTypes = useMemo(
    () => dedupeAuditValues(allItems.map((item) => String(item.scope_type || ""))),
    [allItems]
  )

  const filterState = useMemo<GovernanceAuditFilterState>(
    () => ({
      severity: severityFilter,
      finding_type: findingTypeFilter,
      object_kind: objectKindFilter,
      scope_type: scopeTypeFilter,
      has_related_object_only: hasRelatedObjectOnly
    }),
    [severityFilter, findingTypeFilter, objectKindFilter, scopeTypeFilter, hasRelatedObjectOnly]
  )

  const baseFilteredItems = useMemo(() => {
    return allItems.filter((item) => {
      if (severityFilter !== "all" && item.severity !== severityFilter) {
        return false
      }
      if (findingTypeFilter !== "all" && item.finding_type !== findingTypeFilter) {
        return false
      }
      if (objectKindFilter !== "all" && item.object_kind !== objectKindFilter) {
        return false
      }
      if (scopeTypeFilter !== "all" && item.scope_type !== scopeTypeFilter) {
        return false
      }
      if (
        hasRelatedObjectOnly &&
        !String(item.related_object_label || item.related_object_id || "").trim()
      ) {
        return false
      }
      return true
    })
  }, [
    allItems,
    severityFilter,
    findingTypeFilter,
    objectKindFilter,
    scopeTypeFilter,
    hasRelatedObjectOnly
  ])

  const relatedSummaries = useMemo(
    () => summarizeRelatedObjects(baseFilteredItems),
    [baseFilteredItems]
  )

  const focusedItems = useMemo(() => {
    if (!relatedObjectFocus) return baseFilteredItems
    return baseFilteredItems.filter(
      (item) =>
        String(item.related_object_kind || "").trim() === relatedObjectFocus.kind &&
        String(item.related_object_id || "").trim() === relatedObjectFocus.id
    )
  }, [baseFilteredItems, relatedObjectFocus])

  const filteredCounts = useMemo(() => buildAuditCounts(focusedItems), [focusedItems])
  const groupedItems = useMemo(() => groupAuditFindings(focusedItems), [focusedItems])

  const _buildMarkdownReport = () =>
    buildAuditMarkdownReport({
      generated_at: new Date().toISOString(),
      items: focusedItems,
      filters: filterState,
      counts: filteredCounts,
      related_object_focus: relatedObjectFocus,
      related_summaries: relatedSummaries
    })

  const _copyAuditReport = async () => {
    try {
      await copyToClipboard({
        text: _buildMarkdownReport(),
        formatted: false
      })
      setActionStatus({ type: "success", message: "Audit report copied." })
    } catch {
      setActionStatus({ type: "error", message: "Failed to copy audit report." })
    }
  }

  const _downloadAuditExport = async (format: "json" | "markdown") => {
    try {
      if (format === "json") {
        const payload = buildAuditJsonExport({
          generated_at: new Date().toISOString(),
          items: focusedItems,
          filters: filterState,
          counts: filteredCounts,
          related_object_focus: relatedObjectFocus
        })
        downloadBlob(
          new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" }),
          "mcp-hub-audit.json"
        )
        setActionStatus({ type: "success", message: "JSON export downloaded." })
        return
      }
      downloadBlob(
        new Blob([_buildMarkdownReport()], { type: "text/markdown" }),
        "mcp-hub-audit.md"
      )
      setActionStatus({ type: "success", message: "Markdown export downloaded." })
    } catch {
      setActionStatus({
        type: "error",
        message:
          format === "json"
            ? "Failed to download JSON export."
            : "Failed to download Markdown export."
      })
    }
  }

  const _runInlineAction = async (finding: McpHubGovernanceAuditFinding) => {
    const action = buildAuditInlineAction(finding)
    if (!action) {
      return
    }
    const confirmLines = [action.confirm_title, action.confirm_description].filter(
      (value): value is string => Boolean(value && value.trim())
    )
    if (confirmLines.length > 0 && !window.confirm(confirmLines.join("\n\n"))) {
      return
    }
    setPendingActionObjectId(action.object_id)
    try {
      if (action.kind === "deactivate_external_server") {
        await updateExternalServer(action.object_id, { enabled: false })
        setActionStatus({ type: "success", message: "Server deactivated." })
      } else {
        const objectId = parseInlineActionObjectId(action.object_id)
        if (objectId === null) {
          setActionStatus({
            type: "error",
            message: "The audit action referenced an invalid object identifier."
          })
          return
        }
        if (action.kind === "clear_permission_profile_reference") {
          await updatePolicyAssignment(objectId, { profile_id: null })
          setActionStatus({ type: "success", message: action.success_message })
        } else if (action.object_kind === "policy_assignment") {
          await updatePolicyAssignment(objectId, { path_scope_object_id: null })
          setActionStatus({ type: "success", message: action.success_message })
        } else {
          await updatePermissionProfile(objectId, { path_scope_object_id: null })
          setActionStatus({ type: "success", message: action.success_message })
        }
      }
      await loadAuditFindings()
    } catch {
      setActionStatus({
        type: "error",
        message:
          action.kind === "deactivate_external_server"
            ? "Failed to deactivate server."
            : action.error_message
      })
    } finally {
      setPendingActionObjectId(null)
    }
  }

  return (
    <Space orientation="vertical" size="middle" style={{ width: "100%" }}>
      <Typography.Text type="secondary">
        Audit aggregates concrete governance findings across the visible MCP Hub configuration.
        Multi-root readiness findings are advisory; assignment blockers remain the enforcement
        boundary.
      </Typography.Text>
      {errorMessage ? <Alert type="error" title={errorMessage} showIcon /> : null}
      {actionStatus ? (
        <Alert
          type={actionStatus.type}
          title={actionStatus.message}
          showIcon
          closable
          onClose={() => setActionStatus(null)}
        />
      ) : null}

      <Card loading={loading}>
        <Space wrap align="start">
          <Tag>{`${filteredCounts.total} findings`}</Tag>
          <Tag color="red">{`${filteredCounts.error} errors`}</Tag>
          <Tag color="gold">{`${filteredCounts.warning} warnings`}</Tag>
        </Space>
        <Space wrap style={{ marginTop: 12 }}>
          <Button size="small" onClick={() => void _copyAuditReport()}>
            Copy report
          </Button>
          <Button size="small" onClick={() => void _downloadAuditExport("json")}>
            Download JSON
          </Button>
          <Button size="small" onClick={() => void _downloadAuditExport("markdown")}>
            Download Markdown
          </Button>
        </Space>
        <Space wrap style={{ marginTop: 12 }}>
          <Typography.Text type="secondary">Severity</Typography.Text>
          <Tag.CheckableTag
            checked={severityFilter === "all"}
            onChange={() => setSeverityFilter("all")}
            aria-label="All severities"
          >
            All
          </Tag.CheckableTag>
          <Tag.CheckableTag
            checked={severityFilter === "error"}
            onChange={() => setSeverityFilter("error")}
            aria-label="Error severities"
          >
            Errors
          </Tag.CheckableTag>
          <Tag.CheckableTag
            checked={severityFilter === "warning"}
            onChange={() => setSeverityFilter("warning")}
            aria-label="Warning severities"
          >
            Warnings
          </Tag.CheckableTag>
        </Space>
        <Space wrap style={{ marginTop: 8 }}>
          <Typography.Text type="secondary">Type</Typography.Text>
          <Tag.CheckableTag
            checked={findingTypeFilter === "all"}
            onChange={() => setFindingTypeFilter("all")}
            aria-label="All finding types"
          >
            All
          </Tag.CheckableTag>
          {FINDING_TYPE_ORDER.filter((findingType) =>
            availableFindingTypes.includes(findingType)
          ).map((findingType) => (
            <Tag.CheckableTag
              key={findingType}
              checked={findingTypeFilter === findingType}
              onChange={() => setFindingTypeFilter(findingType)}
              aria-label={`Finding type ${FINDING_TYPE_LABELS[findingType]}`}
            >
              {FINDING_TYPE_LABELS[findingType]}
            </Tag.CheckableTag>
          ))}
        </Space>
        <Space wrap style={{ marginTop: 8 }}>
          <Typography.Text type="secondary">Object Kind</Typography.Text>
          <Tag.CheckableTag
            checked={objectKindFilter === "all"}
            onChange={() => setObjectKindFilter("all")}
            aria-label="All object kinds"
          >
            All
          </Tag.CheckableTag>
          {availableObjectKinds.map((objectKind) => (
            <Tag.CheckableTag
              key={objectKind}
              checked={objectKindFilter === objectKind}
              onChange={() => setObjectKindFilter(objectKind)}
              aria-label={`Object kind ${OBJECT_KIND_LABELS[objectKind] || objectKind}`}
            >
              {OBJECT_KIND_LABELS[objectKind] || objectKind}
            </Tag.CheckableTag>
          ))}
        </Space>
        <Space wrap style={{ marginTop: 8 }}>
          <Typography.Text type="secondary">Scope</Typography.Text>
          <Tag.CheckableTag
            checked={scopeTypeFilter === "all"}
            onChange={() => setScopeTypeFilter("all")}
            aria-label="All scopes"
          >
            All
          </Tag.CheckableTag>
          {availableScopeTypes.map((scopeType) => (
            <Tag.CheckableTag
              key={scopeType}
              checked={scopeTypeFilter === scopeType}
              onChange={() => setScopeTypeFilter(scopeType)}
              aria-label={`Scope ${scopeType}`}
            >
              {scopeType}
            </Tag.CheckableTag>
          ))}
        </Space>
        <Space wrap style={{ marginTop: 8 }}>
          <Typography.Text type="secondary">Relationships</Typography.Text>
          <Tag.CheckableTag
            checked={hasRelatedObjectOnly}
            onChange={(checked) => setHasRelatedObjectOnly(checked)}
            aria-label="Has related object"
          >
            Has related object
          </Tag.CheckableTag>
        </Space>
      </Card>

      {relatedSummaries.length > 0 ? (
        <Card>
          <Space orientation="vertical" size="small" style={{ width: "100%" }}>
            <Typography.Text type="secondary">
              Top related objects in current filtered findings
            </Typography.Text>
            <Space wrap>
              {relatedSummaries.map((summary) => {
                const isActive =
                  relatedObjectFocus?.kind === summary.kind &&
                  relatedObjectFocus?.id === summary.id
                const label = `${summary.label} · ${summary.kind} · ${summary.count} finding${
                  summary.count === 1 ? "" : "s"
                }`
                return (
                  <Button
                    key={summary.key}
                    size="small"
                    type={isActive ? "primary" : "default"}
                    onClick={() =>
                      setRelatedObjectFocus({
                        kind: summary.kind,
                        id: summary.id,
                        label: summary.label
                      })
                    }
                    aria-label={label}
                  >
                    {label}
                  </Button>
                )
              })}
            </Space>
            {relatedObjectFocus ? (
              <Space wrap>
                <Tag>{`Related object: ${relatedObjectFocus.label}`}</Tag>
                <Button
                  size="small"
                  onClick={() => setRelatedObjectFocus(null)}
                  aria-label="Clear related object focus"
                >
                  Clear related object focus
                </Button>
              </Space>
            ) : null}
          </Space>
        </Card>
      ) : null}

      {focusedItems.length === 0 ? (
        <Card>
          <Empty description="No governance findings" />
        </Card>
      ) : null}

      {groupedItems.map(({ finding_type: findingType, items }) => {
        if (items.length === 0) {
          return null
        }
        return (
          <Card
            key={findingType}
            title={`${FINDING_TYPE_LABELS[findingType]} (${items.length})`}
            data-testid={`audit-group-${findingType}`}
          >
            <List
              dataSource={items}
              renderItem={(finding) => (
                <List.Item
                  actions={(() => {
                    const actions = [
                      <Button
                        key="open"
                        size="small"
                        onClick={() => onOpen?.(finding.navigate_to)}
                      >
                        Open
                      </Button>
                    ]
                    const inlineAction = buildAuditInlineAction(finding)
                    if (inlineAction) {
                      actions.push(
                        <Button
                          key={inlineAction.kind}
                          size="small"
                          loading={pendingActionObjectId === inlineAction.object_id}
                          onClick={() => void _runInlineAction(finding)}
                        >
                          {inlineAction.label}
                        </Button>
                      )
                    }
                    return actions
                  })()}
                >
                  <Space orientation="vertical" size={4} style={{ width: "100%" }}>
                    <Space wrap>
                      <Typography.Text strong>{finding.object_label}</Typography.Text>
                      <Tag color={severityColor(finding.severity)}>{finding.severity}</Tag>
                      <Tag>{finding.object_kind}</Tag>
                      <Tag>{finding.scope_type}</Tag>
                      {finding.scope_id !== null && finding.scope_id !== undefined ? (
                        <Tag>{`scope ${finding.scope_id}`}</Tag>
                      ) : null}
                    </Space>
                    <Typography.Text>{finding.message}</Typography.Text>
                    {finding.related_object_label ? (
                      <Typography.Text type="secondary">
                        {`Related to: ${finding.related_object_label}${
                          finding.related_object_kind
                            ? ` (${finding.related_object_kind})`
                            : ""
                        }`}
                      </Typography.Text>
                    ) : null}
                    {(() => {
                      const remediation = buildAuditRemediationSteps(finding)
                      if (remediation.steps.length === 0) {
                        return null
                      }
                      return (
                        <Space orientation="vertical" size={2} style={{ width: "100%" }}>
                          <Typography.Text strong>Suggested next steps</Typography.Text>
                          <ol style={{ margin: 0, paddingLeft: 18 }}>
                            {remediation.steps.map((step) => (
                              <li key={step}>
                                <Typography.Text>{step}</Typography.Text>
                              </li>
                            ))}
                          </ol>
                          {remediation.note ? (
                            <Typography.Text type="secondary">{remediation.note}</Typography.Text>
                          ) : null}
                        </Space>
                      )
                    })()}
                  </Space>
                </List.Item>
              )}
            />
          </Card>
        )
      })}
    </Space>
  )
}
