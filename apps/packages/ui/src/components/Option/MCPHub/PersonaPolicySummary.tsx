import { useEffect, useState } from "react"
import { Button, Card, Empty, Space, Tag, Typography } from "antd"
import { ProductStateAlert as Alert } from "@/components/Option/productStatePrimitives"

import {
  getAssignmentExternalAccess,
  getEffectivePolicy,
  listPolicyAssignments,
  type McpHubEffectiveExternalAccess,
  type McpHubEffectivePolicy,
  type McpHubPermissionPolicyDocument
} from "@/services/tldw/mcp-hub"

import { getPathAllowlistSummary, getPathScopeLabel } from "./policyHelpers"
import { ExternalAccessSummary } from "./ExternalAccessSummary"

type PersonaPolicySummaryProps = {
  personaId?: string | null
}

export const PersonaPolicySummary = ({ personaId }: PersonaPolicySummaryProps) => {
  const [policy, setPolicy] = useState<McpHubEffectivePolicy | null>(null)
  const [externalAccess, setExternalAccess] = useState<McpHubEffectiveExternalAccess | null>(null)
  const [loading, setLoading] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    const load = async () => {
      if (!personaId) {
        setPolicy(null)
        setExternalAccess(null)
        return
      }
      setLoading(true)
      setErrorMessage(null)
      try {
        const [next, assignmentRows] = await Promise.all([
          getEffectivePolicy({ persona_id: personaId }),
          listPolicyAssignments({ target_type: "persona", target_id: personaId })
        ])
        if (!cancelled) {
          setPolicy(next)
          const activeAssignment = (assignmentRows || []).find((row) => row.is_active)
          if (activeAssignment) {
            try {
              const summary = await getAssignmentExternalAccess(activeAssignment.id)
              if (!cancelled) {
                setExternalAccess(summary)
              }
            } catch {
              if (!cancelled) {
                setExternalAccess(null)
              }
            }
          } else {
            setExternalAccess(null)
          }
        }
      } catch {
        if (!cancelled) {
          setPolicy(null)
          setExternalAccess(null)
          setErrorMessage("Failed to load effective tool policy.")
        }
      } finally {
        if (!cancelled) {
          setLoading(false)
        }
      }
    }

    void load()
    return () => {
      cancelled = true
    }
  }, [personaId])

  const provenance = Array.isArray(policy?.provenance) ? policy.provenance : []
  const resolvedPolicyDocument =
    (policy?.resolved_policy_document as McpHubPermissionPolicyDocument | null | undefined) ??
    (policy?.policy_document as McpHubPermissionPolicyDocument | null | undefined) ??
    {}
  const authoredPolicyDocument =
    (policy?.policy_document as McpHubPermissionPolicyDocument | null | undefined) ?? {}
  const capabilityMappingSummary = Array.isArray(policy?.capability_mapping_summary)
    ? policy.capability_mapping_summary
    : []
  const unresolvedCapabilities = Array.isArray(policy?.unresolved_capabilities)
    ? policy.unresolved_capabilities
    : []
  const capabilityWarnings = Array.isArray(policy?.capability_warnings)
    ? policy.capability_warnings
    : []
  const governancePackLabels = Array.from(
    new Set(
      provenance
        .filter((entry) => entry.field === "governance_pack")
        .map((entry) => {
          const value = entry.value as { pack_id?: unknown; pack_version?: unknown } | null
          const packId = typeof value?.pack_id === "string" ? value.pack_id : null
          const packVersion = typeof value?.pack_version === "string" ? value.pack_version : null
          if (!packId || !packVersion) {
            return null
          }
          return `Pack ${packId}@${packVersion}`
        })
        .filter((label): label is string => Boolean(label))
    )
  )

  return (
    <Card
      title="Tool Policy Summary"
      extra={
        <Button size="small" type="link" href="/settings/mcp-hub">
          Open MCP Hub
        </Button>
      }
    >
      {errorMessage ? <Alert type="error" title={errorMessage} showIcon /> : null}
      {!personaId ? (
        <Empty description="Select a persona to review its tool access." />
      ) : loading ? (
        <Typography.Text type="secondary">Loading effective policy...</Typography.Text>
      ) : policy ? (
        <Space orientation="vertical" size="small" style={{ width: "100%" }}>
          {getPathScopeLabel(resolvedPolicyDocument.path_scope_mode) ? (
            <Typography.Text type="secondary">
              {`Local file scope: ${getPathScopeLabel(resolvedPolicyDocument.path_scope_mode)}`}
            </Typography.Text>
          ) : null}
          {getPathAllowlistSummary(resolvedPolicyDocument.path_allowlist_prefixes) ? (
            <Typography.Text type="secondary">
              {`Allowed paths: ${getPathAllowlistSummary(resolvedPolicyDocument.path_allowlist_prefixes)}`}
            </Typography.Text>
          ) : null}
          {policy.selected_assignment_workspace_ids?.length ? (
            <Typography.Text type="secondary">
              {`Allowed workspaces: ${policy.selected_assignment_workspace_ids.join(", ")}`}
            </Typography.Text>
          ) : null}
          {policy.selected_workspace_source_mode === "named" && policy.selected_workspace_set_object_name ? (
            <Typography.Text type="secondary">
              {`Workspace set: ${policy.selected_workspace_set_object_name}`}
            </Typography.Text>
          ) : null}
          {policy.selected_workspace_trust_source ? (
            <Typography.Text type="secondary">
              {`Workspace trust source: ${
                policy.selected_workspace_trust_source === "shared_registry"
                  ? "shared registry"
                  : "user-local"
              }`}
            </Typography.Text>
          ) : null}
          {capabilityMappingSummary.length ? (
            <Space orientation="vertical" size={4}>
              {capabilityMappingSummary.map((summary) => (
                <Typography.Text key={`${summary.capability_name}:${summary.mapping_id ?? "unmapped"}`}>
                  {`${
                    summary.resolution_intent === "deny" ? "Denied" : "Mapped"
                  } ${summary.capability_name} via ${summary.mapping_id ?? "local mapping"}`}
                </Typography.Text>
              ))}
            </Space>
          ) : null}
          {unresolvedCapabilities.length ? (
            <Space orientation="vertical" size={4}>
              {unresolvedCapabilities.map((capability) => (
                <Typography.Text key={capability} type="warning">
                  {`Unresolved capability: ${capability}`}
                </Typography.Text>
              ))}
            </Space>
          ) : null}
          {capabilityWarnings.length ? (
            <Space orientation="vertical" size={4}>
              {capabilityWarnings.map((warning) => (
                <Typography.Text key={warning} type="warning">
                  {warning}
                </Typography.Text>
              ))}
            </Space>
          ) : null}
          <Space wrap>
            {policy.capabilities.map((capability) => (
              <Tag key={capability}>{capability}</Tag>
            ))}
            {policy.approval_mode ? <Tag color="gold">{policy.approval_mode}</Tag> : null}
            {getPathScopeLabel(resolvedPolicyDocument.path_scope_mode) ? (
              <Tag color="cyan">{getPathScopeLabel(resolvedPolicyDocument.path_scope_mode)}</Tag>
            ) : null}
            {resolvedPolicyDocument.path_scope_enforcement ? (
              <Tag color="orange">Path approval fallback</Tag>
            ) : null}
            {getPathAllowlistSummary(resolvedPolicyDocument.path_allowlist_prefixes) ? (
              <Tag color="blue">{`paths ${getPathAllowlistSummary(resolvedPolicyDocument.path_allowlist_prefixes)}`}</Tag>
            ) : null}
            {provenance.some((entry) => entry.source_kind === "assignment_override") ? (
              <Tag color="cyan">Override active</Tag>
            ) : null}
            {provenance.some((entry) => entry.source_kind === "assignment_path_scope_object") ? (
              <Tag color="purple">Named path scope</Tag>
            ) : null}
            {governancePackLabels.map((label) => (
              <Tag key={label} color="geekblue">
                {label}
              </Tag>
            ))}
            {policy.selected_workspace_source_mode === "named" && policy.selected_workspace_set_object_name ? (
              <Tag color="geekblue">{`workspace set ${policy.selected_workspace_set_object_name}`}</Tag>
            ) : null}
            {policy.selected_workspace_trust_source ? (
              <Tag color={policy.selected_workspace_trust_source === "shared_registry" ? "magenta" : "purple"}>
                {policy.selected_workspace_trust_source === "shared_registry"
                  ? "shared registry"
                  : "user-local"}
              </Tag>
            ) : null}
            {unresolvedCapabilities.map((capability) => (
              <Tag key={`unresolved:${capability}`} color="red">
                {`unresolved ${capability}`}
              </Tag>
            ))}
          </Space>
          <Space wrap>
            {policy.allowed_tools.map((tool) => (
              <Tag key={tool} color="green">
                {tool}
              </Tag>
            ))}
            {policy.denied_tools.map((tool) => (
              <Tag key={tool} color="red">
                {tool}
              </Tag>
            ))}
          </Space>
          <Typography.Text strong>External Services</Typography.Text>
          <ExternalAccessSummary
            summary={externalAccess}
            emptyText="No external service access is configured for this persona."
          />
        </Space>
      ) : (
        <Empty description="No tool policy is active for this persona yet." />
      )}
    </Card>
  )
}
