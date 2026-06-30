import { useEffect, useMemo, useState } from "react"
import { Button, Card, Checkbox, Empty, List, Space, Tag, Typography } from "antd"
import { ProductStateAlert as Alert } from "@/components/Option/productStatePrimitives"

import {
  createApprovalPolicy,
  deleteApprovalPolicy,
  listApprovalPolicies,
  updateApprovalPolicy,
  type McpHubApprovalPolicy,
  type McpHubApprovalMode
} from "@/services/tldw/mcp-hub"

import {
  MCP_HUB_APPROVAL_MODE_OPTIONS,
  MCP_HUB_APPROVAL_DURATION_OPTIONS,
  MCP_HUB_SCOPE_OPTIONS,
  toggleStringValue
} from "./policyHelpers"

export const ApprovalPoliciesTab = () => {
  const [policies, setPolicies] = useState<McpHubApprovalPolicy[]>([])
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [createOpen, setCreateOpen] = useState(false)
  const [editingId, setEditingId] = useState<number | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [name, setName] = useState("")
  const [ownerScopeType, setOwnerScopeType] = useState<"global" | "org" | "team" | "user">("global")
  const [mode, setMode] = useState<McpHubApprovalMode>("ask_outside_profile")
  const [durationOptions, setDurationOptions] = useState<string[]>(["session"])
  const [isActive, setIsActive] = useState(true)

  const canSave = useMemo(
    () => name.trim().length > 0 && durationOptions.length > 0 && !saving,
    [durationOptions.length, name, saving]
  )

  const loadPolicies = async () => {
    setLoading(true)
    setErrorMessage(null)
    try {
      const rows = await listApprovalPolicies()
      setPolicies(Array.isArray(rows) ? rows : [])
    } catch {
      setPolicies([])
      setErrorMessage("Failed to load approval policies.")
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void loadPolicies()
  }, [])

  const resetForm = () => {
    setCreateOpen(false)
    setEditingId(null)
    setName("")
    setOwnerScopeType("global")
    setMode("ask_outside_profile")
    setDurationOptions(["session"])
    setIsActive(true)
  }

  const openForEdit = (policy: McpHubApprovalPolicy) => {
    setCreateOpen(true)
    setEditingId(policy.id)
    setName(policy.name)
    setOwnerScopeType(policy.owner_scope_type)
    setMode(policy.mode)
    setDurationOptions(
      Array.isArray(policy.rules?.duration_options)
        ? policy.rules.duration_options.map((entry) => String(entry))
        : ["session"]
    )
    setIsActive(policy.is_active)
  }

  const handleSave = async () => {
    if (!canSave) return
    setSaving(true)
    setErrorMessage(null)
    try {
      const payload = {
        name: name.trim(),
        owner_scope_type: ownerScopeType,
        mode,
        rules: {
          duration_options: durationOptions
        },
        is_active: isActive
      }
      if (editingId) {
        await updateApprovalPolicy(editingId, payload)
      } else {
        await createApprovalPolicy(payload)
      }
      resetForm()
      await loadPolicies()
    } catch {
      setErrorMessage(
        editingId ? "Failed to update approval policy." : "Failed to create approval policy."
      )
    } finally {
      setSaving(false)
    }
  }

  const handleDelete = async (policyId: number) => {
    if (typeof window !== "undefined" && !window.confirm("Delete this approval policy?")) {
      return
    }
    setErrorMessage(null)
    try {
      await deleteApprovalPolicy(policyId)
      await loadPolicies()
    } catch {
      setErrorMessage("Failed to delete approval policy.")
    }
  }

  return (
    <Space orientation="vertical" size="middle" style={{ width: "100%" }}>
      <Typography.Text type="secondary">
        Approval policies define when MCP Hub should interrupt tool execution and ask for runtime consent.
      </Typography.Text>
      {errorMessage ? <Alert type="error" title={errorMessage} showIcon /> : null}

      <Button type="primary" onClick={() => setCreateOpen(true)}>
        New Approval Policy
      </Button>

      {createOpen ? (
        <Card title={editingId ? "Edit Approval Policy" : "Create Approval Policy"}>
          <Space orientation="vertical" size="middle" style={{ width: "100%" }}>
            <Space orientation="vertical" style={{ width: "100%" }}>
              <label htmlFor="mcp-approval-policy-name">Policy Name</label>
              <input
                id="mcp-approval-policy-name"
                aria-label="Policy Name"
                value={name}
                onChange={(event) => setName(event.target.value)}
                placeholder="Outside Profile"
              />
            </Space>
            <Space>
              <Space orientation="vertical">
                <label htmlFor="mcp-approval-policy-scope">Owner Scope</label>
                <select
                  id="mcp-approval-policy-scope"
                  aria-label="Owner Scope"
                  value={ownerScopeType}
                  onChange={(event) => setOwnerScopeType(event.target.value as typeof ownerScopeType)}
                >
                  {MCP_HUB_SCOPE_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </Space>
              <Space orientation="vertical">
                <label htmlFor="mcp-approval-policy-mode">Approval Mode</label>
                <select
                  id="mcp-approval-policy-mode"
                  aria-label="Approval Mode"
                  value={mode}
                  onChange={(event) => setMode(event.target.value as McpHubApprovalMode)}
                >
                  {MCP_HUB_APPROVAL_MODE_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </Space>
            </Space>
            <fieldset
              aria-label="Duration Options"
              className="mcp-approval-policy-duration-options"
              style={{ border: 0, margin: 0, padding: 0 }}
            >
              <legend>Duration Options</legend>
              <Space orientation="vertical" size={4} style={{ width: "100%" }}>
                {MCP_HUB_APPROVAL_DURATION_OPTIONS.map((option) => (
                  <Checkbox
                    key={option.value}
                    checked={durationOptions.includes(option.value)}
                    onChange={(event) =>
                      setDurationOptions((prev) =>
                        toggleStringValue(prev, option.value, event.target.checked)
                      )
                    }
                  >
                    {option.label}
                  </Checkbox>
                ))}
              </Space>
            </fieldset>
            <Checkbox checked={isActive} onChange={(event) => setIsActive(event.target.checked)}>
              Active
            </Checkbox>
            <Space>
              <Button type="primary" onClick={handleSave} disabled={!canSave} loading={saving}>
                {editingId ? "Update Policy" : "Save Policy"}
              </Button>
              <Button onClick={resetForm}>Cancel</Button>
            </Space>
          </Space>
        </Card>
      ) : null}

      <List
        bordered
        loading={loading}
        dataSource={policies}
        locale={{ emptyText: <Empty description="No approval policies yet" /> }}
        renderItem={(policy) => (
          <List.Item>
            <Space orientation="vertical" size={4} style={{ width: "100%" }}>
              <Space wrap>
                <Typography.Text strong>{policy.name}</Typography.Text>
                <Tag>{policy.owner_scope_type}</Tag>
                <Tag color="gold">{policy.mode}</Tag>
                {policy.is_active ? <Tag color="green">active</Tag> : <Tag>inactive</Tag>}
                <Button size="small" onClick={() => openForEdit(policy)}>
                  Edit
                </Button>
                <Button size="small" danger onClick={() => void handleDelete(policy.id)}>
                  Delete
                </Button>
              </Space>
              <Space wrap>
                {Array.isArray(policy.rules?.duration_options)
                  ? policy.rules.duration_options.map((entry) => (
                      <Tag key={String(entry)} color="blue">
                        {String(entry)}
                      </Tag>
                    ))
                  : null}
              </Space>
            </Space>
          </List.Item>
        )}
      />
    </Space>
  )
}
