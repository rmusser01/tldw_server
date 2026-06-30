import { useEffect, useMemo, useState } from "react"
import { Button, Card, Descriptions, Empty, Input, List, Space, Switch, Tag, Typography } from "antd"
import { ProductStateAlert as Alert } from "@/components/Option/productStatePrimitives"

import {
  createCapabilityAdapterMapping,
  listCapabilityAdapterMappings,
  previewCapabilityAdapterMapping,
  updateCapabilityAdapterMapping,
  type McpHubCapabilityAdapterMapping,
  type McpHubCapabilityAdapterMappingInput,
  type McpHubCapabilityAdapterMappingPreview,
  type McpHubCapabilityAdapterScopeType,
  type McpHubPermissionPolicyDocument
} from "@/services/tldw/mcp-hub"

const DEFAULT_POLICY_JSON = JSON.stringify(
  {
    allowed_tools: []
  },
  null,
  2
)

const parseLineList = (value: string) =>
  value
    .split(/[\n,]/)
    .map((entry) => entry.trim())
    .filter(Boolean)

export const CapabilityMappingsTab = () => {
  const [mappings, setMappings] = useState<McpHubCapabilityAdapterMapping[]>([])
  const [selectedMappingId, setSelectedMappingId] = useState<number | null>(null)
  const [mappingId, setMappingId] = useState("")
  const [title, setTitle] = useState("")
  const [description, setDescription] = useState("")
  const [ownerScopeType, setOwnerScopeType] = useState<McpHubCapabilityAdapterScopeType>("global")
  const [ownerScopeId, setOwnerScopeId] = useState("")
  const [capabilityName, setCapabilityName] = useState("")
  const [resolvedPolicyJson, setResolvedPolicyJson] = useState(DEFAULT_POLICY_JSON)
  const [supportedRequirementsText, setSupportedRequirementsText] = useState("")
  const [isActive, setIsActive] = useState(true)
  const [preview, setPreview] = useState<McpHubCapabilityAdapterMappingPreview | null>(null)
  const [loading, setLoading] = useState(false)
  const [previewing, setPreviewing] = useState(false)
  const [saving, setSaving] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [successMessage, setSuccessMessage] = useState<string | null>(null)

  const parsedResolvedPolicy = useMemo<McpHubPermissionPolicyDocument | null>(() => {
    try {
      const parsed = JSON.parse(resolvedPolicyJson) as McpHubPermissionPolicyDocument
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
        return null
      }
      return parsed
    } catch {
      return null
    }
  }, [resolvedPolicyJson])

  const loadMappings = async () => {
    setLoading(true)
    setErrorMessage(null)
    try {
      const rows = await listCapabilityAdapterMappings()
      const safeRows = Array.isArray(rows) ? rows : []
      setMappings(safeRows)
      setSelectedMappingId((current) => {
        if (safeRows.some((row) => row.id === current)) {
          return current
        }
        return safeRows[0]?.id ?? null
      })
    } catch {
      setMappings([])
      setSelectedMappingId(null)
      setErrorMessage("Failed to load capability mappings.")
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void loadMappings()
  }, [])

  useEffect(() => {
    const selected = mappings.find((row) => row.id === selectedMappingId)
    if (!selected) {
      return
    }
    setMappingId(selected.mapping_id)
    setTitle(selected.title)
    setDescription(String(selected.description || ""))
    setOwnerScopeType(selected.owner_scope_type)
    setOwnerScopeId(
      selected.owner_scope_id === null || selected.owner_scope_id === undefined
        ? ""
        : String(selected.owner_scope_id)
    )
    setCapabilityName(selected.capability_name)
    setResolvedPolicyJson(JSON.stringify(selected.resolved_policy_document || {}, null, 2))
    setSupportedRequirementsText((selected.supported_environment_requirements || []).join("\n"))
    setIsActive(selected.is_active)
    setPreview(null)
    setSuccessMessage(null)
  }, [mappings, selectedMappingId])

  const resetForm = () => {
    setSelectedMappingId(null)
    setMappingId("")
    setTitle("")
    setDescription("")
    setOwnerScopeType("global")
    setOwnerScopeId("")
    setCapabilityName("")
    setResolvedPolicyJson(DEFAULT_POLICY_JSON)
    setSupportedRequirementsText("")
    setIsActive(true)
    setPreview(null)
    setErrorMessage(null)
    setSuccessMessage(null)
  }

  const buildPayload = (): {
    payload: McpHubCapabilityAdapterMappingInput | null
    validationError: string | null
  } => {
    if (!mappingId.trim() || !capabilityName.trim() || !parsedResolvedPolicy) {
      return {
        payload: null,
        validationError: "Mapping ID, capability name, and valid policy JSON are required."
      }
    }

    const trimmedOwnerScopeId = ownerScopeId.trim()
    let numericOwnerScopeId: number | null = null

    if (trimmedOwnerScopeId) {
      if (!/^\d+$/.test(trimmedOwnerScopeId)) {
        return {
          payload: null,
          validationError: "Owner Scope ID must be a valid integer."
        }
      }
      numericOwnerScopeId = Number.parseInt(trimmedOwnerScopeId, 10)
    }

    return {
      payload: {
        mapping_id: mappingId.trim(),
        title: title.trim() || null,
        description: description.trim() || null,
        owner_scope_type: ownerScopeType,
        owner_scope_id: numericOwnerScopeId,
        capability_name: capabilityName.trim(),
        adapter_contract_version: 1,
        resolved_policy_document: parsedResolvedPolicy,
        supported_environment_requirements: parseLineList(supportedRequirementsText),
        is_active: isActive
      },
      validationError: null
    }
  }

  const handlePreview = async () => {
    setSuccessMessage(null)
    const { payload, validationError } = buildPayload()
    if (!payload) {
      setErrorMessage(validationError)
      setPreview(null)
      return
    }
    setPreviewing(true)
    setErrorMessage(null)
    try {
      setPreview(await previewCapabilityAdapterMapping(payload))
    } catch {
      setPreview(null)
      setErrorMessage("Failed to preview capability mapping.")
    } finally {
      setPreviewing(false)
    }
  }

  const handleSave = async () => {
    setSuccessMessage(null)
    const { payload, validationError } = buildPayload()
    if (!payload) {
      setErrorMessage(validationError)
      return
    }
    setSaving(true)
    setErrorMessage(null)
    try {
      if (selectedMappingId) {
        await updateCapabilityAdapterMapping(selectedMappingId, payload)
        setSuccessMessage(`Updated ${payload.mapping_id}.`)
      } else {
        const created = await createCapabilityAdapterMapping(payload)
        setSelectedMappingId(created.id)
        setSuccessMessage(`Saved ${payload.mapping_id}.`)
      }
      await loadMappings()
    } catch {
      setErrorMessage(selectedMappingId ? "Failed to update capability mapping." : "Failed to save capability mapping.")
    } finally {
      setSaving(false)
    }
  }

  return (
    <Space orientation="vertical" size="middle" style={{ width: "100%" }}>
      <Typography.Text type="secondary">
        Capability mappings turn portable capability names into concrete local MCP Hub policy effects.
      </Typography.Text>
      {errorMessage ? <Alert type="error" title={errorMessage} showIcon /> : null}
      {successMessage ? <Alert type="success" title={successMessage} showIcon /> : null}

      <Space align="start" size="middle" style={{ width: "100%", justifyContent: "space-between" }}>
        <Card title="Mappings" style={{ flex: 1, minWidth: 320 }}>
          <Space orientation="vertical" size="middle" style={{ width: "100%" }}>
            <Button onClick={resetForm}>New Mapping</Button>
            <List
              bordered
              loading={loading}
              dataSource={mappings}
              locale={{ emptyText: <Empty description="No capability mappings configured yet" /> }}
              renderItem={(mapping) => (
                <List.Item
                  style={{
                    cursor: "pointer",
                    borderColor:
                      mapping.id === selectedMappingId ? "var(--ant-color-primary, #1677ff)" : undefined
                  }}
                  onClick={() => setSelectedMappingId(mapping.id)}
                >
                  <Space orientation="vertical" size={4} style={{ width: "100%" }}>
                    <Space wrap>
                      <Typography.Text strong>{mapping.title || mapping.mapping_id}</Typography.Text>
                      <Tag>{mapping.owner_scope_type}</Tag>
                      <Tag color="blue">{mapping.capability_name}</Tag>
                    </Space>
                    <Typography.Text type="secondary">{mapping.mapping_id}</Typography.Text>
                  </Space>
                </List.Item>
              )}
            />
          </Space>
        </Card>

        <Card title={selectedMappingId ? "Edit Mapping" : "Create Mapping"} style={{ flex: 1, minWidth: 360 }}>
          <Space orientation="vertical" size="middle" style={{ width: "100%" }}>
            <Space orientation="vertical" style={{ width: "100%" }}>
              <label htmlFor="mcp-capability-mapping-id">Mapping ID</label>
              <Input
                id="mcp-capability-mapping-id"
                aria-label="Mapping ID"
                value={mappingId}
                onChange={(event) => setMappingId(event.target.value)}
                placeholder="research.global"
              />
            </Space>
            <Space orientation="vertical" style={{ width: "100%" }}>
              <label htmlFor="mcp-capability-mapping-title">Title</label>
              <Input
                id="mcp-capability-mapping-title"
                aria-label="Title"
                value={title}
                onChange={(event) => setTitle(event.target.value)}
                placeholder="Research Mapping"
              />
            </Space>
            <Space orientation="vertical" style={{ width: "100%" }}>
              <label htmlFor="mcp-capability-mapping-description">Description</label>
              <Input
                id="mcp-capability-mapping-description"
                aria-label="Description"
                value={description}
                onChange={(event) => setDescription(event.target.value)}
                placeholder="Maps portable research access to local search tools"
              />
            </Space>
            <Space wrap>
              <Space orientation="vertical">
                <label htmlFor="mcp-capability-mapping-scope">Owner Scope</label>
                <select
                  id="mcp-capability-mapping-scope"
                  aria-label="Owner Scope"
                  value={ownerScopeType}
                  onChange={(event) =>
                    setOwnerScopeType(event.target.value as McpHubCapabilityAdapterScopeType)
                  }
                >
                  <option value="global">Global</option>
                  <option value="org">Organization</option>
                  <option value="team">Team</option>
                </select>
              </Space>
              <Space orientation="vertical">
                <label htmlFor="mcp-capability-mapping-scope-id">Owner Scope ID</label>
                <Input
                  id="mcp-capability-mapping-scope-id"
                  aria-label="Owner Scope ID"
                  value={ownerScopeId}
                  inputMode="numeric"
                  onChange={(event) => setOwnerScopeId(event.target.value)}
                  placeholder="Optional"
                />
              </Space>
            </Space>
            <Space orientation="vertical" style={{ width: "100%" }}>
              <label htmlFor="mcp-capability-mapping-capability">Capability Name</label>
              <Input
                id="mcp-capability-mapping-capability"
                aria-label="Capability Name"
                value={capabilityName}
                onChange={(event) => setCapabilityName(event.target.value)}
                placeholder="tool.invoke.research"
              />
            </Space>
            <Space orientation="vertical" style={{ width: "100%" }}>
              <label htmlFor="mcp-capability-mapping-requirements">Supported Environment Requirements</label>
              <Input.TextArea
                id="mcp-capability-mapping-requirements"
                aria-label="Supported Environment Requirements"
                value={supportedRequirementsText}
                rows={3}
                onChange={(event) => setSupportedRequirementsText(event.target.value)}
                placeholder="workspace_bounded_read"
              />
            </Space>
            <Space orientation="vertical" style={{ width: "100%" }}>
              <label htmlFor="mcp-capability-mapping-policy">Resolved Policy JSON</label>
              <Input.TextArea
                id="mcp-capability-mapping-policy"
                aria-label="Resolved Policy JSON"
                value={resolvedPolicyJson}
                rows={10}
                onChange={(event) => setResolvedPolicyJson(event.target.value)}
                spellCheck={false}
              />
            </Space>
            <Space align="center">
              <Typography.Text>Active</Typography.Text>
              <Switch checked={isActive} onChange={setIsActive} aria-label="Active" />
            </Space>
            <Space>
              <Button type="primary" onClick={() => void handlePreview()} loading={previewing}>
                Preview Mapping
              </Button>
              <Button onClick={() => void handleSave()} loading={saving}>
                Save Mapping
              </Button>
            </Space>
          </Space>
        </Card>
      </Space>

      {preview ? (
        <Card title="Preview Result">
          <Descriptions bordered column={1} size="small">
            <Descriptions.Item label="Scope">
              {preview.affected_scope_summary.display_scope}
            </Descriptions.Item>
            <Descriptions.Item label="Capability">
              {preview.normalized_mapping.capability_name}
            </Descriptions.Item>
            <Descriptions.Item label="Warnings">
              {preview.warnings.length ? (
                <Space wrap>
                  {preview.warnings.map((warning) => (
                    <Tag key={warning} color="orange">
                      {warning}
                    </Tag>
                  ))}
                </Space>
              ) : (
                <Typography.Text type="secondary">None</Typography.Text>
              )}
            </Descriptions.Item>
            <Descriptions.Item label="Resolved Effects">
              <pre style={{ margin: 0, whiteSpace: "pre-wrap" }}>
                {JSON.stringify(preview.normalized_mapping.resolved_policy_document, null, 2)}
              </pre>
            </Descriptions.Item>
          </Descriptions>
        </Card>
      ) : null}
    </Space>
  )
}
