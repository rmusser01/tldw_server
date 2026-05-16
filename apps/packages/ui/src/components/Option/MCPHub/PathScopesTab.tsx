import { useEffect, useState } from "react"
import { Button, Card, Checkbox, Empty, List, Space, Tag, Typography } from "antd"
import { ProductStateAlert as Alert } from "@/components/Option/productStatePrimitives"

import {
  createPathScopeObject,
  deletePathScopeObject,
  listPathScopeObjects,
  updatePathScopeObject,
  type McpHubPathScopeObject,
  type McpHubPermissionPolicyDocument,
  type McpHubScopeType
} from "@/services/tldw/mcp-hub"

import {
  getPathAllowlistSummary,
  getPathScopeLabel,
  MCP_HUB_SCOPE_OPTIONS
} from "./policyHelpers"
import { PolicyDocumentEditor } from "./PolicyDocumentEditor"

export const PathScopesTab = () => {
  const [objects, setObjects] = useState<McpHubPathScopeObject[]>([])
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [createOpen, setCreateOpen] = useState(false)
  const [editingId, setEditingId] = useState<number | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [name, setName] = useState("")
  const [description, setDescription] = useState("")
  const [ownerScopeType, setOwnerScopeType] = useState<McpHubScopeType>("global")
  const [pathScopeDocument, setPathScopeDocument] = useState<McpHubPermissionPolicyDocument>({})
  const [isActive, setIsActive] = useState(true)

  const loadObjects = async () => {
    setLoading(true)
    setErrorMessage(null)
    try {
      const rows = await listPathScopeObjects()
      setObjects(Array.isArray(rows) ? rows : [])
    } catch {
      setObjects([])
      setErrorMessage("Failed to load path scopes.")
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void loadObjects()
  }, [])

  const resetForm = () => {
    setCreateOpen(false)
    setEditingId(null)
    setName("")
    setDescription("")
    setOwnerScopeType("global")
    setPathScopeDocument({})
    setIsActive(true)
  }

  const openForEdit = (pathScopeObject: McpHubPathScopeObject) => {
    setCreateOpen(true)
    setEditingId(pathScopeObject.id)
    setName(pathScopeObject.name)
    setDescription(String(pathScopeObject.description || ""))
    setOwnerScopeType(pathScopeObject.owner_scope_type)
    setPathScopeDocument(pathScopeObject.path_scope_document || {})
    setIsActive(pathScopeObject.is_active)
  }

  const handleSave = async () => {
    if (!name.trim() || saving) return
    setSaving(true)
    setErrorMessage(null)
    try {
      const payload = {
        name: name.trim(),
        description: description.trim() || null,
        owner_scope_type: ownerScopeType,
        path_scope_document: pathScopeDocument,
        is_active: isActive
      }
      if (editingId) {
        await updatePathScopeObject(editingId, payload)
      } else {
        await createPathScopeObject(payload)
      }
      resetForm()
      await loadObjects()
    } catch {
      setErrorMessage(editingId ? "Failed to update path scope." : "Failed to create path scope.")
    } finally {
      setSaving(false)
    }
  }

  const handleDelete = async (pathScopeObjectId: number) => {
    if (typeof window !== "undefined" && !window.confirm("Delete this path scope?")) {
      return
    }
    setErrorMessage(null)
    try {
      await deletePathScopeObject(pathScopeObjectId)
      await loadObjects()
    } catch {
      setErrorMessage("Failed to delete path scope.")
    }
  }

  return (
    <Space orientation="vertical" size="middle" style={{ width: "100%" }}>
      <Typography.Text type="secondary">
        Reusable path scopes define relative local-file rules once and let profiles or assignments attach
        them to trusted workspaces later.
      </Typography.Text>
      {errorMessage ? <Alert type="error" title={errorMessage} showIcon /> : null}

      <Button type="primary" onClick={() => setCreateOpen(true)}>
        New Path Scope
      </Button>

      {createOpen ? (
        <Card title={editingId ? "Edit Path Scope" : "Create Path Scope"}>
          <Space orientation="vertical" size="middle" style={{ width: "100%" }}>
            <Space orientation="vertical" style={{ width: "100%" }}>
              <label htmlFor="mcp-path-scope-name">Path Scope Name</label>
              <input
                id="mcp-path-scope-name"
                aria-label="Path Scope Name"
                value={name}
                onChange={(event) => setName(event.target.value)}
                placeholder="Docs Only"
              />
            </Space>
            <Space orientation="vertical" style={{ width: "100%" }}>
              <label htmlFor="mcp-path-scope-description">Description</label>
              <input
                id="mcp-path-scope-description"
                aria-label="Path Scope Description"
                value={description}
                onChange={(event) => setDescription(event.target.value)}
                placeholder="Reusable relative path rules for documentation work"
              />
            </Space>
            <Space orientation="vertical">
              <label htmlFor="mcp-path-scope-owner-scope">Owner Scope</label>
              <select
                id="mcp-path-scope-owner-scope"
                aria-label="Path Scope Owner Scope"
                value={ownerScopeType}
                onChange={(event) => setOwnerScopeType(event.target.value as McpHubScopeType)}
              >
                {MCP_HUB_SCOPE_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </Space>

            <PolicyDocumentEditor
              formId="mcp-path-scope"
              policy={pathScopeDocument}
              onChange={setPathScopeDocument}
              registryEntries={[]}
              registryModules={[]}
              pathScopeOnly
            />

            <Checkbox checked={isActive} onChange={(event) => setIsActive(event.target.checked)}>
              Active
            </Checkbox>
            <Space>
              <Button type="primary" onClick={() => void handleSave()} loading={saving}>
                {editingId ? "Update Path Scope" : "Save Path Scope"}
              </Button>
              <Button onClick={resetForm}>Cancel</Button>
            </Space>
          </Space>
        </Card>
      ) : null}

      <List
        bordered
        loading={loading}
        dataSource={objects}
        locale={{ emptyText: <Empty description="No path scopes yet" /> }}
        renderItem={(pathScopeObject) => (
          <List.Item>
            <Space orientation="vertical" size={4} style={{ width: "100%" }}>
              <Space wrap>
                <Typography.Text strong>{pathScopeObject.name}</Typography.Text>
                <Tag>{pathScopeObject.owner_scope_type}</Tag>
                {pathScopeObject.is_active ? <Tag color="green">active</Tag> : <Tag>inactive</Tag>}
                {getPathScopeLabel(pathScopeObject.path_scope_document.path_scope_mode) ? (
                  <Tag color="cyan">
                    {getPathScopeLabel(pathScopeObject.path_scope_document.path_scope_mode)}
                  </Tag>
                ) : null}
                <Button size="small" onClick={() => openForEdit(pathScopeObject)}>
                  Edit
                </Button>
                <Button size="small" danger onClick={() => void handleDelete(pathScopeObject.id)}>
                  Delete
                </Button>
              </Space>
              {pathScopeObject.description ? (
                <Typography.Text type="secondary">{pathScopeObject.description}</Typography.Text>
              ) : null}
              {getPathAllowlistSummary(pathScopeObject.path_scope_document.path_allowlist_prefixes) ? (
                <Typography.Text type="secondary">
                  {`Allowed paths: ${getPathAllowlistSummary(pathScopeObject.path_scope_document.path_allowlist_prefixes)}`}
                </Typography.Text>
              ) : null}
            </Space>
          </List.Item>
        )}
      />
    </Space>
  )
}
