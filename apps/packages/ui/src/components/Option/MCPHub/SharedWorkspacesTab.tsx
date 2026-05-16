import { useEffect, useRef, useState } from "react"
import { Button, Card, Empty, List, Space, Tag, Typography } from "antd"
import { ProductStateAlert as Alert } from "@/components/Option/productStatePrimitives"

import {
  createSharedWorkspace,
  deleteSharedWorkspace,
  listSharedWorkspaces,
  type McpHubDrillTarget,
  updateSharedWorkspace,
  type McpHubSharedWorkspace
} from "@/services/tldw/mcp-hub"

type SharedWorkspacesTabProps = {
  drillTarget?: McpHubDrillTarget | null
  onDrillHandled?: (requestId: number) => void
}

export const SharedWorkspacesTab = ({
  drillTarget = null,
  onDrillHandled
}: SharedWorkspacesTabProps) => {
  const handledDrillRequestRef = useRef<number | null>(null)
  const [entries, setEntries] = useState<McpHubSharedWorkspace[]>([])
  const [entriesLoaded, setEntriesLoaded] = useState(false)
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [createOpen, setCreateOpen] = useState(false)
  const [editingId, setEditingId] = useState<number | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [workspaceId, setWorkspaceId] = useState("")
  const [displayName, setDisplayName] = useState("")
  const [absoluteRoot, setAbsoluteRoot] = useState("")
  const [ownerScopeType, setOwnerScopeType] = useState<"global" | "org" | "team">("team")
  const [ownerScopeId, setOwnerScopeId] = useState("")
  const [isActive, setIsActive] = useState(true)

  const loadEntries = async () => {
    setLoading(true)
    setErrorMessage(null)
    try {
      const rows = await listSharedWorkspaces()
      setEntries(Array.isArray(rows) ? rows : [])
    } catch {
      setEntries([])
      setErrorMessage("Failed to load shared workspaces.")
    } finally {
      setLoading(false)
      setEntriesLoaded(true)
    }
  }

  useEffect(() => {
    void loadEntries()
  }, [])

  useEffect(() => {
    if (
      !drillTarget ||
      drillTarget.tab !== "shared-workspaces" ||
      drillTarget.object_kind !== "shared_workspace"
    ) {
      return
    }
    if (
      handledDrillRequestRef.current === drillTarget.request_id ||
      loading ||
      !entriesLoaded
    ) {
      return
    }
    const entry = entries.find((row) => String(row.id) === String(drillTarget.object_id))
    if (entry) {
      handledDrillRequestRef.current = drillTarget.request_id
      openForEdit(entry)
      onDrillHandled?.(drillTarget.request_id)
    }
  }, [drillTarget, entries, entriesLoaded, loading, onDrillHandled])

  const resetForm = () => {
    setCreateOpen(false)
    setEditingId(null)
    setWorkspaceId("")
    setDisplayName("")
    setAbsoluteRoot("")
    setOwnerScopeType("team")
    setOwnerScopeId("")
    setIsActive(true)
  }

  const openForEdit = (entry: McpHubSharedWorkspace) => {
    setCreateOpen(true)
    setEditingId(entry.id)
    setWorkspaceId(entry.workspace_id)
    setDisplayName(entry.display_name)
    setAbsoluteRoot(entry.absolute_root)
    setOwnerScopeType(entry.owner_scope_type)
    setOwnerScopeId(entry.owner_scope_id ? String(entry.owner_scope_id) : "")
    setIsActive(entry.is_active)
  }

  const handleSave = async () => {
    if (!workspaceId.trim() || !displayName.trim() || !absoluteRoot.trim() || saving) return
    setSaving(true)
    setErrorMessage(null)
    try {
      const payload = {
        workspace_id: workspaceId.trim(),
        display_name: displayName.trim(),
        absolute_root: absoluteRoot.trim(),
        owner_scope_type: ownerScopeType,
        owner_scope_id: ownerScopeType === "global" ? null : Number(ownerScopeId || 0) || null,
        is_active: isActive
      }
      if (editingId) {
        await updateSharedWorkspace(editingId, payload)
      } else {
        await createSharedWorkspace(payload)
      }
      resetForm()
      await loadEntries()
    } catch {
      setErrorMessage(editingId ? "Failed to update shared workspace." : "Failed to create shared workspace.")
    } finally {
      setSaving(false)
    }
  }

  const handleDelete = async (sharedWorkspaceId: number) => {
    if (typeof window !== "undefined" && !window.confirm("Delete this shared workspace?")) {
      return
    }
    setErrorMessage(null)
    try {
      await deleteSharedWorkspace(sharedWorkspaceId)
      await loadEntries()
    } catch {
      setErrorMessage("Failed to delete shared workspace.")
    }
  }

  return (
    <Space orientation="vertical" size="middle" style={{ width: "100%" }}>
      <Typography.Text type="secondary">
        Shared workspaces are admin-managed trusted absolute roots for team, org, and global path
        policies. Workspace sets can reference their ids without depending on user-local workspace
        mappings.
      </Typography.Text>
      {errorMessage ? <Alert type="error" title={errorMessage} showIcon /> : null}

      <Button type="primary" onClick={() => setCreateOpen(true)}>
        New Shared Workspace
      </Button>

      {createOpen ? (
        <Card title={editingId ? "Edit Shared Workspace" : "Create Shared Workspace"}>
          <Space orientation="vertical" size="middle" style={{ width: "100%" }}>
            <Space orientation="vertical" style={{ width: "100%" }}>
              <label htmlFor="mcp-shared-workspace-id">Workspace Id</label>
              <input
                id="mcp-shared-workspace-id"
                aria-label="Shared Workspace Id"
                value={workspaceId}
                onChange={(event) => setWorkspaceId(event.target.value)}
                placeholder="shared-docs"
              />
            </Space>
            <Space orientation="vertical" style={{ width: "100%" }}>
              <label htmlFor="mcp-shared-workspace-display-name">Display Name</label>
              <input
                id="mcp-shared-workspace-display-name"
                aria-label="Shared Workspace Display Name"
                value={displayName}
                onChange={(event) => setDisplayName(event.target.value)}
                placeholder="Shared Docs"
              />
            </Space>
            <Space orientation="vertical" style={{ width: "100%" }}>
              <label htmlFor="mcp-shared-workspace-absolute-root">Absolute Root</label>
              <input
                id="mcp-shared-workspace-absolute-root"
                aria-label="Shared Workspace Absolute Root"
                value={absoluteRoot}
                onChange={(event) => setAbsoluteRoot(event.target.value)}
                placeholder="/srv/shared/docs"
              />
            </Space>
            <Space>
              <Space orientation="vertical">
                <label htmlFor="mcp-shared-workspace-scope-type">Owner Scope</label>
                <select
                  id="mcp-shared-workspace-scope-type"
                  aria-label="Shared Workspace Owner Scope"
                  value={ownerScopeType}
                  onChange={(event) =>
                    setOwnerScopeType(event.target.value as "global" | "org" | "team")
                  }
                >
                  <option value="team">team</option>
                  <option value="org">org</option>
                  <option value="global">global</option>
                </select>
              </Space>
              {ownerScopeType !== "global" ? (
                <Space orientation="vertical">
                  <label htmlFor="mcp-shared-workspace-scope-id">Owner Scope Id</label>
                  <input
                    id="mcp-shared-workspace-scope-id"
                    aria-label="Shared Workspace Owner Scope Id"
                    value={ownerScopeId}
                    onChange={(event) => setOwnerScopeId(event.target.value)}
                    placeholder={ownerScopeType === "team" ? "21" : "9"}
                  />
                </Space>
              ) : null}
            </Space>
            <label>
              <input
                type="checkbox"
                checked={isActive}
                onChange={(event) => setIsActive(event.target.checked)}
              />
              <span style={{ marginLeft: 8 }}>Active</span>
            </label>
            <Space>
              <Button type="primary" onClick={() => void handleSave()} loading={saving}>
                {editingId ? "Update Shared Workspace" : "Save Shared Workspace"}
              </Button>
              <Button onClick={resetForm}>Cancel</Button>
            </Space>
          </Space>
        </Card>
      ) : null}

      <List
        bordered
        loading={loading}
        dataSource={entries}
        locale={{ emptyText: <Empty description="No shared workspaces yet" /> }}
        renderItem={(entry) => (
          <List.Item>
            <Space orientation="vertical" size={4} style={{ width: "100%" }}>
              <Space wrap>
                <Typography.Text strong>{entry.display_name}</Typography.Text>
                <Tag>{entry.workspace_id}</Tag>
                <Tag>{entry.owner_scope_type}</Tag>
                {entry.owner_scope_id ? <Tag>{`scope ${entry.owner_scope_id}`}</Tag> : null}
                {entry.is_active ? <Tag color="green">active</Tag> : <Tag>inactive</Tag>}
                <Button size="small" onClick={() => openForEdit(entry)}>
                  Edit
                </Button>
                <Button size="small" danger onClick={() => void handleDelete(entry.id)}>
                  Delete
                </Button>
              </Space>
              <Typography.Text type="secondary">{entry.absolute_root}</Typography.Text>
              {entry.readiness_summary && !entry.readiness_summary.is_multi_root_ready ? (
                <Alert
                  type="warning"
                  showIcon
                  title={
                    entry.readiness_summary.warning_message ||
                    "May conflict with other visible shared roots in multi-root assignments."
                  }
                  description={[
                    entry.readiness_summary.conflicting_workspace_ids?.length
                      ? `Workspaces: ${entry.readiness_summary.conflicting_workspace_ids.join(", ")}`
                      : null,
                    entry.readiness_summary.conflicting_workspace_roots?.length
                      ? `Roots: ${entry.readiness_summary.conflicting_workspace_roots.join(", ")}`
                      : null,
                  ]
                    .filter(Boolean)
                    .join(" ")}
                />
              ) : null}
            </Space>
          </List.Item>
        )}
      />
    </Space>
  )
}
