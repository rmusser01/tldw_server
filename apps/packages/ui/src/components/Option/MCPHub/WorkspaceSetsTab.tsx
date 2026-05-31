import { useEffect, useRef, useState } from "react"
import { Button, Card, Empty, List, Space, Tag, Typography } from "antd"
import { ProductStateAlert as Alert } from "@/components/Option/productStatePrimitives"

import {
  addWorkspaceSetMember,
  createWorkspaceSetObject,
  deleteWorkspaceSetMember,
  deleteWorkspaceSetObject,
  listSharedWorkspaces,
  listWorkspaceSetMembers,
  listWorkspaceSetObjects,
  updateWorkspaceSetObject,
  type McpHubDrillTarget,
  type McpHubSharedWorkspace,
  type McpHubWorkspaceSetObject,
  type McpHubWorkspaceSetObjectMember
} from "@/services/tldw/mcp-hub"

import { parseLineList } from "./policyHelpers"

type WorkspaceSetsTabProps = {
  drillTarget?: McpHubDrillTarget | null
  focusSource?: string | null
  focusWorkspaceId?: string | null
  onDrillHandled?: (requestId: number) => void
}

export const WorkspaceSetsTab = ({
  drillTarget = null,
  focusSource = null,
  focusWorkspaceId = null,
  onDrillHandled
}: WorkspaceSetsTabProps) => {
  const handledDrillRequestRef = useRef<number | null>(null)
  const [objects, setObjects] = useState<McpHubWorkspaceSetObject[]>([])
  const [objectsLoaded, setObjectsLoaded] = useState(false)
  const [membersByObjectId, setMembersByObjectId] = useState<Record<number, McpHubWorkspaceSetObjectMember[]>>({})
  const [sharedEntries, setSharedEntries] = useState<McpHubSharedWorkspace[]>([])
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [createOpen, setCreateOpen] = useState(false)
  const [editingId, setEditingId] = useState<number | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [name, setName] = useState("")
  const [description, setDescription] = useState("")
  const [ownerScopeType, setOwnerScopeType] = useState<"user" | "team" | "org" | "global">("user")
  const [ownerScopeId, setOwnerScopeId] = useState("")
  const [workspaceIdsText, setWorkspaceIdsText] = useState("")
  const [selectedSharedWorkspaceIds, setSelectedSharedWorkspaceIds] = useState<string[]>([])
  const [isActive, setIsActive] = useState(true)

  const loadObjects = async () => {
    setLoading(true)
    setErrorMessage(null)
    try {
      const [rows, sharedWorkspaceRows] = await Promise.all([
        listWorkspaceSetObjects(),
        listSharedWorkspaces()
      ])
      const nextObjects = Array.isArray(rows) ? rows : []
      setObjects(nextObjects)
      setSharedEntries(Array.isArray(sharedWorkspaceRows) ? sharedWorkspaceRows : [])
      const memberEntries = await Promise.all(
        nextObjects.map(async (workspaceSet) => [
          workspaceSet.id,
          await listWorkspaceSetMembers(workspaceSet.id)
        ] as const)
      )
      setMembersByObjectId(
        Object.fromEntries(
          memberEntries.map(([workspaceSetObjectId, members]) => [
            workspaceSetObjectId,
            Array.isArray(members) ? members : []
          ])
        )
      )
    } catch {
      setObjects([])
      setMembersByObjectId({})
      setSharedEntries([])
      setErrorMessage("Failed to load workspace sets.")
    } finally {
      setLoading(false)
      setObjectsLoaded(true)
    }
  }

  useEffect(() => {
    void loadObjects()
  }, [])

  useEffect(() => {
    if (
      !drillTarget ||
      drillTarget.tab !== "workspace-sets" ||
      drillTarget.object_kind !== "workspace_set_object"
    ) {
      return
    }
    if (
      handledDrillRequestRef.current === drillTarget.request_id ||
      loading ||
      !objectsLoaded
    ) {
      return
    }
    const workspaceSet = objects.find(
      (row) => String(row.id) === String(drillTarget.object_id)
    )
    if (workspaceSet) {
      handledDrillRequestRef.current = drillTarget.request_id
      openForEdit(workspaceSet)
      onDrillHandled?.(drillTarget.request_id)
    }
  }, [drillTarget, loading, objects, objectsLoaded, onDrillHandled])

  const resetForm = () => {
    setCreateOpen(false)
    setEditingId(null)
    setName("")
    setDescription("")
    setOwnerScopeType("user")
    setOwnerScopeId("")
    setWorkspaceIdsText("")
    setSelectedSharedWorkspaceIds([])
    setIsActive(true)
  }

  const openForEdit = (workspaceSet: McpHubWorkspaceSetObject) => {
    setCreateOpen(true)
    setEditingId(workspaceSet.id)
    setName(workspaceSet.name)
    setDescription(String(workspaceSet.description || ""))
    setOwnerScopeType(
      (workspaceSet.owner_scope_type as "user" | "team" | "org" | "global") || "user"
    )
    setOwnerScopeId(workspaceSet.owner_scope_id ? String(workspaceSet.owner_scope_id) : "")
    setWorkspaceIdsText(
      (membersByObjectId[workspaceSet.id] || []).map((member) => member.workspace_id).join("\n")
    )
    setSelectedSharedWorkspaceIds(
      (membersByObjectId[workspaceSet.id] || []).map((member) => member.workspace_id)
    )
    setIsActive(workspaceSet.is_active)
  }

  const syncMembers = async (workspaceSetObjectId: number) => {
    const desiredWorkspaceIds = Array.from(
      new Set(
        ownerScopeType === "user" ? parseLineList(workspaceIdsText) : selectedSharedWorkspaceIds
      )
    )
    const currentWorkspaceIds = (membersByObjectId[workspaceSetObjectId] || []).map(
      (member) => member.workspace_id
    )
    const toAdd = desiredWorkspaceIds.filter((workspaceId) => !currentWorkspaceIds.includes(workspaceId))
    const toDelete = currentWorkspaceIds.filter((workspaceId) => !desiredWorkspaceIds.includes(workspaceId))

    await Promise.all([
      ...toAdd.map((workspaceId) => addWorkspaceSetMember(workspaceSetObjectId, workspaceId)),
      ...toDelete.map((workspaceId) => deleteWorkspaceSetMember(workspaceSetObjectId, workspaceId))
    ])
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
        owner_scope_id: ownerScopeType === "global" ? null : Number(ownerScopeId || 0) || null,
        is_active: isActive
      }
      let workspaceSetObjectId = editingId
      if (editingId) {
        const updated = await updateWorkspaceSetObject(editingId, payload)
        workspaceSetObjectId = updated.id
      } else {
        const created = await createWorkspaceSetObject(payload)
        workspaceSetObjectId = created.id
      }
      if (workspaceSetObjectId) {
        await syncMembers(workspaceSetObjectId)
      }
      resetForm()
      await loadObjects()
    } catch {
      setErrorMessage(editingId ? "Failed to update workspace set." : "Failed to create workspace set.")
    } finally {
      setSaving(false)
    }
  }

  const handleDelete = async (workspaceSetObjectId: number) => {
    if (typeof window !== "undefined" && !window.confirm("Delete this workspace set?")) {
      return
    }
    setErrorMessage(null)
    try {
      await deleteWorkspaceSetObject(workspaceSetObjectId)
      await loadObjects()
    } catch {
      setErrorMessage("Failed to delete workspace set.")
    }
  }

  const normalizedFocusWorkspaceId = String(focusWorkspaceId ?? "").trim()
  const matchingWorkspaceSets = normalizedFocusWorkspaceId
    ? objects.filter((workspaceSet) =>
        (membersByObjectId[workspaceSet.id] || []).some(
          (member) => member.workspace_id === normalizedFocusWorkspaceId
        )
      )
    : []
  const shouldShowFocusState =
    Boolean(normalizedFocusWorkspaceId) && objectsLoaded && !loading && !errorMessage
  const focusTitle =
    matchingWorkspaceSets.length === 1
      ? `${normalizedFocusWorkspaceId} is included in 1 MCP workspace set.`
      : `${normalizedFocusWorkspaceId} is included in ${matchingWorkspaceSets.length} MCP workspace sets.`

  return (
    <Space orientation="vertical" size="middle" style={{ width: "100%" }}>
      <Typography.Text type="secondary">
        Reusable workspace sets let assignments share the same trusted workspace membership without
        duplicating inline ids.
      </Typography.Text>
      {errorMessage ? <Alert type="error" title={errorMessage} showIcon /> : null}
      {shouldShowFocusState && matchingWorkspaceSets.length > 0 ? (
        <Alert
          data-testid="mcp-workspace-context-status"
          type="info"
          showIcon
          title={focusTitle}
          description={
            focusSource === "research-workspace"
              ? "This context came from Research Workspace. MCP Hub owns the workspace-set membership and policy assignment."
              : "MCP Hub owns workspace-set membership and policy assignment."
          }
        />
      ) : null}
      {shouldShowFocusState && matchingWorkspaceSets.length === 0 ? (
        <Alert
          data-testid="mcp-workspace-context-status"
          type="warning"
          showIcon
          title={`No MCP workspace set includes ${normalizedFocusWorkspaceId} yet.`}
          description="Create or edit a workspace set here to make this workspace available to MCP policy assignments."
        />
      ) : null}

      <Button type="primary" onClick={() => setCreateOpen(true)}>
        New Workspace Set
      </Button>

      {createOpen ? (
        <Card title={editingId ? "Edit Workspace Set" : "Create Workspace Set"}>
          <Space orientation="vertical" size="middle" style={{ width: "100%" }}>
            <Space orientation="vertical" style={{ width: "100%" }}>
              <label htmlFor="mcp-workspace-set-name">Workspace Set Name</label>
              <input
                id="mcp-workspace-set-name"
                aria-label="Workspace Set Name"
                value={name}
                onChange={(event) => setName(event.target.value)}
                placeholder="Primary Workspace Set"
              />
            </Space>
            <Space orientation="vertical" style={{ width: "100%" }}>
              <label htmlFor="mcp-workspace-set-description">Description</label>
              <input
                id="mcp-workspace-set-description"
                aria-label="Workspace Set Description"
                value={description}
                onChange={(event) => setDescription(event.target.value)}
                placeholder="Reusable trusted workspace membership for research personas"
              />
            </Space>
            <Space>
              <Space orientation="vertical">
                <label htmlFor="mcp-workspace-set-scope-type">Owner Scope</label>
                <select
                  id="mcp-workspace-set-scope-type"
                  aria-label="Workspace Set Owner Scope"
                  value={ownerScopeType}
                  onChange={(event) =>
                    setOwnerScopeType(event.target.value as "user" | "team" | "org" | "global")
                  }
                >
                  <option value="user">user</option>
                  <option value="team">team</option>
                  <option value="org">org</option>
                  <option value="global">global</option>
                </select>
              </Space>
              {ownerScopeType !== "global" ? (
                <Space orientation="vertical">
                  <label htmlFor="mcp-workspace-set-scope-id">Owner Scope Id</label>
                  <input
                    id="mcp-workspace-set-scope-id"
                    aria-label="Workspace Set Owner Scope Id"
                    value={ownerScopeId}
                    onChange={(event) => setOwnerScopeId(event.target.value)}
                    placeholder={ownerScopeType === "user" ? "7" : ownerScopeType === "team" ? "21" : "9"}
                  />
                </Space>
              ) : null}
            </Space>
            {ownerScopeType === "user" ? (
              <Space orientation="vertical" style={{ width: "100%" }}>
                <label htmlFor="mcp-workspace-set-workspace-ids">Workspace ids</label>
                <textarea
                  id="mcp-workspace-set-workspace-ids"
                  aria-label="Workspace ids"
                  value={workspaceIdsText}
                  onChange={(event) => setWorkspaceIdsText(event.target.value)}
                  rows={4}
                />
                <Typography.Text type="secondary">
                  One trusted workspace id per line. Only server-known workspaces for the current user are
                  accepted.
                </Typography.Text>
              </Space>
            ) : (
              <Space orientation="vertical" style={{ width: "100%" }}>
                <label htmlFor="mcp-workspace-set-shared-workspaces">Shared workspace ids</label>
                <select
                  id="mcp-workspace-set-shared-workspaces"
                  aria-label="Shared workspace ids"
                  multiple
                  value={selectedSharedWorkspaceIds}
                  onChange={(event) =>
                    setSelectedSharedWorkspaceIds(
                      Array.from(event.currentTarget.selectedOptions).map((option) => option.value)
                    )
                  }
                  size={Math.max(4, Math.min(sharedEntries.length, 8))}
                >
                  {sharedEntries.map((entry) => (
                    <option key={entry.id} value={entry.workspace_id}>
                      {`${entry.display_name} (${entry.workspace_id})`}
                    </option>
                  ))}
                </select>
                <Typography.Text type="secondary">
                  Shared-scope workspace sets select from the admin-managed shared registry.
                </Typography.Text>
              </Space>
            )}
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
                {editingId ? "Update Workspace Set" : "Save Workspace Set"}
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
        locale={{ emptyText: <Empty description="No workspace sets yet" /> }}
        renderItem={(workspaceSet) => (
          <List.Item>
            <Space orientation="vertical" size={4} style={{ width: "100%" }}>
              <Space wrap>
                <Typography.Text strong>{workspaceSet.name}</Typography.Text>
                <Tag>{workspaceSet.owner_scope_type}</Tag>
                {workspaceSet.owner_scope_id ? <Tag>{`scope ${workspaceSet.owner_scope_id}`}</Tag> : null}
                {workspaceSet.is_active ? <Tag color="green">active</Tag> : <Tag>inactive</Tag>}
                <Button size="small" onClick={() => openForEdit(workspaceSet)}>
                  Edit
                </Button>
                <Button size="small" danger onClick={() => void handleDelete(workspaceSet.id)}>
                  Delete
                </Button>
              </Space>
              {workspaceSet.description ? (
                <Typography.Text type="secondary">{workspaceSet.description}</Typography.Text>
              ) : null}
              {workspaceSet.readiness_summary && !workspaceSet.readiness_summary.is_multi_root_ready ? (
                <Alert
                  type="warning"
                  showIcon
                  title={
                    workspaceSet.readiness_summary.warning_message ||
                    "This workspace source is not currently multi-root-ready."
                  }
                  description={[
                    workspaceSet.readiness_summary.conflicting_workspace_ids?.length
                      ? `Workspaces: ${workspaceSet.readiness_summary.conflicting_workspace_ids.join(", ")}`
                      : null,
                    workspaceSet.readiness_summary.conflicting_workspace_roots?.length
                      ? `Roots: ${workspaceSet.readiness_summary.conflicting_workspace_roots.join(", ")}`
                      : null,
                    workspaceSet.readiness_summary.unresolved_workspace_ids?.length
                      ? `Unresolved workspaces: ${workspaceSet.readiness_summary.unresolved_workspace_ids.join(", ")}`
                      : null,
                  ]
                    .filter(Boolean)
                    .join(" ")}
                />
              ) : null}
              {(membersByObjectId[workspaceSet.id] || []).length > 0 ? (
                <Space wrap size={4}>
                  {(membersByObjectId[workspaceSet.id] || []).map((member) => (
                    <Tag key={`${workspaceSet.id}-${member.workspace_id}`} color="purple">
                      {member.workspace_id}
                    </Tag>
                  ))}
                </Space>
              ) : (
                <Typography.Text type="secondary">No workspace ids configured yet.</Typography.Text>
              )}
            </Space>
          </List.Item>
        )}
      />
    </Space>
  )
}
