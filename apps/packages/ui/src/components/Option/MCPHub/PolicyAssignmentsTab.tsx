import { useEffect, useMemo, useRef, useState } from "react"
import { Button, Card, Checkbox, Empty, List, Modal, Space, Tag, Typography } from "antd"
import { ProductStateAlert as Alert } from "@/components/Option/productStatePrimitives"

import {
  createPolicyAssignment,
  addPolicyAssignmentWorkspace,
  deletePolicyAssignmentOverride,
  deletePolicyAssignment,
  deleteAssignmentCredentialBinding,
  deletePolicyAssignmentWorkspace,
  getAssignmentExternalAccess,
  getToolRegistrySummary,
  getEffectivePolicy,
  getPolicyAssignmentOverride,
  listPathScopeObjects,
  listWorkspaceSetObjects,
  listPolicyAssignmentWorkspaces,
  listAssignmentCredentialBindings,
  listApprovalPolicies,
  listExternalServers,
  listPermissionProfiles,
  listPolicyAssignments,
  upsertAssignmentCredentialBinding,
  upsertPolicyAssignmentOverride,
  updatePolicyAssignment,
  type McpHubApprovalPolicy,
  type McpHubCredentialBinding,
  type McpHubDrillTarget,
  type McpHubEffectivePolicy,
  type McpHubEffectiveExternalAccess,
  type McpHubExternalServer,
  type McpHubPathScopeObject,
  type McpHubPermissionPolicyDocument,
  type McpHubPermissionProfile,
  type McpHubPolicyAssignment,
  type McpHubPolicyAssignmentWorkspace,
  type McpHubPolicyOverride,
  type McpHubToolRegistryEntry,
  type McpHubToolRegistryModule,
  type McpHubWorkspaceSetObject,
  type McpHubWorkspaceSourceMode
} from "@/services/tldw/mcp-hub"

import {
  getCredentialBindingKey,
  getManagedExternalServers,
  getManagedExternalServerSlots,
  getPathAllowlistSummary,
  getPathScopeLabel,
  MCP_HUB_SCOPE_OPTIONS,
  MCP_HUB_TARGET_OPTIONS,
  parseLineList
} from "./policyHelpers"
import { ExternalAccessSummary } from "./ExternalAccessSummary"
import { PolicyDocumentEditor } from "./PolicyDocumentEditor"

const PROVENANCE_LABELS = {
  profile: "profile",
  profile_path_scope_object: "profile path scope",
  assignment_path_scope_object: "assignment path scope",
  assignment_inline: "assignment",
  assignment_override: "assignment override"
} as const

type AssignmentValidationDetail = {
  message?: string
  conflicting_workspace_ids?: string[]
  conflicting_workspace_roots?: string[]
  unresolved_workspace_ids?: string[]
}

const _coerceAssignmentValidationDetail = (value: unknown): AssignmentValidationDetail | null => {
  if (!value || typeof value !== "object") return null
  const candidate = value as Record<string, unknown>
  const detail =
    candidate.detail && typeof candidate.detail === "object"
      ? (candidate.detail as Record<string, unknown>)
      : candidate
  const message = typeof detail.message === "string" ? detail.message : null
  const conflictingWorkspaceIds = Array.isArray(detail.conflicting_workspace_ids)
    ? detail.conflicting_workspace_ids
        .map((entry) => String(entry ?? "").trim())
        .filter((entry) => entry.length > 0)
    : []
  const conflictingWorkspaceRoots = Array.isArray(detail.conflicting_workspace_roots)
    ? detail.conflicting_workspace_roots
        .map((entry) => String(entry ?? "").trim())
        .filter((entry) => entry.length > 0)
    : []
  const unresolvedWorkspaceIds = Array.isArray(detail.unresolved_workspace_ids)
    ? detail.unresolved_workspace_ids
        .map((entry) => String(entry ?? "").trim())
        .filter((entry) => entry.length > 0)
    : []
  if (
    !message &&
    conflictingWorkspaceIds.length === 0 &&
    conflictingWorkspaceRoots.length === 0 &&
    unresolvedWorkspaceIds.length === 0
  ) {
    return null
  }
  return {
    message: message ?? undefined,
    conflicting_workspace_ids: conflictingWorkspaceIds,
    conflicting_workspace_roots: conflictingWorkspaceRoots,
    unresolved_workspace_ids: unresolvedWorkspaceIds
  }
}

const _formatAssignmentValidationError = (error: unknown, fallback: string): string => {
  const detail = _coerceAssignmentValidationDetail(
    (error as { details?: unknown } | null | undefined)?.details
  )
  if (!detail) return fallback
  const parts = [detail.message || fallback]
  if (detail.conflicting_workspace_ids && detail.conflicting_workspace_ids.length > 0) {
    parts.push(`Workspaces: ${detail.conflicting_workspace_ids.join(", ")}`)
  }
  if (detail.conflicting_workspace_roots && detail.conflicting_workspace_roots.length > 0) {
    parts.push(`Roots: ${detail.conflicting_workspace_roots.join(", ")}`)
  }
  if (detail.unresolved_workspace_ids && detail.unresolved_workspace_ids.length > 0) {
    parts.push(`Unresolved workspaces: ${detail.unresolved_workspace_ids.join(", ")}`)
  }
  return parts.join(" ")
}

type PolicyAssignmentsTabProps = {
  drillTarget?: McpHubDrillTarget | null
  onDrillHandled?: (requestId: number) => void
}

export const PolicyAssignmentsTab = ({
  drillTarget = null,
  onDrillHandled
}: PolicyAssignmentsTabProps) => {
  const handledDrillRequestRef = useRef<number | null>(null)
  const [assignments, setAssignments] = useState<McpHubPolicyAssignment[]>([])
  const [assignmentsLoaded, setAssignmentsLoaded] = useState(false)
  const [profiles, setProfiles] = useState<McpHubPermissionProfile[]>([])
  const [approvalPolicies, setApprovalPolicies] = useState<McpHubApprovalPolicy[]>([])
  const [effectivePolicy, setEffectivePolicy] = useState<McpHubEffectivePolicy | null>(null)
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [createOpen, setCreateOpen] = useState(false)
  const [editingId, setEditingId] = useState<number | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [targetType, setTargetType] = useState<"default" | "group" | "persona">("persona")
  const [targetId, setTargetId] = useState("")
  const [ownerScopeType, setOwnerScopeType] = useState<"global" | "org" | "team" | "user">("user")
  const [profileId, setProfileId] = useState<string>("")
  const [pathScopeSource, setPathScopeSource] = useState<"inline" | "named">("inline")
  const [pathScopeObjectId, setPathScopeObjectId] = useState("")
  const [workspaceSourceMode, setWorkspaceSourceMode] = useState<McpHubWorkspaceSourceMode>("inline")
  const [workspaceSetObjectId, setWorkspaceSetObjectId] = useState("")
  const [workspaceIdsText, setWorkspaceIdsText] = useState("")
  const [approvalPolicyId, setApprovalPolicyId] = useState<string>("")
  const [policyDocument, setPolicyDocument] = useState<McpHubPermissionPolicyDocument>({})
  const [isActive, setIsActive] = useState(true)
  const [overridePolicyDocument, setOverridePolicyDocument] = useState<McpHubPermissionPolicyDocument>(
    {}
  )
  const [overrideIsActive, setOverrideIsActive] = useState(true)
  const [overrideExists, setOverrideExists] = useState(false)
  const [overrideLoading, setOverrideLoading] = useState(false)
  const [overrideSaving, setOverrideSaving] = useState(false)
  const [registryEntries, setRegistryEntries] = useState<McpHubToolRegistryEntry[]>([])
  const [registryModules, setRegistryModules] = useState<McpHubToolRegistryModule[]>([])
  const [externalServers, setExternalServers] = useState<McpHubExternalServer[]>([])
  const [pathScopeObjects, setPathScopeObjects] = useState<McpHubPathScopeObject[]>([])
  const [workspaceSetObjects, setWorkspaceSetObjects] = useState<McpHubWorkspaceSetObject[]>([])
  const [assignmentWorkspaces, setAssignmentWorkspaces] = useState<McpHubPolicyAssignmentWorkspace[]>([])
  const [assignmentBindings, setAssignmentBindings] = useState<McpHubCredentialBinding[]>([])
  const [externalAccess, setExternalAccess] = useState<McpHubEffectiveExternalAccess | null>(null)
  const [bindingsLoading, setBindingsLoading] = useState(false)
  const [bindingServerId, setBindingServerId] = useState<string | null>(null)
  const managedExternalServers = useMemo(
    () => getManagedExternalServers(externalServers),
    [externalServers]
  )
  const selectedWorkspaceSet = useMemo(
    () =>
      workspaceSetObjects.find((workspaceSet) => String(workspaceSet.id) === String(workspaceSetObjectId)) ||
      null,
    [workspaceSetObjectId, workspaceSetObjects]
  )
  const bindingModes = useMemo(
    () =>
      new Map(
        assignmentBindings.map((binding) => [
          getCredentialBindingKey(binding.external_server_id, binding.slot_name),
          binding.binding_mode
        ] as const)
      ),
    [assignmentBindings]
  )

  const canSave = useMemo(
    () => !saving && (targetType === "default" || targetId.trim().length > 0),
    [saving, targetId, targetType]
  )

  const loadAll = async () => {
    setLoading(true)
    setErrorMessage(null)
    try {
      const [assignmentRows, profileRows, approvalRows] = await Promise.all([
        listPolicyAssignments(),
        listPermissionProfiles(),
        listApprovalPolicies()
      ])
      setAssignments(Array.isArray(assignmentRows) ? assignmentRows : [])
      setProfiles(Array.isArray(profileRows) ? profileRows : [])
      setApprovalPolicies(Array.isArray(approvalRows) ? approvalRows : [])

      const firstPersonaAssignment = assignmentRows.find(
        (row) => row.target_type === "persona" && row.target_id
      )
      const firstGroupAssignment = assignmentRows.find(
        (row) => row.target_type === "group" && row.target_id
      )
      const preview = await getEffectivePolicy({
        persona_id: firstPersonaAssignment?.target_id ?? null,
        group_id: firstGroupAssignment?.target_id ?? null
      })
      setEffectivePolicy(preview)
    } catch (err) {
      setAssignments([])
      setProfiles([])
      setApprovalPolicies([])
      setEffectivePolicy(null)
      const msg = err instanceof Error ? err.message : "Unknown error"
      setErrorMessage(`Failed to load policy assignments: ${msg}`)
    } finally {
      setLoading(false)
      setAssignmentsLoaded(true)
    }
  }

  useEffect(() => {
    void loadAll()
  }, [])

  useEffect(() => {
    let cancelled = false
    const loadRegistryAndServers = async () => {
      try {
        const [summary, serverRows, pathScopeRows, workspaceSetRows] = await Promise.all([
          getToolRegistrySummary(),
          listExternalServers(),
          listPathScopeObjects(),
          listWorkspaceSetObjects()
        ])
        if (!cancelled) {
          setRegistryEntries(Array.isArray(summary?.entries) ? summary.entries : [])
          setRegistryModules(Array.isArray(summary?.modules) ? summary.modules : [])
          setExternalServers(Array.isArray(serverRows) ? serverRows : [])
          setPathScopeObjects(Array.isArray(pathScopeRows) ? pathScopeRows : [])
          setWorkspaceSetObjects(Array.isArray(workspaceSetRows) ? workspaceSetRows : [])
        }
      } catch {
        if (!cancelled) {
          setRegistryEntries([])
          setRegistryModules([])
          setExternalServers([])
          setPathScopeObjects([])
          setWorkspaceSetObjects([])
        }
      }
    }
    void loadRegistryAndServers()
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (
      !drillTarget ||
      drillTarget.tab !== "assignments" ||
      drillTarget.object_kind !== "policy_assignment"
    ) {
      return
    }
    if (
      handledDrillRequestRef.current === drillTarget.request_id ||
      loading ||
      !assignmentsLoaded
    ) {
      return
    }
    const assignment = assignments.find(
      (row) => String(row.id) === String(drillTarget.object_id)
    )
    if (assignment) {
      handledDrillRequestRef.current = drillTarget.request_id
      openForEdit(assignment)
      onDrillHandled?.(drillTarget.request_id)
    }
  }, [assignments, assignmentsLoaded, drillTarget, loading, onDrillHandled])

  const resetForm = () => {
    setCreateOpen(false)
    setEditingId(null)
    setTargetType("persona")
    setTargetId("")
    setOwnerScopeType("user")
    setProfileId("")
    setPathScopeSource("inline")
    setPathScopeObjectId("")
    setWorkspaceSourceMode("inline")
    setWorkspaceSetObjectId("")
    setWorkspaceIdsText("")
    setApprovalPolicyId("")
    setPolicyDocument({})
    setIsActive(true)
    setOverridePolicyDocument({})
    setOverrideIsActive(true)
    setOverrideExists(false)
    setOverrideLoading(false)
    setOverrideSaving(false)
    setAssignmentBindings([])
    setExternalAccess(null)
    setBindingsLoading(false)
    setBindingServerId(null)
    setAssignmentWorkspaces([])
  }

  const loadOverride = async (assignmentId: number) => {
    setOverrideLoading(true)
    try {
      const row = await getPolicyAssignmentOverride(assignmentId)
      const overrideRow = row as McpHubPolicyOverride
      setOverridePolicyDocument(overrideRow.override_policy_document || {})
      setOverrideIsActive(Boolean(overrideRow.is_active))
      setOverrideExists(true)
    } catch {
      setOverridePolicyDocument({})
      setOverrideIsActive(true)
      setOverrideExists(false)
    } finally {
      setOverrideLoading(false)
    }
  }

  const loadAssignmentExternalState = async (assignmentId: number) => {
    setBindingsLoading(true)
    try {
      const [bindingRows, summary, serverRows] = await Promise.all([
        listAssignmentCredentialBindings(assignmentId),
        getAssignmentExternalAccess(assignmentId),
        listExternalServers()
      ])
      setAssignmentBindings(Array.isArray(bindingRows) ? bindingRows : [])
      setExternalAccess(summary)
      setExternalServers(Array.isArray(serverRows) ? serverRows : [])
    } catch (err) {
      setAssignmentBindings([])
      setExternalAccess(null)
      const msg = err instanceof Error ? err.message : "Unknown error"
      setErrorMessage(`Failed to load external service bindings: ${msg}`)
    } finally {
      setBindingsLoading(false)
    }
  }

  const loadAssignmentWorkspaces = async (assignmentId: number) => {
    try {
      const rows = await listPolicyAssignmentWorkspaces(assignmentId)
      const nextRows = Array.isArray(rows) ? rows : []
      setAssignmentWorkspaces(nextRows)
      setWorkspaceIdsText(nextRows.map((row) => row.workspace_id).join("\n"))
    } catch {
      setAssignmentWorkspaces([])
      setWorkspaceIdsText("")
      setErrorMessage("Failed to load assignment workspace access.")
    }
  }

  const openForEdit = (assignment: McpHubPolicyAssignment) => {
    setCreateOpen(true)
    setEditingId(assignment.id)
    setTargetType(assignment.target_type)
    setTargetId(String(assignment.target_id || ""))
    setOwnerScopeType(assignment.owner_scope_type)
    setProfileId(assignment.profile_id ? String(assignment.profile_id) : "")
    setPathScopeSource(assignment.path_scope_object_id ? "named" : "inline")
    setPathScopeObjectId(assignment.path_scope_object_id ? String(assignment.path_scope_object_id) : "")
    setWorkspaceSourceMode(assignment.workspace_source_mode || "inline")
    setWorkspaceSetObjectId(
      assignment.workspace_set_object_id ? String(assignment.workspace_set_object_id) : ""
    )
    setApprovalPolicyId(assignment.approval_policy_id ? String(assignment.approval_policy_id) : "")
    setPolicyDocument(assignment.inline_policy_document || {})
    setIsActive(assignment.is_active)
    setOverridePolicyDocument({})
    setOverrideIsActive(assignment.has_override ? Boolean(assignment.override_active) : true)
    setOverrideExists(Boolean(assignment.has_override))
    if (assignment.has_override) {
      void loadOverride(assignment.id)
    }
    void loadAssignmentExternalState(assignment.id)
    void loadAssignmentWorkspaces(assignment.id)
  }

  const syncAssignmentWorkspaces = async (assignmentId: number) => {
    if (workspaceSourceMode !== "inline") {
      return
    }
    const desiredWorkspaceIds = Array.from(new Set(parseLineList(workspaceIdsText)))
    const currentWorkspaceIds = assignmentWorkspaces.map((row) => row.workspace_id)
    const toAdd = desiredWorkspaceIds.filter((workspaceId) => !currentWorkspaceIds.includes(workspaceId))
    const toDelete = currentWorkspaceIds.filter((workspaceId) => !desiredWorkspaceIds.includes(workspaceId))

    await Promise.all([
      ...toAdd.map((workspaceId) => addPolicyAssignmentWorkspace(assignmentId, workspaceId)),
      ...toDelete.map((workspaceId) => deletePolicyAssignmentWorkspace(assignmentId, workspaceId))
    ])

    const nextRows = await listPolicyAssignmentWorkspaces(assignmentId)
    const normalizedRows = Array.isArray(nextRows) ? nextRows : []
    setAssignmentWorkspaces(normalizedRows)
    setWorkspaceIdsText(normalizedRows.map((row) => row.workspace_id).join("\n"))
  }

  const handleSave = async () => {
    if (!canSave) return
    setSaving(true)
    setErrorMessage(null)
    try {
      const payload = {
        target_type: targetType,
        target_id: targetType === "default" ? null : targetId.trim(),
        owner_scope_type: ownerScopeType,
        profile_id: profileId ? Number(profileId) : null,
        path_scope_object_id:
          pathScopeSource === "named" && pathScopeObjectId ? Number(pathScopeObjectId) : null,
        workspace_source_mode: workspaceSourceMode,
        workspace_set_object_id:
          workspaceSourceMode === "named" && workspaceSetObjectId ? Number(workspaceSetObjectId) : null,
        approval_policy_id: approvalPolicyId ? Number(approvalPolicyId) : null,
        inline_policy_document: policyDocument,
        is_active: isActive
      }
      let savedAssignmentId = editingId
      if (editingId) {
        await updatePolicyAssignment(editingId, payload)
      } else {
        const created = await createPolicyAssignment(payload)
        savedAssignmentId = created.id
      }
      if (savedAssignmentId) {
        await syncAssignmentWorkspaces(savedAssignmentId)
      }
      resetForm()
      await loadAll()
    } catch (error) {
      setErrorMessage(
        _formatAssignmentValidationError(
          error,
          editingId ? "Failed to update policy assignment." : "Failed to create policy assignment."
        )
      )
    } finally {
      setSaving(false)
    }
  }

  const handleDelete = (assignment: McpHubPolicyAssignment) => {
    const targetLabel = assignment.target_type === "default"
      ? "the default assignment"
      : assignment.target_id
        ? `the ${assignment.target_type} assignment for "${assignment.target_id}"`
        : `the ${assignment.target_type} assignment`
    Modal.confirm({
      title: "Delete Policy Assignment",
      content: `Are you sure you want to delete ${targetLabel}? This cannot be undone.`,
      okText: "Delete",
      okType: "danger",
      cancelText: "Cancel",
      onOk: async () => {
        setErrorMessage(null)
        try {
          await deletePolicyAssignment(assignment.id)
          await loadAll()
        } catch (err) {
          const msg = err instanceof Error ? err.message : "Unknown error"
          setErrorMessage(`Failed to delete policy assignment: ${msg}`)
        }
      }
    })
  }

  const handleSaveOverride = async () => {
    if (!editingId) return
    setOverrideSaving(true)
    setErrorMessage(null)
    try {
      await upsertPolicyAssignmentOverride(editingId, {
        override_policy_document: overridePolicyDocument,
        is_active: overrideIsActive
      })
      setOverrideExists(true)
      await loadAll()
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Unknown error"
      setErrorMessage(`Failed to save assignment override: ${msg}`)
    } finally {
      setOverrideSaving(false)
    }
  }

  const handleDeleteOverride = () => {
    if (!editingId) return
    Modal.confirm({
      title: "Delete Assignment Override",
      content: "Are you sure you want to delete this assignment override? This cannot be undone.",
      okText: "Delete",
      okType: "danger",
      cancelText: "Cancel",
      onOk: async () => {
        setErrorMessage(null)
        try {
          await deletePolicyAssignmentOverride(editingId)
          setOverridePolicyDocument({})
          setOverrideIsActive(true)
          setOverrideExists(false)
          await loadAll()
        } catch (err) {
          const msg = err instanceof Error ? err.message : "Unknown error"
          setErrorMessage(`Failed to delete assignment override: ${msg}`)
        }
      }
    })
  }

  const handleAssignmentBindingModeChange = async (
    serverId: string,
    slotName: string | null | undefined,
    nextMode: "inherit" | "grant" | "disable"
  ) => {
    if (!editingId) return
    const bindingKey = getCredentialBindingKey(serverId, slotName)
    const currentMode = bindingModes.get(bindingKey) || "inherit"
    if (currentMode === nextMode) {
      return
    }
    setBindingServerId(bindingKey)
    setErrorMessage(null)
    try {
      if (nextMode === "inherit") {
        if (bindingModes.has(bindingKey)) {
          await deleteAssignmentCredentialBinding(editingId, serverId, slotName)
        }
      } else {
        await upsertAssignmentCredentialBinding(
          editingId,
          serverId,
          { binding_mode: nextMode },
          slotName
        )
      }
      await loadAssignmentExternalState(editingId)
    } catch {
      setErrorMessage("Failed to update external service binding.")
    } finally {
      setBindingServerId(null)
    }
  }

  return (
    <Space orientation="vertical" size="middle" style={{ width: "100%" }}>
      <Typography.Text type="secondary">
        Assign profiles to default, group, or persona targets, then layer in exact tool overrides where
        needed.
      </Typography.Text>
      {errorMessage ? <Alert type="error" title={errorMessage} showIcon /> : null}

      <Button type="primary" onClick={() => setCreateOpen(true)}>
        New Assignment
      </Button>

      {createOpen ? (
        <Card title={editingId ? "Edit Policy Assignment" : "Create Policy Assignment"}>
          <Space orientation="vertical" size="middle" style={{ width: "100%" }}>
            <Space>
              <Space orientation="vertical">
                <label htmlFor="mcp-assignment-target-type">Target Type</label>
                <select
                  id="mcp-assignment-target-type"
                  aria-label="Target Type"
                  value={targetType}
                  onChange={(event) => setTargetType(event.target.value as typeof targetType)}
                >
                  {MCP_HUB_TARGET_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </Space>
              <Space orientation="vertical">
                <label htmlFor="mcp-assignment-scope">Owner Scope</label>
                <select
                  id="mcp-assignment-scope"
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
            </Space>
            {targetType !== "default" ? (
              <Space orientation="vertical" style={{ width: "100%" }}>
                <label htmlFor="mcp-assignment-target-id">Target Id</label>
                <input
                  id="mcp-assignment-target-id"
                  aria-label="Target Id"
                  value={targetId}
                  onChange={(event) => setTargetId(event.target.value)}
                  placeholder={targetType === "persona" ? "researcher" : "team-red"}
                />
              </Space>
            ) : null}
            <Space>
              <Space orientation="vertical">
                <label htmlFor="mcp-assignment-profile">Referenced Profile</label>
                <select
                  id="mcp-assignment-profile"
                  aria-label="Referenced Profile"
                  value={profileId}
                  onChange={(event) => setProfileId(event.target.value)}
                >
                  <option value="">Manual only</option>
                  {profiles.map((profile) => (
                    <option key={profile.id} value={profile.id}>
                      {profile.name}
                    </option>
                  ))}
                </select>
              </Space>
              <Space orientation="vertical">
                <label htmlFor="mcp-assignment-approval-policy">Approval Policy</label>
                <select
                  id="mcp-assignment-approval-policy"
                  aria-label="Approval Policy"
                  value={approvalPolicyId}
                  onChange={(event) => setApprovalPolicyId(event.target.value)}
                >
                  <option value="">No runtime approval</option>
                  {approvalPolicies.map((policy) => (
                    <option key={policy.id} value={policy.id}>
                      {policy.name}
                    </option>
                  ))}
                </select>
              </Space>
            </Space>

            <Card size="small" title="Path Scope Source">
              <Space orientation="vertical" size="middle" style={{ width: "100%" }}>
                <Typography.Text type="secondary">
                  Named path scopes provide reusable relative file rules. Inline path fields below stay
                  preserved and can still replace the object values for this assignment.
                </Typography.Text>
                <Space wrap>
                  <label>
                    <input
                      type="radio"
                      name="mcp-assignment-path-scope-source"
                      checked={pathScopeSource === "inline"}
                      onChange={() => setPathScopeSource("inline")}
                    />
                    <span style={{ marginLeft: 8 }}>Use inline rules</span>
                  </label>
                  <label>
                    <input
                      type="radio"
                      name="mcp-assignment-path-scope-source"
                      checked={pathScopeSource === "named"}
                      onChange={() => setPathScopeSource("named")}
                    />
                    <span style={{ marginLeft: 8 }}>Use named path scope</span>
                  </label>
                </Space>
                {pathScopeSource === "named" ? (
                  <Space orientation="vertical" style={{ width: "100%" }}>
                    <label htmlFor="mcp-assignment-path-scope-object">Named path scope</label>
                    <select
                      id="mcp-assignment-path-scope-object"
                      aria-label="Assignment named path scope"
                      value={pathScopeObjectId}
                      onChange={(event) => setPathScopeObjectId(event.target.value)}
                    >
                      <option value="">Select a path scope</option>
                      {pathScopeObjects.map((pathScopeObject) => (
                        <option key={pathScopeObject.id} value={pathScopeObject.id}>
                          {pathScopeObject.name}
                        </option>
                      ))}
                    </select>
                  </Space>
                ) : null}
              </Space>
            </Card>

            <Card size="small" title="Workspace Access">
              <Space orientation="vertical" size="middle" style={{ width: "100%" }}>
                <Typography.Text type="secondary">
                  Select one workspace source. Inline rows stay preserved even when a named workspace set is
                  active, but runtime uses only the selected source.
                </Typography.Text>
                <Space wrap>
                  <label>
                    <input
                      type="radio"
                      name="mcp-assignment-workspace-source"
                      checked={workspaceSourceMode === "inline"}
                      onChange={() => setWorkspaceSourceMode("inline")}
                    />
                    <span style={{ marginLeft: 8 }}>Use inline workspace list</span>
                  </label>
                  <label>
                    <input
                      type="radio"
                      name="mcp-assignment-workspace-source"
                      checked={workspaceSourceMode === "named"}
                      onChange={() => setWorkspaceSourceMode("named")}
                    />
                    <span style={{ marginLeft: 8 }}>Use named workspace set</span>
                  </label>
                </Space>
                {workspaceSourceMode === "named" ? (
                  <Space orientation="vertical" style={{ width: "100%" }}>
                    <label htmlFor="mcp-assignment-workspace-set-object">Assignment named workspace set</label>
                    <select
                      id="mcp-assignment-workspace-set-object"
                      aria-label="Assignment named workspace set"
                      value={workspaceSetObjectId}
                      onChange={(event) => setWorkspaceSetObjectId(event.target.value)}
                    >
                      <option value="">Select a workspace set</option>
                      {workspaceSetObjects.map((workspaceSet) => (
                        <option key={workspaceSet.id} value={workspaceSet.id}>
                          {workspaceSet.readiness_summary && !workspaceSet.readiness_summary.is_multi_root_ready
                            ? `${workspaceSet.name} (overlap warning)`
                            : workspaceSet.name}
                        </option>
                      ))}
                    </select>
                    {selectedWorkspaceSet?.readiness_summary &&
                    !selectedWorkspaceSet.readiness_summary.is_multi_root_ready ? (
                      <Alert
                        type="warning"
                        showIcon
                        title={
                          selectedWorkspaceSet.readiness_summary.warning_message ||
                          "This workspace set is not currently multi-root-ready."
                        }
                      />
                    ) : null}
                    <Typography.Text type="secondary">
                      Preserved inline workspace rows remain stored but inactive while a named workspace set is selected.
                    </Typography.Text>
                  </Space>
                ) : (
                  <Space orientation="vertical" style={{ width: "100%" }}>
                    <label htmlFor="mcp-assignment-workspace-ids">Allowed workspace ids</label>
                    <textarea
                      id="mcp-assignment-workspace-ids"
                      aria-label="Allowed workspace ids"
                      value={workspaceIdsText}
                      onChange={(event) => setWorkspaceIdsText(event.target.value)}
                      rows={4}
                    />
                    <Typography.Text type="secondary">
                      One workspace id per line. If this list is empty, current behavior stays unchanged.
                    </Typography.Text>
                  </Space>
                )}
              </Space>
            </Card>

            <Card size="small" title="Base Assignment Policy">
              <PolicyDocumentEditor
                formId="mcp-assignment"
                policy={policyDocument}
                onChange={setPolicyDocument}
                registryEntries={registryEntries}
                registryModules={registryModules}
              />
            </Card>

            {editingId ? (
              <Card size="small" title="External Service Bindings">
                <Space orientation="vertical" size="middle" style={{ width: "100%" }}>
                  <Typography.Text type="secondary">
                    Grant or disable managed external MCP servers for this assignment. Legacy inventory is
                    visible in External Servers and cannot be selected here until imported into MCP Hub.
                  </Typography.Text>
                  {bindingsLoading ? (
                    <Typography.Text type="secondary">Loading external service bindings...</Typography.Text>
                  ) : managedExternalServers.length > 0 ? (
                    <Space orientation="vertical" size="small" style={{ width: "100%" }}>
                      {managedExternalServers.map((server) => {
                        const slots = getManagedExternalServerSlots(server)
                        return (
                          <Card key={server.id} size="small" title={server.name}>
                            <Space orientation="vertical" size="small" style={{ width: "100%" }}>
                              <Space wrap size={4}>
                                {server.secret_configured ? <Tag color="green">secret configured</Tag> : <Tag>no secret</Tag>}
                              </Space>
                              {slots.length > 0 ? (
                                <Space orientation="vertical" size="small" style={{ width: "100%" }}>
                                  {slots.map((slot) => {
                                    const bindingKey = getCredentialBindingKey(server.id, slot.slot_name)
                                    return (
                                      <Space
                                        key={bindingKey}
                                        wrap
                                        size="small"
                                        style={{ width: "100%", justifyContent: "space-between" }}
                                      >
                                        <Space wrap size={4}>
                                          <Typography.Text>{slot.display_name}</Typography.Text>
                                          <Tag>{slot.slot_name}</Tag>
                                          <Tag>{slot.privilege_class}</Tag>
                                          {slot.secret_configured ? (
                                            <Tag color="green">slot secret configured</Tag>
                                          ) : (
                                            <Tag>slot secret missing</Tag>
                                          )}
                                        </Space>
                                        <select
                                          aria-label={`${server.name} ${slot.display_name}`}
                                          value={bindingModes.get(bindingKey) || "inherit"}
                                          disabled={bindingServerId === bindingKey}
                                          onChange={(event) =>
                                            void handleAssignmentBindingModeChange(
                                              server.id,
                                              slot.slot_name,
                                              event.target.value as "inherit" | "grant" | "disable"
                                            )
                                          }
                                        >
                                          <option value="inherit">Inherit</option>
                                          <option value="grant">Grant</option>
                                          <option value="disable">Disable</option>
                                        </select>
                                      </Space>
                                    )
                                  })}
                                </Space>
                              ) : (
                                <Typography.Text type="secondary">
                                  Define credential slots in External Servers before using slot-level assignment bindings.
                                </Typography.Text>
                              )}
                            </Space>
                          </Card>
                        )
                      })}
                    </Space>
                  ) : (
                    <Empty description="No managed external servers are available yet." />
                  )}
                  <Card size="small" title="Effective External Access">
                    <ExternalAccessSummary
                      summary={externalAccess}
                      emptyText="No external server access is currently configured for this assignment."
                    />
                  </Card>
                </Space>
              </Card>
            ) : null}

            {editingId ? (
              <Card
                size="small"
                title="Assignment Override"
                extra={
                  overrideExists ? (
                    <Tag color={overrideIsActive ? "cyan" : "default"}>
                      {overrideIsActive ? "override active" : "override inactive"}
                    </Tag>
                  ) : (
                    <Tag>no override yet</Tag>
                  )
                }
              >
                <Space orientation="vertical" size="middle" style={{ width: "100%" }}>
                  <Typography.Text type="secondary">
                    Use one explicit override document for this assignment when it needs to differ from
                    the base profile plus assignment policy.
                  </Typography.Text>
                  {overrideLoading ? (
                    <Typography.Text type="secondary">Loading override...</Typography.Text>
                  ) : (
                    <PolicyDocumentEditor
                      formId="mcp-assignment-override"
                      policy={overridePolicyDocument}
                      onChange={setOverridePolicyDocument}
                      registryEntries={registryEntries}
                      registryModules={registryModules}
                    />
                  )}
                  <Checkbox
                    checked={overrideIsActive}
                    onChange={(event) => setOverrideIsActive(event.target.checked)}
                  >
                    Override Active
                  </Checkbox>
                  <Space>
                    <Button
                      type="primary"
                      onClick={() => void handleSaveOverride()}
                      loading={overrideSaving}
                      disabled={overrideLoading}
                    >
                      Save Override
                    </Button>
                    <Button
                      danger
                      onClick={() => handleDeleteOverride()}
                      disabled={!overrideExists || overrideLoading}
                    >
                      Delete Override
                    </Button>
                  </Space>
                </Space>
              </Card>
            ) : null}

            <Checkbox checked={isActive} onChange={(event) => setIsActive(event.target.checked)}>
              Active
            </Checkbox>
            <Space>
              <Button type="primary" onClick={handleSave} disabled={!canSave} loading={saving}>
                {editingId ? "Update Assignment" : "Save Assignment"}
              </Button>
              <Button onClick={resetForm}>Cancel</Button>
            </Space>
          </Space>
        </Card>
      ) : null}

      <Card title="Current Effective Preview">
        {effectivePolicy ? (
          <Space orientation="vertical" size="small" style={{ width: "100%" }}>
            {(() => {
              const effectivePolicyDocument =
                (effectivePolicy.policy_document as McpHubPermissionPolicyDocument | null | undefined) ??
                {}

              return (
                <Space wrap>
                  {effectivePolicy.capabilities.map((capability) => (
                    <Tag key={capability}>{capability}</Tag>
                  ))}
                  {effectivePolicy.allowed_tools.map((tool) => (
                    <Tag key={tool} color="green">
                      {tool}
                    </Tag>
                  ))}
                  {effectivePolicy.denied_tools.map((tool) => (
                    <Tag key={tool} color="red">
                      {tool}
                    </Tag>
                  ))}
                  {effectivePolicy.approval_mode ? (
                    <Tag color="gold">{effectivePolicy.approval_mode}</Tag>
                  ) : null}
                  {getPathScopeLabel(effectivePolicyDocument.path_scope_mode) ? (
                    <Tag color="cyan">{getPathScopeLabel(effectivePolicyDocument.path_scope_mode)}</Tag>
                  ) : null}
                  {effectivePolicyDocument.path_scope_enforcement ? (
                    <Tag color="orange">Path approval fallback</Tag>
                  ) : null}
                  {getPathAllowlistSummary(effectivePolicyDocument.path_allowlist_prefixes) ? (
                    <Tag color="blue">
                      {`paths ${getPathAllowlistSummary(effectivePolicyDocument.path_allowlist_prefixes)}`}
                    </Tag>
                  ) : null}
                  {effectivePolicy.selected_assignment_workspace_ids?.length ? (
                    <Tag color="purple">
                      {`workspaces ${effectivePolicy.selected_assignment_workspace_ids.join(", ")}`}
                    </Tag>
                  ) : null}
                  {effectivePolicy.selected_workspace_set_object_name ? (
                    <Tag color="geekblue">
                      {`workspace set ${effectivePolicy.selected_workspace_set_object_name}`}
                    </Tag>
                  ) : null}
                  {effectivePolicy.selected_workspace_trust_source ? (
                    <Tag color={effectivePolicy.selected_workspace_trust_source === "shared_registry" ? "magenta" : "purple"}>
                      {effectivePolicy.selected_workspace_trust_source === "shared_registry"
                        ? "shared registry"
                        : "user-local"}
                    </Tag>
                  ) : null}
                </Space>
              )
            })()}
            {effectivePolicy.provenance.length > 0 ? (
              <Space orientation="vertical" size={4} style={{ width: "100%" }}>
                <Typography.Text strong>Why This Applies</Typography.Text>
                {effectivePolicy.provenance.map((entry, index) => (
                  <Typography.Text key={`${entry.assignment_id}-${entry.field}-${entry.source_kind}-${index}`}>
                    {`${entry.field} from ${PROVENANCE_LABELS[entry.source_kind]} (${entry.effect})`}
                  </Typography.Text>
                ))}
              </Space>
            ) : null}
          </Space>
        ) : (
          <Empty description="No effective policy preview available yet" />
        )}
      </Card>

      <List
        bordered
        loading={loading}
        dataSource={assignments}
        locale={{
          emptyText: (
            <Empty
              description={
                <Space orientation="vertical" size={4}>
                  <Typography.Text type="secondary">No policy assignments yet</Typography.Text>
                  <Typography.Text type="secondary" style={{ fontSize: 13 }}>
                    Assignments bind permission profiles to users, groups, or personas.
                    Create a permission profile first, then assign it here.
                  </Typography.Text>
                </Space>
              }
            >
              <Button type="primary" onClick={() => setCreateOpen(true)}>
                Create Assignment
              </Button>
            </Empty>
          )
        }}
        renderItem={(assignment) => (
          <List.Item>
            <Space orientation="vertical" size={4} style={{ width: "100%" }}>
              <Space wrap>
                <Typography.Text strong>{assignment.target_id || assignment.target_type}</Typography.Text>
                <Tag>{assignment.target_type}</Tag>
                <Tag>{assignment.owner_scope_type}</Tag>
                {assignment.profile_id ? <Tag color="blue">{`profile ${assignment.profile_id}`}</Tag> : null}
                {assignment.path_scope_object_id ? (
                  <Tag color="purple">
                    {`path scope ${
                      pathScopeObjects.find((row) => row.id === assignment.path_scope_object_id)?.name ||
                      assignment.path_scope_object_id
                    }`}
                  </Tag>
                ) : null}
                {assignment.workspace_source_mode === "named" && assignment.workspace_set_object_id ? (
                  <Tag color="geekblue">
                    {`workspace set ${
                      workspaceSetObjects.find((row) => row.id === assignment.workspace_set_object_id)?.name ||
                      assignment.workspace_set_object_id
                    }`}
                  </Tag>
                ) : null}
                {assignment.approval_policy_id ? (
                  <Tag color="gold">{`approval ${assignment.approval_policy_id}`}</Tag>
                ) : null}
                {assignment.is_active ? <Tag color="green">active</Tag> : <Tag>inactive</Tag>}
                {assignment.has_override ? (
                  <Tag color={assignment.override_active ? "cyan" : "default"}>
                    {assignment.override_active ? "override active" : "override inactive"}
                  </Tag>
                ) : null}
                <Button size="small" onClick={() => openForEdit(assignment)}>
                  Edit
                </Button>
                <Button size="small" danger onClick={() => handleDelete(assignment)}>
                  Delete
                </Button>
              </Space>
              <Space wrap>
                {(assignment.inline_policy_document.capabilities || []).map((capability) => (
                  <Tag key={capability}>{capability}</Tag>
                ))}
                {(assignment.inline_policy_document.allowed_tools || []).map((tool) => (
                  <Tag key={tool} color="green">
                    {tool}
                  </Tag>
                ))}
                {getPathScopeLabel(assignment.inline_policy_document.path_scope_mode) ? (
                  <Tag color="cyan">{getPathScopeLabel(assignment.inline_policy_document.path_scope_mode)}</Tag>
                ) : null}
                {getPathAllowlistSummary(assignment.inline_policy_document.path_allowlist_prefixes) ? (
                  <Tag color="blue">
                    {`paths ${getPathAllowlistSummary(assignment.inline_policy_document.path_allowlist_prefixes)}`}
                  </Tag>
                ) : null}
                {assignmentWorkspaces.length > 0 && editingId === assignment.id ? (
                  <Tag color="purple">
                    {`workspaces ${assignmentWorkspaces.map((row) => row.workspace_id).join(", ")}`}
                  </Tag>
                ) : null}
              </Space>
            </Space>
          </List.Item>
        )}
      />
    </Space>
  )
}
