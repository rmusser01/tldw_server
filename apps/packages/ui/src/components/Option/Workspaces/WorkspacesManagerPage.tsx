import React from "react"
import { Plus, RefreshCw, Search } from "lucide-react"
import { useNavigate } from "react-router-dom"
import { Button } from "@/components/Common/Button"
import { Alert, Badge } from "@/components/ui/primitives"
import { useTldwApiClient } from "@/hooks/useTldwApiClient"
import { buildResearchWorkspaceReturnPath } from "@/routes/route-paths"
import type {
  WorkspaceApiResponse,
  WorkspaceProfile
} from "@/services/tldw/domains/workspace-api"
import { WorkspaceCreateDialog } from "./WorkspaceCreateDialog"
import { WorkspaceList } from "./WorkspaceList"
import { WorkspaceMetadataDialog } from "./WorkspaceMetadataDialog"
import { WorkspaceProjectRootPanel } from "./WorkspaceProjectRootPanel"
import {
  normalizeWorkspaceManagerItem,
  type WorkspaceManagerAttention,
  type WorkspaceManagerItem
} from "./workspace-manager-models"

type ProfileFilter = "all" | WorkspaceProfile
type AttentionFilter = "all" | "needs_attention"
type DialogProfile = WorkspaceProfile | null

const createWorkspaceId = (): string => {
  const randomUUID = globalThis.crypto?.randomUUID
  if (typeof randomUUID === "function") return randomUUID.call(globalThis.crypto)
  return `workspace-${Date.now()}`
}

const errorText = (error: unknown): string =>
  error instanceof Error ? error.message : "Unknown Workspace error"

const filterMatchesAttention = (
  attentionState: WorkspaceManagerAttention,
  filter: AttentionFilter
): boolean => {
  if (filter === "all") return true
  return attentionState === "needs_attention" || attentionState === "blocked"
}

const mergeWorkspaceItem = (
  items: WorkspaceManagerItem[],
  workspace: WorkspaceApiResponse,
  options: { prependIfMissing?: boolean } = {}
): WorkspaceManagerItem[] => {
  const normalized = normalizeWorkspaceManagerItem(workspace)
  const existingIndex = items.findIndex((item) => item.id === workspace.id)
  if (existingIndex === -1) {
    return options.prependIfMissing ? [normalized, ...items] : [...items, normalized]
  }
  return items.map((item, index) =>
    index === existingIndex
      ? {
          ...item,
          name: normalized.name,
          archived: normalized.archived,
          profile: normalized.profile,
          attentionState: normalized.attentionState,
          updatedAt: normalized.updatedAt,
          version: normalized.version
        }
      : item
  )
}

export const WorkspacesManagerPage = () => {
  const api = useTldwApiClient()
  const navigate = useNavigate()
  const [items, setItems] = React.useState<WorkspaceManagerItem[]>([])
  const [loading, setLoading] = React.useState(true)
  const [error, setError] = React.useState<string | null>(null)
  const [partialError, setPartialError] = React.useState<string | null>(null)
  const [searchQuery, setSearchQuery] = React.useState("")
  const [profileFilter, setProfileFilter] = React.useState<ProfileFilter>("all")
  const [attentionFilter, setAttentionFilter] =
    React.useState<AttentionFilter>("all")
  const [showArchived, setShowArchived] = React.useState(false)
  const [createProfile, setCreateProfile] = React.useState<DialogProfile>(null)
  const [editingItem, setEditingItem] = React.useState<WorkspaceManagerItem | null>(
    null
  )
  const [selectedItemId, setSelectedItemId] = React.useState<string | null>(null)
  const [mutationError, setMutationError] = React.useState<string | null>(null)
  const [mutating, setMutating] = React.useState(false)

  const loadWorkspaces = React.useCallback(async () => {
    setLoading(true)
    setError(null)
    setPartialError(null)
    try {
      const response = await api.listWorkspaces()
      const workspaces = Array.isArray(response.items) ? response.items : []
      const normalizedResults = await Promise.allSettled(
        workspaces.map(async (workspace) => {
          const context = await api.getWorkspaceContext(workspace.id)
          return normalizeWorkspaceManagerItem(workspace, context)
        })
      )
      const normalized = normalizedResults.map((result, index) =>
        result.status === "fulfilled"
          ? result.value
          : normalizeWorkspaceManagerItem(workspaces[index])
      )
      if (normalizedResults.some((result) => result.status === "rejected")) {
        setPartialError("Some Workspace details could not load.")
      }
      setItems(normalized)
      setSelectedItemId((current) =>
        current && normalized.some((item) => item.id === current)
          ? current
          : normalized[0]?.id ?? null
      )
    } catch (caught) {
      setError(errorText(caught))
      setItems([])
      setSelectedItemId(null)
    } finally {
      setLoading(false)
    }
  }, [api])

  React.useEffect(() => {
    void loadWorkspaces()
  }, [loadWorkspaces])

  const filteredItems = React.useMemo(() => {
    const query = searchQuery.trim().toLowerCase()
    return items.filter((item) => {
      if (!showArchived && item.archived) return false
      if (profileFilter !== "all" && item.profile !== profileFilter) return false
      if (!filterMatchesAttention(item.attentionState, attentionFilter)) {
        return false
      }
      if (!query) return true
      return (
        item.name.toLowerCase().includes(query) ||
        item.id.toLowerCase().includes(query)
      )
    })
  }, [attentionFilter, items, profileFilter, searchQuery, showArchived])

  const selectedItem = React.useMemo(() => {
    if (filteredItems.length === 0) return null
    return (
      filteredItems.find((item) => item.id === selectedItemId) ?? filteredItems[0]
    )
  }, [filteredItems, selectedItemId])

  const createWorkspace = async (
    name: string,
    profile: WorkspaceProfile
  ): Promise<void> => {
    setMutating(true)
    setMutationError(null)
    try {
      const workspace = await api.upsertWorkspace(createWorkspaceId(), {
        name,
        study_materials_policy: "workspace",
        workspace_profile: profile
      })
      setItems((current) =>
        mergeWorkspaceItem(current, workspace, { prependIfMissing: true })
      )
      setSelectedItemId(workspace.id)
      setCreateProfile(null)
    } catch (caught) {
      setMutationError(errorText(caught))
    } finally {
      setMutating(false)
    }
  }

  const updateWorkspaceName = async (
    item: WorkspaceManagerItem,
    name: string
  ): Promise<void> => {
    setMutating(true)
    setMutationError(null)
    try {
      const workspace = await api.patchWorkspace(item.id, {
        name,
        version: item.version
      })
      setItems((current) => mergeWorkspaceItem(current, workspace))
      setSelectedItemId(workspace.id)
      setEditingItem(null)
    } catch (caught) {
      setMutationError(errorText(caught))
    } finally {
      setMutating(false)
    }
  }

  const updateArchived = async (
    item: WorkspaceManagerItem,
    archived: boolean
  ): Promise<void> => {
    setMutating(true)
    setMutationError(null)
    try {
      const workspace = await api.patchWorkspace(item.id, {
        archived,
        version: item.version
      })
      setItems((current) => mergeWorkspaceItem(current, workspace))
      setSelectedItemId(workspace.id)
    } catch (caught) {
      setMutationError(errorText(caught))
    } finally {
      setMutating(false)
    }
  }

  return (
    <section className="flex h-full min-h-0 w-full flex-col bg-bg text-text">
      <div className="border-b border-border px-4 py-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="text-xl font-semibold text-text">Workspaces</h1>
            <p className="mt-1 text-sm text-text-muted">
              Server-backed research and project Workspace directory.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button
              variant="secondary"
              icon={<Plus className="h-4 w-4" />}
              onClick={() => setCreateProfile("research")}
            >
              New Research Workspace
            </Button>
            <Button
              variant="primary"
              icon={<Plus className="h-4 w-4" />}
              onClick={() => setCreateProfile("project")}
            >
              New Project Workspace
            </Button>
          </div>
        </div>

        <div className="mt-3 flex flex-wrap items-center gap-2">
          <label className="relative min-w-[240px] flex-1 max-w-md">
            <span className="sr-only">Search Workspaces</span>
            <Search
              className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-text-muted"
              aria-hidden="true"
            />
            <input
              type="search"
              aria-label="Search Workspaces"
              value={searchQuery}
              onChange={(event) => setSearchQuery(event.target.value)}
              className={[
                "w-full rounded-md border border-border bg-surface",
                "py-2 pl-9 pr-3 text-sm text-text outline-none",
                "focus:border-primary focus:ring-2 focus:ring-primary/20"
              ].join(" ")}
              placeholder="Search by name or id"
            />
          </label>
          <Button
            size="sm"
            variant={profileFilter === "all" ? "primary" : "outline"}
            onClick={() => setProfileFilter("all")}
          >
            All
          </Button>
          <Button
            size="sm"
            variant={profileFilter === "research" ? "primary" : "outline"}
            onClick={() => setProfileFilter("research")}
          >
            Research
          </Button>
          <Button
            size="sm"
            variant={profileFilter === "project" ? "primary" : "outline"}
            onClick={() => setProfileFilter("project")}
          >
            Project
          </Button>
          <Button
            size="sm"
            variant={attentionFilter === "needs_attention" ? "danger" : "outline"}
            onClick={() =>
              setAttentionFilter((current) =>
                current === "needs_attention" ? "all" : "needs_attention"
              )
            }
          >
            Needs attention
          </Button>
          <label className="ml-auto inline-flex items-center gap-2 text-sm text-text-muted">
            <input
              type="checkbox"
              checked={showArchived}
              onChange={(event) => setShowArchived(event.target.checked)}
            />
            Show archived
          </label>
          <Button
            size="sm"
            variant="ghost"
            icon={<RefreshCw className="h-4 w-4" />}
            onClick={() => void loadWorkspaces()}
            loading={loading}
          >
            Refresh
          </Button>
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-hidden">
        {loading ? (
          <div className="p-6 text-sm text-text-muted">Loading Workspaces</div>
        ) : error ? (
          <div className="p-4">
            <Alert
              variant="error"
              title="Workspaces are unavailable"
              action={{ label: "Retry", onClick: () => void loadWorkspaces() }}
            >
              Reconnect to your tldw server to manage Workspaces.
            </Alert>
          </div>
        ) : items.length === 0 ? (
          <div className="p-6">
            <div className="max-w-xl rounded-lg border border-border bg-surface p-5">
              <h2 className="text-base font-semibold text-text">
                No server-backed Workspaces yet
              </h2>
              <p className="mt-2 text-sm text-text-muted">
                Create a Workspace here when you want a durable server record for
                research sources, notes, project files, and future agent sessions.
              </p>
              <div className="mt-4 flex flex-wrap gap-2">
                <Button onClick={() => setCreateProfile("research")}>
                  Create Research Workspace
                </Button>
                <Button
                  variant="primary"
                  onClick={() => setCreateProfile("project")}
                >
                  Create Project Workspace
                </Button>
              </div>
            </div>
          </div>
        ) : (
          <>
            <div className="flex items-center justify-between gap-3 px-4 py-2 text-sm text-text-muted">
              <span>
                Showing {filteredItems.length} of {items.length} Workspaces
              </span>
              {partialError && (
                <Badge variant="warning" outline>
                  {partialError}
                </Badge>
              )}
            </div>
            {filteredItems.length === 0 ? (
              <div className="p-6 text-sm text-text-muted">
                No Workspaces match the current filters.
              </div>
            ) : (
              <div className="grid h-full min-h-0 grid-cols-1 lg:grid-cols-[minmax(0,1fr)_minmax(320px,380px)]">
                <WorkspaceList
                  items={filteredItems}
                  selectedId={selectedItem?.id ?? null}
                  onSelect={(item) => setSelectedItemId(item.id)}
                  onOpen={(item) =>
                    navigate(
                      buildResearchWorkspaceReturnPath({
                        sourceWorkspaceId: item.id
                      })
                    )
                  }
                  onEdit={setEditingItem}
                  onArchive={(item) => void updateArchived(item, true)}
                  onUnarchive={(item) => void updateArchived(item, false)}
                />
                {selectedItem && (
                  <WorkspaceProjectRootPanel
                    item={selectedItem}
                    onWorkspaceUpdated={(workspace) => {
                      setItems((current) => mergeWorkspaceItem(current, workspace))
                      setSelectedItemId(workspace.id)
                    }}
                    onRootsUpdated={() => void loadWorkspaces()}
                    onRefreshContext={() => void loadWorkspaces()}
                  />
                )}
              </div>
            )}
          </>
        )}
      </div>

      {mutationError && !createProfile && !editingItem && (
        <div className="border-t border-border p-3">
          <Alert variant="error" title={mutationError} />
        </div>
      )}

      <WorkspaceCreateDialog
        open={createProfile != null}
        profile={createProfile ?? "research"}
        submitting={mutating}
        error={mutationError}
        onClose={() => {
          setCreateProfile(null)
          setMutationError(null)
        }}
        onSubmit={createWorkspace}
      />
      <WorkspaceMetadataDialog
        item={editingItem}
        submitting={mutating}
        error={mutationError}
        onClose={() => {
          setEditingItem(null)
          setMutationError(null)
        }}
        onSubmit={updateWorkspaceName}
      />
    </section>
  )
}
