import React from "react"
import { Plus, RefreshCw, Search } from "lucide-react"
import { useNavigate } from "react-router-dom"
import { Button } from "@/components/Common/Button"
import { Alert, Badge } from "@/components/ui/primitives"
import { useTldwApiClient } from "@/hooks/useTldwApiClient"
import { buildResearchWorkspaceReturnPath } from "@/routes/route-paths"
import {
  createOwnedWorkspaceDirectoryContext,
  createOwnedWorkspaceLifecycleContext
} from "@/services/owned-workspace-opening"
import { COOKIE_SESSION_CONFIG_KEY } from "@/services/tldw/browser-networking"
import {
  MANUAL_SESSION_KEY,
  REFRESH_ROTATION_KEY
} from "@/services/tldw/single-user-credential"
import type { OwnedWorkspaceScope } from "@/store/owned-workspace-state"
import type {
  WorkspaceApiResponse,
  WorkspaceProfile
} from "@/services/tldw/domains/workspace-api"
import { WorkspaceCreateDialog } from "./WorkspaceCreateDialog"
import { WorkspaceList } from "./WorkspaceList"
import { WorkspaceMetadataDialog } from "./WorkspaceMetadataDialog"
import { WorkspaceProjectRootPanel } from "./WorkspaceProjectRootPanel"
import { WorkspaceReconciliationPanel } from "./WorkspaceReconciliationPanel"
import {
  normalizeWorkspaceManagerItem,
  type WorkspaceManagerAttention,
  type WorkspaceManagerItem
} from "./workspace-manager-models"

type ProfileFilter = "all" | WorkspaceProfile
type AttentionFilter = "all" | "needs_attention"
type DialogProfile = WorkspaceProfile | null
type Directory = { scope: OwnedWorkspaceScope; request: AbortController }
const sameScope = (left: OwnedWorkspaceScope, right: OwnedWorkspaceScope) =>
  left.serverBase === right.serverBase &&
  left.principalId === right.principalId &&
  left.organizationId === right.organizationId
const LIFECYCLE_REVIEW_MESSAGE =
  "Workspace change could not be confirmed. Refresh and review before trying again."

const isAccountError = (error: unknown): boolean => {
  if (!error || typeof error !== "object") return false
  return (
    ("status" in error && [401, 403, 412].includes(Number(error.status))) ||
    ("reason" in error && error.reason === "denied")
  )
}

const createWorkspaceId = (): string => {
  const cryptoApi = globalThis.crypto
  const randomUUID = cryptoApi?.randomUUID
  if (typeof randomUUID === "function") return randomUUID.call(cryptoApi)
  if (typeof cryptoApi?.getRandomValues === "function") {
    const values = new Uint32Array(2)
    cryptoApi.getRandomValues(values)
    return `workspace-${values[0].toString(36)}-${values[1].toString(36)}`
  }
  const perfTime =
    typeof globalThis.performance?.now === "function"
      ? globalThis.performance.now()
      : 0
  return `workspace-${Date.now().toString(36)}-${perfTime.toString(36)}-${Math.random()
    .toString(36)
    .slice(2)}`
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
    return options.prependIfMissing
      ? [normalized, ...items]
      : [...items, normalized]
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
  const [editingItem, setEditingItem] =
    React.useState<WorkspaceManagerItem | null>(null)
  const [selectedItemId, setSelectedItemId] = React.useState<string | null>(
    null
  )
  const [mutationError, setMutationError] = React.useState<string | null>(null)
  const [mutating, setMutating] = React.useState(false)
  const [directory, setDirectory] = React.useState<Directory | null>(null)
  const viewScope = React.useRef<OwnedWorkspaceScope | null>(null)
  const requestRef = React.useRef<AbortController | null>(null)
  const mutationPending = React.useRef(false)
  const lifecyclePending = React.useRef<OwnedWorkspaceScope | null>(null)
  const reviewRequired = React.useRef<OwnedWorkspaceScope | null>(null)
  const [needsReview, setNeedsReview] = React.useState(false)

  const stopRequests = React.useCallback(() => {
    if (lifecyclePending.current)
      reviewRequired.current = lifecyclePending.current
    lifecyclePending.current = null
    requestRef.current?.abort()
    requestRef.current = null
    mutationPending.current = false
  }, [])

  const discardDirectory = React.useCallback(() => {
    viewScope.current = null
    setDirectory(null)
    setItems([])
    setSelectedItemId(null)
    setCreateProfile(null)
    setEditingItem(null)
    setMutationError(null)
    setMutating(false)
    setNeedsReview(false)
  }, [])

  const clearDirectory = React.useCallback(() => {
    stopRequests()
    discardDirectory()
  }, [discardDirectory, stopRequests])

  const suspendDirectory = React.useCallback(() => {
    stopRequests()
    setLoading(true)
    setMutating(false)
  }, [stopRequests])

  const loadWorkspaces = React.useCallback(
    async (explicitReview = true) => {
      suspendDirectory()
      const request = new AbortController()
      requestRef.current = request
      const current = () =>
        requestRef.current === request && !request.signal.aborted
      setLoading(true)
      setError(null)
      setPartialError(null)
      try {
        const context = await createOwnedWorkspaceDirectoryContext(
          request.signal
        )
        if (!current()) return
        if (viewScope.current && !sameScope(viewScope.current, context.scope))
          discardDirectory()
        const response = await context.list()
        if (!current()) return
        const workspaces = Array.isArray(response.items) ? response.items : []
        const normalizedResults = await Promise.allSettled(
          workspaces.map(async (workspace) => {
            const details = await context.getContext(workspace.id)
            return normalizeWorkspaceManagerItem(workspace, details)
          })
        )
        if (!current()) return
        const accountFailure = normalizedResults.find(
          (result) =>
            result.status === "rejected" && isAccountError(result.reason)
        )
        if (accountFailure?.status === "rejected") throw accountFailure.reason
        const normalized = normalizedResults.map((result, index) =>
          result.status === "fulfilled"
            ? result.value
            : normalizeWorkspaceManagerItem(workspaces[index])
        )
        if (normalizedResults.some((result) => result.status === "rejected")) {
          setPartialError("Some Workspace details could not load.")
        }
        setItems(normalized)
        viewScope.current = context.scope
        setDirectory({ scope: context.scope, request })
        const previousFailure = reviewRequired.current
        const stillNeedsReview =
          !explicitReview &&
          previousFailure != null &&
          sameScope(previousFailure, context.scope)
        if (!stillNeedsReview) reviewRequired.current = null
        setNeedsReview(stillNeedsReview)
        setMutationError(stillNeedsReview ? LIFECYCLE_REVIEW_MESSAGE : null)
        setSelectedItemId((current) =>
          current && normalized.some((item) => item.id === current)
            ? current
            : (normalized[0]?.id ?? null)
        )
      } catch (caught) {
        if (!current()) return
        if (isAccountError(caught)) discardDirectory()
        setError(errorText(caught))
        request.abort()
      } finally {
        if (requestRef.current === request) setLoading(false)
      }
    },
    [discardDirectory, suspendDirectory]
  )

  React.useEffect(() => {
    const start = () => void loadWorkspaces(false)
    const suspend = suspendDirectory
    const auth = (event: Event) => {
      if ((event as CustomEvent<{ kind?: string }>).detail?.kind === "logout") {
        clearDirectory()
        setError("Workspace access ended.")
        setLoading(false)
      } else start()
    }
    const visibility = () => {
      if (document.visibilityState === "hidden") suspend()
      else start()
    }
    const storage = (event: StorageEvent) => {
      if (
        event.key === null ||
        [
          "tldwConfig",
          MANUAL_SESSION_KEY,
          REFRESH_ROTATION_KEY,
          COOKIE_SESSION_CONFIG_KEY
        ].includes(event.key)
      )
        start()
    }
    window.addEventListener("tldw:config-updated", start)
    window.addEventListener("tldw:auth-principal-changed", auth)
    window.addEventListener("focus", start)
    window.addEventListener("pageshow", start)
    window.addEventListener("pagehide", suspend)
    window.addEventListener("storage", storage)
    document.addEventListener("visibilitychange", visibility)
    start()
    return () => {
      stopRequests()
      window.removeEventListener("tldw:config-updated", start)
      window.removeEventListener("tldw:auth-principal-changed", auth)
      window.removeEventListener("focus", start)
      window.removeEventListener("pageshow", start)
      window.removeEventListener("pagehide", suspend)
      window.removeEventListener("storage", storage)
      document.removeEventListener("visibilitychange", visibility)
    }
  }, [clearDirectory, loadWorkspaces, stopRequests, suspendDirectory])

  // Each render's callbacks retain their directory ticket, including child callbacks.
  const isCurrentDirectory = () =>
    directory != null &&
    requestRef.current === directory.request &&
    !directory.request.signal.aborted

  const guardDirectoryEvent = (event: React.SyntheticEvent) => {
    if (isCurrentDirectory()) return
    event.preventDefault()
    event.stopPropagation()
  }
  const viewKey = directory ? JSON.stringify(directory.scope) : "unverified"

  const beginMutation = () => {
    if (
      !isCurrentDirectory() ||
      mutationPending.current ||
      reviewRequired.current
    )
      return false
    mutationPending.current = true
    setMutating(true)
    setMutationError(null)
    return true
  }

  const finishMutation = () => {
    if (!isCurrentDirectory()) return
    mutationPending.current = false
    lifecyclePending.current = null
    setMutating(false)
  }

  const openCreateDialog = (profile: WorkspaceProfile) => {
    if (
      isCurrentDirectory() &&
      !mutationPending.current &&
      !reviewRequired.current
    )
      setCreateProfile(profile)
  }

  const filteredItems = React.useMemo(() => {
    const query = searchQuery.trim().toLowerCase()
    return items.filter((item) => {
      if (!showArchived && item.archived) return false
      if (profileFilter !== "all" && item.profile !== profileFilter)
        return false
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
      filteredItems.find((item) => item.id === selectedItemId) ??
      filteredItems[0]
    )
  }, [filteredItems, selectedItemId])

  const createWorkspace = async (
    name: string,
    profile: WorkspaceProfile
  ): Promise<void> => {
    if (!beginMutation()) return
    try {
      const workspace = await api.upsertWorkspace(createWorkspaceId(), {
        name,
        study_materials_policy: "workspace",
        workspace_profile: profile
      })
      if (!isCurrentDirectory()) return
      setItems((current) =>
        mergeWorkspaceItem(current, workspace, { prependIfMissing: true })
      )
      setSelectedItemId(workspace.id)
      setCreateProfile(null)
    } catch (caught) {
      if (isCurrentDirectory()) setMutationError(errorText(caught))
    } finally {
      finishMutation()
    }
  }

  const updateWorkspaceName = async (
    item: WorkspaceManagerItem,
    name: string
  ): Promise<void> => {
    if (!beginMutation()) return
    try {
      const workspace = await api.patchWorkspace(item.id, {
        name,
        version: item.version
      })
      if (!isCurrentDirectory()) return
      setItems((current) => mergeWorkspaceItem(current, workspace))
      setSelectedItemId(workspace.id)
      setEditingItem(null)
    } catch (caught) {
      if (isCurrentDirectory()) setMutationError(errorText(caught))
    } finally {
      finishMutation()
    }
  }

  const updateArchived = async (
    item: WorkspaceManagerItem,
    archived: boolean
  ): Promise<void> => {
    if (!directory || !beginMutation()) return
    try {
      const context = await createOwnedWorkspaceLifecycleContext(
        item.id,
        directory.scope,
        directory.request.signal
      )
      if (!isCurrentDirectory()) return
      lifecyclePending.current = directory.scope
      const workspace = await context.setArchived(archived, item.version)
      if (!isCurrentDirectory()) return
      setItems((current) => mergeWorkspaceItem(current, workspace))
      setSelectedItemId(workspace.id)
    } catch (caught) {
      if (!isCurrentDirectory()) return
      if (isAccountError(caught)) {
        clearDirectory()
        setError("Workspace access changed.")
        return
      }
      // A failed write may have committed. Never replay its version or intent.
      reviewRequired.current = directory.scope
      setNeedsReview(true)
      setMutationError(LIFECYCLE_REVIEW_MESSAGE)
    } finally {
      finishMutation()
    }
  }

  const handleServerWorkspaceCreated = (
    workspace: WorkspaceApiResponse
  ): void => {
    if (!isCurrentDirectory()) return
    setItems((current) =>
      mergeWorkspaceItem(current, workspace, { prependIfMissing: true })
    )
    setSelectedItemId(workspace.id)
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
              disabled={loading || !!error || mutating || needsReview}
              onClick={() => openCreateDialog("research")}
            >
              New Research Workspace
            </Button>
            <Button
              variant="primary"
              icon={<Plus className="h-4 w-4" />}
              disabled={loading || !!error || mutating || needsReview}
              onClick={() => openCreateDialog("project")}
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
            variant={
              attentionFilter === "needs_attention" ? "danger" : "outline"
            }
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
        ) : null}
        <div
          key={viewKey}
          hidden={loading || !!error}
          className="h-full min-h-0"
          onClickCapture={guardDirectoryEvent}
          onSubmitCapture={guardDirectoryEvent}
        >
          {!directory ? null : items.length === 0 ? (
            <>
              <WorkspaceReconciliationPanel
                serverWorkspaces={items}
                onServerWorkspaceCreated={handleServerWorkspaceCreated}
              />
              <div className="p-6">
                <div className="max-w-xl rounded-lg border border-border bg-surface p-5">
                  <h2 className="text-base font-semibold text-text">
                    No server-backed Workspaces yet
                  </h2>
                  <p className="mt-2 text-sm text-text-muted">
                    Create a Workspace here when you want a durable server
                    record for research sources, notes, project files, and
                    future agent sessions.
                  </p>
                  <div className="mt-4 flex flex-wrap gap-2">
                    <Button onClick={() => openCreateDialog("research")}>
                      Create Research Workspace
                    </Button>
                    <Button
                      variant="primary"
                      onClick={() => openCreateDialog("project")}
                    >
                      Create Project Workspace
                    </Button>
                  </div>
                </div>
              </div>
            </>
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
              <WorkspaceReconciliationPanel
                serverWorkspaces={items}
                onServerWorkspaceCreated={handleServerWorkspaceCreated}
              />
              {filteredItems.length === 0 ? (
                <div className="p-6 text-sm text-text-muted">
                  No Workspaces match the current filters.
                </div>
              ) : (
                <div className="grid h-full min-h-0 grid-cols-1 lg:grid-cols-[minmax(0,1fr)_minmax(320px,380px)]">
                  <WorkspaceList
                    items={filteredItems}
                    selectedId={selectedItem?.id ?? null}
                    onSelect={(item) => {
                      if (isCurrentDirectory()) setSelectedItemId(item.id)
                    }}
                    onOpen={(item) => {
                      if (!isCurrentDirectory()) return
                      navigate(
                        buildResearchWorkspaceReturnPath({
                          sourceWorkspaceId: item.id
                        })
                      )
                    }}
                    onEdit={(item) => {
                      if (
                        isCurrentDirectory() &&
                        !mutationPending.current &&
                        !reviewRequired.current
                      )
                        setEditingItem(item)
                    }}
                    onArchive={(item) => void updateArchived(item, true)}
                    onUnarchive={(item) => void updateArchived(item, false)}
                  />
                  {selectedItem && (
                    <WorkspaceProjectRootPanel
                      item={selectedItem}
                      active={!loading && !error && isCurrentDirectory()}
                      signal={directory.request.signal}
                      onWorkspaceUpdated={(workspace) => {
                        if (!isCurrentDirectory()) return
                        setItems((current) =>
                          mergeWorkspaceItem(current, workspace)
                        )
                        setSelectedItemId(workspace.id)
                      }}
                      onRootsUpdated={() => {
                        if (isCurrentDirectory()) void loadWorkspaces(false)
                      }}
                      onRefreshContext={() => {
                        if (isCurrentDirectory()) void loadWorkspaces()
                      }}
                    />
                  )}
                </div>
              )}
            </>
          )}
        </div>
      </div>

      {!loading &&
        !error &&
        mutationError &&
        !createProfile &&
        !editingItem && (
          <div className="border-t border-border p-3">
            <Alert
              variant="error"
              title={mutationError}
              action={
                needsReview
                  ? {
                      label: "Refresh and review",
                      onClick: () => void loadWorkspaces()
                    }
                  : undefined
              }
            />
          </div>
        )}

      <div
        key={viewKey}
        hidden={loading || !!error}
        onClickCapture={guardDirectoryEvent}
        onSubmitCapture={guardDirectoryEvent}
      >
        <WorkspaceCreateDialog
          open={createProfile != null}
          profile={createProfile ?? "research"}
          submitting={mutating}
          error={mutationError}
          onClose={() => {
            if (!isCurrentDirectory()) return
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
            if (!isCurrentDirectory()) return
            setEditingItem(null)
            setMutationError(null)
          }}
          onSubmit={updateWorkspaceName}
        />
      </div>
    </section>
  )
}
