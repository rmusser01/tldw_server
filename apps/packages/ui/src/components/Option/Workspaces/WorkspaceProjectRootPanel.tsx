import React from "react"
import {
  FolderCog,
  FolderOpen,
  RefreshCw,
  Server,
  UploadCloud
} from "lucide-react"
import { Button } from "@/components/Common/Button"
import { Alert, Badge } from "@/components/ui/primitives"
import { useTldwApiClient } from "@/hooks/useTldwApiClient"
import type {
  WorkspaceApiResponse,
  WorkspaceOperationResponse,
  WorkspaceRootsResponse
} from "@/services/tldw/domains/workspace-api"
import type { WorkspaceManagerItem } from "./workspace-manager-models"

type RootSetupMode = "host_local" | "sandbox_volume" | null

const INITIAL_OPERATION_POLL_DELAY_MS = 750
const MAX_OPERATION_POLL_DELAY_MS = 3000

type WorkspaceProjectRootPanelProps = {
  item: WorkspaceManagerItem
  active?: boolean
  signal?: AbortSignal
  onWorkspaceUpdated?: (workspace: WorkspaceApiResponse) => void
  onRootsUpdated?: (roots: WorkspaceRootsResponse) => void
  onRefreshContext?: () => void
}

const terminalOperationStatuses = new Set([
  "succeeded",
  "failed",
  "conflicted",
  "expired"
])

const operationLabel = (
  operation: WorkspaceOperationResponse | null
): string => {
  if (!operation) return "No active root operation"
  return operation.status.replace(/_/g, " ")
}

const rootStateCopy = (
  state: WorkspaceManagerItem["projectRoot"]["state"]
): string => {
  if (state === "not_configured")
    return "Choose a primary root before scanning files."
  if (state === "unavailable")
    return "Root is unavailable. Refresh context or attach a reachable root."
  if (state === "failed")
    return "Root setup failed. Review diagnostics, then retry the root action."
  if (state === "cleanup_pending")
    return "Previous root cleanup is still pending before replacement."
  if (state === "missing" || state === "detached")
    return "Root is detached or missing. Attach it again to continue."
  if (state === "provisioning") return "Root provisioning is in progress."
  if (state === "archived") return "This root is archived."
  return "Root is attached."
}

const fallbackEntropy = (): string => {
  const cryptoApi = globalThis.crypto
  if (typeof cryptoApi?.getRandomValues === "function") {
    const values = new Uint32Array(2)
    cryptoApi.getRandomValues(values)
    return `${values[0].toString(36)}-${values[1].toString(36)}`
  }
  const perf = globalThis.performance
  const perfTime = typeof perf?.now === "function" ? perf.now() : 0
  return `${Date.now().toString(36)}-${perfTime.toString(36)}-${Math.random()
    .toString(36)
    .slice(2)}`
}

const generateSandboxRootIdempotencyKey = (workspaceId: string): string => {
  const randomUUID = globalThis.crypto?.randomUUID
  const entropy =
    typeof randomUUID === "function"
      ? randomUUID.call(globalThis.crypto)
      : fallbackEntropy()
  return `workspace-sandbox-root:${workspaceId}:${entropy}`
}

export const WorkspaceProjectRootPanel = ({
  item,
  active = true,
  signal,
  onWorkspaceUpdated,
  onRootsUpdated,
  onRefreshContext
}: WorkspaceProjectRootPanelProps) => {
  const api = useTldwApiClient()
  const [mode, setMode] = React.useState<RootSetupMode>(null)
  const [hostPath, setHostPath] = React.useState("")
  const [hostDisplayName, setHostDisplayName] = React.useState("")
  const [sandboxDisplayName, setSandboxDisplayName] = React.useState("")
  const [requestedRuntime, setRequestedRuntime] = React.useState("")
  const [observedOperation, setObservedOperation] =
    React.useState<WorkspaceOperationResponse | null>(null)
  const [operationPollDelayMs, setOperationPollDelayMs] = React.useState(
    INITIAL_OPERATION_POLL_DELAY_MS
  )
  const [submitting, setSubmitting] = React.useState(false)
  const [inventoryScanning, setInventoryScanning] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)
  const [inventoryMessage, setInventoryMessage] = React.useState<string | null>(
    null
  )
  const lifetime = React.useMemo(
    () => ({ active, workspaceId: item.id, signal }),
    [active, item.id, signal]
  )
  const currentLifetime = React.useRef<object | null>(null)
  const enabled = active && !signal?.aborted
  const isCurrent = React.useCallback(
    () => active && !signal?.aborted && currentLifetime.current === lifetime,
    [active, signal, lifetime]
  )

  // Retain editor inputs while retiring all asynchronous work from the old view.
  React.useEffect(() => {
    currentLifetime.current = lifetime
    setObservedOperation(null)
    setOperationPollDelayMs(INITIAL_OPERATION_POLL_DELAY_MS)
    setSubmitting(false)
    setInventoryScanning(false)
    setError(null)
    setInventoryMessage(null)
    return () => {
      currentLifetime.current = null
    }
  }, [lifetime])

  React.useEffect(() => {
    setMode(null)
    setHostPath("")
    setHostDisplayName("")
    setSandboxDisplayName("")
    setRequestedRuntime("")
    setObservedOperation(null)
    setOperationPollDelayMs(INITIAL_OPERATION_POLL_DELAY_MS)
    setError(null)
    setInventoryMessage(null)
  }, [item.id])

  const contextOperation = React.useMemo(
    () =>
      item.activeOperations.find((operation) =>
        operation.command.toLowerCase().includes("sandbox")
      ) ?? null,
    [item.activeOperations]
  )
  const activeOperation = observedOperation ?? contextOperation
  const hasActiveOperation =
    activeOperation != null &&
    !terminalOperationStatuses.has(activeOperation.status)
  const inventoryAvailable = item.projectRoot.fileInventory.available === true
  const inventoryUnavailableCopy =
    item.projectRoot.backend === "sandbox_volume" && !inventoryAvailable
      ? "File inventory is unavailable until the sandbox-managed root is mounted."
      : item.profile === "project"
        ? "Attach an available primary root before scanning files."
        : "Upgrade to a Project Workspace before scanning files."

  const refreshOperation = React.useCallback(
    async (operation: WorkspaceOperationResponse) => {
      if (!isCurrent() || terminalOperationStatuses.has(operation.status))
        return
      try {
        const latest = await api.getWorkspaceOperation(
          item.id,
          operation.operation_id
        )
        if (!isCurrent()) return
        setObservedOperation(latest)
        setOperationPollDelayMs((current) =>
          terminalOperationStatuses.has(latest.status)
            ? INITIAL_OPERATION_POLL_DELAY_MS
            : Math.min(current * 2, MAX_OPERATION_POLL_DELAY_MS)
        )
      } catch (caught) {
        if (!isCurrent()) return
        setError(
          caught instanceof Error
            ? caught.message
            : "Failed to refresh Workspace operation status."
        )
      }
    },
    [api, item.id, isCurrent]
  )

  React.useEffect(() => {
    if (
      !isCurrent() ||
      !activeOperation ||
      terminalOperationStatuses.has(activeOperation.status)
    ) {
      return undefined
    }
    const timeout = window.setTimeout(() => {
      void refreshOperation(activeOperation)
    }, operationPollDelayMs)
    const cancel = () => window.clearTimeout(timeout)
    signal?.addEventListener("abort", cancel, { once: true })
    return () => {
      cancel()
      signal?.removeEventListener("abort", cancel)
    }
  }, [
    activeOperation,
    operationPollDelayMs,
    refreshOperation,
    isCurrent,
    signal
  ])

  const upgradeWorkspace = async (): Promise<void> => {
    if (!isCurrent()) return
    setSubmitting(true)
    setError(null)
    try {
      const workspace = await api.patchWorkspace(item.id, {
        workspace_profile: "project",
        version: item.version
      })
      if (!isCurrent()) return
      onWorkspaceUpdated?.(workspace)
    } catch (caught) {
      if (!isCurrent()) return
      setError(
        caught instanceof Error
          ? caught.message
          : "Failed to upgrade Workspace."
      )
    } finally {
      if (isCurrent()) setSubmitting(false)
    }
  }

  const attachHostRoot = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (!isCurrent()) return
    const absoluteRoot = hostPath.trim()
    if (!absoluteRoot) {
      setError("Root path is required.")
      return
    }
    setSubmitting(true)
    setError(null)
    try {
      const roots = await api.attachWorkspacePrimaryRoot(item.id, {
        backend: "host_local",
        absolute_root: absoluteRoot,
        display_name: hostDisplayName.trim() || null,
        expected_workspace_version: item.version
      })
      if (!isCurrent()) return
      onRootsUpdated?.(roots)
    } catch (caught) {
      if (!isCurrent()) return
      setError(
        caught instanceof Error
          ? caught.message
          : "Failed to attach host-local root."
      )
    } finally {
      if (isCurrent()) setSubmitting(false)
    }
  }

  const provisionSandboxRoot = async (
    event: React.FormEvent<HTMLFormElement>
  ) => {
    event.preventDefault()
    if (!isCurrent()) return
    setSubmitting(true)
    setError(null)
    try {
      const response = await api.provisionWorkspaceSandboxRoot(
        item.id,
        {
          display_name: sandboxDisplayName.trim() || null,
          requested_runtime: requestedRuntime.trim() || null,
          expected_workspace_version: item.version
        },
        generateSandboxRootIdempotencyKey(item.id)
      )
      if (!isCurrent()) return
      setOperationPollDelayMs(INITIAL_OPERATION_POLL_DELAY_MS)
      setObservedOperation(response.operation)
      if (response.primary_root) {
        onRootsUpdated?.({
          workspace_id: response.workspace_id,
          workspace_profile: response.workspace_profile,
          primary_root: response.primary_root,
          roots: [response.primary_root]
        })
      }
      void refreshOperation(response.operation)
    } catch (caught) {
      if (!isCurrent()) return
      setError(
        caught instanceof Error
          ? caught.message
          : "Failed to provision sandbox root."
      )
    } finally {
      if (isCurrent()) setSubmitting(false)
    }
  }

  const scanInventory = async (): Promise<void> => {
    if (!isCurrent() || !inventoryAvailable) return
    setInventoryScanning(true)
    setError(null)
    setInventoryMessage(null)
    try {
      const status = await api.queueWorkspaceFileInventoryScan(item.id, {
        force: true
      })
      if (!isCurrent()) return
      setInventoryMessage(`File inventory scan ${status.state}.`)
    } catch (caught) {
      if (!isCurrent()) return
      setError(
        caught instanceof Error
          ? caught.message
          : "Failed to queue file inventory scan."
      )
    } finally {
      if (isCurrent()) setInventoryScanning(false)
    }
  }

  if (item.profile === "research") {
    return (
      <section className="h-full min-h-0 border-l border-border bg-surface p-4">
        <div className="flex items-start gap-3">
          <FolderCog
            className="mt-0.5 h-5 w-5 text-primary"
            aria-hidden="true"
          />
          <div className="min-w-0 flex-1">
            <h2 className="text-base font-semibold text-text">Project root</h2>
            <p className="mt-1 text-sm text-text-muted">
              Upgrade this Research Workspace when it needs files, a primary
              root, and sandbox-backed project execution.
            </p>
            <Button
              className="mt-4"
              variant="primary"
              loading={submitting}
              disabled={!enabled}
              onClick={() => void upgradeWorkspace()}
            >
              Upgrade to Project Workspace
            </Button>
            {error && <Alert className="mt-3" variant="error" title={error} />}
          </div>
        </div>
      </section>
    )
  }

  return (
    <section className="h-full min-h-0 overflow-auto border-l border-border bg-surface p-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold text-text">Project root</h2>
          <p className="mt-1 text-sm text-text-muted">
            Configure one primary root for files, sandbox sessions, and future
            agent runs.
          </p>
        </div>
        <Button
          size="sm"
          variant="ghost"
          icon={<RefreshCw className="h-4 w-4" />}
          disabled={!enabled}
          onClick={() => {
            if (isCurrent()) onRefreshContext?.()
          }}
        >
          Refresh
        </Button>
      </div>

      <div
        className="mt-4 rounded-md border border-border bg-bg p-3"
        data-testid="workspace-root-summary"
      >
        <div className="flex items-center justify-between gap-2">
          <div className="min-w-0">
            <div className="truncate text-sm font-medium text-text">
              {item.projectRoot.displayName || "No primary root"}
            </div>
            <div className="mt-1 text-xs text-text-muted">
              {item.projectRoot.backend === "host_local"
                ? "Path hidden"
                : item.projectRoot.backend === "sandbox_volume"
                  ? "Sandbox-managed volume"
                  : "No root attached"}
            </div>
          </div>
          <Badge
            variant={
              item.projectRoot.state === "attached" ? "success" : "warning"
            }
          >
            {item.projectRoot.state.replace(/_/g, " ")}
          </Badge>
        </div>
        <p className="mt-3 text-xs text-text-muted">
          {rootStateCopy(item.projectRoot.state)}
        </p>
      </div>

      {(hasActiveOperation || item.projectRoot.state === "provisioning") && (
        <Alert
          className="mt-3"
          variant="info"
          title="Provisioning sandbox root"
        >
          <span>{operationLabel(activeOperation)}</span>
        </Alert>
      )}

      {item.projectRoot.state !== "attached" && (
        <div className="mt-4">
          <div className="flex flex-wrap gap-2">
            <Button
              variant={mode === "host_local" ? "primary" : "outline"}
              disabled={!enabled}
              icon={<FolderOpen className="h-4 w-4" />}
              onClick={() => setMode("host_local")}
            >
              Host-local root
            </Button>
            <Button
              variant={mode === "sandbox_volume" ? "primary" : "outline"}
              disabled={!enabled}
              icon={<Server className="h-4 w-4" />}
              onClick={() => setMode("sandbox_volume")}
            >
              Sandbox-managed root
            </Button>
          </div>

          {mode === "host_local" && (
            <form className="mt-4 space-y-3" onSubmit={attachHostRoot}>
              <label className="block">
                <span className="text-xs font-medium text-text-muted">
                  Root path
                </span>
                <input
                  className="mt-1 w-full rounded-md border border-border bg-bg px-3 py-2 text-sm text-text outline-none focus:border-primary focus:ring-2 focus:ring-primary/20"
                  aria-label="Root path"
                  disabled={!enabled}
                  value={hostPath}
                  onChange={(event) => setHostPath(event.target.value)}
                  placeholder="/path/to/project"
                />
              </label>
              <label className="block">
                <span className="text-xs font-medium text-text-muted">
                  Display name
                </span>
                <input
                  className="mt-1 w-full rounded-md border border-border bg-bg px-3 py-2 text-sm text-text outline-none focus:border-primary focus:ring-2 focus:ring-primary/20"
                  aria-label="Display name"
                  value={hostDisplayName}
                  disabled={!enabled}
                  onChange={(event) => setHostDisplayName(event.target.value)}
                  placeholder="Project repo"
                />
              </label>
              <Button
                htmlType="submit"
                variant="primary"
                loading={submitting}
                disabled={!enabled}
              >
                Attach host-local root
              </Button>
            </form>
          )}

          {mode === "sandbox_volume" && (
            <form className="mt-4 space-y-3" onSubmit={provisionSandboxRoot}>
              <label className="block">
                <span className="text-xs font-medium text-text-muted">
                  Display name
                </span>
                <input
                  className="mt-1 w-full rounded-md border border-border bg-bg px-3 py-2 text-sm text-text outline-none focus:border-primary focus:ring-2 focus:ring-primary/20"
                  aria-label="Display name"
                  value={sandboxDisplayName}
                  disabled={!enabled}
                  onChange={(event) =>
                    setSandboxDisplayName(event.target.value)
                  }
                  placeholder="Project sandbox"
                />
              </label>
              <label className="block">
                <span className="text-xs font-medium text-text-muted">
                  Requested runtime
                </span>
                <input
                  className="mt-1 w-full rounded-md border border-border bg-bg px-3 py-2 text-sm text-text outline-none focus:border-primary focus:ring-2 focus:ring-primary/20"
                  aria-label="Requested runtime"
                  disabled={!enabled}
                  value={requestedRuntime}
                  onChange={(event) => setRequestedRuntime(event.target.value)}
                  placeholder="python"
                />
              </label>
              <Button
                htmlType="submit"
                variant="primary"
                loading={submitting}
                disabled={!enabled}
                icon={<UploadCloud className="h-4 w-4" />}
              >
                Provision sandbox root
              </Button>
            </form>
          )}
        </div>
      )}

      <div className="mt-5 border-t border-border pt-4">
        <div className="flex items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold text-text">File inventory</h3>
            <p className="mt-1 text-xs text-text-muted">
              {inventoryAvailable
                ? `${item.projectRoot.fileInventory.indexedFileCount ?? 0}/${item.projectRoot.fileInventory.totalFileCount ?? 0} files indexed`
                : inventoryUnavailableCopy}
            </p>
          </div>
          <Button
            size="sm"
            variant="outline"
            loading={inventoryScanning}
            disabled={!enabled || !inventoryAvailable}
            onClick={() => void scanInventory()}
          >
            Scan files
          </Button>
        </div>
        {inventoryMessage && (
          <Alert className="mt-3" variant="success" title={inventoryMessage} />
        )}
      </div>

      {error && <Alert className="mt-3" variant="error" title={error} />}
    </section>
  )
}
