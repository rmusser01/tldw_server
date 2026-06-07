import React from "react"
import { FolderCog, FolderOpen, RefreshCw, Server, UploadCloud } from "lucide-react"
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
  onWorkspaceUpdated?: (workspace: WorkspaceApiResponse) => void
  onRootsUpdated?: (roots: WorkspaceRootsResponse) => void
  onRefreshContext?: () => void
}

const terminalOperationStatuses = new Set(["succeeded", "failed", "conflicted", "expired"])

const operationLabel = (operation: WorkspaceOperationResponse | null): string => {
  if (!operation) return "No active root operation"
  return operation.status.replace(/_/g, " ")
}

const rootStateCopy = (state: WorkspaceManagerItem["projectRoot"]["state"]): string => {
  if (state === "not_configured") return "Choose a primary root before scanning files."
  if (state === "unavailable") return "Root is unavailable. Refresh context or attach a reachable root."
  if (state === "failed") return "Root setup failed. Review diagnostics, then retry the root action."
  if (state === "cleanup_pending") return "Previous root cleanup is still pending before replacement."
  if (state === "missing" || state === "detached") return "Root is detached or missing. Attach it again to continue."
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
  const [inventoryMessage, setInventoryMessage] = React.useState<string | null>(null)

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
    activeOperation != null && !terminalOperationStatuses.has(activeOperation.status)
  const inventoryAvailable = item.projectRoot.fileInventory.available === true
  const inventoryUnavailableCopy =
    item.projectRoot.backend === "sandbox_volume" && !inventoryAvailable
      ? "File inventory is unavailable until the sandbox-managed root is mounted."
      : item.profile === "project"
        ? "Attach an available primary root before scanning files."
        : "Upgrade to a Project Workspace before scanning files."

  const refreshOperation = React.useCallback(
    async (operation: WorkspaceOperationResponse) => {
      if (terminalOperationStatuses.has(operation.status)) return
      try {
        const latest = await api.getWorkspaceOperation(
          item.id,
          operation.operation_id
        )
        setObservedOperation(latest)
        setOperationPollDelayMs((current) =>
          terminalOperationStatuses.has(latest.status)
            ? INITIAL_OPERATION_POLL_DELAY_MS
            : Math.min(current * 2, MAX_OPERATION_POLL_DELAY_MS)
        )
      } catch (caught) {
        setError(
          caught instanceof Error
            ? caught.message
            : "Failed to refresh Workspace operation status."
        )
      }
    },
    [api, item.id]
  )

  React.useEffect(() => {
    if (!activeOperation || terminalOperationStatuses.has(activeOperation.status)) {
      return undefined
    }
    const timeout = window.setTimeout(() => {
      void refreshOperation(activeOperation)
    }, operationPollDelayMs)
    return () => window.clearTimeout(timeout)
  }, [activeOperation, operationPollDelayMs, refreshOperation])

  const upgradeWorkspace = async (): Promise<void> => {
    setSubmitting(true)
    setError(null)
    try {
      const workspace = await api.patchWorkspace(item.id, {
        workspace_profile: "project",
        version: item.version
      })
      onWorkspaceUpdated?.(workspace)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Failed to upgrade Workspace.")
    } finally {
      setSubmitting(false)
    }
  }

  const attachHostRoot = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
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
      onRootsUpdated?.(roots)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Failed to attach host-local root.")
    } finally {
      setSubmitting(false)
    }
  }

  const provisionSandboxRoot = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
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
      setError(caught instanceof Error ? caught.message : "Failed to provision sandbox root.")
    } finally {
      setSubmitting(false)
    }
  }

  const scanInventory = async (): Promise<void> => {
    if (!inventoryAvailable) return
    setInventoryScanning(true)
    setError(null)
    setInventoryMessage(null)
    try {
      const status = await api.queueWorkspaceFileInventoryScan(item.id, { force: true })
      setInventoryMessage(`File inventory scan ${status.state}.`)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Failed to queue file inventory scan.")
    } finally {
      setInventoryScanning(false)
    }
  }

  if (item.profile === "research") {
    return (
      <section className="h-full min-h-0 border-l border-border bg-surface p-4">
        <div className="flex items-start gap-3">
          <FolderCog className="mt-0.5 h-5 w-5 text-primary" aria-hidden="true" />
          <div className="min-w-0 flex-1">
            <h2 className="text-base font-semibold text-text">Project root</h2>
            <p className="mt-1 text-sm text-text-muted">
              Upgrade this Research Workspace when it needs files, a primary root,
              and sandbox-backed project execution.
            </p>
            <Button
              className="mt-4"
              variant="primary"
              loading={submitting}
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
          onClick={onRefreshContext}
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
          <Badge variant={item.projectRoot.state === "attached" ? "success" : "warning"}>
            {item.projectRoot.state.replace(/_/g, " ")}
          </Badge>
        </div>
        <p className="mt-3 text-xs text-text-muted">
          {rootStateCopy(item.projectRoot.state)}
        </p>
      </div>

      {(hasActiveOperation || item.projectRoot.state === "provisioning") && (
        <Alert className="mt-3" variant="info" title="Provisioning sandbox root">
          <span>{operationLabel(activeOperation)}</span>
        </Alert>
      )}

      {item.projectRoot.state !== "attached" && (
        <div className="mt-4">
          <div className="flex flex-wrap gap-2">
            <Button
              variant={mode === "host_local" ? "primary" : "outline"}
              icon={<FolderOpen className="h-4 w-4" />}
              onClick={() => setMode("host_local")}
            >
              Host-local root
            </Button>
            <Button
              variant={mode === "sandbox_volume" ? "primary" : "outline"}
              icon={<Server className="h-4 w-4" />}
              onClick={() => setMode("sandbox_volume")}
            >
              Sandbox-managed root
            </Button>
          </div>

          {mode === "host_local" && (
            <form className="mt-4 space-y-3" onSubmit={attachHostRoot}>
              <label className="block">
                <span className="text-xs font-medium text-text-muted">Root path</span>
                <input
                  className="mt-1 w-full rounded-md border border-border bg-bg px-3 py-2 text-sm text-text outline-none focus:border-primary focus:ring-2 focus:ring-primary/20"
                  aria-label="Root path"
                  value={hostPath}
                  onChange={(event) => setHostPath(event.target.value)}
                  placeholder="/path/to/project"
                />
              </label>
              <label className="block">
                <span className="text-xs font-medium text-text-muted">Display name</span>
                <input
                  className="mt-1 w-full rounded-md border border-border bg-bg px-3 py-2 text-sm text-text outline-none focus:border-primary focus:ring-2 focus:ring-primary/20"
                  aria-label="Display name"
                  value={hostDisplayName}
                  onChange={(event) => setHostDisplayName(event.target.value)}
                  placeholder="Project repo"
                />
              </label>
              <Button htmlType="submit" variant="primary" loading={submitting}>
                Attach host-local root
              </Button>
            </form>
          )}

          {mode === "sandbox_volume" && (
            <form className="mt-4 space-y-3" onSubmit={provisionSandboxRoot}>
              <label className="block">
                <span className="text-xs font-medium text-text-muted">Display name</span>
                <input
                  className="mt-1 w-full rounded-md border border-border bg-bg px-3 py-2 text-sm text-text outline-none focus:border-primary focus:ring-2 focus:ring-primary/20"
                  aria-label="Display name"
                  value={sandboxDisplayName}
                  onChange={(event) => setSandboxDisplayName(event.target.value)}
                  placeholder="Project sandbox"
                />
              </label>
              <label className="block">
                <span className="text-xs font-medium text-text-muted">Requested runtime</span>
                <input
                  className="mt-1 w-full rounded-md border border-border bg-bg px-3 py-2 text-sm text-text outline-none focus:border-primary focus:ring-2 focus:ring-primary/20"
                  aria-label="Requested runtime"
                  value={requestedRuntime}
                  onChange={(event) => setRequestedRuntime(event.target.value)}
                  placeholder="python"
                />
              </label>
              <Button
                htmlType="submit"
                variant="primary"
                loading={submitting}
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
            disabled={!inventoryAvailable}
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
