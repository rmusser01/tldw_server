import React from "react"
import { Link2, RefreshCw, ServerCog } from "lucide-react"
import { Button } from "@/components/Common/Button"
import { Alert, Badge } from "@/components/ui/primitives"
import { useTldwApiClient } from "@/hooks/useTldwApiClient"
import type { WorkspaceApiResponse } from "@/services/tldw/domains/workspace-api"
import {
  buildWorkspaceReconciliationDryRun,
  discoverLocalResearchWorkspaceEntriesFromBrowser,
  writeWorkspaceReconciliationMarker,
  type WorkspaceReconciliationDryRunItem,
  type WorkspaceReconciliationServerWorkspace
} from "./workspace-local-reconciliation"
import type { WorkspaceManagerItem } from "./workspace-manager-models"

type WorkspaceReconciliationPanelProps = {
  serverWorkspaces: WorkspaceManagerItem[]
  onServerWorkspaceCreated?: (workspace: WorkspaceApiResponse) => void
}

const stateLabel = (state: WorkspaceReconciliationDryRunItem["state"]): string =>
  state.replace(/_/g, " ")

const stateVariant = (
  state: WorkspaceReconciliationDryRunItem["state"]
): React.ComponentProps<typeof Badge>["variant"] => {
  if (state === "ready_to_create_metadata") return "success"
  if (state === "local_only") return "info"
  if (state === "unsupported_local_payload") return "danger"
  return "warning"
}

const mapServerWorkspace = (
  item: WorkspaceManagerItem
): WorkspaceReconciliationServerWorkspace => ({
  id: item.id,
  name: item.name,
  workspace_profile: item.profile === "project" ? "project" : "research"
})

const markerTimestamp = (): string => new Date().toISOString()

export const WorkspaceReconciliationPanel = ({
  serverWorkspaces,
  onServerWorkspaceCreated
}: WorkspaceReconciliationPanelProps) => {
  const api = useTldwApiClient()
  const [items, setItems] = React.useState<WorkspaceReconciliationDryRunItem[]>([])
  const [error, setError] = React.useState<string | null>(null)
  const [busyWorkspaceId, setBusyWorkspaceId] = React.useState<string | null>(null)

  const refreshLocalEntries = React.useCallback(() => {
    const localEntries = discoverLocalResearchWorkspaceEntriesFromBrowser()
    const dryRun = buildWorkspaceReconciliationDryRun({
      localEntries,
      serverWorkspaces: serverWorkspaces.map(mapServerWorkspace)
    })
    setItems(dryRun.items.filter((item) => !item.marker))
  }, [serverWorkspaces])

  React.useEffect(() => {
    refreshLocalEntries()
  }, [refreshLocalEntries])

  const createServerMetadata = async (
    item: WorkspaceReconciliationDryRunItem
  ): Promise<void> => {
    if (item.state !== "ready_to_create_metadata") return
    setBusyWorkspaceId(item.localWorkspaceId)
    setError(null)
    try {
      const workspace = await api.upsertWorkspace(item.localWorkspaceId, {
        name: item.name,
        study_materials_policy: "workspace",
        workspace_profile: "research"
      })
      writeWorkspaceReconciliationMarker({
        storage: window.localStorage,
        localWorkspaceId: item.localWorkspaceId,
        marker: {
          schemaVersion: 1,
          serverWorkspaceId: workspace.id,
          serverName: workspace.name || item.name,
          serverProfile: workspace.workspace_profile,
          linkedAt: markerTimestamp(),
          status: "metadata_promoted"
        }
      })
      onServerWorkspaceCreated?.(workspace)
      refreshLocalEntries()
    } catch (caught) {
      setError(
        caught instanceof Error
          ? caught.message
          : "Failed to create Workspace metadata."
      )
    } finally {
      setBusyWorkspaceId(null)
    }
  }

  const linkExistingWorkspace = (
    item: WorkspaceReconciliationDryRunItem
  ): void => {
    const target = serverWorkspaces.find(
      (workspace) => workspace.id === item.conflictServerWorkspaceId
    )
    if (!target) {
      setError("Choose an existing Workspace before linking.")
      return
    }
    writeWorkspaceReconciliationMarker({
      storage: window.localStorage,
      localWorkspaceId: item.localWorkspaceId,
      marker: {
        schemaVersion: 1,
        serverWorkspaceId: target.id,
        serverName: target.name,
        serverProfile: target.profile === "project" ? "project" : "research",
        linkedAt: markerTimestamp(),
        status: "linked"
      }
    })
    refreshLocalEntries()
  }

  if (items.length === 0) return null

  return (
    <section className="border-b border-border bg-bg px-4 py-3">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold text-text">
            Local Research Workspaces
          </h2>
          <p className="mt-1 text-xs text-text-muted">
            Local-only Research Workspace records are shown separately until you
            create server metadata or link them to an existing Workspace.
          </p>
        </div>
        <Button
          size="sm"
          variant="ghost"
          icon={<RefreshCw className="h-4 w-4" />}
          onClick={refreshLocalEntries}
        >
          Rescan local
        </Button>
      </div>

      <div className="mt-3 overflow-auto rounded-md border border-border">
        <table className="w-full min-w-[760px] border-collapse text-sm">
          <thead className="bg-surface text-xs uppercase text-text-muted">
            <tr className="border-b border-border">
              <th className="px-3 py-2 text-left font-medium">Local entry</th>
              <th className="px-3 py-2 text-left font-medium">State</th>
              <th className="px-3 py-2 text-left font-medium">Server match</th>
              <th className="px-3 py-2 text-right font-medium">Actions</th>
            </tr>
          </thead>
          <tbody>
            {items.map((item) => (
              <tr key={item.localWorkspaceId} className="border-b border-border/70">
                <td className="px-3 py-3 align-top">
                  <div className="font-medium text-text">{item.name}</div>
                  <div className="mt-1 text-xs text-text-muted">
                    {item.localWorkspaceId}
                    {item.sourceCount != null ? ` · Sources ${item.sourceCount}` : ""}
                  </div>
                </td>
                <td className="px-3 py-3 align-top">
                  <Badge variant={stateVariant(item.state)}>
                    {stateLabel(item.state)}
                  </Badge>
                  {item.reason && (
                    <div className="mt-1 text-xs text-text-muted">{item.reason}</div>
                  )}
                </td>
                <td className="px-3 py-3 align-top text-text-muted">
                  {item.conflictServerName || "No server match"}
                </td>
                <td className="px-3 py-3 align-top">
                  <div className="flex justify-end gap-2">
                    <Button
                      size="sm"
                      variant="primary"
                      icon={<ServerCog className="h-4 w-4" />}
                      disabled={item.state !== "ready_to_create_metadata"}
                      loading={busyWorkspaceId === item.localWorkspaceId}
                      onClick={() => void createServerMetadata(item)}
                    >
                      Create server metadata
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      icon={<Link2 className="h-4 w-4" />}
                      disabled={!item.conflictServerWorkspaceId}
                      onClick={() => linkExistingWorkspace(item)}
                    >
                      Link to existing Workspace
                    </Button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {error && <Alert className="mt-3" variant="error" title={error} />}
    </section>
  )
}
