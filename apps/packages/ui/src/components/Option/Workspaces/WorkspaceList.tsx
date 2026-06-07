import React from "react"
import { Archive, ArchiveRestore, ExternalLink, Pencil } from "lucide-react"
import { Button } from "@/components/Common/Button"
import { Badge, type BadgeVariant } from "@/components/ui/primitives"
import type { WorkspaceManagerItem } from "./workspace-manager-models"

type WorkspaceListProps = {
  items: WorkspaceManagerItem[]
  selectedId?: string | null
  onSelect?: (item: WorkspaceManagerItem) => void
  onOpen: (item: WorkspaceManagerItem) => void
  onEdit: (item: WorkspaceManagerItem) => void
  onArchive: (item: WorkspaceManagerItem) => void
  onUnarchive: (item: WorkspaceManagerItem) => void
}

const profileLabel = (item: WorkspaceManagerItem): string =>
  item.profile === "project"
    ? "Project"
    : item.profile === "research"
      ? "Research"
      : "Unknown"

const attentionLabel = (state: WorkspaceManagerItem["attentionState"]): string =>
  state.replace(/_/g, " ")

const attentionVariant = (
  state: WorkspaceManagerItem["attentionState"]
): BadgeVariant => {
  if (state === "ready") return "success"
  if (state === "working") return "info"
  if (state === "setup_pending") return "warning"
  if (state === "needs_attention" || state === "blocked") return "danger"
  return "secondary"
}

const rootSummary = (item: WorkspaceManagerItem): string => {
  if (item.profile !== "project") return "Research sources"
  if (item.projectRoot.displayName) return item.projectRoot.displayName
  if (item.projectRoot.backend === "sandbox_volume") return "Sandbox-managed root"
  if (item.projectRoot.backend === "host_local") return "Host-local root"
  return "Root not configured"
}

const inventorySummary = (item: WorkspaceManagerItem): string => {
  const inventory = item.projectRoot.fileInventory
  if (inventory.totalFileCount != null || inventory.indexedFileCount != null) {
    return `Files ${inventory.indexedFileCount ?? 0}/${inventory.totalFileCount ?? 0}`
  }
  if (item.profile === "project") return "Inventory not available"
  return "Source inventory"
}

const sourceSummary = (item: WorkspaceManagerItem): string =>
  item.sourceCount == null ? "Sources unknown" : `Sources ${item.sourceCount}`

const formatUpdatedAt = (value: string): string => {
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return value
  return parsed.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit"
  })
}

export const WorkspaceList = ({
  items,
  selectedId,
  onSelect,
  onOpen,
  onEdit,
  onArchive,
  onUnarchive
}: WorkspaceListProps) => {
  return (
    <div className="min-h-0 overflow-auto border-t border-border">
      <table className="w-full min-w-[860px] border-collapse text-sm">
        <caption className="sr-only">Workspaces list</caption>
        <thead className="sticky top-0 z-10 bg-bg text-xs uppercase text-text-muted">
          <tr className="border-b border-border">
            <th className="px-3 py-2 text-left font-medium">Name</th>
            <th className="px-3 py-2 text-left font-medium">Profile</th>
            <th className="px-3 py-2 text-left font-medium">State</th>
            <th className="px-3 py-2 text-left font-medium">Root</th>
            <th className="px-3 py-2 text-left font-medium">Content</th>
            <th className="px-3 py-2 text-left font-medium">Updated</th>
            <th className="px-3 py-2 text-right font-medium">Actions</th>
          </tr>
        </thead>
        <tbody>
          {items.length === 0 && (
            <tr>
              <td className="px-3 py-3 text-center text-text-muted" colSpan={7}>
                No workspaces found
              </td>
            </tr>
          )}
          {items.map((item) => (
            <tr
              key={item.id}
              className={[
                "border-b border-border/70 hover:bg-surface2/50",
                selectedId === item.id ? "bg-primary/5" : null
              ]
                .filter(Boolean)
                .join(" ")}
            >
              <td className="px-3 py-3 align-top">
                <div className="font-medium text-text">{item.name}</div>
                <div className="mt-1 text-xs text-text-muted">{item.id}</div>
              </td>
              <td className="px-3 py-3 align-top">
                <Badge variant={item.profile === "project" ? "primary" : "info"}>
                  {profileLabel(item)}
                </Badge>
                {item.archived && (
                  <Badge className="ml-2" variant="secondary">
                    Archived
                  </Badge>
                )}
              </td>
              <td className="px-3 py-3 align-top">
                <Badge variant={attentionVariant(item.attentionState)} dot>
                  {attentionLabel(item.attentionState)}
                </Badge>
              </td>
              <td className="px-3 py-3 align-top text-text-muted">
                <div>{rootSummary(item)}</div>
                <div className="mt-1 text-xs">{item.projectRoot.state}</div>
              </td>
              <td className="px-3 py-3 align-top text-text-muted">
                <div>{sourceSummary(item)}</div>
                <div className="mt-1 text-xs">{inventorySummary(item)}</div>
              </td>
              <td className="px-3 py-3 align-top text-text-muted">
                {formatUpdatedAt(item.updatedAt)}
              </td>
              <td className="px-3 py-3 align-top">
                <div className="flex justify-end gap-2">
                  {onSelect && (
                    <Button
                      size="sm"
                      variant={selectedId === item.id ? "primary" : "ghost"}
                      ariaLabel={`Manage ${item.name}`}
                      onClick={() => onSelect(item)}
                    >
                      Manage
                    </Button>
                  )}
                  <Button
                    size="sm"
                    variant="ghost"
                    icon={<ExternalLink className="h-4 w-4" />}
                    ariaLabel={`Open ${item.name}`}
                    onClick={() => onOpen(item)}
                  >
                    Open
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    icon={<Pencil className="h-4 w-4" />}
                    ariaLabel={`Edit ${item.name}`}
                    onClick={() => onEdit(item)}
                  >
                    Edit
                  </Button>
                  {item.archived ? (
                    <Button
                      size="sm"
                      variant="outline"
                      icon={<ArchiveRestore className="h-4 w-4" />}
                      ariaLabel={`Unarchive ${item.name}`}
                      onClick={() => onUnarchive(item)}
                    >
                      Unarchive
                    </Button>
                  ) : (
                    <Button
                      size="sm"
                      variant="outline"
                      icon={<Archive className="h-4 w-4" />}
                      ariaLabel={`Archive ${item.name}`}
                      onClick={() => onArchive(item)}
                    >
                      Archive
                    </Button>
                  )}
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
