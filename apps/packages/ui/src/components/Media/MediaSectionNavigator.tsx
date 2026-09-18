import React from "react"
import { ChevronDown, ChevronRight, Search } from "lucide-react"
import type { MediaNavigationNode } from "@/hooks/useMediaNavigation"
import {
  buildMediaNavigationTreeIndex,
  findQuickJumpMatches
} from "@/utils/media-navigation-tree"

type Props = {
  nodes: MediaNavigationNode[]
  selectedNodeId: string | null
  loading?: boolean
  error?: unknown
  onRetry?: () => void
  onSelectNode: (node: MediaNavigationNode) => void
  className?: string
}

const errorToText = (err: unknown) => {
  if (!err) return ""
  if (typeof err === "string") return err
  if (err instanceof Error) return err.message
  return "Failed to load sections."
}

const childLimitForDepth = (depth: number): number => {
  // Keep top-level sections fully visible; trim deeper branches by default.
  if (depth <= 0) return Number.POSITIVE_INFINITY
  if (depth === 1) return 8
  return 6
}

export const MediaSectionNavigator: React.FC<Props> = ({
  nodes,
  selectedNodeId,
  loading = false,
  error,
  onRetry,
  onSelectNode,
  className = ""
}) => {
  const [quickJump, setQuickJump] = React.useState("")
  const [expandedIds, setExpandedIds] = React.useState<Set<string>>(new Set())
  const [expandedChildGroups, setExpandedChildGroups] = React.useState<Set<string>>(
    new Set()
  )

  const tree = React.useMemo(() => buildMediaNavigationTreeIndex(nodes), [nodes])

  React.useEffect(() => {
    setExpandedIds((prev) => {
      const next = new Set(prev)
      for (const root of tree.roots) next.add(root.id)

      if (selectedNodeId && tree.byId[selectedNodeId]) {
        let cursor = tree.byId[selectedNodeId]
        while (cursor?.parent_id) {
          next.add(cursor.parent_id)
          cursor = tree.byId[cursor.parent_id]
        }
      }
      return next
    })
    setExpandedChildGroups((prev) => {
      const next = new Set<string>()
      for (const id of prev) {
        if (tree.byId[id]) next.add(id)
      }

      if (selectedNodeId && tree.byId[selectedNodeId]) {
        let cursor = tree.byId[selectedNodeId]
        while (cursor?.parent_id) {
          next.add(cursor.parent_id)
          cursor = tree.byId[cursor.parent_id]
        }
      }
      return next
    })
  }, [selectedNodeId, tree])

  const quickJumpMatches = React.useMemo(
    () => findQuickJumpMatches(nodes, quickJump),
    [nodes, quickJump]
  )

  const breadcrumbs = React.useMemo(() => {
    if (!selectedNodeId || !tree.byId[selectedNodeId]) return []
    const chain: MediaNavigationNode[] = []
    let cursor: MediaNavigationNode | undefined = tree.byId[selectedNodeId]
    while (cursor) {
      chain.unshift(cursor)
      if (!cursor.parent_id) break
      cursor = tree.byId[cursor.parent_id]
    }
    return chain
  }, [selectedNodeId, tree])

  const toggleExpanded = React.useCallback((nodeId: string) => {
    setExpandedIds((prev) => {
      const next = new Set(prev)
      if (next.has(nodeId)) next.delete(nodeId)
      else next.add(nodeId)
      return next
    })
  }, [])

  const toggleChildGroupExpanded = React.useCallback((nodeId: string) => {
    setExpandedChildGroups((prev) => {
      const next = new Set(prev)
      if (next.has(nodeId)) next.delete(nodeId)
      else next.add(nodeId)
      return next
    })
  }, [])

  const handleSelect = React.useCallback(
    (node: MediaNavigationNode) => {
      onSelectNode(node)
    },
    [onSelectNode]
  )

  const handleQuickJumpEnter = React.useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key !== "Enter") return
      if (quickJumpMatches.length === 0) return
      handleSelect(quickJumpMatches[0])
    },
    [quickJumpMatches, handleSelect]
  )

  const renderNode = React.useCallback(
    (node: MediaNavigationNode, depth: number): React.ReactNode => {
      const children = tree.childrenByParent[node.id] || []
      const hasChildren = children.length > 0
      const isExpanded = expandedIds.has(node.id)
      const isSelected = selectedNodeId === node.id
      const childLimit = childLimitForDepth(depth)
      const hasChildLimit = Number.isFinite(childLimit)
      const isChildGroupExpanded = expandedChildGroups.has(node.id)
      const visibleChildren =
        hasChildren && hasChildLimit && !isChildGroupExpanded
          ? children.slice(0, childLimit)
          : children
      const hiddenChildrenCount = hasChildren
        ? Math.max(0, children.length - visibleChildren.length)
        : 0

      return (
        <div key={node.id}>
          <div
            className={`flex items-center gap-1 rounded-md px-2 py-1 text-sm ${
              isSelected
                ? "bg-primary/10 text-primary"
                : "text-text hover:bg-surface2"
            }`}
            style={{ paddingLeft: `${Math.max(0, depth) * 12 + 8}px` }}
          >
            {hasChildren ? (
              <button
                type="button"
                className="shrink-0 rounded p-0.5 hover:bg-surface"
                onClick={() => toggleExpanded(node.id)}
                aria-label={isExpanded ? "Collapse section" : "Expand section"}
              >
                {isExpanded ? (
                  <ChevronDown className="h-3.5 w-3.5" />
                ) : (
                  <ChevronRight className="h-3.5 w-3.5" />
                )}
              </button>
            ) : (
              <span className="inline-block w-4" aria-hidden="true" />
            )}
            <button
              type="button"
              onClick={() => handleSelect(node)}
              className="flex min-w-0 flex-1 items-center gap-2 text-left"
            >
              {node.path_label ? (
                <span className="shrink-0 rounded bg-surface2 px-1.5 py-0.5 text-[10px] text-text-muted">
                  {node.path_label}
                </span>
              ) : null}
              <span className="truncate">{node.title}</span>
            </button>
          </div>
          {hasChildren && isExpanded ? (
            <div>
              {visibleChildren.map((child) => renderNode(child, depth + 1))}
              {hasChildLimit && children.length > childLimit ? (
                <button
                  type="button"
                  onClick={() => toggleChildGroupExpanded(node.id)}
                  className="ml-6 mt-1 rounded px-2 py-1 text-[11px] text-text-muted hover:bg-surface2 hover:text-text"
                >
                  {isChildGroupExpanded
                    ? "Show fewer"
                    : `Show ${hiddenChildrenCount} more`}
                </button>
              ) : null}
            </div>
          ) : null}
        </div>
      )
    },
    [
      expandedChildGroups,
      expandedIds,
      handleSelect,
      selectedNodeId,
      toggleChildGroupExpanded,
      toggleExpanded,
      tree.childrenByParent
    ]
  )

  return (
    <aside
      className={`w-full min-w-0 max-w-full md:w-72 md:min-w-72 border-b md:border-b-0 md:border-r border-border bg-surface flex flex-col ${className}`}
      aria-label="Chapters and sections"
    >
      <div className="px-3 py-2 border-b border-border">
        <div className="text-sm font-semibold text-text">Chapters/Sections</div>
        <div className="mt-2 relative">
          <Search className="absolute left-2 top-2.5 h-3.5 w-3.5 text-text-muted" />
          <input
            type="text"
            value={quickJump}
            onChange={(e) => setQuickJump(e.target.value)}
            onKeyDown={handleQuickJumpEnter}
            placeholder="Jump to 12.5 or title"
            className="w-full rounded-md border border-border bg-surface2 pl-7 pr-2 py-1.5 text-xs text-text placeholder:text-text-muted focus:outline-none focus:ring-2 focus:ring-primary/20"
          />
        </div>
        {quickJump.trim() && quickJumpMatches.length > 0 ? (
          <div className="mt-2 space-y-1">
            {quickJumpMatches.slice(0, 5).map((node) => (
              <button
                key={`match-${node.id}`}
                type="button"
                onClick={() => handleSelect(node)}
                className="w-full text-left rounded px-2 py-1 text-xs text-text hover:bg-surface2"
              >
                {node.path_label ? `${node.path_label} ` : ""}
                {node.title}
              </button>
            ))}
          </div>
        ) : null}
        {breadcrumbs.length > 0 ? (
          <div className="mt-2 text-[11px] text-text-muted truncate">
            {breadcrumbs.map((n) => n.title).join(" > ")}
          </div>
        ) : null}
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto p-2" role="tree">
        {loading ? (
          <div className="px-2 py-3 text-sm text-text-muted">Loading sections...</div>
        ) : null}
        {!loading && error ? (
          <div className="px-2 py-3 text-sm text-danger">
            <div>{errorToText(error)}</div>
            {onRetry ? (
              <button
                type="button"
                onClick={onRetry}
                className="mt-2 rounded border border-border px-2 py-1 text-xs text-text hover:bg-surface2"
              >
                Retry
              </button>
            ) : null}
          </div>
        ) : null}
        {!loading && !error && nodes.length === 0 ? (
          <div className="px-2 py-3 text-sm text-text-muted">
            No section structure available for this item.
          </div>
        ) : null}
        {!loading && !error && nodes.length > 0 ? (
          <div>{tree.roots.map((node) => renderNode(node, 0))}</div>
        ) : null}
      </div>
    </aside>
  )
}

export default MediaSectionNavigator
