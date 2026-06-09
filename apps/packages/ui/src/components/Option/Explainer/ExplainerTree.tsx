import { ChevronRight, Plus } from "lucide-react"
import type { ExplainerSession } from "./types"
import {
  flattenExplainerTree,
  getExplainerEvidenceLabel,
  getExplainerNodeStatusLabel
} from "./tree"

type ExplainerTreeProps = {
  session: ExplainerSession | null
  selectedNodeId?: string | null
  onSelectNode: (nodeId: string) => void
  onExpandNode: (nodeId: string) => void
}

const evidenceClass = (state: string): string => {
  if (state === "supported") return "bg-success/10 text-success"
  if (state === "partially_supported") return "bg-warn/10 text-warn"
  if (state === "insufficient") return "bg-danger/10 text-danger"
  return "bg-surface2 text-text-muted"
}

export const ExplainerTree = ({
  session,
  selectedNodeId,
  onSelectNode,
  onExpandNode
}: ExplainerTreeProps) => {
  const rows = session
    ? flattenExplainerTree(session.nodes, session.rootNodeIds)
    : []

  return (
    <aside
      aria-label="Explainer tree rail"
      className="flex min-h-0 flex-col border-r border-border bg-surface"
    >
      <div className="border-b border-border px-4 py-3">
        <h2 className="text-sm font-semibold text-text">Outline</h2>
        <p className="text-xs text-text-muted">{rows.length} nodes</p>
      </div>
      <div
        role="tree"
        aria-label="Explainer outline"
        className="min-h-0 flex-1 overflow-auto p-2"
      >
        {rows.length === 0 ? (
          <p className="rounded-md bg-surface2 px-3 py-4 text-sm text-text-muted">
            Start a goal or select sources to create the first node.
          </p>
        ) : (
          rows.map(({ node, depth }) => {
            const selected = node.id === selectedNodeId
            return (
              <div
                key={node.id}
                role="treeitem"
                aria-selected={selected}
                aria-level={depth + 1}
                className={[
                  "group mb-1 rounded-md border px-2 py-2 transition-colors",
                  selected
                    ? "border-primary bg-primary/10"
                    : "border-transparent hover:border-border hover:bg-surface2"
                ].join(" ")}
                style={{ paddingLeft: `${8 + depth * 18}px` }}
              >
                <button
                  type="button"
                  className="flex w-full items-start gap-2 text-left"
                  onClick={() => onSelectNode(node.id)}
                >
                  <ChevronRight
                    className="mt-0.5 h-4 w-4 shrink-0 text-text-muted"
                    aria-hidden="true"
                  />
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-sm font-medium text-text">
                      {node.title}
                    </span>
                    <span className="mt-1 flex flex-wrap gap-1">
                      <span className="rounded-full bg-surface px-2 py-0.5 text-[11px] font-medium text-text-muted">
                        {getExplainerNodeStatusLabel(node.status)}
                      </span>
                      <span
                        className={`rounded-full px-2 py-0.5 text-[11px] font-medium ${evidenceClass(node.evidenceState)}`}
                      >
                        {getExplainerEvidenceLabel(node.evidenceState)}
                      </span>
                    </span>
                  </span>
                </button>
                <button
                  type="button"
                  className="mt-2 inline-flex h-7 items-center gap-1 rounded-md border border-border bg-surface px-2 text-xs font-medium text-text-muted transition-colors hover:bg-surface2 hover:text-text"
                  onClick={() => onExpandNode(node.id)}
                >
                  <Plus className="h-3.5 w-3.5" aria-hidden="true" />
                  Expand
                </button>
              </div>
            )
          })
        )}
      </div>
    </aside>
  )
}
