import { useRef } from "react"
import { Sparkles } from "lucide-react"
import type { ExplainerSession } from "./types"
import {
  flattenExplainerTree,
  getExplainerEvidenceChipClass,
  getExplainerEvidenceDescription,
  getExplainerEvidenceLabel,
  getExplainerNodeStatusLabel
} from "./tree"

type ExplainerTreeProps = {
  session: ExplainerSession | null
  selectedNodeId?: string | null
  generatingNodeId?: string | null
  onSelectNode: (nodeId: string) => void
  onExpandNode: (nodeId: string) => void
}

export const ExplainerTree = ({
  session,
  selectedNodeId,
  generatingNodeId = null,
  onSelectNode,
  onExpandNode
}: ExplainerTreeProps) => {
  const rows = session
    ? flattenExplainerTree(session.nodes, session.rootNodeIds)
    : []
  const rowRefs = useRef<Array<HTMLButtonElement | null>>([])

  const focusRow = (index: number) => {
    const clamped = Math.max(0, Math.min(rows.length - 1, index))
    rowRefs.current[clamped]?.focus()
  }

  const handleRowKeyDown = (event: React.KeyboardEvent<HTMLButtonElement>, index: number) => {
    switch (event.key) {
      case "ArrowDown":
        event.preventDefault()
        focusRow(index + 1)
        break
      case "ArrowUp":
        event.preventDefault()
        focusRow(index - 1)
        break
      case "Home":
        event.preventDefault()
        focusRow(0)
        break
      case "End":
        event.preventDefault()
        focusRow(rows.length - 1)
        break
      default:
        break
    }
  }

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
            Set a goal or add sources to create the first topic.
          </p>
        ) : (
          rows.map(({ node, depth }, index) => {
            const selected = node.id === selectedNodeId
            const generating = node.id === generatingNodeId
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
                  ref={(element) => {
                    rowRefs.current[index] = element
                  }}
                  tabIndex={selected || (!selectedNodeId && index === 0) ? 0 : -1}
                  className="flex w-full items-start gap-2 text-left"
                  onClick={() => onSelectNode(node.id)}
                  onKeyDown={(event) => handleRowKeyDown(event, index)}
                >
                  <span className="min-w-0 flex-1">
                    <span
                      className="block truncate text-sm font-medium text-text"
                      title={node.title}
                    >
                      {node.title}
                    </span>
                    <span className="mt-1 flex flex-wrap gap-1">
                      {generating ? (
                        <span className="rounded-full bg-primary/10 px-2 py-0.5 text-[11px] font-medium text-primary">
                          Generating
                        </span>
                      ) : node.status !== "complete" ? (
                        <span className="rounded-full bg-surface px-2 py-0.5 text-[11px] font-medium text-text-muted">
                          {getExplainerNodeStatusLabel(node.status)}
                        </span>
                      ) : null}
                      <span
                        className={`rounded-full px-2 py-0.5 text-[11px] font-medium ${getExplainerEvidenceChipClass(node.evidenceState)}`}
                        title={getExplainerEvidenceDescription(node.evidenceState)}
                      >
                        {getExplainerEvidenceLabel(node.evidenceState)}
                      </span>
                      {node.childNodeIds.length > 0 ? (
                        <span className="rounded-full bg-surface px-2 py-0.5 text-[11px] font-medium text-text-muted">
                          {node.childNodeIds.length === 1
                            ? "1 subtopic"
                            : `${node.childNodeIds.length} subtopics`}
                        </span>
                      ) : null}
                    </span>
                  </span>
                </button>
                <button
                  type="button"
                  aria-label={`Break down ${node.title}`}
                  className="mt-2 inline-flex h-7 items-center gap-1 rounded-md border border-border bg-surface px-2 text-xs font-medium text-text-muted transition-colors hover:bg-surface2 hover:text-text disabled:cursor-not-allowed disabled:opacity-60"
                  disabled={generating}
                  onClick={() => onExpandNode(node.id)}
                >
                  <Sparkles className="h-3.5 w-3.5" aria-hidden="true" />
                  Break down
                </button>
              </div>
            )
          })
        )}
      </div>
    </aside>
  )
}
