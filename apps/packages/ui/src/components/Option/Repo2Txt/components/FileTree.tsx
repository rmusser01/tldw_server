import type { RepoTreeNode } from "../providers/types"

type FileTreeProps = {
  disabled?: boolean
  nodes: RepoTreeNode[]
  selectedPaths: Set<string>
  onTogglePath: (path: string) => void
}

export function FileTree({
  nodes,
  selectedPaths,
  onTogglePath,
  disabled
}: FileTreeProps) {
  if (nodes.length === 0) {
    return <p className="text-sm text-text-subtle">No files loaded.</p>
  }

  return (
    <section className="space-y-2">
      <h3 className="text-sm font-semibold">File Tree</h3>
      <ul className="max-h-64 space-y-1 overflow-auto rounded border p-2">
        {nodes
          .filter((node) => node.type === "blob")
          .map((node) => (
            <li key={node.path} className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                disabled={disabled}
                checked={selectedPaths.has(node.path)}
                onChange={() => onTogglePath(node.path)}
                aria-label={`Select ${node.path}`}
              />
              <span className="truncate">{node.path}</span>
            </li>
          ))}
      </ul>
    </section>
  )
}
