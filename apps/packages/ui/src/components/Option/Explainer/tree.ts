import type {
  ExplainerEvidenceState,
  ExplainerNode,
  ExplainerNodeStatus,
  FlattenedExplainerNode
} from "./types"

const compareNodes = (a: ExplainerNode, b: ExplainerNode): number => {
  if (a.ordinal !== b.ordinal) return a.ordinal - b.ordinal
  if (a.createdAt !== b.createdAt) return a.createdAt.localeCompare(b.createdAt)
  return a.id.localeCompare(b.id)
}

export const flattenExplainerTree = (
  nodes: Record<string, ExplainerNode>,
  rootNodeIds: string[]
): FlattenedExplainerNode[] => {
  const flattened: FlattenedExplainerNode[] = []
  const seen = new Set<string>()

  const visit = (nodeId: string, depth: number) => {
    const node = nodes[nodeId]
    if (!node || seen.has(nodeId)) return
    seen.add(nodeId)
    flattened.push({ node, depth })
    const children = node.childNodeIds
      .map((childId) => nodes[childId])
      .filter((child): child is ExplainerNode => Boolean(child))
      .sort(compareNodes)
    for (const child of children) {
      visit(child.id, depth + 1)
    }
  }

  const roots = rootNodeIds
    .map((nodeId) => nodes[nodeId])
    .filter((node): node is ExplainerNode => Boolean(node))
    .sort(compareNodes)
  for (const root of roots) {
    visit(root.id, 0)
  }

  const orphans = Object.values(nodes)
    .filter((node) => !seen.has(node.id))
    .sort((a, b) => {
      const parentCompare = (a.parentId ?? "").localeCompare(b.parentId ?? "")
      return parentCompare || compareNodes(a, b)
    })
  for (const orphan of orphans) {
    visit(orphan.id, 0)
  }

  return flattened
}

export const getSelectedExplainerNode = (
  nodes: Record<string, ExplainerNode>,
  rootNodeIds: string[],
  selectedNodeId?: string | null
): ExplainerNode | null => {
  if (selectedNodeId && nodes[selectedNodeId]) {
    return nodes[selectedNodeId]
  }
  const firstRoot = rootNodeIds.find((nodeId) => nodes[nodeId])
  if (firstRoot) {
    return nodes[firstRoot] ?? null
  }
  return Object.values(nodes).sort(compareNodes)[0] ?? null
}

export const pruneExplainerNodeTree = (
  nodes: Record<string, ExplainerNode>,
  rootNodeIds: string[],
  deletedNodeId: string
): { nodes: Record<string, ExplainerNode>; rootNodeIds: string[] } => {
  const idsToDelete = new Set<string>()
  const collect = (nodeId: string) => {
    if (idsToDelete.has(nodeId)) return
    idsToDelete.add(nodeId)
    const node = nodes[nodeId]
    for (const childId of node?.childNodeIds ?? []) {
      collect(childId)
    }
  }
  collect(deletedNodeId)

  const nextNodes: Record<string, ExplainerNode> = {}
  for (const [nodeId, node] of Object.entries(nodes)) {
    if (idsToDelete.has(nodeId)) continue
    nextNodes[nodeId] = {
      ...node,
      childNodeIds: node.childNodeIds.filter((childId) => !idsToDelete.has(childId))
    }
  }

  return {
    nodes: nextNodes,
    rootNodeIds: rootNodeIds.filter((nodeId) => !idsToDelete.has(nodeId))
  }
}

export const getExplainerNodeStatusLabel = (
  status: ExplainerNodeStatus | string
): string => {
  const labels: Record<string, string> = {
    idle: "Idle",
    queued: "Queued",
    generating: "Generating",
    error: "Error",
    complete: "Complete"
  }
  return labels[status] ?? status
}

export const getExplainerEvidenceLabel = (
  evidenceState: ExplainerEvidenceState | string
): string => {
  const labels: Record<string, string> = {
    supported: "Supported",
    partially_supported: "Partially supported",
    uncited: "Uncited",
    insufficient: "Insufficient evidence"
  }
  return labels[evidenceState] ?? evidenceState
}
