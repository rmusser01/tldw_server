import { useCallback, useMemo } from "react"
import { useQuery, useQueryClient } from "@tanstack/react-query"
import { Button, Tree, Typography, Empty, Spin } from "antd"
import type { DataNode } from "antd/es/tree"
import type { TreeProps } from "antd"
import { BookOpen, FileText, FolderOpen, Layers, Plus } from "lucide-react"
import { useWritingPlaygroundStore } from "@/store/writing-playground"
import { getManuscriptStructure, listManuscriptProjects, createManuscriptProject, reorderManuscriptItems } from "@/services/writing-playground"

type ManuscriptTreePanelProps = {
  isOnline: boolean
}

// Extended DataNode that carries the entity's version for reorder
type VersionedDataNode = DataNode & { version?: number; entityType?: "part" | "chapter" | "scene" }

export function ManuscriptTreePanel({ isOnline }: ManuscriptTreePanelProps) {
  const activeProjectId = useWritingPlaygroundStore((s) => s.activeProjectId)
  const setActiveProjectId = useWritingPlaygroundStore((s) => s.setActiveProjectId)
  const activeNodeId = useWritingPlaygroundStore((s) => s.activeNodeId)
  const setActiveNodeId = useWritingPlaygroundStore((s) => s.setActiveNodeId)
  const setActiveNodeType = useWritingPlaygroundStore((s) => s.setActiveNodeType)
  const queryClient = useQueryClient()

  // Fetch project list
  const { data: projectsData } = useQuery({
    queryKey: ["manuscript-projects"],
    queryFn: () => listManuscriptProjects({ limit: 50 }),
    enabled: isOnline,
    staleTime: 30_000,
  })

  // Fetch structure for active project
  const { data: structure, isLoading: structureLoading } = useQuery({
    queryKey: ["manuscript-structure", activeProjectId],
    queryFn: () => getManuscriptStructure(activeProjectId!),
    enabled: isOnline && !!activeProjectId,
    staleTime: 30_000,
  })

  const treeData = useMemo(() => buildTreeData(structure), [structure])

  // Build a flat lookup of node key -> { version, entityType } from the tree
  const nodeVersionMap = useMemo(() => {
    const map = new Map<string, { version: number; entityType: "part" | "chapter" | "scene" }>()
    function walk(nodes: VersionedDataNode[]) {
      for (const n of nodes) {
        if (n.version != null && n.entityType) {
          map.set(String(n.key), { version: n.version, entityType: n.entityType })
        }
        if (n.children) walk(n.children as VersionedDataNode[])
      }
    }
    walk(treeData)
    return map
  }, [treeData])

  const handleDrop: TreeProps["onDrop"] = useCallback(async (info) => {
    if (!activeProjectId) return

    const dragKey = String(info.dragNode.key)
    const dragMeta = nodeVersionMap.get(dragKey)
    if (!dragMeta) return

    // Collect siblings at the drop position to compute new sort orders
    // For simplicity, we only reorder the dragged node with its new sort_order
    const dropKey = String(info.node.key)
    const dropMeta = nodeVersionMap.get(dropKey)
    if (!dropMeta) return

    // Only allow reorder within the same entity type
    if (dragMeta.entityType !== dropMeta.entityType) return

    const entityTypeMap = { part: "parts", chapter: "chapters", scene: "scenes" } as const
    const entityType = entityTypeMap[dragMeta.entityType]

    // Compute a sort_order that places the dragged node near the drop target
    const dropPos = info.dropPosition // -1 = before, 0 = inside, 1 = after
    const dropNodeData = info.node as VersionedDataNode
    const baseSortOrder = (dropNodeData as any).sort_order ?? info.dropPosition
    const newSortOrder = dropPos <= 0 ? baseSortOrder - 0.5 : baseSortOrder + 0.5

    try {
      await reorderManuscriptItems(activeProjectId, entityType, [
        { id: dragKey, sort_order: newSortOrder, version: dragMeta.version },
      ])
      queryClient.invalidateQueries({ queryKey: ["manuscript-structure", activeProjectId] })
    } catch (err) {
      console.error("Reorder failed:", err)
    }
  }, [activeProjectId, nodeVersionMap, queryClient])

  if (!activeProjectId) {
    const projects = (projectsData as any)?.projects || []
    return (
      <div className="flex flex-col gap-3 p-3">
        <Typography.Text strong className="text-sm">
          Manuscripts
        </Typography.Text>
        {projects.length === 0 ? (
          <Empty
            image={<BookOpen className="mx-auto h-10 w-10 text-gray-300" />}
            description="No manuscripts yet"
          >
            <Button
              type="primary"
              size="small"
              icon={<Plus className="h-3 w-3" />}
              onClick={async () => {
                try {
                  const result = await createManuscriptProject({ title: "Untitled Project" }) as any
                  if (result?.id) {
                    setActiveNodeId(null)
                    setActiveProjectId(result.id)
                  }
                  queryClient.invalidateQueries({ queryKey: ["manuscript-projects"] })
                } catch (err) {
                  console.error("Failed to create project:", err)
                }
              }}
            >
              New Project
            </Button>
          </Empty>
        ) : (
          <div className="flex flex-col gap-1">
            {projects.map((p: any) => (
              <div
                key={p.id}
                className="cursor-pointer rounded-md px-2 py-2 hover:bg-gray-100 dark:hover:bg-gray-800"
                onClick={() => {
                  setActiveNodeId(null)
                  setActiveProjectId(p.id)
                }}
              >
                <Typography.Text className="text-sm">{p.title}</Typography.Text>
                <Typography.Text type="secondary" className="ml-2 text-xs">
                  {p.word_count} words
                </Typography.Text>
              </div>
            ))}
          </div>
        )}
      </div>
    )
  }

  if (structureLoading) {
    return (
      <div className="flex items-center justify-center p-8">
        <Spin size="small" />
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-2 p-2">
      <div className="flex items-center justify-between px-1">
        <Typography.Text
          type="secondary"
          className="cursor-pointer text-xs hover:underline"
          onClick={() => {
            setActiveNodeId(null)
            setActiveProjectId(null)
          }}
        >
          &larr; All Projects
        </Typography.Text>
      </div>
      {treeData.length > 0 ? (
        <Tree
          treeData={treeData}
          selectedKeys={activeNodeId ? [activeNodeId] : []}
          onSelect={(keys) => {
            const key = (keys[0] as string) || null
            setActiveNodeId(key)
            setActiveNodeType(key ? nodeVersionMap.get(key)?.entityType ?? null : null)
          }}
          draggable
          onDrop={handleDrop}
          showIcon
          blockNode
          defaultExpandAll
          className="manuscript-tree"
        />
      ) : (
        <Empty description="Empty project" />
      )}
    </div>
  )
}

function buildTreeData(structure: any): VersionedDataNode[] {
  if (!structure) return []

  const sceneNode = (s: any): VersionedDataNode => ({
    key: s.id,
    title: `${s.title} (${s.word_count}w)`,
    icon: <FileText className="h-3.5 w-3.5" />,
    isLeaf: true,
    version: s.version,
    entityType: "scene",
  })

  const chapterNode = (ch: any): VersionedDataNode => ({
    key: ch.id,
    title: `${ch.title} (${ch.word_count}w)`,
    icon: <FolderOpen className="h-3.5 w-3.5" />,
    children: ch.scenes?.map(sceneNode) || [],
    version: ch.version,
    entityType: "chapter",
  })

  const partNodes: VersionedDataNode[] = (structure.parts || []).map((p: any) => ({
    key: p.id,
    title: `${p.title} (${p.word_count}w)`,
    icon: <Layers className="h-3.5 w-3.5" />,
    children: p.chapters?.map(chapterNode) || [],
    version: p.version,
    entityType: "part",
  }))

  const unassigned: VersionedDataNode[] = (structure.unassigned_chapters || []).map(chapterNode)

  return [...partNodes, ...unassigned]
}

export default ManuscriptTreePanel
