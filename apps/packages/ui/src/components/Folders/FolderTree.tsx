/**
 * FolderTree Component
 *
 * Displays folders in a hierarchical tree structure using Ant Design Tree.
 * Shows folders from server's keyword_collections with proper nesting via parent_id.
 */

import { Tree, Empty, Spin } from "antd"
import type { TreeDataNode, TreeProps } from "antd"
import { Folder, FolderOpen, ChevronRight, ChevronDown } from "lucide-react"
import { buildFolderTree, useFolderStore, useFolderUIPrefs, type FolderTreeNode, type FolderOperation } from "@/store/folder"
import { useTranslation } from "react-i18next"
import { useMemo, useCallback } from "react"
import { useShallow } from "zustand/react/shallow"

interface FolderTreeProps {
  operation?: FolderOperation
  onFolderSelect?: (folderId: number) => void
  onConversationSelect?: (conversationId: string) => void
  selectedFolderId?: number | null
  conversations?: Array<{ id: string; title: string }>
  showConversations?: boolean
}

export const FolderTree = ({
  operation,
  onFolderSelect,
  onConversationSelect,
  selectedFolderId,
  conversations = [],
  showConversations = true
}: FolderTreeProps) => {
  const { t } = useTranslation()
  const uiPrefs = useFolderUIPrefs()
  const {
    folders,
    keywords,
    folderKeywordLinks,
    conversationKeywordLinks,
    isLoading,
    toggleFolderOpen
  } = useFolderStore(
    useShallow((s) => ({
      folders: s.folders,
      keywords: s.keywords,
      folderKeywordLinks: s.folderKeywordLinks,
      conversationKeywordLinks: s.conversationKeywordLinks,
      isLoading: s.isLoading,
      toggleFolderOpen: s.toggleFolderOpen
    }))
  )

  const canUse = useCallback(() => {
    if (!operation) return true
    const state = useFolderStore.getState()
    return operation.isCurrent() && state.serverDataScope === operation &&
      !state.isLoading && !state.error && state.folderApiAvailable === true
  }, [operation])

  const folderTree = useMemo(
    () => buildFolderTree(folders, keywords, folderKeywordLinks, conversationKeywordLinks),
    [folders, keywords, folderKeywordLinks, conversationKeywordLinks]
  )

  const conversationIdsByFolderId = useMemo(() => {
    const folderIdsByKeywordId = new Map<number, Set<number>>()
    for (const link of folderKeywordLinks) {
      const existing = folderIdsByKeywordId.get(link.keyword_id) ?? new Set<number>()
      existing.add(link.folder_id)
      folderIdsByKeywordId.set(link.keyword_id, existing)
    }

    const conversationIdsByFolder = new Map<number, Set<string>>()
    for (const link of conversationKeywordLinks) {
      const folderIds = folderIdsByKeywordId.get(link.keyword_id)
      if (!folderIds) continue

      for (const folderId of folderIds) {
        const existing = conversationIdsByFolder.get(folderId) ?? new Set<string>()
        existing.add(link.conversation_id)
        conversationIdsByFolder.set(folderId, existing)
      }
    }

    return conversationIdsByFolder
  }, [folderKeywordLinks, conversationKeywordLinks])

  // Convert folder tree to Ant Design tree data
  const treeData = useMemo((): TreeDataNode[] => {
    const convertNode = (node: FolderTreeNode): TreeDataNode => {
      const isOpen = uiPrefs[node.key]?.isOpen ?? true
      const color = uiPrefs[node.key]?.color

      // Get conversations for this folder
      const folderConversationIds = conversationIdsByFolderId.get(node.key) ?? new Set<string>()
      const folderConversations = showConversations
        ? conversations.filter((c) => folderConversationIds.has(c.id))
        : []

      const children: TreeDataNode[] = [
        // Child folders first
        ...node.children.map(convertNode),
        // Then conversations
        ...folderConversations.map(conv => ({
          key: `conv-${conv.id}`,
          title: conv.title || t("common:untitled"),
          isLeaf: true,
          icon: null,
          className: "folder-tree-conversation"
        }))
      ]

      return {
        key: node.key,
        title: (
          <span className="flex items-center gap-1.5">
            <span
              className="truncate max-w-[150px]"
              style={color ? { color } : undefined}
            >
              {node.title}
            </span>
            {node.itemCount > 0 && (
              <span className="text-xs text-text-subtle">
                ({node.itemCount})
              </span>
            )}
          </span>
        ),
        children: children.length > 0 ? children : undefined,
        isLeaf: children.length === 0,
        icon: ({ expanded }: { expanded?: boolean }) => {
          const IconComponent = expanded || isOpen ? FolderOpen : Folder
          return (
            <IconComponent
              className="w-4 h-4"
              style={color ? { color } : undefined}
            />
          )
        },
        className: `folder-tree-folder ${selectedFolderId === node.key ? 'folder-tree-selected' : ''}`
      }
    }

    return folderTree.map(convertNode)
  }, [folderTree, uiPrefs, conversationIdsByFolderId, conversations, showConversations, selectedFolderId, t])

  // Expanded keys from UI prefs
  const expandedKeys = useMemo(() => {
    return Object.entries(uiPrefs)
      .filter(([, prefs]) => prefs.isOpen)
      .map(([id]) => Number(id))
  }, [uiPrefs])

  // Handle expand/collapse
  const handleExpand: TreeProps['onExpand'] = useCallback((_keys, { node }) => {
    if (!canUse()) return
    const folderId = typeof node.key === 'number' ? node.key : Number(node.key)
    if (!isNaN(folderId) && !String(node.key).startsWith('conv-')) {
      toggleFolderOpen(folderId)
    }
  }, [toggleFolderOpen, canUse])

  // Handle selection
  const handleSelect: TreeProps['onSelect'] = useCallback((keys, { node }) => {
    if (!canUse()) return
    const key = String(node.key)
    if (key.startsWith('conv-')) {
      const conversationId = key.replace('conv-', '')
      onConversationSelect?.(conversationId)
    } else {
      const folderId = Number(key)
      if (!isNaN(folderId)) {
        onFolderSelect?.(folderId)
      }
    }
  }, [onFolderSelect, onConversationSelect, canUse])

  if (operation && !canUse()) return null

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-8">
        <Spin size="small" />
      </div>
    )
  }

  if (treeData.length === 0) {
    return (
      <Empty
        image={Empty.PRESENTED_IMAGE_SIMPLE}
        description={t("common:noFolders")}
        className="py-4"
      />
    )
  }

  return (
    <Tree
      treeData={treeData}
      expandedKeys={expandedKeys}
      onExpand={handleExpand}
      onSelect={handleSelect}
      selectedKeys={selectedFolderId ? [selectedFolderId] : []}
      showIcon
      switcherIcon={({ expanded }: { expanded?: boolean }) =>
        expanded ? (
          <ChevronDown className="w-3 h-3 text-text-subtle" />
        ) : (
          <ChevronRight className="w-3 h-3 text-text-subtle" />
        )
      }
      className="folder-tree bg-transparent"
      blockNode
    />
  )
}

export default FolderTree
