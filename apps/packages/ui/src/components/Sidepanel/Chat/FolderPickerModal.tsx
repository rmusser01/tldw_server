import React from "react"
import { Modal, Tree, Empty, Spin, Input, Button, message } from "antd"
import type { TreeProps } from "antd"
import { Folder, Plus } from "lucide-react"
import { useTranslation } from "react-i18next"
import { useFolderStore, type FolderOperation } from "@/store/folder"

type FolderPickerModalProps = {
  open: boolean
  onClose: () => void
  conversationId: string | null
  onSuccess?: () => void
  /**
   * The account and view the picker was opened for. With it, folders are read
   * and changed only for that account, and nothing is written once it is no
   * longer current (an account switch, another chat).
   */
  operation?: FolderOperation
}

/**
 * Choose the folders one chat belongs to: the ones it is in start ticked, and
 * unticking one removes the chat from it.
 */
export const FolderPickerModal: React.FC<FolderPickerModalProps> = ({
  open,
  onClose,
  conversationId,
  onSuccess,
  operation
}) => {
  const { t } = useTranslation(["sidepanel", "common"])
  const [selectedFolderIds, setSelectedFolderIds] = React.useState<number[]>([])
  const [newFolderName, setNewFolderName] = React.useState("")
  const [isCreating, setIsCreating] = React.useState(false)
  const [isSaving, setIsSaving] = React.useState(false)
  const lifetime = React.useMemo(() => ({ open, operation, conversationId }), [open, operation, conversationId])
  const lifetimeRef = React.useRef<typeof lifetime | null>(null)
  const isCurrent = React.useCallback(() => lifetimeRef.current === lifetime &&
    (!operation || operation.isCurrent()), [lifetime, operation])
  const scopedOperation = React.useMemo<FolderOperation | undefined>(() => operation
    ? { requestScope: operation.requestScope, isCurrent }
    : undefined, [operation, isCurrent])
  React.useLayoutEffect(() => {
    if (open) lifetimeRef.current = lifetime
    return () => {
      if (lifetimeRef.current === lifetime) lifetimeRef.current = null
      if (scopedOperation) useFolderStore.getState().releaseServerData(scopedOperation)
    }
  }, [open, lifetime, scopedOperation])

  const {
    folders,
    isLoading,
    error,
    folderApiAvailable,
    serverDataScope,
    getFolderTree,
    getFoldersForConversation,
    addConversationToFolder,
    removeConversationFromFolder,
    createFolder,
    refreshFromServer
  } = useFolderStore()
  // A captured picker works only on the snapshot it read for its account.
  const ready = scopedOperation
    ? serverDataScope === scopedOperation && isCurrent() && !isLoading && !error && folderApiAvailable === true
    : !isLoading

  // Refresh folder data when modal opens
  React.useEffect(() => {
    if (!open) return
    if (scopedOperation) {
      if (isCurrent()) void refreshFromServer(scopedOperation)
    } else if (folderApiAvailable !== false) {
      refreshFromServer()
    }
  }, [open, folderApiAvailable, refreshFromServer, scopedOperation, isCurrent])

  // Pre-select the folders that already contain this conversation, once their
  // memberships are known.
  const preselectedFor = React.useRef<string | null>(null)
  React.useEffect(() => {
    if (!open) {
      preselectedFor.current = null
      return
    }
    if (!conversationId || !ready || preselectedFor.current === conversationId) return
    preselectedFor.current = conversationId
    setSelectedFolderIds(getFoldersForConversation(conversationId).map((f) => f.id))
  }, [open, conversationId, ready, getFoldersForConversation])

  const folderTree = React.useMemo(() => {
    return getFolderTree()
  }, [getFolderTree, folders])

  const treeData: TreeProps["treeData"] = React.useMemo(() => {
    const buildTreeData = (
      nodes: ReturnType<typeof getFolderTree>
    ): TreeProps["treeData"] => {
      return nodes.map((node) => ({
        key: node.key,
        title: (
          <span className="flex items-center gap-1.5">
            <Folder className="size-3.5 text-text-muted" />
            <span>{node.title}</span>
            {node.itemCount > 0 && (
              <span className="text-xs text-text-subtle">({node.itemCount})</span>
            )}
          </span>
        ),
        children: node.children.length > 0 ? buildTreeData(node.children) : undefined
      }))
    }
    return buildTreeData(folderTree)
  }, [folderTree])

  const handleCheck: TreeProps["onCheck"] = (checked) => {
    if (Array.isArray(checked)) {
      setSelectedFolderIds(checked.map(Number))
      return
    }
    if (checked && typeof checked === "object" && "checked" in checked) {
      setSelectedFolderIds((checked.checked ?? []).map(Number))
    }
  }

  const handleCreateFolder = async () => {
    if (!newFolderName.trim() || !ready || !isCurrent()) return
    setIsCreating(true)
    try {
      const created = await createFolder(newFolderName.trim(), undefined, scopedOperation)
      if (created && isCurrent()) {
        setNewFolderName("")
        // Auto-select the newly created folder
        setSelectedFolderIds((prev) => [...prev, created.id])
      }
    } finally {
      if (isCurrent()) setIsCreating(false)
    }
  }

  const handleSave = async () => {
    if (!conversationId || !ready || !isCurrent() || isSaving) return
    setIsSaving(true)
    try {
      const existingFolders = getFoldersForConversation(conversationId)
      const existingIds = new Set(existingFolders.map((f) => f.id))
      const selectedSet = new Set(selectedFolderIds)
      let failed = false

      // Remove from folders that are no longer selected
      for (const folder of existingFolders) {
        if (!selectedSet.has(folder.id)) {
          if (!isCurrent()) return
          if (!(await removeConversationFromFolder(conversationId, folder.id, scopedOperation))) failed = true
        }
      }

      // Add to newly selected folders
      for (const folderId of selectedFolderIds) {
        if (!existingIds.has(folderId)) {
          if (!isCurrent()) return
          if (!(await addConversationToFolder(conversationId, folderId, scopedOperation))) failed = true
        }
      }

      if (!isCurrent()) return
      if (failed) {
        message.error(t("sidepanel:folderPicker.saveError", "Some folder changes couldn't be saved. Try again."))
        return
      }
      onSuccess?.()
      onClose()
    } finally {
      if (isCurrent()) setIsSaving(false)
    }
  }

  const renderContent = () => {
    if (scopedOperation && !ready && !(serverDataScope === scopedOperation && folderApiAvailable === false)) {
      if (isLoading) return <div className="flex items-center justify-center py-8"><Spin /></div>
      return <div role="status" className="space-y-3">
        <p>{error
          ? t("sidepanel:multiSelect.organizationLoadError", "Unable to load folders and tags for this account.")
          : t("sidepanel:multiSelect.organizationRefreshRequired", "Refresh folders and tags for this account.")}</p>
        <Button onClick={() => { if (isCurrent()) void refreshFromServer(scopedOperation) }}>
          {t("common:retry", "Retry")}
        </Button>
      </div>
    }
    if (folderApiAvailable === false) {
      return (
        <Empty
          description={t(
            "sidepanel:folderPicker.notAvailable",
            "Folder organization is not available on this server"
          )}
          image={Empty.PRESENTED_IMAGE_SIMPLE}
        />
      )
    }

    if (isLoading) {
      return (
        <div className="flex items-center justify-center py-8">
          <Spin />
        </div>
      )
    }

    if (folders.filter((f) => !f.deleted).length === 0) {
      return (
        <div className="space-y-4">
          <Empty
            description={t(
              "sidepanel:folderPicker.noFolders",
              "No folders yet. Create one below."
            )}
            image={Empty.PRESENTED_IMAGE_SIMPLE}
          />
          <div className="flex gap-2">
            <Input
              placeholder={t(
                "sidepanel:folderPicker.newFolderPlaceholder",
                "New folder name"
              )}
              value={newFolderName}
              onChange={(e) => setNewFolderName(e.target.value)}
              onPressEnter={handleCreateFolder}
            />
            <Button
              type="primary"
              icon={<Plus className="size-4" />}
              onClick={handleCreateFolder}
              loading={isCreating}
              disabled={!newFolderName.trim()}
            >
              {t("common:create", "Create")}
            </Button>
          </div>
        </div>
      )
    }

    return (
      <div className="space-y-4">
        <Tree
          checkable
          defaultExpandAll
          checkedKeys={selectedFolderIds}
          onCheck={handleCheck}
          treeData={treeData}
          className="bg-transparent"
        />
        <div className="border-t border-border pt-4">
          <div className="text-caption text-text-muted mb-2">
            {t("sidepanel:folderPicker.createNew", "Create new folder")}
          </div>
          <div className="flex gap-2">
            <Input
              placeholder={t(
                "sidepanel:folderPicker.newFolderPlaceholder",
                "New folder name"
              )}
              value={newFolderName}
              onChange={(e) => setNewFolderName(e.target.value)}
              onPressEnter={handleCreateFolder}
              size="small"
            />
            <Button
              size="small"
              icon={<Plus className="size-3" />}
              onClick={handleCreateFolder}
              loading={isCreating}
              disabled={!newFolderName.trim()}
            >
              {t("common:add", "Add")}
            </Button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <Modal
      open={open}
      title={t("sidepanel:folderPicker.title", "Add to folder")}
      onCancel={onClose}
      onOk={handleSave}
      okText={t("common:save", "Save")}
      cancelText={t("common:cancel", "Cancel")}
      okButtonProps={{ loading: isSaving, disabled: folderApiAvailable === false || !ready }}
      destroyOnHidden
    >
      {renderContent()}
    </Modal>
  )
}

export default FolderPickerModal
