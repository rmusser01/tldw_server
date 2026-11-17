import React from "react"
import { Modal, Tree, Empty, Spin, Input, Button, message } from "antd"
import type { TreeProps } from "antd"
import { Folder, Plus } from "lucide-react"
import { useTranslation } from "react-i18next"
import { shallow } from "zustand/shallow"
import { useFolderStore, type FolderTreeNode, type FolderOperation } from "@/store/folder"

type BulkFolderPickerModalProps = {
  open: boolean
  operation?: FolderOperation
  conversationIds: string[]
  onClose: () => void
  onSuccess?: () => void
}

export const BulkFolderPickerModal: React.FC<BulkFolderPickerModalProps> = ({
  open,
  operation,
  conversationIds,
  onClose,
  onSuccess
}) => {
  const { t } = useTranslation(["sidepanel", "common"])
  const [selectedFolderIds, setSelectedFolderIds] = React.useState<number[]>([])
  const [newFolderName, setNewFolderName] = React.useState("")
  const [isCreating, setIsCreating] = React.useState(false)
  const [isSaving, setIsSaving] = React.useState(false)
  const selectionKey = JSON.stringify(conversationIds)
  const lifetime = React.useMemo(() => ({ open, operation, selectionKey }), [open, operation, selectionKey])
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
    folderTree,
    isLoading,
    folderApiAvailable,
    serverDataScope,
    error,
    addConversationToFolder,
    createFolder,
    refreshFromServer
  } = useFolderStore(
    (state) => ({
      folderTree: state.getFolderTree(),
      isLoading: state.isLoading,
      folderApiAvailable: state.folderApiAvailable,
      serverDataScope: state.serverDataScope,
      error: state.error,
      addConversationToFolder: state.addConversationToFolder,
      createFolder: state.createFolder,
      refreshFromServer: state.refreshFromServer
    }),
    shallow
  )

  React.useEffect(() => {
    if (open && isCurrent()) void refreshFromServer(scopedOperation)
  }, [open, scopedOperation, refreshFromServer, isCurrent])

  const ready = !scopedOperation || (serverDataScope === scopedOperation &&
    isCurrent() && !isLoading && !error && folderApiAvailable === true)

  React.useEffect(() => {
    setIsSaving(false)
    setIsCreating(false)
    if (!open || scopedOperation) {
      setSelectedFolderIds([])
      setNewFolderName("")
    }
  }, [open, scopedOperation, lifetime])

  const treeData: TreeProps["treeData"] = React.useMemo(() => {
    const buildTreeData = (
      nodes: FolderTreeNode[]
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
    if (!isCurrent() || !ready || isCreating || !newFolderName.trim()) return
    setIsCreating(true)
    try {
      const created = await createFolder(newFolderName.trim(), undefined, scopedOperation)
      if (isCurrent() && created) {
        setNewFolderName("")
        setSelectedFolderIds((prev) => [...prev, created.id])
      } else if (isCurrent()) {
        message.error(t("sidepanel:folderPicker.createError", "Unable to create the folder. Try again."))
      }
    } catch {
      if (isCurrent()) message.error(t("sidepanel:folderPicker.createError", "Unable to create the folder. Try again."))
    } finally {
      if (isCurrent()) setIsCreating(false)
    }
  }

  const handleClose = () => {
    if (lifetimeRef.current === lifetime) lifetimeRef.current = null
    if (scopedOperation) useFolderStore.getState().releaseServerData(scopedOperation)
    onClose()
  }

  const handleSave = async () => {
    if (!isCurrent() || !ready || isSaving) return
    if (conversationIds.length === 0 || selectedFolderIds.length === 0) return
    setIsSaving(true)
    try {
      const tasks = conversationIds.flatMap((conversationId) =>
        selectedFolderIds.map((folderId) => ({
          conversationId,
          folderId
        }))
      )
      const results: PromiseSettledResult<{
        ok: boolean
        conversationId: string
        folderId: number
      }>[] = []
      // Limit concurrent requests to avoid large bursts on bulk apply.
      const maxConcurrentRequests = 12

      for (let i = 0; i < tasks.length; i += maxConcurrentRequests) {
        if (!isCurrent()) return
        const chunk = tasks.slice(i, i + maxConcurrentRequests).map(async (task) => {
          if (!isCurrent()) return { ok: false, conversationId: task.conversationId, folderId: task.folderId }
          const ok = await addConversationToFolder(
            task.conversationId,
            task.folderId,
            scopedOperation
          )
          return { ok, conversationId: task.conversationId, folderId: task.folderId }
        })
        results.push(...(await Promise.allSettled(chunk)))
      }
      const failures = results.filter(
        (result) => result.status === "fulfilled" && !result.value.ok
      ).length +
        results.filter((result) => result.status === "rejected").length

      if (!isCurrent()) return
      if (failures > 0) {
        message.error(
          t("sidepanel:multiSelect.folderApplyPartialError", "Some chats could not be added to folders. Your selection is kept so you can try again.")
        )
        return
      } else {
        message.success(
          t("sidepanel:multiSelect.folderApplySuccess", "Folders applied to selected chats.")
        )
      }

      onSuccess?.()
      if (isCurrent()) handleClose()
    } catch {
      if (isCurrent()) message.error(t("sidepanel:multiSelect.folderApplyPartialError", "Some chats could not be added to folders. Your selection is kept so you can try again."))
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

    if (folderTree.length === 0) {
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
              disabled={!ready || isSaving || !newFolderName.trim()}
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
              disabled={!ready || isSaving || !newFolderName.trim()}
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
      title={t("sidepanel:multiSelect.addToFolder", "Add to folders")}
      onCancel={handleClose}
      onOk={handleSave}
      okText={t("common:save", "Save")}
      cancelText={t("common:cancel", "Cancel")}
      okButtonProps={{
        loading: isSaving,
        disabled:
          !ready || isLoading || folderApiAvailable === false ||
          conversationIds.length === 0 ||
          selectedFolderIds.length === 0
      }}
      destroyOnHidden
    >
      {renderContent()}
    </Modal>
  )
}

export default BulkFolderPickerModal
