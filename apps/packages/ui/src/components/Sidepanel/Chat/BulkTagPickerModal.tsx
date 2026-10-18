import React from "react"
import { Modal, Empty, Spin, Select, Button, message } from "antd"
import { useTranslation } from "react-i18next"
import { shallow } from "zustand/shallow"
import { useFolderStore, type FolderOperation } from "@/store/folder"

type BulkTagPickerModalProps = {
  open: boolean
  operation?: FolderOperation
  conversationIds: string[]
  onClose: () => void
  onSuccess?: () => void
}

const normalizeTag = (value: string) => value.trim()

export const BulkTagPickerModal: React.FC<BulkTagPickerModalProps> = ({
  open,
  operation,
  conversationIds,
  onClose,
  onSuccess
}) => {
  const { t } = useTranslation(["sidepanel", "common"])
  const [selectedTags, setSelectedTags] = React.useState<string[]>([])
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
    keywords,
    isLoading,
    folderApiAvailable,
    serverDataScope,
    error,
    ensureKeyword,
    addKeywordToConversation,
    refreshFromServer
  } = useFolderStore(
    (state) => ({
      keywords: state.keywords,
      isLoading: state.isLoading,
      folderApiAvailable: state.folderApiAvailable,
      serverDataScope: state.serverDataScope,
      error: state.error,
      ensureKeyword: state.ensureKeyword,
      addKeywordToConversation: state.addKeywordToConversation,
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
    if (!open || scopedOperation) {
      setSelectedTags([])
    }
  }, [open, scopedOperation, lifetime])

  const keywordOptions = React.useMemo(
    () =>
      keywords
        .filter((keyword) => !keyword.deleted)
        .map((keyword) => ({
          label: keyword.keyword,
          value: keyword.keyword
        })),
    [keywords]
  )

  const handleClose = () => {
    if (lifetimeRef.current === lifetime) lifetimeRef.current = null
    if (scopedOperation) useFolderStore.getState().releaseServerData(scopedOperation)
    onClose()
  }

  const handleSave = async () => {
    if (!isCurrent() || !ready || isSaving) return
    const normalizedTags = selectedTags
      .map(normalizeTag)
      .filter(Boolean)
    const uniqueTags = Array.from(
      new Map(normalizedTags.map((tag) => [tag.toLowerCase(), tag])).values()
    )

    if (uniqueTags.length === 0) return
    if (conversationIds.length === 0) return

    setIsSaving(true)
    try {
      const keywords = await Promise.all(
        uniqueTags.map((tag) => ensureKeyword(tag, scopedOperation))
      )
      if (!isCurrent()) return
      const keywordIds = keywords
        .filter(Boolean)
        .map((keyword) => keyword!.id)

      if (keywordIds.length === 0) {
        message.error(
          t("sidepanel:multiSelect.tagsApplyError", "Unable to create tags.")
        )
        return
      }

      const tasks = conversationIds.flatMap((conversationId) =>
        keywordIds.map(async (keywordId) => {
          if (!isCurrent()) return { ok: false, conversationId, keywordId }
          const ok = await addKeywordToConversation(conversationId, keywordId, scopedOperation)
          return { ok, conversationId, keywordId }
        })
      )
      const results = await Promise.allSettled(tasks)
      const failures = (keywords.length - keywordIds.length) + results.filter(
        (result) => result.status === "fulfilled" && !result.value.ok
      ).length +
        results.filter((result) => result.status === "rejected").length

      if (!isCurrent()) return
      if (failures > 0) {
        message.error(
          t("sidepanel:multiSelect.tagsApplyPartialError", "Some tags could not be created or applied. Your selection is kept so you can try again.")
        )
        return
      } else {
        message.success(
          t("sidepanel:multiSelect.tagsApplySuccess", "Tags applied to selected chats.")
        )
      }

      onSuccess?.()
      if (isCurrent()) handleClose()
    } catch {
      if (isCurrent()) message.error(t("sidepanel:multiSelect.tagsApplyPartialError", "Some tags could not be created or applied. Your selection is kept so you can try again."))
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
            "sidepanel:multiSelect.tagsUnavailable",
            "Tags are not available on this server"
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

    return (
      <div className="space-y-2">
        <div className="text-caption text-text-muted">
          {t("sidepanel:multiSelect.tagsHint", "Add tags to the selected chats.")}
        </div>
        <Select
          mode="tags"
          value={selectedTags}
          onChange={(vals) => setSelectedTags(vals as string[])}
          options={keywordOptions}
          placeholder={t(
            "sidepanel:multiSelect.tagsPlaceholder",
            "Add tags..."
          )}
          className="w-full"
        />
      </div>
    )
  }

  return (
    <Modal
      open={open}
      title={t("sidepanel:multiSelect.addTags", "Add tags")}
      onCancel={handleClose}
      onOk={handleSave}
      okText={t("common:save", "Save")}
      cancelText={t("common:cancel", "Cancel")}
      okButtonProps={{
        loading: isSaving,
        disabled:
          !ready || isLoading || folderApiAvailable === false ||
          conversationIds.length === 0 ||
          selectedTags.length === 0
      }}
      destroyOnHidden
    >
      {renderContent()}
    </Modal>
  )
}

export default BulkTagPickerModal
