import React from "react"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { toWorldBookFormValues } from "../worldBookFormUtils"
import type { ConfirmDangerOptions } from "@/components/Common/confirm-danger"

type ConfirmDanger = (options: ConfirmDangerOptions) => Promise<boolean>

export interface UseWorldBookBulkActionsDeps {
  /** World books list from the query */
  data: any[] | undefined
  /** React Query client for cache invalidation */
  qc: { invalidateQueries: (opts: { queryKey: string[] }) => void }
  /** Notification API */
  notification: {
    success: (opts: { message: string; description?: string }) => void
    error: (opts: { message: string; description?: string }) => void
    info: (opts: { message: string; description?: string }) => void
  }
  /** Confirm danger dialog */
  confirmDanger: ConfirmDanger
  /** Undo notification handler */
  showUndoNotification: (opts: {
    title: string
    description: string
    duration: number
    onUndo: () => void
  }) => void
  /** Delete mutation */
  deleteWB: (id: number) => void
  /** Whether delete mutation is pending */
  deleting: boolean
  /** Whether attachments are loading */
  attachmentsLoading: boolean
  /** Get attached characters for a world book */
  getAttachedCharacters: (worldBookId: number) => any[]
  /** Selected world book keys (controlled externally by filtering hook) */
  selectedWorldBookKeys: React.Key[]
  /** Setter for selected world book keys */
  setSelectedWorldBookKeys: React.Dispatch<React.SetStateAction<React.Key[]>>
}

export function useWorldBookBulkActions(deps: UseWorldBookBulkActionsDeps) {
  const {
    data,
    qc,
    notification,
    confirmDanger,
    showUndoNotification,
    deleteWB,
    deleting,
    attachmentsLoading,
    getAttachedCharacters,
    selectedWorldBookKeys,
    setSelectedWorldBookKeys,
  } = deps

  const [bulkWorldBookAction, setBulkWorldBookAction] = React.useState<
    "enable" | "disable" | "delete" | null
  >(null)
  const [pendingDeleteIds, setPendingDeleteIds] = React.useState<number[]>([])
  const deleteTimersRef = React.useRef<Record<number, any>>({})

  const cancelPendingWorldBookDeletes = React.useCallback(() => {
    Object.values(deleteTimersRef.current).forEach((timer) => clearTimeout(timer))
    deleteTimersRef.current = {}
    setPendingDeleteIds([])
    notification.info({
      message: "Pending deletions canceled",
      description: "No world books are currently scheduled for deletion."
    })
  }, [notification])

  const requestDeleteWorldBook = async (record: any) => {
    const entryCount = record.entry_count || 0
    const attached = attachmentsLoading ? null : getAttachedCharacters(record.id)
    const attachedNames = attached ? attached.map((c: any) => c.name || `Character ${c.id}`) : []
    const attachedSummary = attachmentsLoading
      ? "Attachment info loading"
      : attachedNames.length === 0
        ? "No character attachments"
        : `${attachedNames.length} attached (${attachedNames.slice(0, 3).join(", ")}${attachedNames.length > 3 ? ` +${attachedNames.length - 3} more` : ""})`
    const ok = await confirmDanger({
      title: `Delete "${record.name}"?`,
      content: (
        <div className="space-y-2">
          <p>This will permanently remove:</p>
          <ul className="list-disc list-inside text-sm">
            <li>{entryCount} {entryCount === 1 ? "entry" : "entries"}</li>
            <li>{attachedSummary}</li>
          </ul>
          <p className="text-danger text-sm mt-2">Deletion will run after 10 seconds unless you undo.</p>
          <p className="text-xs text-text-muted">
            Pending deletions are local to this tab. Refreshing or navigating away cancels the timer.
          </p>
        </div>
      ),
      okText: "Delete",
      cancelText: "Cancel",
      autoFocusButton: "ok"
    })
    if (!ok) return

    if (deleteTimersRef.current[record.id]) return
    setPendingDeleteIds((prev) => [...prev, record.id])
    deleteTimersRef.current[record.id] = setTimeout(() => {
      deleteWB(record.id)
      setPendingDeleteIds((prev) => prev.filter((id) => id !== record.id))
      delete deleteTimersRef.current[record.id]
    }, 10000)

    showUndoNotification({
      title: "World book deletion scheduled",
      description:
        `"${record.name}" will be deleted in 10 seconds. ` +
        "Refresh or navigation before timeout cancels pending deletion.",
      duration: 10,
      onUndo: () => {
        if (deleteTimersRef.current[record.id]) {
          clearTimeout(deleteTimersRef.current[record.id])
          delete deleteTimersRef.current[record.id]
        }
        setPendingDeleteIds((prev) => prev.filter((id) => id !== record.id))
      }
    })
  }

  const handleBulkWorldBookAction = async (operation: "enable" | "disable" | "delete") => {
    const selectedIds = selectedWorldBookKeys
      .map((key) => Number(key))
      .filter((id) => Number.isFinite(id) && id > 0)
    if (selectedIds.length === 0 || bulkWorldBookAction) return

    if (operation === "delete") {
      const ok = await confirmDanger({
        title: "Delete selected world books?",
        content: `This will permanently remove ${selectedIds.length} world books and their entries.`,
        okText: "Delete",
        cancelText: "Cancel"
      })
      if (!ok) return
    }

    setBulkWorldBookAction(operation)
    try {
      if (operation === "delete") {
        await Promise.all(selectedIds.map((id) => tldwClient.deleteWorldBook(id)))
      } else {
        const nextEnabled = operation === "enable"
        const booksById = new Map(((data || []) as any[]).map((book: any) => [book.id, book]))
        await Promise.all(
          selectedIds.map((id) => {
            const record = booksById.get(id)
            if (!record) return Promise.resolve(null)
            const values = toWorldBookFormValues(record)
            const payload = {
              name: values.name,
              description: values.description,
              enabled: nextEnabled,
              scan_depth: values.scan_depth,
              token_budget: values.token_budget,
              recursive_scanning: values.recursive_scanning
            }
            if (typeof record?.version === "number") {
              return tldwClient.updateWorldBook(id, payload, {
                expectedVersion: record.version
              })
            }
            return tldwClient.updateWorldBook(id, payload)
          })
        )
      }

      qc.invalidateQueries({ queryKey: ["tldw:listWorldBooks"] })
      setSelectedWorldBookKeys([])
      notification.success({
        message: "Bulk action complete",
        description:
          operation === "delete"
            ? `Deleted ${selectedIds.length} world books.`
            : `${operation === "enable" ? "Enabled" : "Disabled"} ${selectedIds.length} world books.`
      })
    } catch (e: any) {
      notification.error({
        message: "Bulk action failed",
        description: e?.message || "Could not complete world-book bulk operation."
      })
    } finally {
      setBulkWorldBookAction(null)
    }
  }

  // Cleanup timers on unmount
  React.useEffect(() => {
    return () => {
      Object.values(deleteTimersRef.current).forEach((t) => clearTimeout(t))
      deleteTimersRef.current = {}
    }
  }, [])

  return {
    // state
    bulkWorldBookAction,
    pendingDeleteIds,
    deleteTimersRef,
    // callbacks
    cancelPendingWorldBookDeletes,
    requestDeleteWorldBook,
    handleBulkWorldBookAction,
  }
}
