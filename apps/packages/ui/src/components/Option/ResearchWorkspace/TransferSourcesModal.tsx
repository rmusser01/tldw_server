import React from "react"
import { Alert, Button, Checkbox, Input, Modal, Radio, message } from "antd"
import { useTranslation } from "react-i18next"
import { useWorkspaceStore } from "@/store/workspace"
import type {
  WorkspaceSourceTransferConflictResolution,
  WorkspaceSourceTransferEmptyFolderPolicy
} from "@/types/workspace"
import {
  WORKSPACE_UNDO_WINDOW_MS,
  scheduleWorkspaceUndoAction,
  undoWorkspaceAction
} from "./undo-manager"

export type TransferSourcesModalEntryPoint = "sources" | "header"

export interface TransferSourcesModalLaunchRequest {
  entryPoint: TransferSourcesModalEntryPoint
  selectedSourceIds: string[]
  eligibleSelectedSourceIds: string[]
  totalSelectedCount: number
  hiddenSelectedCount: number
  ineligibleSelectedCount: number
}

interface TransferSourcesModalProps {
  open: boolean
  request: TransferSourcesModalLaunchRequest | null
  onCancel: () => void
}

type TransferDestinationKind = "existing" | "new"
type TransferStep =
  | "mode"
  | "destination"
  | "summary"
  | "conflicts"
  | "cleanup"
  | "complete"

type ConflictCandidate = {
  mediaId: number
  title: string
}

const DEFAULT_NEW_WORKSPACE_NAME = "New Research"

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object"

const getSnapshotSources = (snapshot: unknown): Array<{ mediaId?: number }> => {
  if (!isRecord(snapshot) || !Array.isArray(snapshot.sources)) {
    return []
  }
  return snapshot.sources as Array<{ mediaId?: number }>
}

export const TransferSourcesModal: React.FC<TransferSourcesModalProps> = ({
  open,
  request,
  onCancel
}) => {
  const { t } = useTranslation(["playground", "common"])
  const [messageApi, messageContextHolder] = message.useMessage()
  const workspaceId = useWorkspaceStore((s) => s.workspaceId)
  const savedWorkspaces = useWorkspaceStore((s) => s.savedWorkspaces) || []
  const archivedWorkspaces = useWorkspaceStore((s) => s.archivedWorkspaces) || []
  const workspaceSnapshots = useWorkspaceStore((s) => s.workspaceSnapshots) || {}
  const sources = useWorkspaceStore((s) => s.sources) || []
  const captureUndoSnapshot = useWorkspaceStore((s) => s.captureUndoSnapshot)
  const restoreUndoSnapshot = useWorkspaceStore((s) => s.restoreUndoSnapshot)
  const switchWorkspace = useWorkspaceStore((s) => s.switchWorkspace)
  const transferSourcesBetweenWorkspaces = useWorkspaceStore(
    (s) => s.transferSourcesBetweenWorkspaces
  )

  const [mode, setMode] = React.useState<"copy" | "move">("copy")
  const [step, setStep] = React.useState<TransferStep>("mode")
  const [destinationKind, setDestinationKind] =
    React.useState<TransferDestinationKind>("existing")
  const [destinationWorkspaceId, setDestinationWorkspaceId] = React.useState("")
  const [newWorkspaceName, setNewWorkspaceName] = React.useState(
    DEFAULT_NEW_WORKSPACE_NAME
  )
  const [emptyFolderPolicy, setEmptyFolderPolicy] =
    React.useState<WorkspaceSourceTransferEmptyFolderPolicy>("keep")
  const [applyToAllRemaining, setApplyToAllRemaining] = React.useState(false)
  const [conflictResolutions, setConflictResolutions] = React.useState<
    Record<number, WorkspaceSourceTransferConflictResolution>
  >({})
  const [resultSummary, setResultSummary] = React.useState<{
    destinationWorkspaceId: string
    destinationWasCreated: boolean
    transferredCount: number
    conflictsSkipped: number
  } | null>(null)
  const [submitError, setSubmitError] = React.useState<string | null>(null)
  const initializedRequestRef =
    React.useRef<TransferSourcesModalLaunchRequest | null>(null)

  const destinationOptions = React.useMemo(() => {
    const archivedIds = new Set(archivedWorkspaces.map((workspace) => workspace.id))
    return savedWorkspaces.filter(
      (workspace) =>
        workspace.id !== workspaceId && !archivedIds.has(workspace.id)
    )
  }, [archivedWorkspaces, savedWorkspaces, workspaceId])

  React.useEffect(() => {
    if (!open || !request) {
      initializedRequestRef.current = null
      return
    }
    if (initializedRequestRef.current === request) {
      return
    }

    setMode("copy")
    setStep(request.entryPoint === "header" ? "destination" : "mode")
    setDestinationKind(
      request.entryPoint === "header" || destinationOptions.length === 0
        ? "new"
        : "existing"
    )
    setDestinationWorkspaceId(destinationOptions[0]?.id || "")
    setNewWorkspaceName(DEFAULT_NEW_WORKSPACE_NAME)
    setEmptyFolderPolicy("keep")
    setApplyToAllRemaining(false)
    setConflictResolutions({})
    setResultSummary(null)
    setSubmitError(null)
    initializedRequestRef.current = request
  }, [destinationOptions, open, request])

  const eligibleSources = React.useMemo(() => {
    if (!request) return []
    const eligibleIdSet = new Set(request.eligibleSelectedSourceIds)
    return sources.filter((source) => eligibleIdSet.has(source.id))
  }, [request, sources])

  const conflictCandidates = React.useMemo<ConflictCandidate[]>(() => {
    if (!request || destinationKind !== "existing" || !destinationWorkspaceId) {
      return []
    }

    const destinationSnapshot = workspaceSnapshots[destinationWorkspaceId]
    const destinationMediaIds = new Set(
      getSnapshotSources(destinationSnapshot)
        .map((source) => Number(source.mediaId))
        .filter((mediaId) => Number.isFinite(mediaId) && mediaId > 0)
    )
    const seenMediaIds = new Set<number>()

    return eligibleSources
      .filter((source) => {
        if (seenMediaIds.has(source.mediaId)) return false
        seenMediaIds.add(source.mediaId)
        return destinationMediaIds.has(source.mediaId)
      })
      .map((source) => ({
        mediaId: source.mediaId,
        title: source.title
      }))
  }, [
    destinationKind,
    destinationWorkspaceId,
    eligibleSources,
    request,
    workspaceSnapshots
  ])

  const unresolvedConflicts = React.useMemo(
    () =>
      conflictCandidates.filter(
        (candidate) => !conflictResolutions[candidate.mediaId]
      ),
    [conflictCandidates, conflictResolutions]
  )

  const canSubmit = React.useMemo(() => {
    if (!request) return false
    if (request.eligibleSelectedSourceIds.length === 0) return false
    if (destinationKind === "existing") {
      return destinationWorkspaceId.trim().length > 0
    }
    return newWorkspaceName.trim().length > 0
  }, [destinationKind, destinationWorkspaceId, newWorkspaceName, request])

  const handleResolveConflict = React.useCallback(
    (
      mediaId: number,
      resolution: WorkspaceSourceTransferConflictResolution
    ) => {
      setConflictResolutions((current) => {
        if (!applyToAllRemaining) {
          return {
            ...current,
            [mediaId]: resolution
          }
        }

        const next = { ...current, [mediaId]: resolution }
        for (const candidate of conflictCandidates) {
          if (!next[candidate.mediaId]) {
            next[candidate.mediaId] = resolution
          }
        }
        return next
      })
    },
    [applyToAllRemaining, conflictCandidates]
  )

  const submitTransfer = React.useCallback(() => {
    if (!request || !canSubmit) {
      return
    }

    try {
      const postUndoStep: TransferStep =
        mode === "move"
          ? "cleanup"
          : conflictCandidates.length > 0
            ? "conflicts"
            : "summary"
      const undoSnapshot = captureUndoSnapshot()
      let transferResult: ReturnType<typeof transferSourcesBetweenWorkspaces> = null
      const undoHandle = scheduleWorkspaceUndoAction({
        apply: () => {
          const result = transferSourcesBetweenWorkspaces({
            mode,
            destination:
              destinationKind === "existing"
                ? { kind: "existing", workspaceId: destinationWorkspaceId }
                : { kind: "new", name: newWorkspaceName.trim() },
            selectedSourceIds: request.eligibleSelectedSourceIds,
            conflictResolutions,
            emptyFolderPolicy,
            switchToDestinationOnComplete: false
          })

          if (!result) {
            throw new Error(
              t(
                "playground:sources.transferUnavailable",
                "Could not complete the source transfer."
              )
            )
          }

          transferResult = result
        },
        undo: () => {
          restoreUndoSnapshot(undoSnapshot)
        }
      })

      if (!transferResult) {
        throw new Error(
          t(
            "playground:sources.transferUnavailable",
            "Could not complete the source transfer."
          )
        )
      }

      setResultSummary({
        destinationWorkspaceId: transferResult.destinationWorkspaceId,
        destinationWasCreated: transferResult.destinationWasCreated,
        transferredCount: transferResult.transferredMediaIds.length,
        conflictsSkipped: transferResult.conflictsSkipped.length
      })
      setStep("complete")
      setSubmitError(null)

      const undoMessageKey = `workspace-transfer-undo-${undoHandle.id}`
      const messageConfig = {
        key: undoMessageKey,
        type: "warning",
        duration: WORKSPACE_UNDO_WINDOW_MS / 1000,
        content: t(
          "playground:sources.transferUndoAvailable",
          "Sources transferred. You can undo for a few seconds."
        ),
        btn: (
          <Button
            size="small"
            type="link"
            onClick={() => {
              if (undoWorkspaceAction(undoHandle.id)) {
                setResultSummary(null)
                setSubmitError(null)
                setStep(postUndoStep)
                messageApi.success(
                  t(
                    "playground:sources.transferRestored",
                    "Transfer undone."
                  )
                )
              }
              messageApi.destroy(undoMessageKey)
            }}
          >
            {t("common:undo", "Undo")}
          </Button>
        )
      }

      const maybeOpen = (messageApi as { open?: (config: unknown) => void }).open
      if (typeof maybeOpen === "function") {
        maybeOpen(messageConfig)
      } else {
        const maybeWarning = (
          messageApi as { warning?: (content: string) => void }
        ).warning
        if (typeof maybeWarning === "function") {
          maybeWarning(
            t(
              "playground:sources.transferUndoAvailable",
              "Sources transferred. You can undo for a few seconds."
            )
          )
        }
      }
    } catch (error) {
      setSubmitError(
        error instanceof Error
          ? error.message
          : t(
              "playground:sources.transferUnavailable",
              "Could not complete the source transfer."
            )
      )
    }
  }, [
    canSubmit,
    captureUndoSnapshot,
    conflictCandidates.length,
    conflictResolutions,
    destinationKind,
    destinationWorkspaceId,
    emptyFolderPolicy,
    mode,
    messageApi,
    newWorkspaceName,
    request,
    restoreUndoSnapshot,
    t,
    transferSourcesBetweenWorkspaces
  ])

  const handleNext = React.useCallback(() => {
    setSubmitError(null)

    if (step === "mode") {
      setStep("destination")
      return
    }

    if (step === "destination") {
      setStep("summary")
      return
    }

    if (step === "summary") {
      if (conflictCandidates.length > 0) {
        setStep("conflicts")
        return
      }
      if (mode === "move") {
        setStep("cleanup")
        return
      }
      submitTransfer()
      return
    }

    if (step === "conflicts") {
      if (unresolvedConflicts.length > 0) {
        return
      }
      if (mode === "move") {
        setStep("cleanup")
        return
      }
      submitTransfer()
      return
    }

    if (step === "cleanup") {
      submitTransfer()
    }
  }, [conflictCandidates.length, mode, step, submitTransfer, unresolvedConflicts.length])

  const handleBack = React.useCallback(() => {
    setSubmitError(null)

    if (step === "destination") {
      setStep("mode")
      return
    }

    if (step === "summary") {
      setStep("destination")
      return
    }

    if (step === "conflicts") {
      setStep("summary")
      return
    }

    if (step === "cleanup") {
      if (conflictCandidates.length > 0) {
        setStep("conflicts")
        return
      }
      setStep("summary")
    }
  }, [conflictCandidates.length, step])

  const handleOpenDestination = React.useCallback(() => {
    if (!resultSummary || resultSummary.destinationWasCreated) {
      return
    }
    switchWorkspace(resultSummary.destinationWorkspaceId)
    onCancel()
  }, [onCancel, resultSummary, switchWorkspace])

  const footer = (
    <div className="flex items-center justify-between gap-2">
      <div className="text-xs text-text-subtle">
        {request
          ? t(
              "playground:sources.transferProgress",
              "Selected: {{count}}",
              {
                count: request.totalSelectedCount
              }
            )
          : null}
      </div>
      <div className="flex items-center gap-2">
        {step !== "mode" && step !== "complete" ? (
          <Button onClick={handleBack}>{t("common:back", "Back")}</Button>
        ) : null}
        {step === "complete" ? (
          <Button type="primary" onClick={onCancel}>
            {t("common:done", "Done")}
          </Button>
        ) : (
          <Button
            type="primary"
            onClick={handleNext}
            disabled={
              (step === "destination" && !canSubmit) ||
              (step === "conflicts" && unresolvedConflicts.length > 0)
            }
          >
            {step === "cleanup" ||
            (step === "summary" &&
              conflictCandidates.length === 0 &&
              mode !== "move") ||
            (step === "conflicts" && mode !== "move")
              ? t("playground:sources.transferAction", "Transfer sources")
              : t("common:next", "Next")}
          </Button>
        )}
      </div>
    </div>
  )

  return (
    <>
      {messageContextHolder}
      <Modal
        title={t("playground:sources.transferTitle", "Transfer sources")}
        open={open}
        onCancel={onCancel}
        destroyOnHidden
        width={720}
        footer={footer}
      >
        {!request ? null : (
          <div className="space-y-4">
            <p className="text-sm text-text-muted">
              {t(
                "playground:sources.transferIntro",
                "Move or copy the selected sources into another workspace."
              )}
            </p>

            {submitError ? (
              <Alert type="error" showIcon title={submitError} />
            ) : null}

            {step === "mode" ? (
              <Radio.Group
                value={mode}
                onChange={(event) => setMode(event.target.value)}
                className="flex flex-col gap-3"
              >
                <Radio value="copy">Copy selected sources</Radio>
                <Radio value="move">Move selected sources</Radio>
              </Radio.Group>
            ) : null}

            {step === "destination" ? (
              <div className="space-y-4">
                <Radio.Group
                  value={destinationKind}
                  onChange={(event) =>
                    setDestinationKind(event.target.value as TransferDestinationKind)
                  }
                  className="flex flex-col gap-3"
                >
                  <Radio value="existing">Use an existing workspace</Radio>
                  <Radio value="new">Create a new workspace</Radio>
                </Radio.Group>

                {destinationKind === "existing" ? (
                  destinationOptions.length > 0 ? (
                    <Radio.Group
                      value={destinationWorkspaceId}
                      onChange={(event) => setDestinationWorkspaceId(event.target.value)}
                      className="flex flex-col gap-2 rounded-lg border border-border p-3"
                    >
                      {destinationOptions.map((workspace) => (
                        <Radio key={workspace.id} value={workspace.id}>
                          {workspace.name}
                        </Radio>
                      ))}
                    </Radio.Group>
                  ) : (
                    <Alert
                      type="info"
                      showIcon
                      title={t(
                        "playground:sources.transferNoDestinations",
                        "No eligible destination workspaces are available. Create a new workspace instead."
                      )}
                    />
                  )
                ) : (
                  <div className="space-y-2">
                    <label
                      className="block text-sm font-medium text-text"
                      htmlFor="transfer-new-workspace-name"
                    >
                      {t(
                        "playground:sources.transferNewWorkspaceLabel",
                        "Workspace name"
                      )}
                    </label>
                    <Input
                      id="transfer-new-workspace-name"
                      value={newWorkspaceName}
                      onChange={(event) => setNewWorkspaceName(event.target.value)}
                      placeholder={DEFAULT_NEW_WORKSPACE_NAME}
                    />
                  </div>
                )}
              </div>
            ) : null}

            {step === "summary" ? (
              <div className="space-y-3">
                <Alert
                  type="info"
                  showIcon
                  title={t(
                    "playground:sources.transferEligibleSummary",
                    "{{count}} ready sources will transfer.",
                    { count: request.eligibleSelectedSourceIds.length }
                  )}
                />
                {request.hiddenSelectedCount > 0 ? (
                  <p className="text-sm text-text-muted">
                    {t(
                      "playground:sources.transferHiddenSummary",
                      "{{count}} selected sources are hidden by current filters.",
                      { count: request.hiddenSelectedCount }
                    )}
                  </p>
                ) : null}
                {request.ineligibleSelectedCount > 0 ? (
                  <p className="text-sm text-text-muted">
                    {t(
                      "playground:sources.transferIneligibleSummary",
                      "{{count}} processing or errored sources are excluded from transfer.",
                      { count: request.ineligibleSelectedCount }
                    )}
                  </p>
                ) : null}
              </div>
            ) : null}

            {step === "conflicts" ? (
              <div className="space-y-4">
                <Checkbox
                  checked={applyToAllRemaining}
                  onChange={(event) => setApplyToAllRemaining(event.target.checked)}
                >
                  Apply to all remaining conflicts
                </Checkbox>

                {conflictCandidates.map((candidate) => (
                  <div
                    key={candidate.mediaId}
                    className="space-y-2 rounded-lg border border-border p-3"
                  >
                    <p className="text-sm font-medium text-text">{candidate.title}</p>
                    <Radio.Group
                      value={conflictResolutions[candidate.mediaId]}
                      onChange={(event) =>
                        handleResolveConflict(
                          candidate.mediaId,
                          event.target.value as WorkspaceSourceTransferConflictResolution
                        )
                      }
                      className="flex flex-col gap-2"
                    >
                      <Radio value="skip">Skip</Radio>
                      <Radio value="merge-folders">Merge folder memberships</Radio>
                      <Radio value="replace-transferred-folders">
                        Replace transferred folder memberships
                      </Radio>
                    </Radio.Group>
                  </div>
                ))}
              </div>
            ) : null}

            {step === "cleanup" ? (
              <Radio.Group
                value={emptyFolderPolicy}
                onChange={(event) =>
                  setEmptyFolderPolicy(
                    event.target.value as WorkspaceSourceTransferEmptyFolderPolicy
                  )
                }
                className="flex flex-col gap-3"
              >
                <Radio value="keep">Keep empty folders</Radio>
                <Radio value="delete-empty-folders">Delete emptied folders</Radio>
              </Radio.Group>
            ) : null}

            {step === "complete" && resultSummary ? (
              <div className="space-y-3">
                <Alert
                  type="success"
                  showIcon
                  title={t(
                    "playground:sources.transferComplete",
                    "{{count}} sources transferred.",
                    { count: resultSummary.transferredCount }
                  )}
                  description={
                    resultSummary.conflictsSkipped > 0
                      ? t(
                          "playground:sources.transferSkippedConflicts",
                          "{{count}} conflicts were skipped.",
                          { count: resultSummary.conflictsSkipped }
                        )
                      : undefined
                  }
                />
                {!resultSummary.destinationWasCreated ? (
                  <Button onClick={handleOpenDestination}>Open destination</Button>
                ) : null}
              </div>
            ) : null}
          </div>
        )}
      </Modal>
    </>
  )
}

export default TransferSourcesModal
