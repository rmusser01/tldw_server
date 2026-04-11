import React, { useRef, useState } from "react"
import { useMutation, type QueryClient } from "@tanstack/react-query"
import { notification } from "antd"
import {
  autoSyncPrompt,
  pushToStudio,
  pullFromStudio,
  shouldAutoSyncWorkspacePrompts,
  unlinkPrompt as unlinkPromptFromServer,
  getConflictInfo,
  resolveConflict,
  getAllPromptsWithSyncStatus,
  type ConflictInfo,
  type ConflictResolution
} from "@/services/prompt-sync"
import {
  buildSyncBatchPlan,
  type SyncBatchTask
} from "../sync-batch-utils"

type BatchSyncFailure = {
  task: SyncBatchTask
  error: string
}

type BatchSyncState = {
  running: boolean
  completed: number
  total: number
  succeeded: number
  failed: BatchSyncFailure[]
  skippedConflicts: number
  skippedCopilotPending: number
  cancelled: boolean
}

const INITIAL_BATCH_SYNC_STATE: BatchSyncState = {
  running: false,
  completed: 0,
  total: 0,
  succeeded: 0,
  failed: [],
  skippedConflicts: 0,
  skippedCopilotPending: 0,
  cancelled: false
}

export interface UsePromptSyncDeps {
  queryClient: QueryClient
  isOnline: boolean
  t: (key: string, opts?: Record<string, any>) => string
}

export function usePromptSync(deps: UsePromptSyncDeps) {
  const { queryClient, isOnline, t } = deps

  const [projectSelectorOpen, setProjectSelectorOpen] = useState(false)
  const [promptToSync, setPromptToSync] = useState<string | null>(null)
  const [conflictModalOpen, setConflictModalOpen] = useState(false)
  const [conflictPromptId, setConflictPromptId] = useState<string | null>(null)
  const [conflictInfo, setConflictInfo] = useState<ConflictInfo | null>(null)
  const [batchSyncState, setBatchSyncState] = useState<BatchSyncState>(
    INITIAL_BATCH_SYNC_STATE
  )
  const batchSyncCancelRef = useRef(false)

  const syncPromptAfterLocalSave = React.useCallback(async (localId: string) => {
    try {
      const autoSyncEnabled = await shouldAutoSyncWorkspacePrompts()
      if (!autoSyncEnabled) {
        return {
          attempted: false,
          success: true,
          error: undefined
        }
      }

      const result = await autoSyncPrompt(localId)
      if (!result.success) {
        notification.warning({
          message: t("managePrompts.sync.syncFailed", {
            defaultValue: "Sync failed"
          }),
          description: t("managePrompts.sync.syncFailedWithLocalSave", {
            defaultValue: "{{error}} Your changes are saved locally.",
            error: result.error || t("managePrompts.sync.pendingTooltip", {
              defaultValue: "Local changes not yet synced."
            })
          })
        })
      }
      return {
        attempted: true,
        success: result.success,
        error: result.error
      }
    } catch (error: unknown) {
      const fallbackError =
        error instanceof Error
          ? error.message
          : t("managePrompts.sync.pendingTooltip", {
              defaultValue: "Local changes not yet synced"
            })
      notification.warning({
        message: t("managePrompts.sync.syncFailed", {
          defaultValue: "Sync failed"
        }),
        description: t("managePrompts.sync.syncFailedWithLocalSave", {
          defaultValue: "{{error}} Your changes are saved locally.",
          error: fallbackError
        })
      })
      return {
        attempted: true,
        success: false,
        error: fallbackError
      }
    }
  }, [t])

  const { mutate: pushToStudioMutation, isPending: isPushing } = useMutation({
    mutationFn: async ({ localId, projectId }: { localId: string; projectId: number }) => {
      return await pushToStudio(localId, projectId)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["fetchAllPrompts"] })
      setProjectSelectorOpen(false)
      setPromptToSync(null)
      notification.success({
        message: t("managePrompts.sync.pushSuccess", { defaultValue: "Pushed to server" }),
        description: t("managePrompts.sync.pushSuccessDesc", { defaultValue: "Prompt has been synced to Prompt Studio." })
      })
    },
    onError: (error) => {
      notification.error({
        message: t("managePrompts.sync.pushError", { defaultValue: "Failed to push" }),
        description: error?.message || t("managePrompts.notification.someError")
      })
    }
  })

  const { mutate: pullFromStudioMutation, isPending: isPulling } = useMutation({
    mutationFn: async ({ serverId, localId }: { serverId: number; localId?: string }) => {
      return await pullFromStudio(serverId, localId)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["fetchAllPrompts"] })
      notification.success({
        message: t("managePrompts.sync.pullSuccess", { defaultValue: "Pulled from server" }),
        description: t("managePrompts.sync.pullSuccessDesc", { defaultValue: "Prompt has been updated from Prompt Studio." })
      })
    },
    onError: (error) => {
      notification.error({
        message: t("managePrompts.sync.pullError", { defaultValue: "Failed to pull" }),
        description: error?.message || t("managePrompts.notification.someError")
      })
    }
  })

  const { mutate: unlinkPromptMutation } = useMutation({
    mutationFn: unlinkPromptFromServer,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["fetchAllPrompts"] })
      notification.success({
        message: t("managePrompts.sync.unlinkSuccess", { defaultValue: "Unlinked from server" }),
        description: t("managePrompts.sync.unlinkSuccessDesc", { defaultValue: "Prompt is now local-only." })
      })
    },
    onError: (error) => {
      notification.error({
        message: t("managePrompts.sync.unlinkError", { defaultValue: "Failed to unlink" }),
        description: error?.message || t("managePrompts.notification.someError")
      })
    }
  })

  const {
    mutate: loadConflictInfoMutation,
    isPending: isLoadingConflictInfo
  } = useMutation({
    mutationFn: async (localId: string) => {
      return await getConflictInfo(localId)
    },
    onSuccess: (info) => {
      if (!info) {
        notification.warning({
          message: t("managePrompts.sync.conflictUnavailable", {
            defaultValue: "Conflict details unavailable"
          }),
          description: t("managePrompts.sync.conflictUnavailableDesc", {
            defaultValue:
              "We couldn't retrieve local and server versions for comparison."
          })
        })
        setConflictModalOpen(false)
        setConflictPromptId(null)
        setConflictInfo(null)
        return
      }
      setConflictInfo(info)
    },
    onError: (error) => {
      notification.error({
        message: t("managePrompts.sync.pullError", {
          defaultValue: "Failed to load conflict details"
        }),
        description: error?.message || t("managePrompts.notification.someError")
      })
      setConflictModalOpen(false)
      setConflictPromptId(null)
      setConflictInfo(null)
    }
  })

  const {
    mutate: resolveConflictMutation,
    isPending: isResolvingConflict
  } = useMutation({
    mutationFn: async ({
      localId,
      resolution
    }: {
      localId: string
      resolution: ConflictResolution
    }) => {
      return await resolveConflict(localId, resolution)
    },
    onSuccess: (result, variables) => {
      if (!result.success) {
        notification.error({
          message: t("managePrompts.sync.resolveError", {
            defaultValue: "Failed to resolve conflict"
          }),
          description: result.error || t("managePrompts.notification.someError")
        })
        return
      }

      queryClient.invalidateQueries({ queryKey: ["fetchAllPrompts"] })
      setConflictModalOpen(false)
      setConflictPromptId(null)
      setConflictInfo(null)

      const description =
        variables.resolution === "keep_local"
          ? t("managePrompts.sync.keepMineSuccessDesc", {
              defaultValue: "Your local prompt has been pushed to the server."
            })
          : variables.resolution === "keep_server"
            ? t("managePrompts.sync.keepServerSuccessDesc", {
                defaultValue:
                  "Your local prompt has been replaced with the server version."
              })
            : t("managePrompts.sync.keepBothSuccessDesc", {
                defaultValue:
                  "The prompt was unlinked and resynced so both versions are preserved."
              })

      notification.success({
        message: t("managePrompts.sync.resolveSuccess", {
          defaultValue: "Conflict resolved"
        }),
        description
      })
    },
    onError: (error) => {
      notification.error({
        message: t("managePrompts.sync.resolveError", {
          defaultValue: "Failed to resolve conflict"
        }),
        description: error?.message || t("managePrompts.notification.someError")
      })
    }
  })

  const { mutate: importFromStudioMutation, isPending: isImporting } = useMutation({
    mutationFn: async ({ serverId }: { serverId: number }) => {
      return await pullFromStudio(serverId)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["fetchAllPrompts"] })
      notification.success({
        message: t("managePrompts.studio.importSuccess", { defaultValue: "Prompt imported" }),
        description: t("managePrompts.studio.importSuccessDesc", { defaultValue: "The prompt has been saved to your local prompts." })
      })
    },
    onError: (error) => {
      notification.error({
        message: t("managePrompts.studio.importError", { defaultValue: "Failed to import" }),
        description: error?.message || t("managePrompts.notification.someError")
      })
    }
  })

  const openConflictResolution = React.useCallback((localId: string) => {
    setConflictPromptId(localId)
    setConflictInfo(null)
    setConflictModalOpen(true)
    loadConflictInfoMutation(localId)
  }, [loadConflictInfoMutation])

  const closeConflictResolution = React.useCallback(() => {
    setConflictModalOpen(false)
    setConflictPromptId(null)
    setConflictInfo(null)
  }, [])

  const handleResolveConflict = React.useCallback((resolution: ConflictResolution) => {
    if (!conflictPromptId) return
    resolveConflictMutation({ localId: conflictPromptId, resolution })
  }, [conflictPromptId, resolveConflictMutation])

  const cancelBatchSync = React.useCallback(() => {
    batchSyncCancelRef.current = true
  }, [])

  const runBatchSync = React.useCallback(
    async (retryTasks?: SyncBatchTask[]) => {
      if (!isOnline) return

      const plan = retryTasks
        ? {
            tasks: retryTasks,
            skippedConflicts: 0,
            skippedCopilotPending: 0
          }
        : buildSyncBatchPlan(await getAllPromptsWithSyncStatus())

      if (plan.tasks.length === 0) {
        const description = plan.skippedConflicts > 0
          ? t("managePrompts.sync.batchNoActionableWithConflicts", {
              defaultValue:
                "No prompts are ready for batch sync. {{count}} prompt(s) require manual conflict resolution.",
              count: plan.skippedConflicts
            })
          : t("managePrompts.sync.batchNoActionable", {
              defaultValue: "No prompts currently need syncing."
            })
        notification.info({
          message: t("managePrompts.sync.batchNothingToSync", {
            defaultValue: "Nothing to sync"
          }),
          description
        })
        return
      }

      batchSyncCancelRef.current = false
      setBatchSyncState({
        running: true,
        completed: 0,
        total: plan.tasks.length,
        succeeded: 0,
        failed: [],
        skippedConflicts: plan.skippedConflicts,
        skippedCopilotPending: plan.skippedCopilotPending,
        cancelled: false
      })

      let completed = 0
      let succeeded = 0
      const failed: BatchSyncFailure[] = []

      for (const task of plan.tasks) {
        if (batchSyncCancelRef.current) {
          setBatchSyncState({
            running: false,
            completed,
            total: plan.tasks.length,
            succeeded,
            failed: [...failed],
            skippedConflicts: plan.skippedConflicts,
            skippedCopilotPending: plan.skippedCopilotPending,
            cancelled: true
          })
          notification.warning({
            message: t("managePrompts.sync.batchCancelled", {
              defaultValue: "Batch sync cancelled"
            }),
            description: t("managePrompts.sync.batchCancelledDesc", {
              defaultValue:
                "Synced {{completed}} of {{total}} prompts before cancellation.",
              completed,
              total: plan.tasks.length
            })
          })
          await queryClient.invalidateQueries({ queryKey: ["fetchAllPrompts"] })
          return
        }

        try {
          const result =
            task.direction === "pull"
              ? await pullFromStudio(task.serverId!, task.promptId)
              : task.serverId
                ? await pushToStudio(task.promptId, task.preferredProjectId || 1)
                : await autoSyncPrompt(task.promptId, task.preferredProjectId)

          if (result.success) {
            succeeded += 1
          } else {
            failed.push({
              task,
              error:
                result.error ||
                t("managePrompts.notification.someError", {
                  defaultValue: "Something went wrong."
                })
            })
          }
        } catch (error: any) {
          failed.push({
            task,
            error:
              error?.message ||
              t("managePrompts.notification.someError", {
                defaultValue: "Something went wrong."
              })
          })
        }

        completed += 1
        setBatchSyncState((prev) => ({
          ...prev,
          completed,
          succeeded,
          failed: [...failed]
        }))
      }

      await queryClient.invalidateQueries({ queryKey: ["fetchAllPrompts"] })

      setBatchSyncState({
        running: false,
        completed,
        total: plan.tasks.length,
        succeeded,
        failed: [...failed],
        skippedConflicts: plan.skippedConflicts,
        skippedCopilotPending: plan.skippedCopilotPending,
        cancelled: false
      })

      if (failed.length > 0) {
        notification.warning({
          message: t("managePrompts.sync.batchPartialFailure", {
            defaultValue: "Sync completed with issues"
          }),
          description: t("managePrompts.sync.batchPartialFailureDesc", {
            defaultValue:
              "Synced {{succeeded}} of {{total}} prompts. {{failed}} failed.",
            succeeded,
            total: plan.tasks.length,
            failed: failed.length
          })
        })
      } else {
        const extra = plan.skippedConflicts > 0
          ? t("managePrompts.sync.batchConflictReminder", {
              defaultValue:
                " {{count}} conflict prompt(s) still need manual resolution.",
              count: plan.skippedConflicts
            })
          : ""
        notification.success({
          message: t("managePrompts.sync.batchSuccess", {
            defaultValue: "Batch sync complete"
          }),
          description: `${t("managePrompts.sync.batchSuccessDesc", {
            defaultValue: "Synced {{count}} prompt(s).",
            count: succeeded
          })}${extra}`
        })
      }
    },
    [isOnline, queryClient, t]
  )

  const handleBatchSyncAction = React.useCallback(() => {
    if (batchSyncState.running) {
      cancelBatchSync()
      return
    }
    if (batchSyncState.failed.length > 0) {
      void runBatchSync(batchSyncState.failed.map((item) => item.task))
      return
    }
    void runBatchSync()
  }, [batchSyncState.failed, batchSyncState.running, cancelBatchSync, runBatchSync])

  return {
    // state
    projectSelectorOpen,
    setProjectSelectorOpen,
    promptToSync,
    setPromptToSync,
    conflictModalOpen,
    conflictPromptId,
    conflictInfo,
    batchSyncState,
    // callbacks
    syncPromptAfterLocalSave,
    openConflictResolution,
    closeConflictResolution,
    handleResolveConflict,
    handleBatchSyncAction,
    runBatchSync,
    cancelBatchSync,
    // mutations
    pushToStudioMutation,
    isPushing,
    pullFromStudioMutation,
    isPulling,
    unlinkPromptMutation,
    importFromStudioMutation,
    isImporting,
    isLoadingConflictInfo,
    isResolvingConflict
  }
}
