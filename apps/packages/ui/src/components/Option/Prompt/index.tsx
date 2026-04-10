/**
 * PromptBody — Orchestrator for the /prompts workspace.
 *
 * This is a thin routing layer that:
 * 1. Wraps children in PromptWorkspaceProvider (shared stable state)
 * 2. Routes to segment components based on selectedSegment
 * 3. Handles global keyboard shortcuts (N, /, Esc, ?)
 * 4. Renders mobile segment tabs
 * 5. Renders the sidebar (desktop)
 * 6. Wires PromptWorkspaceModals with callbacks from hooks
 * 7. Handles deep-link URL params (?prompt=, ?edit=, ?new=)
 *
 * Segment rendering is delegated to:
 * - CustomSegment.tsx (filter state + hooks live there)
 * - CopilotSegment.tsx
 * - TrashSegment.tsx
 * - Studio/StudioTabContainer.tsx (already extracted)
 */
import React, { Suspense, useCallback, useEffect, useRef } from "react"
import {
  Alert,
  Segmented,
  Tooltip,
  notification,
  type InputRef
} from "antd"
import { AlertTriangle, Cloud, Trash2, WifiOff } from "lucide-react"
import { usePromptWorkspace, PromptWorkspaceProvider, type SegmentType } from "./PromptWorkspaceProvider"
import { usePromptUrlState } from "./usePromptUrlState"
import { PromptSidebar } from "./PromptSidebar"
import { PromptWorkspaceModals } from "./PromptWorkspaceModals"
import { CustomSegment } from "./CustomSegment"
import { CopilotSegment } from "./CopilotSegment"
import { TrashSegment } from "./TrashSegment"
import { usePromptSync } from "./hooks/usePromptSync"
import { usePromptEditor } from "./hooks/usePromptEditor"
import { usePromptBulkActions } from "./hooks/usePromptBulkActions"
import { usePromptCollections } from "./hooks/usePromptCollections"
import { usePromptInteractions } from "./hooks/usePromptInteractions"
import { pullFromStudio } from "@/services/prompt-sync"
import { getAllPrompts } from "@/db/dexie/helpers"

const StudioTabContainer = React.lazy(() =>
  import("./Studio/StudioTabContainer").then((module) => ({
    default: module.StudioTabContainer
  }))
)

// ---------------------------------------------------------------------------
// Inner orchestrator (lives inside the provider)
// ---------------------------------------------------------------------------

function PromptBodyInner() {
  const {
    queryClient,
    isOnline,
    t,
    isCompactViewport,
    selectedSegment,
    setSelectedSegment,
    data,
    dataStatus: status,
    trashData,
    utils
  } = usePromptWorkspace()

  const {
    confirmDanger,
    guardPrivateMode,
    getPromptTexts,
    getPromptKeywords,
    getPromptRecordById,
    getPromptModifiedAt,
    getPromptUsageCount,
    getPromptLastUsedAt,
    isFireFoxPrivateMode
  } = utils

  const urlState = usePromptUrlState()
  const searchInputRef = useRef<InputRef | null>(null)
  const deepLinkProcessedRef = useRef(false)
  // Holds the selected IDs from CustomSegment when bulk keyword modal opens
  const bulkKeywordTargetIdsRef = useRef<string[]>([])
  // Ref to CustomSegment's bulk selection setter for syncing after shared bulk ops
  const customSegmentSelectionSyncRef = useRef<((ids: React.Key[]) => void) | null>(null)
  // Track latest selectedSegment for async deep-link race guard (#1050)
  const selectedSegmentRef = useRef(selectedSegment)
  selectedSegmentRef.current = selectedSegment

  // ---- Hooks for orchestrator-level concerns ----
  // (Sync and editor are needed here for deep-link handling and modal wiring.
  //  CustomSegment also instantiates its own sync/editor — this is intentional;
  //  the orchestrator's instances are for deep-link + shared modal callbacks only.)

  const sync = usePromptSync({ queryClient, isOnline, t })
  const editor = usePromptEditor({
    queryClient, isOnline, t,
    guardPrivateMode, getPromptTexts, getPromptKeywords, getPromptRecordById,
    confirmDanger,
    syncPromptAfterLocalSave: sync.syncPromptAfterLocalSave,
    onEmptyTrashSuccess: () => {}
  })
  const bulk = usePromptBulkActions({
    queryClient, data, isOnline, isFireFoxPrivateMode, t,
    guardPrivateMode, getPromptKeywords,
    buildPromptUpdatePayload: editor.buildPromptUpdatePayload,
    confirmDanger
  })
  const collections = usePromptCollections({
    queryClient, isOnline, t,
    setSelectedRowKeys: bulk.setSelectedRowKeys
  })
  const interactions = usePromptInteractions({
    queryClient, isOnline,
    initialSegment: urlState.tab,
    t,
    getPromptTexts, getPromptKeywords, getPromptRecordById,
    getPromptModifiedAt, getPromptUsageCount, getPromptLastUsedAt,
    editorMarkPromptAsUsed: editor.markPromptAsUsed
  })

  const {
    openCopilotEdit, setOpenCopilotEdit,
    editCopilotId, setEditCopilotId,
    editCopilotForm,
    copilotSearchText, setCopilotSearchText,
    copilotKeyFilter, setCopilotKeyFilter,
    copilotData, copilotStatus,
    copilotPromptIncludesTextPlaceholder,
    copilotPromptKeyOptions, filteredCopilotData,
    updateCopilotPrompt, isUpdatingCopilotPrompt,
    copyCopilotPromptToClipboard, copyPromptShareLink,
    insertPrompt, setInsertPrompt,
    handleInsertChoice, handleUsePromptInChat,
    localQuickTestPrompt, localQuickTestInput, setLocalQuickTestInput,
    localQuickTestOutput, isRunningLocalQuickTest, localQuickTestRunInfo,
    closeLocalQuickTestModal, handleQuickTest, runLocalQuickTest,
    inspectorOpen, inspectorPrompt,
    closeInspector, openPromptInspector,
    shortcutsHelpOpen, setShortcutsHelpOpen,
    hasStudio
  } = interactions

  // ---- Sync URL ↔ segment state ----
  useEffect(() => {
    if (urlState.tab !== selectedSegment) {
      urlState.setTab(selectedSegment)
    }
  }, [selectedSegment, urlState])

  // Initialize segment from URL on mount
  useEffect(() => {
    if (urlState.tab !== "custom") {
      setSelectedSegment(urlState.tab)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // ---- Offline redirect ----
  useEffect(() => {
    if (!isOnline && (selectedSegment === "copilot" || selectedSegment === "studio")) {
      setSelectedSegment("custom")
    }
  }, [isOnline, selectedSegment, setSelectedSegment])

  // ---- Deep-link: ?prompt= ----
  useEffect(() => {
    const promptId = urlState.prompt
    if (!promptId || deepLinkProcessedRef.current) return
    if (status !== "success" || !Array.isArray(data)) return

    deepLinkProcessedRef.current = true
    const openPromptDrawer = (promptRecord: any) => {
      urlState.clearPromptParam()
      editor.setEditId(promptRecord.id)
      editor.setDrawerOpen(true)
      editor.setDrawerInitialValues({
        id: promptRecord?.id,
        name: promptRecord?.name || promptRecord?.title,
        author: promptRecord?.author,
        details: promptRecord?.details,
        system_prompt: promptRecord?.system_prompt || (promptRecord?.is_system ? promptRecord?.content : undefined),
        user_prompt: promptRecord?.user_prompt || (!promptRecord?.is_system ? promptRecord?.content : undefined),
        keywords: promptRecord?.keywords ?? promptRecord?.tags ?? [],
        serverId: promptRecord?.serverId,
        syncStatus: promptRecord?.syncStatus,
        sourceSystem: promptRecord?.sourceSystem,
        studioProjectId: promptRecord?.studioProjectId,
        lastSyncedAt: promptRecord?.lastSyncedAt,
        fewShotExamples: promptRecord?.fewShotExamples,
        modulesConfig: promptRecord?.modulesConfig,
        promptFormat: promptRecord?.promptFormat ?? "legacy",
        promptSchemaVersion: promptRecord?.promptSchemaVersion ?? null,
        structuredPromptDefinition: promptRecord?.structuredPromptDefinition ?? null,
        changeDescription: promptRecord?.changeDescription,
        versionNumber: promptRecord?.versionNumber
      })
    }

    const localPromptRecord = data.find((p: any) => p.id === promptId)
    if (localPromptRecord) { openPromptDrawer(localPromptRecord); return }

    const parsedServerPromptId = Number(promptId)
    const isServerLink = Number.isInteger(parsedServerPromptId) && parsedServerPromptId > 0 &&
      (urlState.source === "studio" || urlState.source === null)

    const getSharedPromptFailureDescription = (errorMessage?: string) => {
      const normalized = String(errorMessage || "").toLowerCase()
      const isAccessDenied =
        normalized.includes("401") || normalized.includes("403") ||
        normalized.includes("forbidden") || normalized.includes("unauthor") ||
        normalized.includes("not authenticated") || normalized.includes("api key")
      if (isAccessDenied) {
        return t("managePrompts.notification.sharedPromptAccessDeniedDesc", {
          defaultValue: "You don't have permission to open this shared prompt. Check your server login and project access."
        })
      }
      return t("managePrompts.notification.sharedPromptNotFoundDesc", {
        defaultValue: "The shared prompt could not be pulled from the server. It may not exist or you may not have access."
      })
    }

    if (isOnline && isServerLink) {
      urlState.clearPromptParam()
      const segmentAtStart = selectedSegment
      void (async () => {
        const syncResult = await pullFromStudio(parsedServerPromptId)
        if (!syncResult.success) {
          notification.warning({ message: t("managePrompts.notification.promptNotFound", { defaultValue: "Prompt not found" }),
            description: getSharedPromptFailureDescription(syncResult.error) })
          return
        }
        try {
          const refreshed = await queryClient.fetchQuery({ queryKey: ["fetchAllPrompts"], queryFn: getAllPrompts })
          const imported = (Array.isArray(refreshed) ? refreshed : []).find(
            (item: any) => item?.id === syncResult.localId || item?.serverId === parsedServerPromptId
          )
          if (imported) {
            if (selectedSegmentRef.current !== segmentAtStart) {
              notification.info({
                message: t("managePrompts.notification.sharedPromptImported", { defaultValue: "Shared prompt imported" }),
                description: t("managePrompts.notification.sharedPromptSegmentChanged", {
                  defaultValue: "The prompt was imported but you navigated away. Switch back to the Custom tab to view it."
                })
              })
              return
            }
            notification.success({ message: t("managePrompts.notification.sharedPromptImported", { defaultValue: "Shared prompt imported" }) })
            openPromptDrawer(imported)
          }
        } catch { /* handled below */ }
      })()
      return
    }

    urlState.clearPromptParam()
    if (!isOnline && isServerLink) {
      notification.warning({ message: t("managePrompts.notification.promptNotFound", { defaultValue: "Prompt not found" }),
        description: t("managePrompts.notification.sharedPromptOfflineDesc", { defaultValue: "This shared prompt link requires an online server connection." }) })
      return
    }
    notification.warning({ message: t("managePrompts.notification.promptNotFound", { defaultValue: "Prompt not found" }),
      description: t("managePrompts.notification.promptNotFoundDesc", { defaultValue: "The requested prompt could not be found. It may have been deleted." }) })
  }, [urlState.prompt, data, status, isOnline, queryClient, t, editor, urlState])

  // ---- Deep-link: ?edit= / ?new= ----
  useEffect(() => {
    if (status !== "success" || !Array.isArray(data)) return
    if (urlState.edit && !editor.fullEditorOpen) {
      const prompt = data.find((p: any) => String(p.id) === urlState.edit)
      if (prompt) editor.openFullEditor(prompt)
    } else if (urlState.isNew && !editor.fullEditorOpen) {
      editor.openFullEditor()
    }
  }, [status, data, urlState.edit, urlState.isNew, editor.fullEditorOpen, editor.openFullEditor])

  // ---- Error banners ----
  const promptLoadFailed = status === "error"
  const copilotLoadFailed = isOnline && copilotStatus === "error"
  const loadErrorDescription = [
    promptLoadFailed ? t("managePrompts.loadErrorDetail", "Custom prompts couldn't be retrieved from local storage.") : null,
    copilotLoadFailed ? t("managePrompts.copilotLoadErrorDetail", "Copilot prompts couldn't be retrieved.") : null
  ].filter(Boolean).join(" ")

  // ---- Global keyboard shortcuts ----
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement
      const isInput = target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.tagName === "SELECT" || target.isContentEditable
      if (e.key === "Escape") {
        if (shortcutsHelpOpen) { setShortcutsHelpOpen(false); return }
        if (editor.drawerOpen) { editor.setDrawerOpen(false) }
        return
      }
      if (isInput) return
      if ((e.key === "?" || (e.key === "/" && e.shiftKey)) && !e.metaKey && !e.ctrlKey && !e.altKey) {
        e.preventDefault(); setShortcutsHelpOpen(true); return
      }
      if (e.key === "n" && !e.metaKey && !e.ctrlKey && !e.altKey) {
        e.preventDefault(); editor.openFullEditor(); return
      }
      if (e.key === "/" && !e.metaKey && !e.ctrlKey && !e.altKey) {
        e.preventDefault(); searchInputRef.current?.focus()
      }
    }
    document.addEventListener("keydown", handler)
    return () => document.removeEventListener("keydown", handler)
  }, [editor.drawerOpen, shortcutsHelpOpen, editor.openFullEditor, setShortcutsHelpOpen])

  // ---- Copilot helpers ----
  const handleCopyCopilotToCustom = useCallback(
    (record: { key?: string; prompt?: string }) => {
      interactions.copyCopilotToCustom(record, editor.openCreateDrawer)
    },
    [interactions.copyCopilotToCustom, editor.openCreateDrawer]
  )

  // Pending sync count for mobile badge
  const pendingSyncCount = Array.isArray(data) ? data.filter((p: any) => p?.syncStatus === "pending").length : 0

  // ---- Sidebar counts (lightweight — sidebar only needs smart counts for non-custom segments) ----
  // For Custom segment, CustomSegment computes its own full sidebarCounts via usePromptFilteredData.
  // Here we just need trash count for the sidebar badge.

  // ---- Render ----
  return (
    <div>
      {/* Screen reader status announcements */}
      <div role="status" aria-live="polite" aria-atomic="true" className="sr-only" id="prompts-status-announcer" />

      {/* Offline banner */}
      {!isOnline && (
        <Alert type="warning" showIcon icon={<WifiOff className="size-4" />} className="mb-4"
          data-testid="prompts-offline-banner"
          title={t("managePrompts.offline.title", { defaultValue: "You are offline" })}
          description={t("managePrompts.offline.description", {
            defaultValue: "Server features like sync, Copilot prompts, and Prompt Studio are unavailable. Your local prompts are still accessible."
          })} />
      )}

      {/* Firefox Private Mode Warning */}
      {isFireFoxPrivateMode && (
        <Alert type="warning" showIcon icon={<AlertTriangle className="size-4" />} className="mb-4"
          title={t("managePrompts.privateMode.title", { defaultValue: "Limited functionality in Private Mode" })}
          description={t("managePrompts.privateMode.description", {
            defaultValue: "Firefox Private Mode doesn't support IndexedDB. You can view existing prompts, but creating, editing, or importing prompts is disabled. Use a normal window for full functionality."
          })} />
      )}
      {(promptLoadFailed || copilotLoadFailed) && (
        <Alert type="error" showIcon className="mb-4"
          title={t("managePrompts.partialLoad", "Some prompt data isn't available")}
          description={loadErrorDescription || t("managePrompts.loadErrorHelp", "Check your server connection and refresh to try again.")} />
      )}

      <div className="flex gap-0">
        {/* Sidebar - desktop only */}
        {!isCompactViewport && (
          <PromptSidebar
            collapsed={false}
            onToggleCollapsed={() => {}}
            selectedSegment={selectedSegment}
            onSegmentChange={(s) => setSelectedSegment(s as SegmentType)}
            trashCount={trashData?.length}
            savedView="all"
            onSavedViewChange={() => {}}
            smartCounts={{ all: data?.length ?? 0, favorites: 0, recent: 0, most_used: 0, untagged: 0 }}
            typeFilter="all"
            onTypeFilterChange={() => {}}
            typeCounts={{ system: 0, quick: 0, mixed: 0 }}
            syncFilter="all"
            onSyncFilterChange={() => {}}
            syncCounts={{ local: 0, pending: 0, synced: 0, conflict: 0 }}
            tagFilter={[]}
            onTagFilterChange={() => {}}
            tagMatchMode="any"
            onTagMatchModeChange={() => {}}
            tagCounts={{}}
            presets={[]}
            onLoadPreset={() => {}}
            onSavePreset={() => {}}
            onDeletePreset={() => {}}
          />
        )}

        {/* Main content area */}
        <div className="flex-1 min-w-0">
          {/* Mobile segment tabs */}
          {isCompactViewport && (
            <div className="flex flex-col items-start gap-1 mb-6 px-4">
              <Segmented
                size="large"
                options={[
                  {
                    label: (
                      <span className="flex items-center gap-1">
                        {t("managePrompts.segmented.custom", { defaultValue: "Custom prompts" })}
                        {pendingSyncCount > 0 && (
                          <Tooltip title={t("managePrompts.sync.pendingCountTooltip", { defaultValue: "{{count}} prompt(s) have local changes pending sync.", count: pendingSyncCount })}>
                            <span data-testid="prompts-pending-sync-count" className="text-xs bg-warn/20 text-warn px-1.5 py-0.5 rounded-full">
                              {pendingSyncCount}
                            </span>
                          </Tooltip>
                        )}
                      </span>
                    ),
                    value: "custom"
                  },
                  {
                    label: (
                      <Tooltip title={t("managePrompts.segmented.copilotTooltip", { defaultValue: "Predefined prompts from your tldw server that help with common tasks" })}>
                        <span>{t("managePrompts.segmented.copilot", { defaultValue: "Copilot prompts" })}</span>
                      </Tooltip>
                    ),
                    value: "copilot",
                    disabled: !isOnline
                  },
                  {
                    label: (
                      <Tooltip title={t("managePrompts.segmented.studioTooltip", { defaultValue: "Browse and import prompts from Prompt Studio projects on the server" })}>
                        <span className="flex items-center gap-1">
                          <Cloud className="size-3" />
                          {t("managePrompts.segmented.studio", { defaultValue: "Studio" })}
                        </span>
                      </Tooltip>
                    ),
                    value: "studio",
                    disabled: !isOnline || hasStudio === false
                  },
                  {
                    label: (
                      <span className="flex items-center gap-1">
                        <Trash2 className="size-3" />
                        {t("managePrompts.segmented.trash", { defaultValue: "Trash" })}
                        {(trashData?.length || 0) > 0 && (
                          <span className="text-xs bg-text-muted/20 px-1.5 py-0.5 rounded-full">{trashData?.length}</span>
                        )}
                      </span>
                    ),
                    value: "trash"
                  }
                ]}
                data-testid="prompts-segmented"
                value={selectedSegment}
                onChange={(value) => setSelectedSegment(value as SegmentType)}
              />
              <p className="text-xs text-text-muted">
                {selectedSegment === "custom"
                  ? t("managePrompts.segmented.helpCustom", { defaultValue: "Create and manage reusable prompts you can insert into chat." })
                  : selectedSegment === "copilot"
                    ? t("managePrompts.segmented.helpCopilot", { defaultValue: "View and tweak predefined Copilot prompts provided by your server." })
                    : selectedSegment === "studio"
                      ? t("managePrompts.segmented.helpStudio", { defaultValue: "Full Prompt Studio: manage projects, prompts, test cases, evaluations, and optimizations." })
                      : t("managePrompts.segmented.helpTrash", { defaultValue: "Restore or permanently delete prompts. Items auto-delete after 30 days." })}
              </p>
            </div>
          )}

          <div className={isCompactViewport ? "px-4" : "p-4"}>
            {selectedSegment === "custom" && (
              <CustomSegment
                projectFilter={urlState.project}
                clearProjectFilter={urlState.clearProjectFilter}
                onOpenShortcutsHelp={() => setShortcutsHelpOpen(true)}
                onQuickTest={handleQuickTest}
                onUsePromptInChat={handleUsePromptInChat}
                onOpenInspector={openPromptInspector}
                closeInspector={closeInspector}
                onOpenBulkKeywordModal={(selectedIds) => {
                  bulkKeywordTargetIdsRef.current = selectedIds
                  bulk.setBulkKeywordModalOpen(true)
                }}
                bulkSelectionSyncRef={customSegmentSelectionSyncRef}
                searchInputRef={searchInputRef}
              />
            )}
            {selectedSegment === "copilot" && (
              <CopilotSegment
                tableDensity="comfortable"
                copilotSearchText={copilotSearchText}
                setCopilotSearchText={setCopilotSearchText}
                copilotKeyFilter={copilotKeyFilter}
                setCopilotKeyFilter={setCopilotKeyFilter}
                copilotData={copilotData}
                copilotStatus={copilotStatus}
                copilotPromptKeyOptions={copilotPromptKeyOptions}
                filteredCopilotData={filteredCopilotData}
                onOpenCopilotEdit={(key, record) => {
                  setEditCopilotId(key)
                  editCopilotForm.setFieldsValue(record)
                  setOpenCopilotEdit(true)
                }}
                onCopyCopilotToCustom={handleCopyCopilotToCustom}
                onCopyCopilotToClipboard={copyCopilotPromptToClipboard}
              />
            )}
            {selectedSegment === "studio" && (
              <Suspense fallback={null}>
                <StudioTabContainer />
              </Suspense>
            )}
            {selectedSegment === "trash" && (
              <TrashSegment tableDensity="comfortable" />
            )}
          </div>
        </div>
      </div>

      {/* Shared modals */}
      <PromptWorkspaceModals
        // Copilot edit
        copilotEditOpen={openCopilotEdit}
        onCopilotEditCancel={() => setOpenCopilotEdit(false)}
        copilotEditForm={editCopilotForm}
        copilotEditId={editCopilotId}
        copilotPromptIncludesTextPlaceholder={copilotPromptIncludesTextPlaceholder}
        onCopilotEditSubmit={(values) => updateCopilotPrompt({ key: values.key, prompt: values.prompt })}
        isUpdatingCopilotPrompt={isUpdatingCopilotPrompt}
        // Bulk keyword
        bulkKeywordModalOpen={bulk.bulkKeywordModalOpen}
        onBulkKeywordCancel={() => { bulk.setBulkKeywordModalOpen(false); bulk.setBulkKeywordValue("") }}
        bulkKeywordValue={bulk.bulkKeywordValue}
        onBulkKeywordValueChange={(v) => bulk.setBulkKeywordValue(v)}
        onBulkKeywordSubmit={() => {
          const ids = bulkKeywordTargetIdsRef.current
          bulk.bulkAddKeyword({ ids, keyword: bulk.bulkKeywordValue }, {
            onSuccess: (result: any) => {
              // Sync selection back to CustomSegment so table reflects failed-only rows
              if (result?.failedIds && customSegmentSelectionSyncRef.current) {
                customSegmentSelectionSyncRef.current(result.failedIds)
              }
            }
          })
        }}
        isBulkAddingKeyword={bulk.isBulkAddingKeyword}
        // Quick test
        quickTestPrompt={localQuickTestPrompt}
        onQuickTestClose={closeLocalQuickTestModal}
        quickTestInput={localQuickTestInput}
        onQuickTestInputChange={setLocalQuickTestInput}
        quickTestOutput={localQuickTestOutput}
        isRunningQuickTest={isRunningLocalQuickTest}
        quickTestRunInfo={localQuickTestRunInfo}
        onRunQuickTest={runLocalQuickTest}
        // Collection create
        collectionModalOpen={collections.createCollectionModalOpen}
        onCollectionModalCancel={() => collections.setCreateCollectionModalOpen(false)}
        collectionName={collections.newCollectionName}
        onCollectionNameChange={collections.setNewCollectionName}
        collectionDescription={collections.newCollectionDescription}
        onCollectionDescriptionChange={collections.setNewCollectionDescription}
        onCollectionCreate={() => collections.createPromptCollectionMutation({ name: collections.newCollectionName, description: collections.newCollectionDescription })}
        isCreatingCollection={collections.isCreatingPromptCollection}
        // Shortcuts
        shortcutsOpen={shortcutsHelpOpen}
        onShortcutsClose={() => setShortcutsHelpOpen(false)}
        // Insert prompt
        insertPrompt={insertPrompt}
        onInsertCancel={() => setInsertPrompt(null)}
        onInsertChoice={handleInsertChoice}
        // Project selector
        projectSelectorOpen={sync.projectSelectorOpen}
        onProjectSelectorClose={() => { sync.setProjectSelectorOpen(false); sync.setPromptToSync(null) }}
        onProjectSelect={(projectId) => { if (sync.promptToSync) sync.pushToStudioMutation({ localId: sync.promptToSync, projectId }) }}
        isPushing={sync.isPushing}
        // Conflict resolution
        conflictModalOpen={sync.conflictModalOpen}
        conflictLoading={sync.isLoadingConflictInfo || sync.isResolvingConflict}
        conflictInfo={sync.conflictInfo}
        onConflictClose={sync.closeConflictResolution}
        onConflictResolve={sync.handleResolveConflict}
        // Drawer
        drawerOpen={editor.drawerOpen}
        onDrawerClose={() => { editor.setDrawerOpen(false); editor.setDrawerInitialValues(null) }}
        drawerMode={editor.drawerMode}
        drawerInitialValues={editor.drawerInitialValues}
        onDrawerSubmit={editor.handleDrawerSubmit}
        drawerLoading={editor.drawerMode === "create" ? editor.savePromptLoading : editor.isUpdatingPrompt}
        allTags={[]}
        // Full page editor
        fullEditorOpen={editor.fullEditorOpen}
        onFullEditorClose={editor.closeFullEditor}
        fullEditorMode={editor.fullEditorMode}
        fullEditorInitialValues={editor.fullEditorInitialValues}
        onFullEditorSubmit={editor.handleFullEditorSubmit}
        fullEditorLoading={editor.fullEditorMode === "create" ? editor.savePromptLoading : editor.isUpdatingPrompt}
        // Inspector panel
        inspectorOpen={inspectorOpen}
        inspectorPrompt={inspectorPrompt}
        onInspectorClose={closeInspector}
        onInspectorEdit={(promptId) => {
          const rec = getPromptRecordById(promptId); if (!rec) return
          closeInspector(); editor.openFullEditor(rec)
        }}
        onInspectorUseInChat={(promptId) => {
          const rec = getPromptRecordById(promptId); if (!rec) return
          closeInspector(); void handleUsePromptInChat(rec)
        }}
        onInspectorDuplicate={(promptId) => {
          const rec = getPromptRecordById(promptId); if (!rec) return
          editor.handleDuplicatePrompt(rec)
        }}
        onInspectorDelete={(promptId) => {
          const rec = getPromptRecordById(promptId); if (!rec) return
          closeInspector(); void editor.handleDeletePrompt(rec)
        }}
      />
    </div>
  )
}

// ---------------------------------------------------------------------------
// Exported component (wraps inner in provider)
// ---------------------------------------------------------------------------

export const PromptBody = () => (
  <PromptWorkspaceProvider>
    <PromptBodyInner />
  </PromptWorkspaceProvider>
)
