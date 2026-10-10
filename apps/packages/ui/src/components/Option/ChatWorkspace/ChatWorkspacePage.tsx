import React from "react"
import { RotateCw } from "lucide-react"

import { useConnectionState } from "@/hooks/useConnectionState"
import { useChatSurfaceCoordinatorStore } from "@/store/chat-surface-coordinator"
import { useWorkspaceStore } from "@/store/workspace"
import {
  WORKSPACE_STORAGE_KEY,
  WORKSPACE_STORAGE_SPLIT_KEY_PREFIX,
  collectResearchWorkspaceLegacyLocalStorageKeys,
  markFreshWorkspaceInitializationForRuntime
} from "@/store/workspace-events"
import { ConnectionPhase } from "@/types/connection"

import { ChatWorkspaceConsole } from "./ChatWorkspaceConsole"
import { ActivatedLocalWorkspace } from "../ResearchWorkspace/ResearchWorkspaceRouteGate"
import { stageWorkspaceSources, unstageWorkspaceSource } from "./staging"
import type { ChatWorkspaceRuntimeState, StagedWorkspaceSource } from "./types"
import { normalizeWorkspaceId } from "./workspaceIdentity"
import type { EffectiveWorkspaceAssistantDefault } from "@/types/workspace"

type WorkspaceContextState = {
  workspaceId: string | null
  browsedSourceId: string | null
  stagedSources: StagedWorkspaceSource[]
}

const createInitialRuntimeState = (
  backendAvailable: boolean,
  effectiveAssistantDefault?: EffectiveWorkspaceAssistantDefault | null
): ChatWorkspaceRuntimeState => ({
  backendAvailable,
  streaming: false,
  sending: false,
  historyLoading: false,
  historyLoadError: null,
  sendError: null,
  selectedModelLabel: "No model selected",
  hasModelSelected: false,
  selectedPersonaLabel: null,
  assistantSource:
    effectiveAssistantDefault?.status === "unavailable"
      ? "unavailable"
      : "none",
  workspaceAssistantDegradedReason:
    effectiveAssistantDefault?.status === "unavailable"
      ? effectiveAssistantDefault.degradedReason
      : null
})

export const ChatWorkspacePage = () => {
  const rawWorkspaceId = useWorkspaceStore((state) => state.workspaceId)
  const storeHydrated = useWorkspaceStore((state) => state.storeHydrated)
  const storeHydrationError = useWorkspaceStore((state) => state.storeHydrationError)
  const serverWorkspace = useWorkspaceStore((state) => state.serverWorkspace)
  const workspaceName = useWorkspaceStore((state) => state.workspaceName)
  const sources = useWorkspaceStore((state) => state.sources)
  const effectiveAssistantDefault = useWorkspaceStore(
    (state) => state.effectiveAssistantDefault
  )
  const sourcesLoading = useWorkspaceStore((state) => state.sourcesLoading)
  const sourcesError = useWorkspaceStore((state) => state.sourcesError)
  const focusSourceById = useWorkspaceStore((state) => state.focusSourceById)
  const setRouteContext = useChatSurfaceCoordinatorStore(
    (state) => state.setRouteContext
  )
  const connectionState = useConnectionState()
  const connectionMode: ChatWorkspaceRuntimeState["connectionMode"] =
    connectionState.mode === "demo"
      ? "demo"
      : connectionState.offlineBypass
        ? "bypass"
        : "live"
  const backendAvailable =
    connectionMode === "live" &&
    connectionState.isConnected &&
    connectionState.phase === ConnectionPhase.CONNECTED
  const workspaceId = React.useMemo(
    () => normalizeWorkspaceId(rawWorkspaceId),
    [rawWorkspaceId]
  )
  const workspaceReady = storeHydrated && workspaceId !== null
  const [hadPersistedWorkspaceContent] = React.useState(() =>
    collectResearchWorkspaceLegacyLocalStorageKeys().some(
      (key) =>
        key === WORKSPACE_STORAGE_KEY ||
        key.startsWith(WORKSPACE_STORAGE_SPLIT_KEY_PREFIX)
    )
  )

  React.useEffect(() => {
    if (!storeHydrated || workspaceId) return
    let cancelled = false
    void Promise.resolve().then(() => {
      const current = useWorkspaceStore.getState()
      if (
        cancelled ||
        !current.storeHydrated ||
        normalizeWorkspaceId(current.workspaceId)
      )
        return
      const initializedWorkspaceId = current.initializeWorkspace()
      if (!hadPersistedWorkspaceContent) {
        markFreshWorkspaceInitializationForRuntime(initializedWorkspaceId)
      }
    })
    return () => {
      cancelled = true
    }
  }, [hadPersistedWorkspaceContent, storeHydrated, workspaceId])

  const [workspaceContext, setWorkspaceContext] =
    React.useState<WorkspaceContextState>(() => ({
      workspaceId,
      browsedSourceId: null,
      stagedSources: []
    }))
  const [runtimeState, setRuntimeState] =
    React.useState<ChatWorkspaceRuntimeState>(() =>
      createInitialRuntimeState(backendAvailable, effectiveAssistantDefault)
    )

  const scopeLabel = workspaceName || "Workspace"
  const contextMatchesWorkspace = workspaceContext.workspaceId === workspaceId
  const browsedSourceId = contextMatchesWorkspace
    ? workspaceContext.browsedSourceId
    : null
  const stagedSources = contextMatchesWorkspace
    ? workspaceContext.stagedSources
    : []
  React.useEffect(() => {
    setRouteContext({
      routeId: "chat-workspace",
      surface: "webui"
    })
  }, [setRouteContext])

  React.useEffect(() => {
    if (!contextMatchesWorkspace) {
      setWorkspaceContext({
        workspaceId,
        browsedSourceId: null,
        stagedSources: []
      })
    }
  }, [contextMatchesWorkspace, workspaceId])

  const handleBrowseSource = React.useCallback(
    (sourceId: string) => {
      const current = useWorkspaceStore.getState()
      if (
        !workspaceReady ||
        !current.storeHydrated ||
        normalizeWorkspaceId(current.workspaceId) !== workspaceId ||
        !current.sources.some((source) => source.id === sourceId)
      )
        return
      focusSourceById(sourceId)
      setWorkspaceContext((current) => ({
        workspaceId,
        browsedSourceId: sourceId,
        stagedSources:
          current.workspaceId === workspaceId ? current.stagedSources : []
      }))
    },
    [focusSourceById, workspaceId, workspaceReady]
  )

  const handleCloseBrowseSource = React.useCallback(() => {
    setWorkspaceContext((current) =>
      current.workspaceId === workspaceId &&
      current.browsedSourceId === browsedSourceId
        ? { ...current, browsedSourceId: null }
        : current
    )
  }, [browsedSourceId, workspaceId])

  const handleStageSources = React.useCallback(
    (sourceIds: string[]) => {
      const selected = sources.filter((source) => sourceIds.includes(source.id))
      setWorkspaceContext((current) => ({
        workspaceId,
        browsedSourceId:
          current.workspaceId === workspaceId ? current.browsedSourceId : null,
        stagedSources: stageWorkspaceSources(
          current.workspaceId === workspaceId ? current.stagedSources : [],
          selected,
          scopeLabel
        )
      }))
    },
    [scopeLabel, sources, workspaceId]
  )

  const handleClearStagedSources = React.useCallback(() => {
    setWorkspaceContext((current) =>
      current.workspaceId === workspaceId
        ? {
            workspaceId,
            browsedSourceId: current.browsedSourceId,
            stagedSources: []
          }
        : current
    )
  }, [workspaceId])

  const handleUnstageSource = React.useCallback(
    (sourceId: string) => {
      setWorkspaceContext((current) =>
        current.workspaceId === workspaceId
          ? {
              workspaceId,
              browsedSourceId: current.browsedSourceId,
              stagedSources: unstageWorkspaceSource(
                current.stagedSources,
                sourceId
              )
            }
          : current
      )
    },
    [workspaceId]
  )

  const handleRuntimeStateChange = React.useCallback(
    (state: ChatWorkspaceRuntimeState) => {
      setRuntimeState((current) => ({ ...current, ...state }))
    },
    []
  )

  if (!storeHydrated && storeHydrationError) {
    return (
      <div data-testid="chat-workspace-page" className="h-full min-w-0 p-4 text-text">
        <h1 className="sr-only">Chat Workspace</h1>
        <div role="alert" className="mb-3 text-sm">
          <h2 className="mb-2 font-semibold">Workspace recovery needed</h2>
          <p>{storeHydrationError}</p>
        </div>
        <button
          type="button"
          onClick={() => { void useWorkspaceStore.persist.rehydrate() }}
          className="inline-flex min-h-11 items-center gap-2 rounded-md border border-border bg-surface px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus">
          <RotateCw size={16} aria-hidden="true" />
          Retry workspace recovery
        </button>
      </div>
    )
  }

  if (workspaceReady && serverWorkspace && serverWorkspace.metadata.id !== workspaceId) {
    return (
      <div data-testid="chat-workspace-page" className="h-full min-w-0 p-4 text-text">
        <h1 className="sr-only">Chat Workspace</h1>
        <h2 className="mb-2 text-sm font-semibold">Workspace not connected</h2>
        <a href="/workspaces" className="inline-flex min-h-11 items-center rounded-md border border-border px-3 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus">
          Open workspaces
        </a>
      </div>
    )
  }

  const console = (
      <ChatWorkspaceConsole
        workspaceId={workspaceId}
        workspaceReady={workspaceReady}
        workspaceName={scopeLabel}
        sources={sources}
        sourcesLoading={sourcesLoading}
        sourcesError={sourcesError}
        browsedSourceId={browsedSourceId}
        stagedSources={stagedSources}
        selectedModelLabel={runtimeState.selectedModelLabel}
        hasModelSelected={runtimeState.hasModelSelected}
        selectedPersonaLabel={runtimeState.selectedPersonaLabel}
        assistantSource={runtimeState.assistantSource}
        workspaceAssistantDegradedReason={
          runtimeState.workspaceAssistantDegradedReason
        }
        sendError={runtimeState.sendError}
        effectiveAssistantDefault={effectiveAssistantDefault}
        backendAvailable={backendAvailable}
        chatBackendAvailable={backendAvailable && workspaceReady}
        streaming={runtimeState.streaming}
        connectionMode={connectionMode}
        sending={runtimeState.sending}
        historyLoading={runtimeState.historyLoading}
        historyLoadError={runtimeState.historyLoadError}
        onBrowseSource={handleBrowseSource}
        onCloseBrowseSource={handleCloseBrowseSource}
        onStageSources={handleStageSources}
        onUnstageSource={handleUnstageSource}
        onClearStagedSources={handleClearStagedSources}
        onRuntimeStateChange={handleRuntimeStateChange}
      />
  )
  return (
    <div data-testid="chat-workspace-page" className="h-full min-h-0 w-full min-w-0">
      <h1 className="sr-only">Chat Workspace</h1>
      {workspaceId && workspaceReady ? (
        <ActivatedLocalWorkspace key={workspaceId} workspaceId={workspaceId} webClip={false}>
          {console}
        </ActivatedLocalWorkspace>
      ) : console}
    </div>
  )
}
