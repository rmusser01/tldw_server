import React from "react"

import { useConnectionState } from "@/hooks/useConnectionState"
import { useChatSurfaceCoordinatorStore } from "@/store/chat-surface-coordinator"
import { useWorkspaceStore } from "@/store/workspace"
import { ConnectionPhase } from "@/types/connection"

import { ChatWorkspaceConsole } from "./ChatWorkspaceConsole"
import { stageWorkspaceSources, unstageWorkspaceSource } from "./staging"
import type {
  ChatWorkspaceRuntimeState,
  StagedWorkspaceSource
} from "./types"
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
  sendError: null,
  selectedModelLabel: "No model selected",
  hasModelSelected: false,
  selectedPersonaLabel: null,
  assistantSource:
    effectiveAssistantDefault?.status === "unavailable" ? "unavailable" : "none",
  workspaceAssistantDegradedReason:
    effectiveAssistantDefault?.status === "unavailable"
      ? effectiveAssistantDefault.degradedReason
      : null
})

export const ChatWorkspacePage = () => {
  const rawWorkspaceId = useWorkspaceStore((state) => state.workspaceId)
  const storeHydrated = useWorkspaceStore((state) => state.storeHydrated)
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
  const backendAvailable =
    connectionState.isConnected &&
    connectionState.phase === ConnectionPhase.CONNECTED
  const workspaceId = React.useMemo(
    () => normalizeWorkspaceId(rawWorkspaceId),
    [rawWorkspaceId]
  )
  const workspaceReady = storeHydrated && workspaceId !== null

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
      focusSourceById(sourceId)
      setWorkspaceContext((current) => ({
        workspaceId,
        browsedSourceId: sourceId,
        stagedSources:
          current.workspaceId === workspaceId ? current.stagedSources : []
      }))
    },
    [focusSourceById, workspaceId]
  )

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

  return (
    <div data-testid="chat-workspace-page" className="h-full min-h-0 w-full min-w-0">
      <h1 className="sr-only">Chat Workspace</h1>
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
        onBrowseSource={handleBrowseSource}
        onStageSources={handleStageSources}
        onUnstageSource={handleUnstageSource}
        onClearStagedSources={handleClearStagedSources}
        onRuntimeStateChange={handleRuntimeStateChange}
      />
    </div>
  )
}
