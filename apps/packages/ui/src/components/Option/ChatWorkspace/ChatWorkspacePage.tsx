import React from "react"

import { useConnectionState } from "@/hooks/useConnectionState"
import { useChatSurfaceCoordinatorStore } from "@/store/chat-surface-coordinator"
import { useWorkspaceStore } from "@/store/workspace"
import { ConnectionPhase } from "@/types/connection"

import { ChatWorkspaceConsole } from "./ChatWorkspaceConsole"
import { stageWorkspaceSources } from "./staging"
import type {
  ChatWorkspaceRuntimeState,
  StagedWorkspaceSource
} from "./types"

const createInitialRuntimeState = (
  backendAvailable: boolean
): ChatWorkspaceRuntimeState => ({
  backendAvailable,
  streaming: false,
  selectedModelLabel: "No model selected",
  selectedPersonaLabel: null
})

export const ChatWorkspacePage = () => {
  const workspaceId = useWorkspaceStore((state) => state.workspaceId)
  const workspaceName = useWorkspaceStore((state) => state.workspaceName)
  const sources = useWorkspaceStore((state) => state.sources)
  const setRouteContext = useChatSurfaceCoordinatorStore(
    (state) => state.setRouteContext
  )
  const connectionState = useConnectionState()
  const backendAvailable =
    connectionState.isConnected &&
    connectionState.phase === ConnectionPhase.CONNECTED

  const [browsedSourceId, setBrowsedSourceId] = React.useState<string | null>(null)
  const [stagedSources, setStagedSources] = React.useState<StagedWorkspaceSource[]>(
    []
  )
  const [runtimeState, setRuntimeState] =
    React.useState<ChatWorkspaceRuntimeState>(() =>
      createInitialRuntimeState(backendAvailable)
    )

  const scopeLabel = workspaceName || "Workspace"

  React.useEffect(() => {
    setRouteContext({
      routeId: "chat-workspace",
      surface: "webui"
    })
  }, [setRouteContext])

  React.useEffect(() => {
    setRuntimeState((current) =>
      current.backendAvailable === backendAvailable
        ? current
        : { ...current, backendAvailable }
    )
  }, [backendAvailable])

  const stagedSourceIds = React.useMemo(
    () => stagedSources.map((source) => source.sourceId),
    [stagedSources]
  )

  const handleBrowseSource = React.useCallback((sourceId: string) => {
    setBrowsedSourceId(sourceId)
  }, [])

  const handleStageSources = React.useCallback(
    (sourceIds: string[]) => {
      const selected = sources.filter((source) => sourceIds.includes(source.id))
      setStagedSources((current) =>
        stageWorkspaceSources(current, selected, scopeLabel)
      )
    },
    [scopeLabel, sources]
  )

  const handleClearStagedSources = React.useCallback(() => {
    setStagedSources([])
  }, [])

  const handleRuntimeStateChange = React.useCallback(
    (state: ChatWorkspaceRuntimeState) => {
      setRuntimeState((current) => ({ ...current, ...state }))
    },
    []
  )

  return (
    <div data-testid="chat-workspace-page" className="h-full min-h-0 w-full">
      <ChatWorkspaceConsole
        workspaceId={workspaceId}
        workspaceName={scopeLabel}
        sources={sources}
        browsedSourceId={browsedSourceId}
        stagedSources={stagedSources}
        stagedSourceIds={stagedSourceIds}
        selectedModelLabel={runtimeState.selectedModelLabel}
        selectedPersonaLabel={runtimeState.selectedPersonaLabel}
        backendAvailable={backendAvailable}
        streaming={runtimeState.streaming}
        onBrowseSource={handleBrowseSource}
        onStageSources={handleStageSources}
        onClearStagedSources={handleClearStagedSources}
        onRuntimeStateChange={handleRuntimeStateChange}
      />
    </div>
  )
}
