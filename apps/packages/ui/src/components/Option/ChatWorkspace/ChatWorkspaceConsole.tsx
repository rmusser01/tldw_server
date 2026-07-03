import type {
  EffectiveWorkspaceAssistantDefault,
  WorkspaceSource
} from "@/types/workspace"

import { InspectorRail } from "./InspectorRail"
import { WorkspaceChatPanel } from "./WorkspaceChatPanel"
import { WorkspaceRail } from "./WorkspaceRail"
import { WorkspaceStatusStrip } from "./WorkspaceStatusStrip"
import type {
  ChatWorkspaceAssistantSource,
  ChatWorkspaceRuntimeState,
  StagedWorkspaceSource
} from "./types"
import { normalizeWorkspaceId } from "./workspaceIdentity"

export type ChatWorkspaceConsoleProps = {
  workspaceId?: string | null
  workspaceName: string
  sources: WorkspaceSource[]
  sourcesLoading?: boolean
  sourcesError?: string | null
  browsedSourceId: string | null
  stagedSources: StagedWorkspaceSource[]
  selectedModelLabel: string
  hasModelSelected: boolean
  selectedPersonaLabel: string | null
  assistantSource: ChatWorkspaceAssistantSource
  workspaceAssistantDegradedReason?: ChatWorkspaceRuntimeState[
    "workspaceAssistantDegradedReason"
  ]
  sendError?: string | null
  effectiveAssistantDefault?: EffectiveWorkspaceAssistantDefault | null
  backendAvailable: boolean
  chatBackendAvailable: boolean
  streaming: boolean
  onBrowseSource: (sourceId: string) => void
  onStageSources: (sourceIds: string[]) => void
  onUnstageSource: (sourceId: string) => void
  onClearStagedSources: () => void
  onRuntimeStateChange: (state: ChatWorkspaceRuntimeState) => void
}

export const ChatWorkspaceConsole = ({
  workspaceId,
  workspaceName,
  sources,
  sourcesLoading,
  sourcesError,
  browsedSourceId,
  stagedSources,
  selectedModelLabel,
  hasModelSelected,
  selectedPersonaLabel,
  assistantSource,
  workspaceAssistantDegradedReason,
  sendError,
  effectiveAssistantDefault,
  backendAvailable,
  chatBackendAvailable,
  streaming,
  onBrowseSource,
  onStageSources,
  onUnstageSource,
  onClearStagedSources,
  onRuntimeStateChange
}: ChatWorkspaceConsoleProps) => {
  const normalizedWorkspaceId = normalizeWorkspaceId(workspaceId)
  const workspaceReady = normalizedWorkspaceId !== null
  const stagedSourceIds = stagedSources.map((source) => source.sourceId)
  const inspectorSources = stagedSources.map((source) => ({
    sourceId: source.sourceId,
    title: source.title
  }))

  return (
    <div
      data-testid="chat-workspace-console"
      className="grid h-full min-h-0 w-full grid-cols-1 overflow-hidden border border-border bg-background text-text lg:grid-cols-[minmax(260px,320px)_minmax(0,1fr)_minmax(280px,340px)]"
    >
      <div className="flex h-full min-h-0 flex-col overflow-y-auto lg:contents">
        <div className="order-2 min-h-0 overflow-y-auto border-t border-border bg-surface2/30 p-2 lg:order-1 lg:border-r lg:border-t-0">
          <WorkspaceRail
            workspaceName={workspaceName}
            sources={sources}
            sourcesLoading={sourcesLoading}
            sourcesError={sourcesError}
            browsedSourceId={browsedSourceId}
            stagedSourceIds={stagedSourceIds}
            onBrowseSource={onBrowseSource}
            onStageSources={onStageSources}
            onUnstageSource={onUnstageSource}
          />
        </div>

        <main className="order-1 flex min-h-[520px] min-w-0 flex-col overflow-hidden bg-background lg:order-2 lg:min-h-0">
          <div className="min-h-0 flex-1 overflow-hidden">
            <WorkspaceChatPanel
              key={normalizedWorkspaceId ?? "global"}
              workspaceId={normalizedWorkspaceId}
              workspaceName={workspaceName}
              stagedSources={stagedSources}
              onClearStagedSources={onClearStagedSources}
              onRemoveStagedSource={onUnstageSource}
              backendAvailable={chatBackendAvailable}
              effectiveAssistantDefault={effectiveAssistantDefault}
              onRuntimeStateChange={onRuntimeStateChange}
            />
          </div>
          <WorkspaceStatusStrip
            backendAvailable={backendAvailable}
            workspaceReady={workspaceReady}
            streaming={streaming}
            sendError={sendError}
            stagedSourceCount={stagedSources.length}
            hasModelSelected={hasModelSelected}
            selectedPersonaLabel={selectedPersonaLabel}
            assistantSource={assistantSource}
          />
        </main>

        <div className="order-3 min-h-0 overflow-y-auto border-t border-border bg-surface2/30 p-2 lg:border-l lg:border-t-0">
          <InspectorRail
            scopeLabel={workspaceName}
            stagedSourceCount={stagedSources.length}
            stagedSources={inspectorSources}
            selectedModelLabel={selectedModelLabel}
            hasModelSelected={hasModelSelected}
            selectedPersonaLabel={selectedPersonaLabel}
            assistantSource={assistantSource}
            workspaceAssistantDegradedReason={workspaceAssistantDegradedReason}
            backendAvailable={backendAvailable}
            workspaceReady={workspaceReady}
            streaming={streaming}
            sendError={sendError}
          />
        </div>
      </div>
    </div>
  )
}
