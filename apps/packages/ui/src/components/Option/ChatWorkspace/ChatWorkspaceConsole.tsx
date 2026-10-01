import { useId, useState } from "react"
import { MessageSquare, Library, Info } from "lucide-react"
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
  ChatWorkspaceRuntimeStatus,
  StagedWorkspaceSource
} from "./types"
import { normalizeWorkspaceId } from "./workspaceIdentity"
import { WorkspaceSourcePreview } from "../ResearchWorkspace/SourcesPane/WorkspaceSourcePreview"

export type ChatWorkspaceConsoleProps = Pick<
  ChatWorkspaceRuntimeStatus,
  "connectionMode" | "sending" | "historyLoading" | "historyLoadError"
> & {
  workspaceId?: string | null
  workspaceReady: boolean
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
  workspaceAssistantDegradedReason?: ChatWorkspaceRuntimeState["workspaceAssistantDegradedReason"]
  sendError?: string | null
  effectiveAssistantDefault?: EffectiveWorkspaceAssistantDefault | null
  backendAvailable: boolean
  chatBackendAvailable: boolean
  streaming: boolean
  onBrowseSource: (sourceId: string) => void
  onCloseBrowseSource?: () => void
  onStageSources: (sourceIds: string[]) => void
  onUnstageSource: (sourceId: string) => void
  onClearStagedSources: () => void
  onRuntimeStateChange: (state: ChatWorkspaceRuntimeState) => void
}

export const ChatWorkspaceConsole = ({
  workspaceId,
  workspaceReady,
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
  connectionMode,
  sending,
  historyLoading,
  historyLoadError,
  onBrowseSource,
  onCloseBrowseSource,
  onStageSources,
  onUnstageSource,
  onClearStagedSources,
  onRuntimeStateChange
}: ChatWorkspaceConsoleProps) => {
  const [activePane, setActivePane] = useState("chat")
  const paneId = useId()
  const normalizedWorkspaceId = normalizeWorkspaceId(workspaceId)
  const browsedSource = workspaceReady && normalizedWorkspaceId
    ? sources.find((source) => source.id === browsedSourceId) ?? null
    : null
  const stagedSourceIds = stagedSources.map((source) => source.sourceId)
  const inspectorSources = stagedSources.map((source) => ({
    sourceId: source.sourceId,
    title: source.title
  }))

  return (
    <div
      data-testid="chat-workspace-console"
      className="flex h-full min-h-0 w-full flex-col overflow-hidden border border-border bg-bg text-text [overflow-wrap:anywhere]"
    >
      <nav
        aria-label="Workspace panels"
        className="flex shrink-0 gap-1 border-b border-border bg-surface2 p-1 xl:hidden"
      >
        {[
          { id: "chat", label: "Chat", Icon: MessageSquare },
          { id: "sources", label: "Sources", Icon: Library },
          { id: "inspector", label: "Inspector", Icon: Info }
        ].map(({ id, label, Icon }) => (
          <button
            key={id}
            type="button"
            aria-pressed={activePane === id}
            aria-controls={`${paneId}-${id}`}
            onClick={() => setActivePane(id)}
            className="flex min-h-11 min-w-0 flex-1 items-center justify-center gap-1 rounded-md px-1 text-xs font-medium text-text hover:bg-surface focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus aria-pressed:bg-surface aria-pressed:font-semibold"
          >
            <Icon size={16} className="shrink-0" aria-hidden="true" />
            {label}
          </button>
        ))}
      </nav>
      <div className="grid min-h-0 flex-1 grid-cols-1 xl:grid-cols-[minmax(220px,280px)_minmax(0,1fr)_minmax(220px,280px)]">
        <div
          id={`${paneId}-sources`}
          onFocusCapture={() => setActivePane("sources")}
          className={`${activePane === "sources" ? "block" : "hidden"} min-h-0 min-w-0 overflow-y-auto bg-surface2/30 p-2 xl:block xl:border-r xl:border-border`}
        >
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

        <div
          id={`${paneId}-chat`}
          onFocusCapture={() => setActivePane("chat")}
          className={`${activePane === "chat" ? "flex" : "hidden"} min-h-0 min-w-0 flex-col overflow-hidden bg-bg xl:flex`}
        >
          <div className="min-h-0 flex-1 overflow-hidden">
            <WorkspaceChatPanel
              key={normalizedWorkspaceId ?? "global"}
              workspaceId={normalizedWorkspaceId}
              workspaceReady={workspaceReady}
              workspaceName={workspaceName}
              stagedSources={stagedSources}
              onClearStagedSources={onClearStagedSources}
              onRemoveStagedSource={onUnstageSource}
              backendAvailable={chatBackendAvailable}
              effectiveAssistantDefault={effectiveAssistantDefault}
              onRuntimeStateChange={onRuntimeStateChange}
            />
          </div>
        </div>

        <div
          id={`${paneId}-inspector`}
          onFocusCapture={() => setActivePane("inspector")}
          className={`${activePane === "inspector" ? "block" : "hidden"} min-h-0 min-w-0 overflow-y-auto bg-surface2/30 p-2 xl:block xl:border-l xl:border-border`}
        >
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
            connectionMode={connectionMode}
            sending={sending}
            historyLoading={historyLoading}
            historyLoadError={historyLoadError}
            sendError={sendError}
          />
        </div>
      </div>
      <WorkspaceStatusStrip
        backendAvailable={backendAvailable}
        workspaceReady={workspaceReady}
        streaming={streaming}
        connectionMode={connectionMode}
        sending={sending}
        historyLoading={historyLoading}
        historyLoadError={historyLoadError}
        sendError={sendError}
        stagedSourceCount={stagedSources.length}
        hasModelSelected={hasModelSelected}
        selectedPersonaLabel={selectedPersonaLabel}
        assistantSource={assistantSource}
      />
      <WorkspaceSourcePreview
        workspaceId={normalizedWorkspaceId}
        source={browsedSource}
        onClose={onCloseBrowseSource ?? (() => undefined)}
      />
    </div>
  )
}
