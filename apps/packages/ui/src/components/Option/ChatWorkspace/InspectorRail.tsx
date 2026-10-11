import type { WorkspaceAssistantDefaultDegradedReason } from "@/types/workspace"
import {
  getChatWorkspaceRuntimeLabel,
  type ChatWorkspaceAssistantSource,
  type ChatWorkspaceRuntimeStatus
} from "./types"

export type InspectorRailStagedSource = {
  sourceId: string
  title: string
}

export type InspectorRailProps = ChatWorkspaceRuntimeStatus & {
  scopeLabel: string
  stagedSourceCount: number
  stagedSources: InspectorRailStagedSource[]
  selectedModelLabel: string
  hasModelSelected: boolean
  selectedPersonaLabel: string | null
  assistantSource: ChatWorkspaceAssistantSource
  workspaceAssistantDegradedReason?: WorkspaceAssistantDefaultDegradedReason | null
}

const panelClass = "rounded-md border border-border bg-surface px-3 py-2"
const headingClass = "text-[11px] font-semibold text-text-muted"
const valueClass = "mt-1 text-sm font-medium text-text"
const mutedClass = "mt-1 text-xs text-text-muted"
const getRuntimeRecoveryCopy = ({
  backendAvailable,
  workspaceReady,
  connectionMode,
  historyLoading,
  historyLoadError,
  sendError
}: ChatWorkspaceRuntimeStatus) => {
  if (connectionMode === "demo" || connectionMode === "bypass") {
    return "Connect to a live server before sending workspace chat."
  }
  if (!backendAvailable) {
    return "Reconnect to the server before sending workspace chat."
  }

  if (!workspaceReady) {
    return "Wait for workspace identity before sending."
  }

  if (historyLoading) return "Wait for chat history before sending."
  if (historyLoadError) {
    return "Reopen the conversation to load its history before sending."
  }

  if (sendError) {
    return "Draft and staged context were preserved for retry."
  }

  return null
}

const degradedReasonLabels: Record<
  WorkspaceAssistantDefaultDegradedReason,
  string
> = {
  invalid_default: "Invalid default",
  permission_denied: "Permission denied",
  persona_deleted: "Persona deleted",
  persona_feature_disabled: "Persona feature disabled",
  persona_unavailable: "Persona unavailable",
  unsupported_assistant_kind: "Unsupported assistant kind"
}

const getAssistantSourceLabel = (
  assistantSource: ChatWorkspaceAssistantSource
) => {
  if (assistantSource === "workspace") return "Inherited from workspace"
  if (assistantSource === "explicit") return "Explicit persona"
  return null
}

export const InspectorRail = ({
  scopeLabel,
  stagedSourceCount,
  stagedSources,
  selectedModelLabel,
  hasModelSelected,
  selectedPersonaLabel,
  assistantSource,
  workspaceAssistantDegradedReason,
  backendAvailable,
  workspaceReady,
  connectionMode,
  streaming,
  sending,
  historyLoading,
  historyLoadError,
  sendError
}: InspectorRailProps) => {
  const runtime = {
    backendAvailable,
    workspaceReady,
    connectionMode,
    streaming,
    sending,
    historyLoading,
    historyLoadError,
    sendError
  }
  const runtimeLabel = getChatWorkspaceRuntimeLabel(runtime)
  const runtimeRecoveryCopy = getRuntimeRecoveryCopy(runtime)
  const assistantSourceLabel = getAssistantSourceLabel(assistantSource)
  const degradedReasonLabel = workspaceAssistantDegradedReason
    ? degradedReasonLabels[workspaceAssistantDegradedReason]
    : null
  const modelRecoveryCopy =
    !hasModelSelected
      ? "Choose a model before sending."
      : null
  const personaRecoveryCopy =
    assistantSource === "none"
      ? "Persona is optional; workspace defaults apply when available."
      : null

  return (
    <aside
      aria-label="Chat workspace inspector"
      className="flex min-w-0 flex-col gap-2 text-sm"
    >
      <section className={panelClass}>
        <h2 className={headingClass}>Scope</h2>
        <p className={valueClass}>{scopeLabel}</p>
      </section>

      <section className={panelClass}>
        <h2 className={headingClass}>Sources</h2>
        <p className={valueClass}>
          {stagedSourceCount} source{stagedSourceCount === 1 ? "" : "s"} staged
        </p>
        {stagedSources.length > 0 ? (
          <ul className="mt-2 space-y-1">
            {stagedSources.map((source) => (
              <li
                key={source.sourceId}
                className="min-w-0 break-words text-xs text-text"
              >
                {source.title}
              </li>
            ))}
          </ul>
        ) : (
          <p className={mutedClass}>No sources staged</p>
        )}
      </section>

      <section className={panelClass}>
        <h2 className={headingClass}>Model / Persona</h2>
        <p className={valueClass}>{selectedModelLabel}</p>
        {assistantSource === "unavailable" ? (
          <>
            <p className={mutedClass}>Workspace default unavailable</p>
            {degradedReasonLabel ? (
              <p className={mutedClass}>{degradedReasonLabel}</p>
            ) : null}
          </>
        ) : (
          <>
            <p className={mutedClass}>
              {selectedPersonaLabel ?? "No persona selected"}
            </p>
            {assistantSourceLabel ? (
              <p className={mutedClass}>{assistantSourceLabel}</p>
            ) : null}
            {modelRecoveryCopy ? (
              <p className={mutedClass}>{modelRecoveryCopy}</p>
            ) : null}
            {personaRecoveryCopy ? (
              <p className={mutedClass}>{personaRecoveryCopy}</p>
            ) : null}
          </>
        )}
      </section>

      <section className={panelClass}>
        <h2 className={headingClass}>Runtime</h2>
        <p className={valueClass}>{runtimeLabel}</p>
        {runtimeRecoveryCopy ? (
          <p className={mutedClass}>{runtimeRecoveryCopy}</p>
        ) : null}
      </section>
    </aside>
  )
}
