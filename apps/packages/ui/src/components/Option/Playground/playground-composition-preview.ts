import type { ConversationContextComposition } from "@/types/conversation-context";
import type { ChatModelUsabilityStatus } from "@/utils/chat-model-availability";
import type {
  PlaygroundContextSource,
  PlaygroundPromptSummary,
} from "./PlaygroundContextRail";
import type {
  RuntimeAssistantSummary,
  RuntimeSettingSummary,
  RuntimeToolSummary,
} from "./PlaygroundRuntimeInspector";

export type PlaygroundCompositionPreviewState =
  | "ready"
  | "degraded"
  | "unavailable";

export type PlaygroundCompositionPreviewEntryState =
  | "active"
  | "disabled"
  | "degraded"
  | "unavailable"
  | "loading";

export type PlaygroundCompositionPreviewEntryKind =
  | "prompt"
  | "assistant"
  | "model"
  | "settings"
  | "context"
  | "tools"
  | "composition"
  | PlaygroundContextSource["kind"];

export type PlaygroundCompositionPreviewEntry = {
  id: string;
  kind: PlaygroundCompositionPreviewEntryKind;
  label: string;
  title: string;
  detail?: string | null;
  state: PlaygroundCompositionPreviewEntryState;
};

export type PlaygroundCompositionPreviewFootprint = {
  providerMessageCount: number;
  previewSectionCount: number;
  contextPieceCount: number;
  warningCount: number;
  readiness: ConversationContextComposition["readiness"] | "unavailable";
};

export type PlaygroundCompositionPreviewInput = {
  promptSummary: PlaygroundPromptSummary;
  assistantSummary: RuntimeAssistantSummary;
  providerRoute: {
    selectedProvider: string | null;
    selectedModel: string | null;
    providerRouteLabel: string | null;
  };
  settingSummaries: RuntimeSettingSummary[];
  contextSources: PlaygroundContextSource[];
  toolSummary: RuntimeToolSummary | null;
  compositionStatus: "idle" | "loading" | "ready" | "error";
  composition: ConversationContextComposition | null;
  modelUsabilityStatus?: ChatModelUsabilityStatus | null;
  modelUsabilityCanSend?: boolean | null;
  modelUsabilityDetail?: string | null;
  modelUnavailable?: boolean;
  modelUnavailableDetail?: string | null;
};

export type PlaygroundCompositionPreviewSummary = {
  overallState: PlaygroundCompositionPreviewState;
  entries: PlaygroundCompositionPreviewEntry[];
  contextStack: PlaygroundCompositionPreviewEntry[];
  settingsScopeLabel: string | null;
  footprint: PlaygroundCompositionPreviewFootprint;
};

const stateRank: Record<PlaygroundCompositionPreviewEntryState, number> = {
  unavailable: 0,
  degraded: 1,
  loading: 2,
  active: 3,
  disabled: 4,
};

const normalizeState = (
  state: PlaygroundContextSource["state"] | undefined,
): PlaygroundCompositionPreviewEntryState => {
  if (state === "degraded") return "degraded";
  if (state === "disabled") return "disabled";
  return "active";
};

const mostSevereState = (
  states: PlaygroundCompositionPreviewEntryState[],
  fallback: PlaygroundCompositionPreviewEntryState,
): PlaygroundCompositionPreviewEntryState =>
  states.reduce(
    (current, next) => (stateRank[next] < stateRank[current] ? next : current),
    fallback,
  );

const countActiveSources = (sources: PlaygroundContextSource[]): number =>
  sources.filter((source) => source.state !== "disabled").length;

const formatSourceCount = (count: number): string =>
  count === 1 ? "1 active source" : `${count} active sources`;

const buildFootprint = (
  composition: ConversationContextComposition | null,
): PlaygroundCompositionPreviewFootprint => ({
  providerMessageCount: composition?.providerMessages.length ?? 0,
  previewSectionCount: composition?.previewSections.length ?? 0,
  contextPieceCount: composition?.pieces.length ?? 0,
  warningCount: composition?.warnings.length ?? 0,
  readiness: composition?.readiness ?? "unavailable",
});

const mapToolState = (
  state: RuntimeToolSummary["state"],
): PlaygroundCompositionPreviewEntryState => {
  if (state === "available") return "active";
  if (state === "degraded") return "degraded";
  if (state === "unavailable") return "unavailable";
  return "disabled";
};

const mapModelUsabilityState = (
  status: ChatModelUsabilityStatus | null | undefined,
  canSend: boolean | null | undefined,
  legacyUnavailable: boolean,
  hasSelectedModel: boolean,
): PlaygroundCompositionPreviewEntryState => {
  if (status === "loading") return "loading";
  if (status === "degraded") {
    if (canSend === false) return "unavailable";
    return hasSelectedModel ? "degraded" : "unavailable";
  }
  if (status === "ready") {
    return hasSelectedModel ? "active" : "unavailable";
  }
  if (status) return "unavailable";
  if (hasSelectedModel && !legacyUnavailable) return "active";
  return "unavailable";
};

export const buildPlaygroundCompositionPreviewSummary = ({
  promptSummary,
  assistantSummary,
  providerRoute,
  settingSummaries,
  contextSources,
  toolSummary,
  compositionStatus,
  composition,
  modelUsabilityStatus = null,
  modelUsabilityCanSend = null,
  modelUsabilityDetail = null,
  modelUnavailable = false,
  modelUnavailableDetail = null,
}: PlaygroundCompositionPreviewInput): PlaygroundCompositionPreviewSummary => {
  const settingsScopeLabel = providerRoute.providerRouteLabel;
  const promptEntry: PlaygroundCompositionPreviewEntry = {
    id: "prompt",
    kind: "prompt",
    label: "Prompt",
    title: promptSummary.label,
    detail: promptSummary.detail,
    state: promptSummary.state === "none" ? "disabled" : "active",
  };
  const assistantName = assistantSummary.name || "No assistant selected";
  const assistantEntry: PlaygroundCompositionPreviewEntry = {
    id: "assistant",
    kind: "assistant",
    label: "Assistant",
    title: assistantName,
    detail: assistantSummary.detail,
    state: assistantSummary.mode === "none" ? "disabled" : "active",
  };
  const modelTitle =
    providerRoute.providerRouteLabel ||
    providerRoute.selectedModel ||
    "No model selected";
  const modelEntry: PlaygroundCompositionPreviewEntry = {
    id: "model",
    kind: "model",
    label: "Model",
    title: modelTitle,
    detail:
      modelUsabilityDetail ||
      (modelUnavailable ? modelUnavailableDetail : null) ||
      providerRoute.selectedProvider ||
      null,
    state: mapModelUsabilityState(
      modelUsabilityStatus,
      modelUsabilityCanSend,
      modelUnavailable,
      Boolean(providerRoute.selectedModel),
    ),
  };
  const settingsEntry: PlaygroundCompositionPreviewEntry = {
    id: "settings",
    kind: "settings",
    label: "Model settings",
    title: settingsScopeLabel || "Default model settings",
    detail:
      settingSummaries.length > 0
        ? settingSummaries
            .map((setting) => `${setting.label}: ${setting.value}`)
            .join(", ")
        : null,
    state: settingsScopeLabel
      ? settingSummaries.some((setting) => setting.source === "override")
        ? "active"
        : "disabled"
      : "disabled",
  };
  const activeSourceCount = countActiveSources(contextSources);
  const activeSourceStates = contextSources
    .map((source) => normalizeState(source.state))
    .filter((state) => state !== "disabled");
  const contextState =
    activeSourceStates.length > 0
      ? mostSevereState(activeSourceStates, "active")
      : "disabled";
  const contextEntry: PlaygroundCompositionPreviewEntry = {
    id: "context",
    kind: "context",
    label: "Context",
    title:
      activeSourceCount > 0 ? formatSourceCount(activeSourceCount) : "No extra context",
    detail:
      contextSources.length > 0
        ? `${contextSources.length} configured source${
            contextSources.length === 1 ? "" : "s"
          }`
        : null,
    state: contextState,
  };
  const toolEntry: PlaygroundCompositionPreviewEntry = toolSummary
    ? {
        id: "tools",
        kind: "tools",
        label: "MCP tools",
        title: toolSummary.label,
        detail: toolSummary.detail,
        state: mapToolState(toolSummary.state),
      }
    : {
        id: "tools",
        kind: "tools",
        label: "MCP tools",
        title: "MCP tools managed from composer",
        detail: null,
        state: "disabled",
      };
  const entries = [
    promptEntry,
    assistantEntry,
    modelEntry,
    settingsEntry,
    contextEntry,
    toolEntry,
  ];

  if (compositionStatus === "loading") {
    entries.push({
      id: "composition-loading",
      kind: "composition",
      label: "Preview",
      title: "Context preview loading",
      detail: null,
      state: "loading",
    });
  } else if (compositionStatus === "error") {
    entries.push({
      id: "composition-error",
      kind: "composition",
      label: "Preview",
      title: "Context preview unavailable",
      detail: null,
      state: "unavailable",
    });
  } else if (composition?.readiness === "partial") {
    entries.push({
      id: "composition-partial",
      kind: "composition",
      label: "Preview",
      title: "Context preview partial",
      detail: composition.warnings.join(", ") || null,
      state: "degraded",
    });
  } else if (composition?.readiness === "blocked") {
    entries.push({
      id: "composition-blocked",
      kind: "composition",
      label: "Preview",
      title: "Context preview blocked",
      detail: composition.warnings.join(", ") || null,
      state: "unavailable",
    });
  }

  const sourceEntries: PlaygroundCompositionPreviewEntry[] = contextSources
    .filter((source) => {
      if (source.kind === "prompt" && promptEntry.state !== "disabled") {
        return false;
      }
      if (source.kind === "assistant" && assistantEntry.state !== "disabled") {
        return false;
      }
      return true;
    })
    .map((source) => ({
      id: `source-${source.id}`,
      kind: source.kind,
      label: source.label,
      title: source.title,
      detail: source.detail,
      state: normalizeState(source.state),
    }));
  const contextStack = [
    ...(promptEntry.state !== "disabled" ? [promptEntry] : []),
    ...(assistantEntry.state !== "disabled" ? [assistantEntry] : []),
    ...sourceEntries,
    ...(toolEntry.state !== "disabled" ? [toolEntry] : []),
  ];
  const hasBlockingModel =
    modelEntry.state === "unavailable" || modelEntry.state === "loading";
  const hasUnhealthyOptionalEntry = [...entries, ...sourceEntries].some(
    (entry) => entry.state === "degraded" || entry.state === "unavailable",
  );
  const overallState: PlaygroundCompositionPreviewState = hasBlockingModel
    ? "unavailable"
    : hasUnhealthyOptionalEntry
      ? "degraded"
      : "ready";

  return {
    overallState,
    entries,
    contextStack,
    settingsScopeLabel,
    footprint: buildFootprint(composition),
  };
};
