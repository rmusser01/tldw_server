import type { AssistantSelection } from "@/types/assistant-selection";
import { parseProviderQualifiedModelSelection } from "@/utils/resolve-api-provider";
import type { PlaygroundPromptSummary } from "./PlaygroundContextRail";
import type {
  RuntimeAssistantSummary,
  RuntimeToolSummary,
} from "./PlaygroundRuntimeInspector";
import type { McpHealthState } from "@/store/mcp-tools";

type LegacyCharacterSelection = {
  id?: string | number;
  name?: string | null;
};

type PersonaMemoryMode = "read_only" | "read_write";
type CockpitMcpHealthState = McpHealthState | "degraded";

type AssistantSummaryCopy = Partial<{
  assistantFallbackName: string;
  characterSelected: string;
  legacyCharacterFallbackName: (id: string | number) => string;
  memoryReadOnly: string;
  memoryReadWrite: string;
  noAssistantSelected: string;
  personaFallbackName: string;
  personaSelected: string;
  personaSelectedWithMemoryMode: (memoryMode: string) => string;
}>;

type PromptSummaryCopy = Partial<{
  customPromptLabel: string;
  inlineSystemPromptActiveDetail: string;
  loadingPromptDetail: string;
  noPromptContextDetail: string;
  noPromptSelectedLabel: string;
  selectedPromptUnavailableDetail: string;
  quickPromptLabel: string;
  selectedPromptDetail: string;
  systemPromptLabel: string;
}>;

type McpSummaryCopy = Partial<{
  availableDetail: (chatToolCount: number, discoveredCount: number) => string;
  emptyDetail: string;
  loadingDetail: string;
  offlineDetail: string;
  toolsLabel: string;
  unavailableDetail: string;
  unavailableLabel: string;
}>;

const defaultAssistantCopy = {
  assistantFallbackName: "Assistant",
  characterSelected: "Character selected",
  legacyCharacterFallbackName: (id: string | number) => `Character ${id}`,
  memoryReadOnly: "memory read-only",
  memoryReadWrite: "memory read/write",
  noAssistantSelected: "No assistant selected",
  personaFallbackName: "Persona",
  personaSelected: "Persona selected",
  personaSelectedWithMemoryMode: (memoryMode: string) =>
    `Persona selected - ${memoryMode}`,
} satisfies Required<AssistantSummaryCopy>;

const defaultPromptCopy = {
  customPromptLabel: "Custom prompt",
  inlineSystemPromptActiveDetail: "Inline system prompt active",
  loadingPromptDetail: "Loading prompt details...",
  noPromptContextDetail: "No prompt context will be added.",
  noPromptSelectedLabel: "No prompt selected",
  selectedPromptUnavailableDetail: "Prompt details unavailable",
  quickPromptLabel: "Quick prompt",
  selectedPromptDetail: "System prompt",
  systemPromptLabel: "System prompt",
} satisfies Required<PromptSummaryCopy>;

const defaultMcpCopy = {
  availableDetail: (chatToolCount: number, discoveredCount: number) => {
    const chatToolsLabel = formatCountLabel(
      chatToolCount,
      "chat tool",
      "chat tools",
    );
    const discoveredSuffix =
      discoveredCount > chatToolCount ? ` (${discoveredCount} discovered)` : "";
    return `${chatToolsLabel} available${discoveredSuffix}`;
  },
  emptyDetail: "No MCP tools available",
  loadingDetail: "Loading tools...",
  offlineDetail: "MCP tools are offline",
  toolsLabel: "MCP tools",
  unavailableDetail: "MCP tools unavailable",
  unavailableLabel: "MCP unavailable",
} satisfies Required<McpSummaryCopy>;

const normalizeText = (value: string | null | undefined): string | null => {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
};

const formatPersonaMemoryMode = (
  personaMemoryMode: PersonaMemoryMode | null | undefined,
  copy: Required<AssistantSummaryCopy>,
): string | null => {
  if (personaMemoryMode === "read_only") return copy.memoryReadOnly;
  if (personaMemoryMode === "read_write") return copy.memoryReadWrite;
  return null;
};

const formatCountLabel = (
  count: number,
  singular: string,
  plural: string,
): string => `${count} ${count === 1 ? singular : plural}`;

export function buildCockpitAssistantSummary(input: {
  selectedAssistant: AssistantSelection | null | undefined;
  selectedCharacter: LegacyCharacterSelection | null | undefined;
  personaMemoryMode?: PersonaMemoryMode | null;
  copy?: AssistantSummaryCopy;
}): RuntimeAssistantSummary {
  const copy = { ...defaultAssistantCopy, ...input.copy };
  const selectedAssistant = input.selectedAssistant;
  if (selectedAssistant?.kind === "persona") {
    const memoryMode = formatPersonaMemoryMode(input.personaMemoryMode, copy);
    return {
      mode: "persona",
      name: normalizeText(selectedAssistant.name) || copy.personaFallbackName,
      detail: memoryMode
        ? copy.personaSelectedWithMemoryMode(memoryMode)
        : copy.personaSelected,
    };
  }

  if (selectedAssistant?.kind === "character") {
    return {
      mode: "character",
      name: normalizeText(selectedAssistant.name) || copy.assistantFallbackName,
      detail: copy.characterSelected,
    };
  }

  const legacyCharacter = input.selectedCharacter;
  if (legacyCharacter) {
    const legacyName =
      normalizeText(legacyCharacter.name) ||
      (legacyCharacter.id != null
        ? copy.legacyCharacterFallbackName(legacyCharacter.id)
        : null);
    if (legacyName) {
      return {
        mode: "character",
        name: legacyName,
        detail: copy.characterSelected,
      };
    }
  }

  return {
    mode: "none",
    name: null,
    detail: copy.noAssistantSelected,
  };
}

export function buildCockpitPromptSummary(input: {
  selectedSystemPrompt: string | null | undefined;
  selectedSystemPromptRecord?: {
    id?: string;
    title?: string;
    name?: string;
  } | null;
  selectedSystemPromptStatus?: "idle" | "loading" | "loaded" | "unavailable";
  selectedQuickPrompt: string | null | undefined;
  systemPrompt: string | null | undefined;
  copy?: PromptSummaryCopy;
}): PlaygroundPromptSummary {
  const copy = { ...defaultPromptCopy, ...input.copy };
  const selectedSystemPrompt = normalizeText(input.selectedSystemPrompt);
  if (selectedSystemPrompt) {
    const selectedSystemPromptRecordId = normalizeText(
      input.selectedSystemPromptRecord?.id,
    );
    const recordMatchesSelection =
      selectedSystemPromptRecordId === selectedSystemPrompt;
    const selectedPromptLabel = recordMatchesSelection
      ? normalizeText(input.selectedSystemPromptRecord?.title) ||
        normalizeText(input.selectedSystemPromptRecord?.name)
      : null;
    const selectedPromptDetail =
      input.selectedSystemPromptStatus === "loading"
        ? copy.loadingPromptDetail
        : input.selectedSystemPromptStatus === "unavailable"
          ? copy.selectedPromptUnavailableDetail
          : selectedPromptLabel
            ? copy.selectedPromptDetail
            : selectedSystemPrompt;
    return {
      state: "system",
      label: selectedPromptLabel || copy.systemPromptLabel,
      detail: selectedPromptDetail,
    };
  }

  const selectedQuickPrompt = normalizeText(input.selectedQuickPrompt);
  if (selectedQuickPrompt) {
    return {
      state: "quick",
      label: copy.quickPromptLabel,
      detail: selectedQuickPrompt,
    };
  }

  if (normalizeText(input.systemPrompt)) {
    return {
      state: "custom",
      label: copy.customPromptLabel,
      detail: copy.inlineSystemPromptActiveDetail,
    };
  }

  return {
    state: "none",
    label: copy.noPromptSelectedLabel,
    detail: copy.noPromptContextDetail,
  };
}

export function buildCockpitMcpSummary(input: {
  hasMcp: boolean;
  healthState: CockpitMcpHealthState;
  toolsLoading: boolean;
  discoveredCount: number;
  chatToolCount: number;
  disabledReason?: string;
  copy?: McpSummaryCopy;
}): RuntimeToolSummary {
  const copy = { ...defaultMcpCopy, ...input.copy };
  if (!input.hasMcp || input.healthState === "unavailable") {
    return {
      state: "unavailable",
      label: copy.unavailableLabel,
      detail: input.disabledReason || copy.unavailableDetail,
    };
  }

  if (input.healthState === "unhealthy" || input.healthState === "degraded") {
    return {
      state: "degraded",
      label: copy.toolsLabel,
      detail: input.disabledReason || copy.offlineDetail,
    };
  }

  if (input.toolsLoading) {
    return {
      state: "disabled",
      label: copy.toolsLabel,
      detail: input.disabledReason || copy.loadingDetail,
    };
  }

  if (input.chatToolCount <= 0) {
    return {
      state: "disabled",
      label: copy.toolsLabel,
      detail: input.disabledReason || copy.emptyDetail,
    };
  }

  return {
    state: "available",
    label: copy.toolsLabel,
    detail: copy.availableDetail(input.chatToolCount, input.discoveredCount),
  };
}

export function buildCockpitProviderRouteSummary(input: {
  selectedProvider: string | null | undefined;
  selectedModel: string | null | undefined;
}): {
  selectedProvider: string | null;
  selectedModel: string | null;
  providerRouteLabel: string | null;
} {
  const rawProvider = normalizeText(input.selectedProvider);
  const rawModel = normalizeText(input.selectedModel);
  if (!rawModel) {
    return {
      selectedProvider: rawProvider,
      selectedModel: null,
      providerRouteLabel: null,
    };
  }

  const providerSelection = parseProviderQualifiedModelSelection(rawModel);
  const apiModelId = providerSelection.modelId;
  const selectedProvider = providerSelection.provider || rawProvider;
  const providerRouteLabel =
    providerSelection.isProviderQualified || !selectedProvider
      ? rawModel
      : `${selectedProvider}:${apiModelId}`;

  return {
    selectedProvider,
    selectedModel: apiModelId,
    providerRouteLabel,
  };
}
