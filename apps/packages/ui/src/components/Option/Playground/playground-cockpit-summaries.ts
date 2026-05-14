import type { AssistantSelection } from "@/types/assistant-selection";
import type { PlaygroundPromptSummary } from "./PlaygroundContextRail";
import type {
  RuntimeAssistantSummary,
  RuntimeToolSummary,
} from "./PlaygroundRuntimeInspector";

type LegacyCharacterSelection = {
  id?: string | number;
  name?: string | null;
};

type PersonaMemoryMode = "read_only" | "read_write";

const normalizeText = (value: string | null | undefined): string | null => {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
};

const formatPersonaMemoryMode = (
  personaMemoryMode: PersonaMemoryMode | null | undefined,
): string | null => {
  if (personaMemoryMode === "read_only") return "memory read-only";
  if (personaMemoryMode === "read_write") return "memory read/write";
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
}): RuntimeAssistantSummary {
  const selectedAssistant = input.selectedAssistant;
  if (selectedAssistant?.kind === "persona") {
    const memoryMode = formatPersonaMemoryMode(input.personaMemoryMode);
    return {
      mode: "persona",
      name: normalizeText(selectedAssistant.name) || "Persona",
      detail: memoryMode ? `Persona selected - ${memoryMode}` : "Persona selected",
    };
  }

  if (selectedAssistant?.kind === "character") {
    return {
      mode: "character",
      name: normalizeText(selectedAssistant.name) || "Assistant",
      detail: "Character selected",
    };
  }

  const legacyCharacter = input.selectedCharacter;
  if (legacyCharacter) {
    const legacyName =
      normalizeText(legacyCharacter.name) ||
      (legacyCharacter.id != null ? `Character ${legacyCharacter.id}` : null);
    if (legacyName) {
      return {
        mode: "character",
        name: legacyName,
        detail: "Character selected",
      };
    }
  }

  return {
    mode: "none",
    name: null,
    detail: "No assistant selected",
  };
}

export function buildCockpitPromptSummary(input: {
  selectedSystemPrompt: string | null | undefined;
  selectedSystemPromptRecord?: {
    id?: string;
    title?: string;
    name?: string;
  } | null;
  selectedQuickPrompt: string | null | undefined;
  systemPrompt: string | null | undefined;
}): PlaygroundPromptSummary {
  const selectedSystemPrompt = normalizeText(input.selectedSystemPrompt);
  if (selectedSystemPrompt) {
    const selectedPromptLabel =
      normalizeText(input.selectedSystemPromptRecord?.title) ||
      normalizeText(input.selectedSystemPromptRecord?.name);
    return {
      state: "system",
      label: selectedPromptLabel || "System prompt",
      detail: selectedPromptLabel ? "System prompt" : selectedSystemPrompt,
    };
  }

  const selectedQuickPrompt = normalizeText(input.selectedQuickPrompt);
  if (selectedQuickPrompt) {
    return {
      state: "quick",
      label: "Quick prompt",
      detail: selectedQuickPrompt,
    };
  }

  if (normalizeText(input.systemPrompt)) {
    return {
      state: "custom",
      label: "Custom prompt",
      detail: "Inline system prompt active",
    };
  }

  return {
    state: "none",
    label: "No prompt selected",
    detail: "No prompt context will be added.",
  };
}

export function buildCockpitMcpSummary(input: {
  hasMcp: boolean;
  healthState: string;
  toolsLoading: boolean;
  discoveredCount: number;
  chatToolCount: number;
  disabledReason?: string;
}): RuntimeToolSummary {
  if (!input.hasMcp || input.healthState === "unavailable") {
    return {
      state: "unavailable",
      label: "MCP unavailable",
      detail: input.disabledReason || "MCP tools unavailable",
    };
  }

  if (input.toolsLoading) {
    return {
      state: "disabled",
      label: "MCP tools",
      detail: input.disabledReason || "Loading tools...",
    };
  }

  if (input.healthState === "unhealthy" || input.healthState === "degraded") {
    return {
      state: "degraded",
      label: "MCP tools",
      detail: input.disabledReason || "MCP tools are offline",
    };
  }

  if (input.chatToolCount <= 0) {
    return {
      state: "disabled",
      label: "MCP tools",
      detail: input.disabledReason || "No MCP tools available",
    };
  }

  const chatToolsLabel = formatCountLabel(
    input.chatToolCount,
    "chat tool",
    "chat tools",
  );
  const discoveredSuffix =
    input.discoveredCount > input.chatToolCount
      ? ` (${input.discoveredCount} discovered)`
      : "";

  return {
    state: "available",
    label: "MCP tools",
    detail: `${chatToolsLabel} available${discoveredSuffix}`,
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

  const providerSeparator = rawModel.indexOf(":");
  const providerFromModel =
    providerSeparator > 0 ? rawModel.slice(0, providerSeparator) : null;
  const apiModelId =
    providerFromModel && providerSeparator < rawModel.length - 1
      ? rawModel.slice(providerSeparator + 1)
      : rawModel;
  const selectedProvider = providerFromModel || rawProvider;
  const providerRouteLabel =
    providerFromModel || !selectedProvider
      ? rawModel
      : `${selectedProvider}:${apiModelId}`;

  return {
    selectedProvider,
    selectedModel: apiModelId,
    providerRouteLabel,
  };
}
