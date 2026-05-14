import { describe, expect, it } from "vitest";

import {
  buildCockpitAssistantSummary,
  buildCockpitMcpSummary,
  buildCockpitPromptSummary,
  buildCockpitProviderRouteSummary,
} from "../playground-cockpit-summaries";
import type { AssistantSelection } from "@/types/assistant-selection";

describe("playground cockpit summary helpers", () => {
  it("prefers selectedAssistant character over legacy selectedCharacter", () => {
    const selectedAssistant: AssistantSelection = {
      kind: "character",
      id: "assistant-character",
      name: "New Character",
    };

    expect(
      buildCockpitAssistantSummary({
        selectedAssistant,
        selectedCharacter: { id: "legacy-character", name: "Legacy Character" },
      }),
    ).toEqual({
      mode: "character",
      name: "New Character",
      detail: "Character selected",
    });
  });

  it("hydrates a legacy character summary when selectedAssistant is null", () => {
    expect(
      buildCockpitAssistantSummary({
        selectedAssistant: null,
        selectedCharacter: { id: 42, name: "Legacy Character" },
      }),
    ).toEqual({
      mode: "character",
      name: "Legacy Character",
      detail: "Character selected",
    });
  });

  it("includes persona mode without exposing Scene Director state", () => {
    const selectedAssistant: AssistantSelection = {
      kind: "persona",
      id: "persona-1",
      name: "Research Persona",
    };

    const summary = buildCockpitAssistantSummary({
      selectedAssistant,
      selectedCharacter: { id: "legacy-character", name: "Legacy Character" },
      personaMemoryMode: "read_write",
    });

    expect(summary).toEqual({
      mode: "persona",
      name: "Research Persona",
      detail: "Persona selected - memory read/write",
    });
    expect(summary).not.toHaveProperty("sceneDirector");
    expect(summary.detail).not.toMatch(/scene director/i);
  });

  it("prefers selected prompt title or name over the raw id when a record exists", () => {
    expect(
      buildCockpitPromptSummary({
        selectedSystemPrompt: "prompt-123",
        selectedSystemPromptRecord: {
          id: "prompt-123",
          title: "Research Brief",
        },
        selectedQuickPrompt: null,
        systemPrompt: null,
      }),
    ).toEqual({
      state: "system",
      label: "Research Brief",
      detail: "System prompt",
    });

    expect(
      buildCockpitPromptSummary({
        selectedSystemPrompt: "prompt-456",
        selectedSystemPromptRecord: {
          id: "prompt-456",
          name: "Fallback Name",
        },
        selectedQuickPrompt: null,
        systemPrompt: null,
      }).label,
    ).toBe("Fallback Name");
  });

  it("ignores stale selected prompt records when the record id no longer matches", () => {
    expect(
      buildCockpitPromptSummary({
        selectedSystemPrompt: "prompt-next",
        selectedSystemPromptRecord: {
          id: "prompt-previous",
          title: "Stale Prompt Title",
        },
        selectedQuickPrompt: null,
        systemPrompt: null,
      }),
    ).toEqual({
      state: "system",
      label: "System prompt",
      detail: "prompt-next",
    });
  });

  it("keeps inline custom prompt distinct from quick and selected prompts", () => {
    expect(
      buildCockpitPromptSummary({
        selectedSystemPrompt: null,
        selectedSystemPromptRecord: null,
        selectedQuickPrompt: "Summarize this",
        systemPrompt: "You are terse.",
      }),
    ).toEqual({
      state: "quick",
      label: "Quick prompt",
      detail: "Summarize this",
    });

    expect(
      buildCockpitPromptSummary({
        selectedSystemPrompt: null,
        selectedSystemPromptRecord: null,
        selectedQuickPrompt: null,
        systemPrompt: "You are terse.",
      }),
    ).toEqual({
      state: "custom",
      label: "Custom prompt",
      detail: "Inline system prompt active",
    });
  });

  it("represents MCP unavailable, loading, degraded, empty, and available states", () => {
    expect(
      buildCockpitMcpSummary({
        hasMcp: false,
        healthState: "unavailable",
        toolsLoading: false,
        discoveredCount: 0,
        chatToolCount: 0,
      }),
    ).toEqual({
      state: "unavailable",
      label: "MCP unavailable",
      detail: "MCP tools unavailable",
    });

    expect(
      buildCockpitMcpSummary({
        hasMcp: true,
        healthState: "unknown",
        toolsLoading: true,
        discoveredCount: 0,
        chatToolCount: 0,
      }),
    ).toEqual({
      state: "disabled",
      label: "MCP tools",
      detail: "Loading tools...",
    });

    expect(
      buildCockpitMcpSummary({
        hasMcp: true,
        healthState: "unhealthy",
        toolsLoading: false,
        discoveredCount: 2,
        chatToolCount: 1,
      }),
    ).toEqual({
      state: "degraded",
      label: "MCP tools",
      detail: "MCP tools are offline",
    });

    expect(
      buildCockpitMcpSummary({
        hasMcp: true,
        healthState: "unhealthy",
        toolsLoading: true,
        discoveredCount: 2,
        chatToolCount: 1,
      }),
    ).toEqual({
      state: "degraded",
      label: "MCP tools",
      detail: "MCP tools are offline",
    });

    expect(
      buildCockpitMcpSummary({
        hasMcp: true,
        healthState: "degraded",
        toolsLoading: true,
        discoveredCount: 2,
        chatToolCount: 1,
      }),
    ).toEqual({
      state: "degraded",
      label: "MCP tools",
      detail: "MCP tools are offline",
    });

    expect(
      buildCockpitMcpSummary({
        hasMcp: true,
        healthState: "healthy",
        toolsLoading: false,
        discoveredCount: 0,
        chatToolCount: 0,
      }),
    ).toEqual({
      state: "disabled",
      label: "MCP tools",
      detail: "No MCP tools available",
    });

    expect(
      buildCockpitMcpSummary({
        hasMcp: true,
        healthState: "healthy",
        toolsLoading: false,
        discoveredCount: 3,
        chatToolCount: 2,
      }),
    ).toEqual({
      state: "available",
      label: "MCP tools",
      detail: "2 chat tools available (3 discovered)",
    });
  });

  it("uses caller-provided copy without coupling helpers to UI translation", () => {
    const selectedAssistant: AssistantSelection = {
      kind: "persona",
      id: "persona-1",
      name: "Research Persona",
    };

    expect(
      buildCockpitAssistantSummary({
        selectedAssistant,
        selectedCharacter: null,
        personaMemoryMode: "read_only",
        copy: {
          personaSelected: "TR persona selected",
          memoryReadOnly: "TR memory read-only",
          personaSelectedWithMemoryMode: (mode) => `TR persona selected (${mode})`,
        },
      }).detail,
    ).toBe("TR persona selected (TR memory read-only)");

    expect(
      buildCockpitPromptSummary({
        selectedSystemPrompt: "prompt-123",
        selectedSystemPromptRecord: {
          id: "prompt-123",
          title: "Research Brief",
        },
        selectedQuickPrompt: null,
        systemPrompt: null,
        copy: {
          selectedPromptDetail: "TR selected prompt",
        },
      }),
    ).toEqual({
      state: "system",
      label: "Research Brief",
      detail: "TR selected prompt",
    });

    expect(
      buildCockpitMcpSummary({
        hasMcp: true,
        healthState: "healthy",
        toolsLoading: false,
        discoveredCount: 4,
        chatToolCount: 2,
        copy: {
          toolsLabel: "TR MCP tools",
          availableDetail: (chatToolCount, discoveredCount) =>
            `TR ${chatToolCount}/${discoveredCount} tools`,
        },
      }),
    ).toEqual({
      state: "available",
      label: "TR MCP tools",
      detail: "TR 2/4 tools",
    });
  });

  it("keeps provider-qualified route distinct from the API model id", () => {
    expect(
      buildCockpitProviderRouteSummary({
        selectedProvider: "openai",
        selectedModel: "openai:gpt-4.1-mini",
      }),
    ).toEqual({
      selectedProvider: "openai",
      selectedModel: "gpt-4.1-mini",
      providerRouteLabel: "openai:gpt-4.1-mini",
    });

    expect(
      buildCockpitProviderRouteSummary({
        selectedProvider: "openai",
        selectedModel: "gpt-4.1-mini",
      }),
    ).toEqual({
      selectedProvider: "openai",
      selectedModel: "gpt-4.1-mini",
      providerRouteLabel: "openai:gpt-4.1-mini",
    });
  });
});
