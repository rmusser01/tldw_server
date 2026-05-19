import { describe, expect, it } from "vitest";

import {
  buildPlaygroundCompositionPreviewSummary,
  type PlaygroundCompositionPreviewInput,
} from "../playground-composition-preview";

const baseInput = (
  overrides: Partial<PlaygroundCompositionPreviewInput> = {},
): PlaygroundCompositionPreviewInput => ({
  promptSummary: {
    state: "none",
    label: "No prompt selected",
    detail: "No prompt context will be added.",
  },
  assistantSummary: {
    mode: "none",
    name: null,
    detail: "No assistant selected",
  },
  providerRoute: {
    selectedProvider: null,
    selectedModel: null,
    providerRouteLabel: null,
  },
  settingSummaries: [],
  contextSources: [],
  toolSummary: null,
  compositionStatus: "idle",
  composition: null,
  ...overrides,
});

describe("playground composition preview summary", () => {
  it("keeps the empty setup explicit instead of hiding missing inputs", () => {
    const summary = buildPlaygroundCompositionPreviewSummary(baseInput());

    expect(summary.overallState).toBe("unavailable");
    expect(summary.entries.map((entry) => [entry.kind, entry.state, entry.title])).toEqual([
      ["prompt", "disabled", "No prompt selected"],
      ["assistant", "disabled", "No assistant selected"],
      ["model", "unavailable", "No model selected"],
      ["settings", "disabled", "Default model settings"],
      ["context", "disabled", "No extra context"],
      ["tools", "disabled", "MCP tools managed from composer"],
    ]);
    expect(summary.settingsScopeLabel).toBeNull();
    expect(summary.contextStack).toEqual([]);
  });

  it("summarizes prompt, persona, provider:model scope, settings, context, and tools together", () => {
    const summary = buildPlaygroundCompositionPreviewSummary(
      baseInput({
        promptSummary: {
          state: "system",
          label: "Research brief",
          detail: "System prompt",
        },
        assistantSummary: {
          mode: "persona",
          name: "Research Persona",
          detail: "Persona selected - memory read/write",
        },
        providerRoute: {
          selectedProvider: "openai",
          selectedModel: "gpt-4.1-mini",
          providerRouteLabel: "openai:gpt-4.1-mini",
        },
        settingSummaries: [
          {
            label: "Temperature",
            value: "0.31",
            source: "override",
          },
        ],
        contextSources: [
          {
            id: "source-knowledge-1",
            kind: "knowledge",
            label: "Knowledge",
            title: "Launch notes",
            detail: "2 snippets",
            state: "active",
          },
        ],
        toolSummary: {
          state: "available",
          label: "MCP tools",
          detail: "2 chat tools available",
        },
        compositionStatus: "ready",
        composition: {
          selection: {
            worldBookIds: [10],
            dictionaryIds: [20],
            providerId: "openai",
            modelId: "gpt-4.1-mini",
          },
          inputText: "Draft",
          transformedInputText: "Draft",
          pieces: [
            {
              kind: "worldbook",
              id: 10,
              name: "Launch worldbook",
              source: "request",
              status: "active",
            },
          ],
          previewSections: [
            {
              name: "worldbook",
              content: "Launch notes context",
              source: "request",
            },
          ],
          providerMessages: [
            {
              role: "system",
              content: "Launch notes context",
            },
          ],
          readiness: "ready",
          warnings: [],
        },
      }),
    );

    expect(summary.overallState).toBe("ready");
    expect(summary.settingsScopeLabel).toBe("openai:gpt-4.1-mini");
    expect(summary.entries.map((entry) => [entry.kind, entry.state, entry.title])).toEqual([
      ["prompt", "active", "Research brief"],
      ["assistant", "active", "Research Persona"],
      ["model", "active", "openai:gpt-4.1-mini"],
      ["settings", "active", "openai:gpt-4.1-mini"],
      ["context", "active", "1 active source"],
      ["tools", "active", "MCP tools"],
    ]);
    expect(summary.contextStack.map((entry) => [entry.kind, entry.state, entry.title])).toEqual([
      ["prompt", "active", "Research brief"],
      ["assistant", "active", "Research Persona"],
      ["knowledge", "active", "Launch notes"],
      ["tools", "active", "MCP tools"],
    ]);
    expect(summary.footprint).toEqual({
      providerMessageCount: 1,
      previewSectionCount: 1,
      contextPieceCount: 1,
      warningCount: 0,
      readiness: "ready",
    });
  });

  it("marks a selected model unavailable when readiness rejects the route", () => {
    const summary = buildPlaygroundCompositionPreviewSummary(
      baseInput({
        providerRoute: {
          selectedProvider: "tldw",
          selectedModel: "gpt-4o",
          providerRouteLabel: "tldw:gpt-4o",
        },
        modelUnavailable: true,
        modelUnavailableDetail: "Choose a chat model before chatting as Ada",
      }),
    );

    expect(summary.overallState).toBe("unavailable");
    expect(summary.entries.find((entry) => entry.kind === "model")).toMatchObject({
      state: "unavailable",
      title: "tldw:gpt-4o",
      detail: "Choose a chat model before chatting as Ada",
    });
  });

  it("keeps MCP unavailable distinct from empty context and preserves character context", () => {
    const summary = buildPlaygroundCompositionPreviewSummary(
      baseInput({
        assistantSummary: {
          mode: "character",
          name: "Ada",
          detail: "Character selected",
        },
        providerRoute: {
          selectedProvider: "ollama",
          selectedModel: "llama3:latest",
          providerRouteLabel: "ollama:llama3:latest",
        },
        contextSources: [
          {
            id: "assistant-ada",
            kind: "assistant",
            label: "Character",
            title: "Ada",
            detail: "Character selected",
            state: "active",
          },
        ],
        toolSummary: {
          state: "unavailable",
          label: "MCP unavailable",
          detail: "MCP tools unavailable",
        },
      }),
    );

    expect(summary.overallState).toBe("degraded");
    expect(summary.entries.find((entry) => entry.kind === "tools")).toMatchObject({
      state: "unavailable",
      title: "MCP unavailable",
      detail: "MCP tools unavailable",
    });
    expect(summary.contextStack.map((entry) => [entry.kind, entry.state, entry.title])).toEqual([
      ["assistant", "active", "Ada"],
      ["tools", "unavailable", "MCP unavailable"],
    ]);
  });

  it("does not count degraded context or tools as ready success", () => {
    const summary = buildPlaygroundCompositionPreviewSummary(
      baseInput({
        providerRoute: {
          selectedProvider: "openai",
          selectedModel: "gpt-4.1-mini",
          providerRouteLabel: "openai:gpt-4.1-mini",
        },
        contextSources: [
          {
            id: "research-failed",
            kind: "research",
            label: "Research",
            title: "Prior research",
            detail: "Refresh failed",
            state: "degraded",
          },
          {
            id: "file-disabled",
            kind: "file",
            label: "File",
            title: "notes.txt",
            detail: "Disabled",
            state: "disabled",
          },
        ],
        toolSummary: {
          state: "degraded",
          label: "MCP tools",
          detail: "MCP tools are offline",
        },
        compositionStatus: "error",
      }),
    );

    expect(summary.overallState).toBe("degraded");
    expect(summary.entries.find((entry) => entry.kind === "context")).toMatchObject({
      state: "degraded",
      title: "1 active source",
    });
    expect(summary.entries.find((entry) => entry.kind === "tools")).toMatchObject({
      state: "degraded",
      title: "MCP tools",
    });
    expect(summary.entries.find((entry) => entry.kind === "composition")).toMatchObject({
      state: "unavailable",
      title: "Context preview unavailable",
    });
  });

  it("keeps an all-disabled context stack disabled", () => {
    const summary = buildPlaygroundCompositionPreviewSummary(
      baseInput({
        providerRoute: {
          selectedProvider: "openai",
          selectedModel: "gpt-4.1-mini",
          providerRouteLabel: "openai:gpt-4.1-mini",
        },
        contextSources: [
          {
            id: "web-search",
            kind: "web",
            label: "Web",
            title: "Web search",
            detail: "Disabled for the next reply.",
            state: "disabled",
          },
        ],
      }),
    );

    expect(summary.entries.find((entry) => entry.kind === "context")).toMatchObject({
      state: "disabled",
      title: "No extra context",
      detail: "1 configured source",
    });
    expect(summary.overallState).toBe("ready");
  });
});
