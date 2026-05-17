import { act, renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { usePlaygroundContextItems } from "../hooks/usePlaygroundContextItems";
import type { RolePlayState } from "../role-play-state";

const t = (key: string, defaultValueOrOptions?: any, options?: any) => {
  const defaultValue =
    typeof defaultValueOrOptions === "string"
      ? defaultValueOrOptions
      : defaultValueOrOptions?.defaultValue;
  const interpolation =
    typeof defaultValueOrOptions === "object" ? defaultValueOrOptions : options;
  return String(defaultValue || key).replace(/\{\{(\w+)\}\}/g, (_match, name) =>
    String(interpolation?.[name] ?? ""),
  );
};

const baseRolePlayState: RolePlayState = {
  active: true,
  identity: null,
  behavior: null,
  scene: null,
  generationStyle: null,
  context: {
    pinnedCount: 0,
    hasExternalContext: false,
  },
};

const createDeps = (
  overrides: Partial<Parameters<typeof usePlaygroundContextItems>[0]> = {},
): Parameters<typeof usePlaygroundContextItems>[0] => ({
  selectedModel: "gpt-4.1",
  modelSummaryLabel: "GPT-4.1",
  isSessionDegraded: false,
  connectionStatusLabel: "Connected",
  compareModeActive: false,
  compareSelectedModels: [],
  currentPreset: null,
  selectedCharacterName: null,
  characterPendingApply: false,
  contextToolsOpen: false,
  ragPinnedResultsLength: 0,
  webSearch: false,
  sessionUsageTotalTokens: 0,
  sessionUsageLabel: "0 tokens",
  selectedSystemPrompt: null,
  selectedQuickPrompt: null,
  systemPrompt: "",
  promptSummaryLabel: "No prompt",
  jsonMode: false,
  showTokenBudgetWarning: false,
  tokenBudgetRiskLevel: "unknown",
  tokenBudgetRiskLabel: "Unknown",
  projectedBudgetUtilizationPercent: null,
  nonMessageContextPercent: null,
  showNonMessageContextWarning: false,
  temporaryChat: false,
  openModelApiSelector: vi.fn(),
  focusConnectionCard: vi.fn(),
  setOpenModelSettings: vi.fn(),
  setOpenActorSettings: vi.fn(),
  setContextToolsOpen: vi.fn(),
  handleToggleWebSearch: vi.fn(),
  openKnowledgePanel: vi.fn(),
  openContextWindowModal: vi.fn(),
  openSessionInsightsModal: vi.fn(),
  updateChatModelSetting: vi.fn(),
  t,
  ...overrides,
});

describe("usePlaygroundContextItems role-play state", () => {
  it("labels applied behavior templates by name instead of as anonymous prompt state", () => {
    const { result } = renderHook(() =>
      usePlaygroundContextItems(
        createDeps({
          rolePlayState: {
            ...baseRolePlayState,
            behavior: {
              source: "template",
              templateId: "character-actor",
              title: "Character Actor",
              modified: false,
            },
          },
        }),
      ),
    );

    expect(result.current).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: "rolePlayBehavior",
          label: "Behavior",
          value: "Character Actor",
          tone: "active",
        }),
      ]),
    );
    expect(result.current.find((item) => item.id === "prompt")).toBeUndefined();
  });

  it("shows freeform system prompt state as a custom system prompt", () => {
    const { result } = renderHook(() =>
      usePlaygroundContextItems(
        createDeps({
          rolePlayState: {
            ...baseRolePlayState,
            behavior: {
              source: "custom",
              title: "Custom",
              modified: false,
            },
          },
        }),
      ),
    );

    expect(result.current).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: "rolePlayBehavior",
          label: "System prompt",
          value: "Custom",
        }),
      ]),
    );
  });

  it("marks edited behavior templates as modified", () => {
    const { result } = renderHook(() =>
      usePlaygroundContextItems(
        createDeps({
          rolePlayState: {
            ...baseRolePlayState,
            behavior: {
              source: "modified-template",
              templateId: "character-actor",
              title: "Character Actor",
              modified: true,
            },
          },
        }),
      ),
    );

    expect(result.current).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: "rolePlayBehavior",
          value: "Character Actor modified",
        }),
      ]),
    );
  });

  it("clears selected character identity through the existing role-play identity action", () => {
    const onClearRolePlayIdentity = vi.fn();
    const { result } = renderHook(() =>
      usePlaygroundContextItems(
        createDeps({
          onClearRolePlayIdentity,
          rolePlayState: {
            ...baseRolePlayState,
            identity: {
              kind: "character",
              id: "char-mira",
              name: "Mira",
            },
          },
        }),
      ),
    );

    act(() => {
      result.current
        .find((item) => item.id === "rolePlayIdentity")
        ?.onClick?.();
    });

    expect(onClearRolePlayIdentity).toHaveBeenCalledTimes(1);
  });

  it("resets generation style through the supplied Balanced reset handler", () => {
    const onResetRolePlayGenerationStyle = vi.fn();
    const { result } = renderHook(() =>
      usePlaygroundContextItems(
        createDeps({
          onResetRolePlayGenerationStyle,
          rolePlayState: {
            ...baseRolePlayState,
            generationStyle: {
              key: "creative",
              label: "Creative",
            },
          },
        }),
      ),
    );

    act(() => {
      result.current
        .find((item) => item.id === "rolePlayGenerationStyle")
        ?.onClick?.();
    });

    expect(onResetRolePlayGenerationStyle).toHaveBeenCalledTimes(1);
  });

  it("summarizes pinned role-play context without a source-management click action", () => {
    const { result } = renderHook(() =>
      usePlaygroundContextItems(
        createDeps({
          rolePlayState: {
            ...baseRolePlayState,
            context: {
              pinnedCount: 2,
              hasExternalContext: true,
            },
          },
        }),
      ),
    );

    expect(result.current).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: "rolePlayContext",
          label: "Context",
          value: "2 pinned + external",
        }),
      ]),
    );
    expect(
      result.current.find((item) => item.id === "rolePlayContext")?.onClick,
    ).toBeUndefined();
  });
});
