import React from "react";
import type { ComposerContextItem } from "../ComposerToolbar";
import type { KnowledgeTab } from "@/components/Knowledge";
import { toText } from "./utils";
import type { RolePlayState } from "../role-play-state";
import type { RolePlayCompatibility } from "../role-play-compatibility";

export type UsePlaygroundContextItemsDeps = {
  selectedModel: string | null | undefined;
  modelSummaryLabel: string;
  isConnectionReady: boolean;
  isSessionDegraded: boolean;
  connectionStatusLabel: string;
  compareModeActive: boolean;
  compareSelectedModels: string[];
  currentPreset: { key: string; label: string } | null | undefined;
  selectedCharacterName: string | null;
  characterPendingApply: boolean;
  contextToolsOpen: boolean;
  ragPinnedResultsLength: number;
  webSearch: boolean;
  sessionUsageTotalTokens: number;
  sessionUsageLabel: string;
  selectedSystemPrompt: string | null | undefined;
  selectedQuickPrompt: string | null | undefined;
  systemPrompt: string | null | undefined;
  promptSummaryLabel: string;
  jsonMode: boolean;
  showTokenBudgetWarning: boolean;
  tokenBudgetRiskLevel: string;
  tokenBudgetRiskLabel: string;
  projectedBudgetUtilizationPercent: number | null | undefined;
  nonMessageContextPercent: number | null | undefined;
  showNonMessageContextWarning: boolean;
  temporaryChat: boolean;
  openModelApiSelector: () => void;
  focusConnectionCard: () => void;
  setOpenModelSettings: (open: boolean) => void;
  setOpenActorSettings: (open: boolean) => void;
  setContextToolsOpen: (open: boolean) => void;
  handleToggleWebSearch: () => void;
  openKnowledgePanel: (tab: KnowledgeTab) => void;
  openContextWindowModal: () => void;
  openSessionInsightsModal: () => void;
  updateChatModelSetting: (key: string, value: any) => void;
  rolePlayState?: RolePlayState | null;
  rolePlayCompatibility?: RolePlayCompatibility | null;
  onClearRolePlayIdentity?: () => void;
  onClearRolePlayBehavior?: () => void;
  onResetRolePlayGenerationStyle?: () => void;
  onDisableCompareMode?: () => void;
  onOpenRolePlaySetup?: () => void;
  t: (key: string, defaultValueOrOptions?: any, options?: any) => string;
};

export function usePlaygroundContextItems(
  deps: UsePlaygroundContextItemsDeps,
): ComposerContextItem[] {
  const {
    selectedModel,
    modelSummaryLabel,
    isConnectionReady,
    isSessionDegraded,
    connectionStatusLabel,
    compareModeActive,
    compareSelectedModels,
    currentPreset,
    selectedCharacterName,
    characterPendingApply,
    contextToolsOpen,
    ragPinnedResultsLength,
    webSearch,
    sessionUsageTotalTokens,
    sessionUsageLabel,
    selectedSystemPrompt,
    selectedQuickPrompt,
    systemPrompt,
    promptSummaryLabel,
    jsonMode,
    showTokenBudgetWarning,
    tokenBudgetRiskLevel,
    tokenBudgetRiskLabel,
    projectedBudgetUtilizationPercent,
    nonMessageContextPercent,
    showNonMessageContextWarning,
    temporaryChat,
    openModelApiSelector,
    focusConnectionCard,
    setOpenModelSettings,
    setOpenActorSettings,
    setContextToolsOpen,
    handleToggleWebSearch,
    openKnowledgePanel,
    openContextWindowModal,
    openSessionInsightsModal,
    updateChatModelSetting,
    rolePlayState,
    rolePlayCompatibility,
    onClearRolePlayIdentity,
    onClearRolePlayBehavior,
    onResetRolePlayGenerationStyle,
    onDisableCompareMode,
    onOpenRolePlaySetup,
    t,
  } = deps;

  return React.useMemo<ComposerContextItem[]>(() => {
    const items: ComposerContextItem[] = [];
    const hasRolePlayState = Boolean(rolePlayState?.active);
    items.push({
      id: "model",
      label: t("playground:composer.context.model", "Model"),
      value: selectedModel ? modelSummaryLabel : t("common:none", "None"),
      tone: selectedModel ? "active" : "warning",
      onClick: openModelApiSelector,
    });
    if (isConnectionReady && isSessionDegraded) {
      items.push({
        id: "sessionStatus",
        label: t("playground:composer.context.sessionStatus", "Session status"),
        value: connectionStatusLabel,
        tone: "warning",
        onClick: focusConnectionCard,
      });
    }

    if (compareModeActive) {
      items.push({
        id: "compare",
        label: t("playground:composer.context.compare", "Compare"),
        value:
          compareSelectedModels.length > 0
            ? String(
                t("playground:composer.context.compareCount", {
                  defaultValue: "{{count}} models",
                  count: compareSelectedModels.length,
                } as any),
              )
            : String(t("playground:composer.context.compareOn", "On")),
        tone: "active",
        onClick: () => setOpenModelSettings(true),
      });
    }

    if (hasRolePlayState && rolePlayState?.identity) {
      const identityLabelKey =
        rolePlayState.identity.kind === "persona"
          ? "playground:composer.context.persona"
          : rolePlayState.identity.kind === "assistant"
            ? "playground:composer.context.assistant"
            : "playground:composer.context.character";
      const identityLabelFallback =
        rolePlayState.identity.kind === "persona"
          ? "Persona"
          : rolePlayState.identity.kind === "assistant"
            ? "Assistant"
            : "Character";
      const identityValue = characterPendingApply
        ? toText(
            t(
              "playground:composer.context.characterNextTurn",
              "{{name}} (next turn)",
              {
                name: rolePlayState.identity.name || identityLabelFallback,
              } as any,
            ),
          )
        : rolePlayState.identity.name ||
          rolePlayState.identity.id ||
          identityLabelFallback;
      items.push({
        id: "rolePlayIdentity",
        label: toText(t(identityLabelKey, identityLabelFallback)),
        value: identityValue,
        tone: "active",
        onClick: onClearRolePlayIdentity ?? (() => setOpenActorSettings(true)),
      });
    }

    if (
      hasRolePlayState &&
      rolePlayState?.identity &&
      rolePlayCompatibility &&
      rolePlayCompatibility.status !== "none"
    ) {
      const isPersona = rolePlayState.identity.kind === "persona";
      const label = isPersona
        ? toText(
            t(
              "playground:composer.context.personaContext",
              "Persona context",
            ),
          )
        : toText(
            t(
              "playground:composer.context.characterContext",
              "Character context",
            ),
          );
      const valueByStatus: Record<
        Exclude<RolePlayCompatibility["status"], "none">,
        string
      > = {
        included: toText(
          t("playground:composer.rolePlayCompatibility.included", "Included"),
        ),
        blended: toText(
          t(
            "playground:composer.rolePlayCompatibility.blended",
            "Blended with sources",
          ),
        ),
        excluded: toText(
          t(
            "playground:composer.rolePlayCompatibility.excluded",
            "Excluded in this mode",
          ),
        ),
        "override-risk": toText(
          t(
            "playground:composer.rolePlayCompatibility.overrideRisk",
            "Prompt override risk",
          ),
        ),
      };
      const actionByReason: Partial<
        Record<RolePlayCompatibility["reasonCode"], () => void>
      > = {
        custom_prompt: onClearRolePlayBehavior,
        rag_sources: () => openKnowledgePanel("search"),
        compare_mode: onDisableCompareMode ?? (() => setOpenModelSettings(true)),
        context_files: openContextWindowModal,
        documents: openContextWindowModal,
        image_command: onOpenRolePlaySetup,
      };
      items.push({
        id: "rolePlayCompatibility",
        label,
        value: valueByStatus[rolePlayCompatibility.status],
        tone:
          rolePlayCompatibility.status === "included" ? "active" : "warning",
        ...(actionByReason[rolePlayCompatibility.reasonCode]
          ? { onClick: actionByReason[rolePlayCompatibility.reasonCode] }
          : {}),
      });
    }

    if (hasRolePlayState && rolePlayState?.behavior) {
      const isCustomBehavior = rolePlayState.behavior.source === "custom";
      const behaviorTitle =
        rolePlayState.behavior.title ||
        (isCustomBehavior
          ? toText(
              t("playground:composer.context.customSystemPrompt", "Custom"),
            )
          : toText(t("playground:composer.context.behavior", "Behavior")));
      const behaviorValue = rolePlayState.behavior.modified
        ? toText(
            t(
              "playground:composer.context.modifiedBehaviorTemplate",
              "{{title}} modified",
              { title: behaviorTitle } as any,
            ),
          )
        : behaviorTitle;
      items.push({
        id: "rolePlayBehavior",
        label: toText(
          t(
            isCustomBehavior
              ? "playground:composer.context.systemPrompt"
              : "playground:composer.context.behavior",
            isCustomBehavior ? "System prompt" : "Behavior",
          ),
        ),
        value: behaviorValue,
        tone: "active",
        ...(onClearRolePlayBehavior
          ? { onClick: onClearRolePlayBehavior }
          : {}),
      });
    }

    if (hasRolePlayState && rolePlayState?.scene) {
      items.push({
        id: "rolePlayScene",
        label: toText(t("playground:composer.context.scene", "Scene")),
        value:
          rolePlayState.scene.summary ||
          toText(t("playground:composer.context.sceneActive", "Active")),
        tone: "active",
        onClick: () => setOpenActorSettings(true),
      });
    }

    if (hasRolePlayState && rolePlayState?.generationStyle) {
      items.push({
        id: "rolePlayGenerationStyle",
        label: toText(
          t("playground:composer.context.generationStyle", "Generation style"),
        ),
        value: rolePlayState.generationStyle.label,
        tone: "active",
        ...(onResetRolePlayGenerationStyle
          ? { onClick: onResetRolePlayGenerationStyle }
          : {}),
      });
    }

    if (
      hasRolePlayState &&
      rolePlayState?.context &&
      (rolePlayState.context.pinnedCount > 0 ||
        rolePlayState.context.hasExternalContext)
    ) {
      const { pinnedCount, hasExternalContext } = rolePlayState.context;
      const contextValue =
        pinnedCount > 0 && hasExternalContext
          ? toText(
              t("playground:composer.context.rolePlayPinnedExternal", {
                defaultValue: "{{count}} pinned + external",
                count: pinnedCount,
              } as any),
            )
          : pinnedCount > 0
            ? toText(
                t("playground:composer.context.rolePlayPinnedCount", {
                  defaultValue: "{{count}} pinned",
                  count: pinnedCount,
                } as any),
              )
            : toText(
                t("playground:composer.context.rolePlayExternal", "External"),
              );
      items.push({
        id: "rolePlayContext",
        label: toText(t("playground:composer.context.context", "Context")),
        value: contextValue,
        tone: "active",
      });
    }

    if (!hasRolePlayState && currentPreset && currentPreset.key !== "custom") {
      items.push({
        id: "preset",
        label: toText(t("playground:composer.context.preset", "Preset")),
        value: toText(
          t(
            `playground:presets.${currentPreset.key}.label`,
            currentPreset.label,
          ),
        ),
        tone: "active",
        onClick: () => setOpenModelSettings(true),
      });
    }

    if (!hasRolePlayState && selectedCharacterName) {
      items.push({
        id: "character",
        label: toText(t("playground:composer.context.character", "Character")),
        value: characterPendingApply
          ? toText(
              t(
                "playground:composer.context.characterNextTurn",
                "{{name}} (next turn)",
                { name: selectedCharacterName } as any,
              ),
            )
          : selectedCharacterName,
        tone: "active",
        onClick: () => setOpenActorSettings(true),
      });
    }

    if (contextToolsOpen) {
      items.push({
        id: "knowledge",
        label: toText(t("playground:composer.context.knowledge", "Knowledge")),
        value: toText(t("common:open", "Open")),
        tone: "active",
        onClick: () => setContextToolsOpen(false),
      });
    }

    if (!hasRolePlayState && ragPinnedResultsLength > 0) {
      items.push({
        id: "ragPinned",
        label: toText(t("playground:composer.context.pinnedSources", "Pinned")),
        value: toText(
          t("playground:composer.context.pinnedCount", {
            defaultValue: "{{count}} sources",
            count: ragPinnedResultsLength,
          } as any),
        ),
        tone: "active",
        onClick: () => openKnowledgePanel("search"),
      });
    }

    if (webSearch) {
      items.push({
        id: "webSearch",
        label: toText(t("playground:composer.context.webSearch", "Web search")),
        value: toText(t("common:on", "On")),
        tone: "active",
        onClick: handleToggleWebSearch,
      });
    }
    if (sessionUsageTotalTokens > 0) {
      items.push({
        id: "sessionUsage",
        label: toText(t("playground:composer.context.session", "Session")),
        value: sessionUsageLabel,
        tone: "neutral",
        onClick: openSessionInsightsModal,
      });
    }
    if (
      !hasRolePlayState &&
      (selectedSystemPrompt ||
        selectedQuickPrompt ||
        String(systemPrompt || "").trim().length > 0)
    ) {
      items.push({
        id: "prompt",
        label: toText(t("playground:composer.context.prompt", "Prompt")),
        value: promptSummaryLabel,
        tone: "active",
      });
    }

    if (jsonMode) {
      items.push({
        id: "json",
        label: toText(t("playground:composer.context.json", "JSON mode")),
        value: toText(
          t("playground:composer.context.jsonShort", "Object responses"),
        ),
        tone: "active",
        onClick: () => updateChatModelSetting("jsonMode", undefined),
      });
    }

    if (showTokenBudgetWarning) {
      items.push({
        id: "budget",
        label: toText(t("playground:composer.context.budget", "Budget")),
        value: `${tokenBudgetRiskLabel}${
          projectedBudgetUtilizationPercent != null
            ? ` • ${Math.round(projectedBudgetUtilizationPercent)}%`
            : ""
        }`,
        tone: "warning",
        onClick: openContextWindowModal,
      });
    }
    if (tokenBudgetRiskLevel !== "unknown" && !showTokenBudgetWarning) {
      items.push({
        id: "truncationRisk",
        label: toText(
          t("playground:composer.context.truncationRisk", "Truncation"),
        ),
        value: tokenBudgetRiskLabel,
        tone:
          tokenBudgetRiskLevel === "high" || tokenBudgetRiskLevel === "critical"
            ? "warning"
            : "neutral",
        onClick: openContextWindowModal,
      });
    }
    if (nonMessageContextPercent != null) {
      items.push({
        id: "contextMix",
        label: toText(
          t("playground:composer.context.contextMix", "Context mix"),
        ),
        value: toText(
          t(
            "playground:composer.context.nonMessageShare",
            "{{percent}}% non-message",
            {
              percent: Math.max(0, Math.round(nonMessageContextPercent)),
            } as any,
          ),
        ),
        tone: showNonMessageContextWarning ? "warning" : "neutral",
        onClick: openContextWindowModal,
      });
    }

    if (temporaryChat) {
      items.push({
        id: "temporary",
        label: toText(t("playground:composer.context.temporary", "Temporary")),
        value: toText(t("playground:composer.context.notSaved", "Not saved")),
        tone: "warning",
      });
    }

    return items;
  }, [
    compareModeActive,
    compareSelectedModels.length,
    connectionStatusLabel,
    contextToolsOpen,
    currentPreset,
    rolePlayState,
    rolePlayCompatibility,
    jsonMode,
    focusConnectionCard,
    handleToggleWebSearch,
    isConnectionReady,
    isSessionDegraded,
    modelSummaryLabel,
    openModelApiSelector,
    openKnowledgePanel,
    openContextWindowModal,
    nonMessageContextPercent,
    characterPendingApply,
    promptSummaryLabel,
    tokenBudgetRiskLevel,
    tokenBudgetRiskLabel,
    ragPinnedResultsLength,
    selectedCharacterName,
    selectedModel,
    selectedQuickPrompt,
    selectedSystemPrompt,
    sessionUsageLabel,
    sessionUsageTotalTokens,
    openSessionInsightsModal,
    setContextToolsOpen,
    setOpenModelSettings,
    setOpenActorSettings,
    showTokenBudgetWarning,
    projectedBudgetUtilizationPercent,
    showNonMessageContextWarning,
    systemPrompt,
    t,
    temporaryChat,
    updateChatModelSetting,
    onClearRolePlayIdentity,
    onClearRolePlayBehavior,
    onResetRolePlayGenerationStyle,
    onDisableCompareMode,
    onOpenRolePlaySetup,
  ]);
}
