import React from "react";
import {
  createStartupTemplateBundle,
  describeStartupTemplatePrompt,
  inferStartupTemplatePromptSource,
  parseStartupTemplateBundles,
  removeStartupTemplateBundle,
  resolveStartupTemplatePrompt,
  sanitizeStartupTemplateName,
  serializeStartupTemplateBundles,
  upsertStartupTemplateBundle,
  type StartupTemplateBundle,
  type StartupTemplateRolePlayMetadata,
} from "../startup-template-bundles";
import { detectCurrentPreset, getPresetByKey } from "../ParameterPresets";
import type { PromptTemplate } from "../SystemPromptTemplates";
import type { Prompt } from "@/db/dexie/types";
import type { ChatModelSettings } from "@/store/model";
import type { Character } from "@/types/character";

// ---------------------------------------------------------------------------
// Deps interface
// ---------------------------------------------------------------------------

export interface UsePromptTemplatesDeps {
  /** Startup templates raw string from storage */
  startupTemplatesRaw: string;
  setStartupTemplatesRaw: (value: string) => void;
  /** Prompt library from query */
  promptLibrary: Prompt[];
  /** Current state snapshots */
  selectedModel: string | undefined | null;
  systemPrompt: string | undefined | null;
  selectedSystemPrompt: string | undefined | null;
  selectedQuickPrompt: string | undefined | null;
  selectedCharacter: any | null;
  ragPinnedResults: any[];
  currentChatModelSettings: Record<string, any>;
  /** Setters for applying templates */
  setSelectedModel: (model: string) => void;
  setSelectedSystemPrompt: (id: string | undefined) => void;
  setSelectedQuickPrompt: (prompt: string | null) => void;
  setSystemPrompt: (prompt: string) => void;
  setSelectedCharacter: (character: any) => void;
  setRagPinnedResults: (results: any[]) => void;
  updateChatModelSettings: (settings: Partial<ChatModelSettings>) => void;
  /** Compare mode (needed when applying template to sync model selection) */
  compareModeActive: boolean;
  setCompareSelectedModels: (
    models: string[] | ((prev: string[]) => string[]),
  ) => void;
  /** Mode announcement */
  setModeAnnouncement: (msg: string | null) => void;
  /** i18n */
  t: (key: string, ...args: any[]) => string;
}

export type SaveRolePlaySetupInput = {
  name: string;
  rolePlay: StartupTemplateRolePlayMetadata;
};

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function usePromptTemplates(deps: UsePromptTemplatesDeps) {
  const {
    startupTemplatesRaw,
    setStartupTemplatesRaw,
    promptLibrary,
    selectedModel,
    systemPrompt,
    selectedSystemPrompt,
    selectedQuickPrompt,
    selectedCharacter,
    ragPinnedResults,
    currentChatModelSettings,
    setSelectedModel,
    setSelectedSystemPrompt,
    setSelectedQuickPrompt,
    setSystemPrompt,
    setSelectedCharacter,
    setRagPinnedResults,
    updateChatModelSettings,
    compareModeActive,
    setCompareSelectedModels,
    setModeAnnouncement,
    t,
  } = deps;

  const [startupTemplateDraftName, setStartupTemplateDraftName] =
    React.useState("");
  const [startupTemplatePreview, setStartupTemplatePreview] =
    React.useState<StartupTemplateBundle | null>(null);

  const currentPresetKey = React.useMemo(
    () =>
      detectCurrentPreset(
        currentChatModelSettings as unknown as ChatModelSettings,
      ),
    [currentChatModelSettings],
  );
  const currentPreset = React.useMemo(
    () => getPresetByKey(currentPresetKey),
    [currentPresetKey],
  );

  const startupTemplates = React.useMemo(
    () => parseStartupTemplateBundles(startupTemplatesRaw),
    [startupTemplatesRaw],
  );

  const selectedSystemPromptRecord = React.useMemo<Prompt | null>(() => {
    if (!selectedSystemPrompt) return null;
    return (
      promptLibrary.find((prompt) => prompt.id === selectedSystemPrompt) || null
    );
  }, [promptLibrary, selectedSystemPrompt]);

  const startupTemplateNameFallback = React.useMemo(() => {
    const nameParts = [
      selectedCharacter?.name?.trim(),
      currentPreset && currentPreset.key !== "custom"
        ? t(
            `playground:presets.${currentPreset.key}.label`,
            currentPreset.label,
          )
        : null,
      selectedModel,
    ].filter((part): part is string => Boolean(part && part.trim().length > 0));
    if (nameParts.length > 0) {
      return sanitizeStartupTemplateName(
        `${nameParts.join(" \u00B7 ")} template`,
        "New startup template",
      );
    }
    return "New startup template";
  }, [currentPreset, selectedCharacter?.name, selectedModel, t]);

  const persistStartupTemplates = React.useCallback(
    (nextTemplates: StartupTemplateBundle[]) => {
      setStartupTemplatesRaw(serializeStartupTemplateBundles(nextTemplates));
    },
    [setStartupTemplatesRaw],
  );

  const handleSaveStartupTemplate = React.useCallback(() => {
    const trimmedSystemPrompt = String(systemPrompt || "").trim();
    const promptSource = inferStartupTemplatePromptSource(
      selectedSystemPromptRecord,
      trimmedSystemPrompt.length > 0,
    );
    const templateName = sanitizeStartupTemplateName(
      startupTemplateDraftName,
      startupTemplateNameFallback,
    );
    const nextTemplate = createStartupTemplateBundle({
      name: templateName,
      selectedModel,
      systemPrompt: trimmedSystemPrompt,
      selectedSystemPromptId: selectedSystemPrompt || null,
      promptStudioPromptId:
        selectedSystemPromptRecord?.studioPromptId ??
        selectedSystemPromptRecord?.serverId ??
        null,
      promptTitle: selectedSystemPromptRecord?.title || null,
      promptSource,
      presetKey: currentPresetKey,
      character: selectedCharacter || null,
      ragPinnedResults,
    });
    const nextTemplates = upsertStartupTemplateBundle(
      startupTemplates,
      nextTemplate,
    );
    persistStartupTemplates(nextTemplates);
    setStartupTemplateDraftName(templateName);
    setModeAnnouncement(
      t(
        "playground:composer.startupTemplateSavedNotice",
        "Startup template saved.",
      ),
    );
  }, [
    currentPresetKey,
    persistStartupTemplates,
    ragPinnedResults,
    selectedCharacter,
    selectedModel,
    selectedSystemPrompt,
    selectedSystemPromptRecord,
    setModeAnnouncement,
    startupTemplateDraftName,
    startupTemplateNameFallback,
    startupTemplates,
    systemPrompt,
    t,
  ]);

  const handleSaveRolePlaySetup = React.useCallback(
    (input: SaveRolePlaySetupInput) => {
      const trimmedSystemPrompt = String(systemPrompt || "").trim();
      const rolePlaySystemPrompt =
        input.rolePlay.behavior?.systemPrompt?.trim() || trimmedSystemPrompt;
      const promptSource =
        input.rolePlay.behavior?.source === "template" ||
        input.rolePlay.behavior?.source === "modified-template"
          ? "system-template"
          : inferStartupTemplatePromptSource(
              selectedSystemPromptRecord,
              rolePlaySystemPrompt.length > 0,
            );
      const templateName = sanitizeStartupTemplateName(
        input.name,
        startupTemplateNameFallback,
      );
      const rolePlayCharacter =
        input.rolePlay.identity?.kind === "character"
          ? ({
              ...(selectedCharacter || {}),
              id: String(input.rolePlay.identity.id),
              name: input.rolePlay.identity.name,
            } as Character)
          : selectedCharacter || null;
      const nextTemplate = createStartupTemplateBundle({
        name: templateName,
        source: "role-play-setup",
        selectedModel,
        systemPrompt: rolePlaySystemPrompt,
        selectedSystemPromptId:
          rolePlaySystemPrompt === trimmedSystemPrompt
            ? selectedSystemPrompt || null
            : null,
        promptStudioPromptId:
          rolePlaySystemPrompt === trimmedSystemPrompt
            ? selectedSystemPromptRecord?.studioPromptId ??
              selectedSystemPromptRecord?.serverId ??
              null
            : null,
        promptTitle:
          rolePlaySystemPrompt === trimmedSystemPrompt
            ? selectedSystemPromptRecord?.title || null
            : input.rolePlay.behavior?.templateTitle || null,
        promptSource,
        presetKey: input.rolePlay.generation?.presetKey ?? currentPresetKey,
        character: rolePlayCharacter,
        ragPinnedResults,
        rolePlay: input.rolePlay,
      });
      const nextTemplates = upsertStartupTemplateBundle(
        startupTemplates,
        nextTemplate,
      );
      persistStartupTemplates(nextTemplates);
      setStartupTemplateDraftName(templateName);
      setModeAnnouncement(
        t(
          "playground:composer.rolePlaySetupSavedNotice",
          "Role-play setup saved.",
        ),
      );
    },
    [
      currentPresetKey,
      persistStartupTemplates,
      ragPinnedResults,
      selectedCharacter,
      selectedModel,
      selectedSystemPrompt,
      selectedSystemPromptRecord,
      setModeAnnouncement,
      startupTemplateNameFallback,
      startupTemplates,
      systemPrompt,
      t,
    ],
  );

  const handleOpenStartupTemplatePreview = React.useCallback(
    (templateId: string) => {
      const template =
        startupTemplates.find((entry) => entry.id === templateId) || null;
      setStartupTemplatePreview(template);
    },
    [startupTemplates],
  );

  const applyStartupTemplateBundle = React.useCallback(
    (template: StartupTemplateBundle) => {
      const promptResolution = resolveStartupTemplatePrompt(
        template,
        promptLibrary,
      );
      const resolvedPromptContent =
        promptResolution.prompt?.content ?? template.systemPrompt;
      const resolvedPromptId = promptResolution.prompt?.id || null;

      if (template.selectedModel) {
        setSelectedModel(template.selectedModel);
        if (compareModeActive) {
          setCompareSelectedModels((prev: string[]) => {
            const updated = new Set(prev || []);
            updated.add(template.selectedModel!);
            return Array.from(updated);
          });
        }
      }

      if (resolvedPromptId) {
        setSelectedSystemPrompt(resolvedPromptId);
      } else {
        setSelectedSystemPrompt(undefined);
      }
      setSystemPrompt(resolvedPromptContent);
      updateChatModelSettings({ systemPromptTemplateId: undefined });

      const preset = getPresetByKey(template.presetKey);
      if (preset && preset.key !== "custom") {
        updateChatModelSettings(preset.settings);
      }

      void setSelectedCharacter(template.character || null);
      setRagPinnedResults(template.ragPinnedResults || []);
    },
    [
      compareModeActive,
      promptLibrary,
      setCompareSelectedModels,
      setRagPinnedResults,
      setSelectedCharacter,
      setSelectedModel,
      setSelectedSystemPrompt,
      setSystemPrompt,
      updateChatModelSettings,
    ],
  );

  const handleApplyStartupTemplate = React.useCallback(() => {
    if (!startupTemplatePreview) return;
    applyStartupTemplateBundle(startupTemplatePreview);
    setStartupTemplatePreview(null);
    setModeAnnouncement(
      t(
        "playground:composer.startupTemplateAppliedNotice",
        "Startup template applied.",
      ),
    );
  }, [
    applyStartupTemplateBundle,
    setModeAnnouncement,
    startupTemplatePreview,
    t,
  ]);

  const handleApplySavedRolePlaySetup = React.useCallback(
    (template: StartupTemplateBundle) => {
      applyStartupTemplateBundle(template);
      setStartupTemplatePreview(null);
      setModeAnnouncement(
        t(
          "playground:composer.rolePlaySetupAppliedNotice",
          "Role-play setup applied.",
        ),
      );
    },
    [applyStartupTemplateBundle, setModeAnnouncement, t],
  );

  const handleDeleteStartupTemplate = React.useCallback(
    (templateId: string) => {
      const nextTemplates = removeStartupTemplateBundle(
        startupTemplates,
        templateId,
      );
      persistStartupTemplates(nextTemplates);
      if (startupTemplatePreview?.id === templateId) {
        setStartupTemplatePreview(null);
      }
      setModeAnnouncement(
        t(
          "playground:composer.startupTemplateRemovedNotice",
          "Startup template removed.",
        ),
      );
    },
    [
      persistStartupTemplates,
      setModeAnnouncement,
      startupTemplatePreview?.id,
      startupTemplates,
      t,
    ],
  );

  const handleRenameStartupTemplate = React.useCallback(
    (templateId: string, nextName: string) => {
      const existing = startupTemplates.find((entry) => entry.id === templateId);
      if (!existing) return;
      const renamed = {
        ...existing,
        name: sanitizeStartupTemplateName(nextName, existing.name),
        updatedAt: Date.now(),
      };
      persistStartupTemplates(
        upsertStartupTemplateBundle(startupTemplates, renamed),
      );
      if (startupTemplatePreview?.id === templateId) {
        setStartupTemplatePreview(renamed);
      }
      setModeAnnouncement(
        t(
          "playground:composer.startupTemplateRenamedNotice",
          "Template renamed.",
        ),
      );
    },
    [
      persistStartupTemplates,
      setModeAnnouncement,
      startupTemplatePreview?.id,
      startupTemplates,
      t,
    ],
  );

  const handleTemplateSelect = React.useCallback(
    (template: Pick<PromptTemplate, "id" | "content">) => {
      setSystemPrompt(template.content);
      updateChatModelSettings({ systemPromptTemplateId: template.id });
      setSelectedSystemPrompt(undefined);
      setSelectedQuickPrompt(null);
    },
    [
      setSystemPrompt,
      setSelectedSystemPrompt,
      setSelectedQuickPrompt,
      updateChatModelSettings,
    ],
  );

  // Prompt summary label
  const promptSummaryLabel = React.useMemo(() => {
    if (selectedSystemPrompt) {
      return t("playground:composer.summary.systemPrompt", "System prompt");
    }
    if (selectedQuickPrompt) {
      return t("playground:composer.summary.customPrompt", "Custom prompt");
    }
    if (String(systemPrompt || "").trim().length > 0) {
      return t("playground:composer.summary.customPrompt", "Custom prompt");
    }
    return t("playground:composer.summary.noPrompt", "No prompt");
  }, [selectedQuickPrompt, selectedSystemPrompt, systemPrompt, t]);

  return {
    // Preset detection
    currentPresetKey,
    currentPreset,
    // Template state
    startupTemplates,
    startupTemplateDraftName,
    setStartupTemplateDraftName,
    startupTemplatePreview,
    setStartupTemplatePreview,
    startupTemplateNameFallback,
    selectedSystemPromptRecord,
    // Handlers
    handleSaveStartupTemplate,
    handleSaveRolePlaySetup,
    handleOpenStartupTemplatePreview,
    handleApplyStartupTemplate,
    handleApplySavedRolePlaySetup,
    handleDeleteStartupTemplate,
    handleRenameStartupTemplate,
    handleTemplateSelect,
    // Labels
    promptSummaryLabel,
  };
}
