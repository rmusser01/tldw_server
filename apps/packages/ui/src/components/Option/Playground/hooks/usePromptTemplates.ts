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
} from "../startup-template-bundles";
import { detectCurrentPreset, getPresetByKey } from "../ParameterPresets";
import type { PromptTemplate } from "../SystemPromptTemplates";
import type { Prompt } from "@/db/dexie/types";
import type { ChatModelSettings } from "@/store/model";

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

  const handleOpenStartupTemplatePreview = React.useCallback(
    (templateId: string) => {
      const template =
        startupTemplates.find((entry) => entry.id === templateId) || null;
      setStartupTemplatePreview(template);
    },
    [startupTemplates],
  );

  const handleApplyStartupTemplate = React.useCallback(() => {
    if (!startupTemplatePreview) return;
    const promptResolution = resolveStartupTemplatePrompt(
      startupTemplatePreview,
      promptLibrary,
    );
    const resolvedPromptContent =
      promptResolution.prompt?.content ?? startupTemplatePreview.systemPrompt;
    const resolvedPromptId = promptResolution.prompt?.id || null;

    if (startupTemplatePreview.selectedModel) {
      setSelectedModel(startupTemplatePreview.selectedModel);
      if (compareModeActive) {
        setCompareSelectedModels((prev: string[]) => {
          const updated = new Set(prev || []);
          updated.add(startupTemplatePreview.selectedModel!);
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

    const preset = getPresetByKey(startupTemplatePreview.presetKey);
    if (preset && preset.key !== "custom") {
      updateChatModelSettings(preset.settings);
    }

    void setSelectedCharacter(startupTemplatePreview.character || null);
    setRagPinnedResults(startupTemplatePreview.ragPinnedResults || []);
    setStartupTemplatePreview(null);
    setModeAnnouncement(
      t(
        "playground:composer.startupTemplateAppliedNotice",
        "Startup template applied.",
      ),
    );
  }, [
    compareModeActive,
    promptLibrary,
    setCompareSelectedModels,
    setModeAnnouncement,
    setRagPinnedResults,
    setSelectedCharacter,
    setSelectedModel,
    setSelectedSystemPrompt,
    setSystemPrompt,
    startupTemplatePreview,
    t,
    updateChatModelSettings,
  ]);

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
    handleOpenStartupTemplatePreview,
    handleApplyStartupTemplate,
    handleDeleteStartupTemplate,
    handleTemplateSelect,
    // Labels
    promptSummaryLabel,
  };
}
