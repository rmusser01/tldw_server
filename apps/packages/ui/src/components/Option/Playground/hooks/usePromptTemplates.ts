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
import { createDefaultActorSettings, type ActorSettings } from "@/types/actor";
import type { RolePlayApplyContext } from "../RolePlaySetupDrawer";
import { captureSelectedModelGuard, type SelectedModelSaveOperation } from "@/hooks/chat/useSelectedModel";
import {
  personaToAssistantSelection,
  characterToAssistantSelection,
  type AssistantSelection,
} from "@/types/assistant-selection";
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
  /** Capture the reviewed source before opening a preview. */
  captureStartupTemplateSource?: () => {
    isCurrent: () => boolean;
    continuationIsCurrent: () => boolean;
  };
  /** Setters for applying templates */
  setSelectedModel: (model: string) => Promise<boolean | void> | boolean | void;
  beginSelectedModelSave?: (model: string) => SelectedModelSaveOperation;
  setSelectedSystemPrompt: (id: string | undefined) => void;
  setSelectedQuickPrompt: (prompt: string | null) => void;
  setSystemPrompt: (prompt: string) => void;
  /** Explicit identity action: detach a different saved target before persistence. */
  applyAssistantSelection: (
    assistant: AssistantSelection | null,
    options?: AssistantSetupApplyOptions,
  ) => Promise<boolean | void> | boolean | void;
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

export type AssistantSetupApplyOptions = RolePlayApplyContext & {
  model?: string | null;
  settings?: Partial<ChatModelSettings>;
  selectedSystemPrompt?: string | null;
  sceneSettings?: ActorSettings;
  hasExplicitContext?: boolean;
  applyIdentity?: boolean;
  freshCharacterRoute?: boolean;
};

export type StartupTemplatePreviewLease = {
  id: object
  isCurrent: () => boolean
  sourceIsCurrent: () => boolean
  continuationIsCurrent: () => boolean
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
    selectedCharacter,
    ragPinnedResults,
    currentChatModelSettings,
    captureStartupTemplateSource,
    setSelectedModel,
    beginSelectedModelSave,
    setSelectedSystemPrompt,
    setSelectedQuickPrompt,
    setSystemPrompt,
    applyAssistantSelection,
    setRagPinnedResults,
    updateChatModelSettings,
    compareModeActive,
    setCompareSelectedModels,
    setModeAnnouncement,
    t,
  } = deps;

  const [startupTemplateDraftName, setStartupTemplateDraftName] =
    React.useState("");
  const previewMountedRef = React.useRef(true)
  React.useEffect(() => {
    previewMountedRef.current = true
    return () => {
      previewMountedRef.current = false
    }
  }, [])
  const previewRef = React.useRef<{
    template: StartupTemplateBundle
    lease: StartupTemplatePreviewLease
  } | null>(null)
  const [previewEntry, setPreviewEntry] =
    React.useState<typeof previewRef.current>(null)
  const setStartupTemplatePreview = React.useCallback(
    (template: StartupTemplateBundle | null) => {
      const id = {}
      const source = template ? captureStartupTemplateSource?.() : undefined
      const entry = template
        ? {
            template,
            lease: {
              id,
              isCurrent: () =>
                previewMountedRef.current && previewRef.current?.lease.id === id,
              sourceIsCurrent: source?.isCurrent ?? (() => true),
              continuationIsCurrent: source?.continuationIsCurrent ?? (() => true)
            }
          }
        : null
      // Retire synchronously, including Cancel/reopen of the same saved bundle.
      previewRef.current = entry
      setPreviewEntry(entry)
    },
    [captureStartupTemplateSource]
  )
  const startupTemplatePreview = previewEntry?.template ?? null
  const startupTemplatePreviewLease = previewEntry?.lease ?? null

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
      const rolePlaySystemPrompt =
        input.rolePlay.behavior?.systemPrompt.trim() ?? ""
      const promptSource = input.rolePlay.behavior ? "system-template" : "none"
      const templateName = sanitizeStartupTemplateName(
        input.name,
        startupTemplateNameFallback
      )
      const rolePlayCharacter =
        input.rolePlay.identity?.kind === "character"
          ? ({
              ...(String(selectedCharacter?.id) ===
              String(input.rolePlay.identity.id)
                ? selectedCharacter
                : {}),
              id: String(input.rolePlay.identity.id),
              name: input.rolePlay.identity.name
            } as Character)
          : null
      const nextTemplate = createStartupTemplateBundle({
        name: templateName,
        source: "role-play-setup",
        selectedModel,
        systemPrompt: rolePlaySystemPrompt,
        selectedSystemPromptId: null,
        promptStudioPromptId: null,
        promptTitle: input.rolePlay.behavior?.templateTitle || null,
        promptSource,
        presetKey: input.rolePlay.generation?.presetKey ?? currentPresetKey,
        character: rolePlayCharacter,
        ragPinnedResults,
        rolePlay: input.rolePlay
      })
      const nextTemplates = upsertStartupTemplateBundle(
        startupTemplates,
        nextTemplate
      )
      persistStartupTemplates(nextTemplates)
      setStartupTemplateDraftName(templateName)
      setModeAnnouncement(
        t(
          "playground:composer.rolePlaySetupSavedNotice",
          "Role-play setup saved."
        )
      )
    },
    [
      currentPresetKey,
      persistStartupTemplates,
      ragPinnedResults,
      selectedCharacter,
      selectedModel,
      setModeAnnouncement,
      startupTemplateNameFallback,
      startupTemplates,
      t
    ]
  )

  const handleOpenStartupTemplatePreview = React.useCallback(
    (templateId: string) => {
      const template =
        startupTemplates.find((entry) => entry.id === templateId) || null;
      setStartupTemplatePreview(template);
    },
    [startupTemplates, setStartupTemplatePreview],
  );

  const createStartupTemplateApplication = React.useCallback(
    (original: StartupTemplateBundle) => {
      const template = JSON.parse(
        JSON.stringify(original)
      ) as StartupTemplateBundle
      const source = captureStartupTemplateSource?.()
      let modelIsCurrent = captureSelectedModelGuard()
      let destinationIsCurrent = source?.isCurrent ?? (() => true)
      let destination: NonNullable<
        Parameters<NonNullable<RolePlayApplyContext["onDestination"]>>[0]
      > | null = null
      let identityAccepted = false
      let modelAccepted = false
      let modelSave: SelectedModelSaveOperation | null = null
      let completed = false
      let retired = false
      const operationIsCurrent = () =>
        !retired &&
        previewMountedRef.current &&
        source?.continuationIsCurrent?.() !== false &&
        destinationIsCurrent() &&
        modelIsCurrent()
      const rolePlayIdentity = template.rolePlay?.identity
      const selection = template.rolePlay
        ? rolePlayIdentity?.kind === "persona"
          ? personaToAssistantSelection({
              id: String(rolePlayIdentity.id),
              name: rolePlayIdentity.name
            })
          : characterToAssistantSelection(
              rolePlayIdentity?.kind === "character"
                ? ({
                    ...(String(template.character?.id) ===
                    String(rolePlayIdentity.id)
                      ? template.character
                      : {}),
                    id: String(rolePlayIdentity.id),
                    name: rolePlayIdentity.name
                  } as Character)
                : null
            )
        : characterToAssistantSelection(template.character)
      const promptResolution = resolveStartupTemplatePrompt(
        template,
        promptLibrary
      )
      const resolvedPromptContent = template.rolePlay
        ? template.rolePlay.behavior?.systemPrompt ?? ""
        : promptResolution.prompt?.content ?? template.systemPrompt
      const resolvedPromptId = promptResolution.prompt?.id || null
      const preset = getPresetByKey(template.presetKey)
      const settings = {
        ...(preset && preset.key !== "custom" ? preset.settings : {}),
        systemPrompt: resolvedPromptContent,
        systemPromptTemplateId:
          template.rolePlay?.behavior?.templateId?.trim() || undefined
      }

      return async (context?: RolePlayApplyContext) => {
        // Retry supplies its new attempt guard; the original operation owns its destination.
        const isCurrent = () =>
          operationIsCurrent() && context?.isCurrent?.() !== false
        if (!isCurrent()) return false
        if (!identityAccepted) {
          const applied = await applyAssistantSelection(selection, {
            ...context,
            onDestination: (next) => {
              modelIsCurrent = captureSelectedModelGuard()
              destinationIsCurrent = next.isCurrent
              destination = { ...next, isCurrent: operationIsCurrent }
              context?.onDestination?.(destination)
            },
            model: template.selectedModel || selectedModel,
            settings,
            selectedSystemPrompt: resolvedPromptId,
            sceneSettings: template.rolePlay
              ? template.rolePlay.scene ?? createDefaultActorSettings()
              : undefined,
            hasExplicitContext: Boolean(template.ragPinnedResults?.length)
          })
          if (applied === false || !isCurrent()) return false
          identityAccepted = true
        } else if (destination) {
          context?.onDestination?.(destination)
        }
        if (!isCurrent()) return false
        if (completed) return true

        if (template.selectedModel && !modelAccepted) {
          let acknowledged: boolean | void
          if (modelSave) {
            acknowledged = await modelSave.retry()
          } else {
            modelSave = beginSelectedModelSave?.(template.selectedModel) ?? null
            if (modelSave) modelIsCurrent = modelSave.isCurrent
            acknowledged = modelSave
              ? await modelSave.completion
              : await setSelectedModel(template.selectedModel)
          }
          if (acknowledged === false) {
            retired = true
            return false
          }
          if (!isCurrent()) return false
          modelAccepted = true
        }
        if (template.selectedModel && compareModeActive) {
          setCompareSelectedModels((prev: string[]) => {
            if (!isCurrent()) return prev
            const updated = new Set(prev || [])
            updated.add(template.selectedModel!)
            return Array.from(updated)
          })
          if (!isCurrent()) return false
        }
        setSelectedSystemPrompt(resolvedPromptId || undefined)
        if (!isCurrent()) return false
        setSystemPrompt(resolvedPromptContent)
        if (!isCurrent()) return false
        updateChatModelSettings(settings)
        if (!isCurrent()) return false
        setRagPinnedResults(template.ragPinnedResults || [])
        completed = isCurrent()
        return completed
      }
    },
    [
      captureStartupTemplateSource,
      compareModeActive,
      promptLibrary,
      selectedModel,
      setCompareSelectedModels,
      setRagPinnedResults,
      applyAssistantSelection,
      setSelectedModel,
      beginSelectedModelSave,
      setSelectedSystemPrompt,
      setSystemPrompt,
      updateChatModelSettings
    ]
  )
  const startupApplicationsRef = React.useRef(
    new WeakMap<
      object,
      {
        intent: string
        apply: ReturnType<typeof createStartupTemplateApplication>
      }
    >()
  )
  const applyStartupTemplateBundle = React.useCallback(
    (template: StartupTemplateBundle, context?: RolePlayApplyContext) => {
      const id = context?.applicationId
      const intent = JSON.stringify(template)
      let application = id ? startupApplicationsRef.current.get(id) : undefined
      if (application && application.intent !== intent)
        return Promise.resolve(false)
      if (!application) {
        application = {
          intent,
          apply: createStartupTemplateApplication(template)
        }
        if (id) startupApplicationsRef.current.set(id, application)
      }
      return application.apply(context)
    },
    [createStartupTemplateApplication]
  )
  const handleSavedRolePlaySetupApplied = React.useCallback(() => {
    setModeAnnouncement(
      t(
        "playground:composer.rolePlaySetupAppliedNotice",
        "Role-play setup applied."
      )
    )
  }, [setModeAnnouncement, t])

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
      setStartupTemplatePreview,
      startupTemplates,
      t,
    ],
  );

  const handleRenameStartupTemplate = React.useCallback(
    (templateId: string, nextName: string) => {
      const existing = startupTemplates.find(
        (entry) => entry.id === templateId,
      );
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
      setStartupTemplatePreview,
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
    if (String(systemPrompt || "").trim().length > 0) {
      return t("playground:composer.summary.customPrompt", "Custom prompt");
    }
    return t("playground:composer.summary.noPrompt", "No prompt");
  }, [selectedSystemPrompt, systemPrompt, t]);

  return {
    // Preset detection
    currentPresetKey,
    currentPreset,
    // Template state
    startupTemplates,
    startupTemplateDraftName,
    setStartupTemplateDraftName,
    startupTemplatePreview,
    startupTemplatePreviewLease,
    setStartupTemplatePreview,
    startupTemplateNameFallback,
    selectedSystemPromptRecord,
    // Handlers
    applyStartupTemplateBundle,
    handleSaveStartupTemplate,
    handleSaveRolePlaySetup,
    handleOpenStartupTemplatePreview,
    handleApplySavedRolePlaySetup: applyStartupTemplateBundle,
    handleSavedRolePlaySetupApplied,
    handleDeleteStartupTemplate,
    handleRenameStartupTemplate,
    handleTemplateSelect,
    // Labels
    promptSummaryLabel,
  };
}
