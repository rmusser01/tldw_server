import { getStructuredApiErrorDetail } from "@/services/tldw/api-error";
import type { ChatModelSettings } from '@/store/model';
import type { ToolChoice } from '@/store/option';
import {
  isAutoModelId,
  parseProviderQualifiedModelSelection,
  resolveExplicitProviderForSelectedModel,
} from '@/utils/resolve-api-provider';

export type NativeCharacterConfiguration = {
  temporary: boolean;
  image?: string;
  unsupportedContext?: boolean;
  toolChoice?: ToolChoice;
  clientToolCount: number;
  actorEnabled?: boolean;
  settings: ChatModelSettings;
  model: string;
  currentModel?: string | null;
};

/** Validate intended configuration without treating the source owner as a destination. */
export const validateNativeCharacterConfiguration = (params: NativeCharacterConfiguration) => {
  if (params.temporary) throw new Error('temporary_history_unavailable');
  if (params.image) throw new Error('native_history_assets_unsupported');
  if (params.unsupportedContext) throw new Error('native_history_explicit_context_unsupported');
  if (params.toolChoice === 'required' || (params.clientToolCount && params.toolChoice !== 'none'))
    throw new Error('native_history_client_tools_unsupported');
  const settings = { ...params.settings };
  const unsupportedSettings: Array<keyof ChatModelSettings> = [
    'f16KV',
    'keepAlive',
    'logitsAll',
    'mirostat',
    'mirostatEta',
    'mirostatTau',
    'numBatch',
    'numCtx',
    'numGpu',
    'numGqa',
    'numKeep',
    'numThread',
    'penalizeNewline',
    'repeatLastN',
    'repeatPenalty',
    'ropeFrequencyBase',
    'ropeFrequencyScale',
    'tfsZ',
    'topK',
    'typicalP',
    'useMLock',
    'useMMap',
    'useMlock',
    'vocabOnly',
    'seed',
    'minP',
    'thinking',
    'historyMessageLimit',
    'historyMessageOrder',
  ];
  const unsupportedSetting = unsupportedSettings.find(
    (key) => settings[key] !== undefined && settings[key] !== null
  );
  if (unsupportedSetting)
    throw new Error(`native_history_setting_unsupported:${unsupportedSetting}`);
  if (settings.reasoningEffort && !['low', 'medium', 'high'].includes(settings.reasoningEffort))
    throw new Error('native_history_reasoning_effort_unsupported');
  if (settings.extraBody?.trim() || settings.extraHeaders?.trim())
    throw new Error('native_history_extra_parameters_unsupported');
  if (settings.systemPromptTemplateId)
    throw new Error('native_history_prompt_template_unsupported');
  if (params.actorEnabled) throw new Error('native_history_actor_context_unsupported');
  const selected = parseProviderQualifiedModelSelection(params.model);
  const model = selected.modelId.replace(/^tldw:/, '').trim();
  const provider = resolveExplicitProviderForSelectedModel({
    currentSelectedModel: params.currentModel ?? params.model,
    requestedSelectedModel: params.model,
    explicitProvider: settings.apiProvider,
  });
  if (!model || isAutoModelId(model) || !provider)
    throw new Error('native_history_explicit_model_provider_required');
  return { model, provider, settings };
};

/** A confirmed original-turn refusal retains transport evidence and its accepted view guard. */
export class NativeHistoryContextRefusal extends Error {
  constructor(
    cause: unknown,
    readonly isCurrent: () => boolean
  ) {
    super('native_history_context_refused', { cause });
  }
}

/** Only a failed pre-dispatch settings read can certify that no input was sent. */
export class NativeHistorySettingsReadError extends Error {
  constructor(cause: unknown, readonly isCurrent: () => boolean) {
    super('native_history_settings_unverified', { cause });
  }
}

export const nativeCharacterPreflightMessage = (
  error: unknown,
  t: (key: string, fallback: string, options?: { setting: string }) => string
): string | null => {
  const code = error instanceof Error ? error.message : '';
  if (code === 'actor_settings_unavailable') return t(
    'playground:composer.sceneLoadError',
    'Scene settings could not be loaded. Existing chat setup was left unchanged.'
  );
  if (error instanceof NativeHistorySettingsReadError)
    return t(
      'playground:composer.native_history_settings_unverified',
      'Conversation settings could not be verified. Your message has not been sent. Reload this saved chat or try again.'
    );
  if (error instanceof NativeHistoryContextRefusal)
    return t(
      'playground:composer.native_history_context_refused',
      'Your message was not added because this saved chat uses unsupported context. Your input is kept separately; copy it and review the saved setup or start a compatible chat.'
    );

  if (code.startsWith('native_history_setting_unsupported:')) {
    return t(
      'playground:composer.characterUnsupportedSetting',
      '{{setting}} is not supported in saved character chat. Review response settings before starting; your current chat and draft are still here.',
      {
        setting: code.slice('native_history_setting_unsupported:'.length),
      }
    );
  }
  const messages: Record<string, string> = {
    assistant_selection_restore_failed:
      'Assistant preferences could not be restored. Your chat and draft are still here; try choosing the assistant again.',
    native_history_greeting_requires_fresh:
      'Start a new saved character chat before choosing an opening. Your current chat and draft are still here.',
    native_history_greeting_not_stored:
      'The server did not save an opening. Your message has not been sent. Review the saved chat in retained drafts before starting again.',
    native_history_response_pending:
      'Wait for the current response to finish before starting a character chat. Your chat and draft are still here.',
    native_history_draft_not_retained:
      'Your draft could not be retained. Keep or copy your message before starting the character opening.',
    stale_selection:
      'The chat or setup changed while saving the opening. Your message has not been sent. Review retained drafts before starting again.',

    temporary_history_unavailable:
      'Start a new saved chat before applying this character. Keep or copy your temporary conversation first.',
    native_history_assets_unsupported:
      'Image attachments are not supported in saved character chat. Keep the attachment in this chat or remove it before starting.',
    native_history_explicit_context_unsupported:
      'Research, files, prompts, or other added context are not supported in saved character chat. Review the active context before starting.',
    native_history_client_tools_unsupported:
      'Client tools are not supported in saved character chat. Set tools to None before starting.',
    native_history_reasoning_effort_unsupported:
      'Choose Low, Medium, or High reasoning effort before starting this character chat.',
    native_history_extra_parameters_unsupported:
      'Extra request parameters are not supported in saved character chat. Review model settings before starting.',
    native_history_prompt_template_unsupported:
      'Behavior templates are not supported in saved character chat. Clear the behavior template before starting.',
    native_history_actor_context_unsupported:
      'Scene overrides are not supported in saved character chat. Turn off scene overrides before starting.',
    native_history_explicit_model_provider_required:
      'Choose a specific model and provider before starting this character chat.',
    native_history_server_unavailable:
      'Connect to your server before starting this character chat. Your current chat and draft are still here.',
    native_history_model_unavailable:
      'Choose an available model with a configured provider before starting this character chat. Your current chat and draft are still here.',
  };
  return messages[code] ? t(`playground:composer.${code}`, messages[code]) : null;
};


/** Regeneration keeps the saved exchange; an uncertain transport never certifies rollback. */
export const nativeRegenerationFeedback = (error: unknown, t: (key: string, fallback: string) => string): string => {
  if (error instanceof NativeHistorySettingsReadError) return t("common:regenerateMessage.settings",
    "Conversation settings could not be verified. No new reply was requested. Reload this saved chat or try again; your original exchange and draft are kept.");
  if (error instanceof NativeHistoryContextRefusal) return t("common:regenerateMessage.context",
    "The server refused regeneration because this saved chat uses unsupported context. Review the saved setup or start a compatible chat; your original exchange and draft are kept.");
  const value = error && typeof error === "object" ? error as { message?: unknown; details?: unknown; code?: unknown; reason?: unknown } : {};
  const detail = getStructuredApiErrorDetail(value.details);
  const code = [value.code, value.reason, detail?.code, detail?.error_code, value.message].find(v => typeof v === "string") as string | undefined;
  if (["message_changed", "version_conflict", "stale_snapshot", "stale_projection", "stale_selection"].includes(code ?? ""))
    return t("common:regenerateMessage.changed", "The saved message or conversation changed. Your original exchange and draft are kept. Reload the intended chat before regenerating again.");
  if (["request_config_scope_changed", "owner_lease_expired", "owner_conversation_mismatch"].includes(code ?? ""))
    return t("common:regenerateMessage.scope", "The server or signed-in account changed. Your original exchange and draft are kept. Return to the intended server and account and reload this chat.");
  if (code === "unsupported_history_regeneration" || code === "unsupported_history_action_context" ||
      code === "native_history_requires_server_owner" || code?.startsWith("native_history_setting_unsupported:") ||
      /^native_history_.*unsupported$/.test(code ?? ""))
    return t("common:regenerateMessage.unsupported", "This reply or its active setup does not support regeneration yet. Your original exchange and draft are kept. Review conversation and response settings or start a compatible saved chat.");
  return t("common:regenerateMessage.failed", "We could not confirm the new reply. Your original exchange and draft are kept. Reload this chat and review recovery before regenerating again.");
};
