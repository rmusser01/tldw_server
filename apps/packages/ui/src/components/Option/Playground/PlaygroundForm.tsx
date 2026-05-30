import {
  ChatComposer,
  useComposerVariantPreference,
} from "@/components/Chat/composer/ChatComposer";
import { useComposerEnabledPreference } from "@/components/Chat/composer/hooks/useComposerEnabledPreference";
import { AudioSourcePicker } from "@/components/Common/AudioSourcePicker";
import { BetaTag } from "@/components/Common/Beta";
// getImageBackendConfigs, normalizeImageBackendConfig, resolveImageBackendConfig moved to usePlaygroundImageGen
import { CharacterSelect } from "@/components/Common/CharacterSelect";
import { ChatQueuePanel } from "@/components/Common/ChatQueuePanel";
import { type KnowledgeTab } from "@/components/Knowledge";
import type { SlashCommandItem } from "@/components/Sidepanel/Chat/SlashCommandMenu";
import { isFirefoxTarget } from "@/config/platform";
import { getDesignSystemState } from "@/design-system";
import { getAllPrompts } from "@/db/dexie/helpers";
import { useChatSettingsRecord } from "@/hooks/chat/useChatSettingsRecord";
import { resolveEffectiveAssistantState } from "@/hooks/chat/effective-assistant-state";
import {
  type CollapsedRange,
  useActionBarVisibility,
  useComposerTokens,
  useDeferredComposerInput,
  useImageBackend,
  useMcpToolsControl,
  useMessageCollapse,
  useModelSelector,
  useSlashCommands,
} from "@/hooks/playground";
// TldwButton moved to extracted sub-components
import { useAntdNotification } from "@/hooks/useAntdNotification";
import { useAudioSourceCatalog } from "@/hooks/useAudioSourceCatalog";
import { useCanonicalConnectionConfig } from "@/hooks/useCanonicalConnectionConfig";
import { useChatMoodBadgePreference } from "@/hooks/useChatMoodBadgePreference";
import {
  resolveStickyComposerTextareaMaxHeight,
  type ComposerDockLayoutMetrics,
} from "./mobile-composer-layout";
import { useConnectionState } from "@/hooks/useConnectionState";
import type { DictationModePreference } from "@/hooks/useDictationStrategy";
import { useMcpTools } from "@/hooks/useMcpTools";
import { useMobile } from "@/hooks/useMediaQuery";
// isMac moved to PlaygroundSendControl
import { useSelectedAssistant } from "@/hooks/useSelectedAssistant";
import { useServerCapabilities } from "@/hooks/useServerCapabilities";
import { useTldwAudioStatus } from "@/hooks/useTldwAudioStatus";
import { useVoiceChatMessages } from "@/hooks/useVoiceChatMessages";
import { useVoiceChatSettings } from "@/hooks/useVoiceChatSettings";
import { useVoiceChatStream } from "@/hooks/useVoiceChatStream";
// useQueuedRequests moved to usePlaygroundQueueManagement
import type { ChatDocuments } from "@/models/ChatTypes";
import { clearSetting, getSetting } from "@/services/settings/registry";
import {
  DISCUSS_MEDIA_PROMPT_SETTING,
  DISCUSS_WATCHLIST_PROMPT_SETTING,
} from "@/services/settings/ui-settings";
import { fetchChatModels, fetchImageModels } from "@/services/tldw-server";
import {
  type ResearchRunCreateRequest,
  type ResearchRunFollowUpBackground,
  tldwClient,
} from "@/services/tldw/TldwApiClient";
// ChatRequestDebugSnapshot moved to usePlaygroundRawPreview
import {
  buildDiscussMediaHint,
  getMediaChatHandoffMode,
  normalizeMediaChatHandoffPayload,
  parseMediaIdAsNumber,
} from "@/services/tldw/media-chat-handoff";
import {
  normalizeVoiceConversationRuntimeError,
  resolveVoiceConversationAvailability,
  resolveVoiceConversationTtsConfig,
  shouldProbeVoiceConversationAudioHealth,
} from "@/services/tldw/voice-conversation";
import {
  buildWatchlistChatHint,
  normalizeWatchlistChatHandoffPayload,
} from "@/services/tldw/watchlist-chat-handoff";
import { saveActorSettingsForChat } from "@/services/actor-settings";
import {
  shouldEnableOptionalResource,
  useChatSurfaceCoordinatorStore,
} from "@/store/chat-surface-coordinator";
import { useActorStore } from "@/store/actor";
import {
  type ChatModelSettings,
  useStoreChatModelSettings,
} from "@/store/model";
import { useStoreMessageOption } from "@/store/option";
import { useUiModeStore } from "@/store/ui-mode";
import { createDefaultActorSettings } from "@/types/actor";
import type { Character } from "@/types/character";
import { DEFAULT_CHAT_SETTINGS } from "@/types/chat-settings";
import {
  assistantSelectionToCharacter,
  characterToAssistantSelection,
  getAssistantSelectionMode,
} from "@/types/assistant-selection";
import { ConnectionPhase, deriveConnectionUxState } from "@/types/connection";
import { PASTED_TEXT_CHAR_LIMIT } from "@/utils/constant";
import {
  normalizeFocusSelector,
  scheduleFocusFirstVisibleElement,
} from "@/utils/focus-return";
// resolveApiProviderForModel moved to usePlaygroundRawPreview and usePlaygroundImageGen
import {
  DEFAULT_CHARACTER_STORAGE_KEY,
  defaultCharacterStorage,
  isFreshChatState,
  resolveCharacterSelectionId,
  shouldApplyDefaultCharacter,
  shouldResetDefaultCharacterBootstrap,
} from "@/utils/default-character-preference";
import {
  type ImageGenerationEventSyncMode,
  type ImageGenerationEventSyncPolicy,
  type ImageGenerationRefineMetadata,
  type ImageGenerationRequestSnapshot,
  PLAYGROUND_IMAGE_EVENT_SYNC_DEFAULT_STORAGE_KEY,
  normalizeImageGenerationEventSyncMode,
  normalizeImageGenerationEventSyncPolicy,
  resolveImageGenerationEventSyncMode,
} from "@/utils/image-generation-chat";
import { isFireFoxPrivateMode } from "@/utils/is-private-mode";
import { handleChatInputKeyDown } from "@/utils/key-down";
import { dispatchOpenAssistantSelect } from "@/utils/assistant-select-events";
import { resolveStartupSelectedModel } from "@/utils/model-startup-selection";
import { trackOnboardingChatSubmitSuccess } from "@/utils/onboarding-ingestion-telemetry";
import type { ChatModelUsability } from "@/utils/chat-model-availability";
import { getProviderDisplayName } from "@/utils/provider-registry";
import { getVariable } from "@/utils/select-variable";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Button,
  Checkbox,
  Dropdown,
  Input,
  InputNumber,
  Modal,
  Radio,
  Select,
  Switch,
  Tooltip,
} from "antd";
import {
  ChevronRight,
  CornerUpLeft,
  Headphones,
  ImageIcon,
  X,
} from "lucide-react";
import React from "react";
import { useTranslation } from "react-i18next";
import { Link, useNavigate } from "react-router-dom";

import { useStorage } from "@plasmohq/storage/hook";

import { useFocusShortcuts } from "~/hooks/keyboard";
import { useMessageOption } from "~/hooks/useMessageOption";
import { useTabMentions } from "~/hooks/useTabMentions";
import { useWebUI } from "~/store/webui";

import { AttachedResearchContextChip } from "./AttachedResearchContextChip";
import { AttachmentsSummary } from "./AttachmentsSummary";
import { ChatModelSelectorDropdown } from "./ChatModelSelectorDropdown";
// ContextFootprintPanel moved to PlaygroundContextWindowModal
import { CompareToggle } from "./CompareToggle";
import { ComposerTextarea } from "./ComposerTextarea";
import { ComposerToolbar } from "./ComposerToolbar";
import { MentionsDropdown } from "./MentionsDropdown";
import { getPresetByKey } from "./ParameterPresets";
import { PlaygroundComposerNotices } from "./PlaygroundComposerNotices";
import {
  PlaygroundChatErrorBanner,
  usePlaygroundChatErrorBanner,
} from "./PlaygroundChatErrorBanner";
import { PlaygroundKnowledgeSection } from "./PlaygroundKnowledgeSection";
import { PlaygroundMcpControl } from "./PlaygroundMcpControl";
import { PlaygroundModelCatalogControls } from "./PlaygroundModelCatalogControls";
import { PlaygroundModeLauncher } from "./PlaygroundModeLauncher";
import {
  PlaygroundAttachmentButton,
  PlaygroundSendControl,
  type PlaygroundSendBlocker,
} from "./PlaygroundSendControl";
import { SidepanelImportedContextBanner } from "./SidepanelImportedContextBanner";
import { PlaygroundToolsPopover } from "./PlaygroundToolsPopover";
import type { RolePlaySetupApplyPayload } from "./RolePlaySetupDrawer";
import { TokenProgressBar } from "./TokenProgressBar";
import { VoiceChatIndicator } from "./VoiceChatIndicator";
import { buildConversationSummaryCheckpointPrompt } from "./conversation-summary-checkpoint";
import {
  OPEN_ACTOR_SETTINGS_EVENT,
  OPEN_KNOWLEDGE_PANEL_EVENT,
  OPEN_MCP_SETTINGS_EVENT,
  OPEN_MCP_TOOLS_EVENT,
  OPEN_MODEL_SELECTOR_EVENT,
  OPEN_MODEL_SETTINGS_EVENT,
  OPEN_TURN_TOOLS_EVENT,
  SET_TEMPORARY_CHAT_EVENT,
  TOGGLE_WEB_SEARCH_EVENT,
  type McpSettingsOpenDetail,
  type ModelSelectorOpenDetail,
  type ModelSettingsOpenDetail,
} from "./playground-cockpit-actions";
import { deriveRolePlayState, type RolePlayIdentity } from "./role-play-state";
import { summarizeRolePlayScene } from "./role-play-scene";
import { getPromptTemplateById } from "./SystemPromptTemplates";
// buildImagePromptRefineMessages, extractImagePromptRefineCandidate moved to usePlaygroundImageGen
// QueuedRequest moved to usePlaygroundQueueManagement
// WeightedImagePromptContextEntry moved to usePlaygroundImageGen
// CompareResponseDiff moved to usePlaygroundImageGen
import {
  estimateTokensFromText,
  toText,
  useComposerInput,
  useContextWindow,
  useModelComparison,
  usePlaygroundAttachments,
  usePlaygroundContextItems,
  usePlaygroundImageGen,
  usePlaygroundPersistence,
  usePlaygroundQueueManagement,
  usePlaygroundRawPreview,
  usePlaygroundSettings,
  usePlaygroundSubmit,
  usePlaygroundVoiceChat,
  usePromptTemplates,
} from "./hooks";
import { useSidepanelChatHandoffImport } from "./hooks/useSidepanelChatHandoffImport";
// SessionInsightsPanel moved to PlaygroundContextWindowModal
import { type ModelRecommendationAction } from "./model-recommendations";
import {
  type AttachedResearchContext,
  type ResearchFollowUpTarget,
  applyAttachedResearchContextEdits,
  resetAttachedResearchContext,
  toChatResearchContext,
} from "./research-chat-context";
import {
  type StartupTemplateBundle,
  describeStartupTemplatePrompt,
  resolveStartupTemplatePrompt,
} from "./startup-template-bundles";
import { useMobileComposerViewport } from "./useMobileComposerViewport";

type Props = {
  droppedFiles: File[];
  attachedResearchContext?: AttachedResearchContext | null;
  attachedResearchContextBaseline?: AttachedResearchContext | null;
  attachedResearchContextPinned?: AttachedResearchContext | null;
  attachedResearchContextHistory?: AttachedResearchContext[];
  onApplyAttachedResearchContext?: (context: AttachedResearchContext) => void;
  onResetAttachedResearchContext?: () => void;
  onRemoveAttachedResearchContext?: () => void;
  onPinAttachedResearchContext?: () => void;
  onUnpinAttachedResearchContext?: () => void;
  onRestorePinnedResearchContext?: () => void;
  onPinAttachedResearchContextHistory?: (
    context: AttachedResearchContext,
  ) => void;
  onSelectAttachedResearchContextHistory?: (
    context: AttachedResearchContext,
  ) => void;
  onPrepareResearchFollowUp?: (target: ResearchFollowUpTarget) => void;
  stickyDockEnabled?: boolean;
  mobileCockpitModeActive?: boolean;
  onComposerLayoutChange?: (metrics: ComposerDockLayoutMetrics) => void;
  onDraftPresenceChange?: (hasDraft: boolean) => void;
  characterChatSendBlocker?: PlaygroundSendBlocker | null;
  characterChatModelUsability?: ChatModelUsability | null;
  characterChatModelUsabilityLabel?: string | null;
  characterChatModelUsabilityTitle?: string | null;
};

type DefaultCharacterPreferenceQueryResult = {
  defaultCharacterId: string | null;
};

type PlaygroundQueuedSourceContext = {
  documents?: ChatDocuments;
  imageBackendOverride?: string;
  isImageCommand?: boolean;
};

type ComposerPopoverKey =
  | "context"
  | "model"
  | "mcp"
  | "tools"
  | "attachment"
  | "send";

const FOLLOW_UP_RESEARCH_PROMPT_PREFIX = "Follow up on this research:";
const CASUAL_COMPOSER_MAX_HEIGHT_PX = 120;
const PRO_COMPOSER_MAX_HEIGHT_PX = 160;

const buildFollowUpResearchBackground = (
  context: AttachedResearchContext,
): ResearchRunFollowUpBackground => {
  const unsupportedClaimCount =
    typeof context.verification_summary?.unsupported_claim_count === "number" &&
    Number.isFinite(context.verification_summary.unsupported_claim_count) &&
    context.verification_summary.unsupported_claim_count >= 0
      ? Math.trunc(context.verification_summary.unsupported_claim_count)
      : 0;
  const highTrustCount =
    typeof context.source_trust_summary?.high_trust_count === "number" &&
    Number.isFinite(context.source_trust_summary.high_trust_count) &&
    context.source_trust_summary.high_trust_count >= 0
      ? Math.trunc(context.source_trust_summary.high_trust_count)
      : 0;

  return {
    question: context.question || context.query,
    outline: Array.isArray(context.outline)
      ? context.outline
          .filter(
            (section) =>
              typeof section?.title === "string" &&
              section.title.trim().length > 0,
          )
          .map((section) => ({ title: section.title.trim() }))
      : [],
    key_claims: Array.isArray(context.key_claims)
      ? context.key_claims
          .map((claim, index) =>
            typeof claim?.text === "string" && claim.text.trim().length > 0
              ? {
                  claim_id: `claim_${index + 1}`,
                  text: claim.text.trim(),
                }
              : null,
          )
          .filter(
            (claim): claim is { claim_id: string; text: string } =>
              claim !== null,
          )
      : [],
    unresolved_questions: Array.isArray(context.unresolved_questions)
      ? context.unresolved_questions.filter(
          (question): question is string =>
            typeof question === "string" && question.trim().length > 0,
        )
      : [],
    verification_summary: {
      supported_claim_count: 0,
      unsupported_claim_count: unsupportedClaimCount,
    },
    source_trust_summary: {
      high_trust_count: highTrustCount,
      low_trust_count: 0,
    },
  };
};

const cloneAttachedResearchContext = (
  context: AttachedResearchContext | null,
): AttachedResearchContext | null =>
  context
    ? {
        ...context,
        outline: context.outline.map((section) => ({ ...section })),
        key_claims: context.key_claims.map((claim) => ({ ...claim })),
        unresolved_questions: [...context.unresolved_questions],
        verification_summary: context.verification_summary
          ? { ...context.verification_summary }
          : undefined,
        source_trust_summary: context.source_trust_summary
          ? { ...context.source_trust_summary }
          : undefined,
      }
    : null;

const stringifyOutline = (context: AttachedResearchContext | null): string =>
  context?.outline.map((section) => section.title).join("\n") ?? "";

const stringifyKeyClaims = (context: AttachedResearchContext | null): string =>
  context?.key_claims.map((claim) => claim.text).join("\n") ?? "";

const stringifyUnresolvedQuestions = (
  context: AttachedResearchContext | null,
): string => context?.unresolved_questions.join("\n") ?? "";

const LazyCurrentChatModelSettings = React.lazy(() =>
  import("@/components/Common/Settings/CurrentChatModelSettings").then(
    (module) => ({ default: module.CurrentChatModelSettings }),
  ),
);

const LazyActorPopout = React.lazy(() =>
  import("@/components/Common/Settings/ActorPopout").then((module) => ({
    default: module.ActorPopout,
  })),
);

const LazyDocumentGeneratorDrawer = React.lazy(
  () => import("@/components/Common/Playground/DocumentGeneratorDrawer"),
);

const LazyVoiceModeSelector = React.lazy(() =>
  import("./VoiceModeSelector").then((module) => ({
    default: module.VoiceModeSelector,
  })),
);

const LazyPlaygroundImageGenModal = React.lazy(() =>
  import("./PlaygroundImageGenModal").then((module) => ({
    default: module.PlaygroundImageGenModal,
  })),
);

const LazyPlaygroundRawRequestModal = React.lazy(() =>
  import("./PlaygroundRawRequestModal").then((module) => ({
    default: module.PlaygroundRawRequestModal,
  })),
);

const LazyPlaygroundStartupTemplateModal = React.lazy(() =>
  import("./PlaygroundStartupTemplateModal").then((module) => ({
    default: module.PlaygroundStartupTemplateModal,
  })),
);

const LazyRolePlaySetupDrawer = React.lazy(() =>
  import("./RolePlaySetupDrawer").then((module) => ({
    default: module.RolePlaySetupDrawer,
  })),
);

const LazyPlaygroundContextWindowModal = React.lazy(() =>
  import("./PlaygroundContextWindowModal").then((module) => ({
    default: module.PlaygroundContextWindowModal,
  })),
);

const LazyPlaygroundMcpSettingsModal = React.lazy(() =>
  import("./PlaygroundMcpSettingsModal").then((module) => ({
    default: module.PlaygroundMcpSettingsModal,
  })),
);

const useIsomorphicLayoutEffect =
  typeof window !== "undefined" ? React.useLayoutEffect : React.useEffect;

export const PlaygroundForm = ({
  droppedFiles,
  attachedResearchContext = null,
  attachedResearchContextBaseline = null,
  attachedResearchContextPinned = null,
  attachedResearchContextHistory = [],
  onApplyAttachedResearchContext,
  onResetAttachedResearchContext,
  onRemoveAttachedResearchContext,
  onPinAttachedResearchContext,
  onUnpinAttachedResearchContext,
  onRestorePinnedResearchContext,
  onPinAttachedResearchContextHistory,
  onSelectAttachedResearchContextHistory,
  onPrepareResearchFollowUp,
  stickyDockEnabled = false,
  mobileCockpitModeActive = false,
  onComposerLayoutChange,
  onDraftPresenceChange,
  characterChatSendBlocker = null,
  characterChatModelUsability = null,
  characterChatModelUsabilityLabel = null,
  characterChatModelUsabilityTitle = null,
}: Props) => {
  const { t: translate } = useTranslation(["playground", "common", "option"]);
  const t = React.useCallback(
    (key: string, defaultValueOrOptions?: any, options?: any) =>
      translate(
        key as any,
        defaultValueOrOptions as any,
        options as any,
      ) as string,
    [translate],
  );
  const notificationApi = useAntdNotification();
  const navigate = useNavigate();
  const [attachedResearchContextDraft, setAttachedResearchContextDraft] =
    React.useState<AttachedResearchContext | null>(null);
  const [followUpResearchModalOpen, setFollowUpResearchModalOpen] =
    React.useState(false);
  const [
    includeAttachedResearchAsBackground,
    setIncludeAttachedResearchAsBackground,
  ] = React.useState(Boolean(attachedResearchContext));
  const [pendingAttachmentFollowUp, setPendingAttachmentFollowUp] =
    React.useState<ResearchFollowUpTarget | null>(null);
  const [followUpResearchPending, setFollowUpResearchPending] =
    React.useState(false);
  const followUpResearchPendingRef = React.useRef(false);
  const voiceChatSubmitFormRef = React.useRef<() => void>(() => undefined);

  const [checkWideMode] = useStorage("checkWideMode", false);
  const [allowExternalImages, setAllowExternalImages] = useStorage(
    "allowExternalImages",
    DEFAULT_CHAT_SETTINGS.allowExternalImages,
  );
  const [showMoodBadge, setShowMoodBadge] = useChatMoodBadgePreference();
  const researchContext = React.useMemo(
    () =>
      attachedResearchContext
        ? toChatResearchContext(attachedResearchContext)
        : undefined,
    [attachedResearchContext],
  );
  const {
    onSubmit,
    messages,
    selectedModel,
    selectedModelIsLoading,
    setSelectedModel,
    chatMode,
    setChatMode,
    compareMode,
    setCompareMode,
    compareFeatureEnabled,
    setCompareFeatureEnabled,
    compareSelectedModels,
    setCompareSelectedModels,
    compareMaxModels,
    setCompareMaxModels,
    speechToTextLanguage,
    stopStreamingRequest,
    streaming: isSending,
    webSearch,
    setWebSearch,
    toolChoice,
    setToolChoice,
    selectedQuickPrompt,
    textareaRef,
    setSelectedQuickPrompt,
    selectedSystemPrompt,
    setSelectedSystemPrompt,
    temporaryChat,
    setTemporaryChat,
    clearChat,
    useOCR,
    setUseOCR,
    defaultInternetSearchOn,
    setHistory,
    historyId,
    history,
    uploadedFiles,
    fileRetrievalEnabled,
    setFileRetrievalEnabled,
    handleFileUpload,
    removeUploadedFile,
    clearUploadedFiles,
    queuedMessages,
    setQueuedMessages,
    serverChatId,
    setServerChatId,
    serverChatState,
    setServerChatState,
    serverChatSource,
    setServerChatSource,
    setServerChatVersion,
    setServerChatCharacterId,
    setServerChatAssistantKind,
    setServerChatAssistantId,
    setServerChatPersonaMemoryMode,
    replyTarget,
    clearReplyTarget,
    ragPinnedResults,
    messageSteeringMode,
    messageSteeringForceNarrate,
    contextFiles,
    documentContext,
    selectedKnowledge,
    ragMediaIds,
    chatLoopState = {
      status: "idle",
      pendingApprovals: [],
      inflightToolCallIds: [],
    },
  } = useMessageOption();
  const setRagMediaIds = useStoreMessageOption((s) => s.setRagMediaIds);
  const setRagPinnedResults = useStoreMessageOption(
    (s) => s.setRagPinnedResults,
  );

  // Experimental Primer composer wire-up — opt in via ?nextgenComposer=1
  // query param OR the Settings "Enable new composer" toggle. When ON,
  // the existing ComposerTextarea + ComposerToolbar render inside
  // <ChatComposer variant={pref}> via the slot APIs.
  //
  // URL flag is read once at mount; the toggle uses the live-syncing
  // hook so cross-tab flips and server-profile hydrates flow through
  // without a reload.
  const [urlFlagEnabled] = React.useState(() => {
    if (typeof window === "undefined") return false;
    try {
      const params = new URLSearchParams(window.location.search);
      return params.get("nextgenComposer") === "1";
    } catch {
      return false;
    }
  });
  const [toggleEnabled] = useComposerEnabledPreference();
  const nextgenComposerEnabled = urlFlagEnabled || toggleEnabled;
  const [nextgenComposerVariant] = useComposerVariantPreference();
  const { settings: chatSettings, updateSettings: updateChatSettings } =
    useChatSettingsRecord({
      historyId,
      serverChatId,
    });
  const [imageEventSyncGlobalDefault, setImageEventSyncGlobalDefault] =
    useStorage<ImageGenerationEventSyncMode>(
      PLAYGROUND_IMAGE_EVENT_SYNC_DEFAULT_STORAGE_KEY,
      "off",
    );
  const imageEventSyncChatMode = React.useMemo(
    () =>
      normalizeImageGenerationEventSyncMode(
        chatSettings?.imageEventSyncMode,
        "off",
      ),
    [chatSettings?.imageEventSyncMode],
  );

  const [autoSubmitVoiceMessage] = useStorage("autoSubmitVoiceMessage", false);
  const isMobileViewport = useMobile();
  const suppressComposerToolbarForMobileCockpit =
    mobileCockpitModeActive && isMobileViewport;
  const mobileComposerViewport = useMobileComposerViewport(isMobileViewport);
  const [viewportHeightPx, setViewportHeightPx] = React.useState(() =>
    typeof window === "undefined" ? 0 : window.innerHeight,
  );
  const [openModelSettings, setOpenModelSettings] = React.useState(false);
  const [modelSettingsScopeOverride, setModelSettingsScopeOverride] =
    React.useState<string | null>(null);
  const modelSettingsReturnFocusSelectorRef = React.useRef<string | null>(null);
  const modelSelectorReturnFocusSelectorRef = React.useRef<string | null>(null);
  const mcpSettingsReturnFocusSelectorRef = React.useRef<string | null>(null);
  const [openActorSettings, setOpenActorSettings] = React.useState(false);
  const [rolePlaySetupOpen, setRolePlaySetupOpen] = React.useState(false);
  const [noticesExpanded, setNoticesExpanded] = React.useState(false);
  const actorSettings = useActorStore((state) => state.settings);
  const setActorSettings = useActorStore((state) => state.setSettings);
  const setActorPreviewAndTokens = useActorStore(
    (state) => state.setPreviewAndTokens,
  );
  const systemPrompt = useStoreChatModelSettings((state) => state.systemPrompt);
  const setSystemPrompt = useStoreChatModelSettings(
    (state) => state.setSystemPrompt,
  );
  const currentChatModelSettings = useStoreChatModelSettings((state) => ({
    temperature: state.temperature,
    numPredict: state.numPredict,
    topP: state.topP,
    topK: state.topK,
    frequencyPenalty: state.frequencyPenalty,
    presencePenalty: state.presencePenalty,
    repeatPenalty: state.repeatPenalty,
    reasoningEffort: state.reasoningEffort,
    historyMessageLimit: state.historyMessageLimit,
    historyMessageOrder: state.historyMessageOrder,
    slashCommandInjectionMode: state.slashCommandInjectionMode,
    apiProvider: state.apiProvider,
    extraHeaders: state.extraHeaders,
    extraBody: state.extraBody,
    llamaThinkingBudgetTokens: state.llamaThinkingBudgetTokens,
    llamaGrammarMode: state.llamaGrammarMode,
    llamaGrammarId: state.llamaGrammarId,
    llamaGrammarInline: state.llamaGrammarInline,
    llamaGrammarOverride: state.llamaGrammarOverride,
    jsonMode: state.jsonMode,
    systemPromptTemplateId: state.systemPromptTemplateId,
  }));
  const numCtx = useStoreChatModelSettings((state) => state.numCtx);
  const updateChatModelSetting = useStoreChatModelSettings(
    (state) => state.updateSetting,
  );
  const updateChatModelSettings = useStoreChatModelSettings(
    (state) => state.updateSettings,
  );
  const setActiveChatModelSettingsScope = useStoreChatModelSettings(
    (state) => state.setActiveSettingsScope,
  );
  const { data: promptLibrary = [] } = useQuery({
    queryKey: ["playgroundStartupPromptLibrary"],
    queryFn: getAllPrompts,
  });
  const {
    voiceChatEnabled,
    setVoiceChatEnabled,
    voiceChatModel,
    setVoiceChatModel,
    voiceChatPauseMs,
    setVoiceChatPauseMs,
    voiceChatTriggerPhrases,
    setVoiceChatTriggerPhrases,
    voiceChatAutoResume,
    setVoiceChatAutoResume,
    voiceChatBargeIn,
    setVoiceChatBargeIn,
    voiceChatTtsMode,
    setVoiceChatTtsMode,
  } = useVoiceChatSettings();
  const voiceChatMessages = useVoiceChatMessages();
  const { devices: audioInputDevices } = useAudioSourceCatalog();
  const {
    config: canonicalConnectionConfig,
    loading: canonicalConnectionLoading,
  } = useCanonicalConnectionConfig();
  const [ttsProvider] = useStorage("ttsProvider", "browser");
  const [tldwTtsModel] = useStorage("tldwTtsModel", "kokoro");
  const [tldwTtsVoice] = useStorage("tldwTtsVoice", "af_heart");
  const [tldwTtsSpeed] = useStorage("tldwTtsSpeed", 1);
  const [tldwTtsResponseFormat] = useStorage("tldwTtsResponseFormat", "mp3");
  const [openAITTSModel] = useStorage("openAITTSModel", "tts-1");
  const [openAITTSVoice] = useStorage("openAITTSVoice", "alloy");
  const [elevenLabsModel] = useStorage("elevenLabsModel", "");
  const [elevenLabsVoiceId] = useStorage("elevenLabsVoiceId", "");
  const [speechPlaybackSpeed] = useStorage("speechPlaybackSpeed", 1);
  const [voiceChatTriggerInput, setVoiceChatTriggerInput] = React.useState(
    voiceChatTriggerPhrases.join(", "),
  );
  React.useEffect(() => {
    const next = voiceChatTriggerPhrases.join(", ");
    setVoiceChatTriggerInput((prev) => (prev === next ? prev : next));
  }, [voiceChatTriggerPhrases]);

  const connectionState = useConnectionState();
  const { phase, isConnected } = connectionState;
  const connectionUxState = React.useMemo(
    () => deriveConnectionUxState(connectionState),
    [connectionState],
  );
  const setOptionalPanelVisible = useChatSurfaceCoordinatorStore(
    (state) => state.setPanelVisible,
  );
  const markOptionalPanelEngaged = useChatSurfaceCoordinatorStore(
    (state) => state.markPanelEngaged,
  );
  const mcpToolsEnabled = useChatSurfaceCoordinatorStore((state) =>
    shouldEnableOptionalResource(state, "mcp-tools"),
  );
  const audioHealthEnabled = useChatSurfaceCoordinatorStore((state) =>
    shouldEnableOptionalResource(state, "audio-health"),
  );
  const modelCatalogEnabled = useChatSurfaceCoordinatorStore((state) =>
    shouldEnableOptionalResource(state, "model-catalog"),
  );
  const isConnectionReady = isConnected && phase === ConnectionPhase.CONNECTED;
  const { capabilities, loading: capsLoading } = useServerCapabilities();
  const {
    hasMcp,
    healthState: mcpHealthState,
    discoveredTools: discoveredMcpTools,
    chatTools: chatMcpTools,
    toolCounts: mcpToolCounts,
    toolsLoading: mcpToolsLoading,
    catalogs: mcpCatalogs,
    catalogsLoading: mcpCatalogsLoading,
    toolCatalog,
    toolCatalogId,
    toolModules,
    moduleOptions,
    moduleOptionsLoading,
    toolCatalogStrict,
    setToolCatalog,
    setToolCatalogId,
    setToolModules,
    setToolCatalogStrict,
    setToolEnabled: setMcpToolEnabled,
    resetToolFilter: resetMcpToolFilter,
  } = useMcpTools({ enabled: mcpToolsEnabled });
  const mcpCtrl = useMcpToolsControl({
    hasMcp,
    mcpHealthState,
    mcpTools: chatMcpTools,
    mcpToolsLoading,
    mcpCatalogs,
    toolCatalog,
    toolCatalogId,
    setToolCatalog,
    setToolCatalogId,
    toolChoice,
  });
  const setMcpPopoverOpen = mcpCtrl.setMcpPopoverOpen;
  const { mcpSettingsOpen, setMcpSettingsOpen } = mcpCtrl;
  const restoreMcpSettingsFocus = React.useCallback(() => {
    const returnFocusSelector = mcpSettingsReturnFocusSelectorRef.current;
    if (!returnFocusSelector) return;
    mcpSettingsReturnFocusSelectorRef.current = null;

    scheduleFocusFirstVisibleElement(returnFocusSelector);
  }, []);
  const setMcpSettingsOpenWithFocusRestore = React.useCallback(
    (nextOpen: boolean) => {
      setMcpSettingsOpen(nextOpen);
      if (!nextOpen) {
        restoreMcpSettingsFocus();
      }
    },
    [restoreMcpSettingsFocus, setMcpSettingsOpen],
  );
  const handleModuleSelect = React.useCallback(
    (value?: string[]) => {
      setToolModules(Array.isArray(value) ? value : []);
    },
    [setToolModules],
  );
  const hasServerVoiceChat =
    isConnectionReady &&
    !capsLoading &&
    Boolean(
      capabilities?.hasVoiceChat ??
      (capabilities?.hasStt && capabilities?.hasTts),
    );
  const hasServerStt =
    isConnectionReady && !capsLoading && Boolean(capabilities?.hasStt);
  const shouldProbeAudioHealth = React.useMemo(
    () =>
      shouldProbeVoiceConversationAudioHealth({
        isConnectionReady,
        hasServerVoiceChat,
        hasServerStt,
        optionalAudioHealthEnabled: audioHealthEnabled,
      }),
    [audioHealthEnabled, hasServerStt, hasServerVoiceChat, isConnectionReady],
  );
  const {
    healthState: audioHealthState,
    sttHealthState,
    hasVoiceConversationTransport,
  } = useTldwAudioStatus({
    enabled: shouldProbeAudioHealth,
    ttsProvider,
    tldwTtsModel,
  });
  const canUseServerAudio =
    hasServerVoiceChat && audioHealthState !== "unhealthy";
  const canUseServerStt = hasServerStt && sttHealthState !== "unhealthy";
  const voiceConversationTtsConfig = React.useMemo(
    () =>
      resolveVoiceConversationTtsConfig({
        ttsProvider,
        tldwTtsModel,
        tldwTtsVoice,
        tldwTtsSpeed,
        tldwTtsResponseFormat,
        openAITTSModel,
        openAITTSVoice,
        elevenLabsModel,
        elevenLabsVoiceId,
        speechPlaybackSpeed,
        voiceChatTtsMode,
      }),
    [
      elevenLabsModel,
      elevenLabsVoiceId,
      openAITTSModel,
      openAITTSVoice,
      speechPlaybackSpeed,
      tldwTtsModel,
      tldwTtsResponseFormat,
      tldwTtsSpeed,
      tldwTtsVoice,
      ttsProvider,
      voiceChatTtsMode,
    ],
  );
  const voiceConversationAvailability = React.useMemo(
    () =>
      resolveVoiceConversationAvailability({
        isConnectionReady: isConnectionReady && !canonicalConnectionLoading,
        hasVoiceConversationTransport,
        authReady: Boolean(
          canonicalConnectionConfig?.serverUrl &&
          (canonicalConnectionConfig?.authMode === "multi-user"
            ? canonicalConnectionConfig.accessToken
            : canonicalConnectionConfig.apiKey),
        ),
        sttHealthState,
        ttsHealthState: audioHealthState,
        selectedModel,
        allowBackendDefaultModel: true,
        ttsConfigReady: voiceConversationTtsConfig.ok,
      }),
    [
      audioHealthState,
      canonicalConnectionConfig?.apiKey,
      canonicalConnectionConfig?.authMode,
      canonicalConnectionConfig?.accessToken,
      canonicalConnectionConfig?.serverUrl,
      canonicalConnectionLoading,
      hasVoiceConversationTransport,
      isConnectionReady,
      selectedModel,
      sttHealthState,
      voiceConversationTtsConfig.ok,
    ],
  );
  const voiceChatAvailable = voiceConversationAvailability.available;
  const voiceChatUnavailableReason = React.useMemo(() => {
    const fallback = t(
      "playground:voiceChat.unavailableBody",
      "Connect to a tldw server with audio chat streaming enabled.",
    );
    return voiceConversationAvailability.message
      ? t(voiceConversationAvailability.message, fallback)
      : fallback;
  }, [t, voiceConversationAvailability.message]);
  const voiceChat = useVoiceChatStream({
    active: voiceChatEnabled && voiceChatAvailable,
    onTranscript: (text) => {
      voiceChatMessages.beginTurn(text);
    },
    onAssistantDelta: (delta) => {
      voiceChatMessages.appendAssistantDelta(delta);
    },
    onAssistantMessage: (text) => {
      void voiceChatMessages.finalizeAssistant(text);
    },
    onError: (msg) => {
      const runtimeError = normalizeVoiceConversationRuntimeError(msg);
      notificationApi.error({
        message: t("playground:voiceChat.errorTitle", "Voice chat error"),
        description: runtimeError.message,
      });
      void voiceChatMessages.failTurn(runtimeError.reason);
      setVoiceChatEnabled(false);
    },
    onWarning: (msg) => {
      notificationApi.warning({
        message: t("playground:voiceChat.warningTitle", "Voice chat warning"),
        description: msg,
      });
    },
  });
  const [hasShownConnectBanner, setHasShownConnectBanner] =
    React.useState(false);
  const [showConnectBanner, setShowConnectBanner] = React.useState(false);
  const [documentGeneratorOpen, setDocumentGeneratorOpen] =
    React.useState(false);
  const [voiceModeSelectorOpen, setVoiceModeSelectorOpen] =
    React.useState(false);
  const [modeLauncherOpen, setModeLauncherOpen] = React.useState(false);
  const [modeAnnouncement, setModeAnnouncement] = React.useState<string | null>(
    null,
  );
  const previousPresetKeyRef = React.useRef<string | null>(null);
  const previousJsonModeRef = React.useRef<boolean | null>(null);
  const previousCharacterNameRef = React.useRef<string | null>(null);
  const [toolsPopoverOpen, setToolsPopoverOpen] = React.useState(false);
  const [attachmentMenuOpen, setAttachmentMenuOpen] = React.useState(false);
  const [sendMenuOpen, setSendMenuOpen] = React.useState(false);
  const [documentGeneratorSeed, setDocumentGeneratorSeed] = React.useState<{
    conversationId?: string | null;
    message?: string | null;
    messageId?: string | null;
  }>({});
  const [autoStopTimeout] = useStorage("autoStopTimeout", 2000);
  const [dictationAutoFallbackEnabled] = useStorage(
    "dictation_auto_fallback",
    false,
  );
  const [dictationModeOverride] = useStorage<DictationModePreference | null>(
    "dictationModeOverride",
    null,
  );
  const [sttModel] = useStorage("sttModel", "whisper-1");
  const [sttUseSegmentation] = useStorage("sttUseSegmentation", false);
  const [sttTimestampGranularities] = useStorage(
    "sttTimestampGranularities",
    "segment",
  );
  const [sttPrompt] = useStorage("sttPrompt", "");
  const [sttTask] = useStorage("sttTask", "transcribe");
  const [sttResponseFormat] = useStorage("sttResponseFormat", "json");
  const [sttTemperature] = useStorage("sttTemperature", 0);
  const [sttSegK] = useStorage("sttSegK", 6);
  const [sttSegMinSegmentSize] = useStorage("sttSegMinSegmentSize", 5);
  const [sttSegLambdaBalance] = useStorage("sttSegLambdaBalance", 0.01);
  const [sttSegUtteranceExpansionWidth] = useStorage(
    "sttSegUtteranceExpansionWidth",
    2,
  );
  const [sttSegEmbeddingsProvider] = useStorage("sttSegEmbeddingsProvider", "");
  const [sttSegEmbeddingsModel] = useStorage("sttSegEmbeddingsModel", "");
  const [selectedAssistant, setSelectedAssistant] = useSelectedAssistant(null);
  const selectedAssistantMode = React.useMemo(
    () => getAssistantSelectionMode(selectedAssistant),
    [selectedAssistant]
  );
  const selectedCharacter = React.useMemo(
    () => assistantSelectionToCharacter<Character>(selectedAssistant),
    [selectedAssistant]
  );
  const setSelectedCharacter = React.useCallback(
    async (next: Character | null) => {
      const nextSelection = characterToAssistantSelection(
        next as (Character & Record<string, unknown>) | null
      );
      if (nextSelection && selectedAssistantMode) {
        nextSelection.metadata = {
          ...(nextSelection.metadata ?? {}),
          selectionMode: selectedAssistantMode,
        };
      }
      await setSelectedAssistant(nextSelection);
    },
    [selectedAssistantMode, setSelectedAssistant]
  );
  const pendingAssistantState = React.useMemo(
    () =>
      resolveEffectiveAssistantState({
        settings: chatSettings ?? null,
        draftSelection: selectedAssistant
      }),
    [chatSettings, selectedAssistant]
  );
  const [defaultCharacter, setDefaultCharacter] = useStorage<Character | null>(
    {
      key: DEFAULT_CHARACTER_STORAGE_KEY,
      instance: defaultCharacterStorage,
    },
    null,
  );
  const { data: defaultCharacterPreference } =
    useQuery<DefaultCharacterPreferenceQueryResult>({
      queryKey: ["tldw:defaultCharacterPreference:playground"],
      queryFn: async () => {
        await tldwClient.initialize();
        const defaultCharacterId =
          await tldwClient.getDefaultCharacterPreference();
        return { defaultCharacterId };
      },
      staleTime: 60 * 1000,
      throwOnError: false,
    });
  const [showMoodConfidence, setShowMoodConfidence] = useStorage(
    "chatShowMoodConfidence",
    Boolean(selectedCharacter?.id) && !compareMode,
  );
  const [startupTemplatesRaw, setStartupTemplatesRaw] = useStorage(
    "playgroundStartupTemplateBundles",
    "[]",
  );
  const [serverPersistenceHintSeen, setServerPersistenceHintSeen] = useStorage(
    "serverPersistenceHintSeen",
    false,
  );
  // showServerPersistenceHint and serverSaveInFlightRef moved to usePlaygroundPersistence
  const uiMode = useUiModeStore((state) => state.mode);
  const isProMode = uiMode === "pro";

  React.useEffect(() => {
    if (typeof window === "undefined") return;

    const updateViewportHeight = () => {
      setViewportHeightPx((previousHeightPx) => {
        const nextHeightPx = window.innerHeight;
        return previousHeightPx === nextHeightPx
          ? previousHeightPx
          : nextHeightPx;
      });
    };

    updateViewportHeight();
    window.addEventListener("resize", updateViewportHeight);
    window.visualViewport?.addEventListener("resize", updateViewportHeight);

    return () => {
      window.removeEventListener("resize", updateViewportHeight);
      window.visualViewport?.removeEventListener(
        "resize",
        updateViewportHeight,
      );
    };
  }, []);

  const [contextToolsOpen, setContextToolsOpen] = useStorage(
    "playgroundKnowledgeSearchOpen",
    false,
  );
  const [simpleInternetSearch, setSimpleInternetSearch] = useStorage(
    "isSimpleInternetSearch",
    true,
  );
  const [, setDefaultInternetSearchOnSetting] = useStorage(
    "defaultInternetSearchOn",
    false,
  );
  const [knowledgePanelTab, setKnowledgePanelTab] =
    React.useState<KnowledgeTab>("search");
  const [knowledgePanelTabRequestId, setKnowledgePanelTabRequestId] =
    React.useState(0);
  const [lastSubmittedContext, setLastSubmittedContext] = React.useState<{
    model: string | null;
    compareEnabled: boolean;
    compareCount: number;
    characterName: string | null;
    promptSummary: string;
    jsonMode: boolean;
    temporaryChat: boolean;
    webSearch: boolean;
    contextToolsOpen: boolean;
  } | null>(null);
  const replyLabel = replyTarget
    ? [
        t("common:replyingTo", "Replying to"),
        replyTarget.name ? `${replyTarget.name}:` : null,
        replyTarget.preview,
      ]
        .filter(Boolean)
        .join(" ")
    : "";

  const storedCharacterId = React.useMemo(
    () => resolveCharacterSelectionId(selectedCharacter),
    [selectedCharacter],
  );
  const localDefaultCharacterId = React.useMemo(
    () => resolveCharacterSelectionId(defaultCharacter),
    [defaultCharacter],
  );
  const serverDefaultCharacterId =
    defaultCharacterPreference?.defaultCharacterId;
  const effectiveDefaultCharacter = React.useMemo<Character | null>(() => {
    if (typeof serverDefaultCharacterId === "undefined") {
      return defaultCharacter;
    }
    if (!serverDefaultCharacterId) {
      return null;
    }
    if (
      localDefaultCharacterId === serverDefaultCharacterId &&
      defaultCharacter
    ) {
      return defaultCharacter;
    }
    return { id: serverDefaultCharacterId } as Character;
  }, [defaultCharacter, localDefaultCharacterId, serverDefaultCharacterId]);
  const effectiveDefaultCharacterId = React.useMemo(
    () => resolveCharacterSelectionId(effectiveDefaultCharacter),
    [effectiveDefaultCharacter],
  );
  const isFreshChat = React.useMemo(
    () => isFreshChatState(serverChatId, messages.length),
    [messages.length, serverChatId],
  );
  const defaultCharacterBootstrapAppliedRef = React.useRef(false);
  const previousFreshChatRef = React.useRef(isFreshChat);

  React.useEffect(() => {
    if (typeof serverDefaultCharacterId === "undefined") return;

    if (!serverDefaultCharacterId) {
      if (localDefaultCharacterId) {
        void setDefaultCharacter(null);
      }
      return;
    }

    if (localDefaultCharacterId === serverDefaultCharacterId) return;
    void setDefaultCharacter({ id: serverDefaultCharacterId } as Character);
  }, [localDefaultCharacterId, serverDefaultCharacterId, setDefaultCharacter]);

  React.useEffect(() => {
    defaultCharacterBootstrapAppliedRef.current = false;
  }, [effectiveDefaultCharacterId]);

  React.useEffect(() => {
    if (
      shouldResetDefaultCharacterBootstrap(
        previousFreshChatRef.current,
        isFreshChat,
      )
    ) {
      defaultCharacterBootstrapAppliedRef.current = false;
    }
    previousFreshChatRef.current = isFreshChat;
  }, [isFreshChat]);

  React.useEffect(() => {
    if (!effectiveDefaultCharacter || !effectiveDefaultCharacterId) return;
    if (
      !shouldApplyDefaultCharacter({
        defaultCharacterId: effectiveDefaultCharacterId,
        selectedCharacterId: storedCharacterId,
        isFreshChat,
        hasAppliedInSession: defaultCharacterBootstrapAppliedRef.current,
      })
    ) {
      return;
    }

    defaultCharacterBootstrapAppliedRef.current = true;
    void setSelectedCharacter(effectiveDefaultCharacter);
  }, [
    effectiveDefaultCharacter,
    effectiveDefaultCharacterId,
    isFreshChat,
    setSelectedCharacter,
    storedCharacterId,
  ]);

  const restoreModelSettingsFocus = React.useCallback(() => {
    const returnFocusSelector = modelSettingsReturnFocusSelectorRef.current;
    if (!returnFocusSelector) return;
    modelSettingsReturnFocusSelectorRef.current = null;

    scheduleFocusFirstVisibleElement(returnFocusSelector);
  }, []);

  const setOpenModelSettingsWithFocusRestore = React.useCallback(
    (nextOpen: boolean) => {
      setOpenModelSettings(nextOpen);
      if (!nextOpen) {
        setModelSettingsScopeOverride(null);
        restoreModelSettingsFocus();
      }
    },
    [restoreModelSettingsFocus],
  );

  React.useEffect(() => {
    if (typeof window === "undefined") return;
    const handler = () => setOpenActorSettings(true);
    window.addEventListener(OPEN_ACTOR_SETTINGS_EVENT, handler);
    return () => {
      window.removeEventListener(OPEN_ACTOR_SETTINGS_EVENT, handler);
    };
  }, []);

  React.useEffect(() => {
    if (typeof window === "undefined") return;
    const handler = (event: Event) => {
      const detail = (event as CustomEvent<ModelSettingsOpenDetail>).detail;
      modelSettingsReturnFocusSelectorRef.current = normalizeFocusSelector(
        detail?.returnFocusSelector,
      );
      setModelSettingsScopeOverride(
        typeof detail?.settingsScope === "string" && detail.settingsScope.trim()
          ? detail.settingsScope.trim()
          : null,
      );
      setOpenModelSettings(true);
    };
    window.addEventListener(OPEN_MODEL_SETTINGS_EVENT, handler);
    return () => {
      window.removeEventListener(OPEN_MODEL_SETTINGS_EVENT, handler);
    };
  }, []);

  React.useEffect(() => {
    if (!modeAnnouncement) return;
    const timer = window.setTimeout(() => {
      setModeAnnouncement(null);
    }, 3000);
    return () => {
      window.clearTimeout(timer);
    };
  }, [modeAnnouncement]);

  React.useEffect(() => {
    if (typeof window === "undefined") return;
    const handler = (event: Event) => {
      const detail = (event as CustomEvent)?.detail || {};
      setDocumentGeneratorSeed({
        conversationId: detail?.conversationId ?? serverChatId ?? null,
        message: detail?.message ?? null,
        messageId: detail?.messageId ?? null,
      });
      setDocumentGeneratorOpen(true);
    };
    window.addEventListener("tldw:open-document-generator", handler);
    return () => {
      window.removeEventListener("tldw:open-document-generator", handler);
    };
  }, [serverChatId]);

  const {
    tabMentionsEnabled,
    showMentions,
    mentionPosition,
    filteredTabs,
    availableTabs,
    selectedDocuments,
    handleTextChange,
    insertMention,
    closeMentions,
    addDocument,
    removeDocument,
    clearSelectedDocuments,
    reloadTabs,
    handleMentionsOpen,
  } = useTabMentions(textareaRef);

  const { data: composerModels } = useQuery({
    queryKey: ["playground:chatModels"],
    queryFn: () => fetchChatModels({ returnEmpty: true }),
    enabled: modelCatalogEnabled,
  });
  const { data: imageModels = [] } = useQuery({
    queryKey: ["playground:imageModels"],
    queryFn: () => fetchImageModels({ returnEmpty: true }),
    enabled: true,
  });

  const {
    modelDropdownOpen,
    setModelDropdownOpen,
    modelSearchQuery,
    setModelSearchQuery,
    modelSortMode,
    setModelSortMode,
    modelListScope = "configured",
    setModelListScope = () => undefined,
    selectedModelMeta,
    selectedModelKey,
    modelContextLength,
    modelCapabilities,
    resolvedMaxContext,
    resolvedProviderKey,
    providerLabel,
    modelSummaryLabel,
    apiModelLabel,
    modelSelectorWarning,
    favoriteModels,
    favoriteModelsIsLoading,
    favoriteModelSet,
    toggleFavoriteModel,
    filteredModels,
    modelDropdownMenuItems,
    isSmallModel,
  } = useModelSelector({
    composerModels,
    selectedModel,
    setSelectedModel,
    navigate,
  });
  const restoreModelSelectorFocus = React.useCallback(() => {
    const returnFocusSelector = modelSelectorReturnFocusSelectorRef.current;
    if (!returnFocusSelector) return;
    modelSelectorReturnFocusSelectorRef.current = null;

    scheduleFocusFirstVisibleElement(returnFocusSelector);
  }, []);
  const setModelDropdownOpenWithFocusRestore = React.useCallback(
    (nextOpen: boolean) => {
      setModelDropdownOpen(nextOpen);
      if (!nextOpen) {
        restoreModelSelectorFocus();
      }
    },
    [restoreModelSelectorFocus, setModelDropdownOpen],
  );
  React.useEffect(() => {
    setActiveChatModelSettingsScope(selectedModelKey ?? null);
  }, [selectedModelKey, setActiveChatModelSettingsScope]);

  const closeComposerPopoversExcept = React.useCallback(
    (activePopover: ComposerPopoverKey) => {
      if (activePopover !== "context") {
        setContextToolsOpen(false);
      }
      if (activePopover !== "model") {
        setModelDropdownOpenWithFocusRestore(false);
      }
      if (activePopover !== "mcp") {
        setMcpPopoverOpen(false);
      }
      if (activePopover !== "tools") {
        setToolsPopoverOpen(false);
      }
      if (activePopover !== "attachment") {
        setAttachmentMenuOpen(false);
      }
      if (activePopover !== "send") {
        setSendMenuOpen(false);
      }
    },
    [
      setAttachmentMenuOpen,
      setContextToolsOpen,
      setMcpPopoverOpen,
      setModelDropdownOpenWithFocusRestore,
      setSendMenuOpen,
      setToolsPopoverOpen,
    ],
  );

  React.useEffect(() => {
    if (typeof window === "undefined") return;
    const handler = () => {
      closeComposerPopoversExcept("mcp");
      setMcpPopoverOpen(true);
    };
    window.addEventListener(OPEN_MCP_TOOLS_EVENT, handler);
    return () => {
      window.removeEventListener(OPEN_MCP_TOOLS_EVENT, handler);
    };
  }, [closeComposerPopoversExcept, setMcpPopoverOpen]);

  React.useEffect(() => {
    if (typeof window === "undefined") return;
    const handler = (event: Event) => {
      const detail = (event as CustomEvent<McpSettingsOpenDetail>).detail;
      mcpSettingsReturnFocusSelectorRef.current = normalizeFocusSelector(
        detail?.returnFocusSelector,
      );
      closeComposerPopoversExcept("mcp");
      setMcpPopoverOpen(false);
      setMcpSettingsOpen(true);
    };
    window.addEventListener(OPEN_MCP_SETTINGS_EVENT, handler);
    return () => {
      window.removeEventListener(OPEN_MCP_SETTINGS_EVENT, handler);
    };
  }, [closeComposerPopoversExcept, setMcpPopoverOpen, setMcpSettingsOpen]);

  React.useEffect(() => {
    if (typeof window === "undefined") return;
    const handler = () => {
      closeComposerPopoversExcept("tools");
      setToolsPopoverOpen(true);
    };
    window.addEventListener(OPEN_TURN_TOOLS_EVENT, handler);
    return () => {
      window.removeEventListener(OPEN_TURN_TOOLS_EVENT, handler);
    };
  }, [closeComposerPopoversExcept, setToolsPopoverOpen]);

  React.useEffect(() => {
    setOptionalPanelVisible("model-catalog", modelDropdownOpen);
    if (modelDropdownOpen) {
      markOptionalPanelEngaged("model-catalog");
    }

    return () => {
      setOptionalPanelVisible("model-catalog", false);
    };
  }, [markOptionalPanelEngaged, modelDropdownOpen, setOptionalPanelVisible]);

  React.useEffect(() => {
    setOptionalPanelVisible("audio-health", voiceChatEnabled);
    if (voiceChatEnabled) {
      markOptionalPanelEngaged("audio-health");
    }

    return () => {
      setOptionalPanelVisible("audio-health", false);
    };
  }, [markOptionalPanelEngaged, setOptionalPanelVisible, voiceChatEnabled]);

  // Auto-select model on initial load when no model is selected
  // Priority: 1) First favorite model, 2) First available model
  React.useEffect(() => {
    const nextModel = resolveStartupSelectedModel({
      currentModel: selectedModel,
      models: (composerModels as any[]) || [],
      preferredModelIds: favoriteModels,
      isCurrentModelHydrating: selectedModelIsLoading,
      arePreferencesHydrating: favoriteModelsIsLoading,
    });
    if (nextModel) {
      setSelectedModel(nextModel);
    }
  }, [
    composerModels,
    favoriteModels,
    favoriteModelsIsLoading,
    selectedModel,
    selectedModelIsLoading,
    setSelectedModel,
  ]);

  const hasPromptContext = React.useMemo(
    () =>
      Boolean(selectedSystemPrompt) ||
      Boolean(selectedQuickPrompt) ||
      String(systemPrompt || "").trim().length > 0,
    [selectedQuickPrompt, selectedSystemPrompt, systemPrompt],
  );

  const modelComparison = useModelComparison({
    composerModels,
    selectedModel,
    setSelectedModel,
    compareFeatureEnabled,
    compareMode,
    setCompareMode,
    compareSelectedModels,
    setCompareSelectedModels,
    compareMaxModels,
    selectedCharacterName: selectedCharacter?.name || null,
    ragPinnedResultsLength: ragPinnedResults.length,
    webSearch,
    hasPromptContext,
    jsonMode: Boolean(currentChatModelSettings.jsonMode),
    voiceChatEnabled,
    t,
  });
  const {
    compareModeActive,
    compareModelMetaById,
    availableCompareModels,
    compareModelLabelById,
    compareSelectedModelLabels,
    compareNeedsMoreModels,
    compareModelsSupportCapability,
    compareCapabilityIncompatibilities,
    toggleCompareMode,
    handleAddCompareModel,
    handleRemoveCompareModel,
    sendLabel,
  } = modelComparison;

  const voiceChatModelOptions = React.useMemo(() => {
    const options = [
      {
        value: "",
        label: t("playground:voiceChat.useChatModel", "Use chat model"),
      },
    ];
    const models = (composerModels as any[]) || [];
    for (const model of models) {
      const pLabel = getProviderDisplayName(model.provider || "");
      const modelLabel = model.nickname || model.model || model.name;
      const label = pLabel ? `${pLabel} - ${modelLabel}` : modelLabel;
      options.push({
        value: model.model || model.name,
        label,
      });
    }
    return options;
  }, [composerModels, t]);

  const promptTemplates = usePromptTemplates({
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
    setSelectedAssistant,
    setRagPinnedResults,
    updateChatModelSettings,
    compareModeActive,
    setCompareSelectedModels,
    setModeAnnouncement,
    t,
  });
  const {
    currentPresetKey,
    currentPreset,
    startupTemplates,
    startupTemplateDraftName,
    setStartupTemplateDraftName,
    startupTemplatePreview,
    setStartupTemplatePreview,
    startupTemplateNameFallback,
    selectedSystemPromptRecord,
    handleSaveStartupTemplate,
    handleOpenStartupTemplatePreview,
    handleApplyStartupTemplate,
    handleDeleteStartupTemplate,
    handleTemplateSelect,
    promptSummaryLabel,
    handleSaveRolePlaySetup,
    handleApplySavedRolePlaySetup,
    handleRenameStartupTemplate,
  } = promptTemplates;
  React.useEffect(() => {
    if (previousPresetKeyRef.current == null) {
      previousPresetKeyRef.current = currentPresetKey;
      return;
    }
    if (previousPresetKeyRef.current !== currentPresetKey) {
      if (currentPresetKey === "custom") {
        setModeAnnouncement(
          t(
            "playground:composer.presetChangedCustom",
            "Preset switched to Custom.",
          ),
        );
      } else {
        const presetLabel = currentPreset
          ? t(
              `playground:presets.${currentPreset.key}.label`,
              currentPreset.label,
            )
          : currentPresetKey;
        setModeAnnouncement(
          toText(
            t(
              "playground:composer.presetChanged",
              "{{preset}} preset applied.",
              {
                preset: presetLabel,
              } as any,
            ),
          ),
        );
      }
    }
    previousPresetKeyRef.current = currentPresetKey;
  }, [currentPreset, currentPresetKey, t]);
  const isJsonModeActive = Boolean(currentChatModelSettings.jsonMode);
  React.useEffect(() => {
    if (previousJsonModeRef.current == null) {
      previousJsonModeRef.current = isJsonModeActive;
      return;
    }
    if (previousJsonModeRef.current !== isJsonModeActive) {
      setModeAnnouncement(
        isJsonModeActive
          ? t("playground:composer.jsonModeEnabledNotice", "JSON mode enabled.")
          : t(
              "playground:composer.jsonModeDisabledNotice",
              "JSON mode disabled.",
            ),
      );
    }
    previousJsonModeRef.current = isJsonModeActive;
  }, [isJsonModeActive, t]);
  React.useEffect(() => {
    const currentCharacterName =
      typeof selectedCharacter?.name === "string" &&
      selectedCharacter.name.trim().length > 0
        ? selectedCharacter.name.trim()
        : null;
    if (previousCharacterNameRef.current == null) {
      previousCharacterNameRef.current = currentCharacterName;
      return;
    }
    if (currentCharacterName !== previousCharacterNameRef.current) {
      setModeAnnouncement(
        currentCharacterName
          ? t(
              "playground:composer.characterAppliesNextTurn",
              "Character updates apply on the next turn.",
            )
          : t(
              "playground:composer.characterClearedNotice",
              "Character mode cleared.",
            ),
      );
    }
    previousCharacterNameRef.current = currentCharacterName;
  }, [selectedCharacter?.name, t]);
  const connectionStatusLabel = React.useMemo(() => {
    if (!isConnectionReady) {
      return t("playground:composer.providerStatusOffline", "Offline");
    }
    if (connectionUxState === "connected_degraded") {
      return t(
        "playground:composer.providerStatusDegraded",
        getDesignSystemState("degraded")?.label ?? "",
      );
    }
    return t("playground:composer.providerStatusHealthy", "Healthy");
  }, [connectionUxState, isConnectionReady, t]);
  const isSessionDegraded = React.useMemo(
    () => !isConnectionReady || connectionUxState === "connected_degraded",
    [connectionUxState, isConnectionReady],
  );
  const currentContextSnapshot = React.useMemo(
    () => ({
      model: selectedModel || null,
      compareEnabled: compareModeActive,
      compareCount: compareSelectedModels.length,
      characterName: selectedCharacter?.name || null,
      promptSummary: promptSummaryLabel,
      jsonMode: Boolean(currentChatModelSettings.jsonMode),
      temporaryChat,
      webSearch,
      contextToolsOpen,
    }),
    [
      compareModeActive,
      compareSelectedModels.length,
      contextToolsOpen,
      currentChatModelSettings.jsonMode,
      promptSummaryLabel,
      selectedCharacter?.name,
      selectedModel,
      temporaryChat,
      webSearch,
    ],
  );
  const contextDeltaLabels = React.useMemo(() => {
    if (!lastSubmittedContext) return [];
    const deltas: string[] = [];
    if (lastSubmittedContext.model !== currentContextSnapshot.model) {
      deltas.push(t("playground:composer.delta.model", "Model changed"));
    }
    if (
      lastSubmittedContext.compareEnabled !==
        currentContextSnapshot.compareEnabled ||
      lastSubmittedContext.compareCount !== currentContextSnapshot.compareCount
    ) {
      deltas.push(
        t("playground:composer.delta.compare", "Compare settings changed"),
      );
    }
    if (
      lastSubmittedContext.characterName !==
      currentContextSnapshot.characterName
    ) {
      deltas.push(
        t("playground:composer.delta.character", "Character changed"),
      );
    }
    if (
      lastSubmittedContext.promptSummary !==
      currentContextSnapshot.promptSummary
    ) {
      deltas.push(
        t("playground:composer.delta.prompt", "Prompt settings changed"),
      );
    }
    if (lastSubmittedContext.jsonMode !== currentContextSnapshot.jsonMode) {
      deltas.push(t("playground:composer.delta.json", "JSON mode changed"));
    }
    if (
      lastSubmittedContext.temporaryChat !==
      currentContextSnapshot.temporaryChat
    ) {
      deltas.push(
        t("playground:composer.delta.temporary", "Save mode changed"),
      );
    }
    if (lastSubmittedContext.webSearch !== currentContextSnapshot.webSearch) {
      deltas.push(
        t("playground:composer.delta.webSearch", "Web search changed"),
      );
    }
    if (
      lastSubmittedContext.contextToolsOpen !==
      currentContextSnapshot.contextToolsOpen
    ) {
      deltas.push(
        t(
          "playground:composer.delta.knowledge",
          "Knowledge panel state changed",
        ),
      );
    }
    return deltas;
  }, [currentContextSnapshot, lastSubmittedContext, t]);
  const characterPendingApply = React.useMemo(() => {
    const currentName =
      typeof selectedCharacter?.name === "string"
        ? selectedCharacter.name.trim()
        : "";
    const previousName =
      typeof lastSubmittedContext?.characterName === "string"
        ? lastSubmittedContext.characterName.trim()
        : "";
    if (!currentName) return false;
    if (!lastSubmittedContext) return false;
    return currentName !== previousName;
  }, [lastSubmittedContext, selectedCharacter?.name]);
  const selectedCharacterGreeting = React.useMemo(() => {
    const raw =
      typeof selectedCharacter?.greeting === "string"
        ? selectedCharacter.greeting
        : "";
    const trimmed = raw.trim();
    return trimmed.length > 0 ? trimmed : null;
  }, [selectedCharacter?.greeting]);

  // Enable focus shortcuts (Shift+Esc to focus textarea)
  useFocusShortcuts(textareaRef, true);

  const [pasteLargeTextAsFile] = useStorage("pasteLargeTextAsFile", false);

  const msgCollapse = useMessageCollapse({ textareaRef });
  const {
    isMessageCollapsed,
    setIsMessageCollapsed,
    collapsedRange,
    setCollapsedRange,
    hasExpandedLargeText,
    setHasExpandedLargeText,
    pendingCaretRef,
    lastDisplaySelectionRef,
    pendingCollapsedStateRef,
    pointerDownRef,
    selectionFromPointerRef,
    normalizeCollapsedRange,
    parseCollapsedRange,
    buildCollapsedMessageLabel,
    getCollapsedDisplayMeta,
    getDisplayCaretFromMessage,
    getMessageCaretFromDisplay,
    collapseLargeMessage,
    expandLargeMessage,
    restoreMessageValue: restoreCollapseState,
  } = msgCollapse;

  const stickyTextareaMaxHeight = React.useMemo(() => {
    if (!stickyDockEnabled) {
      return undefined;
    }

    return resolveStickyComposerTextareaMaxHeight({
      viewportHeightPx,
      keyboardInsetPx: mobileComposerViewport.keyboardInsetPx,
      isMobileViewport,
      defaultMaxHeightPx: isProMode
        ? PRO_COMPOSER_MAX_HEIGHT_PX
        : CASUAL_COMPOSER_MAX_HEIGHT_PX,
    });
  }, [
    isMobileViewport,
    isProMode,
    mobileComposerViewport.keyboardInsetPx,
    stickyDockEnabled,
    viewportHeightPx,
  ]);

  const composerInput = useComposerInput({
    textareaRef,
    isMessageCollapsed,
    setIsMessageCollapsed,
    collapsedRange,
    setCollapsedRange,
    hasExpandedLargeText,
    setHasExpandedLargeText,
    collapseLargeMessage,
    restoreCollapseState,
    getCollapsedDisplayMeta,
    getDisplayCaretFromMessage,
    getMessageCaretFromDisplay,
    normalizeCollapsedRange,
    expandLargeMessage,
    pendingCaretRef,
    lastDisplaySelectionRef,
    pendingCollapsedStateRef,
    pointerDownRef,
    selectionFromPointerRef,
    tabMentionsEnabled,
    handleTextChange,
    isProMode,
    textareaMaxHeightOverride: stickyTextareaMaxHeight,
  });
  const {
    form,
    typing,
    setMessageValue,
    restoreMessageValue,
    messageDisplayValue,
    collapsedDisplayMeta,
    textAreaFocus,
    syncCollapsedCaret,
    commitCollapsedEdit,
    replaceCollapsedRange,
    handleCompositionStart,
    handleCompositionEnd,
    handleTextareaMouseDown,
    handleTextareaMouseUp,
    handleTextareaChange,
    handleTextareaSelect,
    markComposerPerf,
    measureComposerPerf,
    onComposerRenderProfile,
    wrapComposerProfile,
    draftSaved,
  } = composerInput;

  const hasDraft = (form.values.message || "").trim().length > 0;
  const {
    importedContext: importedSidepanelContext,
    removeImportedContext: clearImportedSidepanelContext,
    conflict: sidepanelHandoffConflict,
    insertHandoffDraft,
    replaceWithHandoffDraft,
    cancelHandoffImport,
  } = useSidepanelChatHandoffImport({
    draftValue: form.values.message || "",
    setMessageValue,
    notificationApi,
    t,
  });
  useIsomorphicLayoutEffect(() => {
    onDraftPresenceChange?.(hasDraft);
  }, [hasDraft, onDraftPresenceChange]);

  const { deferredInput: deferredComposerInput } = useDeferredComposerInput(
    form.values.message || "",
  );

  const {
    draftTokenCount,
    conversationTokenCount,
    tokenUsageLabel,
    tokenUsageCompactLabel,
    tokenUsageTooltip,
    estimateTokensForText,
  } = useComposerTokens({
    message: form.values.message || "",
    messages,
    systemPrompt,
    resolvedMaxContext,
    apiModelLabel,
    isSending,
  });
  const tokenUsageDisplay = isProMode
    ? tokenUsageLabel
    : tokenUsageCompactLabel;

  const contextWindow = useContextWindow({
    draftTokenCount,
    conversationTokenCount,
    resolvedMaxContext,
    modelContextLength,
    numCtx,
    updateChatModelSetting,
    selectedCharacter,
    systemPrompt,
    selectedQuickPrompt,
    selectedSystemPrompt,
    ragPinnedResults,
    messages,
    selectedModel,
    resolvedProviderKey,
    deferredComposerInput,
    modelCapabilities,
    webSearch,
    jsonMode: Boolean(currentChatModelSettings.jsonMode),
    hasImageAttachment: Boolean(form.values.image),
    measureComposerPerf,
    t,
  });
  const {
    contextWindowModalOpen,
    setContextWindowModalOpen,
    contextWindowDraftValue,
    setContextWindowDraftValue,
    sessionInsightsOpen,
    setSessionInsightsOpen,
    sessionUsageSummary,
    sessionUsageLabel,
    sessionInsights,
    projectedBudget,
    tokenBudgetRisk,
    tokenBudgetRiskLabel,
    showTokenBudgetWarning,
    tokenBudgetWarningText,
    characterContextTokenEstimate,
    systemPromptTokenEstimate,
    pinnedSourceTokenEstimate,
    historyTokenEstimate,
    summaryCheckpointSuggestion,
    modelRecommendations,
    visibleModelRecommendations,
    dismissModelRecommendation,
    contextFootprintRows,
    nonMessageContextTokenEstimate,
    nonMessageContextPercent,
    showNonMessageContextWarning,
    largestContextContributor,
    formatContextWindowValue,
    isContextWindowOverrideActive,
    requestedContextWindowOverride,
    isContextWindowOverrideClamped,
    openContextWindowModal,
    saveContextWindowSetting,
    resetContextWindowSetting,
    openSessionInsightsModal,
  } = contextWindow;
  const handleModelRecommendationAction = React.useCallback(
    (action: ModelRecommendationAction) => {
      if (action === "open_model_settings") {
        setOpenModelSettings(true);
        return;
      }
      if (action === "enable_json_mode") {
        if (!currentChatModelSettings.jsonMode) {
          updateChatModelSetting("jsonMode", true);
        }
        setOpenModelSettings(true);
        return;
      }
      if (action === "open_context_window") {
        openContextWindowModal();
        return;
      }
      if (action === "open_session_insights") {
        openSessionInsightsModal();
      }
    },
    [
      currentChatModelSettings.jsonMode,
      openContextWindowModal,
      openSessionInsightsModal,
      setOpenModelSettings,
      updateChatModelSetting,
    ],
  );
  const openModelApiSelector = React.useCallback(() => {
    closeComposerPopoversExcept("model");
    setModelDropdownOpenWithFocusRestore(true);
  }, [closeComposerPopoversExcept, setModelDropdownOpenWithFocusRestore]);
  React.useEffect(() => {
    if (typeof window === "undefined") return;
    const handler = (event: Event) => {
      const detail = (event as CustomEvent<ModelSelectorOpenDetail>).detail;
      modelSelectorReturnFocusSelectorRef.current = normalizeFocusSelector(
        detail?.returnFocusSelector,
      );
      openModelApiSelector();
      scheduleFocusFirstVisibleElement("[data-testid='model-selector']");
    };
    window.addEventListener(OPEN_MODEL_SELECTOR_EVENT, handler);
    return () => {
      window.removeEventListener(OPEN_MODEL_SELECTOR_EVENT, handler);
    };
  }, [openModelApiSelector]);
  const getModelRecommendationActionLabel = React.useCallback(
    (action: ModelRecommendationAction) => {
      if (action === "enable_json_mode") {
        return t("playground:composer.recommendationEnableJson", "Enable JSON");
      }
      if (action === "open_context_window") {
        return t(
          "playground:composer.recommendationAdjustContext",
          "Adjust context",
        );
      }
      if (action === "open_session_insights") {
        return t(
          "playground:composer.recommendationOpenInsights",
          "Open insights",
        );
      }
      return t(
        "playground:composer.recommendationReviewModels",
        "Review models",
      );
    },
    [t],
  );
  const clearPromptContext = React.useCallback(() => {
    setSelectedQuickPrompt(null);
    setSelectedSystemPrompt("");
    setSystemPrompt("");
    updateChatModelSetting("systemPromptTemplateId", undefined);
  }, [
    setSelectedQuickPrompt,
    setSelectedSystemPrompt,
    setSystemPrompt,
    updateChatModelSetting,
  ]);
  const clearBehaviorTemplateIdentity = React.useCallback(() => {
    updateChatModelSetting("systemPromptTemplateId", undefined);
  }, [updateChatModelSetting]);
  const setSelectedSystemPromptForComposer = React.useCallback(
    (id: string | undefined) => {
      clearBehaviorTemplateIdentity();
      setSelectedSystemPrompt(id);
    },
    [clearBehaviorTemplateIdentity, setSelectedSystemPrompt],
  );
  const setSelectedQuickPromptForComposer = React.useCallback(
    (prompt: string | undefined) => {
      clearBehaviorTemplateIdentity();
      setSelectedQuickPrompt(prompt);
    },
    [clearBehaviorTemplateIdentity, setSelectedQuickPrompt],
  );
  const clearRolePlayIdentity = React.useCallback(() => {
    void setSelectedCharacter(null);
    void setSelectedAssistant(null);
  }, [setSelectedAssistant, setSelectedCharacter]);
  const resetRolePlayGenerationStyle = React.useCallback(() => {
    const preset = getPresetByKey("balanced");
    if (!preset) return;
    updateChatModelSettings(preset.settings);
  }, [updateChatModelSettings]);
  const handleApplyRolePlaySetup = React.useCallback(
    async (payload: RolePlaySetupApplyPayload) => {
      if (payload.clearIdentity) {
        clearRolePlayIdentity();
      }
      if (payload.clearBehavior) {
        clearPromptContext();
      } else if (payload.behaviorTemplate) {
        handleTemplateSelect(payload.behaviorTemplate);
      }
      if (payload.resetGenerationStyle) {
        resetRolePlayGenerationStyle();
      } else if (payload.generationPresetKey) {
        const preset = getPresetByKey(payload.generationPresetKey);
        if (preset && preset.key !== "custom") {
          updateChatModelSettings(preset.settings);
        }
      }
    },
    [
      clearPromptContext,
      clearRolePlayIdentity,
      handleTemplateSelect,
      resetRolePlayGenerationStyle,
      updateChatModelSettings,
    ],
  );
  const handleApplyStartupTemplatePreview = React.useCallback(async () => {
    if (!startupTemplatePreview?.rolePlay) {
      handleApplyStartupTemplate();
      return;
    }

    if (startupTemplatePreview.rolePlay.source === "role-play-setup") {
      const nextScene =
        startupTemplatePreview.rolePlay.scene ?? createDefaultActorSettings();
      await saveActorSettingsForChat({
        historyId,
        serverChatId,
        settings: nextScene,
      });
      setActorSettings(nextScene);
      const preview = summarizeRolePlayScene(nextScene);
      setActorPreviewAndTokens(preview.prompt, preview.tokenCount);
    }

    await handleApplySavedRolePlaySetup(startupTemplatePreview);
  }, [
    handleApplySavedRolePlaySetup,
    handleApplyStartupTemplate,
    historyId,
    serverChatId,
    setActorPreviewAndTokens,
    setActorSettings,
    startupTemplatePreview,
  ]);
  const clearPinnedSourceContext = React.useCallback(() => {
    setRagPinnedResults([]);
  }, [setRagPinnedResults]);
  const clearHistoryContext = React.useCallback(() => {
    clearChat();
  }, [clearChat]);
  const trimLargestContextContributor = React.useCallback(() => {
    if (!largestContextContributor) return;
    if (largestContextContributor.id === "character") {
      setOpenActorSettings(true);
      return;
    }
    if (largestContextContributor.id === "prompt") {
      clearPromptContext();
      return;
    }
    if (largestContextContributor.id === "pinned") {
      clearPinnedSourceContext();
      return;
    }
    if (largestContextContributor.id === "history") {
      clearHistoryContext();
    }
  }, [
    clearHistoryContext,
    clearPinnedSourceContext,
    clearPromptContext,
    largestContextContributor,
  ]);
  const insertSummaryCheckpointPrompt = React.useCallback(() => {
    const checkpointPrompt = buildConversationSummaryCheckpointPrompt(
      messages,
      {
        maxRecentMessages: 10,
      },
    );
    setMessageValue(checkpointPrompt, {
      collapseLarge: true,
      forceCollapse: true,
    });
    textAreaFocus();
  }, [messages, setMessageValue, textAreaFocus]);

  const {
    imageBackendDefault: imageBackendDefaultTrimmed,
    setImageBackendDefault,
    imageBackendOptions,
    imageBackendLabel,
    imageBackendActiveKey,
    imageBackendMenuItems,
    imageBackendBadgeLabel,
  } = useImageBackend({ imageModels });
  const translateModelCatalogControl = React.useCallback(
    (key: string, defaultValueOrOptions?: unknown, options?: unknown) =>
      toText(t(key, defaultValueOrOptions as never, options as never)),
    [t],
  );
  const modelCatalogControls = (
    <PlaygroundModelCatalogControls
      t={translateModelCatalogControl}
      modelListScope={modelListScope}
      setModelListScope={setModelListScope}
      modelSearchQuery={modelSearchQuery}
      setModelSearchQuery={setModelSearchQuery}
      modelSortMode={modelSortMode}
      setModelSortMode={setModelSortMode}
    />
  );

  const modelSelectButton = (
    <ChatModelSelectorDropdown
      apiModelLabel={apiModelLabel}
      catalogControls={modelCatalogControls}
      connectionStatusLabel={connectionStatusLabel}
      connectionStatusWarning={
        !isConnectionReady || connectionUxState === "connected_degraded"
      }
      modelDropdownMenuItems={modelDropdownMenuItems}
      modelDropdownOpen={modelDropdownOpen}
      modelSearchQuery={modelSearchQuery}
      modelSelectorWarning={modelSelectorWarning}
      modelSortMode={modelSortMode}
      resolvedProviderKey={resolvedProviderKey}
      selectedModel={selectedModel}
      setModelDropdownOpen={setModelDropdownOpenWithFocusRestore}
      setModelSearchQuery={setModelSearchQuery}
      setModelSortMode={setModelSortMode}
    />
  );

  const modelUsageBadge = wrapComposerProfile(
    "token-progress",
    <TokenProgressBar
      conversationTokens={conversationTokenCount}
      draftTokens={draftTokenCount}
      maxTokens={resolvedMaxContext}
      modelLabel={isProMode ? apiModelLabel : undefined}
      compact={!isProMode}
      onClick={openContextWindowModal}
    />,
  );
  const compareControl = (
    <CompareToggle
      featureEnabled={compareFeatureEnabled}
      active={compareModeActive}
      onToggle={toggleCompareMode}
      selectedModels={compareSelectedModels}
      availableModels={availableCompareModels}
      maxModels={compareMaxModels}
      onAddModel={handleAddCompareModel}
      onRemoveModel={handleRemoveCompareModel}
      onOpenSettings={() => setOpenModelSettings(true)}
    />
  );
  const imageProviderControl = (
    <Dropdown
      menu={{
        items: imageBackendMenuItems,
        activeKey: imageBackendActiveKey,
      }}
      trigger={["hover", "click"]}
      placement="topRight"
    >
      <button
        type="button"
        title={t(
          "playground:imageBackend.tooltip",
          "Default image provider for /generate-image.",
        )}
        aria-label={imageBackendBadgeLabel}
        className="flex w-full items-center justify-between rounded-md px-2 py-1 text-sm text-text transition hover:bg-surface2"
      >
        <span className="flex min-w-0 items-center gap-2">
          <ImageIcon className="h-4 w-4 text-text-subtle" />
          <span className="truncate">
            {t("playground:imageBackend.menuLabel", "Default image provider")}
          </span>
        </span>
        <span className="flex min-w-0 items-center gap-1 text-xs text-text-muted">
          <span className="truncate max-w-[140px]">
            {imageBackendDefaultTrimmed
              ? imageBackendLabel
              : t("playground:imageBackend.none", "None")}
          </span>
          <ChevronRight className="h-3.5 w-3.5" />
        </span>
      </button>
    </Dropdown>
  );

  // Allow other components (e.g., connection card) to request focus
  React.useEffect(() => {
    const handler = () => {
      if (document.visibilityState === "visible") {
        textAreaFocus();
      }
    };
    window.addEventListener("tldw:focus-composer", handler);
    return () => window.removeEventListener("tldw:focus-composer", handler);
  }, [textAreaFocus]);

  // Allow other components (e.g., empty state) to set the composer message
  React.useEffect(() => {
    const handler = (event: CustomEvent<{ message: string }>) => {
      if (event.detail?.message) {
        form.setFieldValue("message", event.detail.message);
      }
    };
    window.addEventListener(
      "tldw:set-composer-message",
      handler as EventListener,
    );
    return () =>
      window.removeEventListener(
        "tldw:set-composer-message",
        handler as EventListener,
      );
  }, [form]);

  React.useEffect(() => {
    const handleToggleCompareMode = () => {
      toggleCompareMode();
    };
    const handleToggleModeLauncher = () => {
      setModeLauncherOpen((prev) => !prev);
    };

    window.addEventListener(
      "tldw:toggle-compare-mode",
      handleToggleCompareMode,
    );
    window.addEventListener(
      "tldw:toggle-mode-launcher",
      handleToggleModeLauncher,
    );
    return () => {
      window.removeEventListener(
        "tldw:toggle-compare-mode",
        handleToggleCompareMode,
      );
      window.removeEventListener(
        "tldw:toggle-mode-launcher",
        handleToggleModeLauncher,
      );
    };
  }, [toggleCompareMode]);

  const applyDiscussMediaPayload = React.useCallback(
    (
      rawPayload: unknown,
      options?: {
        clearAfterUse?: boolean;
      },
    ) => {
      const payload = normalizeMediaChatHandoffPayload(rawPayload);
      if (!payload) {
        if (options?.clearAfterUse) {
          void clearSetting(DISCUSS_MEDIA_PROMPT_SETTING);
        }
        return;
      }
      if (options?.clearAfterUse) {
        void clearSetting(DISCUSS_MEDIA_PROMPT_SETTING);
      }
      const mode = getMediaChatHandoffMode(payload);
      if (mode === "rag_media") {
        const mediaId = parseMediaIdAsNumber(payload);
        if (mediaId != null) {
          setChatMode("rag");
          setRagMediaIds([mediaId]);
        }
      } else {
        setChatMode("normal");
        setRagMediaIds(null);
      }
      const hint = buildDiscussMediaHint(payload);
      if (!hint) return;
      setMessageValue(hint, { collapseLarge: true, forceCollapse: true });
      textAreaFocus();
    },
    [setChatMode, setMessageValue, setRagMediaIds, textAreaFocus],
  );

  // Seed composer when a media item requests discussion (e.g., from Quick ingest or Review page)
  React.useEffect(() => {
    let cancelled = false;
    void (async () => {
      const payload = await getSetting(DISCUSS_MEDIA_PROMPT_SETTING);
      if (cancelled || !payload) return;
      applyDiscussMediaPayload(payload, { clearAfterUse: true });
    })();
    return () => {
      cancelled = true;
    };
  }, [applyDiscussMediaPayload]);

  React.useEffect(() => {
    const handler = (event: Event) => {
      const detail = (event as CustomEvent).detail;
      applyDiscussMediaPayload(detail);
    };
    window.addEventListener("tldw:discuss-media", handler as any);
    return () => {
      window.removeEventListener("tldw:discuss-media", handler as any);
    };
  }, [applyDiscussMediaPayload]);

  const applyDiscussWatchlistPayload = React.useCallback(
    (rawPayload: unknown, options?: { clearAfterUse?: boolean }) => {
      const payload = normalizeWatchlistChatHandoffPayload(rawPayload);
      if (!payload) {
        if (options?.clearAfterUse) {
          void clearSetting(DISCUSS_WATCHLIST_PROMPT_SETTING);
        }
        return;
      }
      if (options?.clearAfterUse) {
        void clearSetting(DISCUSS_WATCHLIST_PROMPT_SETTING);
      }
      setChatMode("normal");
      setRagMediaIds(null);
      const hint = buildWatchlistChatHint(payload);
      if (!hint) return;
      setMessageValue(hint, { collapseLarge: true, forceCollapse: true });
      textAreaFocus();
    },
    [setChatMode, setMessageValue, setRagMediaIds, textAreaFocus],
  );

  // Seed composer when a watchlist item requests discussion
  React.useEffect(() => {
    let cancelled = false;
    void (async () => {
      const payload = await getSetting(DISCUSS_WATCHLIST_PROMPT_SETTING);
      if (cancelled || !payload) return;
      applyDiscussWatchlistPayload(payload, { clearAfterUse: true });
    })();
    return () => {
      cancelled = true;
    };
  }, [applyDiscussWatchlistPayload]);

  React.useEffect(() => {
    const handler = (event: Event) => {
      const detail = (event as CustomEvent).detail;
      applyDiscussWatchlistPayload(detail);
    };
    window.addEventListener("tldw:discuss-watchlist", handler as any);
    return () => {
      window.removeEventListener("tldw:discuss-watchlist", handler as any);
    };
  }, [applyDiscussWatchlistPayload]);

  React.useEffect(() => {
    textAreaFocus();
  }, [textAreaFocus]);

  React.useEffect(() => {
    // Apply global default when the preference changes, but do not override per-chat toggles.
    if (defaultInternetSearchOn) {
      setWebSearch(true);
    }
  }, [defaultInternetSearchOn, setWebSearch]);

  React.useEffect(() => {
    if (isConnectionReady) {
      setShowConnectBanner(false);
    }
  }, [isConnectionReady]);

  const notifyImageAttachmentDisabled = React.useCallback(() => {
    notificationApi.warning({
      message: t(
        "playground:attachments.imageDisabledTitle",
        "Image attachments disabled",
      ),
      description: t(
        "playground:attachments.imageDisabledBody",
        "Disable Knowledge Search to attach images.",
      ),
    });
  }, [notificationApi, t]);

  const attachments = usePlaygroundAttachments({
    chatMode,
    setFieldValue: form.setFieldValue,
    handleFileUpload,
    notifyImageAttachmentDisabled,
  });
  const {
    inputRef,
    fileInputRef,
    onFileInputChange,
    onInputChange,
    handleImageUpload,
    handleDocumentUpload,
    useDroppedFiles,
  } = attachments;

  // Process dropped files
  useDroppedFiles(droppedFiles);

  const handlePaste = React.useCallback(
    async (e: React.ClipboardEvent) => {
      if (e.clipboardData.files.length > 0) {
        try {
          await onInputChange(e.clipboardData.files[0]);
        } catch (error) {
          console.error("Failed to handle pasted file:", error);
        }
        return;
      }

      const pastedText = e.clipboardData.getData("text/plain");
      if (!pastedText) return;

      if (pasteLargeTextAsFile && pastedText.length > PASTED_TEXT_CHAR_LIMIT) {
        e.preventDefault();
        const blob = new Blob([pastedText], { type: "text/plain" });
        const file = new File([blob], `pasted-text-${Date.now()}.txt`, {
          type: "text/plain",
        });

        await handleFileUpload(file);
        return;
      }

      if (isMessageCollapsed && collapsedRange) {
        e.preventDefault();
        const currentValue = form.values.message || "";
        const meta = getCollapsedDisplayMeta(currentValue, collapsedRange);
        const textarea = textareaRef.current;
        const rawStart = textarea?.selectionStart ?? meta.labelEnd;
        const rawEnd = textarea?.selectionEnd ?? rawStart;
        const displayStart = Math.min(rawStart, rawEnd);
        const displayEnd = Math.max(rawStart, rawEnd);
        const hasSelection = displayStart !== displayEnd;
        const selectionTouchesLabel =
          displayStart < meta.labelEnd && displayEnd > meta.labelStart;
        if (hasSelection) {
          const startPrefer =
            displayStart > meta.labelStart && displayStart < meta.labelEnd
              ? "before"
              : undefined;
          const endPrefer =
            displayEnd > meta.labelStart && displayEnd < meta.labelEnd
              ? "after"
              : undefined;
          let editStart = getMessageCaretFromDisplay(displayStart, meta, {
            prefer: startPrefer,
          });
          let editEnd = getMessageCaretFromDisplay(displayEnd, meta, {
            prefer: endPrefer,
          });
          if (editStart > editEnd) {
            [editStart, editEnd] = [editEnd, editStart];
          }
          if (selectionTouchesLabel) {
            editStart = Math.min(editStart, meta.rangeStart);
            editEnd = Math.max(editEnd, meta.rangeEnd);
          }
          replaceCollapsedRange(
            currentValue,
            meta,
            editStart,
            editEnd,
            pastedText,
          );
          return;
        }
        const caretPrefer =
          rawStart > meta.labelStart && rawStart < meta.labelEnd
            ? pendingCaretRef.current !== null &&
              pendingCaretRef.current <= meta.rangeStart
              ? "before"
              : "after"
            : undefined;
        let caret = getMessageCaretFromDisplay(rawStart, meta, {
          prefer: caretPrefer,
        });
        if (caret > meta.rangeStart && caret < meta.rangeEnd) {
          caret = meta.rangeEnd;
        }
        const insertAt =
          caret <= meta.rangeStart
            ? caret
            : caret >= meta.rangeEnd
              ? caret
              : meta.rangeEnd;
        replaceCollapsedRange(
          currentValue,
          meta,
          insertAt,
          insertAt,
          pastedText,
        );
        return;
      }

      const currentValue = form.values.message || "";
      const textarea = textareaRef.current;
      const selectionStart = textarea?.selectionStart ?? currentValue.length;
      const selectionEnd = textarea?.selectionEnd ?? selectionStart;
      const nextValue =
        currentValue.slice(0, selectionStart) +
        pastedText +
        currentValue.slice(selectionEnd);

      if (nextValue.length > PASTED_TEXT_CHAR_LIMIT) {
        e.preventDefault();
        const blockRange = {
          start: selectionStart,
          end: selectionStart + pastedText.length,
        };
        pendingCaretRef.current = blockRange.end;
        pendingCollapsedStateRef.current = {
          message: nextValue,
          range: blockRange,
          caret: blockRange.end,
        };
        setMessageValue(nextValue, {
          collapseLarge: true,
          forceCollapse: true,
          collapsedRange: blockRange,
        });
      }
    },
    [
      collapsedRange,
      form.values.message,
      handleFileUpload,
      isMessageCollapsed,
      getCollapsedDisplayMeta,
      getMessageCaretFromDisplay,
      onInputChange,
      pasteLargeTextAsFile,
      replaceCollapsedRange,
      setMessageValue,
      textareaRef,
    ],
  );
  const handleDisconnectedFocus = React.useCallback(() => {
    if (!isConnectionReady && !hasShownConnectBanner) {
      setShowConnectBanner(true);
      setHasShownConnectBanner(true);
    }
  }, [hasShownConnectBanner, isConnectionReady]);

  const handleTextareaKeyDown = (
    e: React.KeyboardEvent<HTMLTextAreaElement>,
  ) => {
    if (handleCollapsedKeyDown(e)) return;
    handleKeyDown(e);
  };

  const handleTextareaFocus = React.useCallback(() => {
    handleDisconnectedFocus();
    if (!isMessageCollapsed) return;
    const wasPointer = pointerDownRef.current;
    pointerDownRef.current = false;
    if (wasPointer) return;
    const textarea = textareaRef.current;
    if (pendingCaretRef.current === null && textarea) {
      lastDisplaySelectionRef.current = {
        start: textarea.selectionStart ?? 0,
        end: textarea.selectionEnd ?? textarea.selectionStart ?? 0,
      };
    }
    syncCollapsedCaret();
  }, [
    handleDisconnectedFocus,
    isMessageCollapsed,
    syncCollapsedCaret,
    textareaRef,
  ]);

  const handleMentionSelect = React.useCallback(
    (tab: any) =>
      insertMention(tab, form.values.message, (value: string) =>
        form.setFieldValue("message", value),
      ),
    [insertMention, form],
  );

  const handleMentionRefetch = React.useCallback(async () => {
    await reloadTabs();
  }, [reloadTabs]);

  const handleKnowledgeInsert = React.useCallback(
    (text: string) => {
      const current = textareaRef.current?.value || "";
      const next = current ? `${current}\n\n${text}` : text;
      setMessageValue(next, { collapseLarge: true });
      textAreaFocus();
    },
    [setMessageValue, textAreaFocus, textareaRef],
  );
  const handleKnowledgePanelOpenChange = React.useCallback(
    (nextOpen: boolean) => {
      if (nextOpen) {
        closeComposerPopoversExcept("context");
      }
      setContextToolsOpen(nextOpen);
    },
    [closeComposerPopoversExcept, setContextToolsOpen],
  );
  const handleKnowledgeRemoveImage = React.useCallback(() => {
    form.setFieldValue("image", "");
  }, [form.setFieldValue]);
  const handleKnowledgeAddFile = React.useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const voiceChatHook = usePlaygroundVoiceChat({
    voiceConversationAvailability,
    voiceChatEnabled,
    setVoiceChatEnabled,
    voiceChat,
    voiceChatMessages,
    canUseServerStt,
    sttModel,
    sttTemperature,
    sttTask,
    sttResponseFormat,
    sttTimestampGranularities,
    sttPrompt,
    sttUseSegmentation,
    sttSegK,
    sttSegMinSegmentSize,
    sttSegLambdaBalance,
    sttSegUtteranceExpansionWidth,
    sttSegEmbeddingsProvider,
    sttSegEmbeddingsModel,
    dictationModeOverride,
    dictationAutoFallbackEnabled,
    autoStopTimeout,
    autoSubmitVoiceMessage,
    speechToTextLanguage,
    setMessageValue,
    submitForm: () => voiceChatSubmitFormRef.current(),
    notificationApi,
    isSending,
    isListening: false,
    isServerDictating: false,
    t,
  });
  const {
    isListening,
    browserSupportsSpeechRecognition,
    dictationAudioSourcePreference,
    dictationResolvedSourceKind,
    setDictationAudioSourcePreference,
    isServerDictating,
    speechAvailable,
    speechUsesServer,
    voiceChatStatusLabel,
    speechTooltipText,
    handleVoiceChatToggle,
    handleDictationToggle,
    stopListening,
  } = voiceChatHook;
  const { sendWhenEnter, setSendWhenEnter } = useWebUI();

  React.useEffect(() => {
    if (!selectedQuickPrompt) {
      return;
    }

    const currentMessage = form.values.message || "";
    const promptText = selectedQuickPrompt;

    const applyOverwrite = () => {
      const word = getVariable(promptText);
      setMessageValue(promptText, { collapseLarge: true });
      if (word) {
        textareaRef.current?.focus();
        const interval = setTimeout(() => {
          textareaRef.current?.setSelectionRange(word.start, word.end);
          setSelectedQuickPrompt(null);
        }, 100);
        return () => {
          clearInterval(interval);
        };
      }
      setSelectedQuickPrompt(null);
      return;
    };

    const applyAppend = () => {
      const next =
        currentMessage.trim().length > 0
          ? `${currentMessage}\n\n${promptText}`
          : promptText;
      setMessageValue(next, { collapseLarge: true });
      setSelectedQuickPrompt(null);
    };

    if (!currentMessage.trim()) {
      applyOverwrite();
      return;
    }

    Modal.confirm({
      title: t("option:promptInsert.confirmTitle", {
        defaultValue: "Use prompt in chat?",
      }),
      content: t("option:promptInsert.confirmDescription", {
        defaultValue:
          "Your message already has text. Do you want to overwrite it with this prompt or append the prompt below it?",
      }),
      okText: t("option:promptInsert.overwrite", {
        defaultValue: "Overwrite message",
      }),
      cancelText: t("option:promptInsert.append", {
        defaultValue: "Append",
      }),
      closable: false,
      maskClosable: false,
      onOk: () => {
        applyOverwrite();
      },
      onCancel: () => {
        applyAppend();
      },
    });
  }, [
    selectedQuickPrompt,
    form.values.message,
    setMessageValue,
    setSelectedQuickPrompt,
    t,
    textareaRef,
  ]);

  const queryClient = useQueryClient();
  const invalidateServerChatHistory = React.useCallback(() => {
    queryClient.invalidateQueries({ queryKey: ["serverChatHistory"] });
  }, [queryClient]);
  const {
    visibleError: chatErrorBanner,
    dismissError: dismissChatErrorBanner,
    dismissAfterSuccessfulSubmit: dismissChatErrorAfterSuccessfulSubmit,
  } = usePlaygroundChatErrorBanner(messages);

  const { mutateAsync: sendMessage } = useMutation({
    mutationFn: onSubmit,
    onMutate: () => ({
      errorKeyToDismiss: chatErrorBanner?.key ?? null,
    }),
    onSuccess: (_data, _variables, context) => {
      dismissChatErrorAfterSuccessfulSubmit(context?.errorKeyToDismiss ?? null);
      void trackOnboardingChatSubmitSuccess(
        typeof window !== "undefined" ? window.location.pathname : "/chat",
      );
      textAreaFocus();
      queryClient.invalidateQueries({
        queryKey: ["fetchChatHistory"],
      });
    },
    onError: (error) => {
      textAreaFocus();
    },
  });

  const followUpResearchDraftQuery = React.useMemo(() => {
    const trimmed = form.values.message.trim();
    if (!trimmed.startsWith(FOLLOW_UP_RESEARCH_PROMPT_PREFIX)) {
      return trimmed;
    }
    const unwrapped = trimmed
      .slice(FOLLOW_UP_RESEARCH_PROMPT_PREFIX.length)
      .trim();
    return unwrapped || trimmed;
  }, [form.values.message]);
  const canLaunchFollowUpResearch =
    !temporaryChat &&
    Boolean(serverChatId) &&
    followUpResearchDraftQuery.length > 0;
  const showFollowUpResearchButton =
    Boolean(attachedResearchContext) && canLaunchFollowUpResearch;

  const openFollowUpResearchModal = React.useCallback(() => {
    if (!canLaunchFollowUpResearch) return;
    setIncludeAttachedResearchAsBackground(Boolean(attachedResearchContext));
    setFollowUpResearchModalOpen(true);
  }, [attachedResearchContext, canLaunchFollowUpResearch]);

  const closeFollowUpResearchModal = React.useCallback(() => {
    if (followUpResearchPendingRef.current) return;
    setFollowUpResearchModalOpen(false);
  }, []);

  const handleStartFollowUpResearch = React.useCallback(async () => {
    if (followUpResearchPendingRef.current) return;
    if (!serverChatId || temporaryChat) return;
    if (!followUpResearchDraftQuery) return;

    const payload: ResearchRunCreateRequest = {
      query: followUpResearchDraftQuery,
      source_policy: "balanced",
      autonomy_mode: "checkpointed",
      chat_handoff: {
        chat_id: serverChatId,
      },
      follow_up: {
        question: followUpResearchDraftQuery,
        background:
          includeAttachedResearchAsBackground && attachedResearchContext
            ? buildFollowUpResearchBackground(attachedResearchContext)
            : undefined,
      },
    };

    followUpResearchPendingRef.current = true;
    setFollowUpResearchPending(true);
    try {
      await tldwClient.createResearchRun(payload);
      void queryClient.invalidateQueries({
        queryKey: ["playground:chat-linked-research-runs", serverChatId],
      });
      setFollowUpResearchModalOpen(false);
      notificationApi.success?.({
        message: t(
          "playground:actions.followUpResearchStarted",
          "Follow-up research started.",
        ),
      });
    } catch (error) {
      notificationApi.error({
        message: t(
          "playground:actions.followUpResearchFailed",
          "Unable to start follow-up research.",
        ),
        description: error instanceof Error ? error.message : undefined,
      });
    } finally {
      followUpResearchPendingRef.current = false;
      setFollowUpResearchPending(false);
    }
  }, [
    attachedResearchContext,
    followUpResearchDraftQuery,
    includeAttachedResearchAsBackground,
    notificationApi,
    queryClient,
    serverChatId,
    t,
    temporaryChat,
  ]);

  const queueMgmt = usePlaygroundQueueManagement({
    composerModels,
    isConnectionReady,
    isSending,
    selectedModel,
    chatMode,
    webSearch,
    compareMode,
    compareModeActive,
    compareSelectedModels,
    selectedSystemPrompt,
    selectedQuickPrompt,
    toolChoice,
    useOCR,
    selectedDocuments,
    uploadedFiles,
    contextFiles,
    documentContext,
    queuedMessages,
    setQueuedMessages,
    historyId,
    serverChatId,
    conversationTokenCount,
    resolvedMaxContext,
    estimateTokensForText: estimateTokensForText as any,
    characterContextTokenEstimate,
    pinnedSourceTokenEstimate,
    currentContextSnapshot,
    setLastSubmittedContext,
    setSelectedModel,
    setChatMode,
    setWebSearch,
    setCompareMode,
    setCompareSelectedModels,
    setSelectedSystemPrompt,
    setSelectedQuickPrompt,
    setToolChoice,
    setUseOCR,
    compareModelsSupportCapability,
    sendMessage,
    stopStreamingRequest,
    form,
    clearSelectedDocuments,
    clearUploadedFiles,
    textAreaFocus,
    notificationApi,
    t,
  });
  const {
    availableChatModelIds,
    isQueuedDispatchBlockedByComposerState,
    queuedRequestActions,
    queueSubmission,
    cancelCurrentAndRunDisabledReason,
    handleRunQueuedRequest,
    handleRunNextQueuedRequest,
    validateSelectedChatModelsAvailability,
  } = queueMgmt;

  const handleToggleWebSearch = React.useCallback(() => {
    setWebSearch(!webSearch);
  }, [setWebSearch, webSearch]);
  const handleOpenModelSettings = React.useCallback(() => {
    modelSettingsReturnFocusSelectorRef.current = null;
    closeComposerPopoversExcept("model");
    setOpenModelSettings(true);
  }, [closeComposerPopoversExcept, setOpenModelSettings]);
  const {
    showSlashMenu,
    slashActiveIndex,
    setSlashActiveIndex,
    filteredSlashCommands,
    resolveSubmissionIntent,
    activeImageCommand,
    handleSlashCommandSelect: slashHandleSelect,
  } = useSlashCommands({
    chatMode,
    setChatMode,
    webSearch,
    setWebSearch,
    handleImageUpload,
    imageBackendDefaultTrimmed,
    imageBackendLabel,
    setOpenModelSettings,
    currentMessage: form.values.message,
  });

  const handleSlashCommandSelect = React.useCallback(
    (command: SlashCommandItem) => {
      slashHandleSelect(command, form.setFieldValue.bind(form), textareaRef);
    },
    [slashHandleSelect, form, textareaRef],
  );

  const { submitForm, submitFormRef } = usePlaygroundSubmit({
    form,
    isSending,
    isConnectionReady,
    webSearch,
    compareModeActive,
    compareSelectedModels,
    selectedModel,
    fileRetrievalEnabled,
    ragPinnedResults,
    selectedDocuments,
    uploadedFiles,
    currentContextSnapshot,
    conversationTokenCount,
    characterContextTokenEstimate,
    pinnedSourceTokenEstimate,
    resolvedMaxContext,
    jsonMode: Boolean(currentChatModelSettings.jsonMode),
    researchContext,
    importedSidepanelContext,
    clearImportedSidepanelContext,
    sendMessage,
    clearSelectedDocuments,
    clearUploadedFiles,
    textAreaFocus,
    setLastSubmittedContext,
    estimateTokensForText: estimateTokensForText as any,
    resolveSubmissionIntent,
    queueSubmission,
    validateSelectedChatModelsAvailability,
    compareModelsSupportCapability,
    notificationApi,
    t,
  });
  const runCharacterChatSendBlocker = React.useCallback(() => {
    if (characterChatSendBlocker?.active) {
      stopListening();
      characterChatSendBlocker.onAction();
    }
  }, [characterChatSendBlocker, stopListening]);
  const handleComposerSend = React.useCallback(() => {
    if (characterChatSendBlocker?.active) {
      runCharacterChatSendBlocker();
      return;
    }
    submitForm();
  }, [characterChatSendBlocker, runCharacterChatSendBlocker, submitForm]);
  React.useEffect(() => {
    voiceChatSubmitFormRef.current = () => {
      handleComposerSend();
    };
  }, [handleComposerSend]);

  const handleKnowledgeAsk = React.useCallback(
    (text: string, options?: { ignorePinnedResults?: boolean }) => {
      const trimmed = text.trim();
      if (!trimmed) return;
      setMessageValue(trimmed, { collapseLarge: true });
      queueMicrotask(() => {
        if (characterChatSendBlocker?.active) {
          runCharacterChatSendBlocker();
          return;
        }
        submitFormRef.current({
          ignorePinnedResults: options?.ignorePinnedResults,
        });
      });
    },
    [
      characterChatSendBlocker,
      runCharacterChatSendBlocker,
      setMessageValue,
      submitFormRef,
    ],
  );

  const persistence = usePlaygroundPersistence({
    isFireFoxPrivateMode,
    isConnectionReady,
    temporaryChat,
    setTemporaryChat,
    serverChatId,
    setServerChatId,
    historyId,
    serverChatState,
    setServerChatState,
    serverChatSource,
    setServerChatSource,
    setServerChatVersion,
    setServerChatCharacterId,
    setServerChatAssistantKind,
    setServerChatAssistantId,
    setServerChatPersonaMemoryMode,
    history,
    clearChat,
    selectedCharacter,
    selectedAssistantMode,
    assistantOverlayActive: pendingAssistantState.mode === "overlay",
    serverPersistenceHintSeen,
    setServerPersistenceHintSeen,
    invalidateServerChatHistory,
    navigate,
    notificationApi,
    t,
  });
  const {
    persistenceTooltip,
    focusConnectionCard,
    getPersistenceModeLabel,
    privateChatLocked,
    showServerPersistenceHint,
    handleToggleTemporaryChat,
    handleSaveChatToServer,
    persistChatMetadata,
    handleDismissServerPersistenceHint,
  } = persistence;

  React.useEffect(() => {
    if (typeof window === "undefined") return;
    const handler = () => handleToggleWebSearch();
    window.addEventListener(TOGGLE_WEB_SEARCH_EVENT, handler);
    return () => {
      window.removeEventListener(TOGGLE_WEB_SEARCH_EVENT, handler);
    };
  }, [handleToggleWebSearch]);

  React.useEffect(() => {
    if (typeof window === "undefined") return;
    const handler = (event: Event) => {
      const next = (event as CustomEvent<{ next?: unknown }>).detail?.next;
      if (typeof next !== "boolean") return;
      handleToggleTemporaryChat(next);
    };
    window.addEventListener(SET_TEMPORARY_CHAT_EVENT, handler);
    return () => {
      window.removeEventListener(SET_TEMPORARY_CHAT_EVENT, handler);
    };
  }, [handleToggleTemporaryChat]);

  const handleClearContext = React.useCallback(() => {
    // Only show confirmation if there's history to clear
    if (history.length === 0) {
      return;
    }

    Modal.confirm({
      title: t(
        "playground:composer.clearContextConfirmTitle",
        "Clear conversation?",
      ),
      content: t(
        "playground:composer.clearContextConfirmContent",
        "This will remove all messages from the current conversation. This action cannot be undone.",
      ),
      okText: t("common:confirm", "Confirm"),
      okButtonProps: { danger: true },
      cancelText: t("common:cancel", "Cancel"),
      onOk: () => {
        setHistory([]);
        notificationApi.success({
          message: t(
            "playground:composer.clearContextSuccess",
            "Conversation cleared",
          ),
          duration: 2,
        });
      },
    });
  }, [history.length, notificationApi, setHistory, t]);

  const requestKnowledgePanelTab = React.useCallback((tab: KnowledgeTab) => {
    setKnowledgePanelTab(tab);
    setKnowledgePanelTabRequestId((id) => id + 1);
  }, []);

  const openKnowledgePanel = React.useCallback(
    (tab: KnowledgeTab) => {
      requestKnowledgePanelTab(tab);
      closeComposerPopoversExcept("context");
      setContextToolsOpen(true);
    },
    [closeComposerPopoversExcept, requestKnowledgePanelTab, setContextToolsOpen],
  );

  const toggleKnowledgePanel = React.useCallback(
    (tab: KnowledgeTab = "search") => {
      const nextOpen = !contextToolsOpen;
      if (nextOpen) {
        requestKnowledgePanelTab(tab);
        closeComposerPopoversExcept("context");
      }
      setContextToolsOpen(nextOpen);
    },
    [
      closeComposerPopoversExcept,
      contextToolsOpen,
      requestKnowledgePanelTab,
      setContextToolsOpen,
    ],
  );

  React.useEffect(() => {
    if (typeof window === "undefined") return;
    const handler = (event: Event) => {
      const detail = (event as CustomEvent<{ tab?: KnowledgeTab }>).detail;
      const tab = detail?.tab === "context" ? "context" : "search";
      openKnowledgePanel(tab);
      setModeAnnouncement(
        t(
          "playground:starter.noticeKnowledge",
          "Opened Search & Context panel.",
        ),
      );
    };
    window.addEventListener(
      OPEN_KNOWLEDGE_PANEL_EVENT,
      handler as EventListener,
    );
    return () => {
      window.removeEventListener(
        OPEN_KNOWLEDGE_PANEL_EVENT,
        handler as EventListener,
      );
    };
  }, [openKnowledgePanel, t]);

  React.useEffect(() => {
    if (typeof window === "undefined") return;
    const handler = (event: Event) => {
      const detail = (event as CustomEvent<{ mode?: string; prompt?: string }>)
        .detail;
      const mode = String(detail?.mode || "")
        .trim()
        .toLowerCase();
      if (mode === "compare") {
        if (!compareFeatureEnabled) {
          notificationApi.warning({
            message: t(
              "playground:starter.compareUnavailable",
              "Compare mode unavailable",
            ),
          });
          return;
        }
        setCompareMode(true);
        if (selectedModel && compareSelectedModels.length === 0) {
          setCompareSelectedModels([selectedModel]);
        }
        setModeAnnouncement(
          t(
            "playground:starter.noticeCompare",
            "Compare mode enabled. Select models and send your first prompt.",
          ),
        );
        textAreaFocus();
        return;
      }
      if (mode === "character") {
        const hasCharacterIdentity =
          selectedAssistant?.kind === "character" || Boolean(selectedCharacter?.id);
        if (hasCharacterIdentity) {
          setModeAnnouncement(
            t(
              "playground:starter.noticeCharacterActive",
              "Character Chat mode active. Continue with the selected character.",
            ),
          );
          textAreaFocus();
          return;
        }
        dispatchOpenAssistantSelect({
          tab: "character",
          source: "playground-starter",
        });
        setModeAnnouncement(
          t(
            "playground:starter.noticeCharacter",
            "Character mode starter selected. Choose a character before sending.",
          ),
        );
        return;
      }
      if (mode === "rag" || mode === "knowledge") {
        setChatMode("rag");
        openKnowledgePanel("search");
        setModeAnnouncement(
          t(
            "playground:starter.noticeRag",
            "Knowledge starter selected. Search and pin sources before sending.",
          ),
        );
        if (detail?.prompt) {
          form.setFieldValue("message", String(detail.prompt));
        }
        textAreaFocus();
        return;
      }
      if (detail?.prompt) {
        form.setFieldValue("message", String(detail.prompt));
      }
      setModeAnnouncement(
        t("playground:starter.noticeGeneral", "General chat starter selected."),
      );
      textAreaFocus();
    };
    window.addEventListener(
      "tldw:playground-starter",
      handler as EventListener,
    );
    return () => {
      window.removeEventListener(
        "tldw:playground-starter",
        handler as EventListener,
      );
    };
  }, [
    compareFeatureEnabled,
    compareSelectedModels.length,
    form,
    notificationApi,
    openKnowledgePanel,
    selectedModel,
    selectedAssistant,
    selectedCharacter?.id,
    setChatMode,
    setCompareMode,
    setCompareSelectedModels,
    t,
    textAreaFocus,
  ]);

  const voiceChatSettingsFields = (
    <>
      <div className="flex flex-col gap-1">
        <span className="text-[11px] text-text-muted">
          {t("playground:voiceChat.modelLabel", "Voice model")}
        </span>
        <Select
          size="small"
          value={voiceChatModel || ""}
          options={voiceChatModelOptions}
          onChange={(value) => setVoiceChatModel(value)}
        />
      </div>
      <div className="flex flex-col gap-1">
        <span className="text-[11px] text-text-muted">
          {t("playground:voiceChat.pauseLabel", "Auto-send pause (ms)")}
        </span>
        <InputNumber
          size="small"
          min={200}
          max={5000}
          value={voiceChatPauseMs}
          onChange={(value) =>
            setVoiceChatPauseMs(typeof value === "number" ? value : 900)
          }
        />
      </div>
      <div className="flex flex-col gap-1">
        <span className="text-[11px] text-text-muted">
          {t("playground:voiceChat.triggerLabel", "Trigger phrases")}
        </span>
        <Input
          size="small"
          placeholder={t(
            "playground:voiceChat.triggerPlaceholder",
            "e.g. send it, over",
          )}
          value={voiceChatTriggerInput}
          onChange={(e) => {
            const value = e.target.value;
            setVoiceChatTriggerInput(value);
            const parsed = value
              .split(",")
              .map((entry) => entry.trim())
              .filter(Boolean);
            setVoiceChatTriggerPhrases(parsed);
          }}
        />
      </div>
      <div className="flex flex-col gap-1">
        <span className="text-[11px] text-text-muted">
          {t("playground:voiceChat.ttsModeLabel", "Response audio")}
        </span>
        <Radio.Group
          size="small"
          optionType="button"
          value={voiceChatTtsMode}
          onChange={(e) => setVoiceChatTtsMode(e.target.value)}
        >
          <Radio.Button value="stream">
            {t("playground:voiceChat.ttsModeStream", "Stream")}
          </Radio.Button>
          <Radio.Button value="full">
            {t("playground:voiceChat.ttsModeFull", "Full")}
          </Radio.Button>
        </Radio.Group>
      </div>
      <div className="flex flex-col gap-1">
        <span className="text-[11px] text-text-muted">
          {t("playground:voiceChat.sourceLabel", "Input source")}
        </span>
        <AudioSourcePicker
          ariaLabel={t(
            "playground:voiceChat.sourcePickerLabel",
            "Dictation input source",
          )}
          devices={audioInputDevices}
          requestedSourceKind={dictationAudioSourcePreference.sourceKind}
          resolvedSourceKind={dictationResolvedSourceKind}
          requestedDeviceId={dictationAudioSourcePreference.deviceId}
          lastKnownLabel={dictationAudioSourcePreference.lastKnownLabel}
          onChange={(nextValue) =>
            setDictationAudioSourcePreference({
              featureGroup: "dictation",
              sourceKind: nextValue.sourceKind,
              deviceId: nextValue.deviceId ?? null,
              lastKnownLabel: nextValue.lastKnownLabel ?? null,
            })
          }
        />
      </div>
      <div className="flex items-center justify-between gap-2">
        <span className="text-[11px] text-text-muted">
          {t(
            "playground:voiceChat.autoResume",
            "Continue listening after response",
          )}
        </span>
        <Switch
          size="small"
          checked={voiceChatAutoResume}
          onChange={(checked) => setVoiceChatAutoResume(checked)}
        />
      </div>
      <div className="flex items-center justify-between gap-2">
        <span className="text-[11px] text-text-muted">
          {t("playground:voiceChat.bargeIn", "Interrupt while speaking")}
        </span>
        <Switch
          size="small"
          checked={voiceChatBargeIn}
          onChange={(checked) => setVoiceChatBargeIn(checked)}
        />
      </div>
    </>
  );

  React.useEffect(() => {
    if (contextToolsOpen) {
      reloadTabs();
    }
  }, [contextToolsOpen, reloadTabs]);

  // State for collapsible advanced section in tools popover
  const [advancedToolsExpanded, setAdvancedToolsExpanded] =
    React.useState(isProMode);

  const imageGen = usePlaygroundImageGen({
    imageBackendDefaultTrimmed: imageBackendDefaultTrimmed,
    imageBackendOptions,
    imageEventSyncChatMode,
    imageEventSyncGlobalDefault,
    updateChatSettings,
    setImageEventSyncGlobalDefault: setImageEventSyncGlobalDefault as any,
    messages: messages as any,
    selectedCharacterName: selectedCharacter?.name ?? null,
    selectedModel,
    currentApiProvider: currentChatModelSettings.apiProvider,
    formMessage: form.values.message || "",
    sendMessage,
    textAreaFocus,
    notificationApi,
    t,
    setToolsPopoverOpen,
  });
  const {
    imageGenerateModalOpen,
    imageGenerateBackend,
    setImageGenerateBackend,
    imageGeneratePrompt,
    setImageGeneratePrompt,
    imageGeneratePromptMode,
    setImageGeneratePromptMode,
    imageGenerateFormat,
    setImageGenerateFormat,
    imageGenerateNegativePrompt,
    setImageGenerateNegativePrompt,
    imageGenerateWidth,
    setImageGenerateWidth,
    imageGenerateHeight,
    setImageGenerateHeight,
    imageGenerateSteps,
    setImageGenerateSteps,
    imageGenerateCfgScale,
    setImageGenerateCfgScale,
    imageGenerateSeed,
    setImageGenerateSeed,
    imageGenerateSampler,
    setImageGenerateSampler,
    imageGenerateModel,
    setImageGenerateModel,
    imageGenerateExtraParams,
    setImageGenerateExtraParams,
    imageGenerateReferenceFileId,
    setImageGenerateReferenceFileId,
    imageGenerateSyncPolicy,
    setImageGenerateSyncPolicy,
    referenceImageCandidates,
    referenceImageCandidatesLoading,
    imagePromptContextBreakdown,
    imagePromptRefineSubmitting,
    imagePromptRefineBaseline,
    imagePromptRefineCandidate,
    imagePromptRefineModel,
    imagePromptRefineLatencyMs,
    imagePromptRefineDiff,
    imageGenerateRefineMetadata,
    imageGenerateSubmitting,
    imagePromptStrategies,
    imageGenerationCharacterMood,
    imageGenerateBackendOptions,
    imageGenerateBusy,
    imageEventSyncBaselineMode,
    imageGenerateResolvedSyncMode,
    clearImagePromptRefineState,
    closeImageGenerateModal,
    hydrateImageGenerateSettings,
    openImageGenerateModal,
    handleCreateImagePromptDraft,
    handleRefineImagePromptDraft,
    applyRefinedImagePromptCandidate,
    rejectRefinedImagePromptCandidate,
    submitImageGenerateModal,
    normalizeImageGenerationEventSyncMode: normalizeImageGenSyncMode,
    normalizeImageGenerationEventSyncPolicy: normalizeImageGenSyncPolicy,
  } = imageGen;
  React.useEffect(() => {
    setOptionalPanelVisible("mcp-tools", mcpSettingsOpen);
    if (mcpSettingsOpen || toolChoice !== "none") {
      markOptionalPanelEngaged("mcp-tools");
    }

    return () => {
      setOptionalPanelVisible("mcp-tools", false);
    };
  }, [
    markOptionalPanelEngaged,
    mcpSettingsOpen,
    setOptionalPanelVisible,
    toolChoice,
  ]);

  // Image generation logic is now in usePlaygroundImageGen hook above.

  // handleCreateImagePromptDraft - extracted to usePlaygroundImageGen

  const rawPreview = usePlaygroundRawPreview({
    composerModels,
    selectedModel,
    compareModeActive,
    compareSelectedModels,
    compareMaxModels,
    currentChatModelSettings,
    history,
    systemPrompt,
    hasMcp,
    mcpHealthState,
    mcpTools: chatMcpTools,
    toolChoice,
    temporaryChat,
    serverChatId,
    serverChatState,
    serverChatSource,
    selectedCharacter,
    hasPersona: selectedAssistant?.kind === "persona",
    messageSteeringMode,
    messageSteeringForceNarrate,
    ragMediaIds,
    selectedKnowledge: selectedKnowledge as any,
    ragPinnedResults,
    fileRetrievalEnabled,
    contextFiles,
    documentContext,
    selectedDocuments,
    imageBackendDefaultTrimmed: imageBackendDefaultTrimmed,
    resolveSubmissionIntent,
    formImage: form.values.image || "",
    formMessage: form.values.message || "",
    importedSidepanelContext,
    researchContext: compareModeActive ? undefined : researchContext,
    notificationApi,
    t,
    setToolsPopoverOpen,
  });
  const {
    rawRequestModalOpen,
    setRawRequestModalOpen,
    rawRequestSnapshot,
    rawRequestJson,
    refreshRawRequestSnapshot,
    openRawRequestModal,
    copyRawRequestJson,
  } = rawPreview;
  React.useEffect(() => {
    setAttachedResearchContextDraft(
      cloneAttachedResearchContext(attachedResearchContext),
    );
  }, [attachedResearchContext]);
  const attachedResearchPreviewSuppressed =
    Boolean(attachedResearchContext) &&
    Boolean(rawRequestSnapshot) &&
    !(rawRequestSnapshot?.body as any)?.research_context;

  const applyAttachedResearchDraft = React.useCallback(() => {
    if (!attachedResearchContext || !attachedResearchContextDraft) return;
    const nextContext = applyAttachedResearchContextEdits(
      attachedResearchContext,
      attachedResearchContextDraft,
    );
    onApplyAttachedResearchContext?.(nextContext);
    setAttachedResearchContextDraft(cloneAttachedResearchContext(nextContext));
  }, [
    attachedResearchContext,
    attachedResearchContextDraft,
    onApplyAttachedResearchContext,
  ]);

  const handleResetAttachedResearchDraft = React.useCallback(() => {
    const resetContext = resetAttachedResearchContext(
      attachedResearchContextBaseline,
    );
    if (!resetContext) return;
    onResetAttachedResearchContext?.();
    setAttachedResearchContextDraft(cloneAttachedResearchContext(resetContext));
  }, [attachedResearchContextBaseline, onResetAttachedResearchContext]);

  const handleAttachedResearchDraftQuestionChange = React.useCallback(
    (event: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
      const nextQuestion = event.target.value;
      setAttachedResearchContextDraft((current) =>
        current ? { ...current, question: nextQuestion } : current,
      );
    },
    [],
  );

  const handleAttachedResearchDraftOutlineChange = React.useCallback(
    (event: React.ChangeEvent<HTMLTextAreaElement>) => {
      const nextOutline = event.target.value
        .split("\n")
        .map((title) => ({ title }));
      setAttachedResearchContextDraft((current) =>
        current ? { ...current, outline: nextOutline } : current,
      );
    },
    [],
  );

  const handleAttachedResearchDraftClaimsChange = React.useCallback(
    (event: React.ChangeEvent<HTMLTextAreaElement>) => {
      const nextClaims = event.target.value
        .split("\n")
        .map((text) => ({ text }));
      setAttachedResearchContextDraft((current) =>
        current ? { ...current, key_claims: nextClaims } : current,
      );
    },
    [],
  );

  const handleAttachedResearchDraftUnresolvedChange = React.useCallback(
    (event: React.ChangeEvent<HTMLTextAreaElement>) => {
      const nextQuestions = event.target.value.split("\n");
      setAttachedResearchContextDraft((current) =>
        current ? { ...current, unresolved_questions: nextQuestions } : current,
      );
    },
    [],
  );

  const navigateToWebSearchSettings = React.useCallback(() => {
    navigate("/settings");
  }, [navigate]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (!isFirefoxTarget) {
      if (e.key === "Process" || e.key === "229") return;
    }

    if (showSlashMenu) {
      if (e.key === "ArrowDown" && filteredSlashCommands.length > 0) {
        e.preventDefault();
        setSlashActiveIndex((prev) =>
          prev + 1 >= filteredSlashCommands.length ? 0 : prev + 1,
        );
        return;
      }
      if (e.key === "ArrowUp" && filteredSlashCommands.length > 0) {
        e.preventDefault();
        setSlashActiveIndex((prev) =>
          prev <= 0 ? filteredSlashCommands.length - 1 : prev - 1,
        );
        return;
      }
      if (
        (e.key === "Enter" || (e.key === "Tab" && !e.shiftKey)) &&
        filteredSlashCommands.length > 0
      ) {
        e.preventDefault();
        const command = filteredSlashCommands[slashActiveIndex];
        if (command) {
          handleSlashCommandSelect(command);
        }
        return;
      }
      if (e.key === "Escape") {
        e.preventDefault();
        form.setFieldValue(
          "message",
          form.values.message.replace(/^\s*\//, ""),
        );
        return;
      }
    }

    if (
      showMentions &&
      (e.key === "ArrowDown" ||
        e.key === "ArrowUp" ||
        e.key === "Enter" ||
        e.key === "Escape")
    ) {
      return;
    }

    if (!isConnectionReady) {
      if (e.key === "Enter") {
        e.preventDefault();
      }
      return;
    }

    if (
      handleChatInputKeyDown({
        e,
        sendWhenEnter,
        typing,
        isSending: false,
      })
    ) {
      e.preventDefault();
      stopListening();
      handleComposerSend();
    }
  };

  const handleCollapsedKeyDown = React.useCallback(
    (e: React.KeyboardEvent) => {
      if (!isMessageCollapsed || !collapsedRange) return false;

      const shouldSend = handleChatInputKeyDown({
        e,
        sendWhenEnter,
        typing,
        isSending: false,
      });
      if (shouldSend) return false;

      const currentValue = form.values.message || "";
      const meta = getCollapsedDisplayMeta(currentValue, collapsedRange);
      const textarea = textareaRef.current;
      const rawStart = textarea?.selectionStart ?? meta.labelEnd;
      const rawEnd = textarea?.selectionEnd ?? rawStart;
      const displayStart = Math.min(rawStart, rawEnd);
      const displayEnd = Math.max(rawStart, rawEnd);
      const hasSelection = displayStart !== displayEnd;
      const selectionTouchesLabel =
        displayStart < meta.labelEnd && displayEnd > meta.labelStart;
      const startPrefer =
        displayStart > meta.labelStart && displayStart < meta.labelEnd
          ? "before"
          : undefined;
      const endPrefer =
        displayEnd > meta.labelStart && displayEnd < meta.labelEnd
          ? "after"
          : undefined;
      let selectionStart = getMessageCaretFromDisplay(displayStart, meta, {
        prefer: startPrefer,
      });
      let selectionEnd = getMessageCaretFromDisplay(displayEnd, meta, {
        prefer: endPrefer,
      });
      if (selectionStart > selectionEnd) {
        [selectionStart, selectionEnd] = [selectionEnd, selectionStart];
      }
      if (selectionTouchesLabel) {
        selectionStart = Math.min(selectionStart, meta.rangeStart);
        selectionEnd = Math.max(selectionEnd, meta.rangeEnd);
      }
      const caretPrefer =
        rawStart > meta.labelStart && rawStart < meta.labelEnd
          ? pendingCaretRef.current !== null &&
            pendingCaretRef.current <= meta.rangeStart
            ? "before"
            : "after"
          : undefined;
      let caret = getMessageCaretFromDisplay(rawStart, meta, {
        prefer: caretPrefer,
      });
      if (caret > meta.rangeStart && caret < meta.rangeEnd) {
        caret = meta.rangeEnd;
      }

      const deleteCollapsedBlock = () => {
        const nextValue =
          currentValue.slice(0, meta.rangeStart) +
          currentValue.slice(meta.rangeEnd);
        const nextCaret = Math.min(meta.rangeStart, nextValue.length);
        commitCollapsedEdit(nextValue, nextCaret, null);
      };

      const insertAtCaret = (text: string) => {
        const insertAt =
          caret <= meta.rangeStart
            ? caret
            : caret >= meta.rangeEnd
              ? caret
              : meta.rangeEnd;
        replaceCollapsedRange(currentValue, meta, insertAt, insertAt, text);
      };

      if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
        if (e.shiftKey) return false;
        e.preventDefault();
        let nextCaret = caret;
        if (hasSelection) {
          nextCaret = e.key === "ArrowLeft" ? selectionStart : selectionEnd;
        } else {
          nextCaret =
            e.key === "ArrowLeft"
              ? Math.max(0, caret - 1)
              : Math.min(meta.messageLength, caret + 1);
          if (nextCaret > meta.rangeStart && nextCaret < meta.rangeEnd) {
            nextCaret = e.key === "ArrowLeft" ? meta.rangeStart : meta.rangeEnd;
          }
        }
        pendingCaretRef.current = nextCaret;
        syncCollapsedCaret({ caret: nextCaret });
        return true;
      }

      if (e.key === "Backspace") {
        if (hasSelection) {
          e.preventDefault();
          replaceCollapsedRange(
            currentValue,
            meta,
            selectionStart,
            selectionEnd,
            "",
          );
          return true;
        }
        if (caret === meta.rangeEnd) {
          e.preventDefault();
          deleteCollapsedBlock();
          return true;
        }
        if (caret === 0) {
          e.preventDefault();
          return true;
        }
        if (caret > meta.rangeStart && caret <= meta.rangeEnd) {
          e.preventDefault();
          deleteCollapsedBlock();
          return true;
        }
        e.preventDefault();
        replaceCollapsedRange(
          currentValue,
          meta,
          Math.max(0, caret - 1),
          caret,
          "",
        );
        return true;
      }

      if (e.key === "Delete") {
        if (hasSelection) {
          e.preventDefault();
          replaceCollapsedRange(
            currentValue,
            meta,
            selectionStart,
            selectionEnd,
            "",
          );
          return true;
        }
        if (caret === meta.rangeStart) {
          e.preventDefault();
          deleteCollapsedBlock();
          return true;
        }
        if (caret >= meta.messageLength) {
          e.preventDefault();
          return true;
        }
        if (caret >= meta.rangeStart && caret < meta.rangeEnd) {
          e.preventDefault();
          deleteCollapsedBlock();
          return true;
        }
        e.preventDefault();
        replaceCollapsedRange(currentValue, meta, caret, caret + 1, "");
        return true;
      }

      if (e.key === "Enter") {
        e.preventDefault();
        if (hasSelection) {
          replaceCollapsedRange(
            currentValue,
            meta,
            selectionStart,
            selectionEnd,
            "\n",
          );
        } else {
          insertAtCaret("\n");
        }
        return true;
      }

      if (e.key === " " || e.key === "Spacebar") {
        e.preventDefault();
        if (hasSelection) {
          replaceCollapsedRange(
            currentValue,
            meta,
            selectionStart,
            selectionEnd,
            " ",
          );
        } else {
          insertAtCaret(" ");
        }
        return true;
      }

      const isPrintable =
        e.key.length === 1 && !e.metaKey && !e.ctrlKey && !e.altKey;
      if (isPrintable) {
        e.preventDefault();
        if (hasSelection) {
          replaceCollapsedRange(
            currentValue,
            meta,
            selectionStart,
            selectionEnd,
            e.key,
          );
        } else {
          insertAtCaret(e.key);
        }
        return true;
      }

      return false;
    },
    [
      collapsedRange,
      commitCollapsedEdit,
      form.values.message,
      getCollapsedDisplayMeta,
      getMessageCaretFromDisplay,
      isMessageCollapsed,
      isSending,
      replaceCollapsedRange,
      sendWhenEnter,
      syncCollapsedCaret,
      textareaRef,
      typing,
    ],
  );

  const behaviorTemplateMetadata = React.useMemo(
    () =>
      getPromptTemplateById(currentChatModelSettings.systemPromptTemplateId),
    [currentChatModelSettings.systemPromptTemplateId],
  );
  const rolePlayScenePreview = React.useMemo(
    () => summarizeRolePlayScene(actorSettings),
    [actorSettings],
  );
  const rolePlayDocumentContextCount = Array.isArray(documentContext)
    ? documentContext.length
    : 0;
  const rolePlayIdentity = React.useMemo<RolePlayIdentity | null>(() => {
    if (
      selectedAssistant?.kind === "character" ||
      selectedAssistant?.kind === "persona"
    ) {
      return {
        kind: selectedAssistant.kind,
        id: selectedAssistant.id,
        name: selectedAssistant.name,
      };
    }
    if (selectedCharacter?.id || selectedCharacter?.name) {
      return {
        kind: "character",
        id:
          selectedCharacter?.id != null
            ? String(selectedCharacter.id)
            : undefined,
        name: selectedCharacter?.name,
      };
    }
    return null;
  }, [selectedAssistant, selectedCharacter]);
  const rolePlayState = React.useMemo(
    () =>
      deriveRolePlayState({
        identity: rolePlayIdentity,
        systemPrompt,
        selectedSystemPrompt,
        selectedQuickPrompt,
        behaviorTemplate: behaviorTemplateMetadata,
        scene: rolePlayScenePreview.active
          ? {
              active: true,
              summary: rolePlayScenePreview.summary,
            }
          : null,
        generationStyle:
          currentPreset && currentPreset.key !== "custom"
            ? {
                key: currentPreset.key,
                label: toText(
                  t(
                    `playground:presets.${currentPreset.key}.label`,
                    currentPreset.label,
                  ),
                ),
              }
            : null,
        context: {
          pinnedCount: ragPinnedResults.length,
          hasExternalContext:
            selectedDocuments.length > 0 ||
            uploadedFiles.length > 0 ||
            contextFiles.length > 0 ||
            rolePlayDocumentContextCount > 0 ||
            Boolean(attachedResearchContext),
        },
      }),
    [
      attachedResearchContext,
      behaviorTemplateMetadata,
      contextFiles.length,
      currentPreset,
      ragPinnedResults.length,
      rolePlayDocumentContextCount,
      rolePlayIdentity,
      rolePlayScenePreview.active,
      rolePlayScenePreview.summary,
      selectedDocuments.length,
      selectedQuickPrompt,
      selectedSystemPrompt,
      systemPrompt,
      t,
      uploadedFiles.length,
    ],
  );

  const contextItems = usePlaygroundContextItems({
    selectedModel,
    modelSummaryLabel,
    isSessionDegraded,
    connectionStatusLabel,
    compareModeActive,
    compareSelectedModels,
    currentPreset,
    selectedCharacterName: selectedCharacter?.name || null,
    characterPendingApply,
    contextToolsOpen,
    ragPinnedResultsLength: ragPinnedResults.length,
    webSearch,
    sessionUsageTotalTokens: sessionUsageSummary.totalTokens,
    sessionUsageLabel,
    selectedSystemPrompt,
    selectedQuickPrompt,
    systemPrompt,
    promptSummaryLabel,
    jsonMode: Boolean(currentChatModelSettings.jsonMode),
    showTokenBudgetWarning,
    tokenBudgetRiskLevel: tokenBudgetRisk.level,
    tokenBudgetRiskLabel,
    projectedBudgetUtilizationPercent: projectedBudget.utilizationPercent,
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
    rolePlayCompatibility: rawPreview.rolePlayCompatibility,
    onClearRolePlayIdentity: clearRolePlayIdentity,
    onClearRolePlayBehavior: clearPromptContext,
    onResetRolePlayGenerationStyle: resetRolePlayGenerationStyle,
    onDisableCompareMode: compareModeActive ? toggleCompareMode : undefined,
    onOpenRolePlaySetup: () => setRolePlaySetupOpen(true),
    t,
  });

  const settingsHook = usePlaygroundSettings({
    selectedCharacterName: selectedCharacter?.name || null,
    selectedSystemPrompt,
    selectedQuickPrompt,
    systemPrompt,
    ragPinnedResultsLength: ragPinnedResults.length,
    webSearch,
    jsonMode: Boolean(currentChatModelSettings.jsonMode),
    compareModeActive,
    compareNeedsMoreModels,
    compareCapabilityIncompatibilities,
    voiceChatEnabled,
    showTokenBudgetWarning,
    tokenBudgetWarningText,
    summaryCheckpointSuggestion,
    messagesLength: messages.length,
    showNonMessageContextWarning,
    nonMessageContextPercent,
    largestContextContributor,
    openKnowledgePanel,
    openContextWindowModal,
    openModelApiSelector,
    setOpenModelSettings,
    setModeLauncherOpen,
    setOpenActorSettings,
    trimLargestContextContributor,
    insertSummaryCheckpointPrompt,
    t,
  });
  const {
    compareSharedContextLabels,
    compareInteroperabilityNotices,
    contextConflictWarnings,
  } = settingsHook;

  const modeLauncherButton = (
    <PlaygroundModeLauncher
      open={modeLauncherOpen}
      onOpenChange={setModeLauncherOpen}
      compareModeActive={compareModeActive}
      compareFeatureEnabled={compareFeatureEnabled}
      onToggleCompare={toggleCompareMode}
      selectedCharacterName={selectedCharacter?.name || null}
      onOpenActorSettings={() => setOpenActorSettings(true)}
      contextToolsOpen={contextToolsOpen}
      onToggleKnowledgePanel={toggleKnowledgePanel}
      voiceChatEnabled={voiceChatEnabled}
      voiceChatAvailable={voiceChatAvailable}
      voiceChatUnavailableReason={voiceChatUnavailableReason}
      isSending={isSending}
      onVoiceChatToggle={handleVoiceChatToggle}
      webSearch={webSearch}
      hasWebSearch={!!capabilities?.hasWebSearch}
      onToggleWebSearch={handleToggleWebSearch}
      onModeAnnouncement={setModeAnnouncement}
      t={t}
    />
  );

  const externalPinSources =
    contextToolsOpen ||
    contextWindowModalOpen ||
    sessionInsightsOpen ||
    imageGenerateModalOpen ||
    mcpSettingsOpen ||
    openModelSettings ||
    openActorSettings ||
    rolePlaySetupOpen ||
    documentGeneratorOpen ||
    voiceModeSelectorOpen ||
    modelDropdownOpen ||
    mcpCtrl.mcpPopoverOpen ||
    modeLauncherOpen ||
    toolsPopoverOpen ||
    attachmentMenuOpen ||
    sendMenuOpen;

  const {
    actionBarVisible,
    composerFocusWithin,
    actionBarVisibilityClass,
    handlers: actionBarHandlers,
  } = useActionBarVisibility({ externalPinSources });
  const [composerOptionsExpanded, setComposerOptionsExpanded] = useStorage(
    "playgroundComposerOptionsExpanded",
    true,
  );
  const shouldCompactComposerTextarea =
    !composerFocusWithin &&
    !contextToolsOpen &&
    !showSlashMenu &&
    !showMentions &&
    !isSending &&
    !isMessageCollapsed &&
    messageDisplayValue.trim().length === 0;
  const composerShellRef = React.useRef<HTMLDivElement>(null);

  const keepComposerBottomInView = React.useCallback(() => {
    if (stickyDockEnabled) return;
    if (typeof window === "undefined") return;
    const composerEl = composerShellRef.current;
    if (!composerEl) return;

    const bottomPadding = 8;
    const scrollingElement = document.scrollingElement as HTMLElement | null;

    const adjustAncestor = (ancestor: HTMLElement | null) => {
      if (!ancestor) return;
      const composerRect = composerEl.getBoundingClientRect();

      if (ancestor === scrollingElement) {
        const viewportBottom = window.innerHeight - bottomPadding;
        const overflow = composerRect.bottom - viewportBottom;
        if (overflow > 0) {
          window.scrollBy({ top: overflow, behavior: "auto" });
        }
        return;
      }

      const ancestorRect = ancestor.getBoundingClientRect();
      const overflow =
        composerRect.bottom - (ancestorRect.bottom - bottomPadding);
      if (overflow > 0) {
        ancestor.scrollTop += overflow;
      }
    };

    let parent = composerEl.parentElement;
    while (parent) {
      const style = window.getComputedStyle(parent);
      const allowsVerticalScroll = /(auto|scroll|overlay)/.test(
        `${style.overflowY} ${style.overflow}`,
      );
      if (
        allowsVerticalScroll &&
        parent.scrollHeight > parent.clientHeight + 1
      ) {
        adjustAncestor(parent);
      }
      parent = parent.parentElement;
    }

    adjustAncestor(scrollingElement);
  }, [stickyDockEnabled]);

  const notifyComposerLayoutChange = React.useCallback(() => {
    if (!onComposerLayoutChange) return;

    const composerEl = composerShellRef.current;
    onComposerLayoutChange({
      occupiedHeightPx: composerEl
        ? Math.round(composerEl.getBoundingClientRect().height)
        : 0,
      keyboardInsetPx:
        stickyDockEnabled && isMobileViewport
          ? mobileComposerViewport.keyboardInsetPx
          : 0,
    });
  }, [
    isMobileViewport,
    mobileComposerViewport.keyboardInsetPx,
    onComposerLayoutChange,
    stickyDockEnabled,
  ]);

  React.useEffect(() => {
    if (!onComposerLayoutChange) return;
    if (typeof window === "undefined") return;

    let rafId = 0;
    const scheduleMeasurement = () => {
      if (rafId) {
        window.cancelAnimationFrame(rafId);
      }
      rafId = window.requestAnimationFrame(() => {
        rafId = 0;
        notifyComposerLayoutChange();
      });
    };

    const composerEl = composerShellRef.current;
    scheduleMeasurement();

    const resizeObserver =
      typeof ResizeObserver !== "undefined" && composerEl
        ? new ResizeObserver(() => {
            scheduleMeasurement();
          })
        : null;
    resizeObserver?.observe(composerEl);
    window.addEventListener("resize", scheduleMeasurement);

    return () => {
      if (rafId) {
        window.cancelAnimationFrame(rafId);
      }
      resizeObserver?.disconnect();
      window.removeEventListener("resize", scheduleMeasurement);
    };
  }, [notifyComposerLayoutChange, onComposerLayoutChange]);

  React.useEffect(() => {
    if (!onComposerLayoutChange) return;

    return () => {
      onComposerLayoutChange({
        occupiedHeightPx: 0,
        keyboardInsetPx: 0,
      });
    };
  }, [onComposerLayoutChange]);

  const previousActionBarVisibleRef = React.useRef(actionBarVisible);

  React.useEffect(() => {
    const wasVisible = previousActionBarVisibleRef.current;
    previousActionBarVisibleRef.current = actionBarVisible;

    if (!actionBarVisible || wasVisible) return;

    let timeoutId: number | null = null;
    const rafId = window.requestAnimationFrame(() => {
      keepComposerBottomInView();
      timeoutId = window.setTimeout(() => {
        keepComposerBottomInView();
      }, 220);
    });

    return () => {
      window.cancelAnimationFrame(rafId);
      if (timeoutId !== null) {
        window.clearTimeout(timeoutId);
      }
    };
  }, [actionBarVisible, keepComposerBottomInView]);

  const previousKeyboardOpenRef = React.useRef(false);
  React.useEffect(() => {
    if (!isMobileViewport) return;
    if (typeof window === "undefined") return;

    const wasOpen = previousKeyboardOpenRef.current;
    previousKeyboardOpenRef.current = mobileComposerViewport.keyboardOpen;

    if (!mobileComposerViewport.keyboardOpen && !wasOpen) {
      return;
    }

    let timeoutId: number | null = null;
    const rafId = window.requestAnimationFrame(() => {
      keepComposerBottomInView();
      timeoutId = window.setTimeout(() => {
        keepComposerBottomInView();
      }, 120);
    });

    return () => {
      window.cancelAnimationFrame(rafId);
      if (timeoutId != null) {
        window.clearTimeout(timeoutId);
      }
    };
  }, [
    isMobileViewport,
    keepComposerBottomInView,
    mobileComposerViewport.keyboardInsetPx,
    mobileComposerViewport.keyboardOpen,
  ]);

  React.useEffect(() => {
    if (typeof document === "undefined") return;

    const handleFocusIn = (event: FocusEvent) => {
      const target = event.target;
      if (!(target instanceof Node)) return;
      if (!composerShellRef.current?.contains(target)) return;
      keepComposerBottomInView();
      if (typeof window !== "undefined") {
        window.setTimeout(() => {
          keepComposerBottomInView();
        }, 80);
      }
    };

    document.addEventListener("focusin", handleFocusIn);
    return () => {
      document.removeEventListener("focusin", handleFocusIn);
    };
  }, [keepComposerBottomInView]);

  const previousSendStateRef = React.useRef(isSending);
  React.useEffect(() => {
    if (!isMobileViewport) {
      previousSendStateRef.current = isSending;
      return;
    }
    if (typeof window === "undefined") {
      previousSendStateRef.current = isSending;
      return;
    }

    const wasSending = previousSendStateRef.current;
    previousSendStateRef.current = isSending;

    if (!isSending && !wasSending) {
      return;
    }

    let timeoutId: number | null = null;
    const rafId = window.requestAnimationFrame(() => {
      keepComposerBottomInView();
      timeoutId = window.setTimeout(() => {
        keepComposerBottomInView();
      }, 100);
    });

    return () => {
      window.cancelAnimationFrame(rafId);
      if (timeoutId != null) {
        window.clearTimeout(timeoutId);
      }
    };
  }, [isMobileViewport, isSending, keepComposerBottomInView]);

  const toolRunStatusLabel = React.useMemo(() => {
    if (chatLoopState.pendingApprovals.length > 0) {
      return t("playground:composer.toolRunPending", "Pending approval");
    }
    if (
      chatLoopState.inflightToolCallIds.length > 0 ||
      chatLoopState.status === "running"
    ) {
      return t("playground:composer.toolRunRunning", "Running");
    }
    if (chatLoopState.status === "error") {
      return t("playground:composer.toolRunFailed", "Failed");
    }
    if (chatLoopState.status === "complete") {
      return t("playground:composer.toolRunDone", "Done");
    }
    return t("playground:composer.toolRunIdle", "Idle");
  }, [chatLoopState, t]);

  const handleMcpPopoverChange = React.useCallback(
    (open: boolean) => {
      if (open) {
        closeComposerPopoversExcept("mcp");
      }
      setMcpPopoverOpen(open);
    },
    [closeComposerPopoversExcept, setMcpPopoverOpen],
  );
  const handleToolsPopoverChange = React.useCallback(
    (open: boolean) => {
      if (open) {
        closeComposerPopoversExcept("tools");
      }
      setToolsPopoverOpen(open);
    },
    [closeComposerPopoversExcept],
  );
  const handleAttachmentMenuChange = React.useCallback(
    (open: boolean) => {
      if (open) {
        closeComposerPopoversExcept("attachment");
      }
      setAttachmentMenuOpen(open);
    },
    [closeComposerPopoversExcept],
  );
  const handleSendMenuChange = React.useCallback(
    (open: boolean) => {
      if (open) {
        closeComposerPopoversExcept("send");
      }
      setSendMenuOpen(open);
    },
    [closeComposerPopoversExcept],
  );

  const mcpControl = (
    <PlaygroundMcpControl
      hasMcp={hasMcp}
      mcpHealthState={mcpHealthState}
      mcpToolsLoading={mcpToolsLoading}
      mcpToolsCount={chatMcpTools.length}
      toolChoice={toolChoice}
      onToolChoiceChange={setToolChoice}
      toolRunStatusLabel={toolRunStatusLabel}
      mcpAriaLabel={mcpCtrl.mcpAriaLabel}
      mcpSummaryLabel={mcpCtrl.mcpSummaryLabel}
      mcpChoiceLabel={mcpCtrl.mcpChoiceLabel}
      mcpDisabledReason={mcpCtrl.mcpDisabledReason}
      mcpPopoverOpen={mcpCtrl.mcpPopoverOpen}
      onMcpPopoverChange={handleMcpPopoverChange}
      onOpenMcpSettings={() => {
        mcpSettingsReturnFocusSelectorRef.current = null;
        setMcpSettingsOpen(true);
      }}
      t={t}
    />
  );

  const voiceChatButton = voiceChatAvailable ? (
    <Tooltip
      title={
        voiceChatEnabled
          ? t("playground:voiceChat.toolbarStop", "Stop voice chat")
          : t("playground:voiceChat.toolbarStart", "Start voice chat")
      }
    >
      <button
        type="button"
        onClick={handleVoiceChatToggle}
        disabled={isSending}
        aria-label={voiceChatStatusLabel}
        data-testid="voice-chat-button"
        className={`flex items-center gap-1.5 rounded-full px-2 py-1.5 text-sm transition ${
          voiceChat.state === "error"
            ? "bg-danger/10 text-danger"
            : voiceChatEnabled && voiceChat.state !== "idle"
              ? "bg-primary/10 text-primary animate-pulse"
              : "hover:bg-surface2 text-text-muted"
        }`}
      >
        <Headphones className="h-4 w-4" />
        {voiceChatEnabled && voiceChat.state !== "idle" && (
          <span className="text-xs">{voiceChatStatusLabel}</span>
        )}
      </button>
    </Tooltip>
  ) : null;

  const toolsButton = (
    <PlaygroundToolsPopover
      toolsPopoverOpen={toolsPopoverOpen}
      onToolsPopoverChange={handleToolsPopoverChange}
      isProMode={isProMode}
      onOpenImageGenerate={openImageGenerateModal}
      onOpenKnowledgePanel={openKnowledgePanel}
      useOCR={useOCR}
      onUseOCRChange={setUseOCR}
      hasWebSearch={!!capabilities?.hasWebSearch}
      webSearch={webSearch}
      onWebSearchChange={setWebSearch}
      simpleInternetSearch={simpleInternetSearch}
      onSimpleInternetSearchChange={setSimpleInternetSearch}
      defaultInternetSearchOn={defaultInternetSearchOn}
      onDefaultInternetSearchOnChange={setDefaultInternetSearchOnSetting}
      onNavigateWebSearchSettings={navigateToWebSearchSettings}
      advancedToolsExpanded={advancedToolsExpanded}
      onAdvancedToolsExpandedChange={setAdvancedToolsExpanded}
      allowExternalImages={allowExternalImages}
      onAllowExternalImagesChange={setAllowExternalImages}
      showMoodBadge={showMoodBadge}
      onShowMoodBadgeChange={setShowMoodBadge}
      showMoodConfidence={showMoodConfidence}
      onShowMoodConfidenceChange={setShowMoodConfidence}
      onOpenRawRequest={openRawRequestModal}
      voiceChatAvailable={voiceChatAvailable}
      voiceChatEnabled={voiceChatEnabled}
      voiceChatUnavailableReason={voiceChatUnavailableReason}
      voiceChatState={voiceChat.state}
      voiceChatStatusLabel={voiceChatStatusLabel}
      onVoiceChatToggle={handleVoiceChatToggle}
      isSending={isSending}
      voiceChatSettingsFields={voiceChatSettingsFields}
      imageProviderControl={imageProviderControl}
      historyLength={history.length}
      onClearContext={handleClearContext}
      t={t}
    />
  );

  const attachmentButton = (
    <PlaygroundAttachmentButton
      isProMode={isProMode}
      isMobileViewport={isMobileViewport}
      chatMode={chatMode}
      onImageUpload={handleImageUpload}
      onDocumentUpload={handleDocumentUpload}
      onOpenKnowledgePanel={openKnowledgePanel}
      attachmentMenuOpen={attachmentMenuOpen}
      onAttachmentMenuChange={handleAttachmentMenuChange}
      t={t}
    />
  );
  const sendControl = (
    <PlaygroundSendControl
      isProMode={isProMode}
      isMobileViewport={isMobileViewport}
      isSending={isSending}
      isConnectionReady={isConnectionReady}
      sendWhenEnter={sendWhenEnter}
      onSendWhenEnterChange={setSendWhenEnter}
      sendLabel={sendLabel}
      compareNeedsMoreModels={compareNeedsMoreModels}
      onStopStreaming={stopStreamingRequest}
      onStopListening={stopListening}
      onSubmitForm={handleComposerSend}
      characterChatSendBlocker={characterChatSendBlocker}
      sendMenuOpen={sendMenuOpen}
      onSendMenuChange={handleSendMenuChange}
      t={t}
    />
  );

  const startupTemplatePromptResolution = startupTemplatePreview
    ? resolveStartupTemplatePrompt(startupTemplatePreview, promptLibrary)
    : null;
  const startupTemplatePromptDescription = startupTemplatePreview
    ? describeStartupTemplatePrompt(startupTemplatePreview, promptLibrary)
    : null;
  const startupTemplatePreset = startupTemplatePreview
    ? getPresetByKey(startupTemplatePreview.presetKey)
    : undefined;
  const rawRequestModalFooterExtras = attachedResearchContextDraft ? (
    <>
      <Button onClick={handleResetAttachedResearchDraft}>
        {t(
          "playground:actions.resetAttachedResearchContext",
          "Reset to Attached Run",
        )}
      </Button>
      <Button type="primary" onClick={applyAttachedResearchDraft}>
        {t("playground:actions.applyAttachedResearchContext", "Apply")}
      </Button>
    </>
  ) : null;
  const rawRequestAttachedResearchPanel = attachedResearchContextDraft ? (
    <div
      data-testid="attached-research-context-panel"
      className="space-y-3 rounded-md border border-border bg-surface px-3 py-3"
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="text-sm font-medium text-text">
            {t(
              "playground:tools.attachedResearchContextTitle",
              "Attached Research Context",
            )}
          </h3>
          <p className="text-xs text-text-muted">
            {attachedResearchContextDraft.query}
          </p>
        </div>
        <div className="space-y-1 text-right text-[11px] text-text-muted">
          <div>
            {t("playground:tools.attachedResearchRunId", "Run ID")}:{" "}
            <span className="font-mono">
              {attachedResearchContextDraft.run_id}
            </span>
          </div>
          <div>
            {t("playground:tools.attachedResearchAttachedAt", "Attached")}:{" "}
            {new Date(
              attachedResearchContextDraft.attached_at,
            ).toLocaleString()}
          </div>
        </div>
      </div>
      {attachedResearchPreviewSuppressed ? (
        <p className="text-xs text-text-muted">
          {t(
            "playground:tools.attachedResearchContextSuppressed",
            "Attached research is active but omitted from this request preview.",
          )}
        </p>
      ) : null}
      <div className="grid gap-3 md:grid-cols-2">
        <div className="space-y-1">
          <label className="text-xs font-medium text-text-muted">
            {t("playground:composer.context.question", "Question")}
          </label>
          <Input
            data-testid="attached-research-context-question-input"
            value={attachedResearchContextDraft.question}
            onChange={handleAttachedResearchDraftQuestionChange}
          />
        </div>
        <div className="space-y-1">
          <label className="text-xs font-medium text-text-muted">
            {t("playground:tools.attachedResearchContextLink", "Research link")}
          </label>
          <Input value={attachedResearchContextDraft.research_url} readOnly />
        </div>
        <div className="space-y-1">
          <label className="text-xs font-medium text-text-muted">
            {t("playground:composer.context.outline", "Outline")}
          </label>
          <Input.TextArea
            data-testid="attached-research-context-outline-input"
            value={stringifyOutline(attachedResearchContextDraft)}
            onChange={handleAttachedResearchDraftOutlineChange}
            autoSize={{ minRows: 3, maxRows: 6 }}
          />
        </div>
        <div className="space-y-1">
          <label className="text-xs font-medium text-text-muted">
            {t("playground:composer.context.claims", "Key claims")}
          </label>
          <Input.TextArea
            data-testid="attached-research-context-claims-input"
            value={stringifyKeyClaims(attachedResearchContextDraft)}
            onChange={handleAttachedResearchDraftClaimsChange}
            autoSize={{ minRows: 3, maxRows: 6 }}
          />
        </div>
        <div className="space-y-1 md:col-span-2">
          <label className="text-xs font-medium text-text-muted">
            {t(
              "playground:composer.context.unresolvedQuestions",
              "Unresolved questions",
            )}
          </label>
          <Input.TextArea
            data-testid="attached-research-context-unresolved-input"
            value={stringifyUnresolvedQuestions(attachedResearchContextDraft)}
            onChange={handleAttachedResearchDraftUnresolvedChange}
            autoSize={{ minRows: 3, maxRows: 6 }}
          />
        </div>
      </div>
    </div>
  ) : null;

  return (
    <React.Profiler
      id="playground-form-root"
      onRender={onComposerRenderProfile}
    >
      <div className="flex w-full flex-col items-center px-4 pb-6">
        <div
          data-checkwidemode={checkWideMode}
          data-ui-mode={uiMode}
          className="relative z-10 flex w-full max-w-[64rem] flex-col items-center justify-center gap-2 text-base data-[checkwidemode='true']:max-w-none"
        >
          <div className="relative flex w-full flex-row justify-center">
            <div
              ref={composerShellRef}
              data-istemporary-chat={temporaryChat}
              data-mobile-keyboard={
                isMobileViewport
                  ? mobileComposerViewport.keyboardOpen
                    ? "open"
                    : "closed"
                  : "desktop"
              }
              onMouseEnter={actionBarHandlers.onMouseEnter}
              onMouseLeave={actionBarHandlers.onMouseLeave}
              onFocusCapture={actionBarHandlers.onFocusCapture}
              onBlurCapture={actionBarHandlers.onBlurCapture}
              style={
                isMobileViewport
                  ? {
                      scrollMarginBottom: `${Math.max(
                        mobileComposerViewport.keyboardInsetPx,
                        16,
                      )}px`,
                      paddingBottom:
                        "calc(env(safe-area-inset-bottom, 0px) + 0.75rem)",
                    }
                  : undefined
              }
              className={`relative w-full rounded-3xl border border-transparent bg-surface/95 p-3 text-text shadow-card backdrop-blur-lg transition-all duration-200 data-[istemporary-chat='true']:border-t-4 data-[istemporary-chat='true']:border-t-purple-500 data-[istemporary-chat='true']:border-dashed data-[istemporary-chat='true']:opacity-90 ${
                !isConnectionReady ? "opacity-80" : ""
              }`}
            >
              {importedSidepanelContext
                ? wrapComposerProfile(
                    "sidepanel-imported-context",
                    <SidepanelImportedContextBanner
                      context={importedSidepanelContext}
                      onRemove={clearImportedSidepanelContext}
                      labels={{
                        defaultTitle: t(
                          "playground:sidepanelHandoff.importedContextTitle",
                          "Imported sidepanel context",
                        ),
                        regionLabel: t(
                          "playground:sidepanelHandoff.importedContextLabel",
                          "Imported sidepanel context",
                        ),
                        removeContext: (title) =>
                          t(
                            "playground:sidepanelHandoff.removeContext",
                            "Remove imported context from {{title}}",
                            { title },
                          ),
                        snippetSummary: (count) =>
                          t(
                            "playground:sidepanelHandoff.snippetSummary",
                            count === 1
                              ? "{{count}} snippet"
                              : "{{count}} snippets",
                            { count },
                          ),
                        noSnippets: t(
                          "playground:sidepanelHandoff.noSnippets",
                          "No snippets",
                        ),
                      }}
                    />,
                  )
                : null}
              {sidepanelHandoffConflict ? (
                <div
                  role="status"
                  aria-label={t(
                    "playground:sidepanelHandoff.conflictLabel",
                    "Sidepanel handoff conflict",
                  )}
                  className="mb-2 rounded-md border border-border bg-surface2 px-3 py-2 text-xs text-text"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="min-w-0">
                      <div className="font-medium">
                        {t(
                          "playground:sidepanelHandoff.conflictTitle",
                          "Import sidepanel draft?",
                        )}
                      </div>
                      <div className="text-text-muted">
                        {t(
                          "playground:sidepanelHandoff.conflictDescription",
                          "Your composer already has a draft. Insert the sidepanel draft, replace yours, or cancel the import.",
                        )}
                      </div>
                    </div>
                    <div className="flex flex-wrap items-center gap-2">
                      <button
                        type="button"
                        onClick={insertHandoffDraft}
                        className="min-h-7 rounded border border-border bg-surface px-3 py-1 text-xs text-text hover:bg-surface3 focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
                      >
                        {t(
                          "playground:sidepanelHandoff.insert",
                          "Insert handoff draft",
                        )}
                      </button>
                      <button
                        type="button"
                        onClick={replaceWithHandoffDraft}
                        className="min-h-7 rounded border border-border bg-surface px-3 py-1 text-xs text-text hover:bg-surface3 focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
                      >
                        {t(
                          "playground:sidepanelHandoff.replace",
                          "Replace current draft",
                        )}
                      </button>
                      <button
                        type="button"
                        onClick={cancelHandoffImport}
                        className="min-h-7 rounded border border-border bg-surface px-3 py-1 text-xs text-text hover:bg-surface3 focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
                      >
                        {t("playground:sidepanelHandoff.cancel", "Cancel import")}
                      </button>
                    </div>
                  </div>
                </div>
              ) : null}
              {/* Attachments summary (collapsed context management) */}
              {wrapComposerProfile(
                "attachments-summary",
                <AttachmentsSummary
                  image={form.values.image}
                  documents={selectedDocuments}
                  files={uploadedFiles}
                  onRemoveImage={() => form.setFieldValue("image", "")}
                  onRemoveDocument={removeDocument}
                  onClearDocuments={clearSelectedDocuments}
                  onRemoveFile={removeUploadedFile}
                  onClearFiles={clearUploadedFiles}
                  onOpenKnowledgePanel={() => openKnowledgePanel("context")}
                  readOnly
                />,
              )}
              {/* Link to Model Playground for Compare mode */}
              <div>
                <div className="flex w-full min-w-0 bg-transparent">
                  <form
                    onSubmit={(event) => {
                      event.preventDefault();
                      stopListening();
                      handleComposerSend();
                    }}
                    className="flex w-full min-w-0 flex-col items-center"
                  >
                    <input
                      id="file-upload"
                      name="file-upload"
                      type="file"
                      className="sr-only"
                      ref={inputRef}
                      accept="image/*"
                      multiple={false}
                      tabIndex={-1}
                      aria-hidden="true"
                      aria-label={
                        t(
                          "playground:actions.attachImage",
                          "Attach image",
                        ) as string
                      }
                      onChange={onInputChange}
                    />
                    <input
                      id="document-upload"
                      name="document-upload"
                      type="file"
                      className="sr-only"
                      ref={fileInputRef}
                      accept=".pdf,.doc,.docx,.txt,.csv,.md,.markdown,text/markdown"
                      multiple={false}
                      tabIndex={-1}
                      aria-hidden="true"
                      aria-label={
                        t(
                          "playground:actions.attachDocument",
                          "Attach document",
                        ) as string
                      }
                      onChange={onFileInputChange}
                    />

                    <div
                      className={`w-full flex flex-col px-2 ${
                        !isConnectionReady
                          ? "rounded-md border border-dashed border-border bg-surface2"
                          : ""
                      }`}
                    >
                      <PlaygroundChatErrorBanner
                        error={chatErrorBanner}
                        diagnosticsLabel={
                          t(
                            "playground:composer.errorBanner.openDiagnostics",
                            "View in Health & Diagnostics",
                          ) as string
                        }
                        dismissLabel={t("common:close", "Dismiss") as string}
                        onDismiss={dismissChatErrorBanner}
                      />
                      {showFollowUpResearchButton ? (
                        <div className="mb-2 flex flex-wrap items-center gap-2">
                          <Button
                            onClick={openFollowUpResearchModal}
                            disabled={followUpResearchPending}
                          >
                            {t(
                              "playground:actions.followUpResearch",
                              "Follow-up Research",
                            )}
                          </Button>
                        </div>
                      ) : null}
                      {attachedResearchContext ? (
                        <AttachedResearchContextChip
                          context={attachedResearchContext}
                          pinned={attachedResearchContextPinned}
                          history={attachedResearchContextHistory}
                          onPreview={openRawRequestModal}
                          onRemove={() => onRemoveAttachedResearchContext?.()}
                          onPin={() => onPinAttachedResearchContext?.()}
                          onUnpin={() => onUnpinAttachedResearchContext?.()}
                          onRestorePinned={() =>
                            onRestorePinnedResearchContext?.()
                          }
                          onPrepareResearchFollowUp={onPrepareResearchFollowUp}
                          onPinHistory={onPinAttachedResearchContextHistory}
                          onSelectHistory={
                            onSelectAttachedResearchContextHistory
                          }
                        />
                      ) : null}
                      {!attachedResearchContext &&
                      (attachedResearchContextPinned ||
                        attachedResearchContextHistory.length > 0) ? (
                        <div className="mb-2 flex flex-col gap-2">
                          {attachedResearchContextPinned ? (
                            <div
                              data-testid="pinned-research-fallback-card"
                              className="rounded-md border border-border bg-surface2 px-3 py-3 text-xs text-text"
                            >
                              <div className="flex flex-col gap-2">
                                <div className="flex flex-wrap items-center gap-2">
                                  <span className="font-medium text-text-muted">
                                    {t(
                                      "playground:composer.pinnedResearch",
                                      "Pinned research",
                                    )}
                                  </span>
                                  <span className="rounded border border-border bg-surface px-2 py-0.5 text-[11px] text-text">
                                    {attachedResearchContextPinned.query}
                                  </span>
                                </div>
                                <p className="text-[11px] text-text-muted">
                                  {t(
                                    "playground:composer.pinnedResearchFallbackDescription",
                                    "This thread keeps this research as its default context.",
                                  )}
                                </p>
                                <div className="flex flex-wrap items-center gap-2">
                                  <button
                                    type="button"
                                    onClick={() =>
                                      onRestorePinnedResearchContext?.()
                                    }
                                    className="rounded border border-border bg-surface px-2 py-0.5 text-[11px] text-text hover:bg-surface3"
                                  >
                                    {t(
                                      "playground:actions.usePinnedResearchNow",
                                      "Use now",
                                    )}
                                  </button>
                                  <button
                                    type="button"
                                    onClick={() =>
                                      onUnpinAttachedResearchContext?.()
                                    }
                                    className="rounded border border-border bg-surface px-2 py-0.5 text-[11px] text-text hover:bg-surface3"
                                  >
                                    {t(
                                      "playground:actions.unpinResearchContext",
                                      "Unpin",
                                    )}
                                  </button>
                                  <Link
                                    to={
                                      attachedResearchContextPinned.research_url
                                    }
                                    className="rounded border border-border bg-surface px-2 py-0.5 text-[11px] text-text hover:bg-surface3"
                                  >
                                    {t(
                                      "playground:actions.openInResearch",
                                      "Open in Research",
                                    )}
                                  </Link>
                                  {onPrepareResearchFollowUp ? (
                                    <button
                                      type="button"
                                      onClick={() =>
                                        setPendingAttachmentFollowUp({
                                          run_id:
                                            attachedResearchContextPinned.run_id,
                                          query:
                                            attachedResearchContextPinned.query,
                                        })
                                      }
                                      className="rounded border border-border bg-surface px-2 py-0.5 text-[11px] text-text hover:bg-surface3"
                                    >
                                      {t(
                                        "playground:actions.followUp",
                                        "Follow up",
                                      )}
                                    </button>
                                  ) : null}
                                </div>
                                {pendingAttachmentFollowUp?.run_id ===
                                attachedResearchContextPinned.run_id ? (
                                  <div className="space-y-2 rounded border border-border bg-surface px-3 py-2 text-[11px] text-text">
                                    <div className="font-medium">
                                      {t(
                                        "playground:actions.prepareFollowUpTitle",
                                        "Prepare follow-up?",
                                      )}
                                    </div>
                                    <div className="text-text-muted">
                                      {t(
                                        "playground:actions.prepareFollowUpBody",
                                        'This will use "{{query}}" and prefill a follow-up research prompt in the composer.',
                                        {
                                          query:
                                            pendingAttachmentFollowUp.query,
                                        },
                                      )}
                                    </div>
                                    <div className="flex flex-wrap items-center gap-2">
                                      <button
                                        type="button"
                                        onClick={() => {
                                          onPrepareResearchFollowUp?.(
                                            pendingAttachmentFollowUp,
                                          );
                                          setPendingAttachmentFollowUp(null);
                                        }}
                                        className="rounded border border-border bg-surface2 px-2 py-0.5 text-[11px] text-text hover:bg-surface3"
                                      >
                                        {t(
                                          "playground:actions.prepareFollowUp",
                                          "Prepare follow-up",
                                        )}
                                      </button>
                                      <button
                                        type="button"
                                        onClick={() =>
                                          setPendingAttachmentFollowUp(null)
                                        }
                                        className="rounded border border-border bg-surface2 px-2 py-0.5 text-[11px] text-text hover:bg-surface3"
                                      >
                                        {t("common:cancel", "Cancel")}
                                      </button>
                                    </div>
                                  </div>
                                ) : null}
                              </div>
                            </div>
                          ) : null}
                          {attachedResearchContextHistory.length > 0 ? (
                            <div
                              data-testid="pinned-research-history-block"
                              className="rounded-md border border-border bg-surface2 px-3 py-3 text-xs text-text"
                            >
                              <div className="mb-2 font-medium text-text-muted">
                                {t(
                                  "playground:composer.recentResearch",
                                  "Recent research",
                                )}
                              </div>
                              <div className="flex flex-col gap-2">
                                {attachedResearchContextHistory.map((entry) => (
                                  <div
                                    key={entry.run_id}
                                    className="rounded border border-border bg-surface px-3 py-2"
                                  >
                                    <div className="flex flex-wrap items-center justify-between gap-2">
                                      <div className="min-w-0">
                                        <div className="truncate font-medium text-text">
                                          {entry.query}
                                        </div>
                                        <div className="truncate text-[11px] text-text-muted">
                                          {entry.question}
                                        </div>
                                      </div>
                                      <div className="flex flex-wrap items-center gap-2">
                                        <button
                                          type="button"
                                          onClick={() =>
                                            onSelectAttachedResearchContextHistory?.(
                                              entry,
                                            )
                                          }
                                          className="rounded border border-border bg-surface2 px-2 py-0.5 text-[11px] text-text hover:bg-surface3"
                                        >
                                          {t(
                                            "playground:actions.usePinnedResearchNow",
                                            "Use now",
                                          )}
                                        </button>
                                        <button
                                          type="button"
                                          onClick={() =>
                                            onPinAttachedResearchContextHistory?.(
                                              entry,
                                            )
                                          }
                                          className="rounded border border-border bg-surface2 px-2 py-0.5 text-[11px] text-text hover:bg-surface3"
                                        >
                                          {t(
                                            "playground:actions.pinResearchContext",
                                            "Pin",
                                          )}
                                        </button>
                                        {onPrepareResearchFollowUp ? (
                                          <button
                                            type="button"
                                            onClick={() =>
                                              setPendingAttachmentFollowUp({
                                                run_id: entry.run_id,
                                                query: entry.query,
                                              })
                                            }
                                            className="rounded border border-border bg-surface2 px-2 py-0.5 text-[11px] text-text hover:bg-surface3"
                                          >
                                            {t(
                                              "playground:actions.followUp",
                                              "Follow up",
                                            )}
                                          </button>
                                        ) : null}
                                      </div>
                                    </div>
                                    {pendingAttachmentFollowUp?.run_id ===
                                    entry.run_id ? (
                                      <div className="mt-2 space-y-2 rounded border border-border bg-surface2 px-3 py-2 text-[11px] text-text">
                                        <div className="font-medium">
                                          {t(
                                            "playground:actions.prepareFollowUpTitle",
                                            "Prepare follow-up?",
                                          )}
                                        </div>
                                        <div className="text-text-muted">
                                          {t(
                                            "playground:actions.prepareFollowUpBody",
                                            'This will use "{{query}}" and prefill a follow-up research prompt in the composer.',
                                            {
                                              query:
                                                pendingAttachmentFollowUp.query,
                                            },
                                          )}
                                        </div>
                                        <div className="flex flex-wrap items-center gap-2">
                                          <button
                                            type="button"
                                            onClick={() => {
                                              onPrepareResearchFollowUp?.(
                                                pendingAttachmentFollowUp,
                                              );
                                              setPendingAttachmentFollowUp(
                                                null,
                                              );
                                            }}
                                            className="rounded border border-border bg-surface px-2 py-0.5 text-[11px] text-text hover:bg-surface3"
                                          >
                                            {t(
                                              "playground:actions.prepareFollowUp",
                                              "Prepare follow-up",
                                            )}
                                          </button>
                                          <button
                                            type="button"
                                            onClick={() =>
                                              setPendingAttachmentFollowUp(null)
                                            }
                                            className="rounded border border-border bg-surface px-2 py-0.5 text-[11px] text-text hover:bg-surface3"
                                          >
                                            {t("common:cancel", "Cancel")}
                                          </button>
                                        </div>
                                      </div>
                                    ) : null}
                                  </div>
                                ))}
                              </div>
                            </div>
                          ) : null}
                        </div>
                      ) : null}
                      <PlaygroundKnowledgeSection
                        contextToolsOpen={contextToolsOpen}
                        isConnectionReady={isConnectionReady}
                        knowledgePanelTab={knowledgePanelTab}
                        knowledgePanelTabRequestId={knowledgePanelTabRequestId}
                        deferredComposerInput={deferredComposerInput}
                        attachedImage={form.values.image}
                        attachedTabs={selectedDocuments}
                        availableTabs={availableTabs}
                        attachedFiles={uploadedFiles}
                        fileRetrievalEnabled={fileRetrievalEnabled}
                        onInsert={handleKnowledgeInsert}
                        onAsk={handleKnowledgeAsk}
                        onOpenChange={handleKnowledgePanelOpenChange}
                        onRemoveImage={handleKnowledgeRemoveImage}
                        onRemoveTab={removeDocument}
                        onAddTab={addDocument}
                        onClearTabs={clearSelectedDocuments}
                        onRefreshTabs={reloadTabs}
                        onAddFile={handleKnowledgeAddFile}
                        onRemoveFile={removeUploadedFile}
                        onClearFiles={clearUploadedFiles}
                        onFileRetrievalChange={setFileRetrievalEnabled}
                        wrapComposerProfile={wrapComposerProfile}
                        t={t}
                      />
                      {(() => {
                        const replyTargetNode =
                          isProMode && replyTarget ? (
                            <div className="mb-2 flex items-center justify-between gap-2 rounded-md border border-border bg-surface2 px-3 py-2 text-xs text-text">
                              <div className="flex min-w-0 items-center gap-2">
                                <CornerUpLeft className="h-3.5 w-3.5 text-text-subtle" />
                                <span className="min-w-0 truncate">
                                  {replyLabel}
                                </span>
                              </div>
                              <button
                                type="button"
                                onClick={clearReplyTarget}
                                aria-label={t(
                                  "common:clearReply",
                                  "Clear reply target",
                                )}
                                title={
                                  t(
                                    "common:clearReply",
                                    "Clear reply target",
                                  ) as string
                                }
                                className="rounded p-1 text-text-subtle hover:bg-surface hover:text-text focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
                              >
                                <X className="h-3.5 w-3.5" aria-hidden="true" />
                              </button>
                            </div>
                          ) : null;

                        const composerTextareaNode = (
                          <div className="relative">
                            {replyTargetNode}
                            <div
                              className={
                                isMobileViewport
                                  ? "grid grid-cols-[auto_minmax(0,1fr)] items-end gap-2"
                                  : "flex items-end gap-2"
                              }
                            >
                              <button
                                type="button"
                                data-testid="composer-options-toggle"
                                aria-controls="composer-options-panel"
                                aria-expanded={composerOptionsExpanded}
                                aria-label={
                                  composerOptionsExpanded
                                    ? (t(
                                        "playground:composer.hideOptions",
                                        "Hide composer options",
                                      ) as string)
                                    : (t(
                                        "playground:composer.showOptions",
                                        "Show composer options",
                                      ) as string)
                                }
                                title={
                                  composerOptionsExpanded
                                    ? (t(
                                        "playground:composer.hideOptions",
                                        "Hide composer options",
                                      ) as string)
                                    : (t(
                                        "playground:composer.showOptions",
                                        "Show composer options",
                                      ) as string)
                                }
                                onClick={() =>
                                  setComposerOptionsExpanded(
                                    !composerOptionsExpanded,
                                  )
                                }
                                className={`${
                                  suppressComposerToolbarForMobileCockpit
                                    ? "hidden"
                                    : "inline-flex"
                                } h-11 w-11 shrink-0 items-center justify-center rounded-2xl border border-border/70 bg-surface/80 text-text-muted transition hover:bg-surface2 hover:text-text focus:outline-none focus-visible:ring-2 focus-visible:ring-focus/40 ${
                                  composerOptionsExpanded
                                    ? "border-primary/40 text-primaryStrong"
                                    : ""
                                }`}
                              >
                                <ChevronRight
                                  className={`h-4 w-4 transition-transform ${
                                    composerOptionsExpanded ? "rotate-90" : ""
                                  }`}
                                  aria-hidden="true"
                                />
                              </button>
                              <div
                                className={
                                  isMobileViewport
                                    ? suppressComposerToolbarForMobileCockpit
                                      ? "col-span-2 min-w-0"
                                      : "min-w-0"
                                    : "min-w-0 flex-1"
                                }
                              >
                                {wrapComposerProfile(
                                  "composer-textarea",
                                  <ComposerTextarea
                                    textareaRef={textareaRef}
                                    value={form.values.message}
                                    displayValue={messageDisplayValue}
                                    onChange={handleTextareaChange}
                                    onKeyDown={handleTextareaKeyDown}
                                    onPaste={handlePaste}
                                    onFocus={handleTextareaFocus}
                                    onSelect={handleTextareaSelect}
                                    onCompositionStart={handleCompositionStart}
                                    onCompositionEnd={handleCompositionEnd}
                                    onMouseDown={handleTextareaMouseDown}
                                    onMouseUp={handleTextareaMouseUp}
                                    placeholder={
                                      isConnectionReady
                                        ? t(
                                            "playground:composer.placeholderWithMentions",
                                            "Type a message... (/ commands, @ mentions)",
                                          )
                                        : t(
                                            "playground:composer.connectionPlaceholder",
                                            "Connect to tldw to start chatting.",
                                          )
                                    }
                                    isProMode={isProMode}
                                    isMobile={isMobileViewport}
                                    isConnectionReady={isConnectionReady}
                                    isCollapsed={isMessageCollapsed}
                                    ariaExpanded={!isMessageCollapsed}
                                    compactWhenInactive={
                                      shouldCompactComposerTextarea
                                    }
                                    formInputProps={form.getInputProps(
                                      "message",
                                    )}
                                    showSlashMenu={showSlashMenu}
                                    slashCommands={filteredSlashCommands}
                                    slashActiveIndex={slashActiveIndex}
                                    onSlashSelect={handleSlashCommandSelect}
                                    onSlashActiveIndexChange={
                                      setSlashActiveIndex
                                    }
                                    slashEmptyLabel={t(
                                      "common:commandPalette.noResults",
                                      "No results found",
                                    )}
                                    showMentions={showMentions}
                                    filteredTabs={filteredTabs}
                                    mentionPosition={mentionPosition}
                                    onMentionSelect={handleMentionSelect}
                                    onMentionsClose={closeMentions}
                                    onMentionRefetch={handleMentionRefetch}
                                    onMentionsOpen={handleMentionsOpen}
                                    draftSaved={draftSaved}
                                  />,
                                )}
                              </div>
                              <div
                                data-testid="composer-inline-send-control"
                                className={
                                  isMobileViewport
                                    ? "col-span-2 flex justify-end self-end"
                                    : "flex shrink-0 items-end self-end"
                                }
                              >
                                {sendControl}
                              </div>
                            </div>
                          </div>
                        );

                        const composerInlineMessagesNode = (
                          <>
                            {form.errors.message && (
                              <div
                                role="alert"
                                aria-live="assertive"
                                aria-atomic="true"
                                className="flex items-center justify-between gap-2 px-2 py-1 text-xs text-danger animate-shake"
                              >
                                <div className="flex items-center gap-2">
                                  <svg
                                    className="h-3.5 w-3.5 flex-shrink-0"
                                    fill="currentColor"
                                    viewBox="0 0 20 20"
                                  >
                                    <path
                                      fillRule="evenodd"
                                      d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z"
                                      clipRule="evenodd"
                                    />
                                  </svg>
                                  <span>{form.errors.message}</span>
                                </div>
                                <button
                                  type="button"
                                  onClick={() =>
                                    form.clearFieldError("message")
                                  }
                                  className="flex-shrink-0 text-danger hover:text-danger"
                                  aria-label={
                                    t("common:dismiss", "Dismiss") as string
                                  }
                                  title={
                                    t("common:dismiss", "Dismiss") as string
                                  }
                                >
                                  <X className="h-3 w-3" />
                                </button>
                              </div>
                            )}
                            {!form.errors.message &&
                              isConnectionReady &&
                              !isSending &&
                              isProMode && (
                                <div className="px-2 py-1 text-label text-text-subtle">
                                  {!selectedModel && !activeImageCommand ? (
                                    <span className="flex items-center gap-1">
                                      <svg
                                        className="h-3 w-3"
                                        fill="none"
                                        stroke="currentColor"
                                        viewBox="0 0 24 24"
                                      >
                                        <path
                                          strokeLinecap="round"
                                          strokeLinejoin="round"
                                          strokeWidth={2}
                                          d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                                        />
                                      </svg>
                                      {t(
                                        "sidepanel:composer.hints.selectModel",
                                        "Select a model above to start chatting",
                                      )}
                                    </span>
                                  ) : form.values.message.trim().length === 0 &&
                                    form.values.image.length === 0 ? (
                                    <span>
                                      {sendWhenEnter
                                        ? t(
                                            "sidepanel:composer.hints.typeAndEnter",
                                            "Type a message and press Enter to send",
                                          )
                                        : t(
                                            "sidepanel:composer.hints.typeAndClick",
                                            "Type a message and click Send",
                                          )}
                                    </span>
                                  ) : null}
                                </div>
                              )}
                          </>
                        );
                        /* Guarded top-level notice/modal contract:
                          Changed since last send:
                          playground:composer.providerDegraded
                          playground:composer.compareActivationTitle
                          playground:composer.compareActivationBody
                          playground:composer.compareActivationInteroperability
                          compare-interoperability-notices
                          playground:composer.validationCompareMinModelsInline
                          playground:composer.jsonModeHint
                          playground:composer.characterPendingNotice
                          { mode: "voice" }
                          previousSendStateRef
                          tldw:focus-composer
                          el.focus()
                          tldw:toggle-compare-mode
                          tldw:toggle-mode-launcher
                          ContextFootprintPanel
                          playground:composer.context.sessionStatus
                          playground:composer.context.truncationRisk
                          playground:composer.context.contextMix
                          isSessionDegraded
                          playground:composer.conflict.summaryCheckpointBudget
                          playground:composer.conflict.contextFootprint
                          evaluateSummaryCheckpointSuggestion
                          resolveTokenBudgetRisk
                          playground:tokens.truncationRisk
                          tldw:playground-starter-selected
                          SessionInsightsPanel
                          ModelRecommendationsPanel
                          buildSessionInsights
                          buildModelRecommendations
                          buildCompareInteroperabilityNotices
                          data-testid="startup-template-controls"
                          startup-template-preview-modal
                          image-refine-with-llm
                          image-prompt-refine-diff
                          imageGenerationRefine
                          playground:insights.modalTitle
                          playground:composer.startupTemplatePreviewTitle */
                        const composerNoticesNode = (
                          <PlaygroundComposerNotices
                            modeAnnouncement={modeAnnouncement}
                            characterPendingApply={characterPendingApply}
                            selectedCharacterGreeting={
                              selectedCharacterGreeting
                            }
                            selectedCharacterName={
                              selectedCharacter?.name || null
                            }
                            compareModeActive={compareModeActive}
                            compareSelectedModels={compareSelectedModels}
                            compareSelectedModelLabels={
                              compareSelectedModelLabels
                            }
                            compareNeedsMoreModels={compareNeedsMoreModels}
                            compareSharedContextLabels={
                              compareSharedContextLabels
                            }
                            compareInteroperabilityNotices={
                              compareInteroperabilityNotices
                            }
                            noticesExpanded={noticesExpanded}
                            setNoticesExpanded={setNoticesExpanded}
                            contextDeltaLabels={contextDeltaLabels}
                            contextConflictWarnings={contextConflictWarnings}
                            visibleModelRecommendations={
                              visibleModelRecommendations
                            }
                            sessionInsightsTotalTokens={
                              sessionInsights.totals.totalTokens
                            }
                            jsonMode={Boolean(
                              currentChatModelSettings.jsonMode,
                            )}
                            isConnectionReady={isConnectionReady}
                            connectionUxState={connectionUxState}
                            isProMode={isProMode}
                            selectedModel={selectedModel}
                            systemPrompt={systemPrompt}
                            selectedCharacter={selectedCharacter}
                            ragPinnedResultsLength={ragPinnedResults.length}
                            startupTemplateDraftName={startupTemplateDraftName}
                            setStartupTemplateDraftName={
                              setStartupTemplateDraftName
                            }
                            startupTemplates={startupTemplates}
                            handleSaveStartupTemplate={
                              handleSaveStartupTemplate
                            }
                            handleOpenStartupTemplatePreview={
                              handleOpenStartupTemplatePreview
                            }
                            setOpenModelSettings={setOpenModelSettings}
                            setOpenActorSettings={setOpenActorSettings}
                            setMessageValue={setMessageValue}
                            textAreaFocus={textAreaFocus}
                            openModelApiSelector={openModelApiSelector}
                            openSessionInsightsModal={openSessionInsightsModal}
                            handleModelRecommendationAction={
                              handleModelRecommendationAction
                            }
                            dismissModelRecommendation={
                              dismissModelRecommendation
                            }
                            getModelRecommendationActionLabel={
                              getModelRecommendationActionLabel
                            }
                            wrapComposerProfile={wrapComposerProfile}
                            t={t}
                          />
                        );

                        const composerToolbarNode = (
                          <div
                            id="composer-options-panel"
                            aria-hidden={!actionBarVisible}
                            className={`w-full transition-all duration-200 overflow-hidden ${actionBarVisibilityClass}`}
                          >
                            {wrapComposerProfile(
                              "composer-toolbar",
                              <ComposerToolbar
                                isProMode={isProMode}
                                isMobile={isMobileViewport}
                                isConnectionReady={isConnectionReady}
                                isSending={isSending}
                                optionsExpanded={composerOptionsExpanded}
                                modeLauncherButton={modeLauncherButton}
                                compareControl={compareControl}
                                modelSelectButton={modelSelectButton}
                                mcpControl={mcpControl}
                                sendControl={sendControl}
                                sendControlPlacement="external"
                                attachmentButton={attachmentButton}
                                toolsButton={toolsButton}
                                voiceChatButton={voiceChatButton}
                                modelUsageBadge={modelUsageBadge}
                                selectedSystemPrompt={selectedSystemPrompt}
                                systemPrompt={systemPrompt}
                                setSystemPrompt={setSystemPrompt}
                                setSelectedSystemPrompt={
                                  setSelectedSystemPromptForComposer
                                }
                                setSelectedQuickPrompt={
                                  setSelectedQuickPromptForComposer
                                }
                                temporaryChat={temporaryChat}
                                onToggleTemporaryChat={
                                  handleToggleTemporaryChat
                                }
                                privateChatLocked={privateChatLocked}
                                isFireFoxPrivateMode={isFireFoxPrivateMode}
                                persistenceTooltip={persistenceTooltip}
                                contextToolsOpen={contextToolsOpen}
                                onToggleKnowledgePanel={toggleKnowledgePanel}
                                webSearch={webSearch}
                                onToggleWebSearch={handleToggleWebSearch}
                                hasWebSearch={!!capabilities?.hasWebSearch}
                                onOpenModelSettings={handleOpenModelSettings}
                                modelSummaryLabel={modelSummaryLabel}
                                promptSummaryLabel={promptSummaryLabel}
                                hasDictation={
                                  !!(
                                    browserSupportsSpeechRecognition ||
                                    hasServerStt
                                  )
                                }
                                speechAvailable={speechAvailable}
                                speechUsesServer={speechUsesServer}
                                isListening={isListening}
                                isServerDictating={isServerDictating}
                                voiceChatEnabled={voiceChatEnabled}
                                speechTooltip={speechTooltipText}
                                onDictationToggle={handleDictationToggle}
                                onTemplateSelect={handleTemplateSelect}
                                selectedModel={selectedModel}
                                resolvedProviderKey={resolvedProviderKey}
                                messages={messages}
                                selectedDocumentsCount={
                                  selectedDocuments.length
                                }
                                uploadedFilesCount={uploadedFiles.length}
                                serverChatId={serverChatId}
                                showServerPersistenceHint={
                                  showServerPersistenceHint
                                }
                                onDismissServerPersistenceHint={
                                  handleDismissServerPersistenceHint
                                }
                                onFocusConnectionCard={focusConnectionCard}
                                contextItems={contextItems}
                                rolePlayActions={{
                                  onOpenRolePlaySetup: () =>
                                    setRolePlaySetupOpen(true),
                                }}
                              />,
                            )}
                          </div>
                        );
                        const composerToolbarSlot =
                          suppressComposerToolbarForMobileCockpit
                            ? (
                                <div className="hidden">{composerToolbarNode}</div>
                              )
                            : composerToolbarNode;

                        if (nextgenComposerEnabled) {
                          const sharedNoticesSlot = (
                            <>
                              {composerInlineMessagesNode}
                              {composerNoticesNode}
                            </>
                          );
                          const tokensProp =
                            resolvedMaxContext > 0
                              ? {
                                  used: conversationTokenCount,
                                  max: resolvedMaxContext,
                                }
                              : undefined;

                          // V3 brief panel + V5 facet row data sourced from the
                          // same Playground state the legacy composer reads.
                          // Click handlers route to the same actions the
                          // legacy toolbar exposes — model picker, character
                          // sheet, knowledge panel, web-search toggle.
                          const briefFields: Array<{
                            id: string;
                            fieldKey: string;
                            value: string;
                            active?: boolean;
                            onClick?: () => void;
                            "aria-label"?: string;
                          }> = [
                            {
                              id: "model",
                              fieldKey: "mdl",
                              value: selectedModel || "—",
                              active: Boolean(selectedModel),
                              onClick: handleOpenModelSettings,
                              "aria-label": selectedModel
                                ? `Change model (current: ${selectedModel})`
                                : "Pick a model",
                            },
                            {
                              id: "character",
                              fieldKey: "chr",
                              value: selectedCharacter?.name || "—",
                              active: Boolean(selectedCharacter?.name),
                              onClick: () => setOpenActorSettings(true),
                              "aria-label": selectedCharacter?.name
                                ? `Change character (current: ${selectedCharacter.name})`
                                : "Pick a character",
                            },
                            {
                              id: "sources",
                              fieldKey: "src",
                              value: String(
                                selectedDocuments.length + uploadedFiles.length,
                              ),
                              active:
                                selectedDocuments.length +
                                  uploadedFiles.length >
                                0,
                              onClick: () => toggleKnowledgePanel(),
                              "aria-label": "Open knowledge panel",
                            },
                            {
                              id: "web",
                              fieldKey: "web",
                              value: webSearch ? "on" : "off",
                              active: webSearch,
                              onClick: handleToggleWebSearch,
                              "aria-label": webSearch
                                ? "Turn web search off"
                                : "Turn web search on",
                            },
                          ];
                          const briefSections = [
                            {
                              id: "playground-brief",
                              label: "Brief",
                              fields: briefFields,
                            },
                          ];

                          const variantNode =
                            nextgenComposerVariant === "v5" ? (
                              <ChatComposer
                                variant="v5"
                                message={form.values.message}
                                onMessageChange={(value) =>
                                  form.setFieldValue("message", value)
                                }
                                onSend={handleComposerSend}
                                sending={isSending}
                                stopStreaming={stopStreamingRequest}
                                tokens={tokensProp}
                                onPaletteTrigger={() => {
                                  // V5 design: ⌘K injects `/` into the
                                  // textarea, opening the existing
                                  // ComposerTextarea slash menu instead
                                  // of a separate palette surface.
                                  const current = form.values.message;
                                  form.setFieldValue(
                                    "message",
                                    current.endsWith("/")
                                      ? current
                                      : `${current}/`,
                                  );
                                  textareaRef.current?.focus();
                                }}
                                textareaSlot={composerTextareaNode}
                                facetsSlot={composerToolbarSlot}
                                noticesSlot={sharedNoticesSlot}
                              />
                            ) : nextgenComposerVariant === "v3" ? (
                              <ChatComposer
                                variant="v3"
                                message={form.values.message}
                                onMessageChange={(value) =>
                                  form.setFieldValue("message", value)
                                }
                                onSend={handleComposerSend}
                                sending={isSending}
                                stopStreaming={stopStreamingRequest}
                                tokens={tokensProp}
                                briefSections={briefSections}
                                textareaSlot={composerTextareaNode}
                                bottomBarSlot={composerToolbarSlot}
                                noticesSlot={sharedNoticesSlot}
                              />
                            ) : (
                              <ChatComposer
                                variant="v1"
                                message={form.values.message}
                                onMessageChange={(value) =>
                                  form.setFieldValue("message", value)
                                }
                                onSend={handleComposerSend}
                                sending={isSending}
                                stopStreaming={stopStreamingRequest}
                                tokens={tokensProp}
                                textareaSlot={composerTextareaNode}
                                bottomBarSlot={composerToolbarSlot}
                                noticesSlot={sharedNoticesSlot}
                              />
                            );

                          return (
                            <div data-testid="nextgen-composer-wrapper">
                              {variantNode}
                            </div>
                          );
                        }

                        return (
                          <>
                            {composerTextareaNode}
                            {composerToolbarSlot}
                            {composerInlineMessagesNode}
                            {composerNoticesNode}
                          </>
                        );
                      })()}
                      {showConnectBanner && !isConnectionReady && (
                        <div className="mt-2 flex flex-wrap items-center justify-between gap-2 rounded-md border border-warn/30 bg-warn/10 px-3 py-2 text-xs text-warn">
                          <p className="max-w-xs text-left">
                            {t(
                              "playground:composer.connectNotice",
                              "Connect to your tldw server in Settings to send messages.",
                            )}
                          </p>
                          <div className="flex flex-wrap items-center gap-2">
                            <Link
                              to="/settings/tldw"
                              className="text-xs font-medium text-warn underline hover:text-warn"
                            >
                              {t("settings:tldw.setupLink", "Set up server")}
                            </Link>
                            <Link
                              to="/settings/health"
                              className="text-xs font-medium text-warn underline hover:text-warn"
                            >
                              {t(
                                "settings:healthSummary.diagnostics",
                                "Health & diagnostics",
                              )}
                            </Link>
                            <button
                              type="button"
                              onClick={() => setShowConnectBanner(false)}
                              className="inline-flex items-center rounded-full p-1 text-warn hover:bg-warn/10"
                              aria-label={t("common:close", "Dismiss")}
                              title={t("common:close", "Dismiss") as string}
                            >
                              <X className="h-3 w-3" />
                            </button>
                          </div>
                        </div>
                      )}
                      <ChatQueuePanel
                        queue={queuedMessages}
                        isConnectionReady={isConnectionReady}
                        isStreaming={isSending}
                        onRunNext={handleRunNextQueuedRequest}
                        onRunNow={handleRunQueuedRequest}
                        onDelete={queuedRequestActions.remove}
                        onMove={queuedRequestActions.move}
                        onUpdate={queuedRequestActions.update}
                        onClearAll={queuedRequestActions.clear}
                        onOpenDiagnostics={() => navigate("/settings/health")}
                        forceRunDisabledReason={
                          cancelCurrentAndRunDisabledReason
                        }
                      />
                    </div>
                  </form>
                </div>
              </div>
            </div>
          </div>
        </div>
        {imageGenerateModalOpen && (
          <React.Suspense fallback={null}>
            <LazyPlaygroundImageGenModal
              open={imageGenerateModalOpen}
              onClose={closeImageGenerateModal}
              busy={imageGenerateBusy}
              backend={imageGenerateBackend}
              backendOptions={imageGenerateBackendOptions}
              onBackendChange={setImageGenerateBackend}
              onHydrateSettings={hydrateImageGenerateSettings}
              promptMode={imageGeneratePromptMode}
              onPromptModeChange={setImageGeneratePromptMode}
              promptStrategies={imagePromptStrategies}
              syncPolicy={imageGenerateSyncPolicy}
              onSyncPolicyChange={setImageGenerateSyncPolicy}
              syncChatMode={imageEventSyncChatMode}
              onSyncChatModeChange={(next) =>
                void updateChatSettings({ imageEventSyncMode: next })
              }
              syncGlobalDefault={imageEventSyncGlobalDefault}
              onSyncGlobalDefaultChange={(next) =>
                void setImageEventSyncGlobalDefault(next)
              }
              resolvedSyncMode={imageGenerateResolvedSyncMode}
              prompt={imageGeneratePrompt}
              onPromptChange={setImageGeneratePrompt}
              contextBreakdown={imagePromptContextBreakdown}
              onClearRefineState={clearImagePromptRefineState}
              refineSubmitting={imagePromptRefineSubmitting}
              refineBaseline={imagePromptRefineBaseline}
              refineCandidate={imagePromptRefineCandidate}
              refineModel={imagePromptRefineModel}
              refineLatencyMs={imagePromptRefineLatencyMs}
              refineDiff={imagePromptRefineDiff}
              onCreateDraft={handleCreateImagePromptDraft}
              onRefine={handleRefineImagePromptDraft}
              onApplyRefined={applyRefinedImagePromptCandidate}
              onRejectRefined={rejectRefinedImagePromptCandidate}
              format={imageGenerateFormat}
              onFormatChange={setImageGenerateFormat}
              width={imageGenerateWidth}
              onWidthChange={setImageGenerateWidth}
              height={imageGenerateHeight}
              onHeightChange={setImageGenerateHeight}
              steps={imageGenerateSteps}
              onStepsChange={setImageGenerateSteps}
              cfgScale={imageGenerateCfgScale}
              onCfgScaleChange={setImageGenerateCfgScale}
              seed={imageGenerateSeed}
              onSeedChange={setImageGenerateSeed}
              sampler={imageGenerateSampler}
              onSamplerChange={setImageGenerateSampler}
              model={imageGenerateModel}
              onModelChange={setImageGenerateModel}
              negativePrompt={imageGenerateNegativePrompt}
              onNegativePromptChange={setImageGenerateNegativePrompt}
              extraParams={imageGenerateExtraParams}
              onExtraParamsChange={setImageGenerateExtraParams}
              referenceFileId={imageGenerateReferenceFileId}
              onReferenceFileIdChange={setImageGenerateReferenceFileId}
              referenceImageCandidates={referenceImageCandidates}
              referenceImageCandidatesLoading={referenceImageCandidatesLoading}
              submitting={imageGenerateSubmitting}
              onSubmit={submitImageGenerateModal}
              t={t}
            />
          </React.Suspense>
        )}
        <Modal
          open={followUpResearchModalOpen}
          onCancel={closeFollowUpResearchModal}
          destroyOnHidden
          title={t("playground:actions.followUpResearch", "Follow-up Research")}
          footer={
            <div className="flex flex-wrap justify-end gap-2">
              <Button
                onClick={closeFollowUpResearchModal}
                disabled={followUpResearchPending}
              >
                {t("common:cancel", "Cancel")}
              </Button>
              <Button
                type="primary"
                onClick={() => {
                  void handleStartFollowUpResearch();
                }}
                disabled={!canLaunchFollowUpResearch || followUpResearchPending}
                loading={followUpResearchPending}
              >
                {t("playground:actions.startResearch", "Start research")}
              </Button>
            </div>
          }
        >
          <div className="space-y-3">
            <p className="text-sm text-text-muted">
              {t(
                "playground:actions.followUpResearchBody",
                "Start a new linked research run from the current draft without sending a chat message.",
              )}
            </p>
            <div className="rounded-md border border-border bg-surface px-3 py-2">
              <div className="text-xs font-semibold uppercase tracking-wide text-text-muted">
                {t("playground:actions.followUpResearchQuery", "Query")}
              </div>
              <div className="mt-1 text-sm text-text">
                {followUpResearchDraftQuery}
              </div>
            </div>
            {attachedResearchContext ? (
              <Checkbox
                checked={includeAttachedResearchAsBackground}
                onChange={(event) =>
                  setIncludeAttachedResearchAsBackground(event.target.checked)
                }
                disabled={followUpResearchPending}
              >
                {t(
                  "playground:actions.followUpResearchUseAttachedBackground",
                  "Use attached research as background",
                )}
              </Checkbox>
            ) : null}
          </div>
        </Modal>
        {rawRequestModalOpen && (
          <React.Suspense fallback={null}>
            <LazyPlaygroundRawRequestModal
              open={rawRequestModalOpen}
              onClose={() => setRawRequestModalOpen(false)}
              snapshot={rawRequestSnapshot}
              json={rawRequestJson}
              onRefresh={refreshRawRequestSnapshot}
              onCopy={copyRawRequestJson}
              extraFooter={rawRequestModalFooterExtras}
              beforeJson={rawRequestAttachedResearchPanel}
              t={t}
            />
          </React.Suspense>
        )}
        {startupTemplatePreview && (
          <React.Suspense fallback={null}>
            <LazyPlaygroundStartupTemplateModal
              preview={startupTemplatePreview}
              onClose={() => setStartupTemplatePreview(null)}
              onDelete={handleDeleteStartupTemplate}
              onApply={handleApplyStartupTemplatePreview}
              promptDescription={startupTemplatePromptDescription}
              promptResolution={startupTemplatePromptResolution}
              preset={startupTemplatePreset}
              t={t}
            />
          </React.Suspense>
        )}
        {(contextWindowModalOpen || sessionInsightsOpen) && (
          <React.Suspense fallback={null}>
            <LazyPlaygroundContextWindowModal
              contextWindowModalOpen={contextWindowModalOpen}
              onCloseContextWindow={() => setContextWindowModalOpen(false)}
              onSaveContextWindow={saveContextWindowSetting}
              onResetContextWindow={resetContextWindowSetting}
              contextWindowDraftValue={contextWindowDraftValue}
              onContextWindowDraftChange={setContextWindowDraftValue}
              resolvedMaxContext={resolvedMaxContext}
              requestedContextWindowOverride={requestedContextWindowOverride}
              modelContextLength={modelContextLength}
              isContextWindowOverrideActive={isContextWindowOverrideActive}
              isContextWindowOverrideClamped={isContextWindowOverrideClamped}
              nonMessageContextPercent={nonMessageContextPercent}
              showNonMessageContextWarning={showNonMessageContextWarning}
              tokenBudgetRiskLabel={tokenBudgetRiskLabel}
              tokenBudgetRisk={tokenBudgetRisk}
              contextFootprintRows={contextFootprintRows}
              formatContextWindowValue={formatContextWindowValue}
              onClearPromptContext={clearPromptContext}
              onClearPinnedSourceContext={clearPinnedSourceContext}
              onClearHistoryContext={clearHistoryContext}
              onCreateSummaryCheckpoint={insertSummaryCheckpointPrompt}
              onReviewCharacterContext={() => setOpenActorSettings(true)}
              onTrimLargestContextContributor={trimLargestContextContributor}
              sessionInsightsOpen={sessionInsightsOpen}
              onCloseSessionInsights={() => setSessionInsightsOpen(false)}
              sessionInsights={sessionInsights}
              t={t}
            />
          </React.Suspense>
        )}
        {mcpSettingsOpen && (
          <React.Suspense fallback={null}>
            <LazyPlaygroundMcpSettingsModal
              open={mcpSettingsOpen}
              onClose={() => setMcpSettingsOpenWithFocusRestore(false)}
              hasMcp={hasMcp}
              mcpStatusLabel={mcpCtrl.mcpStatusLabel}
              catalogsLoading={mcpCatalogsLoading}
              catalogGroups={mcpCtrl.catalogGroups}
              catalogDraft={mcpCtrl.catalogDraft}
              onCatalogDraftChange={mcpCtrl.setCatalogDraft}
              onCatalogCommit={mcpCtrl.commitCatalog}
              onCatalogSelect={mcpCtrl.handleCatalogSelect}
              toolCatalogId={toolCatalogId}
              onToolCatalogIdChange={setToolCatalogId}
              toolCatalogStrict={toolCatalogStrict}
              onToolCatalogStrictChange={setToolCatalogStrict}
              moduleOptions={moduleOptions}
              moduleOptionsLoading={moduleOptionsLoading}
              toolModules={toolModules}
              onModuleSelect={handleModuleSelect}
              discoveredTools={discoveredMcpTools}
              toolCounts={mcpToolCounts}
              toolsLoading={mcpToolsLoading}
              mcpHealthState={mcpHealthState}
              onToolEnabledChange={setMcpToolEnabled}
              onResetToolFilter={resetMcpToolFilter}
              isSmallModel={isSmallModel}
              t={t}
            />
          </React.Suspense>
        )}
        {openModelSettings && (
          <React.Suspense fallback={null}>
            <LazyCurrentChatModelSettings
              open={openModelSettings}
              setOpen={setOpenModelSettingsWithFocusRestore}
              settingsScope={modelSettingsScopeOverride}
              isOCREnabled={useOCR}
            />
          </React.Suspense>
        )}
        {openActorSettings && (
          <React.Suspense fallback={null}>
            <LazyActorPopout
              open={openActorSettings}
              setOpen={setOpenActorSettings}
            />
          </React.Suspense>
        )}
        {rolePlaySetupOpen && (
          <React.Suspense fallback={null}>
            <LazyRolePlaySetupDrawer
              open={rolePlaySetupOpen}
              beforeState={rolePlayState}
              historyId={historyId}
              serverChatId={serverChatId}
              characterId={
                selectedCharacter?.id ??
                (rolePlayIdentity?.kind === "character"
                  ? rolePlayIdentity.id
                  : null)
              }
              currentSystemPrompt={systemPrompt}
              ragPinnedResultIds={ragPinnedResults.map((result) =>
                String(result.id)
              )}
              savedRolePlaySetups={startupTemplates}
              savedSetupDraftName={startupTemplateDraftName}
              savedSetupNameFallback={startupTemplateNameFallback}
              onClose={() => setRolePlaySetupOpen(false)}
              onApply={handleApplyRolePlaySetup}
              onSavedSetupDraftNameChange={setStartupTemplateDraftName}
              onSaveRolePlaySetup={handleSaveRolePlaySetup}
              onPreviewSavedSetup={handleOpenStartupTemplatePreview}
              onApplySavedSetup={handleApplySavedRolePlaySetup}
              onRenameSavedSetup={handleRenameStartupTemplate}
              onDeleteSavedSetup={handleDeleteStartupTemplate}
            />
          </React.Suspense>
        )}
        {documentGeneratorOpen && (
          <React.Suspense fallback={null}>
            <LazyDocumentGeneratorDrawer
              open={documentGeneratorOpen}
              onClose={() => {
                setDocumentGeneratorOpen(false);
                setDocumentGeneratorSeed({});
              }}
              conversationId={
                documentGeneratorSeed?.conversationId ?? serverChatId ?? null
              }
              defaultModel={selectedModel || null}
              seedMessage={documentGeneratorSeed?.message ?? null}
              seedMessageId={documentGeneratorSeed?.messageId ?? null}
            />
          </React.Suspense>
        )}
        {voiceChatEnabled && voiceChat.state !== "idle" && (
          <VoiceChatIndicator
            state={voiceChat.state}
            statusLabel={voiceChatStatusLabel}
            onStop={handleVoiceChatToggle}
          />
        )}
        {voiceModeSelectorOpen && (
          <React.Suspense fallback={null}>
            <LazyVoiceModeSelector
              open={voiceModeSelectorOpen}
              onClose={() => setVoiceModeSelectorOpen(false)}
              onSelectDictation={handleDictationToggle}
              onSelectConversation={handleVoiceChatToggle}
              dictationAvailable={speechAvailable}
              conversationAvailable={voiceChatAvailable}
            />
          </React.Suspense>
        )}
      </div>
    </React.Profiler>
  );
};
