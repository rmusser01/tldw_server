import { useModelSettingsOverlayState } from "@/hooks/useModelSettingsOverlayState"
import { shallow } from "zustand/shallow";
import { SelectedModelSaveFeedback } from "@/components/Common/SelectedModelSaveFeedback";
import { useExplicitStopFocus } from "@/hooks/useExplicitStopFocus"
import { captureChatDraftSource, hasTemporaryChatContent } from "@/hooks/chat/chat-draft-transition";
import { useChatModelCatalog } from "@/hooks/useChatModelCatalog"
import { isSavedRoleplayConversation } from "@/hooks/chat/roleplay-resume-source"
import { ChatGreetingPicker } from "@/components/Common/ChatGreetingPicker";
import { useCharacterGreetingStart } from "@/hooks/useCharacterGreetingStart";
import { ChatDraftRecovery, type RestoreChatDraft } from "@/components/Chat/composer/ChatDraftRecovery";
import { watchChatAccountChanges } from "@/services/chat-account-boundary";
import { useDefaultCharacterSelection } from "@/hooks/useDefaultCharacterSelection";
import {
  ChatComposer,
  useComposerVariantPreference,
} from "@/components/Chat/composer/ChatComposer";
import { PromptAssistComposerAction } from "@/components/Chat/composer/PromptAssistComposerAction";
import { useComposerEnabledPreference } from "@/components/Chat/composer/hooks/useComposerEnabledPreference";
import { AudioSourcePicker } from "@/components/Common/AudioSourcePicker";
import { BetaTag } from "@/components/Common/Beta";
// getImageBackendConfigs, normalizeImageBackendConfig, resolveImageBackendConfig moved to usePlaygroundImageGen
import { CharacterSelect } from "@/components/Common/CharacterSelect";
import { ChatQueuePanel } from "@/components/Common/ChatQueuePanel";
import { BuddyManagementButton } from "@/components/Common/PersonaBuddy/BuddyManagementButton";
import { type KnowledgeTab } from "@/components/Knowledge";
import type { SlashCommandItem } from "@/components/Sidepanel/Chat/SlashCommandMenu";
import { isFirefoxTarget } from "@/config/platform";
import { STT_DEFAULTS } from "@/config/ui-constants";
import { getDesignSystemState } from "@/design-system";
import { generateID } from "@/db/dexie/helpers";
import { usePromptInventory } from "@/hooks/usePromptInventory";
import type {
  DocumentProcessingTurnMetadata,
  UploadedFile,
} from "@/db/dexie/types";
import { useChatSettingsRecord } from "@/hooks/chat/useChatSettingsRecord";
import { useHistorySelectionContext } from "@/hooks/chat/useHistorySelection";
import { validateNativeCharacterConfiguration, nativeCharacterPreflightMessage } from "@/hooks/chat/native-character-preflight";
import { useMcpToolsStore } from "@/store/mcp-tools";
import {
  getActorSceneForChat,
  type ActorSceneRead
} from "@/services/actor-settings"
import { captureCurrentActorSceneTarget } from "@/hooks/chat/actor-scene-source"
import {
  captureActorSettingsTarget,
  type ActorSettingsTarget
} from "@/utils/actor-settings-target"
import { chatDraftOwnerKey } from "@/utils/chat-draft-owner"
import { freshChatDraftAddress } from "@/utils/chat-draft-address"
import { loadServicePromptSnapshot } from "@/services/service-prompts";
import { requestSettingsNavigation } from "@/utils/settings-return";
import { dispatchChatRouteReplacement } from "@/utils/character-chat-mode-intent";
import { buildChatModelUsability } from "@/utils/chat-model-availability";
import {
  captureModelSettingsOperation,
  type ModelSettingsOperation
} from "@/hooks/chat/model-settings-operation"
import { resolveModelSelectorSelection } from "@/hooks/playground/modelSelectorUtils"
import { resolveExplicitProviderForSelectedModel } from "@/utils/resolve-api-provider"
import { PLAYGROUND_APPEND_FORMATTING_GUIDE_PROMPT_STORAGE_KEY, resolveOutputFormattingGuideSuffix } from "@/utils/output-formatting-guide";
import { resolveMessageSteering } from "@/utils/message-steering";
import type { AssistantSetupApplyOptions } from "./hooks/usePromptTemplates";
import { captureSelectedModelGuard, createSelectedModelGuard } from "@/hooks/chat/useSelectedModel";
import type {
  RolePlayApplyContext,
  RolePlayApplyGuard
} from "./RolePlaySetupDrawer"
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
import { usePlaygroundSessionStore } from "@/store/playground-session";
import { useHomeMilestoneScope } from "@/hooks/useHomeMilestoneScope";
import { useMilestoneStore } from "@/store/milestones";
import { useCanonicalConnectionConfig } from "@/hooks/useCanonicalConnectionConfig";
import { usePromptLocalAuthority } from "@/hooks/usePromptLocalAuthority";
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
import { hasAcknowledgedServerChatMirror, serverChatMirrorOwnerKey } from "@/db/dexie/server-chat-mirror";
import { getSelectedAssistantOperationRevision, useSelectedAssistant } from "@/hooks/useSelectedAssistant";
import { useServerCapabilities } from "@/hooks/useServerCapabilities";
import { useTldwAudioStatus } from "@/hooks/useTldwAudioStatus";
import { useVoiceChatMessages } from "@/hooks/useVoiceChatMessages";
import { useVoiceChatSettings } from "@/hooks/useVoiceChatSettings";
import { useVoiceChatStream } from "@/hooks/useVoiceChatStream";
// useQueuedRequests moved to usePlaygroundQueueManagement
import type { ChatDocuments } from "@/models/ChatTypes";
import { clearSetting, getSetting } from "@/services/settings/registry";
import {
  buildChatSurfaceScopeKeyFromConfig,
  derivePromptAssistAuthorizationRevision,
} from "@/services/chat-surface-scope";
import {
  DISCUSS_WATCHLIST_PROMPT_SETTING,
} from "@/services/settings/ui-settings";
import {
  isChatSubmitSuccess,
  normalizeChatSubmitResult,
} from "@/hooks/chat/chat-action-utils";
import { fetchImageModels } from "@/services/tldw-server";
import {
  type ResearchRunCreateRequest,
  type ResearchRunFollowUpBackground,
  tldwClient,
} from "@/services/tldw/TldwApiClient";
// ChatRequestDebugSnapshot moved to usePlaygroundRawPreview
import {
  buildDiscussMediaHint,
  getMediaChatHandoffMode,
  parseMediaIdAsNumber,
  consumeMediaChatHandoff,
  removeMediaChatHandoff,
  MEDIA_CHAT_HANDOFF_PARAM,
  type MediaChatHandoffPayload,
} from "@/services/tldw/media-chat-handoff";
import {
  normalizeVoiceConversationRuntimeError,
  isVoiceConversationAuthReady,
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
  isProtectedTrackedPersonaUnavailable,
  getAssistantSelectionMode,
  type AssistantSelection,
} from "@/types/assistant-selection";
import { ConnectionPhase, deriveConnectionUxState } from "@/types/connection";
import { PASTED_TEXT_CHAR_LIMIT } from "@/utils/constant";
import {
  normalizeFocusSelector,
  focusFirstVisibleElement,
  scheduleFocusFirstVisibleElement,
} from "@/utils/focus-return";
// resolveApiProviderForModel moved to usePlaygroundRawPreview and usePlaygroundImageGen
import {
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
import { QuickPromptInsertionReview } from "@/components/Chat/composer/QuickPromptInsertionReview";
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
  LayoutTemplate,
  X,
} from "lucide-react";
import React from "react";
import { useTranslation } from "react-i18next";
import { Link, useLocation, useNavigate } from "react-router-dom";

import { useStorage } from "@plasmohq/storage/hook";

import { useFocusShortcuts } from "~/hooks/keyboard";
import { isFreshChatSelectionPending } from "~/hooks/chat/fresh-chat-selection";
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
import { DocumentProcessingChoices } from "./DocumentProcessingChoices";
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
import { openModelSelector } from "./playground-cockpit-actions";
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
import { blockQueuedRequest } from "@/utils/chat-request-queue";
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

export type ApplyCharacterRoute = (characterId: string, isCurrent: () => boolean) => Promise<"applied" | "missing" | "load-error" | "failed" | "retired">;

type Props = {
  droppedFiles: File[];
  onCharacterRouteApplyReady?: (apply: ApplyCharacterRoute | null) => void;
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
  forceWideMode?: boolean;
  onComposerLayoutChange?: (metrics: ComposerDockLayoutMetrics) => void;
  onDraftPresenceChange?: (hasDraft: boolean) => void;
  connectionSetupInTimeline?: boolean;
  composerMessageCount?: number;
  composerMessageCountLabel?: string | null;
  characterWorkflowActive?: boolean;
  characterChatSendBlocker?: PlaygroundSendBlocker | null;
  characterChatModelUsability?: ChatModelUsability | null;
  characterChatModelUsabilityLabel?: string | null;
  characterChatModelUsabilityTitle?: string | null;
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
  onCharacterRouteApplyReady,
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
  forceWideMode = false,
  onComposerLayoutChange,
  onDraftPresenceChange,
  connectionSetupInTimeline = false,
  composerMessageCount = 0,
  composerMessageCountLabel = null,
  characterWorkflowActive = false,
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
  const location = useLocation();
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
  const effectiveWideMode = checkWideMode || forceWideMode;
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
    onSubmit: submitMessageAction,
    regenerateLastMessage,
    messages,
    setMessages,
    selectedModel,
    selectedModelIsLoading,
    selectedModelSaveFailed,
    retrySelectedModelSave,
    selectedAssistant: resolvedAssistantSelection,
    effectiveAssistantState,
    setSelectedModel,
    beginSelectedModelSave,
    chatMode,
    setChatMode,
    compareMode,
    setCompareMode,
    compareFeatureEnabled,
    compareFeatureReady,
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
    setContextFiles,
    documentContext,
    selectedKnowledge,
    ragMediaIds,
    chatLoopState = {
      status: "idle",
      pendingApprovals: [],
      inflightToolCallIds: [],
    },
  } = useMessageOption();
  const greetingSubmitGateRef = React.useRef<() => boolean>(() => false);
  const personaSubmitDisabledReasonRef = React.useRef<string | null>(null);
  const characterSubmitBlockerRef = React.useRef(characterChatSendBlocker);
  characterSubmitBlockerRef.current = characterChatSendBlocker;
  React.useLayoutEffect(() => {
    if (!characterChatSendBlocker?.active || !queuedMessages.some(item => item.status === "queued")) return;
    setQueuedMessages(previous => previous.map(item => item.status === "queued"
      ? blockQueuedRequest(item, "chat_setup_changed") : item));
  }, [characterChatSendBlocker?.active, queuedMessages, setQueuedMessages]);
  const getSubmitDisabledReason = React.useCallback(() => personaSubmitDisabledReasonRef.current ?? (characterSubmitBlockerRef.current?.active
    ? characterSubmitBlockerRef.current.title
    : greetingSubmitGateRef.current() ? t("sidepanel:greetingPicker.startingHelp", "Saving the character’s opening. Your message has not been sent.") : null), [t]);
  const onSubmit = React.useCallback((input: Parameters<typeof submitMessageAction>[0]) => {
    const reason = getSubmitDisabledReason();
    return reason ? Promise.resolve({ status: "skipped" as const, reason }) : submitMessageAction(input);
  }, [getSubmitDisabledReason, submitMessageAction]);
  const setUploadedFiles = useStoreMessageOption((s) => s.setUploadedFiles);
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
  const mobileComposerViewport = useMobileComposerViewport(isMobileViewport);
  const [viewportHeightPx, setViewportHeightPx] = React.useState(() =>
    typeof window === "undefined" ? 0 : window.innerHeight,
  );
  const modelSettingsDialog = useModelSettingsOverlayState();
  const { open: openModelSettings, setOpen: setModelSettingsOpen, afterClose: finishModelSettingsClose, isCurrent: modelSettingsCloseIsCurrent } = modelSettingsDialog;
  const [modelSettingsScopeOverride, setModelSettingsScopeOverride] =
    React.useState<string | null>(null);
  const modelSettingsReturnFocusRef = React.useRef<{ selector: string; isCurrent: () => boolean } | null>(null);
  const captureModelSettingsFocusSourceRef = React.useRef<(() => (() => boolean)) | null>(null);
  const setOpenModelSettings = React.useCallback<React.Dispatch<React.SetStateAction<boolean>>>((requested) => {
    setModelSettingsOpen(current => {
      const nextOpen = typeof requested === "function" ? requested(current) : requested;
      if (nextOpen) {
        setModelSettingsScopeOverride(null);
        modelSettingsReturnFocusRef.current = null;
      }
      return nextOpen;
    });
  }, [setModelSettingsOpen]);
  const modelSelectorReturnFocusSelectorRef = React.useRef<string | null>(null);
  const modelSelectorFocusFrameRef = React.useRef<number | null>(null);
  const mcpSettingsReturnFocusSelectorRef = React.useRef<string | null>(null);
  const [openActorSettings, setOpenActorSettings] = React.useState(false);
  const [rolePlaySetupOpen, setRolePlaySetupOpen] = React.useState(false);
  const rolePlaySetupReturnFocusRef = React.useRef<HTMLElement | null>(null);
  const openRolePlaySetup = React.useCallback(() => {
    const active = document.activeElement;
    if (
      active instanceof HTMLElement &&
      active !== document.body &&
      !active.closest('[role="dialog"]')
    ) {
      rolePlaySetupReturnFocusRef.current = active;
    } else if (!rolePlaySetupReturnFocusRef.current?.isConnected) {
      rolePlaySetupReturnFocusRef.current =
        document.querySelector<HTMLElement>(
          '[data-testid="composer-role-play-setup"]',
        ) ??
        document.querySelector<HTMLElement>('[data-testid="composer-casual-advanced-chip"]') ??
        textareaRef.current;
    }
    setRolePlaySetupOpen(true);
  }, [textareaRef]);
  const [noticesExpanded, setNoticesExpanded] = React.useState(false);
  const actorSettings = useActorStore((state) => state.settings);
  const applyActorScene = useActorStore((state) => state.applyScene);
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
  }), shallow);
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
  const { data: promptLibrary = [] } = usePromptInventory("playgroundStartupPromptLibrary");
  const startupPromptAuthority = usePromptLocalAuthority();
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
  const homeScope = useHomeMilestoneScope();
  const homeScopeRef = React.useRef(homeScope);
  homeScopeRef.current = homeScope;
  const promptAssistBackendKey = React.useMemo(
    () =>
      canonicalConnectionLoading || !canonicalConnectionConfig
        ? null
        : buildChatSurfaceScopeKeyFromConfig(canonicalConnectionConfig),
    [canonicalConnectionConfig, canonicalConnectionLoading],
  );
  const promptAssistAuthorizationRevision = React.useMemo(
    () =>
      canonicalConnectionLoading || !canonicalConnectionConfig
        ? null
        : derivePromptAssistAuthorizationRevision(canonicalConnectionConfig),
    [canonicalConnectionConfig, canonicalConnectionLoading],
  );
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
  const isConnectionReady = isConnected && phase === ConnectionPhase.CONNECTED;
  const needsConnectionSetup = ["unconfigured", "configuring_url", "configuring_auth"].includes(connectionUxState);
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
        authReady: isVoiceConversationAuthReady({
          serverUrl: canonicalConnectionConfig?.serverUrl,
          authMode: canonicalConnectionConfig?.authMode,
          authSource: canonicalConnectionConfig?.authSource,
          apiKey: canonicalConnectionConfig?.apiKey,
          accessToken: canonicalConnectionConfig?.accessToken,
        }),
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
      canonicalConnectionConfig?.authSource,
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
  const [documentGeneratorOpen, setDocumentGeneratorOpen] =
    React.useState(false);
  const [voiceModeSelectorOpen, setVoiceModeSelectorOpen] =
    React.useState(false);
  const [modeLauncherOpen, setModeLauncherOpen] = React.useState(false);
  const [openUIRequestMode, setOpenUIRequestMode] = React.useState(false);
  const openUIRequestRevisionRef = React.useRef(0);
  const openUIRequestRevision = openUIRequestRevisionRef.current;
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
  const [sttModel] = useStorage("sttModel", STT_DEFAULTS.MODEL);
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
  const [selectedAssistant, setSelectedAssistant, assistantMeta] = useSelectedAssistant(null);
  const selectedAssistantMode = React.useMemo(
    () => getAssistantSelectionMode(selectedAssistant),
    [selectedAssistant]
  );
  const storedCharacter = React.useMemo(
    () => assistantSelectionToCharacter<Character>(selectedAssistant),
    [selectedAssistant]
  );
  const activeAssistantSelection = effectiveAssistantState?.mode === "plain"
    ? null
    : resolvedAssistantSelection;
  const selectedCharacter = React.useMemo(
    () => assistantSelectionToCharacter<Character>(activeAssistantSelection),
    [activeAssistantSelection]
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
  const [defaultCharacter, setDefaultCharacter, defaultCharacterMeta] = useDefaultCharacterSelection();
  const defaultCharacterPreference = defaultCharacterMeta.preference;
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
  const suppressComposerToolbarForMobileCockpit =
    mobileCockpitModeActive && isMobileViewport && isProMode;

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
    () => resolveCharacterSelectionId(storedCharacter),
    [storedCharacter],
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
    if (assistantMeta?.isLoading || defaultCharacterMeta.isLoading || !effectiveDefaultCharacter || !effectiveDefaultCharacterId) return;
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
    assistantMeta?.isLoading,
    defaultCharacterMeta.isLoading,
    effectiveDefaultCharacter,
    effectiveDefaultCharacterId,
    isFreshChat,
    setSelectedCharacter,
    storedCharacterId,
  ]);

  const restoreModelSettingsFocus = React.useCallback(() => {
    const target = modelSettingsReturnFocusRef.current;
    if (!target) return;
    modelSettingsReturnFocusRef.current = null;
    const closeIsCurrent = modelSettingsCloseIsCurrent;
    scheduleFocusFirstVisibleElement(target.selector, undefined, () =>
      closeIsCurrent() && target.isCurrent());
  }, [modelSettingsCloseIsCurrent]);

  const completeModelSettingsClose = React.useCallback(() => {
    if (!finishModelSettingsClose()) return;
    setModelSettingsScopeOverride(null);
    restoreModelSettingsFocus();
  }, [finishModelSettingsClose, restoreModelSettingsFocus]);

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
      const selector = normalizeFocusSelector(detail?.returnFocusSelector);
      const isCurrent = captureModelSettingsFocusSourceRef.current?.();
      modelSettingsReturnFocusRef.current = selector && isCurrent ? { selector, isCurrent } : null;
      setModelSettingsScopeOverride(
        typeof detail?.settingsScope === "string" && detail.settingsScope.trim()
          ? detail.settingsScope.trim()
          : null,
      );
      setModelSettingsOpen(true);
    };
    window.addEventListener(OPEN_MODEL_SETTINGS_EVENT, handler);
    return () => {
      window.removeEventListener(OPEN_MODEL_SETTINGS_EVENT, handler);
    };
  }, [setModelSettingsOpen]);

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

  const { data: composerModels, isFetching: composerModelsFetching, isError: composerModelsFailed, refetch: retryComposerModels } = useChatModelCatalog(isConnectionReady);
  // True while the catalog is actively being fetched and none have arrived yet.
  // Used to show a loading affordance instead of a "connect your server" error.
  const composerModelsLoading =
    isConnectionReady &&
    composerModelsFetching &&
    ((composerModels as any[] | undefined)?.length ?? 0) === 0;
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
    selectedModelSettingsScope,
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
    modelsLoading: composerModelsLoading,
    modelsError: composerModelsFailed,
    onRetryModels: () => { void retryComposerModels() },
  });
  const cancelModelSelectorFocus = React.useCallback(() => {
    if (modelSelectorFocusFrameRef.current == null) return;
    window.cancelAnimationFrame(modelSelectorFocusFrameRef.current);
    modelSelectorFocusFrameRef.current = null;
  }, []);
  React.useEffect(() => cancelModelSelectorFocus, [cancelModelSelectorFocus]);
  const restoreModelSelectorFocus = React.useCallback((restoreFocus = true) => {
    const returnFocusSelector = modelSelectorReturnFocusSelectorRef.current;
    if (!returnFocusSelector) return;
    modelSelectorReturnFocusSelectorRef.current = null;
    cancelModelSelectorFocus();
    if (!restoreFocus) return;
    const closingFocus = document.activeElement;
    if (closingFocus !== document.body &&
      !(closingFocus instanceof HTMLElement && closingFocus.matches("[data-testid='model-selector']"))) return;
    modelSelectorFocusFrameRef.current = window.requestAnimationFrame(() => {
      modelSelectorFocusFrameRef.current = null;
      if (document.activeElement !== closingFocus && document.activeElement !== document.body) return;
      focusFirstVisibleElement(returnFocusSelector);
    });
  }, [cancelModelSelectorFocus]);
  const setModelDropdownOpenWithFocusRestore = React.useCallback(
    (nextOpen: boolean, restoreFocus = true) => {
      if (nextOpen) cancelModelSelectorFocus();
      setModelDropdownOpen(nextOpen);
      if (!nextOpen) restoreModelSelectorFocus(restoreFocus);
    },
    [cancelModelSelectorFocus, restoreModelSelectorFocus, setModelDropdownOpen],
  );
  React.useEffect(() => {
    setActiveChatModelSettingsScope(selectedModelSettingsScope ?? null);
  }, [selectedModelSettingsScope, setActiveChatModelSettingsScope]);

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
  // Priority: available favorite, verified server default, then available fallback.
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
      String(systemPrompt || "").trim().length > 0,
    [selectedSystemPrompt, systemPrompt],
  );

  const modelComparison = useModelComparison({
    composerModels,
    selectedModel,
    setSelectedModel,
    compareFeatureEnabled,
    compareFeatureReady,
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

  const assistantActionMountedRef = React.useRef(true);
  const composerImageRef = React.useRef("");
  const assistantComposerRef = React.useRef<{
    getRevision: () => number
    getSetupInputRevision: () => number
    retainCurrentDraft: () => Promise<boolean>
    draftReady: boolean
  } | null>(null)
  const assistantConnectionRevisionRef = React.useRef(0);
  React.useLayoutEffect(() => { assistantConnectionRevisionRef.current++; }, [phase, isConnected, canonicalConnectionConfig, canonicalConnectionLoading]);
  const [appendFormattingGuidePrompt] = useStorage(PLAYGROUND_APPEND_FORMATTING_GUIDE_PROMPT_STORAGE_KEY, false);
  const assistantPreparationViewRef = React.useRef({ researchContext, openUIRequestMode, appendFormattingGuidePrompt, isConnectionReady, composerModels, composerModelsLoading });
  React.useLayoutEffect(() => {
    assistantPreparationViewRef.current = { researchContext, openUIRequestMode, appendFormattingGuidePrompt, isConnectionReady, composerModels, composerModelsLoading };
  }, [researchContext, openUIRequestMode, appendFormattingGuidePrompt, isConnectionReady, composerModels, composerModelsLoading]);
  const historySelection = useHistorySelectionContext();
  const personaHistoryUnavailable = isProtectedTrackedPersonaUnavailable(Boolean(historySelection),
    effectiveAssistantState?.mode === "tracked_persona" ? "persona" : null, "tracked");
  const personaSubmitDisabledReason = personaHistoryUnavailable
    ? t("option:assistant.trackedPersonaUnavailable", "Tracked persona chat is unavailable here. Use a persona prompt overlay in a regular chat, or choose a character for tracked chat.") : null;
  personaSubmitDisabledReasonRef.current = personaSubmitDisabledReason;
  const captureSelectionFence = historySelection?.fence;
  const resetSelectedHistory = historySelection?.reset;
  const assistantOwnerIsCurrent = assistantMeta?.isCurrent;
  captureModelSettingsFocusSourceRef.current = () => {
    const historyIsCurrent = captureSelectionFence?.() ?? (() => true);
    const owner = historySelection?.getCurrent().owner;
    const modelRevision = useStoreMessageOption.getState().selectedModelRevision;
    return captureChatDraftSource(() => assistantActionMountedRef.current &&
      historyIsCurrent() && (assistantOwnerIsCurrent?.() ?? true) &&
      (owner?.kind !== "native" || owner.validate_lease()) &&
      useStoreMessageOption.getState().selectedModelRevision === modelRevision).isCurrent;
  }
  const assistantActionRevisionRef = React.useRef(0);
  React.useLayoutEffect(() => {
    if (characterChatSendBlocker?.active) assistantActionRevisionRef.current++;
  }, [characterChatSendBlocker?.active]);
  React.useEffect(() => {
    assistantActionMountedRef.current = true;
    return () => { assistantActionMountedRef.current = false; };
  }, []);
  const captureAssistantActionGuard =
    React.useCallback((): RolePlayApplyGuard => {
      const selectionIsCurrent = captureSelectionFence?.() ?? (() => true)
      let actionRevision = assistantActionRevisionRef.current
      const address = useStoreMessageOption.getState().chatDraftAddress
      const source = captureChatDraftSource(
        () =>
          assistantActionMountedRef.current &&
          selectionIsCurrent() &&
          (assistantOwnerIsCurrent?.() ?? true),
        (current, original) => address?.kind === "native" || current === original
      )
      return Object.assign(
        () =>
          assistantActionRevisionRef.current === actionRevision &&
          source.isCurrent(),
        {
          acceptNextAction: (revision: number) => {
            if (
              !Number.isSafeInteger(revision) ||
              revision !== actionRevision + 1 ||
              assistantActionRevisionRef.current !== revision ||
              !source.isCurrent()
            )
              return false
            actionRevision = revision
            return true
          },
          acceptRestoreCancelled: (revision: number) =>
            assistantActionRevisionRef.current === actionRevision &&
            source.acceptRestoreCancelled(revision)
        }
      )
    }, [captureSelectionFence, assistantOwnerIsCurrent])
  const captureAssistantPreparationGuard = React.useCallback(() => {
    const sourceIsCurrent = captureAssistantActionGuard()
    const inputRevision = assistantComposerRef.current?.getRevision()
    const connectionRevision = assistantConnectionRevisionRef.current
    const inputsAreCurrent = () =>
      assistantComposerRef.current?.getRevision() === inputRevision &&
      assistantConnectionRevisionRef.current === connectionRevision
    return Object.assign(() => inputsAreCurrent() && sourceIsCurrent(), {
      acceptRestoreCancelled: (revision: number) =>
        inputsAreCurrent() &&
        sourceIsCurrent.acceptRestoreCancelled?.(revision) === true
    })
  }, [captureAssistantActionGuard])
  const readAssistantPreparationInputs = React.useCallback(() => {
    const chat = useStoreMessageOption.getState();
    const view = assistantPreparationViewRef.current;
    return [chat.selectedModel, chat.toolChoice, chat.temporaryChat, chat.history, chat.messages, chat.webSearch,
      chat.selectedSystemPrompt, chat.uploadedFiles, chat.contextFiles, chat.documentContext, chat.selectedKnowledge,
      chat.ragMediaIds, chat.ragPinnedResults, chat.replyTarget, chat.messageSteeringMode, chat.messageSteeringForceNarrate, chat.messageSteeringRevision,
      chat.isProcessing, chat.isLoading, chat.streaming, composerImageRef.current, view.researchContext, view.openUIRequestMode,
      view.appendFormattingGuidePrompt, view.isConnectionReady, view.composerModels, view.composerModelsLoading];
  }, []);
  const applyAssistantSelection = React.useCallback(
    async (
      selection: AssistantSelection | null,
      options: AssistantSetupApplyOptions = {}
    ): Promise<boolean> => {
      const settingsOperation =
        options.settingsOperation ?? captureModelSettingsOperation()
      if (options.isCurrent?.() === false || !settingsOperation.isCurrent())
        return false
      if (options.applyIdentity !== false && isProtectedTrackedPersonaUnavailable(
        Boolean(historySelection), selection?.kind, getAssistantSelectionMode(selection)
      )) throw new Error("native_history_persona_unsupported")
      const activeChat = useStoreMessageOption.getState()
      let modelIsCurrent = captureSelectedModelGuard()
      const toolsState = useMcpToolsStore.getState()
      const toolChoice = activeChat.toolChoice
      const configurationIsCurrent = () =>
        useStoreMessageOption.getState().toolChoice === toolChoice &&
        useMcpToolsStore.getState() === toolsState
      const publicationInputsAreCurrent = () =>
        modelIsCurrent() && configurationIsCurrent()
      const initialSceneTarget = captureCurrentActorSceneTarget(historySelection)
      const savedSettingsOnly = options.applyIdentity === false && initialSceneTarget?.address?.kind === "native"
      const mirrorOwnerKey = savedSettingsOnly && initialSceneTarget.account ? serverChatMirrorOwnerKey({ requestScope: initialSceneTarget.account.requestScope }) : null
      const sourceHistoryIdCurrent = (current: string | null, original: string | null) => current === original || Boolean(mirrorOwnerKey && initialSceneTarget?.address?.kind === "native" && hasAcknowledgedServerChatMirror({ historyId: current, ownerKey: mirrorOwnerKey, chatId: initialSceneTarget.address.id, scope: initialSceneTarget.address.scope }))
      const entrySource = captureChatDraftSource(undefined, sourceHistoryIdCurrent)
      const actionRevision = ++assistantActionRevisionRef.current
      if (
        options.onActionStarted?.(actionRevision) === false ||
        assistantActionRevisionRef.current !== actionRevision ||
        !entrySource.isCurrent() ||
        !publicationInputsAreCurrent() ||
        !settingsOperation.isCurrent() ||
        options.isCurrent?.() === false
      )
        return false
      const sourceAddress = activeChat.chatDraftAddress
      const sourceRevision = activeChat.chatDraftRevision
      const originalSceneTarget = savedSettingsOnly ? captureCurrentActorSceneTarget(historySelection, () => true, sourceHistoryIdCurrent) : initialSceneTarget
      const savedSettingsAreCurrent = () => !savedSettingsOnly || Boolean(
        originalSceneTarget?.address?.kind === "native" &&
        originalSceneTarget.isCurrent() &&
        historySelection?.settingsMode(activeChat.serverChatId,
          originalSceneTarget.address?.scope) !== "pending"
      )
      const draftAccount = historySelection?.draftAccount
      const originalAccount =
        draftAccount?.requestScope && draftAccount.isCurrent()
          ? {
              requestScope: structuredClone(draftAccount.requestScope),
              isCurrent: draftAccount.isCurrent
            }
          : !draftAccount
            ? originalSceneTarget?.account
            : undefined
      const explicitScene = options.sceneSettings
        ? structuredClone(options.sceneSettings)
        : undefined
      const inputRevision = assistantComposerRef.current?.getSetupInputRevision()
      const connectionRevision = assistantConnectionRevisionRef.current
      const sourceSession = usePlaygroundSessionStore.getState()
      const sourcePreparationIsCurrent = captureAssistantPreparationGuard()
      let sourceRestoreRevision = sourceSession.restoreRevision
      let routeReceiptRejected = false
      const sourceIsCurrent = () =>
        settingsOperation.isCurrent() &&
        savedSettingsAreCurrent() &&
        !routeReceiptRejected &&
        sourcePreparationIsCurrent() &&
        publicationInputsAreCurrent() &&
        originalAccount?.isCurrent() !== false
      let destinationIsCurrent = () =>
        settingsOperation.isCurrent() &&
        savedSettingsAreCurrent() &&
        !routeReceiptRejected &&
        sourcePreparationIsCurrent() &&
        configurationIsCurrent() &&
        originalAccount?.isCurrent() !== false
      const publicationIsCurrent = () =>
        destinationIsCurrent() && publicationInputsAreCurrent()
      let sceneTarget = originalSceneTarget
      let destinationHistoryId = activeChat.historyId
      let destinationServerChatId = activeChat.serverChatId
      const matchesActiveChat =
        selection?.kind === "character"
          ? activeChat.serverChatAssistantKind === "character" &&
            String(activeChat.serverChatCharacterId) === selection.id
          : selection?.kind === "persona"
            ? activeChat.serverChatAssistantKind === "persona" &&
              String(activeChat.serverChatAssistantId) === selection.id
            : activeChat.serverChatMetaLoaded &&
              (activeChat.serverChatAssistantKind === null ||
                activeChat.serverChatAssistantKind === undefined) &&
              (activeChat.serverChatAssistantId === null ||
                activeChat.serverChatAssistantId === undefined) &&
              (activeChat.serverChatCharacterId === null ||
                activeChat.serverChatCharacterId === undefined)
      const nativeCharacter = Boolean(
        !savedSettingsOnly && historySelection &&
          selection?.kind === "character" &&
          getAssistantSelectionMode(selection) !== "overlay"
      )
      const replacing = !savedSettingsOnly && (
        Boolean(
          nativeCharacter &&
            (options.freshCharacterRoute ||
              activeChat.temporaryChat ||
              historySelection?.getCurrent().owner?.kind === "local")
        ) ||
        (!matchesActiveChat &&
          getAssistantSelectionMode(selection) !== "overlay" &&
          Boolean(
            activeChat.serverChatId ||
              (nativeCharacter &&
                (activeChat.historyId || historySelection?.getCurrent().owner))
          )))
      let snapshot:
        | Awaited<ReturnType<typeof loadServicePromptSnapshot>>
        | undefined
      let preparationIsCurrent = () =>
        sourceIsCurrent() && options.isCurrent?.() !== false
      let preparedModelScope: string | null = null
      const proposedModel = options.model || activeChat.selectedModel || ""
      const replaceRoute = (
        detail: Parameters<typeof dispatchChatRouteReplacement>[0]
      ) => {
        let receiving = true
        let accepted = false
        try {
          dispatchChatRouteReplacement({
            ...detail,
            onRestoreCancelled: (revision) => {
              if (!receiving || accepted) return false
              if (
                revision !== sourceRestoreRevision + 1 ||
                !sourcePreparationIsCurrent.acceptRestoreCancelled(revision) ||
                options.onRestoreCancelled?.(revision) === false
              ) {
                routeReceiptRejected = true
                return false
              }
              sourceRestoreRevision = revision
              accepted = true
              if (!preparationIsCurrent()) {
                routeReceiptRejected = true
                return false
              }
              return true
            }
          })
        } finally {
          receiving = false
        }
      }
      try {
        if (nativeCharacter) {
          if (
            activeChat.isProcessing ||
            activeChat.isLoading ||
            activeChat.streaming
          )
            throw new Error("native_history_response_pending")
          const settingsState = useStoreChatModelSettings.getState()
          const capturedInputs = readAssistantPreparationInputs()
          const sourceAssistantRevision = getSelectedAssistantOperationRevision()
          let writingIdentity = false
          preparationIsCurrent = () =>
            sourceIsCurrent() &&
            options.isCurrent?.() !== false &&
            (getSelectedAssistantOperationRevision() ===
              sourceAssistantRevision ||
              (writingIdentity &&
                getSelectedAssistantOperationRevision() ===
                  sourceAssistantRevision + 1)) &&
            !snapshot?.scopeSignal.aborted &&
            !snapshot?.scopeInvalidatedSignal.aborted &&
            useStoreChatModelSettings.getState() === settingsState &&
            useMcpToolsStore.getState() === toolsState &&
            readAssistantPreparationInputs().every(
              (value, index) => value === capturedInputs[index]
            )
          const proposedProvider = resolveExplicitProviderForSelectedModel({
            currentSelectedModel: activeChat.selectedModel,
            requestedSelectedModel: proposedModel,
            explicitProvider:
              options.settings?.apiProvider ?? settingsState.apiProvider
          })
          preparedModelScope = resolveModelSelectorSelection(
            composerModels,
            proposedModel,
            proposedProvider,
            {
              scope: settingsState.activeSettingsScope,
              providerOverride: settingsState.activeSettingsScope
                ? settingsState.scopedSettingsByModelKey[
                    settingsState.activeSettingsScope
                  ]?.apiProvider
                : undefined
            }
          ).settingsScope
          const proposedSettings = settingsState.getEffectiveSettings(
            preparedModelScope,
            options.settings
          )
          if (!explicitScene && !replacing && !originalSceneTarget)
            throw new Error("actor_settings_unavailable")
          let actor = explicitScene ?? createDefaultActorSettings()
          if (!explicitScene && !replacing) {
            try {
              actor = (
                await getActorSceneForChat({
                  target: originalSceneTarget!,
                  validate_lease: preparationIsCurrent
                })
              ).settings
            } catch (error) {
              if (!preparationIsCurrent()) return false
              throw error
            }
          }
          if (!preparationIsCurrent()) return false
          const steering = resolveMessageSteering({
            mode: activeChat.messageSteeringMode,
            forceNarrate: activeChat.messageSteeringForceNarrate
          })
          const prepared = validateNativeCharacterConfiguration({
            temporary:
              Boolean(
                activeChat.temporaryChat &&
                  (activeChat.history.length || activeChat.messages.length)
              ) || activeChat.historyId === "temp",
            image: composerImageRef.current,
            unsupportedContext: Boolean(
              activeChat.webSearch ||
                (options.selectedSystemPrompt !== undefined
                  ? options.selectedSystemPrompt
                  : activeChat.selectedSystemPrompt) ||
                activeChat.uploadedFiles?.length ||
                activeChat.contextFiles?.length ||
                activeChat.documentContext?.length ||
                activeChat.selectedKnowledge ||
                activeChat.ragMediaIds?.length ||
                (options.hasExplicitContext ??
                  Boolean(activeChat.ragPinnedResults?.length)) ||
                activeChat.replyTarget ||
                steering.continueAsUser ||
                steering.impersonateUser ||
                steering.forceNarrate ||
                researchContext ||
                openUIRequestMode ||
                resolveOutputFormattingGuideSuffix(
                  Boolean(appendFormattingGuidePrompt)
                )
            ),
            toolChoice: activeChat.toolChoice,
            clientToolCount: toolsState.tools.length,
            actorEnabled: actor?.isEnabled,
            model: proposedModel,
            currentModel: activeChat.selectedModel,
            settings: proposedSettings
          })
          if (!isConnectionReady)
            throw new Error("native_history_server_unavailable")
          const usability = buildChatModelUsability({
            selectedModel: `${prepared.provider}:${prepared.model}`,
            availableModels: composerModels,
            modelsLoading: composerModelsLoading,
            isServerConnected: isConnectionReady
          })
          if (!usability.canSend)
            throw new Error("native_history_model_unavailable")
          const owner = historySelection?.getCurrent().owner
          snapshot = await loadServicePromptSnapshot([], {
            requestScope:
              originalAccount?.requestScope ??
              (!replacing && owner?.kind === "native"
                ? owner.request_scope
                : undefined)
          })
          if (!preparationIsCurrent()) return false
          if (!(await assistantComposerRef.current?.retainCurrentDraft()))
            throw new Error("native_history_draft_not_retained")
          if (
            !preparationIsCurrent() ||
            (replacing && !requestSettingsNavigation("/chat"))
          )
            return false
          if (!preparationIsCurrent()) return false
          writingIdentity = options.applyIdentity !== false
          // A failed identity write must not release the acknowledged source owner.
          if (
            options.applyIdentity !== false &&
            !(await setSelectedAssistant(selection, {
              isCurrent: preparationIsCurrent
            }))
          )
            return false
          if (!preparationIsCurrent()) return false
        }
        if (replacing) {
          if (!preparationIsCurrent()) return false
          if (!nativeCharacter || !options.freshCharacterRoute)
            replaceRoute({
              serverChatId: activeChat.serverChatId,
              historyId: activeChat.historyId,
              restoreRevision:
                usePlaygroundSessionStore.getState().restoreRevision,
              characterId:
                selection?.kind === "character" ? String(selection.id) : null
            })
          if (
            !preparationIsCurrent() ||
            (nativeCharacter && !originalAccount?.isCurrent())
          )
            return false
          const address = freshChatDraftAddress(
            nativeCharacter ? false : activeChat.temporaryChat
          )
          const revision = sourceRevision + 1
          if (
            !activeChat.setChatDraftAddress(address, {
              sourceAddress,
              expectedRevision: sourceRevision
            })
          )
            return false
          let restoreRevision = sourceRestoreRevision
          let selectionIsCurrent = () => true
          destinationIsCurrent = () => {
            const current = useStoreMessageOption.getState()
            const session = usePlaygroundSessionStore.getState()
            return (
              assistantActionMountedRef.current &&
              settingsOperation.isCurrent() &&
              configurationIsCurrent() &&
              assistantActionRevisionRef.current === actionRevision &&
              (assistantOwnerIsCurrent?.() ?? true) &&
              originalAccount?.isCurrent() !== false &&
              selectionIsCurrent() &&
              assistantComposerRef.current?.getSetupInputRevision() ===
                inputRevision &&
              assistantConnectionRevisionRef.current === connectionRevision &&
              current.chatDraftAddress === address &&
              current.chatDraftRevision === revision &&
              current.historyId === destinationHistoryId &&
              current.serverChatId === destinationServerChatId &&
              session.restoreRevision === restoreRevision &&
              session.sourceSelectionRevision ===
                sourceSession.sourceSelectionRevision &&
              !snapshot?.scopeSignal.aborted &&
              !snapshot?.scopeInvalidatedSignal.aborted
            )
          }
          if (!publicationIsCurrent()) return false
          if (nativeCharacter)
            activeChat.setQueuedMessages((previous) =>
              previous.map((item) =>
                blockQueuedRequest(item, "chat_setup_changed")
              )
            )
          if (!publicationIsCurrent()) return false
          resetSelectedHistory?.()
          if (!publicationIsCurrent()) return false
          selectionIsCurrent = captureSelectionFence?.() ?? (() => true)
          restoreRevision++
          usePlaygroundSessionStore.getState().clearSession()
          if (!publicationIsCurrent()) return false
          destinationHistoryId = null
          activeChat.setHistoryId(null, { preserveServerChatId: false })
          if (!publicationIsCurrent()) return false
          destinationServerChatId = null
          activeChat.setServerChatId(null)
          if (!publicationIsCurrent()) return false
          for (const publish of [
            () => activeChat.setHistory([]),
            () => activeChat.setMessages([]),
            () => activeChat.setServerChatCharacterId(null),
            () => activeChat.setServerChatAssistantKind(null),
            () => activeChat.setServerChatAssistantId(null),
            () => activeChat.setServerChatPersonaMemoryMode(null),
            () => activeChat.setServerChatMetaLoaded(false)
          ]) {
            if (!publicationIsCurrent()) return false
            publish()
          }
          if (!publicationIsCurrent()) return false
          sceneTarget = captureActorSettingsTarget({
            address,
            requestScope: originalAccount?.requestScope,
            accountIsCurrent: originalAccount?.isCurrent,
            isCurrent: () => {
              const current = useStoreMessageOption.getState()
              return (
                current.chatDraftAddress === address &&
                current.chatDraftRevision === revision &&
                current.historyId === null &&
                current.serverChatId === null
              )
            }
          })
        }
        if (nativeCharacter && !replacing && !options.freshCharacterRoute) {
          const target = useStoreMessageOption.getState()
          const owner = historySelection?.getCurrent().owner
          replaceRoute({
            serverChatId: target.serverChatId,
            historyId: target.historyId,
            restoreRevision: usePlaygroundSessionStore.getState().restoreRevision,
            characterId: String(selection!.id),
            nextChatId:
              owner?.kind === "native" &&
              owner.conversation_id === target.serverChatId
                ? target.serverChatId
                : null
          })
        }
        if (!publicationIsCurrent()) return false
        if (nativeCharacter) {
          activeChat.setTemporaryChat(false)
          if (!publicationIsCurrent()) return false
          if (options.model) {
            const revision =
              useStoreMessageOption.getState().selectedModelRevision + 1
            modelIsCurrent = createSelectedModelGuard(options.model, revision)
            activeChat.setSelectedModel(options.model)
          }
          if (!publicationIsCurrent() || !modelIsCurrent()) return false
          if (
            !settingsOperation.write(() =>
              useStoreChatModelSettings
                .getState()
                .setActiveSettingsScope(preparedModelScope)
            )
          )
            return false
          if (!publicationIsCurrent() || !modelIsCurrent()) return false
        }
        if (!publicationIsCurrent() || !modelIsCurrent()) return false
        const actionIsCurrent = captureAssistantActionGuard()
        const isCurrent = () => destinationIsCurrent() && actionIsCurrent()
        options.onDestination?.({
          historyId: destinationHistoryId,
          serverChatId: destinationServerChatId,
          sceneTarget: sceneTarget ?? undefined,
          settingsOperation,
          isCurrent
        })
        if (
          !nativeCharacter &&
          options.applyIdentity !== false &&
          !(await setSelectedAssistant(selection, {
            isCurrent: () => isCurrent() && publicationInputsAreCurrent()
          }))
        )
          return false
        return (
          isCurrent() &&
          publicationInputsAreCurrent() &&
          options.isCurrent?.() !== false
        )
      } finally {
        snapshot?.release()
      }
    },
    [
      appendFormattingGuidePrompt,
      assistantOwnerIsCurrent,
      captureSelectionFence,
      captureAssistantActionGuard,
      captureAssistantPreparationGuard,
      composerModels,
      composerModelsLoading,
      historySelection,
      isConnectionReady,
      openUIRequestMode,
      readAssistantPreparationInputs,
      researchContext,
      resetSelectedHistory,
      setSelectedAssistant
    ]
  )
  const applyTrackedAssistantSelection = React.useCallback(async (selection: AssistantSelection, callerIsCurrent?: () => boolean) => {
    try { return await applyAssistantSelection(selection, { isCurrent: callerIsCurrent }); }
    catch (error) {
      notificationApi.error({ message: t("common:error", "Error"),
        description: nativeCharacterPreflightMessage(error, t) ?? t("playground:composer.rolePlayApplyError", "Settings could not be applied. Your draft is still here; retry Apply.") });
      return false;
    }
  }, [applyAssistantSelection, notificationApi, t]);

  const captureStartupTemplateSource = React.useCallback(() => {
    const selectionIsCurrent = captureSelectionFence?.() ?? (() => true)
    const inputRevision = assistantComposerRef.current?.getRevision()
    const inputEditRevision =
      assistantComposerRef.current?.getSetupInputRevision()
    const connectionRevision = assistantConnectionRevisionRef.current
    const source = captureChatDraftSource(
      () =>
        assistantActionMountedRef.current &&
        selectionIsCurrent() &&
        (assistantOwnerIsCurrent?.() ?? true) &&
        assistantComposerRef.current?.getRevision() === inputRevision &&
        assistantConnectionRevisionRef.current === connectionRevision
    )
    return {
      isCurrent: source.isCurrent,
      acceptRestoreCancelled: source.acceptRestoreCancelled,
      continuationIsCurrent: () =>
        assistantActionMountedRef.current &&
        (assistantOwnerIsCurrent?.() ?? true) &&
        assistantComposerRef.current?.getSetupInputRevision() ===
          inputEditRevision &&
        assistantConnectionRevisionRef.current === connectionRevision
    }
  }, [captureSelectionFence, assistantOwnerIsCurrent])

  const startupTemplateOwnerReady = startupPromptAuthority.ready || startupPromptAuthority.localOnlyReady;
  const promptTemplates = usePromptTemplates({
    promptOwnerId: startupPromptAuthority.ownerId,
    promptOwnerReady: startupTemplateOwnerReady,
    promptOwnerBoundaryKey: startupPromptAuthority.boundaryKey,
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
    composerModels,
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
  });
  const {
    currentPresetKey,
    currentPreset,
    startupTemplates,
    startupTemplateDraftName,
    setStartupTemplateDraftName,
    startupTemplatePreview,
    startupTemplatePreviewLease,
    setStartupTemplatePreview,
    startupTemplateNameFallback,
    selectedSystemPromptRecord,
    handleSaveStartupTemplate,
    handleOpenStartupTemplatePreview,
    applyStartupTemplateBundle,
    captureStartupTemplateGuard,
    handleDeleteStartupTemplate,
    handleTemplateSelect,
    promptSummaryLabel,
    handleSaveRolePlaySetup,
    handleApplySavedRolePlaySetup,
    handleSavedRolePlaySetupApplied,
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
    return t("playground:composer.serverStatusConnected", "Server connected");
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

  const msgCollapse = useMessageCollapse();
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
    messageRevision,
    promptAssistMutation,
    beginPromptAssistReset,
    restorePromptAssistAttempt,
    markPromptAssistAttemptSaved,
    promptAssistSavedAttemptId,
    captureDraftOwnership,
    typing,
    getSetupInputRevision,
    setMessageValue,
    restoreMessageValue,
    messageDisplayValue,
    attachTextarea,
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
    draftReady,
    draftOwnerVerificationState,
    retryDraftOwnerVerification,
    draftOwnerKey,
    isDraftOwnerCurrent,
    captureCompletionGuard,
    captureStopGuard,
    retainCurrentDraft,
    creationRecovery,
    reviewCreatedChat,
    copyCreationText,
    listRetainedDrafts,
    copyRetainedDraft,
  } = composerInput;
  const stopStreamingFromUi = useExplicitStopFocus(
    stopStreamingRequest,
    captureStopGuard
  )
  React.useLayoutEffect(() => {
    composerImageRef.current = form.values.image
    assistantComposerRef.current = {
      getRevision: form.getRevision,
      getSetupInputRevision: getSetupInputRevision ?? form.getRevision,
      retainCurrentDraft,
      draftReady
    }
  }, [
    form.getRevision,
    getSetupInputRevision,
    form.values.image,
    retainCurrentDraft,
    draftReady
  ])

  const applyCharacterRoute = React.useCallback<ApplyCharacterRoute>(async (characterId, routeIsCurrent) => {
    let sourceIsCurrent = captureAssistantPreparationGuard();
    const selectionRevision = getSelectedAssistantOperationRevision();
    const inputs = readAssistantPreparationInputs();
    const settings = useStoreChatModelSettings.getState();
    const tools = useMcpToolsStore.getState();
    const sourceCurrent = () => sourceIsCurrent() && routeIsCurrent() &&
      assistantComposerRef.current?.draftReady === true &&
      useStoreChatModelSettings.getState() === settings && useMcpToolsStore.getState() === tools &&
      readAssistantPreparationInputs().every((value, index) => value === inputs[index]);
    let snapshot: Awaited<ReturnType<typeof loadServicePromptSnapshot>> | undefined;
    const loadIsCurrent = () => sourceCurrent() && getSelectedAssistantOperationRevision() === selectionRevision &&
      !snapshot?.scopeSignal.aborted && !snapshot?.scopeInvalidatedSignal.aborted;
    let cardLoaded = false;
    try {
      if (!loadIsCurrent()) return "retired";
      snapshot = await loadServicePromptSnapshot([]);
      if (!loadIsCurrent()) return "retired";
      const character = await tldwClient.getCharacter(characterId, { requestScope: snapshot.requestScope, signal: snapshot.scopeSignal });
      if (!loadIsCurrent()) return "retired";
      if (!character || String(character.id) !== characterId) return "missing";
      cardLoaded = true;
      const application = applyAssistantSelection(characterToAssistantSelection({ ...character, metadata: { ...character.metadata, selectionMode: "tracked" } }), {
        freshCharacterRoute: true,
        isCurrent: () => routeIsCurrent() && !snapshot?.scopeSignal.aborted && !snapshot?.scopeInvalidatedSignal.aborted,
      });
      // Apply starts its own action revision synchronously, before preparation awaits.
      sourceIsCurrent = captureAssistantPreparationGuard();
      const accepted = await application;
      if (accepted) return "applied";
      return sourceCurrent() ? "failed" : "retired";
    } catch (error) {
      if (!sourceCurrent()) return "retired";
      if (!cardLoaded) return "load-error";
      notificationApi.error({ message: t("common:error", "Error"),
        description: nativeCharacterPreflightMessage(error, t) ?? t("playground:composer.rolePlayApplyError", "Settings could not be applied. Your draft is still here; retry Apply.") });
      return "failed";
    } finally { snapshot?.release(); }
  }, [applyAssistantSelection, captureAssistantPreparationGuard, notificationApi, readAssistantPreparationInputs, t]);
  const characterRouteApplyRef = React.useRef(applyCharacterRoute);
  React.useLayoutEffect(() => { characterRouteApplyRef.current = applyCharacterRoute; }, [applyCharacterRoute]);
  React.useEffect(() => {
    if (!onCharacterRouteApplyReady || !draftReady) return;
    onCharacterRouteApplyReady((...args) => characterRouteApplyRef.current(...args));
    return () => onCharacterRouteApplyReady(null);
  }, [onCharacterRouteApplyReady, draftReady]);

  const restoreRetainedDraft: RestoreChatDraft = async (draft, copy) => {
    const source = useStoreMessageOption.getState();
    const current = () => useStoreMessageOption.getState().chatDraftAddress === source.chatDraftAddress &&
      useStoreMessageOption.getState().chatDraftRevision === source.chatDraftRevision;
    if (!requestSettingsNavigation("/") || !(await copy(draft.surfaceKey)) || !current()) return false;
    if (clearChat() === false) return false;
    return useStoreMessageOption.getState().setChatDraftAddress(draft.handle.address);
  };

  const receiverDraftAddress = useStoreMessageOption(state => state.chatDraftAddress);
  const receiverDraftRevision = useStoreMessageOption(state => state.chatDraftRevision);
  const handoffDestinationKey = JSON.stringify([draftOwnerKey, historyId, serverChatId, receiverDraftAddress, receiverDraftRevision]);
  const captureHandoffGuard = React.useCallback(() => {
    const selectionIsCurrent = historySelection?.fence() ?? (() => true);
    const expected = useStoreMessageOption.getState();
    return () => {
      const current = useStoreMessageOption.getState();
      return assistantActionMountedRef.current && isDraftOwnerCurrent() && selectionIsCurrent() &&
        current.chatDraftAddress === expected.chatDraftAddress && current.chatDraftRevision === expected.chatDraftRevision &&
        current.historyId === expected.historyId && current.serverChatId === expected.serverChatId;
    };
  }, [historySelection, isDraftOwnerCurrent]);
  const hasDraft = (form.values.message || "").trim().length > 0;
  const {
    importedContext: importedSidepanelContext,
    removeImportedContext: clearImportedSidepanelContext,
    conflict: sidepanelHandoffConflict,
    importPending: sidepanelImportPending,
    conflictFeedback: sidepanelImportFeedback,
    skippedFeedback: sidepanelSkippedFeedback,
    importReceipt: sidepanelImportReceipt,
    dismissImportReceipt: dismissSidepanelImportReceipt,
    literalOptionsReview: sidepanelOptionReview,
    includeLiteralOptions: includeSidepanelOptions,
    setIncludeLiteralOptions: setIncludeSidepanelOptions,
    applyHandoffOptions: applySidepanelOptions,
    insertHandoffDraft,
    replaceWithHandoffDraft,
    cancelHandoffImport,
  } = useSidepanelChatHandoffImport({
    ready: draftReady,
    destinationKey: handoffDestinationKey,
    captureGuard: captureHandoffGuard,
    draftValue: form.values.message || "",
    getDraftRevision: form.getRevision,
    literalOptions: {
      current: { selectedModel, chatMode, webSearch, toolChoice, useOCR },
      setSelectedModel, setChatMode, setWebSearch, setToolChoice, setUseOCR,
      canApply: () => historySelection?.settingsMode(serverChatId) !== "pending",
    },
    setMessageValue,
    notificationApi,
    t,
  });
  const sidepanelContextOnlyCopy = Boolean(
    sidepanelHandoffConflict?.handoff.pageContext &&
    sidepanelHandoffConflict.handoff.hasText !== false &&
    sidepanelHandoffConflict.handoff.draft.text.length === 0,
  );
  const sidepanelCopyShortened = Boolean(
    sidepanelHandoffConflict?.handoff.draft.truncated ||
    sidepanelHandoffConflict?.handoff.pageContext?.truncated ||
    sidepanelHandoffConflict?.handoff.pageContext?.snippets.some(
      (snippet) => snippet.truncated,
    ),
  );
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
      focusFirstVisibleElement("[data-testid='model-selector']");
      modelSelectorReturnFocusSelectorRef.current = normalizeFocusSelector(
        detail?.returnFocusSelector,
      );
      openModelApiSelector();
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
  const clearPromptContext = React.useCallback(
    (
      isCurrent: () => boolean = () => true,
      settingsOperation?: ModelSettingsOperation
    ) => {
      for (const clear of [
        () => setSelectedQuickPrompt(null),
        () => setSelectedSystemPrompt("")
      ]) {
        if (!isCurrent()) return false
        clear()
      }
      for (const clear of [
        () => setSystemPrompt(""),
        () => updateChatModelSetting("systemPromptTemplateId", undefined)
      ]) {
        if (!isCurrent()) return false
        if (settingsOperation) {
          if (!settingsOperation.write(clear)) return false
        } else clear()
      }
      return isCurrent()
    },
    [
      setSelectedQuickPrompt,
      setSelectedSystemPrompt,
      setSystemPrompt,
      updateChatModelSetting
    ]
  )
  const clearBehaviorTemplateIdentity = React.useCallback(() => {
    updateChatModelSetting("systemPromptTemplateId", undefined);
  }, [updateChatModelSetting]);
  const setSelectedSystemPromptForComposer = React.useCallback(
    (id: string | undefined) => {
      if (id === selectedSystemPrompt) return;
      clearBehaviorTemplateIdentity();
      updateChatModelSetting("systemPrompt", undefined);
      setSelectedSystemPrompt(id);
    },
    [
      clearBehaviorTemplateIdentity,
      selectedSystemPrompt,
      setSelectedSystemPrompt,
      updateChatModelSetting,
    ],
  );
  const setSelectedQuickPromptForComposer = React.useCallback(
    (prompt: string | undefined) => {
      clearBehaviorTemplateIdentity();
      setSelectedQuickPrompt(prompt);
    },
    [clearBehaviorTemplateIdentity, setSelectedQuickPrompt],
  );
  const clearRolePlayIdentity = React.useCallback(() => {
    void applyAssistantSelection(null);
  }, [applyAssistantSelection]);
  const resetRolePlayGenerationStyle = React.useCallback(
    (
      isCurrent: () => boolean = () => true,
      settingsOperation?: ModelSettingsOperation
    ) => {
      const preset = getPresetByKey("balanced")
      if (!preset || !isCurrent()) return false
      if (settingsOperation) {
        if (
          !settingsOperation.write(() => updateChatModelSettings(preset.settings))
        )
          return false
      } else updateChatModelSettings(preset.settings)
      return isCurrent()
    },
    [updateChatModelSettings]
  )
  const handleApplyRolePlaySetup = React.useCallback(
    async (
      original: RolePlaySetupApplyPayload,
      context?: RolePlayApplyContext
    ) => {
      const payload = structuredClone(original)
      let destination:
        | Parameters<NonNullable<RolePlayApplyContext["onDestination"]>>[0]
        | null = null
      let modelIsCurrent = () => false
      const isCurrent = () =>
        destination?.isCurrent() === true &&
        modelIsCurrent() &&
        context?.isCurrent?.() !== false
      const preset = payload.resetGenerationStyle
        ? getPresetByKey("balanced")
        : payload.generationPresetKey
          ? getPresetByKey(payload.generationPresetKey)
          : undefined
      const settings = {
        ...(preset && preset.key !== "custom" ? preset.settings : {}),
        ...(payload.clearBehavior
          ? { systemPrompt: "", systemPromptTemplateId: undefined }
          : payload.behaviorTemplate
            ? {
                systemPrompt: payload.behaviorTemplate.content,
                systemPromptTemplateId: payload.behaviorTemplate.id
              }
            : {})
      }
      const active = useStoreMessageOption.getState()
      const existingIdentity =
        resolvedAssistantSelection ??
        selectedAssistant ??
        (active.serverChatAssistantKind === "character" &&
        active.serverChatCharacterId != null
          ? {
              kind: "character" as const,
              id: String(active.serverChatCharacterId),
              name: String(active.serverChatCharacterId)
            }
          : null)
      if (
        !(await applyAssistantSelection(
          payload.clearIdentity
            ? null
            : payload.identitySelection ?? existingIdentity,
          {
            ...context,
            settings,
            sceneSettings: payload.sceneSettings,
            onDestination: (next) => {
              modelIsCurrent = captureSelectedModelGuard()
              destination = {
                ...next,
                isCurrent: () => next.isCurrent() && modelIsCurrent()
              }
              context?.onDestination?.(destination)
            },
            applyIdentity: Boolean(
              payload.clearIdentity || payload.identitySelection
            ),
            ...(payload.clearBehavior || payload.behaviorTemplate
              ? { selectedSystemPrompt: null }
              : {})
          }
        ))
      )
        return false
      if (!isCurrent()) return false
      if (payload.clearBehavior) {
        if (!clearPromptContext(isCurrent, destination?.settingsOperation))
          return false
      } else if (payload.behaviorTemplate) {
        if (
          !handleTemplateSelect(
            payload.behaviorTemplate,
            isCurrent,
            destination?.settingsOperation
          )
        )
          return false
      }
      if (payload.resetGenerationStyle) {
        if (
          !resetRolePlayGenerationStyle(isCurrent, destination?.settingsOperation)
        )
          return false
      } else if (payload.generationPresetKey) {
        const preset = getPresetByKey(payload.generationPresetKey)
        if (preset && preset.key !== "custom") {
          if (!isCurrent()) return false
          if (
            !destination?.settingsOperation?.write(() =>
              updateChatModelSettings(preset.settings)
            )
          )
            return false
        }
      }
      return isCurrent()
    },
    [
      clearPromptContext,
      handleTemplateSelect,
      resetRolePlayGenerationStyle,
      applyAssistantSelection,
      resolvedAssistantSelection,
      selectedAssistant,
      updateChatModelSettings
    ]
  )
  const startupTemplateProgressRef = React.useRef<{
    leaseId: object
    template: StartupTemplateBundle
    originalScene: ActorSceneRead | null
    originalAccount: ActorSettingsTarget["account"]
    settingsOperation: ModelSettingsOperation
    destination: {
      historyId: string | null
      serverChatId: string | null
      isCurrent: () => boolean
      sceneTarget: ActorSettingsTarget | null
      expectedRecord: ActorSceneRead["originalRecord"] | undefined
    }
    settingsAccepted: boolean
  } | null>(null)
  const startupTemplateAttemptRef = React.useRef<{
    leaseId: object
    id: object
  } | null>(null)
  const [startupTemplateBusyLease, setStartupTemplateBusyLease] = React.useState<object | null>(null);
  const startupTemplateApplying = Boolean(
    startupTemplatePreviewLease &&
      startupTemplateBusyLease === startupTemplatePreviewLease.id
  )
  const handleApplyStartupTemplatePreview = React.useCallback(async () => {
    const lease = startupTemplatePreviewLease
    if (
      !startupTemplatePreview ||
      !lease?.isCurrent() ||
      startupTemplateAttemptRef.current?.leaseId === lease.id
    )
      return false
    let progress = startupTemplateProgressRef.current
    if (progress?.leaseId !== lease.id) {
      if (!lease.sourceIsCurrent()) return false
      const destination = useStoreMessageOption.getState()
      const settingsOperation = captureModelSettingsOperation()
      const modelIsCurrent = captureSelectedModelGuard()
      const toolChoice = destination.toolChoice
      const toolsState = useMcpToolsStore.getState()
      const sceneTarget = startupTemplatePreview.rolePlay
        ? captureCurrentActorSceneTarget(historySelection)
        : null
      const draftAccount = historySelection?.draftAccount
      const originalAccount =
        draftAccount?.requestScope &&
        draftAccount.isCurrent() &&
        chatDraftOwnerKey(draftAccount.requestScope) === sceneTarget?.accountKey
          ? {
              requestScope: structuredClone(draftAccount.requestScope),
              isCurrent: draftAccount.isCurrent
            }
          : !draftAccount
            ? sceneTarget?.account
            : undefined
      progress = {
        leaseId: lease.id,
        originalScene: null,
        originalAccount,
        settingsOperation,
        template: JSON.parse(
          JSON.stringify(startupTemplatePreview)
        ) as StartupTemplateBundle,
        destination: {
          historyId: destination.historyId,
          serverChatId: destination.serverChatId,
          isCurrent: () =>
            lease.sourceIsCurrent() &&
            settingsOperation.isCurrent() &&
            modelIsCurrent() &&
            useStoreMessageOption.getState().toolChoice === toolChoice &&
            useMcpToolsStore.getState() === toolsState,
          sceneTarget,
          expectedRecord: undefined
        },
        settingsAccepted: false
      }
      startupTemplateProgressRef.current = progress
    }
    if (!progress.destination.isCurrent()) return false
    const proposal = progress
    const attempt = {}
    startupTemplateAttemptRef.current = { leaseId: lease.id, id: attempt }
    setStartupTemplateBusyLease(lease.id)
    const isCurrent = () =>
      assistantActionMountedRef.current &&
      lease.isCurrent() &&
      lease.continuationIsCurrent() &&
      startupTemplateAttemptRef.current?.id === attempt &&
      proposal.settingsOperation.isCurrent() &&
      proposal.destination.isCurrent()
    try {
      if (proposal.template.rolePlay && !proposal.originalScene) {
        const target = proposal.destination.sceneTarget
        if (!target) throw new Error("actor_settings_unavailable")
        const read = await getActorSceneForChat({
          target,
          validate_lease: isCurrent
        })
        if (!isCurrent()) return false
        proposal.originalScene = read
        proposal.destination.expectedRecord = read.originalRecord
      }
      if (!proposal.settingsAccepted) {
        const applied = await applyStartupTemplateBundle(proposal.template, {
          applicationId: lease.id,
          settingsOperation: proposal.settingsOperation,
          isCurrent,
          onRestoreCancelled: (revision) => {
            if (
              !assistantActionMountedRef.current ||
              !lease.isCurrent() ||
              !lease.continuationIsCurrent() ||
              startupTemplateAttemptRef.current?.id !== attempt ||
              !proposal.settingsOperation.isCurrent() ||
              proposal.originalAccount?.isCurrent() === false ||
              proposal.destination.sceneTarget?.isCurrent() === false ||
              lease.acceptRestoreCancelled?.(revision) !== true
            )
              return false
            return isCurrent()
          },
          onDestination: (next) => {
            const original = proposal.originalScene
            const sameTarget =
              !next.sceneTarget || next.sceneTarget.key === original?.target.key
            const target = sameTarget
              ? original?.target ?? null
              : next.sceneTarget!
            if (
              original &&
              (!target?.isCurrent() ||
                (!sameTarget &&
                  (target.address?.kind !== "fresh" ||
                    proposal.originalAccount?.isCurrent() !== true ||
                    target.accountKey !== original.target.accountKey)))
            ) {
              proposal.destination.isCurrent = () => false
              return
            }
            proposal.destination = {
              ...next,
              sceneTarget: target,
              expectedRecord: sameTarget ? original?.originalRecord : null,
              isCurrent: () =>
                lease.isCurrent() &&
                next.isCurrent() &&
                (!original || target?.isCurrent() === true) &&
                proposal.originalAccount?.isCurrent() !== false
            }
          }
        })
        if (applied === false || !isCurrent()) return false
        proposal.settingsAccepted = true
      }
      if (!isCurrent()) return false
      if (proposal.template.rolePlay) {
        const target = proposal.destination.sceneTarget
        if (!target) throw new Error("actor_settings_unavailable")
        const nextScene =
          proposal.template.rolePlay.scene ?? createDefaultActorSettings()
        const saved = await saveActorSettingsForChat({
          historyId: proposal.destination.historyId,
          serverChatId: proposal.destination.serverChatId,
          settings: nextScene,
          target,
          expectedRecord: proposal.destination.expectedRecord,
          validate_lease: isCurrent
        })
        if (!isCurrent()) return false
        if (!saved) throw new Error("Scene settings save failed")
        const preview = summarizeRolePlayScene(nextScene)
        if (
          !applyActorScene({
            target,
            settings: nextScene,
            preview: preview.prompt,
            tokenCount: preview.tokenCount
          })
        )
          return false
        if (!isCurrent()) return false
      }
      setModeAnnouncement(
        proposal.template.rolePlay
          ? t(
              "playground:composer.rolePlaySetupAppliedNotice",
              "Role-play setup applied."
            )
          : t(
              "playground:composer.startupTemplateAppliedNotice",
              "Startup template applied."
            )
      )
      if (!isCurrent()) return false
      setStartupTemplatePreview(null)
      return true
    } catch (error) {
      if (!isCurrent()) return false
      notificationApi.error({
        message: t("common:error", "Error"),
        description:
          nativeCharacterPreflightMessage(error, t) ??
          (proposal.settingsAccepted && proposal.template.rolePlay
            ? t(
                "playground:composer.rolePlayScenePartialSaveError",
                "Identity and behavior were applied, but scene settings could not be saved. Your draft is still here; retry Apply."
              )
            : t(
                "playground:composer.rolePlayApplyError",
                "Settings could not be applied. Your draft is still here; retry Apply."
              ))
      })
      return false
    } finally {
      if (startupTemplateAttemptRef.current?.id === attempt) {
        startupTemplateAttemptRef.current = null
        if (assistantActionMountedRef.current)
          setStartupTemplateBusyLease((current) =>
            current === lease.id ? null : current
          )
      }
    }
  }, [
    applyStartupTemplateBundle,
    notificationApi,
    setModeAnnouncement,
    setStartupTemplatePreview,
    t,
    applyActorScene,
    historySelection,
    startupTemplatePreview,
    startupTemplatePreviewLease
  ])
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
    <>
    <SelectedModelSaveFeedback failed={selectedModelSaveFailed} onRetry={retrySelectedModelSave} />
    <ChatModelSelectorDropdown
      apiModelLabel={apiModelLabel}
      catalogControls={modelCatalogControls}
      connectionStatusLabel={connectionStatusLabel}
      connectionStatusWarning={
        !isConnectionReady || connectionUxState === "connected_degraded"
      }
      modelUsabilityLabel={characterChatModelUsabilityLabel ?? undefined}
      modelUsabilityTitle={characterChatModelUsabilityTitle ?? undefined}
      modelUsabilityWarning={Boolean(
        characterChatModelUsability && !characterChatModelUsability.canSend
      )}
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
    </>
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

  const mediaHandoffToken = new URLSearchParams(location.search).get(MEDIA_CHAT_HANDOFF_PARAM);
  const mediaDestinationRef = React.useRef({ token: mediaHandoffToken, owner: homeScope, draftReady, text: form.values.message });
  mediaDestinationRef.current = { token: mediaHandoffToken, owner: homeScope, draftReady, text: form.values.message };
  const handoffBoundaryRevision = React.useRef(0);
  React.useLayoutEffect(() => watchChatAccountChanges(invalidated => {
    if (invalidated) handoffBoundaryRevision.current += 1;
  }), []);
  const mediaLifetime = React.useRef<AbortController | null>(null);
  React.useLayoutEffect(() => {
    const lifetime = new AbortController();
    mediaLifetime.current = lifetime;
    return () => lifetime.abort();
  }, []);
  const handledMediaHandoffs = React.useRef(new Set<string>());
  const [mediaHandoffConflict, setMediaHandoffConflict] = React.useState<{ token: string; owner: string } | null>(null);
  const cleanMediaHandoffRoute = () => {
    const params = new URLSearchParams(location.search);
    params.delete(MEDIA_CHAT_HANDOFF_PARAM);
    navigate({ pathname: location.pathname, search: params.toString() ? `?${params}` : "", hash: location.hash }, { replace: true });
  };

  const applyDiscussMediaPayload = (
    payload: MediaChatHandoffPayload,
    message: string,
  ) => {
      useStoreMessageOption.getState().setSelectedKnowledge(null);
      const mode = getMediaChatHandoffMode(payload);
      if (mode === "rag_media") {
        const mediaId = parseMediaIdAsNumber(payload);
        const mediaIds = payload.mediaIds || (mediaId == null ? [] : [mediaId]);
        if (mediaIds.length) {
          usePlaygroundSessionStore.getState().markSourceSelectionIntent();
          setChatMode("rag");
          setRagMediaIds(mediaIds);
          setFileRetrievalEnabled(true);
        }
      } else {
        usePlaygroundSessionStore.getState().markSourceSelectionIntent();
        setChatMode("normal");
        setRagMediaIds(null);
      }
      if (!message) return;
      setMessageValue(message, { collapseLarge: true, forceCollapse: true });
      textAreaFocus();
  };
  const applyMediaRef = React.useRef(applyDiscussMediaPayload);
  applyMediaRef.current = applyDiscussMediaPayload;
  const cleanMediaRouteRef = React.useRef(cleanMediaHandoffRoute);
  cleanMediaRouteRef.current = cleanMediaHandoffRoute;

  const acceptMediaHandoff = React.useCallback(async (
    token: string, owner: string, action: "auto" | "insert" | "replace", current: () => boolean = () => true,
  ) => {
    const boundaryRevision = handoffBoundaryRevision.current;
    const lifetime = mediaLifetime.current;
    if (!lifetime || lifetime.signal.aborted) return;
    await consumeMediaChatHandoff(token, owner, payload => {
      const destination = mediaDestinationRef.current;
      if (boundaryRevision !== handoffBoundaryRevision.current || lifetime.signal.aborted || !current() || !destination.draftReady || destination.token !== token || destination.owner !== owner || handledMediaHandoffs.current.has(token)) return false;
      const hint = buildDiscussMediaHint(payload);
      if (action === "auto" && destination.text.trim()) {
        setMediaHandoffConflict({ token, owner });
        return false;
      }
      const text = action === "insert" && destination.text ? `${destination.text}\n\n${hint}` : hint;
      handledMediaHandoffs.current.add(token);
      applyMediaRef.current(payload, text);
      setMediaHandoffConflict(null);
      cleanMediaRouteRef.current();
      return true;
    });
  }, []);

  // Only this tab's explicit destination token may deliver source text. Legacy
  // shared pending settings and unaddressed window events are never consumed.
  React.useEffect(() => {
    if (!mediaHandoffToken || !homeScope || !draftReady || handledMediaHandoffs.current.has(mediaHandoffToken)) return;
    let cancelled = false;
    void acceptMediaHandoff(mediaHandoffToken, homeScope, "auto", () => !cancelled);
    return () => { cancelled = true; };
  }, [acceptMediaHandoff, mediaHandoffToken, homeScope, draftReady]);

  const previousMediaOwner = React.useRef(homeScope);
  React.useLayoutEffect(() => {
    const previous = previousMediaOwner.current;
    previousMediaOwner.current = homeScope;
    if (previous && previous !== homeScope) {
      setMediaHandoffConflict(null);
      if (mediaHandoffToken) {
        handledMediaHandoffs.current.add(mediaHandoffToken);
        // Drop this destination request without clearing another owner’s token.
        // Expiration handles a transfer abandoned across an account boundary.
        cleanMediaRouteRef.current();
      }
    }
  }, [homeScope, mediaHandoffToken]);

  const cancelMediaHandoff = () => {
    if (!mediaHandoffConflict) return;
    handledMediaHandoffs.current.add(mediaHandoffConflict.token);
    void removeMediaChatHandoff(mediaHandoffConflict.token).catch(() => undefined);
    setMediaHandoffConflict(null);
    cleanMediaHandoffRoute();
  };

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
    captureImageOwnership: () => captureDraftOwnership().isImageCurrent,
    setFieldValue: form.setFieldValue,
    handleFileUpload,
    notifyImageAttachmentDisabled,
    notifyImageReadError: () => notificationApi.error({
      message: t("playground:composer.invalidImage", "This image could not be opened. Choose a valid image file."),
    }),
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
  const handleTextareaKeyDown = (
    e: React.KeyboardEvent<HTMLTextAreaElement>,
  ) => {
    if (handleCollapsedKeyDown(e)) return;
    handleKeyDown(e);
  };

  const handleTextareaFocus = React.useCallback(() => {
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
    messageValue: form.values.message || "",
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
      ownerScope: homeScope,
      isCurrent: captureCompletionGuard(),
      canFocus: captureCompletionGuard(true),
    }),
    onSuccess: (data, _variables, context) => {
      const result = normalizeChatSubmitResult(data);
      if (isChatSubmitSuccess(result)) {
        if (context?.ownerScope) {
          useMilestoneStore.getState().markScopedMilestone(context.ownerScope, "first_chat");
        }
        if (!context?.isCurrent()) return;
        dismissChatErrorAfterSuccessfulSubmit(context?.errorKeyToDismiss ?? null);
        void trackOnboardingChatSubmitSuccess(
          typeof window !== "undefined" ? window.location.pathname : "/chat",
          context.isCurrent,
        );
      }
      if (!context?.isCurrent()) return;
      if (context.canFocus()) textAreaFocus();
      queryClient.invalidateQueries({
        queryKey: ["fetchChatHistory"],
      });
    },
    onError: (_error, _variables, context) => {
      if (context?.canFocus()) textAreaFocus();
    },
  });

  const handleRetryChatError = React.useCallback(() => {
    void regenerateLastMessage().catch(() => textAreaFocus())
  }, [regenerateLastMessage, textAreaFocus])

  const handleEditChatProvider = React.useCallback(() => {
    setOpenModelSettings(true)
  }, [setOpenModelSettings])

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

  const queuedCompletionIsCurrent = React.useMemo(captureCompletionGuard, [captureCompletionGuard]);
  const queueMgmt = usePlaygroundQueueManagement({
    historySelection,
    canCommitDispatchResult: queuedCompletionIsCurrent,
    getDispatchDisabledReason: getSubmitDisabledReason,
    captureDispatchFence: captureAssistantActionGuard,
    composerModels,
    composerModelsLoading,
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
    setToolChoice,
    setUseOCR,
    compareModelsSupportCapability,
    sendMessage,
    stopStreamingRequest,
    form,
    clearSelectedDocuments,
    clearUploadedFiles,
    textAreaFocus,
    onEnqueueSuccess: () => {
      const attemptId = beginPromptAssistReset();
      markPromptAssistAttemptSaved(attemptId);
    },
    notificationApi,
    t,
  });
  const {
    availableChatModelIds,
    isQueuedDispatchBlockedByComposerState,
    queuedRequestActions,
    queueSubmission,
    cancelCurrentAndRunDisabledReason,
    handleRunQueuedRequest: runQueuedRequest,
    handleRunNextQueuedRequest: runNextQueuedRequest,
    validateSelectedChatModelsAvailability,
  } = queueMgmt;
  const handleRunQueuedRequest = (...args: Parameters<typeof runQueuedRequest>) => {
    if (getSubmitDisabledReason()) return;
    return runQueuedRequest(...args);
  };
  const handleRunNextQueuedRequest = () => {
    if (getSubmitDisabledReason()) return;
    return runNextQueuedRequest();
  };

  const handleToggleWebSearch = React.useCallback(() => {
    setWebSearch(!webSearch);
  }, [setWebSearch, webSearch]);
  const handleOpenModelSettings = React.useCallback(() => {
    modelSettingsReturnFocusRef.current = null;
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

  const reserveDocumentProcessingTurn = React.useCallback(
    ({
      message,
      metadata,
    }: {
      message: string;
      metadata: DocumentProcessingTurnMetadata;
    }) => {
      const id = generateID();
      setMessages((prev) => [
        ...prev,
        {
          isBot: false,
          role: "user",
          name: "You",
          message,
          sources: [],
          createdAt: Date.now(),
          id,
          metadataExtra: {
            documentProcessing: metadata,
          },
        },
      ]);
      return id;
    },
    [setMessages],
  );

  const updateDocumentProcessingTurn = React.useCallback(
    (userMessageId: string, metadata: DocumentProcessingTurnMetadata) => {
      setMessages((prev) =>
        prev.map((message) =>
          message.id === userMessageId && !message.isBot
            ? {
                ...message,
                metadataExtra: {
                  ...(message.metadataExtra ?? {}),
                  documentProcessing: metadata,
                },
              }
            : message,
        ),
      );
    },
    [setMessages],
  );

  const historySelection = useHistorySelectionContext();
  const { submitForm, submitFormRef, isPreparingDocuments } = usePlaygroundSubmit({
    historySelection,
    captureCompletionGuard,
    form,
    draftReady,
    beginPromptAssistReset,
    restorePromptAssistAttempt,
    markPromptAssistAttemptSaved,
    captureDraftOwnership,
    captureHistorySelection: () => historySelection?.fence() ?? (() => false),
    isSending,
    isConnectionReady,
    webSearch,
    chatMode,
    systemPrompt,
    selectedSystemPrompt,
    systemPromptTemplateId: currentChatModelSettings.systemPromptTemplateId,
    compareModeActive,
    compareSelectedModels,
    selectedModel,
    historyId,
    serverChatId,
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
    openUIRequestMode,
    researchContext,
    importedSidepanelContext,
    clearImportedSidepanelContext,
    sendMessage,
    clearOpenUIRequestMode: () => {
      if (openUIRequestRevisionRef.current !== openUIRequestRevision) return;
      openUIRequestRevisionRef.current++;
      setOpenUIRequestMode(false);
    },
    clearSelectedDocuments,
    clearUploadedFiles,
    reserveDocumentProcessingTurn,
    updateDocumentProcessingTurn,
    textAreaFocus,
    setLastSubmittedContext,
    estimateTokensForText: estimateTokensForText as any,
    resolveSubmissionIntent,
    queueSubmission,
    getSubmitDisabledReason,
    validateSelectedChatModelsAvailability,
    compareModelsSupportCapability,
    notificationApi,
    t,
  });
  // CS-01 (#3106): a chat that addresses no conversation (after New chat or
  // Clear, or while a restored one is still loading) sends only once its
  // history-selection controller is idle; until then Send stays disabled and
  // the draft stays in the composer.
  // CS-04: the same while any chat's selected history is loading, such as the
  // view that follows a stopped or interrupted reply. The send path refuses a
  // selection that is not ready, which used to drop the draft.
  const historySelectionPending =
    !isSending &&
    (historySelection?.status === "loading" ||
      isFreshChatSelectionPending(historySelection, {
        historyId,
        serverChatId,
      }));
  const runCharacterChatSendBlocker = React.useCallback(() => {
    if (characterChatSendBlocker?.active) {
      stopListening();
      characterChatSendBlocker.onAction();
    }
  }, [characterChatSendBlocker, stopListening]);
  const greetingDraftAddress = useStoreMessageOption(state => state.chatDraftAddress);
  const greetingDraftRevision = useStoreMessageOption(state => state.chatDraftRevision);
  const { starting: greetingStarting, isPreparing: isGreetingPreparing, start: startCharacterGreeting } = useCharacterGreetingStart({
    form, retainCurrentDraft, submit: submitMessageAction, controller: historySelection, focusComposer: () => textareaRef.current?.focus(),
    ready: draftReady, isConnectionReady, busy: isSending || isPreparingDocuments,
    readContext: () => [isConnectionReady, Boolean(characterSubmitBlockerRef.current?.active), composerModels, composerModelsLoading, JSON.stringify(activeAssistantSelection),
      chatMode, compareModeActive, selectedDocuments, importedSidepanelContext, researchContext, openUIRequestMode, appendFormattingGuidePrompt],
    preflight: () => {
      if (!isConnectionReady) throw new Error("native_history_server_unavailable");
      if (!buildChatModelUsability({ selectedModel, availableModels: composerModels,
        modelsLoading: composerModelsLoading, isServerConnected: isConnectionReady }).canSend)
        throw new Error("native_history_model_unavailable");
      if (chatMode !== "normal" || compareModeActive || selectedDocuments.length || importedSidepanelContext ||
          researchContext || openUIRequestMode || resolveOutputFormattingGuideSuffix(Boolean(appendFormattingGuidePrompt)))
        throw new Error("native_history_explicit_context_unsupported");
    },
  });
  greetingSubmitGateRef.current = isGreetingPreparing;
  const handleComposerSend = React.useCallback(() => {
    if (greetingStarting || historySelectionPending || personaSubmitDisabledReason) return;
    if (characterChatSendBlocker?.active) {
      runCharacterChatSendBlocker();
      return;
    }
    submitForm();
  }, [personaSubmitDisabledReason, greetingStarting, historySelectionPending, characterChatSendBlocker, runCharacterChatSendBlocker, submitForm]);
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
      // The question stays in the composer until the selection is idle.
      if (historySelectionPending) return;
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
      historySelectionPending,
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
    setServerChatMetaLoaded: useStoreMessageOption.getState().setServerChatMetaLoaded,
    setServerChatCharacterId,
    setServerChatAssistantKind,
    setServerChatAssistantId,
    setServerChatPersonaMemoryMode,
    history,
    messages,
    setMessages,
    clearChat,
    selectedCharacter,
    selectedAssistantMode,
    characterWorkflowActive,
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
    privateChatLocked,
    showServerPersistenceHint,
    handleToggleTemporaryChat,
    handleSaveChatToServer,
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

    const source = captureChatDraftSource(captureAssistantPreparationGuard());
    const clear = () => {
      if (!source.isCurrent()) return;
      // CS-05 (#3106): reset both transcript and request history together.
      if (clearChat() === false) return;
      notificationApi.success({
        message: t(
          "playground:composer.clearContextSuccess",
          "Conversation cleared",
        ),
        duration: 2,
      });
    };
    if (hasTemporaryChatContent(useStoreMessageOption.getState())) {
      clear();
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
      onOk: clear,
    });
  }, [captureAssistantPreparationGuard, clearChat, history.length, notificationApi, t]);

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
          applyAs: "tracked",
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
    compareFeatureReady,
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
    draftOwnerKey,
    isDraftOwnerCurrent,
    captureCompletionGuard,
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
    referenceImageCandidatesFailed,
    retryReferenceImageCandidates,
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
      if (handleChatInputKeyDown({ e, sendWhenEnter, typing, isSending: false })) e.preventDefault();
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
      activeAssistantSelection?.kind === "character" ||
      activeAssistantSelection?.kind === "persona"
    ) {
      return {
        kind: activeAssistantSelection.kind,
        id: activeAssistantSelection.id,
        name: activeAssistantSelection.name,
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
  }, [activeAssistantSelection, selectedCharacter]);
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
    isConnectionReady,
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
    composerMessageCount,
    composerMessageCountLabel,
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
    onClearRolePlayBehavior: () => { clearPromptContext(); },
    onResetRolePlayGenerationStyle: () => { resetRolePlayGenerationStyle(); },
    onDisableCompareMode: compareModeActive ? toggleCompareMode : undefined,
    onOpenRolePlaySetup: openRolePlaySetup,
    t,
  });

  const handleDocumentProcessingFilesChange = React.useCallback(
    (files: UploadedFile[]) => {
      setUploadedFiles(files);
      setContextFiles(
        files.filter((file) => file.processingMode !== "ingest_to_library"),
      );
    },
    [setContextFiles, setUploadedFiles],
  );

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

  const composerOverlayOpen =
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
  const externalPinSources = contextToolsOpen || composerOverlayOpen;

  const {
    actionBarVisible,
    composerTextareaExpanded,
    actionBarVisibilityClass,
    handlers: actionBarHandlers,
  } = useActionBarVisibility({ externalPinSources });
  const keepComposerToolbarVisible = !isMobileViewport || !isProMode;
  const [composerOptionsExpanded, setComposerOptionsExpanded] = useStorage(
    "playgroundComposerOptionsExpanded",
    true,
  );
  const shouldCompactComposerTextarea =
    !composerTextareaExpanded &&
    !contextToolsOpen &&
    !showSlashMenu &&
    !showMentions &&
    !isSending &&
    !isMessageCollapsed &&
    messageDisplayValue.trim().length === 0;
  const composerShellRef = React.useRef<HTMLDivElement>(null);

  const keepComposerFocusInView = React.useCallback(() => {
    if (typeof document === "undefined") return
    const composerEl = composerShellRef.current
    if (!composerEl) return
    const focused = document.activeElement
    const railHasFocus = isMobileViewport && mobileCockpitModeActive &&
      focused instanceof HTMLElement && Boolean(focused.closest('[data-testid="playground-cockpit-mobile-rails"]'))
    if (stickyDockEnabled && !railHasFocus) return
    const target =
      focused && composerEl.contains(focused)
        ? focused
        : !focused || focused === document.body || railHasFocus
          ? textareaRef.current
          : null
    if (target && composerEl.contains(target)) {
      target.scrollIntoView?.({ block: "nearest", inline: "nearest" })
    }
  }, [isMobileViewport, mobileCockpitModeActive, stickyDockEnabled, textareaRef])

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
      keepComposerFocusInView();
      timeoutId = window.setTimeout(() => {
        keepComposerFocusInView();
      }, 220);
    });

    return () => {
      window.cancelAnimationFrame(rafId);
      if (timeoutId !== null) {
        window.clearTimeout(timeoutId);
      }
    };
  }, [actionBarVisible, keepComposerFocusInView]);

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
      keepComposerFocusInView();
      timeoutId = window.setTimeout(() => {
        keepComposerFocusInView();
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
    keepComposerFocusInView,
    mobileComposerViewport.keyboardInsetPx,
    mobileComposerViewport.keyboardOpen,
  ]);

  React.useEffect(() => {
    if (typeof document === "undefined") return;

    const handleFocusIn = (event: FocusEvent) => {
      const target = event.target;
      if (!(target instanceof Node)) return;
      if (!composerShellRef.current?.contains(target) && !(
        isMobileViewport && mobileCockpitModeActive && target instanceof HTMLElement &&
        target.closest('[data-testid="playground-cockpit-mobile-rails"]')
      )) return;
      keepComposerFocusInView();
      if (typeof window !== "undefined") {
        window.setTimeout(() => {
          keepComposerFocusInView();
        }, 80);
      }
    };

    document.addEventListener("focusin", handleFocusIn);
    return () => {
      document.removeEventListener("focusin", handleFocusIn);
    };
  }, [isMobileViewport, mobileCockpitModeActive, keepComposerFocusInView]);

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
      keepComposerFocusInView();
      timeoutId = window.setTimeout(() => {
        keepComposerFocusInView();
      }, 100);
    });

    return () => {
      window.cancelAnimationFrame(rafId);
      if (timeoutId != null) {
        window.clearTimeout(timeoutId);
      }
    };
  }, [isMobileViewport, isSending, keepComposerFocusInView]);

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

  const openUIRequestButton = (
    <Tooltip
      title={
        t(
          "playground:composer.openuiModeTooltip",
          "Answer with an OpenUI interface for the next message.",
        ) as string
      }
    >
      <button
        type="button"
        onClick={() => { openUIRequestRevisionRef.current++; setOpenUIRequestMode((current) => !current); }}
        disabled={isSending}
        aria-pressed={openUIRequestMode}
        aria-label={t("playground:composer.openuiMode", "OpenUI") as string}
        data-testid="openui-request-mode-toggle"
        className={`inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs font-medium transition hover:bg-surface2 hover:text-text disabled:cursor-not-allowed disabled:opacity-50 ${
          openUIRequestMode
            ? "bg-primary/10 text-primaryStrong"
            : "text-text-muted"
        }`}
      >
        <LayoutTemplate className="h-3.5 w-3.5" aria-hidden="true" />
        <span>{t("playground:composer.openuiMode", "OpenUI")}</span>
      </button>
    </Tooltip>
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
  const modelCatalogSubmitDisabledReason = composerModelsLoading && !activeImageCommand &&
    !isSavedRoleplayConversation(historySelection)
    ? toText(t("playground:composer.modelsLoading", "Loading models…")) : null;
  const sendControl = (
    <PlaygroundSendControl
      isProMode={isProMode}
      isMobileViewport={isMobileViewport}
      isSending={isSending}
      isPreparingDocuments={isPreparingDocuments}
      isHistorySelectionPending={historySelectionPending}
      isConnectionReady={isConnectionReady}
      sendWhenEnter={sendWhenEnter}
      onSendWhenEnterChange={setSendWhenEnter}
      sendLabel={modelCatalogSubmitDisabledReason ?? sendLabel}
      compareNeedsMoreModels={compareNeedsMoreModels}
      onStopStreaming={stopStreamingFromUi}
      onStopListening={stopListening}
      onSubmitForm={handleComposerSend}
      characterChatSendBlocker={characterChatSendBlocker}
      submitDisabledReason={personaSubmitDisabledReason ?? modelCatalogSubmitDisabledReason}
      sendMenuOpen={sendMenuOpen}
      onSendMenuChange={handleSendMenuChange}
      t={t}
    />
  );
  const promptAssistAction = (
    <PromptAssistComposerAction
      form={form}
      messageRevision={messageRevision}
      promptAssistMutation={promptAssistMutation}
      promptAssistSavedAttemptId={promptAssistSavedAttemptId}
      modelSelection={selectedModel?.trim()
        ? {
            selected_model: selectedModel,
            provider_hint: currentChatModelSettings.apiProvider,
          }
        : null}
      promptAssistContextKey={serverChatId
        ? `server:${serverChatId}`
        : historyId
          ? `local:${historyId}`
          : "local:playground-draft"}
      promptAssistBackendKey={promptAssistBackendKey}
      promptAssistAuthorizationRevision={promptAssistAuthorizationRevision}
      sending={isSending}
      surfaceOpen
      narrow={isMobileViewport}
      onSelectModel={() =>
        openModelSelector({
          returnFocusSelector: "[aria-label='Improve prompt']",
        })
      }
      onReturnFocus={textAreaFocus}
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
      <div className="flex w-full flex-col items-center px-2 pb-0 sm:px-4 sm:pb-0">
        {!needsConnectionSetup && <div className="flex w-full justify-end"><ChatDraftRecovery creationRecovery={creationRecovery} onReviewCreatedChat={reviewCreatedChat} onCopyCreationText={copyCreationText} onFindConversations={() => window.dispatchEvent(new Event("tldw:open-chat-sidebar"))} listDrafts={listRetainedDrafts} copyDraft={copyRetainedDraft} onRestore={restoreRetainedDraft} disabled={!draftReady} draftOwnerVerificationState={draftOwnerVerificationState} retryDraftOwnerVerification={retryDraftOwnerVerification} /></div>}
        <ChatGreetingPicker selectedCharacter={selectedCharacter} selectedCharacterMode={getAssistantSelectionMode(activeAssistantSelection)}
          messages={messages} historyId={historyId} serverChatId={serverChatId}
          ready={draftReady && !assistantMeta?.isLoading && !temporaryChat && historySelection?.getCurrent().status === "idle" && !historySelection.getCurrent().owner}
          scope={greetingDraftAddress?.kind === "fresh" ? greetingDraftAddress.scope : undefined}
          viewKey={String(greetingDraftRevision)}
          disabled={isSending || isPreparingDocuments || greetingStarting || Boolean(characterChatSendBlocker?.active)}
          blockedReason={creationRecovery.unavailable || (creationRecovery.active && !["prepared", "settled"].includes(creationRecovery.active.state)) ? t("sidepanel:greetingPicker.reviewCreation", "Review the previous save in Recover drafts before saving another opening.") : null}
          inputRevision={form.getRevision()} onStartGreeting={startCharacterGreeting} onStartWithMessage={() => textareaRef.current?.focus()}
          className="mb-3 max-w-[64rem]" />
        <div
          data-checkwidemode={effectiveWideMode}
          data-ui-mode={uiMode}
          className={`relative z-10 flex w-full ${
            effectiveWideMode ? "max-w-none" : "max-w-[64rem]"
          } flex-col items-center justify-center gap-2 text-base`}
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
              className={`relative w-full rounded-xl border border-transparent bg-surface/95 p-2 text-text shadow-sm backdrop-blur-lg transition-all duration-200 data-[istemporary-chat='true']:border-t-4 data-[istemporary-chat='true']:border-t-purple-500 data-[istemporary-chat='true']:border-dashed data-[istemporary-chat='true']:opacity-90 sm:rounded-3xl sm:p-3 sm:shadow-card ${
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
              <QuickPromptInsertionReview form={form} ready={draftReady} destinationKey={handoffDestinationKey}
                captureGuard={captureHandoffGuard} textareaRef={textareaRef}
                setMessageValue={(value) => restoreMessageValue(value, { wasExpanded: true })} />
              {mediaHandoffConflict ? (
                <div role="status" aria-label={t("mediaHandoff.conflictLabel", "Media handoff conflict")} className="mb-2 rounded-md border border-border bg-surface2 px-3 py-2 text-xs text-text">
                  <p>{t("mediaHandoff.conflict", "Your composer already has a draft. Insert the media source, replace your draft, or cancel the import.")}</p>
                  <div className="flex flex-wrap gap-2">
                    <button type="button" className="min-h-7 rounded border border-border px-3 py-1" onClick={() => { void acceptMediaHandoff(mediaHandoffConflict.token, mediaHandoffConflict.owner, "insert"); }}>{t("mediaHandoff.insert", "Insert media source")}</button>
                    <button type="button" className="min-h-7 rounded border border-border px-3 py-1" onClick={() => { void acceptMediaHandoff(mediaHandoffConflict.token, mediaHandoffConflict.owner, "replace"); }}>{t("mediaHandoff.replace", "Replace current draft")}</button>
                    <button type="button" className="min-h-7 rounded border border-border px-3 py-1" onClick={cancelMediaHandoff}>{t("mediaHandoff.cancel", "Cancel import")}</button>
                  </div>
                </div>
              ) : null}
              {sidepanelHandoffConflict ? (
                <div
                  role="region"
                  aria-label={t(
                    "playground:sidepanelHandoff.conflictLabel",
                    "Review copied chat",
                  )}
                  className="mb-2 rounded-md border border-border bg-surface2 px-3 py-2 text-xs text-text"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="min-w-0">
                      <div className="font-medium">
                        {t(
                          sidepanelHandoffConflict.handoff.hasText === false ? "playground:sidepanelHandoff.noTextTitle" : sidepanelContextOnlyCopy ? "playground:sidepanelHandoff.contextOnlyTitle" : "playground:sidepanelHandoff.conflictTitle",
                          sidepanelHandoffConflict.handoff.hasText === false ? "Copy contains no draft text" : sidepanelContextOnlyCopy ? "Import page references?" : "Import sidepanel draft?",
                        )}
                      </div>
                      <div className="text-text-muted">
                        {t(
                          sidepanelHandoffConflict.handoff.hasText === false ? sidepanelOptionReview.length > 0 ? "playground:sidepanelHandoff.settingsOnlyDescription" : "playground:sidepanelHandoff.noTextDescription" : sidepanelContextOnlyCopy ? "playground:sidepanelHandoff.contextOnlyDescription" : "playground:sidepanelHandoff.conflictDescription",
                          sidepanelHandoffConflict.handoff.hasText === false ? sidepanelOptionReview.length > 0 ? "Your draft remains here. Review the copied settings below, apply them if wanted, or dismiss this copy." : "Your draft remains here. Dismiss this copy, then choose the settings or attachments you need." : sidepanelContextOnlyCopy ? "Review the copied page references, add them to this chat, or cancel. Your current draft will be kept." : "Review the copied draft below. Insert it, replace your current draft, or cancel the import.",
                        )}
                      </div>
                    </div>
                    {!sidepanelHandoffConflict.route.webUi ? (
                      <p role="status" className="basis-full text-text-muted">
                        {t(
                          "playground:sidepanelHandoff.sourceAccountUnverified",
                          "This copy does not identify its source account. Review the content before importing.",
                        )}
                      </p>
                    ) : null}
                    {sidepanelCopyShortened ? (
                      <p className="basis-full text-text-muted">
                        {t(
                          "playground:sidepanelHandoff.copyShortened",
                          "Some copied text or page references were shortened.",
                        )}
                      </p>
                    ) : null}
                    {sidepanelHandoffConflict.handoff.hasText !== false && !sidepanelContextOnlyCopy ? <div className="basis-full min-w-0">
                      <p className="font-medium">{t("playground:sidepanelHandoff.copiedDraft", "Copied draft")}</p>
                      <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap break-words font-sans text-sm text-text">{sidepanelHandoffConflict.handoff.draft.text}</pre>
                    </div> : null}
                    {sidepanelHandoffConflict.handoff.pageContext ? (
                      <details className="basis-full min-w-0">
                        <summary className="min-h-11 cursor-pointer py-3 font-medium focus-visible:ring-2 focus-visible:ring-focus">
                          {t(
                            "playground:sidepanelHandoff.copiedPageReferences",
                            "Copied page references",
                          )}
                        </summary>
                        <div
                          role="region"
                          aria-label={t(
                            "playground:sidepanelHandoff.copiedPageReferences",
                            "Copied page references",
                          )}
                          tabIndex={0}
                          className="mt-1 max-h-48 space-y-2 overflow-auto rounded border border-border p-2 focus-visible:ring-2 focus-visible:ring-focus"
                        >
                          {sidepanelHandoffConflict.handoff.pageContext.title ? (
                            <p className="break-words font-medium">
                              {sidepanelHandoffConflict.handoff.pageContext.title}
                            </p>
                          ) : null}
                          {sidepanelHandoffConflict.handoff.pageContext.url ? (
                            <p className="break-all text-text-muted">
                              {sidepanelHandoffConflict.handoff.pageContext.url}
                            </p>
                          ) : null}
                          {sidepanelHandoffConflict.handoff.pageContext.snippets.map(
                            (snippet, index) => (
                              <div key={index}>
                                <p className="font-medium">
                                  {snippet.label ||
                                    t(
                                      "playground:sidepanelHandoff.copiedSnippet",
                                      "Snippet {{number}}",
                                      { number: index + 1 },
                                    )}
                                </p>
                                <pre className="whitespace-pre-wrap break-words font-sans text-sm">
                                  {snippet.text}
                                </pre>
                              </div>
                            ),
                          )}
                        </div>
                      </details>
                    ) : null}
                    {sidepanelOptionReview.length > 0 ? (
                      <div className="basis-full min-w-0 space-y-2">
                        <table
                          aria-label={t(
                            "playground:sidepanelHandoff.copiedSettings",
                            "Copied settings"
                          )}
                          className="w-full table-fixed text-left"
                        >
                          <thead>
                            <tr>
                              <th scope="col">
                                {t("playground:sidepanelHandoff.setting", "Setting")}
                              </th>
                              <th scope="col">
                                {t("playground:sidepanelHandoff.currentValue", "Current")}
                              </th>
                              <th scope="col">
                                {t("playground:sidepanelHandoff.copiedValue", "Copied")}
                              </th>
                            </tr>
                          </thead>
                          <tbody>
                            {sidepanelOptionReview.map((row) => (
                              <tr key={row.key}>
                                <th scope="row" className="py-1 pr-2 font-normal">
                                  {row.label}
                                </th>
                                {[row.current, row.copied].map((value, index) => (
                                  <td key={index} className="break-words py-1 pr-2">
                                    {typeof value === "boolean"
                                      ? value
                                        ? t("common:on", "On")
                                        : t("common:off", "Off")
                                      : (value ?? t("common:none", "None"))}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                        <label className="flex min-h-11 items-center gap-2">
                          <input
                            type="checkbox"
                            checked={includeSidepanelOptions}
                            disabled={sidepanelImportPending}
                            onChange={(event) =>
                              setIncludeSidepanelOptions(event.target.checked)
                            }
                          />
                          {t(
                            "playground:sidepanelHandoff.applySupportedSettings",
                            "Apply supported settings"
                          )}
                        </label>
                        <p className="text-text-muted">
                          {t(
                            "playground:sidepanelHandoff.settingsIntent",
                            "These preferences do not copy prompts, chat privacy, attachments or retrieval sources. Availability is checked when sending."
                          )}
                        </p>
                      </div>
                    ) : null}
                    {sidepanelSkippedFeedback ? <p className="basis-full text-text-muted">{sidepanelSkippedFeedback}</p> : null}
                    {(sidepanelImportPending || sidepanelImportFeedback) ? (
                      <p role="status" className="basis-full text-text-muted">
                        {sidepanelImportPending
                          ? t("playground:sidepanelHandoff.checkingCopy", "Checking the copy…")
                          : sidepanelImportFeedback}
                      </p>
                    ) : null}
                    <div className="flex flex-wrap items-center gap-2">
                      {sidepanelHandoffConflict.handoff.hasText !== false ? <>
                      <button
                        type="button"
                        onClick={insertHandoffDraft}
                        disabled={sidepanelImportPending}
                        aria-busy={sidepanelImportPending || undefined}
                        className="min-h-11 rounded border border-border bg-surface px-3 py-1 text-xs text-text hover:bg-surface3 focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
                      >
                        {t(
                          sidepanelContextOnlyCopy ? "playground:sidepanelHandoff.addPageReferences" : "playground:sidepanelHandoff.insert",
                          sidepanelContextOnlyCopy ? "Add page references" : "Insert handoff draft",
                        )}
                      </button>
                      {!sidepanelContextOnlyCopy ? <button
                        type="button"
                        onClick={replaceWithHandoffDraft}
                        disabled={sidepanelImportPending}
                        aria-busy={sidepanelImportPending || undefined}
                        className="min-h-11 rounded border border-border bg-surface px-3 py-1 text-xs text-text hover:bg-surface3 focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
                      >
                        {t(
                          "playground:sidepanelHandoff.replace",
                          "Replace current draft",
                        )}
                      </button> : null}
                      </> : null}
                      {sidepanelHandoffConflict.handoff.hasText === false && sidepanelOptionReview.length > 0 ? <button
                        type="button" onClick={applySidepanelOptions} disabled={sidepanelImportPending || !includeSidepanelOptions}
                        className="min-h-11 rounded border border-border bg-surface px-3 py-1 text-xs text-text focus-visible:ring-2 focus-visible:ring-focus">
                        {t("playground:sidepanelHandoff.applySettings", "Apply settings")}
                      </button> : null}
                      <button
                        type="button"
                        onClick={cancelHandoffImport}
                        className="min-h-11 rounded border border-border bg-surface px-3 py-1 text-xs text-text hover:bg-surface3 focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
                      >
                        {sidepanelHandoffConflict.handoff.hasText === false ? t("playground:sidepanelHandoff.dismissCopy", "Dismiss copy") : t("playground:sidepanelHandoff.cancel", "Cancel import")}
                      </button>
                    </div>
                  </div>
                </div>
              ) : null}
              {sidepanelImportReceipt ? (
                <div role="status" className="mb-2 flex flex-wrap items-center justify-between gap-2 rounded-md border border-border bg-surface2 px-3 py-2 text-xs text-text">
                  <p className="min-w-0 flex-1">{sidepanelImportReceipt.description}</p>
                  <button type="button" onClick={dismissSidepanelImportReceipt} className="min-h-7 rounded border border-border px-3 py-1 focus-visible:ring-2 focus-visible:ring-focus">
                    {t("playground:sidepanelHandoff.dismissReceipt", "Dismiss notice")}
                  </button>
                </div>
              ) : null}
              {/* Attachments summary (collapsed context management) */}
              {wrapComposerProfile(
                "attachments-summary",
                <AttachmentsSummary
                  image={form.values.image}
                  imageFilename={form.values.imageFilename}
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
              {uploadedFiles.length > 0
                ? wrapComposerProfile(
                    "document-processing-choices",
                    <DocumentProcessingChoices
                      files={uploadedFiles}
                      onChangeFiles={handleDocumentProcessingFilesChange}
                    />,
                  )
                : null}
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
                      accept=".pdf,.docx,.txt,.csv,.md,.markdown,text/markdown"
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
                      className={`w-full flex flex-col px-0 sm:px-2 ${
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
                        retryLabel={
                          t(
                            "playground:composer.errorBanner.retry",
                            "Retry chat",
                          ) as string
                        }
                        editProviderLabel={
                          t(
                            "playground:composer.errorBanner.editProvider",
                            "Edit provider",
                          ) as string
                        }
                        switchProviderLabel={
                          t(
                            "playground:composer.errorBanner.switchProvider",
                            "Switch provider",
                          ) as string
                        }
                        dismissLabel={t("common:close", "Dismiss") as string}
                        onRetry={handleRetryChatError}
                        onEditProvider={handleEditChatProvider}
                        onSwitchProvider={openModelApiSelector}
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
                        onFileRetrievalChange={(enabled) => {
                          usePlaygroundSessionStore.getState().markSourceSelectionIntent();
                          setFileRetrievalEnabled(enabled);
                        }}
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
                              {isProMode && <button
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
                              </button>}
                              <div
                                className={
                                  !isProMode
                                    ? "min-w-0 flex-1 col-span-2"
                                    : isMobileViewport
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
                                    attachTextarea={attachTextarea}
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
                                    placeholder={t(
                                      "playground:composer.placeholderWithMentions",
                                      "Type a message... (/ commands, @ mentions)",
                                    )}
                                    isProMode={isProMode}
                                    isMobile={isMobileViewport}
                                    isConnectionReady={isConnectionReady}
                                    isCollapsed={isMessageCollapsed}
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
                                    ? "col-span-2 flex shrink-0 items-end justify-end gap-2 self-end"
                                    : "flex shrink-0 items-end gap-2 self-end"
                                }
                              >
                                {promptAssistAction}
                                {sendControl}
                              </div>
                            </div>
                          </div>
                        );

                        const composerInlineMessagesNode = (
                          <>
                            {personaSubmitDisabledReason && <p role="status" className="px-2 py-1 text-sm text-text-muted">{personaSubmitDisabledReason}</p>}
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
                            showDisconnectedNotice={!(needsConnectionSetup && connectionSetupInTimeline)}
                            connectionUxState={connectionUxState}
                            isProMode={isProMode}
                            selectedModel={selectedModel}
                            systemPrompt={systemPrompt}
                            selectedCharacter={selectedCharacter}
                            deferCharacterAppearanceSetup={Boolean(selectedCharacter && getAssistantSelectionMode(activeAssistantSelection) === "tracked" && !historyId && !serverChatId && messages.length === 0)}
                            serverScope={canonicalConnectionConfig?.serverUrl}
                            userScope={
                              canonicalConnectionConfig
                                ? buildChatSurfaceScopeKeyFromConfig(
                                    canonicalConnectionConfig,
                                  )
                                : undefined
                            }
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
                            aria-hidden={
                              keepComposerToolbarVisible
                                ? false
                                : !actionBarVisible
                            }
                            className={`w-full transition-all duration-200 ${
                              !isProMode
                                ? "opacity-100 visible"
                                : keepComposerToolbarVisible
                                  ? "max-h-[480px] overflow-hidden opacity-100 visible"
                                  : `overflow-hidden ${actionBarVisibilityClass}`
                            }`}
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
                                openUIRequestButton={openUIRequestButton}
                                openUIRequestMode={openUIRequestMode}
                                toolChoice={toolChoice}
                                nestedDisclosureOpen={composerOverlayOpen}
                                focusFallbackRef={textareaRef}
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
                                currentProvider={
                                  currentChatModelSettings.apiProvider
                                }
                                resolvedProviderKey={resolvedProviderKey}
                                messages={messages}
                                selectedDocumentsCount={
                                  selectedDocuments.length
                                }
                                uploadedFilesCount={uploadedFiles.length}
                                serverChatId={serverChatId}
                                promptAssistContextKey={
                                  serverChatId
                                    ? `server:${serverChatId}`
                                    : historyId
                                      ? `local:${historyId}`
                                      : "local:playground-draft"
                                }
                                promptAssistBackendKey={promptAssistBackendKey}
                                promptAssistAuthorizationRevision={
                                  promptAssistAuthorizationRevision
                                }
                                showServerPersistenceHint={
                                  showServerPersistenceHint
                                }
                                onDismissServerPersistenceHint={
                                  handleDismissServerPersistenceHint
                                }
                                onFocusConnectionCard={focusConnectionCard}
                                contextItems={contextItems}
                                rolePlayActions={{
                                  onOpenRolePlaySetup: openRolePlaySetup,
                                }}
                                onTrackedAssistantSelection={applyTrackedAssistantSelection}
                                trackedAssistantSelectionLoading={Boolean(historySelection && composerModelsLoading)}
                              />,
                            )}
                          </div>
                        );
                        const composerToolbarSlot =
                          suppressComposerToolbarForMobileCockpit
                            ? (
                                <>
                                  <div
                                    className="flex min-w-0 justify-end"
                                    onClickCapture={(event) => {
                                      rolePlaySetupReturnFocusRef.current =
                                        event.currentTarget.querySelector("button");
                                    }}
                                  >
                                    <BuddyManagementButton
                                      target={serverChatId ? {
                                        scope_type: "conversation",
                                        scope_id: serverChatId,
                                      } : null}
                                      onConversationSettings={openRolePlaySetup}
                                    />
                                  </div>
                                  <div className="hidden">{composerToolbarNode}</div>
                                </>
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
                                stopStreaming={stopStreamingFromUi}
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
                                stopStreaming={stopStreamingFromUi}
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
                                stopStreaming={stopStreamingFromUi}
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
                      <ChatQueuePanel
                        dispatchDisabledReason={getSubmitDisabledReason()}
                        queue={queuedMessages}
                        isConnectionReady={isConnectionReady}
                        isStreaming={isSending || greetingStarting}
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
              referenceImageCandidatesFailed={referenceImageCandidatesFailed}
              onRetryReferenceImages={retryReferenceImageCandidates}
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
              busy={startupTemplateApplying}
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
              onClearPromptContext={() => { clearPromptContext(); }}
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
        {modelSettingsDialog.present && (
          <React.Suspense fallback={null}>
            <LazyCurrentChatModelSettings
              open={openModelSettings}
              setOpen={setOpenModelSettings}
              afterClose={completeModelSettingsClose}
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
              returnFocusRef={rolePlaySetupReturnFocusRef}
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
              captureApplyGuard={captureAssistantActionGuard}
              onSavedSetupDraftNameChange={setStartupTemplateDraftName}
              onSaveRolePlaySetup={handleSaveRolePlaySetup}
              onPreviewSavedSetup={handleOpenStartupTemplatePreview}
              onApplySavedSetup={handleApplySavedRolePlaySetup}
              onSavedSetupApplied={handleSavedRolePlaySetupApplied}
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
