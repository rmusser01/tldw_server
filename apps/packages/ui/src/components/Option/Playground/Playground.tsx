import React from "react";
import { PlaygroundForm } from "./PlaygroundForm";
import { PlaygroundChat } from "./PlaygroundChat";
import {
  PlaygroundCockpitShell,
  type PlaygroundCockpitMode,
} from "./PlaygroundCockpitShell";
import {
  PlaygroundContextRail,
  type PlaygroundContextSource,
  type PlaygroundPromptSummary,
} from "./PlaygroundContextRail";
import {
  PlaygroundRuntimeInspector,
  type RuntimeSettingSummary,
  type RuntimeToolChoice,
} from "./PlaygroundRuntimeInspector";
import { PlaygroundStatusStrip } from "./PlaygroundStatusStrip";
import type { PlaygroundSendBlocker } from "./PlaygroundSendControl";
import {
  CharacterChatReadinessPanel,
  type MissingCharacterRecovery,
} from "./CharacterChatReadinessPanel";
import { CharacterChatSessionsPanel } from "./CharacterChatSessionsPanel";
import {
  buildCockpitAssistantSummary,
  buildCockpitMcpSummary,
  buildCockpitPromptSummary,
  buildCockpitProviderRouteSummary,
  buildCockpitSessionSummary,
} from "./playground-cockpit-summaries";
import {
  openActorSettings,
  openAssistantSelector,
  openMcpSettings,
  openModelSettings,
  openPromptSelector,
  openSearchAndContext,
  setTemporaryChatFromCockpit,
  toggleWebSearchFromCockpit,
} from "./playground-cockpit-actions";
import { getCockpitMessageCount } from "./playground-cockpit-state";
import { buildPlaygroundCompositionPreviewSummary } from "./playground-composition-preview";
import { ChatErrorBoundary } from "@/components/Common/Playground/ChatErrorBoundary";
import { hasVisibleAssistantResponse } from "@/components/Common/Playground/message-visibility";
import { useMessageOption } from "@/hooks/useMessageOption";
import { usePlaygroundSessionPersistence } from "@/hooks/usePlaygroundSessionPersistence";
import { shouldRestorePersistedPlaygroundSession } from "@/hooks/playground-session-restore";
import { webUIResumeLastChat } from "@/services/app";
import {
  formatToChatHistory,
  formatToMessage,
  getHistoryByServerChatId,
  getPromptById,
  getRecentChatFromWebUI,
} from "@/db/dexie/helpers";
import {
  type ChatModelSettings,
  useStoreChatModelSettings,
} from "@/store/model";
import { getDesignSystemState } from "@/design-system";
import { useSmartScroll } from "@/hooks/useSmartScroll";
import { ChevronDown, Keyboard, Search, X } from "lucide-react";
import { CHAT_BACKGROUND_IMAGE_SETTING } from "@/services/settings/ui-settings";
import { otherUnsupportedTypes } from "../Knowledge/utils/unsupported-types";
import { useTranslation } from "react-i18next";
import { useStoreMessageOption } from "@/store/option";
import { useArtifactsStore } from "@/store/artifacts";
import { DEGRADED_STATE_LABEL, READY_STATE_LABEL } from "@/design-system";
import { useSetting } from "@/hooks/useSetting";
import { useStorage } from "@plasmohq/storage/hook";
import { DEFAULT_CHAT_SETTINGS } from "@/types/chat-settings";
import { useMcpToolsStore } from "@/store/mcp-tools";
import { useMobile } from "@/hooks/useMediaQuery";
import { useLoadLocalConversation } from "@/hooks/useLoadLocalConversation";
import { tldwClient } from "@/services/tldw/TldwApiClient";
import { resolvePlaygroundShortcutAction } from "./playground-shortcuts";
import {
  EDIT_MESSAGE_EVENT,
  OPEN_HISTORY_EVENT,
  TIMELINE_ACTION_EVENT,
  type OpenHistoryDetail,
  type TimelineActionDetail,
} from "@/utils/timeline-actions";
import { useCharacterGreeting } from "@/hooks/useCharacterGreeting";
import {
  applyChatSettingsPatch,
  syncChatSettingsForServerChat,
} from "@/services/chat-settings";
import { fetchChatModels } from "@/services/tldw-server";
import {
  buildResearchFollowUpPrompt,
  clearAttachedResearchContext,
  deriveAttachedResearchContext,
  fromPersistedDeepResearchAttachment,
  pinAttachedResearchContext,
  resetAttachedResearchContext,
  restorePinnedResearchContext,
  setAttachedResearchContextActive,
  toPersistedDeepResearchAttachment,
  unpinAttachedResearchContext,
  type AttachedResearchContext,
  type ResearchFollowUpTarget,
} from "./research-chat-context";
import {
  collectThreadSearchMatches,
  getWrappedMatchIndex,
} from "./playground-thread-search";
import {
  RESEARCH_RETURN_RUN_ID_PARAM,
  SETTINGS_HISTORY_ID_PARAM,
  SETTINGS_SERVER_CHAT_ID_PARAM,
} from "@/utils/settings-return";
import { useChatSurfaceCoordinatorStore } from "@/store/chat-surface-coordinator";
import { useLocation, useNavigate } from "react-router-dom";
import {
  resolveComposerBottomOffsetPx,
  type ComposerDockLayoutMetrics,
} from "./mobile-composer-layout";
import { buildPersonaGardenRoute } from "@/utils/persona-garden-route";
import { scheduleFocusFirstVisibleElement } from "@/utils/focus-return";
import {
  CHARACTER_CHAT_MODE_INTENT_EVENT,
  getCharacterChatRouteIntent,
} from "@/utils/character-chat-mode-intent";
import {
  buildChatModelUsability,
  buildCharacterChatReadiness,
  getCharacterChatReadinessCopy,
  getMatchingCharacterChatModelUsabilityCopy,
  type CharacterChatReadinessAction,
} from "@/utils/chat-model-availability";
import type { Character } from "@/types/character";

type ChatModelCatalog = Awaited<ReturnType<typeof fetchChatModels>>;

const toText = (value: unknown): string =>
  typeof value === "string" ? value : String(value);

const isPlaygroundContextSource = (
  item: PlaygroundContextSource | null,
): item is PlaygroundContextSource => Boolean(item);

const UNAVAILABLE_DESIGN_STATE_LABEL =
  getDesignSystemState("unavailable").label;

const LazyArtifactsPanel = React.lazy(() =>
  import("@/components/Sidepanel/Chat/ArtifactsPanel").then((module) => ({
    default: module.ArtifactsPanel,
  })),
);

const renderArtifactsPanel = () => (
  <React.Suspense
    fallback={
      <div className="flex h-full items-center justify-center text-sm text-text-muted">
        Loading artifacts...
      </div>
    }
  >
    <LazyArtifactsPanel />
  </React.Suspense>
);

const SERVER_READINESS_STATE_EVENT = "tldw:server-readiness-state";
const CHAT_WORKFLOW_MODE_STORAGE_KEY = "playgroundChatWorkflowMode";
type PlaygroundChatWorkflowMode = "standard" | "character";
type ServerReadinessState = "ready" | "degraded" | "blocked" | null;
const COCKPIT_ASSISTANT_SELECT_TRIGGER_SELECTOR =
  "[data-cockpit-assistant-select-trigger]";
const COCKPIT_PROMPT_SELECT_TRIGGER_SELECTOR =
  "[data-cockpit-prompt-select-trigger]";
const COCKPIT_MODEL_SETTINGS_TRIGGER_SELECTOR =
  "[data-cockpit-model-settings-trigger]";
const COCKPIT_MCP_SETTINGS_TRIGGER_SELECTOR =
  "[data-cockpit-mcp-settings-trigger]";

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value);

const getRecordString = (value: unknown, keys: string[]): string | null => {
  if (!isRecord(value)) return null;
  for (const key of keys) {
    const fieldValue = value[key];
    if (typeof fieldValue === "string" && fieldValue.trim().length > 0) {
      return fieldValue.trim();
    }
    if (
      (typeof fieldValue === "number" || typeof fieldValue === "bigint") &&
      String(fieldValue).trim().length > 0
    ) {
      return String(fieldValue);
    }
  }
  return null;
};

const normalizeChatWorkflowMode = (
  value: unknown,
): PlaygroundChatWorkflowMode =>
  value === "character" ? "character" : "standard";

export const Playground = () => {
  const drop = React.useRef<HTMLDivElement>(null);
  const artifactsTriggerRef = React.useRef<HTMLButtonElement>(null);
  const threadSearchInputRef = React.useRef<HTMLInputElement>(null);
  const shortcutsTriggerRef = React.useRef<HTMLButtonElement>(null);
  const shortcutsCloseRef = React.useRef<HTMLButtonElement>(null);
  const composerDockRef = React.useRef<HTMLDivElement>(null);
  const [droppedFiles, setDroppedFiles] = React.useState<File[]>([]);
  const [attachedResearchContext, setAttachedResearchContext] =
    React.useState<AttachedResearchContext | null>(null);
  const [attachedResearchContextBaseline, setAttachedResearchContextBaseline] =
    React.useState<AttachedResearchContext | null>(null);
  const [attachedResearchContextPinned, setAttachedResearchContextPinned] =
    React.useState<AttachedResearchContext | null>(null);
  const [attachedResearchContextHistory, setAttachedResearchContextHistory] =
    React.useState<AttachedResearchContext[]>([]);
  const [pendingReturnedResearchRunId, setPendingReturnedResearchRunId] =
    React.useState<string | null>(null);
  const [dismissedReturnedResearchRunId, setDismissedReturnedResearchRunId] =
    React.useState<string | null>(null);
  const [serverDegradedChecks, setServerDegradedChecks] = React.useState<
    string[]
  >([]);
  const [serverReadinessState, setServerReadinessState] =
    React.useState<ServerReadinessState>(null);
  const [routeCharacterRecovery, setRouteCharacterRecovery] =
    React.useState<MissingCharacterRecovery | null>(null);
  const [routeCharacterRetryToken, setRouteCharacterRetryToken] =
    React.useState(0);
  const [characterChatAvailableModels, setCharacterChatAvailableModels] =
    React.useState<ChatModelCatalog | null>(null);
  const [composerDockMetrics, setComposerDockMetrics] =
    React.useState<ComposerDockLayoutMetrics | null>(null);
  const [composerHasDraft, setComposerHasDraft] = React.useState(false);
  const { t } = useTranslation(["playground", "common"]);
  const navigate = useNavigate();
  const [chatBackgroundImage] = useSetting(CHAT_BACKGROUND_IMAGE_SETTING);
  const [stickyChatInput] = useStorage(
    "stickyChatInput",
    DEFAULT_CHAT_SETTINGS.stickyChatInput,
  );
  const isMobileViewport = useMobile();
  const location = useLocation();
  const defaultChatLayoutMode: PlaygroundCockpitMode = isMobileViewport
    ? "focus"
    : "cockpit";
  const [chatWorkflowMode, setChatWorkflowMode] =
    useStorage<PlaygroundChatWorkflowMode>(
      CHAT_WORKFLOW_MODE_STORAGE_KEY,
      "standard",
    );

  const refreshCharacterChatModels = React.useCallback(
    async (isCancelled?: () => boolean) => {
      setCharacterChatAvailableModels(null);
      try {
        const models = await fetchChatModels({
          returnEmpty: true,
          forceRefresh: true,
        });
        if (isCancelled?.()) return;
        setCharacterChatAvailableModels(Array.isArray(models) ? models : []);
      } catch {
        if (isCancelled?.()) return;
        setCharacterChatAvailableModels([]);
      }
    },
    [],
  );

  React.useEffect(() => {
    let cancelled = false;
    void refreshCharacterChatModels(() => cancelled);
    return () => {
      cancelled = true;
    };
  }, [refreshCharacterChatModels]);
  const [characterModeIntentActive, setCharacterModeIntentActive] =
    React.useState(false);
  const [chatLayoutMode, setChatLayoutMode] = useStorage<PlaygroundCockpitMode>(
    "playgroundChatLayoutMode",
    defaultChatLayoutMode,
  );
  const [cockpitContextRailVisible, setCockpitContextRailVisible] =
    useStorage<boolean>("playgroundChatContextRailVisible", true);
  const [cockpitRuntimeRailVisible, setCockpitRuntimeRailVisible] =
    useStorage<boolean>("playgroundChatRuntimeRailVisible", true);
  const [mobileCockpitPanel, setMobileCockpitPanel] = useStorage<
    "context" | "runtime" | null
  >("playgroundChatMobileCockpitPanel", "context");
  const {
    messages,
    history,
    historyId,
    serverChatId,
    serverChatTitle,
    serverChatLoadState,
    serverChatLoadError,
    serverChatState,
    serverChatTopic,
    serverChatSource,
    isLoading,
    selectedModel,
    setHistoryId,
    setHistory,
    setMessages,
    selectedQuickPrompt,
    setSelectedQuickPrompt,
    selectedSystemPrompt,
    setSelectedSystemPrompt,
    setSelectedModel,
    setServerChatId,
    contextFiles,
    setContextFiles,
    createChatBranch,
    streaming,
    isProcessing,
    selectedCharacter,
    setSelectedCharacter,
    compareMode,
    compareFeatureEnabled,
    temporaryChat,
    webSearch,
    toolChoice,
    setToolChoice,
    selectedKnowledge,
    setSelectedKnowledge,
    ragMediaIds,
    setRagMediaIds,
    stopStreamingRequest,
    regenerateLastMessage,
    selectedAssistant,
    setSelectedAssistant,
    serverChatPersonaMemoryMode,
  } = useMessageOption();
  const {
    systemPrompt,
    setSystemPrompt,
    temperature,
    topP,
    topK,
    numCtx,
    numPredict,
    reasoningEffort,
    apiProvider,
    activeSettingsScope,
    setActiveSettingsScope,
    scopedSettingsByModelKey,
  } = useStoreChatModelSettings();
  const [selectedSystemPromptRecord, setSelectedSystemPromptRecord] =
    React.useState<{ id?: string; title?: string; name?: string } | null>(null);
  const [selectedSystemPromptStatus, setSelectedSystemPromptStatus] =
    React.useState<"idle" | "loading" | "loaded" | "unavailable">("idle");
  const mcpHealthState = useMcpToolsStore((state) => state.healthState);
  const mcpToolsLoading = useMcpToolsStore((state) => state.toolsLoading);
  const discoveredMcpToolCount = useMcpToolsStore(
    (state) => state.discoveredTools.length,
  );
  const chatMcpToolCount = useMcpToolsStore((state) => state.chatTools.length);
  const mcpToolCounts = useMcpToolsStore((state) => state.toolCounts);

  React.useEffect(() => {
    let cancelled = false;
    const promptId = String(selectedSystemPrompt || "").trim();
    if (!promptId) {
      setSelectedSystemPromptRecord(null);
      setSelectedSystemPromptStatus("idle");
      return;
    }

    setSelectedSystemPromptStatus("loading");
    void getPromptById(promptId)
      .then((prompt) => {
        if (cancelled) return;
        setSelectedSystemPromptRecord(
          prompt
            ? {
                id: String(prompt.id || promptId),
                title:
                  typeof prompt.title === "string" ? prompt.title : undefined,
                name: typeof prompt.name === "string" ? prompt.name : undefined,
              }
            : null,
        );
        setSelectedSystemPromptStatus(prompt ? "loaded" : "unavailable");
      })
      .catch((error) => {
        if (cancelled) return;
        console.warn("[Playground] Failed to resolve selected system prompt", {
          promptId,
          error,
        });
        setSelectedSystemPromptRecord(null);
        setSelectedSystemPromptStatus("unavailable");
      });

    return () => {
      cancelled = true;
    };
  }, [selectedSystemPrompt]);
  const composerBottomOffsetPx = stickyChatInput
    ? resolveComposerBottomOffsetPx(composerDockMetrics)
    : 0;
  const handleComposerLayoutChange = React.useCallback(
    (metrics: ComposerDockLayoutMetrics) => {
      if (metrics.occupiedHeightPx === 0 && metrics.keyboardInsetPx === 0) {
        setComposerDockMetrics(null);
        return;
      }

      const dockEl = composerDockRef.current;
      setComposerDockMetrics({
        occupiedHeightPx: dockEl
          ? Math.round(dockEl.getBoundingClientRect().height)
          : metrics.occupiedHeightPx,
        keyboardInsetPx: metrics.keyboardInsetPx,
      });
    },
    [],
  );
  const handleComposerDraftPresenceChange = React.useCallback(
    (hasDraft: boolean) => {
      setComposerHasDraft(hasDraft);
    },
    [],
  );
  const { containerRef, isAutoScrollToBottom, autoScrollToBottom } =
    useSmartScroll(messages, streaming, 120, {
      bottomOffsetPx: composerBottomOffsetPx,
    });

  const [dropState, setDropState] = React.useState<
    "idle" | "dragging" | "error"
  >("idle");
  const [threadSearchOpen, setThreadSearchOpen] = React.useState(false);
  const [threadSearchQuery, setThreadSearchQuery] = React.useState("");
  const [debouncedSearchQuery, setDebouncedSearchQuery] = React.useState("");
  const [threadSearchActiveIndex, setThreadSearchActiveIndex] =
    React.useState(0);
  const [shortcutsHelpOpen, setShortcutsHelpOpen] = React.useState(false);
  const [dropFeedback, setDropFeedback] = React.useState<{
    type: "info" | "error" | "warning";
    message: string;
  } | null>(null);
  const [playgroundReady, setPlaygroundReady] = React.useState(false);
  const feedbackTimerRef = React.useRef<ReturnType<typeof setTimeout> | null>(
    null,
  );
  const timelineActionRetryTimeoutRef = React.useRef<ReturnType<
    typeof setTimeout
  > | null>(null);
  const initializePlaygroundRef = React.useRef(false);
  const routeCharacterIntentAppliedRef = React.useRef<string | null>(null);
  const routeCharacterIntentInFlightRef = React.useRef<string | null>(null);
  const routeCharacterIntentRequestRef = React.useRef(0);
  const translationRef = React.useRef(t);
  const previousThreadRef = React.useRef<string | null>(null);
  const stableHistoryId = historyId && historyId !== "temp" ? historyId : null;
  const showStarterDeck =
    messages.length === 0 &&
    history.length === 0 &&
    !stableHistoryId &&
    !serverChatId &&
    !composerHasDraft;
  const routeCharacterIntent = React.useMemo(
    () => getCharacterChatRouteIntent(location.search),
    [location.search],
  );
  const routeCharacterIntentId = routeCharacterIntent?.characterId ?? null;
  const routeRequestsCharacterMode = Boolean(routeCharacterIntent);
  const normalizedChatWorkflowMode =
    normalizeChatWorkflowMode(chatWorkflowMode);
  const characterWorkflowActive =
    routeRequestsCharacterMode ||
    characterModeIntentActive ||
    normalizedChatWorkflowMode === "character" ||
    selectedAssistant?.kind === "character" ||
    Boolean(selectedCharacter?.id);
  const activeCharacterModeLabel =
    selectedAssistant?.kind === "character"
      ? selectedAssistant.name
      : selectedAssistant
        ? null
        : (selectedCharacter?.name ?? null);
  const setRouteContext = useChatSurfaceCoordinatorStore(
    (state) => state.setRouteContext,
  );
  const normalizedChatLayoutMode: PlaygroundCockpitMode =
    chatLayoutMode === "focus" || chatLayoutMode === "cockpit"
      ? chatLayoutMode
      : defaultChatLayoutMode;
  const normalizedCockpitContextRailVisible =
    cockpitContextRailVisible !== false;
  const normalizedCockpitRuntimeRailVisible =
    cockpitRuntimeRailVisible !== false;
  const handleChatLayoutModeChange = React.useCallback(
    (mode: PlaygroundCockpitMode) => {
      if (
        mode === "cockpit" &&
        !normalizedCockpitContextRailVisible &&
        !normalizedCockpitRuntimeRailVisible
      ) {
        void setCockpitContextRailVisible(true);
        void setCockpitRuntimeRailVisible(true);
      }
      void setChatLayoutMode(mode);
    },
    [
      normalizedCockpitContextRailVisible,
      normalizedCockpitRuntimeRailVisible,
      setChatLayoutMode,
      setCockpitContextRailVisible,
      setCockpitRuntimeRailVisible,
    ],
  );

  React.useEffect(() => {
    setRouteContext({ routeId: "chat", surface: "webui" });
  }, [setRouteContext]);

  React.useEffect(() => {
    translationRef.current = t;
  }, [t]);

  React.useEffect(() => {
    if (typeof window === "undefined") return;
    const handleCharacterModeIntent = () => {
      setCharacterModeIntentActive(true);
      void setChatWorkflowMode("character");
    };
    const handleStarterSelected = (event: Event) => {
      const detail = (event as CustomEvent<{ mode?: unknown }>).detail;
      const mode = String(detail?.mode || "")
        .trim()
        .toLowerCase();
      if (mode === "character") {
        setCharacterModeIntentActive(true);
        void setChatWorkflowMode("character");
        return;
      }
      if (mode) {
        setCharacterModeIntentActive(false);
        void setChatWorkflowMode("standard");
      }
    };
    window.addEventListener(
      CHARACTER_CHAT_MODE_INTENT_EVENT,
      handleCharacterModeIntent as EventListener,
    );
    window.addEventListener(
      "tldw:playground-starter-selected",
      handleStarterSelected as EventListener,
    );
    return () => {
      window.removeEventListener(
        CHARACTER_CHAT_MODE_INTENT_EVENT,
        handleCharacterModeIntent as EventListener,
      );
      window.removeEventListener(
        "tldw:playground-starter-selected",
        handleStarterSelected as EventListener,
      );
    };
  }, [setChatWorkflowMode]);

  React.useEffect(() => {
    if (!routeRequestsCharacterMode) return;
    setCharacterModeIntentActive(true);
    void setChatWorkflowMode("character");
  }, [routeRequestsCharacterMode, setChatWorkflowMode]);

  React.useEffect(() => {
    if (!routeCharacterIntentId) return;
    if (routeCharacterIntentAppliedRef.current === routeCharacterIntentId) {
      return;
    }

    const requestId = routeCharacterIntentRequestRef.current + 1;
    routeCharacterIntentRequestRef.current = requestId;
    routeCharacterIntentInFlightRef.current = routeCharacterIntentId;
    const fallbackCharacterName = toText(
      translationRef.current(
        "playground:characterChat.characterFallback",
        "Character {{id}}",
        {
          id: routeCharacterIntentId,
        },
      ),
    ).replace("{{id}}", routeCharacterIntentId);
    const fallbackCharacter: Character = {
      id: routeCharacterIntentId,
      name: fallbackCharacterName,
    };
    void tldwClient
      .getCharacter(routeCharacterIntentId)
      .then((character) => {
        if (routeCharacterIntentRequestRef.current !== requestId) return;
        routeCharacterIntentAppliedRef.current = routeCharacterIntentId;
        if (character) {
          setRouteCharacterRecovery(null);
          void setSelectedCharacter(character);
          return;
        }
        setRouteCharacterRecovery({
          id: routeCharacterIntentId,
          reason: "missing",
        });
        void setSelectedCharacter(fallbackCharacter);
      })
      .catch(() => {
        if (routeCharacterIntentRequestRef.current !== requestId) return;
        routeCharacterIntentAppliedRef.current = routeCharacterIntentId;
        setRouteCharacterRecovery({
          id: routeCharacterIntentId,
          reason: "load-error",
        });
        void setSelectedCharacter(fallbackCharacter);
      })
      .finally(() => {
        if (routeCharacterIntentRequestRef.current !== requestId) return;
        routeCharacterIntentInFlightRef.current = null;
      });
    return () => {
      if (routeCharacterIntentRequestRef.current === requestId) {
        routeCharacterIntentRequestRef.current += 1;
      }
    };
  }, [routeCharacterIntentId, routeCharacterRetryToken, setSelectedCharacter]);

  React.useEffect(() => {
    if (!routeCharacterIntentId) {
      routeCharacterIntentAppliedRef.current = null;
      routeCharacterIntentInFlightRef.current = null;
      setRouteCharacterRecovery(null);
    }
  }, [routeCharacterIntentId]);

  React.useEffect(() => {
    if (!routeCharacterRecovery || !selectedCharacter?.id) return;
    if (routeCharacterIntentInFlightRef.current === routeCharacterRecovery.id) {
      return;
    }
    if (String(selectedCharacter.id) !== routeCharacterRecovery.id) {
      setRouteCharacterRecovery(null);
    }
  }, [routeCharacterRecovery, selectedCharacter?.id]);

  React.useEffect(() => {
    const handleReadinessState = (event: Event) => {
      const detail = (
        event as CustomEvent<{
          state?: string;
          degradedChecks?: unknown;
        }>
      ).detail;
      const nextState =
        detail?.state === "ready" ||
        detail?.state === "degraded" ||
        detail?.state === "blocked"
          ? detail.state
          : null;
      setServerReadinessState(nextState);
      if (nextState !== "degraded") {
        setServerDegradedChecks([]);
        return;
      }
      const checks = Array.isArray(detail.degradedChecks)
        ? detail.degradedChecks
            .map((check) => (typeof check === "string" ? check.trim() : ""))
            .filter((check) => check.length > 0)
        : [];
      setServerDegradedChecks(checks);
    };

    window.addEventListener(SERVER_READINESS_STATE_EVENT, handleReadinessState);
    return () => {
      window.removeEventListener(
        SERVER_READINESS_STATE_EVENT,
        handleReadinessState,
      );
    };
  }, []);

  // Debounce search query to avoid running collectThreadSearchMatches on every keystroke
  React.useEffect(() => {
    const timer = setTimeout(
      () => setDebouncedSearchQuery(threadSearchQuery),
      200,
    );
    return () => clearTimeout(timer);
  }, [threadSearchQuery]);

  const showDropFeedback = React.useCallback(
    (feedback: { type: "info" | "error" | "warning"; message: string }) => {
      setDropFeedback(feedback);
      if (feedbackTimerRef.current) {
        clearTimeout(feedbackTimerRef.current);
      }
      feedbackTimerRef.current = setTimeout(() => {
        setDropFeedback(null);
        feedbackTimerRef.current = null;
      }, 6000);
    },
    [],
  );

  React.useEffect(() => {
    if (!drop.current) {
      return;
    }
    const handleDragOver = (e: DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
    };

    const handleDrop = (e: DragEvent) => {
      e.preventDefault();
      e.stopPropagation();

      setDropState("idle");

      const files = Array.from(e.dataTransfer?.files || []);

      const hasUnsupportedFiles = files.some((file) =>
        otherUnsupportedTypes.includes(file.type),
      );

      if (hasUnsupportedFiles) {
        setDropState("error");
        showDropFeedback({
          type: "error",
          message: t(
            "playground:drop.unsupported",
            "That file type isn’t supported. Try images or text-based files.",
          ),
        });
        return;
      }

      const FILE_LIMIT = 5;
      const allFiles = Array.from(e.dataTransfer?.files || []).filter(
        (file) => !otherUnsupportedTypes.includes(file.type),
      );
      const newFiles = allFiles.slice(0, FILE_LIMIT);
      const droppedExtra = allFiles.length - newFiles.length;

      if (newFiles.length > 0) {
        setDroppedFiles(newFiles);

        // Show warning if files were truncated
        if (droppedExtra > 0) {
          showDropFeedback({
            type: "warning",
            message: t("playground:drop.limitWarning", {
              count: newFiles.length,
              extra: droppedExtra,
              limit: FILE_LIMIT,
              defaultValue: `Attached first ${newFiles.length} files. ${droppedExtra} additional file(s) were not attached (limit: ${FILE_LIMIT}).`,
            }),
          });
        } else {
          showDropFeedback({
            type: "info",
            message:
              newFiles.length > 1
                ? t("playground:drop.readyMultiple", {
                    count: newFiles.length,
                  })
                : t("playground:drop.readySingle", {
                    name:
                      newFiles[0]?.name ||
                      t("playground:drop.defaultFileName", "File"),
                  }),
          });
        }
      }
    };
    const handleDragEnter = (e: DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setDropState("dragging");
      showDropFeedback({
        type: "info",
        message: t(
          "playground:drop.hint",
          "Drop files to attach them to your message",
        ),
      });
    };

    const handleDragLeave = (e: DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setDropState("idle");
    };

    drop.current.addEventListener("dragover", handleDragOver);
    drop.current.addEventListener("drop", handleDrop);
    drop.current.addEventListener("dragenter", handleDragEnter);
    drop.current.addEventListener("dragleave", handleDragLeave);

    return () => {
      if (drop.current) {
        drop.current.removeEventListener("dragover", handleDragOver);
        drop.current.removeEventListener("drop", handleDrop);
        drop.current.removeEventListener("dragenter", handleDragEnter);
        drop.current.removeEventListener("dragleave", handleDragLeave);
      }
    };
  }, [showDropFeedback, t]);

  React.useEffect(() => {
    return () => {
      if (feedbackTimerRef.current) {
        clearTimeout(feedbackTimerRef.current);
      }
      if (timelineActionRetryTimeoutRef.current) {
        clearTimeout(timelineActionRetryTimeoutRef.current);
      }
      pendingTimelineActionRef.current = null;
    };
  }, []);

  React.useEffect(() => {
    const currentThreadKey = `${serverChatId ?? ""}::${historyId ?? ""}`;
    if (
      previousThreadRef.current !== null &&
      previousThreadRef.current !== currentThreadKey
    ) {
      setAttachedResearchContext(null);
      setAttachedResearchContextBaseline(null);
      setAttachedResearchContextPinned(null);
      setAttachedResearchContextHistory([]);
    }
    previousThreadRef.current = currentThreadKey;
  }, [historyId, serverChatId]);

  const persistAttachedResearchContext = React.useCallback(
    async (
      context: AttachedResearchContext | null,
      pinned: AttachedResearchContext | null,
      history: AttachedResearchContext[],
    ) => {
      if (!serverChatId || !stableHistoryId) {
        return;
      }
      try {
        await applyChatSettingsPatch({
          historyId: stableHistoryId,
          serverChatId,
          patch: {
            deepResearchAttachment: context
              ? toPersistedDeepResearchAttachment(context)
              : null,
            deepResearchPinnedAttachment: pinned
              ? toPersistedDeepResearchAttachment(pinned)
              : null,
            deepResearchAttachmentHistory: history.map((entry) =>
              toPersistedDeepResearchAttachment(entry, entry.attached_at),
            ),
          },
        });
      } catch {
        // Attachment persistence is best-effort and should never block chat use.
      }
    },
    [serverChatId, stableHistoryId],
  );

  React.useEffect(() => {
    if (!playgroundReady || !serverChatId || !stableHistoryId) {
      return;
    }
    let cancelled = false;
    const threadKey = `${serverChatId}::${stableHistoryId}`;

    const restorePersistedAttachment = async () => {
      try {
        const settings = await syncChatSettingsForServerChat({
          historyId: stableHistoryId,
          serverChatId,
        });
        if (cancelled || previousThreadRef.current !== threadKey) {
          return;
        }
        const restoredAttachment = settings?.deepResearchAttachment
          ? fromPersistedDeepResearchAttachment(settings.deepResearchAttachment)
          : null;
        const restoredPinnedAttachment = settings?.deepResearchPinnedAttachment
          ? fromPersistedDeepResearchAttachment(
              settings.deepResearchPinnedAttachment,
            )
          : null;
        const restoredHistory = Array.isArray(
          settings?.deepResearchAttachmentHistory,
        )
          ? settings.deepResearchAttachmentHistory.map(
              fromPersistedDeepResearchAttachment,
            )
          : [];
        const restoredActive = restoredAttachment ?? restoredPinnedAttachment;
        setAttachedResearchContext((current) => current ?? restoredActive);
        setAttachedResearchContextBaseline(
          (current) => current ?? restoredActive,
        );
        setAttachedResearchContextPinned(
          (current) => current ?? restoredPinnedAttachment,
        );
        setAttachedResearchContextHistory((current) =>
          current.length > 0 ? current : restoredHistory,
        );
      } catch {
        // Silent, non-blocking auxiliary restore.
      }
    };

    void restorePersistedAttachment();

    return () => {
      cancelled = true;
    };
  }, [playgroundReady, serverChatId, stableHistoryId]);

  const handleAttachResearchContext = React.useCallback(
    (context: AttachedResearchContext) => {
      const nextState = setAttachedResearchContextActive({
        active: attachedResearchContext,
        baseline: attachedResearchContextBaseline,
        pinned: attachedResearchContextPinned,
        history: attachedResearchContextHistory,
        nextActive: context,
      });
      setAttachedResearchContext(nextState.active);
      setAttachedResearchContextBaseline(nextState.baseline);
      setAttachedResearchContextPinned(nextState.pinned);
      setAttachedResearchContextHistory(nextState.history);
      void persistAttachedResearchContext(
        nextState.active,
        nextState.pinned,
        nextState.history,
      );
    },
    [
      attachedResearchContext,
      attachedResearchContextBaseline,
      attachedResearchContextPinned,
      attachedResearchContextHistory,
      persistAttachedResearchContext,
    ],
  );

  const handlePrepareResearchFollowUp = React.useCallback(
    async (target: ResearchFollowUpTarget) => {
      if (attachedResearchContext?.run_id !== target.run_id) {
        try {
          await tldwClient.initialize().catch(() => null);
          const bundle = await tldwClient.getResearchBundle(target.run_id);
          handleAttachResearchContext(
            deriveAttachedResearchContext(bundle, target.run_id, target.query),
          );
        } catch {
          // Keep prompt preparation available even if bundle reload fails.
        }
      }

      setSelectedQuickPrompt(buildResearchFollowUpPrompt(target.query));
    },
    [
      attachedResearchContext?.run_id,
      handleAttachResearchContext,
      setSelectedQuickPrompt,
    ],
  );

  const handleApplyAttachedResearchContext = React.useCallback(
    (context: AttachedResearchContext) => {
      setAttachedResearchContext(context);
      void persistAttachedResearchContext(
        context,
        attachedResearchContextPinned,
        attachedResearchContextHistory,
      );
    },
    [
      attachedResearchContextHistory,
      attachedResearchContextPinned,
      persistAttachedResearchContext,
    ],
  );

  const handleResetAttachedResearchContext = React.useCallback(() => {
    const resetContext = resetAttachedResearchContext(
      attachedResearchContextBaseline,
    );
    setAttachedResearchContext(resetContext);
    void persistAttachedResearchContext(
      resetContext,
      attachedResearchContextPinned,
      attachedResearchContextHistory,
    );
  }, [
    attachedResearchContextBaseline,
    attachedResearchContextHistory,
    attachedResearchContextPinned,
    persistAttachedResearchContext,
  ]);

  const handleRemoveAttachedResearchContext = React.useCallback(() => {
    const cleared = clearAttachedResearchContext({
      active: attachedResearchContext,
      baseline: attachedResearchContextBaseline,
      pinned: attachedResearchContextPinned,
      history: attachedResearchContextHistory,
    });
    setAttachedResearchContext(cleared.active);
    setAttachedResearchContextBaseline(cleared.baseline);
    setAttachedResearchContextPinned(cleared.pinned);
    setAttachedResearchContextHistory(cleared.history);
    void persistAttachedResearchContext(
      cleared.active,
      cleared.pinned,
      cleared.history,
    );
  }, [
    attachedResearchContext,
    attachedResearchContextBaseline,
    attachedResearchContextPinned,
    attachedResearchContextHistory,
    persistAttachedResearchContext,
  ]);

  const handleSelectAttachedResearchContextHistory = React.useCallback(
    (context: AttachedResearchContext) => {
      const nextState = setAttachedResearchContextActive({
        active: attachedResearchContext,
        baseline: attachedResearchContextBaseline,
        pinned: attachedResearchContextPinned,
        history: attachedResearchContextHistory,
        nextActive: context,
      });
      setAttachedResearchContext(nextState.active);
      setAttachedResearchContextBaseline(nextState.baseline);
      setAttachedResearchContextPinned(nextState.pinned);
      setAttachedResearchContextHistory(nextState.history);
      void persistAttachedResearchContext(
        nextState.active,
        nextState.pinned,
        nextState.history,
      );
    },
    [
      attachedResearchContext,
      attachedResearchContextBaseline,
      attachedResearchContextPinned,
      attachedResearchContextHistory,
      persistAttachedResearchContext,
    ],
  );

  const handlePinAttachedResearchContext = React.useCallback(() => {
    const nextState = pinAttachedResearchContext({
      active: attachedResearchContext,
      baseline: attachedResearchContextBaseline,
      pinned: attachedResearchContextPinned,
      history: attachedResearchContextHistory,
    });
    setAttachedResearchContext(nextState.active);
    setAttachedResearchContextBaseline(nextState.baseline);
    setAttachedResearchContextPinned(nextState.pinned);
    setAttachedResearchContextHistory(nextState.history);
    void persistAttachedResearchContext(
      nextState.active,
      nextState.pinned,
      nextState.history,
    );
  }, [
    attachedResearchContext,
    attachedResearchContextBaseline,
    attachedResearchContextPinned,
    attachedResearchContextHistory,
    persistAttachedResearchContext,
  ]);

  const handlePinAttachedResearchContextHistory = React.useCallback(
    (context: AttachedResearchContext) => {
      const nextState = pinAttachedResearchContext({
        active: attachedResearchContext,
        baseline: attachedResearchContextBaseline,
        pinned: attachedResearchContextPinned,
        history: attachedResearchContextHistory,
        nextPinned: context,
      });
      setAttachedResearchContext(nextState.active);
      setAttachedResearchContextBaseline(nextState.baseline);
      setAttachedResearchContextPinned(nextState.pinned);
      setAttachedResearchContextHistory(nextState.history);
      void persistAttachedResearchContext(
        nextState.active,
        nextState.pinned,
        nextState.history,
      );
    },
    [
      attachedResearchContext,
      attachedResearchContextBaseline,
      attachedResearchContextPinned,
      attachedResearchContextHistory,
      persistAttachedResearchContext,
    ],
  );

  const handleUnpinAttachedResearchContext = React.useCallback(() => {
    const nextState = unpinAttachedResearchContext({
      active: attachedResearchContext,
      baseline: attachedResearchContextBaseline,
      pinned: attachedResearchContextPinned,
      history: attachedResearchContextHistory,
    });
    setAttachedResearchContext(nextState.active);
    setAttachedResearchContextBaseline(nextState.baseline);
    setAttachedResearchContextPinned(nextState.pinned);
    setAttachedResearchContextHistory(nextState.history);
    void persistAttachedResearchContext(
      nextState.active,
      nextState.pinned,
      nextState.history,
    );
  }, [
    attachedResearchContext,
    attachedResearchContextBaseline,
    attachedResearchContextPinned,
    attachedResearchContextHistory,
    persistAttachedResearchContext,
  ]);

  const handleRestorePinnedResearchContext = React.useCallback(() => {
    const nextState = restorePinnedResearchContext({
      active: attachedResearchContext,
      baseline: attachedResearchContextBaseline,
      pinned: attachedResearchContextPinned,
      history: attachedResearchContextHistory,
    });
    setAttachedResearchContext(nextState.active);
    setAttachedResearchContextBaseline(nextState.baseline);
    setAttachedResearchContextPinned(nextState.pinned);
    setAttachedResearchContextHistory(nextState.history);
    void persistAttachedResearchContext(
      nextState.active,
      nextState.pinned,
      nextState.history,
    );
  }, [
    attachedResearchContext,
    attachedResearchContextBaseline,
    attachedResearchContextPinned,
    attachedResearchContextHistory,
    persistAttachedResearchContext,
  ]);

  // Session persistence for draft restoration
  const {
    restoreSession,
    sessionScopeReady,
    hasPersistedSession,
    persistedHistoryId,
    persistedServerChatId,
  } = usePlaygroundSessionPersistence();

  const initializePlayground = React.useCallback(async () => {
    // 1. Try session persistence first (restores exact state from nav-away)
    const shouldRestorePersistedSession =
      shouldRestorePersistedPlaygroundSession({
        hasPersistedSession,
        persistedHistoryId,
        persistedServerChatId,
        currentHistoryId: historyId ?? null,
        currentServerChatId: serverChatId ?? null,
        currentMessagesLength: messages.length,
        currentHistoryLength: history.length,
      });

    if (shouldRestorePersistedSession) {
      const restored = await restoreSession();
      if (restored) return;
    }

    // 2. Fall back to existing webUIResumeLastChat behavior
    const isEnabled = await webUIResumeLastChat();
    if (!isEnabled) return;

    if (messages.length === 0 && history.length === 0) {
      const recentChat = await getRecentChatFromWebUI();
      if (recentChat) {
        setHistoryId(recentChat.history.id);
        setHistory(formatToChatHistory(recentChat.messages));
        setMessages(formatToMessage(recentChat.messages));

        const lastUsedPrompt = recentChat?.history?.last_used_prompt;
        if (lastUsedPrompt) {
          if (lastUsedPrompt.prompt_id) {
            const prompt = await getPromptById(lastUsedPrompt.prompt_id);
            if (prompt) {
              setSelectedSystemPrompt(lastUsedPrompt.prompt_id);
              if (!lastUsedPrompt.prompt_content?.trim()) {
                setSystemPrompt(prompt.content);
              }
            }
          }
          if (lastUsedPrompt.prompt_content?.trim()) {
            setSystemPrompt(lastUsedPrompt.prompt_content);
          }
        }
      }
    }
  }, [
    history.length,
    historyId,
    hasPersistedSession,
    messages.length,
    persistedHistoryId,
    persistedServerChatId,
    restoreSession,
    serverChatId,
    setHistory,
    setHistoryId,
    setMessages,
    setSelectedSystemPrompt,
    setSystemPrompt,
  ]);

  React.useEffect(() => {
    if (!sessionScopeReady) {
      return;
    }
    if (initializePlaygroundRef.current) {
      return;
    }
    initializePlaygroundRef.current = true;
    let cancelled = false;
    const run = async () => {
      await initializePlayground();
      if (!cancelled) {
        setPlaygroundReady(true);
      }
    };
    void run();
    return () => {
      cancelled = true;
    };
  }, [initializePlayground, sessionScopeReady]);

  useCharacterGreeting({
    playgroundReady,
    selectedCharacter,
    serverChatId,
    historyId,
    messagesLength: messages.length,
    setMessages,
    setHistory,
    setSelectedCharacter,
  });

  const loadLocalConversation = useLoadLocalConversation(
    {
      setServerChatId,
      setHistoryId: (id) => setHistoryId(id, { preserveServerChatId: false }),
      setHistory,
      setMessages,
      setSelectedModel: (id) => setSelectedModel(id),
      setSelectedSystemPrompt: (id) => {
        if (id) {
          setSelectedSystemPrompt(id);
        }
      },
      setSystemPrompt,
      setContextFiles,
    },
    {
      t,
      errorLogPrefix: t(
        "playground:errors.loadLocalHistoryPrefix",
        "Failed to load local chat history",
      ),
      errorDefaultMessage: t(
        "playground:errors.loadLocalHistoryDefault",
        "Something went wrong while loading local chat history.",
      ),
    },
  );

  const settingsReturnContext = React.useMemo(() => {
    if (typeof window === "undefined") {
      return {
        historyId: null as string | null,
        serverChatId: null as string | null,
        researchReturnRunId: null as string | null,
      };
    }
    const params = new URLSearchParams(window.location.search);
    const historyId = params.get(SETTINGS_HISTORY_ID_PARAM)?.trim() || null;
    const serverChatId =
      params.get(SETTINGS_SERVER_CHAT_ID_PARAM)?.trim() || null;
    const researchReturnRunId =
      params.get(RESEARCH_RETURN_RUN_ID_PARAM)?.trim() || null;
    return { historyId, serverChatId, researchReturnRunId };
  }, []);

  const returnHistoryIdFromSettings = settingsReturnContext.historyId;
  const returnServerChatIdFromSettings = settingsReturnContext.serverChatId;
  const returnResearchRunIdFromSettings =
    settingsReturnContext.researchReturnRunId;

  React.useEffect(() => {
    if (!playgroundReady) return;
    if (
      !returnHistoryIdFromSettings &&
      !returnServerChatIdFromSettings &&
      !returnResearchRunIdFromSettings
    ) {
      return;
    }

    let cancelled = false;

    const restoreFromSettingsReturnTarget = async () => {
      if (
        returnHistoryIdFromSettings &&
        returnHistoryIdFromSettings !== historyId
      ) {
        await loadLocalConversation(returnHistoryIdFromSettings);
      } else if (
        !returnHistoryIdFromSettings &&
        returnServerChatIdFromSettings &&
        returnServerChatIdFromSettings !== serverChatId
      ) {
        const existingHistory = await getHistoryByServerChatId(
          returnServerChatIdFromSettings,
        );
        const fallbackHistoryId =
          existingHistory?.id && existingHistory.id.trim().length > 0
            ? existingHistory.id
            : null;
        if (fallbackHistoryId) {
          await loadLocalConversation(fallbackHistoryId);
        }
      }

      if (cancelled) return;

      if (
        returnServerChatIdFromSettings &&
        returnServerChatIdFromSettings !== serverChatId
      ) {
        setServerChatId(returnServerChatIdFromSettings);
      }

      if (
        returnResearchRunIdFromSettings &&
        returnResearchRunIdFromSettings !== dismissedReturnedResearchRunId
      ) {
        setPendingReturnedResearchRunId(returnResearchRunIdFromSettings);
      }

      if (typeof window !== "undefined") {
        const url = new URL(window.location.href);
        url.searchParams.delete(SETTINGS_HISTORY_ID_PARAM);
        url.searchParams.delete(SETTINGS_SERVER_CHAT_ID_PARAM);
        url.searchParams.delete(RESEARCH_RETURN_RUN_ID_PARAM);
        const nextQuery = url.searchParams.toString();
        const nextPath = `${url.pathname}${nextQuery ? `?${nextQuery}` : ""}${url.hash}`;
        window.history.replaceState(window.history.state, "", nextPath);
      }
    };

    void restoreFromSettingsReturnTarget();

    return () => {
      cancelled = true;
    };
  }, [
    historyId,
    loadLocalConversation,
    playgroundReady,
    returnHistoryIdFromSettings,
    returnResearchRunIdFromSettings,
    returnServerChatIdFromSettings,
    serverChatId,
    setServerChatId,
    dismissedReturnedResearchRunId,
  ]);

  const pendingTimelineActionRef = React.useRef<TimelineActionDetail | null>(
    null,
  );
  const threadSearchMatches = React.useMemo(
    () => collectThreadSearchMatches(messages, debouncedSearchQuery),
    [messages, debouncedSearchQuery],
  );
  const threadSearchMatchSet = React.useMemo(
    () => new Set(threadSearchMatches),
    [threadSearchMatches],
  );
  const threadSearchActiveMessageIndex =
    threadSearchMatches.length > 0
      ? threadSearchMatches[
          Math.max(
            0,
            Math.min(threadSearchActiveIndex, threadSearchMatches.length - 1),
          )
        ]
      : null;

  const findMessageIndex = React.useCallback(
    (messageId: string) =>
      messages.findIndex(
        (message) =>
          message.id === messageId || message.serverMessageId === messageId,
      ),
    [messages],
  );

  const scrollToMessage = React.useCallback(
    (messageId: string) => {
      const container = containerRef.current;
      if (!container) return false;
      const target = container.querySelector<HTMLElement>(
        `[data-message-id="${messageId}"], [data-server-message-id="${messageId}"]`,
      );
      if (!target) return false;
      target.scrollIntoView({ block: "center", behavior: "smooth" });
      return true;
    },
    [containerRef],
  );
  const scrollToMessageIndex = React.useCallback(
    (index: number) => {
      const container = containerRef.current;
      if (!container) return false;
      const target = container.querySelector<HTMLElement>(
        `[data-index="${index}"]`,
      );
      if (!target) return false;
      target.scrollIntoView({ block: "center", behavior: "smooth" });
      return true;
    },
    [containerRef],
  );

  const dispatchEditMessage = React.useCallback((messageId: string) => {
    if (typeof window === "undefined") return;
    window.dispatchEvent(
      new CustomEvent(EDIT_MESSAGE_EVENT, { detail: { messageId } }),
    );
  }, []);

  const performTimelineAction = React.useCallback(
    (detail: TimelineActionDetail) => {
      if (!detail?.historyId) return true;
      if (detail.historyId !== historyId) return false;

      if (detail.action === "branch") {
        if (!detail.messageId) return true;
        if (messages.length === 0) return false;
        const index = findMessageIndex(detail.messageId);
        if (index < 0) return true;
        void createChatBranch(index);
        return true;
      }

      if (!detail.messageId) return true;

      const scrolled = scrollToMessage(detail.messageId);
      if (!scrolled) {
        if (!containerRef.current) return false;
        if (timelineActionRetryTimeoutRef.current) {
          clearTimeout(timelineActionRetryTimeoutRef.current);
        }
        timelineActionRetryTimeoutRef.current = setTimeout(() => {
          timelineActionRetryTimeoutRef.current = null;
          const retry = scrollToMessage(detail.messageId);
          if (retry && detail.action === "edit") {
            dispatchEditMessage(detail.messageId);
          }
        }, 80);
        return true;
      }

      if (detail.action === "edit") {
        dispatchEditMessage(detail.messageId);
      }
      return true;
    },
    [
      containerRef,
      createChatBranch,
      dispatchEditMessage,
      findMessageIndex,
      historyId,
      messages.length,
      scrollToMessage,
      timelineActionRetryTimeoutRef,
    ],
  );

  const enqueueTimelineAction = React.useCallback(
    (detail: TimelineActionDetail) => {
      if (!detail?.historyId) return;
      if (detail.historyId !== historyId) {
        pendingTimelineActionRef.current = detail;
        void loadLocalConversation(detail.historyId);
        return;
      }

      const handled = performTimelineAction(detail);
      if (!handled) {
        pendingTimelineActionRef.current = detail;
      }
    },
    [historyId, loadLocalConversation, performTimelineAction],
  );

  React.useEffect(() => {
    const pending = pendingTimelineActionRef.current;
    if (!pending) return;
    const handled = performTimelineAction(pending);
    if (handled) {
      pendingTimelineActionRef.current = null;
    }
  }, [historyId, messages, performTimelineAction]);

  React.useEffect(() => {
    if (typeof window === "undefined") return;

    const handleTimelineActionEvent = (event: Event) => {
      const detail = (event as CustomEvent<TimelineActionDetail>).detail;
      if (!detail?.historyId) return;
      enqueueTimelineAction(detail);
    };

    const handleOpenHistoryEvent = (event: Event) => {
      const detail = (event as CustomEvent<OpenHistoryDetail>).detail;
      if (!detail?.historyId) return;
      enqueueTimelineAction({
        action: "go",
        historyId: detail.historyId,
        messageId: detail.messageId,
      });
    };
    const handleScrollToLatestEvent = () => {
      autoScrollToBottom();
    };

    window.addEventListener(TIMELINE_ACTION_EVENT, handleTimelineActionEvent);
    window.addEventListener(OPEN_HISTORY_EVENT, handleOpenHistoryEvent);
    window.addEventListener("tldw:scroll-to-latest", handleScrollToLatestEvent);
    return () => {
      window.removeEventListener(
        TIMELINE_ACTION_EVENT,
        handleTimelineActionEvent,
      );
      window.removeEventListener(OPEN_HISTORY_EVENT, handleOpenHistoryEvent);
      window.removeEventListener(
        "tldw:scroll-to-latest",
        handleScrollToLatestEvent,
      );
    };
  }, [autoScrollToBottom, enqueueTimelineAction]);

  const compareParentByHistory = useStoreMessageOption(
    (state) => state.compareParentByHistory,
  );
  const artifactsOpen = useArtifactsStore((state) => state.isOpen);
  const activeArtifact = useArtifactsStore((state) => state.active);
  const artifactsPinned = useArtifactsStore((state) => state.isPinned);
  const artifactHistory = useArtifactsStore((state) => state.history);
  const artifactUnreadCount = useArtifactsStore((state) => state.unreadCount);
  const setArtifactsOpen = useArtifactsStore((state) => state.setOpen);
  const closeArtifacts = useArtifactsStore((state) => state.closeArtifact);
  const markArtifactsRead = useArtifactsStore((state) => state.markRead);

  const parentMeta =
    historyId && compareParentByHistory
      ? compareParentByHistory[historyId]
      : undefined;
  const branchDepth = React.useMemo(() => {
    if (!historyId || !compareParentByHistory) return 0;
    let depth = 0;
    let cursor = historyId;
    const seen = new Set<string>();
    while (cursor && !seen.has(cursor)) {
      seen.add(cursor);
      const meta = compareParentByHistory[cursor];
      if (!meta?.parentHistoryId) break;
      depth += 1;
      cursor = meta.parentHistoryId;
    }
    return depth;
  }, [compareParentByHistory, historyId]);
  const branchForkPointLabel = React.useMemo(() => {
    if (!parentMeta?.parentHistoryId) return null;
    if (parentMeta.clusterId) {
      return toText(
        t("playground:branching.forkPointCluster", "Fork point: {{cluster}}", {
          cluster: parentMeta.clusterId,
        } as any),
      );
    }
    return toText(
      t("playground:branching.forkPointParent", "Fork point: {{historyId}}", {
        historyId: parentMeta.parentHistoryId,
      } as any),
    );
  }, [parentMeta?.clusterId, parentMeta?.parentHistoryId, t]);
  const branchDepthLabel = React.useMemo(() => {
    if (branchDepth <= 0) return null;
    return toText(
      t("playground:branching.depth", "Depth {{depth}}", {
        depth: branchDepth,
      } as any),
    );
  }, [branchDepth, t]);
  const compareActive = compareFeatureEnabled && compareMode;
  const compactFeatureNoticeVisible =
    isMobileViewport && (compareActive || Boolean(parentMeta?.parentHistoryId));
  const artifactPinnedCount = activeArtifact && artifactsPinned ? 1 : 0;
  const artifactHistoryCount = artifactHistory.length;
  const artifactBadgeLabel = artifactsOpen
    ? toText(t("playground:regions.artifactsOpen", "Artifacts panel open"))
    : activeArtifact
      ? toText(t("playground:regions.artifactsAvailable", "Artifacts ready"))
      : toText(
          t("playground:regions.artifactsClosed", "Artifacts panel closed"),
        );
  const closeArtifactsWithFocusReturn = React.useCallback(() => {
    closeArtifacts();
    requestAnimationFrame(() => {
      artifactsTriggerRef.current?.focus();
    });
  }, [closeArtifacts]);

  React.useEffect(() => {
    if (typeof window === "undefined") return;

    const handleShortcut = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const isEditableTarget = Boolean(
        target &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.isContentEditable),
      );
      if (
        (event.metaKey || event.ctrlKey) &&
        !event.altKey &&
        !event.shiftKey &&
        event.key.toLowerCase() === "f"
      ) {
        event.preventDefault();
        setThreadSearchOpen(true);
        requestAnimationFrame(() => {
          threadSearchInputRef.current?.focus();
          threadSearchInputRef.current?.select();
        });
        return;
      }
      if (
        !event.altKey &&
        !event.ctrlKey &&
        !event.metaKey &&
        event.shiftKey &&
        event.key === "?"
      ) {
        event.preventDefault();
        setShortcutsHelpOpen(true);
        return;
      }
      if (shortcutsHelpOpen && event.key === "Escape") {
        event.preventDefault();
        setShortcutsHelpOpen(false);
        requestAnimationFrame(() => {
          shortcutsTriggerRef.current?.focus();
        });
        return;
      }
      if (threadSearchOpen && event.key === "Escape") {
        event.preventDefault();
        setThreadSearchOpen(false);
        return;
      }

      const action = resolvePlaygroundShortcutAction(event);
      if (!action) return;
      if (isEditableTarget) return;
      event.preventDefault();

      if (action === "toggle_artifacts") {
        if (artifactsOpen) {
          closeArtifacts();
          return;
        }
        if (!activeArtifact) return;
        setArtifactsOpen(true);
        markArtifactsRead();
        return;
      }

      if (action === "toggle_compare") {
        window.dispatchEvent(new CustomEvent("tldw:toggle-compare-mode"));
        return;
      }

      if (action === "toggle_modes") {
        window.dispatchEvent(new CustomEvent("tldw:toggle-mode-launcher"));
      }
    };

    window.addEventListener("keydown", handleShortcut);
    return () => {
      window.removeEventListener("keydown", handleShortcut);
    };
  }, [
    activeArtifact,
    artifactsOpen,
    closeArtifacts,
    markArtifactsRead,
    setArtifactsOpen,
    shortcutsHelpOpen,
    threadSearchOpen,
  ]);

  React.useEffect(() => {
    if (typeof window === "undefined") return;
    const handleFocusArtifactsTrigger = () => {
      artifactsTriggerRef.current?.focus();
    };
    window.addEventListener(
      "tldw:focus-artifacts-trigger",
      handleFocusArtifactsTrigger,
    );
    return () => {
      window.removeEventListener(
        "tldw:focus-artifacts-trigger",
        handleFocusArtifactsTrigger,
      );
    };
  }, []);

  React.useEffect(() => {
    if (typeof window === "undefined") return;
    const handleOpenShortcutHelp = () => {
      setShortcutsHelpOpen(true);
    };
    window.addEventListener(
      "tldw:open-playground-shortcuts",
      handleOpenShortcutHelp,
    );
    return () => {
      window.removeEventListener(
        "tldw:open-playground-shortcuts",
        handleOpenShortcutHelp,
      );
    };
  }, []);

  React.useEffect(() => {
    if (!shortcutsHelpOpen) return;
    requestAnimationFrame(() => {
      shortcutsCloseRef.current?.focus();
    });
  }, [shortcutsHelpOpen]);

  React.useEffect(() => {
    if (!threadSearchOpen) return;
    if (threadSearchMatches.length === 0) {
      setThreadSearchActiveIndex(0);
      return;
    }
    setThreadSearchActiveIndex((previous) => {
      const bounded =
        previous >= 0 && previous < threadSearchMatches.length ? previous : 0;
      const messageIndex = threadSearchMatches[bounded];
      if (typeof messageIndex === "number") {
        requestAnimationFrame(() => {
          scrollToMessageIndex(messageIndex);
        });
      }
      return bounded;
    });
  }, [scrollToMessageIndex, threadSearchMatches, threadSearchOpen]);

  const stepThreadSearchMatch = React.useCallback(
    (direction: 1 | -1) => {
      if (threadSearchMatches.length === 0) return;
      setThreadSearchActiveIndex((previous) => {
        const next = getWrappedMatchIndex(
          previous,
          threadSearchMatches.length,
          direction,
        );
        const messageIndex = threadSearchMatches[next];
        if (typeof messageIndex === "number") {
          requestAnimationFrame(() => {
            scrollToMessageIndex(messageIndex);
          });
        }
        return next;
      });
    },
    [scrollToMessageIndex, threadSearchMatches],
  );

  const contextFileCount = Array.isArray(contextFiles)
    ? contextFiles.length
    : 0;
  const selectedKnowledgeCount = Array.isArray(selectedKnowledge)
    ? selectedKnowledge.length
    : selectedKnowledge
      ? 1
      : 0;
  const ragMediaIdCount = Array.isArray(ragMediaIds) ? ragMediaIds.length : 0;
  const trimmedSystemPrompt = String(systemPrompt || "").trim();
  const hasPromptContext = Boolean(
    selectedSystemPrompt ||
    selectedQuickPrompt ||
    trimmedSystemPrompt.length > 0,
  );
  const promptSummary: PlaygroundPromptSummary = buildCockpitPromptSummary({
    selectedSystemPrompt,
    selectedSystemPromptRecord,
    selectedSystemPromptStatus,
    selectedQuickPrompt,
    systemPrompt,
    copy: {
      customPromptLabel: toText(
        t("playground:cockpit.customPrompt", "Custom prompt"),
      ),
      inlineSystemPromptActiveDetail: toText(
        t(
          "playground:cockpit.inlineSystemPromptActive",
          "Inline system prompt active",
        ),
      ),
      loadingPromptDetail: toText(
        t(
          "playground:cockpit.loadingPromptDetails",
          "Loading prompt details...",
        ),
      ),
      noPromptContextDetail: toText(
        t(
          "playground:cockpit.noPromptContext",
          "No prompt context will be added.",
        ),
      ),
      noPromptSelectedLabel: toText(
        t("playground:cockpit.noPromptSelected", "No prompt selected"),
      ),
      selectedPromptUnavailableDetail: toText(
        t(
          "playground:cockpit.promptDetailsUnavailable",
          "Prompt details unavailable",
        ),
      ),
      quickPromptLabel: toText(
        t("playground:cockpit.quickPrompt", "Quick prompt"),
      ),
      selectedPromptDetail: toText(
        t("playground:cockpit.systemPrompt", "System prompt"),
      ),
      systemPromptLabel: toText(
        t("playground:cockpit.systemPrompt", "System prompt"),
      ),
    },
  });
  const clearPromptContextFromCockpit = React.useCallback(() => {
    setSelectedQuickPrompt(null);
    setSelectedSystemPrompt("");
    setSystemPrompt("");
  }, [setSelectedQuickPrompt, setSelectedSystemPrompt, setSystemPrompt]);
  const openPromptSelectorFromCockpit = React.useCallback(() => {
    openPromptSelector({
      returnFocusSelector: COCKPIT_PROMPT_SELECT_TRIGGER_SELECTOR,
    });
  }, []);
  const cockpitAssistantSelectTab =
    selectedAssistant?.kind === "persona" ? "persona" : "character";
  const openAssistantSelectorFromCockpit = React.useCallback(() => {
    openAssistantSelector({
      tab: cockpitAssistantSelectTab,
      returnFocusSelector: COCKPIT_ASSISTANT_SELECT_TRIGGER_SELECTOR,
    });
  }, [cockpitAssistantSelectTab]);
  const clearAssistantFromCockpit = React.useCallback(() => {
    void setSelectedAssistant(null);
    setSelectedCharacter(null);
    scheduleFocusFirstVisibleElement(COCKPIT_ASSISTANT_SELECT_TRIGGER_SELECTOR);
  }, [setSelectedAssistant, setSelectedCharacter]);
  const inspectAssistantFromCockpit = React.useCallback(() => {
    if (selectedAssistant?.kind === "persona") {
      navigate(
        buildPersonaGardenRoute({
          personaId: selectedAssistant.id,
          tab: "profiles",
        }),
      );
      return;
    }
    if (selectedAssistant?.kind === "character" || selectedCharacter) {
      navigate("/settings/characters");
    }
  }, [navigate, selectedAssistant, selectedCharacter]);
  const cockpitAssistantSummary = buildCockpitAssistantSummary({
    selectedAssistant,
    selectedCharacter,
    personaMemoryMode: serverChatPersonaMemoryMode,
    copy: {
      assistantFallbackName: toText(
        t("playground:cockpit.assistantFallback", "Assistant"),
      ),
      characterSelected: toText(
        t("playground:cockpit.characterSelected", "Character selected"),
      ),
      legacyCharacterFallbackName: (id) =>
        toText(
          t("playground:cockpit.characterFallbackById", `Character ${id}`, {
            id,
          }),
        ),
      memoryReadOnly: toText(
        t("playground:cockpit.personaMemoryReadOnly", "memory read-only"),
      ),
      memoryReadWrite: toText(
        t("playground:cockpit.personaMemoryReadWrite", "memory read/write"),
      ),
      noAssistantSelected: toText(
        t("playground:cockpit.noAssistantSelected", "No assistant selected"),
      ),
      personaFallbackName: toText(
        t("playground:cockpit.personaFallback", "Persona"),
      ),
      personaSelected: toText(
        t("playground:cockpit.personaSelected", "Persona selected"),
      ),
      personaSelectedWithMemoryMode: (memoryMode) =>
        toText(
          t(
            "playground:cockpit.personaSelectedWithMemoryMode",
            `Persona selected - ${memoryMode}`,
            { memoryMode },
          ),
        ),
    },
  });
  const latestAssistantMessage = React.useMemo(() => {
    for (let index = messages.length - 1; index >= 0; index -= 1) {
      const message = messages[index];
      if (message?.role === "assistant" || message?.isBot) {
        return message;
      }
    }
    return null;
  }, [messages]);
  const canRegenerateLastResponse = Boolean(latestAssistantMessage);
  const emptyAssistantResponse = React.useMemo(() => {
    if (!latestAssistantMessage || streaming || isProcessing) return false;
    return !hasVisibleAssistantResponse(latestAssistantMessage);
  }, [isProcessing, latestAssistantMessage, streaming]);
  const runtimeStatusDetail =
    serverReadinessState === "blocked"
      ? toText(
          t(
            "playground:cockpit.blockedServerHealth",
            "Server is unavailable. Check the server connection before sending.",
          ),
        )
      : serverReadinessState === "degraded"
        ? serverDegradedChecks.length > 0
          ? `${toText(
              t("playground:cockpit.degraded", DEGRADED_STATE_LABEL),
            )}: ${serverDegradedChecks.join(", ")}`
          : toText(
              t(
                "playground:cockpit.degradedServerHealth",
                "Server health is degraded",
              ),
            )
        : null;
  const hasChatContext = Boolean(
    attachedResearchContext ||
    webSearch ||
    contextFileCount > 0 ||
    selectedKnowledgeCount > 0 ||
    ragMediaIdCount > 0 ||
    hasPromptContext ||
    cockpitAssistantSummary.mode !== "none",
  );
  const sessionSummary = buildCockpitSessionSummary({
    temporaryChat,
    serverChatId,
    historyId,
    serverChatTitle,
    serverChatLoadState,
    serverChatLoadError,
    serverChatState,
    serverChatTopic,
    serverChatSource,
    copy: {
      failedDetail: toText(
        t(
          "playground:cockpit.sessionLoadFailedDetail",
          "Failed to load conversation",
        ),
      ),
      failedStatusLabel: toText(
        t("playground:cockpit.sessionLoadFailed", "Load failed"),
      ),
      historyLinkedDetail: toText(
        t("playground:cockpit.historyLinked", "History linked"),
      ),
      idleStatusLabel: toText(t("playground:cockpit.idle", "Idle")),
      loadingDetail: toText(
        t("playground:cockpit.sessionLoading", "Loading conversation"),
      ),
      loadingStatusLabel: toText(
        t("playground:cockpit.sessionLoading", "Loading conversation"),
      ),
      localChatLabel: toText(
        t("playground:cockpit.sessionLocal", "Local chat"),
      ),
      localHistoryStatusLabel: toText(
        t("playground:cockpit.localHistory", "Local history"),
      ),
      noSavedHistoryDetail: toText(
        t("playground:cockpit.noSavedHistory", "No saved history yet"),
      ),
      readyDetail: toText(
        t("playground:cockpit.sessionReadyDetail", "Conversation ready"),
      ),
      readyStatusLabel: toText(
        t("playground:cockpit.sessionReady", READY_STATE_LABEL),
      ),
      serverChatLabel: toText(
        t("playground:cockpit.sessionServer", "Server chat"),
      ),
      temporaryChatLabel: toText(
        t("playground:cockpit.sessionTemporary", "Temporary chat"),
      ),
      temporaryDetail: toText(t("playground:cockpit.notSaved", "Not saved")),
      temporaryStatusLabel: toText(
        t("playground:cockpit.localOnly", "Local only"),
      ),
    },
  });
  const sessionLabel = sessionSummary.label;
  const contextSummary: string[] = [];
  const providerRouteSummary = buildCockpitProviderRouteSummary({
    selectedProvider: apiProvider,
    selectedModel,
  });
  const characterChatModelUsability = React.useMemo(
    () =>
      buildChatModelUsability({
        isServerConnected: serverReadinessState !== "blocked",
        selectedModel: providerRouteSummary.selectedModel,
        availableModels: characterChatAvailableModels,
        modelsLoading: !Array.isArray(characterChatAvailableModels),
        serverDegraded: serverReadinessState === "degraded",
        allowDegradedSend: false,
      }),
    [
      characterChatAvailableModels,
      providerRouteSummary.selectedModel,
      serverReadinessState,
    ],
  );
  const characterChatReadiness = React.useMemo(
    () =>
      buildCharacterChatReadiness({
        isServerConnected: serverReadinessState !== "blocked",
        selectedCharacter,
        selectedModel: providerRouteSummary.selectedModel,
        availableModels: characterChatAvailableModels,
        modelsLoading: !Array.isArray(characterChatAvailableModels),
        serverDegraded: serverReadinessState === "degraded",
        allowDegradedSend: false,
        isSendBlocked: Boolean(streaming || isProcessing || isLoading),
      }),
    [
      characterChatAvailableModels,
      isLoading,
      isProcessing,
      providerRouteSummary.selectedModel,
      selectedCharacter,
      serverReadinessState,
      streaming,
    ],
  );
  const characterChatBlocked =
    characterWorkflowActive && characterChatReadiness.status === "blocked";
  const characterChatModelUnavailable =
    characterChatBlocked &&
    characterChatReadiness.missingRequirement === "chat-model";
  const characterChatReadinessCopy = React.useMemo(
    () =>
      characterChatBlocked
        ? getCharacterChatReadinessCopy(characterChatReadiness, t, {
            characterName: activeCharacterModeLabel,
          })
        : null,
    [activeCharacterModeLabel, characterChatBlocked, characterChatReadiness, t],
  );
  const activeCharacterChatModelUsability = characterWorkflowActive
    ? characterChatModelUsability
    : null;
  const characterChatModelUsabilityMessage =
    getMatchingCharacterChatModelUsabilityCopy({
      modelUsability: activeCharacterChatModelUsability,
      readiness: characterChatReadiness,
      readinessTitle: characterChatReadinessCopy?.title ?? null,
    });
  const characterChatModelSelectorLabel = React.useMemo(() => {
    if (
      !activeCharacterChatModelUsability ||
      activeCharacterChatModelUsability.status === "ready" ||
      (activeCharacterChatModelUsability.status === "degraded" &&
        activeCharacterChatModelUsability.canSend)
    ) {
      return null;
    }

    switch (activeCharacterChatModelUsability.status) {
      case "loading":
        return toText(
          t(
            "playground:composer.modelUsabilityChecking",
            "Checking model readiness",
          ),
        );
      case "no_server":
        return toText(
          t("playground:composer.modelUsabilityServer", "Server unavailable"),
        );
      case "no_selection":
        return toText(
          t("playground:composer.modelUsabilityChoose", "Choose model"),
        );
      case "no_models":
        return toText(
          t(
            "playground:composer.modelUsabilityNoModels",
            "No chat models configured",
          ),
        );
      case "selected_missing":
        return toText(
          t(
            "playground:composer.modelUsabilityUnavailable",
            "Model unavailable",
          ),
        );
      case "provider_unconfigured":
        return toText(
          t(
            "playground:composer.modelUsabilityProviderSetup",
            "Provider setup needed",
          ),
        );
      case "model_unavailable":
        return toText(
          t("playground:composer.modelUsabilityNotCallable", "Not callable"),
        );
      case "degraded":
        return toText(
          t("playground:composer.modelUsabilityBlocked", "Model blocked"),
        );
      default:
        return null;
    }
  }, [activeCharacterChatModelUsability, t]);
  const characterChatModelSelectorTitle = characterChatModelSelectorLabel
    ? characterChatModelUsabilityMessage ?? characterChatModelSelectorLabel
    : null;
  const characterChatModelUsabilityBlocks = Boolean(
    activeCharacterChatModelUsability &&
      activeCharacterChatModelUsability.status !== "ready" &&
      !activeCharacterChatModelUsability.canSend,
  );
  React.useEffect(() => {
    if (typeof setActiveSettingsScope === "function") {
      setActiveSettingsScope(providerRouteSummary.providerRouteLabel ?? null);
    }
  }, [providerRouteSummary.providerRouteLabel, setActiveSettingsScope]);
  const contextFileItems = Array.isArray(contextFiles) ? contextFiles : [];
  const selectedKnowledgeItems = Array.isArray(selectedKnowledge)
    ? selectedKnowledge
    : selectedKnowledge
      ? [selectedKnowledge]
      : [];
  const ragMediaItems = Array.isArray(ragMediaIds) ? ragMediaIds : [];
  const removeContextFileAt = React.useCallback(
    (index: number) => {
      setContextFiles(
        contextFileItems.filter((_, itemIndex) => itemIndex !== index),
      );
    },
    [contextFileItems, setContextFiles],
  );
  const removeRagMediaAt = React.useCallback(
    (index: number) => {
      const nextIds = ragMediaItems.filter(
        (_, itemIndex) => itemIndex !== index,
      );
      setRagMediaIds(nextIds.length > 0 ? nextIds : null);
    },
    [ragMediaItems, setRagMediaIds],
  );
  const removeSelectedKnowledgeAt = React.useCallback(
    (index: number) => {
      const nextItems = selectedKnowledgeItems.filter(
        (_, itemIndex) => itemIndex !== index,
      );
      const nextValue = Array.isArray(selectedKnowledge)
        ? nextItems.length > 0
          ? nextItems
          : null
        : nextItems[0] || null;
      (setSelectedKnowledge as (value: unknown) => void)(nextValue);
    },
    [selectedKnowledge, selectedKnowledgeItems, setSelectedKnowledge],
  );
  const contextSources = (
    [
      webSearch
        ? {
            id: "web-search",
            kind: "web" as const,
            label: toText(t("playground:cockpit.web", "Web")),
            title: toText(t("playground:cockpit.webSearch", "Web search")),
            detail: toText(
              t(
                "playground:cockpit.webSearchDetail",
                "Enabled for the next reply.",
              ),
            ),
            state: "active" as const,
            onRemove: toggleWebSearchFromCockpit,
            removeLabel: toText(
              t("playground:cockpit.disableWebSearch", "Disable web search"),
            ),
          }
        : null,
      hasPromptContext
        ? {
            id: `prompt-${promptSummary.state}`,
            kind: "prompt" as const,
            label: toText(t("playground:cockpit.prompt", "Prompt")),
            title: promptSummary.label,
            detail: promptSummary.detail,
            state: "active" as const,
            onOpen: openPromptSelectorFromCockpit,
            onRemove: clearPromptContextFromCockpit,
            openLabel: toText(
              t("playground:cockpit.selectPrompt", "Select a prompt"),
            ),
            removeLabel: toText(
              t(
                "playground:cockpit.clearPromptContext",
                "Clear prompt context",
              ),
            ),
          }
        : null,
      cockpitAssistantSummary.mode !== "none" && cockpitAssistantSummary.name
        ? {
            id: `assistant-${cockpitAssistantSummary.mode}-${cockpitAssistantSummary.name}`,
            kind: "assistant" as const,
            label:
              cockpitAssistantSummary.mode === "persona"
                ? toText(t("playground:cockpit.persona", "Persona"))
                : toText(t("playground:cockpit.character", "Character")),
            title: cockpitAssistantSummary.name,
            detail: cockpitAssistantSummary.detail,
            state: "active" as const,
            onOpen: openAssistantSelectorFromCockpit,
            onRemove: clearAssistantFromCockpit,
            openLabel: toText(
              t(
                "playground:cockpit.selectCharacterPersona",
                "Select character or persona",
              ),
            ),
            removeLabel: toText(
              t("playground:cockpit.clearAssistant", "Clear assistant"),
            ),
          }
        : null,
      attachedResearchContext
        ? {
            id: `research-${attachedResearchContext.run_id || "active"}`,
            kind: "research" as const,
            label: toText(t("playground:cockpit.research", "Research")),
            title:
              attachedResearchContext.query ||
              attachedResearchContext.question ||
              toText(
                t("playground:cockpit.researchContext", "Research context"),
              ),
            detail: attachedResearchContext.run_id
              ? toText(
                  t("playground:cockpit.researchRun", "Run {{runId}}", {
                    runId: attachedResearchContext.run_id,
                  }),
                )
              : null,
            state: "active" as const,
            onOpen: () => openSearchAndContext({ tab: "context" }),
            onRemove: handleRemoveAttachedResearchContext,
            removeLabel: toText(
              t(
                "playground:cockpit.clearResearchContext",
                "Clear research context",
              ),
            ),
          }
        : null,
      ...contextFileItems.map((file, index) => {
        const title =
          getRecordString(file, ["name", "filename", "title", "id"]) ||
          toText(
            t("playground:cockpit.fileFallback", "File {{index}}", {
              index: index + 1,
            }),
          );
        return {
          id: `file-${getRecordString(file, ["id"]) || index}`,
          kind: "file" as const,
          label: toText(t("playground:cockpit.file", "File")),
          title,
          detail: toText(
            t("playground:cockpit.nextReply", "Used on next reply"),
          ),
          state: "active" as const,
          onRemove: () => removeContextFileAt(index),
          removeLabel: toText(
            t("playground:cockpit.removeFileSource", `Remove ${title}`, {
              title,
            }),
          ),
        };
      }),
      ...selectedKnowledgeItems.map((knowledge, index) => {
        const title =
          getRecordString(knowledge, ["title", "name", "id"]) ||
          toText(
            t("playground:cockpit.knowledgeFallback", "Knowledge {{index}}", {
              index: index + 1,
            }),
          );
        return {
          id: `knowledge-${getRecordString(knowledge, ["id"]) || index}`,
          kind: "knowledge" as const,
          label: toText(t("playground:cockpit.knowledge", "Knowledge")),
          title,
          detail: toText(
            t("playground:cockpit.nextReply", "Used on next reply"),
          ),
          state: "active" as const,
          onOpen: () => openSearchAndContext({ tab: "context" }),
          onRemove: () => removeSelectedKnowledgeAt(index),
          openLabel: toText(
            t("playground:cockpit.openKnowledgeSource", `Open ${title}`, {
              title,
            }),
          ),
          removeLabel: toText(
            t("playground:cockpit.removeKnowledgeSource", `Remove ${title}`, {
              title,
            }),
          ),
        };
      }),
      ...ragMediaItems.map((mediaId, index) => ({
        id: `media-${mediaId}`,
        kind: "media" as const,
        label: toText(t("playground:cockpit.media", "Media")),
        title: toText(
          t("playground:cockpit.mediaScopeLabel", "Media scope {{id}}", {
            id: mediaId,
          }),
        ),
        detail: toText(t("playground:cockpit.nextReply", "Used on next reply")),
        state: "active" as const,
        onOpen: () => openSearchAndContext({ tab: "context" }),
        onRemove: () => removeRagMediaAt(index),
        openLabel: toText(
          t("playground:cockpit.openMediaSource", "Open media scope"),
        ),
        removeLabel: toText(
          t("playground:cockpit.removeMediaSource", "Remove media scope"),
        ),
      })),
    ] satisfies Array<PlaygroundContextSource | null>
  ).filter(isPlaygroundContextSource);
  const activeScopedModelSettings =
    activeSettingsScope && scopedSettingsByModelKey
      ? scopedSettingsByModelKey[activeSettingsScope]
      : undefined;
  const getRuntimeSettingSource = (
    key: keyof ChatModelSettings,
  ): RuntimeSettingSummary["source"] =>
    activeSettingsScope
      ? Object.prototype.hasOwnProperty.call(
          activeScopedModelSettings || {},
          key,
        )
        ? "override"
        : "default"
      : undefined;
  const runtimeSettingSummaryItems: Array<RuntimeSettingSummary | null> = [
    typeof temperature === "number"
      ? {
          label: toText(t("playground:cockpit.temperature", "Temperature")),
          value: String(temperature),
          source: getRuntimeSettingSource("temperature"),
        }
      : null,
    typeof topP === "number"
      ? {
          label: toText(t("playground:cockpit.topP", "Top P")),
          value: String(topP),
          source: getRuntimeSettingSource("topP"),
        }
      : null,
    typeof topK === "number"
      ? {
          label: toText(t("playground:cockpit.topK", "Top K")),
          value: String(topK),
          source: getRuntimeSettingSource("topK"),
        }
      : null,
    typeof numCtx === "number"
      ? {
          label: toText(t("playground:cockpit.contextWindow", "Context")),
          value: String(numCtx),
          source: getRuntimeSettingSource("numCtx"),
        }
      : null,
    typeof numPredict === "number"
      ? {
          label: toText(t("playground:cockpit.maxTokens", "Max tokens")),
          value: String(numPredict),
          source: getRuntimeSettingSource("numPredict"),
        }
      : null,
    typeof reasoningEffort === "string" && reasoningEffort.length > 0
      ? {
          label: toText(t("playground:cockpit.reasoning", "Reasoning")),
          value: reasoningEffort,
          source: getRuntimeSettingSource("reasoningEffort"),
        }
      : null,
  ];
  const runtimeSettingSummaries: RuntimeSettingSummary[] =
    runtimeSettingSummaryItems.filter((item): item is RuntimeSettingSummary =>
      Boolean(item),
    );
  const cockpitToolSummary = buildCockpitMcpSummary({
    hasMcp: mcpHealthState !== "unavailable",
    healthState: mcpHealthState,
    toolsLoading: mcpToolsLoading,
    discoveredCount: discoveredMcpToolCount,
    chatToolCount: chatMcpToolCount,
    toolCounts: mcpToolCounts,
    copy: {
      availableDetail: (chatToolCount, discoveredCount) => {
        const chatToolsLabel =
          chatToolCount === 1
            ? toText(
                t(
                  "playground:cockpit.mcpChatToolsAvailableOne",
                  "1 chat tool available",
                ),
              )
            : toText(
                t(
                  "playground:cockpit.mcpChatToolsAvailableMany",
                  `${chatToolCount} chat tools available`,
                  { count: chatToolCount },
                ),
              );
        if (discoveredCount <= chatToolCount) return chatToolsLabel;
        const discoveredSuffix = toText(
          t(
            "playground:cockpit.mcpDiscoveredSuffix",
            ` (${discoveredCount} discovered)`,
            { count: discoveredCount },
          ),
        );
        return `${chatToolsLabel}${discoveredSuffix}`;
      },
      chatEnabledLabel: toText(
        t("playground:cockpit.mcpChatEnabledLabel", "Chat-enabled"),
      ),
      discoveredLabel: toText(
        t("playground:cockpit.mcpDiscoveredLabel", "Discovered"),
      ),
      emptyDetail: toText(
        t("playground:composer.mcpToolsEmpty", "No MCP tools available"),
      ),
      executableLabel: toText(
        t("playground:cockpit.mcpExecutableLabel", "Executable"),
      ),
      loadingDetail: toText(
        t("playground:composer.mcpToolsLoading", "Loading tools..."),
      ),
      nameConflictsLabel: toText(
        t("playground:cockpit.mcpNameConflictsLabel", "Name conflicts"),
      ),
      offlineDetail: toText(
        t("playground:composer.mcpToolsUnhealthy", "MCP tools are offline"),
      ),
      toolsLabel: toText(t("playground:cockpit.mcpTools", "MCP tools")),
      unavailableDetail: toText(
        t("playground:composer.mcpToolsUnavailable", "MCP tools unavailable"),
      ),
      unavailableLabel: toText(
        t("playground:composer.mcpUnavailable", "MCP unavailable"),
      ),
      unavailableToolsLabel: toText(
        t(
          "playground:cockpit.mcpUnavailableToolsLabel",
          UNAVAILABLE_DESIGN_STATE_LABEL,
        ),
      ),
      userDisabledLabel: toText(
        t("playground:cockpit.mcpUserDisabledLabel", "User-disabled"),
      ),
    },
  });
  const compositionStatus = "idle" as const;
  const compositionPreviewSummary = buildPlaygroundCompositionPreviewSummary({
    promptSummary,
    assistantSummary: cockpitAssistantSummary,
    providerRoute: providerRouteSummary,
    settingSummaries: runtimeSettingSummaries,
    contextSources,
    toolSummary: cockpitToolSummary,
    compositionStatus,
    composition: null,
    modelUsabilityStatus: activeCharacterChatModelUsability?.status ?? null,
    modelUsabilityCanSend: activeCharacterChatModelUsability?.canSend ?? null,
    modelUsabilityDetail: characterChatModelUsabilityMessage,
    modelUnavailable: characterChatModelUnavailable,
    modelUnavailableDetail: characterChatModelUnavailable
      ? characterChatReadinessCopy?.title ?? null
      : null,
  });
  const openModelSettingsFromCockpit = React.useCallback(() => {
    if (typeof setActiveSettingsScope === "function") {
      setActiveSettingsScope(providerRouteSummary.providerRouteLabel ?? null);
    }
    openModelSettings({
      returnFocusSelector: COCKPIT_MODEL_SETTINGS_TRIGGER_SELECTOR,
      settingsScope: providerRouteSummary.providerRouteLabel ?? null,
    });
  }, [providerRouteSummary.providerRouteLabel, setActiveSettingsScope]);
  const openServerSettingsFromCockpit = React.useCallback(() => {
    if (typeof setActiveSettingsScope === "function") {
      setActiveSettingsScope(null);
    }
    openModelSettings({
      returnFocusSelector: COCKPIT_MODEL_SETTINGS_TRIGGER_SELECTOR,
      settingsScope: null,
    });
  }, [setActiveSettingsScope]);
  const openCharacterSelectorFromReadiness = React.useCallback(() => {
    openAssistantSelector({
      tab: "character",
      returnFocusSelector: COCKPIT_ASSISTANT_SELECT_TRIGGER_SELECTOR,
    });
  }, []);
  const retryRouteCharacterRecovery = React.useCallback(() => {
    if (!routeCharacterRecovery) return;
    routeCharacterIntentAppliedRef.current = null;
    routeCharacterIntentInFlightRef.current = null;
    setRouteCharacterRecovery(null);
    setRouteCharacterRetryToken((previous) => previous + 1);
  }, [routeCharacterRecovery]);
  const handleCharacterChatReadinessAction = React.useCallback(
    (action: CharacterChatReadinessAction) => {
      if (action === "choose-character") {
        openCharacterSelectorFromReadiness();
        return;
      }
      if (action === "open-model-settings") {
        openModelSettingsFromCockpit();
        return;
      }
      if (action === "open-server-settings") {
        openServerSettingsFromCockpit();
        return;
      }
      if (action === "retry") {
        void refreshCharacterChatModels();
      }
    },
    [
      openCharacterSelectorFromReadiness,
      openModelSettingsFromCockpit,
      refreshCharacterChatModels,
      openServerSettingsFromCockpit,
    ],
  );
  const characterChatSendBlocker = React.useMemo<PlaygroundSendBlocker | null>(
    () =>
      characterWorkflowActive &&
      characterChatReadiness.status === "blocked" &&
      characterChatReadiness.missingRequirement === "chat-model" &&
      characterChatReadinessCopy
        ? {
            active: true,
            title: characterChatReadinessCopy.title,
            actionLabel: characterChatReadinessCopy.actionLabel,
            onAction: () =>
              handleCharacterChatReadinessAction(
                characterChatReadiness.recommendedAction ??
                  "open-model-settings",
              ),
          }
        : null,
    [
      characterChatReadiness,
      characterChatReadinessCopy,
      characterWorkflowActive,
      handleCharacterChatReadinessAction,
    ],
  );
  const openMcpSettingsFromCockpit = React.useCallback(() => {
    openMcpSettings({
      returnFocusSelector: COCKPIT_MCP_SETTINGS_TRIGGER_SELECTOR,
    });
  }, []);
  const statusContextSummary = [
    hasPromptContext ? promptSummary.label : null,
    webSearch
      ? toText(t("playground:cockpit.webSearchOn", "Web search on"))
      : null,
    contextFileCount > 0
      ? contextFileCount === 1
        ? toText(t("playground:cockpit.contextFilesCountOne", "1 file"))
        : toText(
            t(
              "playground:cockpit.contextFilesCountMany",
              `${contextFileCount} files`,
              { count: contextFileCount },
            ),
          )
      : null,
    selectedKnowledgeCount > 0
      ? selectedKnowledgeCount === 1
        ? toText(
            t(
              "playground:cockpit.contextKnowledgeCountOne",
              "1 knowledge item",
            ),
          )
        : toText(
            t(
              "playground:cockpit.contextKnowledgeCountMany",
              `${selectedKnowledgeCount} knowledge items`,
              { count: selectedKnowledgeCount },
            ),
          )
      : null,
    ragMediaIdCount > 0
      ? ragMediaIdCount === 1
        ? toText(t("playground:cockpit.contextMediaCountOne", "1 media scope"))
        : toText(
            t(
              "playground:cockpit.contextMediaCountMany",
              `${ragMediaIdCount} media scopes`,
              { count: ragMediaIdCount },
            ),
          )
      : null,
  ].filter((item): item is string => Boolean(item));
  const cockpitMessageCount = getCockpitMessageCount(messages, history);
  const activeCharacterSessionId =
    selectedAssistant?.kind === "character"
      ? selectedAssistant.id
      : selectedCharacter?.id;
  const characterSessionsPanel = characterWorkflowActive ? (
    <CharacterChatSessionsPanel
      activeCharacterId={activeCharacterSessionId ?? null}
      activeCharacterName={activeCharacterModeLabel}
      activeServerChatId={serverChatId}
    />
  ) : null;
  const cockpitLeftRail = (
    <PlaygroundContextRail
      hasContext={hasChatContext}
      contextSummary={contextSummary}
      contextSources={contextSources}
      sessionLabel={sessionLabel}
      sessionTitle={sessionSummary.title}
      sessionStatus={sessionSummary.status}
      sessionStatusLabel={sessionSummary.statusLabel}
      sessionDetail={sessionSummary.detail}
      sessionError={sessionSummary.error}
      historyLinked={Boolean(historyId)}
      webSearch={webSearch}
      onToggleWebSearch={toggleWebSearchFromCockpit}
      temporaryChat={temporaryChat}
      onToggleTemporaryChat={setTemporaryChatFromCockpit}
      contextCounts={{
        files: contextFileCount,
        knowledge: selectedKnowledgeCount,
        media: ragMediaIdCount,
        research: attachedResearchContext ? 1 : 0,
      }}
      promptSummary={promptSummary}
      promptSelectControl={
        <button
          type="button"
          className="inline-flex min-h-[30px] items-center rounded-md border border-border bg-surface2 px-2.5 py-1 text-xs font-medium text-text hover:bg-surface focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
          data-cockpit-prompt-select-trigger
          aria-label={toText(
            t("playground:cockpit.selectPrompt", "Select a prompt"),
          )}
          onClick={openPromptSelectorFromCockpit}
        >
          {toText(t("playground:cockpit.selectPrompt", "Select prompt"))}
        </button>
      }
      characterSessionsPanel={characterSessionsPanel}
      onClearPrompt={clearPromptContextFromCockpit}
      onOpenSearchContext={() => openSearchAndContext({ tab: "search" })}
      onClearFiles={() => setContextFiles([])}
      onClearKnowledge={() => setSelectedKnowledge(null)}
      onClearMedia={() => setRagMediaIds(null)}
      onClearResearch={handleRemoveAttachedResearchContext}
      compositionPreviewSummary={compositionPreviewSummary}
    />
  );
  const cockpitRightRail = (
    <PlaygroundRuntimeInspector
      streaming={streaming}
      selectedProvider={providerRouteSummary.selectedProvider}
      selectedModel={providerRouteSummary.selectedModel}
      providerRouteLabel={providerRouteSummary.providerRouteLabel}
      modelUsabilityStatus={activeCharacterChatModelUsability?.status ?? null}
      modelUsabilityCanSend={activeCharacterChatModelUsability?.canSend ?? null}
      modelUsabilityDetail={characterChatModelUsabilityMessage}
      runtimeStatus={
        serverReadinessState === "blocked"
          ? "error"
          : streaming
            ? "streaming"
            : activeCharacterChatModelUsability?.status === "loading"
              ? "loading"
            : characterChatModelUsabilityBlocks
              ? "error"
              : serverReadinessState === "degraded"
                ? "degraded"
                : "ready"
      }
      runtimeStatusDetail={
        characterChatReadinessCopy?.title ?? runtimeStatusDetail
      }
      messageCount={cockpitMessageCount}
      threadSearchOpen={threadSearchOpen}
      assistantSummary={cockpitAssistantSummary}
      onOpenModelSettings={openModelSettingsFromCockpit}
      onOpenAssistantSelect={openAssistantSelectorFromCockpit}
      onClearAssistant={clearAssistantFromCockpit}
      onInspectAssistant={inspectAssistantFromCockpit}
      onOpenSceneDirector={openActorSettings}
      canStopStreaming={streaming}
      onStopStreaming={() => stopStreamingRequest()}
      canRegenerate={canRegenerateLastResponse}
      onRegenerate={() => regenerateLastMessage()}
      emptyAssistantResponse={emptyAssistantResponse}
      settingSummaries={runtimeSettingSummaries}
      toolChoice={toolChoice as RuntimeToolChoice}
      onToolChoiceChange={(nextChoice) => setToolChoice(nextChoice)}
      onOpenMcpSettings={openMcpSettingsFromCockpit}
      toolSummary={cockpitToolSummary}
    />
  );
  const cockpitStatusStrip = (
    <PlaygroundStatusStrip
      mode={normalizedChatLayoutMode}
      streaming={streaming}
      selectedProvider={providerRouteSummary.selectedProvider}
      selectedModel={providerRouteSummary.selectedModel}
      messageCount={cockpitMessageCount}
      sessionLabel={sessionLabel}
      sessionTitle={sessionSummary.title}
      sessionStatus={sessionSummary.status}
      sessionStatusLabel={sessionSummary.statusLabel}
      sessionDetail={sessionSummary.detail}
      sessionError={sessionSummary.error}
      hasContext={hasChatContext}
      contextSummary={statusContextSummary}
      temporaryChat={temporaryChat}
      characterChatActive={characterWorkflowActive}
      degraded={serverReadinessState === "degraded"}
      degradedChecks={serverDegradedChecks}
      errorMessage={null}
      serverBlocked={serverReadinessState === "blocked"}
      modelUsabilityStatus={activeCharacterChatModelUsability?.status ?? null}
      modelUsabilityCanSend={activeCharacterChatModelUsability?.canSend ?? null}
      modelUsabilityMessage={characterChatModelUsabilityMessage}
      modelUnavailable={characterChatModelUnavailable}
      modelUnavailableMessage={
        characterChatModelUnavailable ? characterChatReadinessCopy?.title ?? null : null
      }
      compositionStatus={compositionStatus}
      onStopStreaming={() => stopStreamingRequest()}
      onOpenSearchContext={() => openSearchAndContext({ tab: "context" })}
      onOpenModelSettings={openModelSettings}
    />
  );

  return (
    <div
      ref={drop}
      data-is-dragging={dropState === "dragging"}
      className="relative flex h-full min-h-0 flex-col items-center bg-bg text-text data-[is-dragging=true]:bg-surface2"
      style={
        chatBackgroundImage
          ? {
              backgroundImage: `url(${chatBackgroundImage})`,
              backgroundSize: "cover",
              backgroundPosition: "center",
              backgroundRepeat: "no-repeat",
            }
          : {}
      }
    >
      {/* Background overlay for opacity effect */}
      {chatBackgroundImage && (
        <div
          className="absolute inset-0 bg-bg"
          style={{ opacity: 0.9, pointerEvents: "none" }}
        />
      )}

      {dropState === "dragging" && (
        <div className="pointer-events-none absolute inset-0 z-30 flex flex-col items-center justify-center">
          <div className="rounded-2xl border border-dashed border-border bg-elevated px-6 py-4 text-center text-sm font-medium text-text shadow-card">
            {t(
              "playground:drop.hint",
              "Drop files to attach them to your message",
            )}
          </div>
        </div>
      )}

      {dropFeedback && (
        <div className="pointer-events-none absolute top-4 left-0 right-0 z-30 flex justify-center px-4">
          <div
            role="status"
            aria-live="polite"
            className={`max-w-lg rounded-full px-4 py-2 text-sm shadow-lg backdrop-blur-sm ${
              dropFeedback.type === "error"
                ? "border border-danger bg-danger text-white"
                : dropFeedback.type === "warning"
                  ? "border border-warn bg-warn/10 text-warn"
                  : "border border-border bg-elevated text-text"
            }`}
          >
            {dropFeedback.message}
          </div>
        </div>
      )}

      <div className="relative z-10 flex h-full min-h-0 w-full">
        <PlaygroundCockpitShell
          mode={normalizedChatLayoutMode}
          onModeChange={handleChatLayoutModeChange}
          leftRailVisible={normalizedCockpitContextRailVisible}
          rightRailVisible={normalizedCockpitRuntimeRailVisible}
          onLeftRailVisibleChange={setCockpitContextRailVisible}
          onRightRailVisibleChange={setCockpitRuntimeRailVisible}
          mobilePanel={mobileCockpitPanel}
          onMobilePanelChange={setMobileCockpitPanel}
          leftRail={cockpitLeftRail}
          rightRail={cockpitRightRail}
          statusStrip={cockpitStatusStrip}
        >
          <div className="flex h-full min-h-0 min-w-0 flex-1 flex-col">
            {parentMeta?.parentHistoryId && (
              <div className="flex w-full justify-center px-5 pt-2">
                <div className="inline-flex flex-wrap items-center justify-center gap-2">
                  <button
                    type="button"
                    className="inline-flex items-center gap-2 rounded-full border border-primary bg-surface2 px-3 py-1 text-[11px] font-medium text-primaryStrong hover:bg-surface focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
                    title={t(
                      "playground:composer.compareBreadcrumb",
                      "Back to comparison chat",
                    )}
                    onClick={() => {
                      window.dispatchEvent(
                        new CustomEvent("tldw:open-history", {
                          detail: { historyId: parentMeta.parentHistoryId },
                        }),
                      );
                    }}
                  >
                    <span aria-hidden="true">←</span>
                    <span>
                      {t(
                        "playground:composer.compareBreadcrumb",
                        "Back to comparison chat",
                      )}
                    </span>
                  </button>
                  {branchForkPointLabel && (
                    <span
                      data-testid="playground-branch-fork-point"
                      className="inline-flex items-center rounded-full border border-border bg-surface2 px-2 py-0.5 text-[10px] text-text-muted"
                    >
                      {branchForkPointLabel}
                    </span>
                  )}
                  {branchDepthLabel && (
                    <span
                      data-testid="playground-branch-depth"
                      className="inline-flex items-center rounded-full border border-border bg-surface2 px-2 py-0.5 text-[10px] text-text-muted"
                    >
                      {branchDepthLabel}
                    </span>
                  )}
                </div>
              </div>
            )}
            <div className="px-4 pt-2">
              <div className="mx-auto flex w-full max-w-[64rem] items-center justify-between text-[11px] text-text-muted">
                <div className="flex min-w-0 flex-wrap items-center gap-1.5">
                  <span
                    data-testid="playground-active-chat-mode"
                    className={`inline-flex max-w-full items-center rounded-full border px-2 py-0.5 font-medium ${
                      characterWorkflowActive
                        ? "border-primary/40 bg-primary/10 text-primaryStrong"
                        : "border-border bg-surface2 text-text-muted"
                    }`}
                  >
                    {characterWorkflowActive
                      ? toText(
                          t(
                            "playground:characterChat.modeLabel",
                            "Character Chat",
                          ),
                        )
                      : toText(
                          t("playground:regions.standardChat", "Standard chat"),
                        )}
                    {characterWorkflowActive && activeCharacterModeLabel ? (
                      <span className="ml-1 max-w-[14rem] truncate text-text">
                        {activeCharacterModeLabel}
                      </span>
                    ) : null}
                  </span>
                  <span className="inline-flex items-center rounded-full border border-border bg-surface2 px-2 py-0.5">
                    {t("playground:regions.timeline", "Conversation timeline")}
                  </span>
                </div>
                <div className="flex items-center gap-1.5">
                  <button
                    ref={shortcutsTriggerRef}
                    type="button"
                    data-testid="playground-shortcuts-help-trigger"
                    onClick={() =>
                      setShortcutsHelpOpen((previous) => !previous)
                    }
                    title={
                      t(
                        "playground:shortcuts.openHelp",
                        "Open keyboard shortcuts (Shift+/)",
                      ) as string
                    }
                    className="inline-flex items-center gap-1 rounded-full border border-border bg-surface2 px-2 py-0.5 text-text hover:bg-surface"
                  >
                    <Keyboard className="h-3 w-3" aria-hidden="true" />
                    {t("playground:shortcuts.title", "Shortcuts")}
                  </button>
                  <button
                    ref={artifactsTriggerRef}
                    type="button"
                    data-testid="playground-artifacts-trigger"
                    disabled={!activeArtifact && !artifactsOpen}
                    onClick={() => {
                      if (artifactsOpen) {
                        closeArtifacts();
                        return;
                      }
                      if (!activeArtifact) {
                        return;
                      }
                      setArtifactsOpen(true);
                      markArtifactsRead();
                    }}
                    title={artifactBadgeLabel as string}
                    className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 transition ${
                      !activeArtifact && !artifactsOpen
                        ? "cursor-not-allowed border-border bg-surface text-text-subtle opacity-70"
                        : "border-border bg-surface2 text-text hover:bg-surface"
                    }`}
                  >
                    <span>{artifactBadgeLabel}</span>
                    {artifactUnreadCount > 0 && (
                      <span
                        data-testid="playground-artifacts-unread"
                        className="rounded-full bg-primary px-1.5 py-0.5 text-[10px] font-semibold text-white"
                      >
                        {toText(
                          t(
                            "playground:regions.artifactsNew",
                            "New {{count}}",
                            {
                              count: artifactUnreadCount,
                            } as any,
                          ),
                        )}
                      </span>
                    )}
                    {artifactPinnedCount > 0 && (
                      <span
                        data-testid="playground-artifacts-pinned"
                        className="rounded-full border border-border bg-surface px-1.5 py-0.5 text-[10px] font-medium text-text-subtle"
                      >
                        {toText(
                          t(
                            "playground:regions.artifactsPinned",
                            "Pinned {{count}}",
                            {
                              count: artifactPinnedCount,
                            } as any,
                          ),
                        )}
                      </span>
                    )}
                    {artifactHistoryCount > 0 && (
                      <span
                        data-testid="playground-artifacts-count"
                        className="rounded-full border border-border bg-surface px-1.5 py-0.5 text-[10px] text-text-subtle"
                      >
                        {toText(
                          t(
                            "playground:regions.artifactsCount",
                            "{{count}} total",
                            {
                              count: artifactHistoryCount,
                            } as any,
                          ),
                        )}
                      </span>
                    )}
                  </button>
                </div>
              </div>
              {shortcutsHelpOpen && (
                <div
                  data-testid="playground-shortcuts-help-panel"
                  role="dialog"
                  aria-modal="false"
                  aria-label={t("playground:shortcuts.title", "Shortcuts")}
                  className="mx-auto mt-1 w-full max-w-[64rem] rounded-md border border-border bg-surface2 px-2 py-1.5 text-[11px] text-text"
                >
                  <div className="mb-1 flex items-center justify-between gap-2">
                    <span className="font-semibold">
                      {t("playground:shortcuts.title", "Shortcuts")}
                    </span>
                    <button
                      ref={shortcutsCloseRef}
                      type="button"
                      data-testid="playground-shortcuts-help-close"
                      onClick={() => {
                        setShortcutsHelpOpen(false);
                        requestAnimationFrame(() => {
                          shortcutsTriggerRef.current?.focus();
                        });
                      }}
                      className="rounded border border-border bg-surface px-2 py-0.5 text-[10px] font-medium text-text hover:bg-surface2"
                    >
                      {t("common:close", "Close")}
                    </button>
                  </div>
                  <div className="grid gap-1 sm:grid-cols-2">
                    <p>
                      <span className="font-medium">Shift+Esc</span>{" "}
                      {t(
                        "playground:shortcuts.focusComposer",
                        "Focus composer",
                      )}
                    </p>
                    <p>
                      <span className="font-medium">
                        {t("playground:shortcuts.findCombo", "Cmd/Ctrl+F")}
                      </span>{" "}
                      {t(
                        "playground:shortcuts.searchThread",
                        "Search this thread",
                      )}
                    </p>
                    <p>
                      <span className="font-medium">
                        {t("playground:shortcuts.helpCombo", "Shift+/")}
                      </span>{" "}
                      {t(
                        "playground:shortcuts.openHelp",
                        "Open keyboard shortcuts (Shift+/)",
                      )}
                    </p>
                    <p>
                      <span className="font-medium">Alt+Shift+A</span>{" "}
                      {t(
                        "playground:shortcuts.toggleArtifacts",
                        "Toggle artifacts panel",
                      )}
                    </p>
                    <p>
                      <span className="font-medium">Alt+Shift+C</span>{" "}
                      {t(
                        "playground:shortcuts.toggleCompare",
                        "Toggle compare mode",
                      )}
                    </p>
                    <p>
                      <span className="font-medium">Alt+Shift+M</span>{" "}
                      {t(
                        "playground:shortcuts.toggleModes",
                        "Open mode launcher",
                      )}
                    </p>
                    <p>
                      <span className="font-medium">Alt+Shift+← / →</span>{" "}
                      {t(
                        "playground:shortcuts.variantSwitch",
                        "Switch response variant",
                      )}
                    </p>
                    <p>
                      <span className="font-medium">Alt+Shift+B / R</span>{" "}
                      {t(
                        "playground:shortcuts.branchRegenerate",
                        "Fork branch / regenerate",
                      )}
                    </p>
                  </div>
                </div>
              )}
              {threadSearchOpen && (
                <div className="mx-auto mt-1 flex w-full max-w-[64rem] flex-wrap items-center gap-2 rounded-md border border-border bg-surface2 px-2 py-1">
                  <div className="relative min-w-[200px] flex-1">
                    <Search
                      className="pointer-events-none absolute left-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-text-subtle"
                      aria-hidden="true"
                    />
                    <input
                      ref={threadSearchInputRef}
                      value={threadSearchQuery}
                      onChange={(event) => {
                        setThreadSearchQuery(event.target.value);
                        setThreadSearchActiveIndex(0);
                      }}
                      onKeyDown={(event) => {
                        if (event.key === "Enter") {
                          event.preventDefault();
                          stepThreadSearchMatch(event.shiftKey ? -1 : 1);
                        }
                      }}
                      placeholder={t(
                        "playground:search.placeholder",
                        "Search messages in this conversation",
                      )}
                      className="h-7 w-full rounded border border-border bg-surface pl-7 pr-2 text-xs text-text placeholder:text-text-subtle focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
                    />
                  </div>
                  <span
                    className="rounded-full border border-border bg-surface px-2 py-0.5 text-[10px] text-text-subtle"
                    aria-live="polite"
                  >
                    {threadSearchMatches.length > 0
                      ? toText(
                          t(
                            "playground:search.matchCount",
                            "{{current}} / {{total}}",
                            {
                              current: Math.min(
                                threadSearchActiveIndex + 1,
                                threadSearchMatches.length,
                              ),
                              total: threadSearchMatches.length,
                            } as any,
                          ),
                        )
                      : toText(t("playground:search.noMatches", "No matches"))}
                  </span>
                  <button
                    type="button"
                    onClick={() => stepThreadSearchMatch(-1)}
                    disabled={threadSearchMatches.length === 0}
                    className={`rounded border px-2 py-0.5 text-[10px] font-medium ${
                      threadSearchMatches.length === 0
                        ? "cursor-not-allowed border-border bg-surface text-text-subtle opacity-60"
                        : "border-border bg-surface text-text hover:bg-surface2"
                    }`}
                  >
                    {t("common:previous", "Previous")}
                  </button>
                  <button
                    type="button"
                    onClick={() => stepThreadSearchMatch(1)}
                    disabled={threadSearchMatches.length === 0}
                    className={`rounded border px-2 py-0.5 text-[10px] font-medium ${
                      threadSearchMatches.length === 0
                        ? "cursor-not-allowed border-border bg-surface text-text-subtle opacity-60"
                        : "border-border bg-surface text-text hover:bg-surface2"
                    }`}
                  >
                    {t("common:next", "Next")}
                  </button>
                  <button
                    type="button"
                    onClick={() => setThreadSearchOpen(false)}
                    title={t("common:close", "Close") as string}
                    className="inline-flex items-center rounded border border-border bg-surface px-2 py-0.5 text-[10px] font-medium text-text hover:bg-surface2"
                  >
                    <X className="mr-1 h-3 w-3" aria-hidden="true" />
                    {t("common:close", "Close")}
                  </button>
                </div>
              )}
              {compactFeatureNoticeVisible && (
                <div
                  data-testid="playground-mobile-parity-notice"
                  className="mx-auto mt-1 w-full max-w-[64rem] rounded-md border border-warn/30 bg-warn/10 px-2 py-1 text-[10px] text-warn"
                >
                  {t(
                    "playground:regions.compactFeatureNotice",
                    "Limited on this device: compare and branch workflows use compact controls. Use full-chat opens from model cards for detailed review.",
                  )}
                </div>
              )}
              {characterWorkflowActive ? (
                <CharacterChatReadinessPanel
                  readiness={characterChatReadiness}
                  characterName={activeCharacterModeLabel}
                  missingCharacter={routeCharacterRecovery}
                  onAction={handleCharacterChatReadinessAction}
                  onChooseCharacter={openCharacterSelectorFromReadiness}
                  onRetryMissingCharacter={retryRouteCharacterRecovery}
                />
              ) : null}
            </div>
            <div
              ref={containerRef}
              data-testid={
                stickyChatInput ? "playground-chat-transcript" : undefined
              }
              role="log"
              aria-live="polite"
              aria-relevant="additions"
              aria-label={t("playground:aria.chatTranscript", "Chat messages")}
              className="custom-scrollbar flex-1 min-h-0 w-full overflow-x-hidden overflow-y-auto px-4"
            >
              <div className="mx-auto w-full max-w-[64rem] pb-6">
                <ChatErrorBoundary>
                  <PlaygroundChat
                    showStarterDeck={showStarterDeck}
                    searchQuery={threadSearchQuery.trim()}
                    matchedMessageIndices={threadSearchMatchSet}
                    activeSearchMessageIndex={threadSearchActiveMessageIndex}
                    onAttachResearchContext={handleAttachResearchContext}
                    onPrepareResearchFollowUp={handlePrepareResearchFollowUp}
                    returnedResearchRunId={pendingReturnedResearchRunId}
                    onDismissReturnedResearchRun={() => {
                      if (!pendingReturnedResearchRunId) {
                        return;
                      }
                      setDismissedReturnedResearchRunId(
                        pendingReturnedResearchRunId,
                      );
                      setPendingReturnedResearchRunId(null);
                    }}
                  />
                </ChatErrorBoundary>
              </div>
            </div>
            <div
              ref={composerDockRef}
              data-testid={
                stickyChatInput ? "playground-chat-composer-dock" : undefined
              }
              className={`relative w-full shrink-0 ${
                stickyChatInput
                  ? "sticky bottom-0 z-20 border-t border-border bg-surface/95 backdrop-blur"
                  : ""
              }`}
            >
              <div className="mx-auto w-full max-w-[64rem] px-4 pt-2 text-[11px] text-text-muted">
                <span className="inline-flex items-center rounded-full border border-border bg-surface2 px-2 py-0.5">
                  {t("playground:regions.composer", "Composer")}
                </span>
              </div>
              {!isAutoScrollToBottom && (
                <div className="pointer-events-none absolute -top-12 left-0 right-0 flex justify-center">
                  <button
                    onClick={() => autoScrollToBottom()}
                    aria-label={t(
                      "playground:composer.scrollToLatest",
                      "Scroll to latest messages",
                    )}
                    title={
                      t(
                        "playground:composer.scrollToLatest",
                        "Scroll to latest messages",
                      ) as string
                    }
                    className="pointer-events-auto rounded-full border border-border bg-surface p-2.5 text-text-subtle shadow-md transition-all duration-200 animate-in fade-in zoom-in-95 hover:bg-surface2 focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
                  >
                    <ChevronDown
                      className="size-4 text-text-subtle"
                      aria-hidden="true"
                    />
                  </button>
                </div>
              )}
              <PlaygroundForm
                droppedFiles={droppedFiles}
                stickyDockEnabled={stickyChatInput}
                onComposerLayoutChange={
                  stickyChatInput ? handleComposerLayoutChange : undefined
                }
                attachedResearchContext={attachedResearchContext}
                attachedResearchContextBaseline={
                  attachedResearchContextBaseline
                }
                attachedResearchContextPinned={attachedResearchContextPinned}
                attachedResearchContextHistory={attachedResearchContextHistory}
                onApplyAttachedResearchContext={
                  handleApplyAttachedResearchContext
                }
                onResetAttachedResearchContext={
                  handleResetAttachedResearchContext
                }
                onRemoveAttachedResearchContext={
                  handleRemoveAttachedResearchContext
                }
                onPinAttachedResearchContext={handlePinAttachedResearchContext}
                onPinAttachedResearchContextHistory={
                  handlePinAttachedResearchContextHistory
                }
                onUnpinAttachedResearchContext={
                  handleUnpinAttachedResearchContext
                }
                onRestorePinnedResearchContext={
                  handleRestorePinnedResearchContext
                }
                onPrepareResearchFollowUp={handlePrepareResearchFollowUp}
                onSelectAttachedResearchContextHistory={
                  handleSelectAttachedResearchContextHistory
                }
                onDraftPresenceChange={handleComposerDraftPresenceChange}
                characterChatSendBlocker={characterChatSendBlocker}
                characterChatModelUsability={activeCharacterChatModelUsability}
                characterChatModelUsabilityLabel={
                  characterChatModelSelectorLabel
                }
                characterChatModelUsabilityTitle={
                  characterChatModelSelectorTitle
                }
              />
            </div>
          </div>
        </PlaygroundCockpitShell>
        {artifactsOpen && (
          <>
            <div className="hidden h-full w-[36%] min-w-[280px] max-w-[520px] shrink-0 lg:flex">
              {renderArtifactsPanel()}
            </div>
            <div className="lg:hidden">
              <button
                type="button"
                aria-label={
                  t(
                    "playground:regions.closeArtifactsDrawer",
                    "Close artifacts drawer",
                  ) as string
                }
                title={
                  t(
                    "playground:regions.closeArtifactsDrawer",
                    "Close artifacts drawer",
                  ) as string
                }
                onClick={closeArtifactsWithFocusReturn}
                className="fixed inset-0 z-40 bg-black/40"
              />
              <div
                data-testid="playground-mobile-artifacts-sheet"
                role="dialog"
                aria-modal="true"
                aria-label={t(
                  "playground:regions.artifacts",
                  "Artifacts panel",
                )}
                className="fixed inset-y-0 right-0 z-50 flex w-full max-w-[520px] flex-col border-l border-border bg-surface"
              >
                <div className="flex items-center justify-between border-b border-border px-3 py-2 text-xs text-text">
                  <span
                    data-testid="playground-mobile-artifacts-title"
                    className="font-semibold"
                  >
                    {t("playground:regions.artifacts", "Artifacts panel")}
                  </span>
                  <button
                    type="button"
                    data-testid="playground-mobile-artifacts-return"
                    onClick={closeArtifactsWithFocusReturn}
                    className="rounded border border-border bg-surface2 px-2 py-0.5 text-[11px] font-medium text-text hover:bg-surface"
                  >
                    {t(
                      "playground:regions.returnToTimeline",
                      "Back to timeline",
                    )}
                  </button>
                </div>
                <div className="min-h-0 flex-1">{renderArtifactsPanel()}</div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
};
