import {
  dispatchOpenAssistantSelect,
  OPEN_ASSISTANT_SELECT_EVENT,
  type AssistantSelectTab,
  type AssistantSelectOpenDetail,
} from "@/utils/assistant-select-events";
import {
  dispatchOpenPromptSelect,
  OPEN_PROMPT_SELECT_EVENT,
  type PromptSelectOpenDetail,
} from "@/utils/prompt-select-events";

export {
  OPEN_ASSISTANT_SELECT_EVENT,
  type AssistantSelectOpenDetail,
} from "@/utils/assistant-select-events";
export {
  OPEN_PROMPT_SELECT_EVENT,
  type PromptSelectOpenDetail,
} from "@/utils/prompt-select-events";

export const OPEN_KNOWLEDGE_PANEL_EVENT = "tldw:open-knowledge-panel";
export const OPEN_MODEL_SETTINGS_EVENT = "tldw:open-model-settings";
export const OPEN_ACTOR_SETTINGS_EVENT = "tldw:open-actor-settings";
export const OPEN_MCP_TOOLS_EVENT = "tldw:open-mcp-tools";
export const OPEN_MCP_SETTINGS_EVENT = "tldw:open-mcp-settings";
export const OPEN_TURN_TOOLS_EVENT = "tldw:open-turn-tools";
export const TOGGLE_WEB_SEARCH_EVENT = "tldw:cockpit-toggle-web-search";
export const SET_TEMPORARY_CHAT_EVENT = "tldw:cockpit-set-temporary-chat";

export type SearchAndContextTab = "search" | "context";

const dispatchCockpitEvent = <TDetail>(
  eventName: string,
  detail?: TDetail,
) => {
  if (typeof window === "undefined") return;

  window.dispatchEvent(
    detail === undefined
      ? new CustomEvent(eventName)
      : new CustomEvent(eventName, { detail }),
  );
};

export const openSearchAndContext = (
  options: { tab?: SearchAndContextTab } = {},
) => {
  dispatchCockpitEvent(OPEN_KNOWLEDGE_PANEL_EVENT, {
    tab: options.tab ?? "search",
  });
};

export const openModelSettings = () => {
  dispatchCockpitEvent(OPEN_MODEL_SETTINGS_EVENT);
};

export const openActorSettings = () => {
  dispatchCockpitEvent(OPEN_ACTOR_SETTINGS_EVENT);
};

export const openAssistantSelector = (
  options: { tab?: AssistantSelectTab } = {},
) => {
  dispatchOpenAssistantSelect({
    tab: options.tab,
    source: "playground-cockpit",
  });
};

export const openPromptSelector = () => {
  dispatchOpenPromptSelect({
    source: "playground-cockpit",
  });
};

export const openMcpTools = () => {
  dispatchCockpitEvent(OPEN_MCP_TOOLS_EVENT);
};

export const openMcpSettings = () => {
  dispatchCockpitEvent(OPEN_MCP_SETTINGS_EVENT);
};

export const openTurnTools = () => {
  dispatchCockpitEvent(OPEN_TURN_TOOLS_EVENT);
};

export const toggleWebSearchFromCockpit = () => {
  dispatchCockpitEvent(TOGGLE_WEB_SEARCH_EVENT);
};

export const setTemporaryChatFromCockpit = (next: boolean) => {
  dispatchCockpitEvent(SET_TEMPORARY_CHAT_EVENT, { next });
};
