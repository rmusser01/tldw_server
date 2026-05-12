export const OPEN_KNOWLEDGE_PANEL_EVENT = "tldw:open-knowledge-panel";
export const OPEN_MODEL_SETTINGS_EVENT = "tldw:open-model-settings";
export const OPEN_ACTOR_SETTINGS_EVENT = "tldw:open-actor-settings";
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

export const toggleWebSearchFromCockpit = () => {
  dispatchCockpitEvent(TOGGLE_WEB_SEARCH_EVENT);
};

export const setTemporaryChatFromCockpit = (next: boolean) => {
  dispatchCockpitEvent(SET_TEMPORARY_CHAT_EVENT, { next });
};
