/**
 * Query parameters that carry a return to chat from Settings or a research
 * run. The Playground reads them once and removes them from the address.
 * Kept free of other imports so address helpers can use them cheaply.
 */
export const SETTINGS_HISTORY_ID_PARAM = "settingsHistoryId"
export const SETTINGS_SERVER_CHAT_ID_PARAM = "settingsServerChatId"
export const RESEARCH_RETURN_RUN_ID_PARAM = "researchReturnRunId"
