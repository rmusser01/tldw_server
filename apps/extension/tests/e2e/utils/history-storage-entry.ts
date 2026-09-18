/** Test-only entry injected by Playwright. Every mutation below is the production API. */
export { db } from '@/db/dexie/schema'
export { historyDigest, getLocalHistoryOwner, captureLocalHistorySnapshot } from '@/db/dexie/history-selection'
export { getFullChatData, restoreChat, deleteByHistoryId, importChatHistoryV2, removeFileFromSession, addFileToSession, updateMessageById, removeMessageById } from '@/db/dexie/helpers'
export { getChatSettingsForKey, saveChatSettingsForKey, withPlainLocalForkSettings, chatSettingsStorageForKey } from '@/services/chat-settings'
export { saveHistoryBookmark, loadHistoryBookmark, savePendingHistoryConfirmation, markPendingHistoryConfirmationDispatched, saveHistoryTurnRecovery, loadHistoryTurnRecoveries } from '@/db/dexie/history-selection'
