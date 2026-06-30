import { STORAGE_KEYS } from "@/config/ui-constants"

export const SIDEPANEL_OVERLAY_RESUME_MARKER_PREFIX =
  "sidepanelChatOverlayResume"

export const getSidepanelDraftStorageKey = (
  tabId: string | null | undefined
) =>
  tabId
    ? `${STORAGE_KEYS.SIDEPANEL_CHAT_DRAFT}:${tabId}`
    : STORAGE_KEYS.SIDEPANEL_CHAT_DRAFT

export const getSidepanelOverlayResumeMarkerKey = (
  draftKey: string | null | undefined
) =>
  draftKey
    ? `${SIDEPANEL_OVERLAY_RESUME_MARKER_PREFIX}:${draftKey}`
    : null
