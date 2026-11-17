/**
 * Where the side panel opens (XS-17, #3111).
 *
 * Chat is the side panel's main job. Companion Home only helps once the
 * server offers personalization and the person has turned the Companion on;
 * otherwise it is a page of "Setup required" cards with Chat far below. So
 * the side panel opens to Companion Home only when the Companion is ready,
 * and to chat in every other case, including when the check fails.
 */
import { fetchPersonalizationProfile } from "@/services/companion"
import { getServerCapabilities } from "@/services/tldw/server-capabilities"
import { hasResumableSidepanelChat } from "./sidepanel-chat-resume"

export type SidepanelHomeTarget = "chat" | "companion"

/** True when the server offers personalization and the Companion is turned on. */
export const isSidepanelCompanionReady = async (): Promise<boolean> => {
  const capabilities = await getServerCapabilities()
  if (!capabilities?.hasPersonalization) return false
  const profile = await fetchPersonalizationProfile()
  return Boolean(profile?.enabled)
}

export const resolveSidepanelHomeTarget = async (): Promise<SidepanelHomeTarget> => {
  try {
    if (await hasResumableSidepanelChat()) return "chat"
    return (await isSidepanelCompanionReady()) ? "companion" : "chat"
  } catch {
    return "chat"
  }
}
