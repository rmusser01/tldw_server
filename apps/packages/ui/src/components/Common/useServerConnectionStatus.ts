import { useTranslation } from "react-i18next"
import { getDesignSystemState } from "@/design-system"
import { useConnectionState } from "@/hooks/useConnectionState"
import { ConnectionPhase, deriveConnectionUxState } from "@/types/connection"
import { isAwaitingFirstConnectionCheck } from "@/components/Option/Playground/composer-connection"

export type ServerConnectionStatus = {
  /** "Server connected", "Connecting…", "Offline" or the degraded label. */
  label: string
  /** The server cannot take a send yet, so a model's own readiness is moot. */
  blocked: boolean
}

/**
 * The tldw server connection, named as the full chat page's model chip names
 * it (CC-01, CC-03). The side panel's model chip shows it when it has no
 * evidence about the selected model (#3343).
 */
export const useServerConnectionStatus = (): ServerConnectionStatus => {
  const { t } = useTranslation(["playground"])
  const state = useConnectionState()
  if (!state.isConnected || state.phase !== ConnectionPhase.CONNECTED) {
    return isAwaitingFirstConnectionCheck(state)
      ? { label: t("playground:composer.providerStatusConnecting", "Connecting…"), blocked: true }
      : { label: t("playground:composer.providerStatusOffline", "Offline"), blocked: true }
  }
  if (deriveConnectionUxState(state) === "connected_degraded") {
    return {
      label: t("playground:composer.providerStatusDegraded", getDesignSystemState("degraded")?.label ?? ""),
      blocked: false
    }
  }
  return { label: t("playground:composer.serverStatusConnected", "Server connected"), blocked: false }
}
