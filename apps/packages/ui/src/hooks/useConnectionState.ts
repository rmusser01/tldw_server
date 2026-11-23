import { shallow } from "zustand/shallow"

import { useConnectionStore } from "@/store/connection"
import { deriveConnectionUxState, type ConnectionState } from "@/types/connection"

/**
 * Whole connection state, or a shallow-compared selection of it. Every health
 * poll replaces the state object, so a caller that reads only a few fields
 * should pass a selector to re-render only when those fields change.
 */
export function useConnectionState<T = ConnectionState>(
  selector?: (state: ConnectionState) => T
): T {
  return useConnectionStore(
    (s) => (selector ? selector(s.state) : (s.state as T)),
    selector ? shallow : undefined
  )
}

export const useConnectionPhase = () =>
  useConnectionStore((s) => s.state.phase)

export const useIsConnected = () =>
  useConnectionStore((s) => s.state.isConnected)

export const useConnectionActions = () =>
  useConnectionStore((s) => ({
    checkOnce: s.checkOnce,
    setServerUrl: s.setServerUrl,
    enableOfflineBypass: s.enableOfflineBypass,
    disableOfflineBypass: s.disableOfflineBypass,
    beginOnboarding: s.beginOnboarding,
    restartOnboarding: s.restartOnboarding,
    setConfigPartial: s.setConfigPartial,
    testConnectionFromOnboarding: s.testConnectionFromOnboarding,
    setDemoMode: s.setDemoMode,
    markFirstRunComplete: s.markFirstRunComplete,
    setUserPersona: s.setUserPersona
  }), shallow)

export const useKnowledgeStatus = () =>
  useConnectionStore((s) => ({
    knowledgeStatus: s.state.knowledgeStatus,
    knowledgeLastCheckedAt: s.state.knowledgeLastCheckedAt,
    knowledgeError: s.state.knowledgeError
  }), shallow)

export const useConnectionUxState = () =>
  useConnectionStore((s) => {
    const { mode, configStep, errorKind, hasCompletedFirstRun } = s.state
    const uxState = deriveConnectionUxState(s.state)
    const isConnectedUx =
      uxState === "connected_ok" ||
      uxState === "connected_degraded" ||
      uxState === "demo_mode"
    const isChecking = uxState === "testing"
    const isConfigOrError =
      uxState === "unconfigured" ||
      uxState === "configuring_url" ||
      uxState === "configuring_auth" ||
      uxState === "error_unreachable" ||
      uxState === "error_auth"

    return {
      uxState,
      mode,
      errorKind,
      configStep,
      hasCompletedFirstRun,
      isConnectedUx,
      isChecking,
      isConfigOrError
    }
  }, shallow)
