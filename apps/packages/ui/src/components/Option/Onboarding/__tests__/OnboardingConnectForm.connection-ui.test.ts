import { describe, expect, it } from "vitest"
import {
  buildAuthValidationFailureProgress,
  connectionUiReducer,
  initialConnectionUiState,
} from "../OnboardingConnectForm"

describe("OnboardingConnectForm connection UI gate", () => {
  it("keeps stale store results ignored while a candidate connection is only validating auth", () => {
    const started = connectionUiReducer(
      {
        ...initialConnectionUiState,
        hasRunConnectionTest: false,
        showSuccess: true,
      },
      { type: "START_CONNECT" }
    )

    expect(started.hasRunConnectionTest).toBe(false)
    expect(started.showSuccess).toBe(false)
    expect(started.progress.serverReachable).toBe("checking")
  })

  it("enables store-result handling only when the full backend readiness result is ready", () => {
    const started = connectionUiReducer(initialConnectionUiState, {
      type: "START_CONNECT",
    })
    const readyForStoreResult = connectionUiReducer(started, {
      type: "SET_HAS_RUN_TEST",
      hasRunConnectionTest: true,
    })

    expect(readyForStoreResult.hasRunConnectionTest).toBe(true)
  })

  it("marks server reachability failed when API key validation fails for connectivity", () => {
    const authChecking = connectionUiReducer(
      connectionUiReducer(initialConnectionUiState, { type: "START_CONNECT" }),
      {
        type: "UPDATE_PROGRESS",
        updater: (previous) => ({
          ...previous,
          serverReachable: "success",
          authentication: "checking",
        }),
      }
    )

    const failed = connectionUiReducer(authChecking, {
      type: "UPDATE_PROGRESS",
      updater: (previous) =>
        buildAuthValidationFailureProgress(previous, "refused"),
    })

    expect(failed.progress.serverReachable).toBe("error")
    expect(failed.progress.authentication).toBe("error")
  })
})
