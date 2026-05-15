import {
  CANONICAL_STATE_KEYS,
  DESIGN_SYSTEM_STATES,
  EMPTY_STATE_LABEL,
  READY_STATE_LABEL,
  getDesignSystemState,
  isDesignSystemStateKey
} from "../states"
import { describe, expect, it } from "vitest"

const EXPECTED_STATES = {
  ready: {
    key: "ready",
    label: "Ready",
    severity: "success",
    iconRole: "success",
    token: "--state-ready",
    copyPattern: "Confirm the feature is available and the user can continue.",
    primaryAction: "continue",
    secondaryActions: [],
    diagnostics: "hidden_by_default",
    testExpectation: "Readable ready label is present."
  },
  unavailable: {
    key: "unavailable",
    label: "Unavailable",
    severity: "error",
    iconRole: "error",
    token: "--state-unavailable",
    copyPattern: "Name the unreachable target and give the shortest recovery path.",
    primaryAction: "retry",
    secondaryActions: ["reload", "open_diagnostics", "open_settings", "switch_server"],
    diagnostics: "visible_when_available",
    testExpectation: "Retry action and failing target are present."
  },
  setup_required: {
    key: "setup_required",
    label: "Setup required",
    severity: "warning",
    iconRole: "warning",
    token: "--state-setup-required",
    copyPattern: "Explain the missing setup step and link directly to setup.",
    primaryAction: "start_setup",
    secondaryActions: ["open_settings", "open_diagnostics"],
    diagnostics: "collapsed_unless_details",
    testExpectation: "Setup action is present."
  },
  auth_required: {
    key: "auth_required",
    label: "Sign in required",
    severity: "warning",
    iconRole: "warning",
    token: "--state-auth-required",
    copyPattern: "Tell the user which credential or account action is required.",
    primaryAction: "sign_in",
    secondaryActions: ["open_settings", "switch_account"],
    diagnostics: "collapsed_unless_details",
    testExpectation: "Auth action is present."
  },
  permission_denied: {
    key: "permission_denied",
    label: "Permission denied",
    severity: "error",
    iconRole: "error",
    token: "--state-permission-denied",
    copyPattern: "Explain the missing permission without exposing sensitive internals.",
    primaryAction: "request_access",
    secondaryActions: ["switch_account", "open_diagnostics"],
    diagnostics: "visible_when_available",
    testExpectation: "Permission label and recovery action are present."
  },
  degraded: {
    key: "degraded",
    label: "Degraded",
    severity: "warning",
    iconRole: "warning",
    token: "--state-degraded",
    copyPattern: "Name what still works and what dependency is limited.",
    primaryAction: "review_status",
    secondaryActions: ["open_diagnostics", "copy_diagnostics", "dismiss"],
    diagnostics: "visible_when_available",
    testExpectation: "Limitation and diagnostics affordance are present."
  },
  retrying: {
    key: "retrying",
    label: "Retrying",
    severity: "info",
    iconRole: "busy",
    token: "--state-retrying",
    copyPattern: "Announce that the system is trying again and avoid implying failure yet.",
    primaryAction: "wait",
    secondaryActions: ["cancel", "open_diagnostics"],
    diagnostics: "collapsed_unless_details",
    testExpectation: "Busy state is announced."
  },
  blocked: {
    key: "blocked",
    label: "Blocked",
    severity: "error",
    iconRole: "error",
    token: "--state-blocked",
    copyPattern: "Name the blocking reason and the next required fix.",
    primaryAction: "resolve_blocker",
    secondaryActions: ["open_diagnostics", "copy_diagnostics"],
    diagnostics: "visible_when_available",
    testExpectation: "Blocking reason and next action are present."
  },
  empty: {
    key: "empty",
    label: "Empty",
    severity: "neutral",
    iconRole: "neutral",
    token: "--state-empty",
    copyPattern: "Explain why there is no content and provide creation or import guidance.",
    primaryAction: "browse",
    secondaryActions: ["dismiss"],
    diagnostics: "hidden_by_default",
    testExpectation: "Empty state includes an action or explanation."
  },
  loading: {
    key: "loading",
    label: "Loading",
    severity: "neutral",
    iconRole: "busy",
    token: "--state-loading",
    copyPattern: "Name what is loading with a non-color-only busy signal.",
    primaryAction: "wait",
    secondaryActions: [],
    diagnostics: "hidden_by_default",
    testExpectation: "Loading state is accessible and non-color-only."
  },
  error: {
    key: "error",
    label: "Error",
    severity: "error",
    iconRole: "error",
    token: "--state-error",
    copyPattern: "Name the failure and show retry or diagnostics when available.",
    primaryAction: "retry",
    secondaryActions: ["reload", "open_diagnostics", "copy_diagnostics"],
    diagnostics: "visible_when_available",
    testExpectation: "Error label, retry path, and diagnostics affordance are present."
  }
}

describe("design-system state registry", () => {
  it("defines every v1 canonical state with stable labels and tokens", () => {
    expect(CANONICAL_STATE_KEYS).toEqual([
      "ready",
      "unavailable",
      "setup_required",
      "auth_required",
      "permission_denied",
      "degraded",
      "retrying",
      "blocked",
      "empty",
      "loading",
      "error"
    ])

    expect(getDesignSystemState("permission_denied")).toMatchObject({
      label: "Permission denied",
      severity: "error",
      token: "--state-permission-denied",
      primaryAction: "request_access"
    })
    expect(isDesignSystemStateKey("ready")).toBe(true)
    expect(isDesignSystemStateKey("healthy")).toBe(false)
    expect(isDesignSystemStateKey("toString")).toBe(false)
    expect(isDesignSystemStateKey("constructor")).toBe(false)
  })

  it("locks labels, severities, tokens, actions, diagnostics, and test expectations for every state", () => {
    expect(DESIGN_SYSTEM_STATES).toEqual(EXPECTED_STATES)
  })

  it("keeps state definitions aligned to the canonical key order", () => {
    expect(Object.keys(DESIGN_SYSTEM_STATES)).toEqual(CANONICAL_STATE_KEYS)
  })

  it("exports canonical state labels through defensive fallbacks", () => {
    expect(READY_STATE_LABEL).toBe(DESIGN_SYSTEM_STATES.ready.label)
    expect(EMPTY_STATE_LABEL).toBe(DESIGN_SYSTEM_STATES.empty.label)
  })
})
