export const CANONICAL_STATE_KEYS = [
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
] as const

export type DesignSystemStateKey = (typeof CANONICAL_STATE_KEYS)[number]

export type DesignSystemSeverity = "success" | "error" | "warning" | "info" | "neutral"

export type DesignSystemPrimaryAction =
  | "continue"
  | "retry"
  | "start_setup"
  | "sign_in"
  | "request_access"
  | "review_status"
  | "resolve_blocker"
  | "browse"
  | "wait"
  | "view_details"

export type DesignSystemSecondaryAction =
  | "reload"
  | "open_diagnostics"
  | "open_settings"
  | "switch_server"
  | "switch_account"
  | "copy_diagnostics"
  | "dismiss"
  | "cancel"
  | "request_access"

export type DesignSystemDiagnosticBehavior =
  | "hidden_by_default"
  | "visible_when_available"
  | "collapsed_unless_details"

export type DesignSystemIconRole =
  | "success"
  | "warning"
  | "error"
  | "info"
  | "neutral"
  | "busy"

export interface DesignSystemStateDefinition {
  key: DesignSystemStateKey
  label: string
  severity: DesignSystemSeverity
  iconRole: DesignSystemIconRole
  token: `--state-${string}`
  copyPattern: string
  primaryAction: DesignSystemPrimaryAction
  secondaryActions: DesignSystemSecondaryAction[]
  diagnostics: DesignSystemDiagnosticBehavior
  testExpectation: string
}

export const DESIGN_SYSTEM_STATES: Record<DesignSystemStateKey, DesignSystemStateDefinition> = {
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

export function getDesignSystemState(key: DesignSystemStateKey): DesignSystemStateDefinition {
  return DESIGN_SYSTEM_STATES[key]
}

export function getDesignSystemStateLabel(
  key: DesignSystemStateKey,
  fallback: string
): string {
  return getDesignSystemState(key)?.label ?? fallback
}

export const READY_STATE_LABEL = getDesignSystemStateLabel("ready", "Ready")
export const EMPTY_STATE_LABEL = getDesignSystemStateLabel("empty", "Empty")
export const LOADING_STATE_LABEL = getDesignSystemStateLabel("loading", "Loading")
export const DEGRADED_STATE_LABEL = getDesignSystemStateLabel("degraded", "Degraded")
export const ERROR_STATE_LABEL = getDesignSystemStateLabel("error", "Error")
export const BLOCKED_STATE_LABEL = getDesignSystemStateLabel("blocked", "Blocked")

export function isDesignSystemStateKey(value: unknown): value is DesignSystemStateKey {
  return (
    typeof value === "string" &&
    Object.prototype.hasOwnProperty.call(DESIGN_SYSTEM_STATES, value)
  )
}
