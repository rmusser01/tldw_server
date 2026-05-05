import React from "react"
import { type StatePanelProps, StatePanel } from "./StatePanel"

export type RecoveryState =
  | "unavailable"
  | "retrying"
  | "blocked"
  | "degraded"
  | "error"
  | "auth_required"
  | "setup_required"

export interface RecoveryCalloutProps extends Omit<StatePanelProps, "state"> {
  state: RecoveryState
}

export function RecoveryCallout(props: RecoveryCalloutProps) {
  return <StatePanel {...props} data-ds-component="RecoveryCallout" />
}
