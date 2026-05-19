import React from "react"
import { type StatePanelProps, StatePanel } from "./StatePanel"

export interface SetupRequiredPanelProps extends Omit<StatePanelProps, "state"> {
  state?: "setup_required"
}

export function SetupRequiredPanel({ state = "setup_required", ...props }: SetupRequiredPanelProps) {
  return <StatePanel state={state} {...props} />
}
