import React from "react"
import { type StatePanelProps, StatePanel } from "./StatePanel"

export interface PermissionNoticeProps extends Omit<StatePanelProps, "state"> {
  state?: "permission_denied"
}

export function PermissionNotice({ state = "permission_denied", ...props }: PermissionNoticeProps) {
  return <StatePanel state={state} {...props} />
}
