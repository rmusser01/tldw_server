import React from "react"

import type { DesignSystemStateKey } from "@/design-system"
import { cn } from "@/libs/utils"

import {
  StatePanel,
  type StatePanelDiagnostic
} from "../ui/state/StatePanel"
import type {
  PersonaBuddyDiagnosticState,
  PersonaBuddyDiagnostics
} from "./personaBuddyDiagnostics"

export interface PersonaBuddyDiagnosticsPanelProps {
  diagnostics: PersonaBuddyDiagnostics
  className?: string
}

const panelStateByDiagnosticState: Record<
  PersonaBuddyDiagnosticState,
  DesignSystemStateKey
> = {
  healthy: "ready",
  unavailable: "unavailable",
  degraded: "degraded",
  recovering: "retrying"
}

const mapDiagnosticRows = (
  diagnostics: PersonaBuddyDiagnostics
): StatePanelDiagnostic[] =>
  diagnostics.rows.map((row) => {
    return {
      label: row.label,
      value: row.detail ? `${row.value} - ${row.detail}` : row.value
    }
  })

export const PersonaBuddyDiagnosticsPanel: React.FC<
  PersonaBuddyDiagnosticsPanelProps
> = ({ diagnostics, className }) => {
  return (
    <StatePanel
      state={panelStateByDiagnosticState[diagnostics.state]}
      title={diagnostics.title}
      message={diagnostics.message}
      diagnostics={mapDiagnosticRows(diagnostics)}
      className={cn("shadow-none", className)}
      data-testid="persona-buddy-diagnostics"
      aria-live={diagnostics.state === "recovering" ? "polite" : undefined}
    />
  )
}
