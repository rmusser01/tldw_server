import React from "react"
import { useTranslation } from "react-i18next"
import {
  type DesignSystemSeverity,
  type DesignSystemStateKey,
  getDesignSystemState
} from "@/design-system"
import { cn } from "@/libs/utils"
import { ActionGroup, type StateAction } from "./ActionGroup"
import { DiagnosticRow, type DiagnosticRowProps } from "./DiagnosticRow"

export type StatePanelDiagnostic = Pick<DiagnosticRowProps, "label" | "value" | "code" | "copyLabel">

export interface StatePanelProps {
  state: DesignSystemStateKey
  title: React.ReactNode
  message?: React.ReactNode
  diagnostics?: StatePanelDiagnostic[]
  primaryAction?: StateAction
  secondaryActions?: StateAction[]
  className?: string
  children?: React.ReactNode
  role?: React.AriaRole
  "aria-live"?: React.AriaAttributes["aria-live"]
  "aria-atomic"?: React.AriaAttributes["aria-atomic"]
  "data-testid"?: string
  "data-ds-component"?: string
}

const severityClasses: Record<DesignSystemSeverity, string> = {
  success: "border-state-ready/30 bg-state-ready/10 text-state-ready",
  error: "border-state-error/30 bg-state-error/10 text-state-error",
  warning: "border-state-degraded/30 bg-state-degraded/10 text-state-degraded",
  info: "border-state-retrying/30 bg-state-retrying/10 text-state-retrying",
  neutral: "border-state-empty/30 bg-state-empty/10 text-state-empty"
}

const stateToneClasses: Partial<Record<DesignSystemStateKey, string>> = {
  unavailable: "border-state-unavailable/30 bg-state-unavailable/10 text-state-unavailable",
  setup_required: "border-state-setupRequired/30 bg-state-setupRequired/10 text-state-setupRequired",
  auth_required: "border-state-authRequired/30 bg-state-authRequired/10 text-state-authRequired",
  permission_denied:
    "border-state-permissionDenied/30 bg-state-permissionDenied/10 text-state-permissionDenied",
  blocked: "border-state-blocked/30 bg-state-blocked/10 text-state-blocked",
  loading: "border-state-loading/30 bg-state-loading/10 text-state-loading"
}

export function StatePanel({
  state,
  title,
  message,
  diagnostics,
  primaryAction,
  secondaryActions,
  className,
  children,
  role,
  "aria-live": ariaLive,
  "aria-atomic": ariaAtomic,
  "data-testid": dataTestId,
  "data-ds-component": dataDesignSystemComponent = "StatePanel"
}: StatePanelProps) {
  const { t } = useTranslation("common")
  const definition = getDesignSystemState(state)
  const toneClass = stateToneClasses[state] ?? severityClasses[definition.severity]
  const hasDiagnostics = diagnostics && diagnostics.length > 0
  const diagnosticsLabel = t("common:diagnostics", "Diagnostics")

  return (
    <section
      className={cn("rounded-lg border bg-surface p-4 text-text shadow-sm", className)}
      role={role}
      aria-live={ariaLive}
      aria-atomic={ariaAtomic}
      data-testid={dataTestId}
      data-ds-component={dataDesignSystemComponent}
    >
      <div className="flex flex-col gap-3">
        <div className="flex flex-col gap-2">
          <span
            className={cn(
              "inline-flex w-fit items-center rounded-full border px-2 py-0.5 text-xs font-semibold",
              toneClass
            )}
          >
            {definition.label}
          </span>
          <div>
            <h2 className="text-base font-semibold text-text">{title}</h2>
            {message ? <div className="mt-1 text-sm text-text-muted">{message}</div> : null}
          </div>
        </div>

        {children}

        {hasDiagnostics ? (
          <details className="rounded-md border border-border bg-surface2 px-3 py-2">
            <summary className="cursor-pointer text-xs font-semibold text-text-muted">
              {diagnosticsLabel}
            </summary>
            <dl aria-label={diagnosticsLabel} className="mt-2">
              {diagnostics.map((diagnostic, index) => (
                <DiagnosticRow key={index} {...diagnostic} />
              ))}
            </dl>
          </details>
        ) : null}

        <ActionGroup primaryAction={primaryAction} secondaryActions={secondaryActions} />
      </div>
    </section>
  )
}
