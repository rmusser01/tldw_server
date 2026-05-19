import React from "react"
import { useTranslation } from "react-i18next"
import { Palette } from "lucide-react"
import { Link } from "react-router-dom"

import type { PersonaBuddySummary } from "@/types/persona-buddy"
import { buildPersonaGardenRoute } from "@/utils/persona-garden-route"

import {
  getPersonaVisualDiagnosticToneClassName,
  type PersonaVisualDiagnostic
} from "./personaVisualDiagnostics"

type BuddyShellPopoverProps = {
  buddySummary: PersonaBuddySummary
  personaId?: string | null
  visualDiagnostic?: PersonaVisualDiagnostic | null
}

export const BuddyShellPopover: React.FC<BuddyShellPopoverProps> = ({
  buddySummary,
  personaId = null,
  visualDiagnostic = null
}) => {
  const { t } = useTranslation("common")
  const normalizedPersonaId = String(personaId ?? "").trim()
  const visualsRoute = normalizedPersonaId
    ? buildPersonaGardenRoute({
        personaId: normalizedPersonaId,
        tab: "visuals"
      })
    : null

  return (
    <div
      data-testid="persona-buddy-popover"
      className="min-w-[220px] rounded-2xl border border-border bg-bg/95 p-3 shadow-xl backdrop-blur"
    >
      <div className="text-xs uppercase tracking-[0.18em] text-text-muted">
        {t("personaBuddy.title", "Persona Buddy")}
      </div>
      <div className="mt-2 text-sm font-semibold text-text">
        {buddySummary.persona_name}
      </div>
      {buddySummary.role_summary ? (
        <div className="mt-1 text-xs leading-5 text-text-muted">
          {buddySummary.role_summary}
        </div>
      ) : null}
      {visualDiagnostic ? (
        <div
          data-testid="persona-buddy-visual-diagnostic-detail"
          data-severity={visualDiagnostic.severity}
          className={`mt-3 rounded-lg border px-2.5 py-2 text-xs leading-5 ${getPersonaVisualDiagnosticToneClassName(visualDiagnostic.severity)}`}
        >
          <div className="font-medium text-inherit">{visualDiagnostic.title}</div>
          <div>{visualDiagnostic.message}</div>
        </div>
      ) : null}
      {visualsRoute ? (
        <Link
          data-testid="persona-buddy-open-visuals-link"
          to={visualsRoute}
          className="mt-3 inline-flex items-center gap-1.5 rounded-md border border-border bg-surface px-2.5 py-1.5 text-xs font-medium text-text hover:bg-surface2"
        >
          <Palette aria-hidden="true" className="h-3.5 w-3.5" />
          <span>{t("personaBuddy.openVisuals", "Open Visuals")}</span>
        </Link>
      ) : null}
    </div>
  )
}

export default BuddyShellPopover
