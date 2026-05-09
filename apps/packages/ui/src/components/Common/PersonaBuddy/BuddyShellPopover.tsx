import React from "react"
import { useTranslation } from "react-i18next"
import { Palette } from "lucide-react"

import type { PersonaBuddySummary } from "@/types/persona-buddy"
import { buildPersonaGardenRoute } from "@/utils/persona-garden-route"

type BuddyShellPopoverProps = {
  buddySummary: PersonaBuddySummary
  personaId?: string | null
}

export const BuddyShellPopover: React.FC<BuddyShellPopoverProps> = ({
  buddySummary,
  personaId = null
}) => {
  const { t } = useTranslation("common")
  const visualsHref = buildPersonaGardenRoute({
    personaId,
    tab: "visuals"
  })

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
      <a
        data-testid="persona-buddy-open-visuals-link"
        href={visualsHref}
        className="mt-3 inline-flex items-center gap-1.5 rounded-md border border-border bg-surface px-2.5 py-1.5 text-xs font-medium text-text hover:bg-surface2"
      >
        <Palette aria-hidden="true" className="h-3.5 w-3.5" />
        <span>{t("personaBuddy.openVisuals", "Open Visuals")}</span>
      </a>
    </div>
  )
}

export default BuddyShellPopover
