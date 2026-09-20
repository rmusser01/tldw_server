import React from "react"
import { useTranslation } from "react-i18next"

import { AssistantDefaultsPanel } from "@/components/PersonaGarden/AssistantDefaultsPanel"
import {
  PersonaSetupAnalyticsCard,
  type PersonaSetupAnalyticsResponse
} from "@/components/PersonaGarden/PersonaSetupAnalyticsCard"
import type { PersonaVoiceAnalytics } from "@/components/PersonaGarden/CommandAnalyticsSummary"
import { PersonaSetupStatusCard } from "@/components/PersonaGarden/PersonaSetupStatusCard"
import type { PersonaSetupState } from "@/hooks/usePersonaSetupWizard"

import { buildPersonaSetupProgress } from "./personaSetupProgress"

type ProfilePanelProps = {
  selectedPersonaId: string
  selectedPersonaName: string
  personaCount: number
  connected: boolean
  sessionId: string | null
  setup?: PersonaSetupState | null
  onStartSetup?: () => void
  onResumeSetup?: () => void
  onResetSetup?: () => void
  onRerunSetup?: () => void
  onDefaultsSaved?: () => void
  isActive?: boolean
  setupAnalytics?: PersonaSetupAnalyticsResponse | null
  setupAnalyticsLoading?: boolean
  analytics?: PersonaVoiceAnalytics | null
  analyticsLoading?: boolean
  handoffFocusRequest?: {
    section: "assistant_defaults" | "confirmation_mode"
    token: number
  } | null
  onSetupHandoffFocusConsumed?: (token: number) => void
}

export const ProfilePanel: React.FC<ProfilePanelProps> = ({
  selectedPersonaId,
  selectedPersonaName,
  personaCount,
  connected,
  sessionId: _sessionId,
  setup = null,
  onStartSetup,
  onResumeSetup,
  onResetSetup,
  onRerunSetup,
  onDefaultsSaved,
  isActive = false,
  setupAnalytics = null,
  setupAnalyticsLoading = false,
  analytics = null,
  analyticsLoading = false,
  handoffFocusRequest = null,
  onSetupHandoffFocusConsumed
}) => {
  const { t } = useTranslation(["sidepanel", "common"])
  const setupProgressItems = React.useMemo(() => buildPersonaSetupProgress(setup), [setup])

  return (
    <div className="space-y-3">
      <div className="rounded-lg border border-border bg-surface p-3">
        <div className="text-base font-semibold text-text">
          {t("sidepanel:personaGarden.profile.heading", {
            defaultValue: "Persona Profile"
          })}
        </div>
        <div className="mt-2 space-y-2 text-sm text-text">
          <div>
            <div className="font-medium">
              {selectedPersonaName ||
                t("sidepanel:personaGarden.profile.noneSelected", {
                  defaultValue: "No persona selected"
                })}
            </div>
          </div>
          <div className="flex flex-wrap gap-3 text-xs text-text-muted">
            <span>
              {t("sidepanel:personaGarden.profile.catalogCount", {
                defaultValue: "Catalog personas: {{count}}",
                count: personaCount
              })}
            </span>
            <span>
              {connected
                ? t("sidepanel:personaGarden.profile.sessionConnected", {
                    defaultValue: "Session connected"
                  })
                : t("sidepanel:personaGarden.profile.sessionDisconnected", {
                    defaultValue: "Session disconnected"
                  })}
            </span>

          </div>
          <p className="text-xs text-text-muted">
            {t("sidepanel:personaGarden.profile.description", {
              defaultValue:
                "Edit this persona’s voice, behavior, and setup. Your connected conversation keeps its current persona."
            })}
          </p>
        </div>
      </div>
      <PersonaSetupStatusCard
        setup={setup}
        progressItems={setupProgressItems}
        onStartSetup={onStartSetup}
        onResumeSetup={onResumeSetup}
        onResetSetup={onResetSetup}
        onRerunSetup={onRerunSetup}
      />
      <PersonaSetupAnalyticsCard
        analytics={setupAnalytics}
        loading={setupAnalyticsLoading}
      />
      <AssistantDefaultsPanel
        selectedPersonaId={selectedPersonaId}
        selectedPersonaName={selectedPersonaName}
        isActive={isActive}
        analytics={analytics}
        analyticsLoading={analyticsLoading}
        handoffFocusRequest={handoffFocusRequest}
        onSetupHandoffFocusConsumed={onSetupHandoffFocusConsumed}
        onSaved={() => {
          onDefaultsSaved?.()
        }}
      />
    </div>
  )
}
