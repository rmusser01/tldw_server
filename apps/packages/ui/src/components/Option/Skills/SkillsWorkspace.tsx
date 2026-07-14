import React from "react"
import { useTranslation } from "react-i18next"
import { PageShell } from "@/components/Common/PageShell"
import { useServerCapabilities } from "@/hooks/useServerCapabilities"
import WorkspaceConnectionGate from "@/components/Common/WorkspaceConnectionGate"
import {
  RecoveryCallout,
  StatePanel,
  buildCapabilityState
} from "@/components/ui/state"
import { SkillsManager } from "./Manager"

export const SkillsWorkspace: React.FC = () => {
  const { t } = useTranslation(["option", "common"])
  const {
    capabilities,
    loading: capsLoading,
    refresh: refreshCapabilities
  } = useServerCapabilities()
  const hasSkills = capabilities?.hasSkills
  const unsupportedState = buildCapabilityState({
    featureName: "Skills",
    capabilityName: "Skills API",
    endpoint: "/api/v1/skills",
    method: "GET",
    reason: "unsupported",
    title: t("option:skillsEmpty.unavailableTitle", {
      defaultValue: "Skills are not available on this server"
    }),
    message: t("option:skillsEmpty.unavailableDescription", {
      defaultValue:
        "Update tldw_server to a build that advertises the Skills API, then refresh capabilities."
    })
  })
  const pageHeader = (
    <section className="mb-4 flex flex-col gap-1" aria-labelledby="skills-workspace-title">
      <h1 id="skills-workspace-title" className="m-0 text-xl font-semibold text-text">
        {t("option:skills.title", { defaultValue: "Skills" })}
      </h1>
      <p className="m-0 max-w-2xl text-sm text-text-muted">
        {t("option:skills.description", {
          defaultValue: "Discover, test, create, import, and manage reusable instructions."
        })}
      </p>
    </section>
  )

  return (
    <WorkspaceConnectionGate
      featureName={t("option:header.modeSkills", {
        defaultValue: "Skills"
      })}
      setupDescription={t("option:skillsEmpty.connectDescription", {
        defaultValue:
          "To use Skills, connect to your tldw server so skill definitions can be stored and executed."
      })}
      unreachableDescription={t("option:skillsEmpty.unreachableDescription", {
        defaultValue:
          "To use Skills, reconnect to your tldw server so skill definitions can be stored and executed."
      })}
      pageHeader={pageHeader}
    >
      {capsLoading ? (
        <PageShell>
          {pageHeader}
          <StatePanel
            state="loading"
            title={t("option:skillsEmpty.loadingTitle", {
              defaultValue: "Checking Skills API support"
            })}
            message={t("option:skillsEmpty.loadingDescription", {
              defaultValue:
                "The Skills manager will load after this server confirms Skills support."
            })}
            role="status"
            aria-live="polite"
            data-testid="skills-capability-loading"
          />
        </PageShell>
      ) : !hasSkills ? (
        <PageShell>
          {pageHeader}
          <RecoveryCallout
            state={unsupportedState.state}
            title={unsupportedState.title}
            message={unsupportedState.message}
            diagnostics={unsupportedState.diagnostics}
            primaryAction={{
              label: t("option:skillsEmpty.refreshCapabilities", {
                defaultValue: "Refresh capabilities"
              }),
              onClick: () => {
                void refreshCapabilities()
              }
            }}
            data-testid="skills-capability-state"
          />
        </PageShell>
      ) : (
        <PageShell>
          <SkillsManager />
        </PageShell>
      )}
    </WorkspaceConnectionGate>
  )
}
