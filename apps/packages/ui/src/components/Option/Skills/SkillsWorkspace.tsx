import React from "react"
import { Skeleton } from "antd"
import { useTranslation } from "react-i18next"
import { useNavigate } from "react-router-dom"
import FeatureEmptyState from "@/components/Common/FeatureEmptyState"
import { PageShell } from "@/components/Common/PageShell"
import { useServerCapabilities } from "@/hooks/useServerCapabilities"
import WorkspaceConnectionGate from "@/components/Common/WorkspaceConnectionGate"
import { SkillsManager } from "./Manager"

export const SkillsWorkspace: React.FC = () => {
  const { t } = useTranslation(["option", "common"])
  const navigate = useNavigate()
  const { capabilities, loading: capsLoading } = useServerCapabilities()
  const hasSkills = capabilities?.hasSkills

  return (
    <WorkspaceConnectionGate
      featureName={t("option:header.modeSkills", {
        defaultValue: "Skills"
      })}
      setupDescription={t("option:skillsEmpty.connectDescription", {
        defaultValue:
          "To use Skills, connect to your tldw server so skill definitions can be stored and executed."
      })}
    >
      {capsLoading ? (
        <PageShell>
          <Skeleton active />
        </PageShell>
      ) : !hasSkills ? (
        <PageShell>
          <FeatureEmptyState
            title={t("option:skillsEmpty.unavailableTitle", {
              defaultValue: "Skills not available"
            })}
            description={t("option:skillsEmpty.unavailableDescription", {
              defaultValue:
                "The connected server does not support the Skills API. Update the server to enable this feature."
            })}
            primaryActionLabel={t("option:skillsEmpty.checkServerSetup", {
              defaultValue: "Check server setup"
            })}
            onPrimaryAction={() => navigate("/settings/health")}
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
