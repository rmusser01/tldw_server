import React from "react"
import { useTranslation } from "react-i18next"
import { useSidepanelInit } from "~/hooks/useSidepanelInit"
import { platformConfig } from "@/config/platform"
import {
  BuddyShellHost,
  BuddyShellRenderContextProvider
} from "@/components/Common/PersonaBuddy"
import { QuickChatHelperButton } from "@/components/Common/QuickChatHelper"
import { patchStaticAntdNotificationCompat } from "@/utils/antd-notification-compat"
import { AppShell } from "./AppShell"
import { SidepanelRouteShell } from "@/routes/sidepanel-route-shell"
import {
  HashRouterWithFuture,
  SidepanelMemoryRouter
} from "./router-utils"

const PageHelpModal = React.lazy(() =>
  import("@/components/Common/PageHelpModal").then((m) => ({
    default: m.PageHelpModal
  }))
)

const WorkflowIntegrationHost = React.lazy(() =>
  import("@/components/Common/Workflow/WorkflowIntegrationHost").then((module) => ({
    default: module.WorkflowIntegrationHost
  }))
)

patchStaticAntdNotificationCompat()

export const SidepanelApp: React.FC = () => {
  const { direction, t } = useSidepanelInit({
    titleDefaultValue: "tldw Assistant — Sidebar"
  })
  const Router =
    platformConfig.routers.sidepanel === "hash"
      ? HashRouterWithFuture
      : SidepanelMemoryRouter
  const extras = (
    <>
      {platformConfig.features.showQuickChatHelper && <QuickChatHelperButton />}
      {platformConfig.features.showKeyboardShortcutsModal && (
        <React.Suspense fallback={null}>
          <PageHelpModal />
        </React.Suspense>
      )}
    </>
  )

  return (
    <BuddyShellRenderContextProvider>
      <AppShell
        router={Router}
        direction={direction}
        emptyDescription={t("common:noData", { defaultValue: "No data" })}
        suspendWhenHidden={platformConfig.features.suspendSidepanelWhenHidden}
        includeAntdApp={platformConfig.features.includeAntdApp}
        extras={
          <>
            {extras}
            <BuddyShellHost root="sidepanel" />
          </>
        }
      >
        <SidepanelRouteShell />
        <React.Suspense fallback={null}>
          <WorkflowIntegrationHost justChatPath="/" autoShow={false} />
        </React.Suspense>
      </AppShell>
    </BuddyShellRenderContextProvider>
  )
}

export default SidepanelApp
