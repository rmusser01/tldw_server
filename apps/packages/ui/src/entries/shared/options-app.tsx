import React, { useEffect, useState } from "react"
import { useTranslation } from "react-i18next"
import { platformConfig } from "@/config/platform"
import { patchStaticAntdNotificationCompat } from "@/utils/antd-notification-compat"
import { AppShell } from "./AppShell"
import { OptionsRouteShell } from "@/routes/options-route-shell"
import { IndependentBuddyHost } from "@/components/Common/PersonaBuddy/IndependentBuddyHost"
import { resolveRouter } from "./router-utils"

const WorkflowIntegrationHost = React.lazy(() =>
  import("@/components/Common/Workflow/WorkflowIntegrationHost").then((module) => ({
    default: module.WorkflowIntegrationHost
  }))
)

patchStaticAntdNotificationCompat()

export const OptionsApp: React.FC = () => {
  const { t, i18n } = useTranslation()
  const [direction, setDirection] = useState<"ltr" | "rtl">("ltr")
  const Router = resolveRouter(platformConfig.routers.options)

  useEffect(() => {
    if (i18n.resolvedLanguage) {
      document.documentElement.lang = i18n.resolvedLanguage
      document.documentElement.dir = i18n.dir(i18n.resolvedLanguage)
      setDirection(i18n.dir(i18n.resolvedLanguage))
    }
  }, [i18n, i18n.resolvedLanguage])

  useEffect(() => {
    document.title = t("common:titles.options", {
      defaultValue: "tldw Assistant — Options"
    })
  }, [t])

  return (
    <AppShell
      router={Router}
      direction={direction}
      emptyDescription={t("common:noData", { defaultValue: "No data" })}
      suspendWhenHidden={platformConfig.features.suspendOptionsWhenHidden}
      includeAntdApp={platformConfig.features.includeAntdApp}
      extras={<IndependentBuddyHost />}
    >
      <OptionsRouteShell />
      <React.Suspense fallback={null}>
        <WorkflowIntegrationHost autoShowPaths={["/"]} />
      </React.Suspense>
    </AppShell>
  )
}

export default OptionsApp
