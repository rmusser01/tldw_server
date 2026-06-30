import React, { useEffect, useState } from "react"
import { HashRouter, MemoryRouter } from "react-router-dom"
import { useTranslation } from "react-i18next"
import { useSidepanelInit } from "~/hooks/useSidepanelInit"
import "~/i18n"
import { AppShell } from "./AppShell"
import { OptionsRouteShell } from "@/routes/options-route-shell"
import { SidepanelRouteShell } from "@/routes/sidepanel-route-shell"
import { platformConfig } from "@/config/platform"
import { QuickChatHelperButton } from "@/components/Common/QuickChatHelper"
import { WorkflowIntegrationHost } from "@/components/Common/Workflow"
import { patchStaticAntdNotificationCompat } from "@/utils/antd-notification-compat"

const PageHelpModal = React.lazy(() =>
  import("@/components/Common/PageHelpModal").then((m) => ({
    default: m.PageHelpModal
  }))
)

patchStaticAntdNotificationCompat()

const routerFutureConfig = {
  v7_startTransition: true,
  v7_relativeSplatPath: true
}

const HashRouterWithFuture: React.FC<{ children: React.ReactNode }> = ({
  children
}) => <HashRouter future={routerFutureConfig}>{children}</HashRouter>

const MemoryRouterWithFuture: React.FC<{ children: React.ReactNode }> = ({
  children
}) => <MemoryRouter future={routerFutureConfig}>{children}</MemoryRouter>

const resolveRouter = (mode: "hash" | "memory") =>
  mode === "hash" ? HashRouterWithFuture : MemoryRouterWithFuture

const resolveMemoryInitialEntry = () => {
  if (typeof window === "undefined") {
    return "/"
  }
  const rawHash = window.location.hash || ""
  const trimmed = rawHash.startsWith("#") ? rawHash.slice(1) : rawHash
  if (!trimmed || trimmed === "/") {
    return "/"
  }
  return trimmed.startsWith("/") ? trimmed : `/${trimmed}`
}

const SidepanelMemoryRouter: React.FC<{ children: React.ReactNode }> = ({
  children
}) => {
  const initialEntries = React.useMemo(
    () => [resolveMemoryInitialEntry()],
    []
  )
  return (
    <MemoryRouter
      initialEntries={initialEntries}
      future={routerFutureConfig}
    >
      {children}
    </MemoryRouter>
  )
}

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
    <AppShell
      router={Router}
      direction={direction}
      emptyDescription={t("common:noData", { defaultValue: "No data" })}
      suspendWhenHidden={platformConfig.features.suspendSidepanelWhenHidden}
      includeAntdApp={platformConfig.features.includeAntdApp}
      extras={extras}
    >
      <SidepanelRouteShell />
      <WorkflowIntegrationHost justChatPath="/" autoShow={false} />
    </AppShell>
  )
}

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
    >
      <OptionsRouteShell />
      <WorkflowIntegrationHost autoShowPaths={["/"]} />
    </AppShell>
  )
}
