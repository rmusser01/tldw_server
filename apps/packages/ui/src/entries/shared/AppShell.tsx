import React, { useEffect, useRef, useState } from "react"
import { App as AntdApp, ConfigProvider, Empty } from "antd"
import { StyleProvider } from "@ant-design/cssinjs"
import { QueryClientProvider } from "@tanstack/react-query"
import { useTheme } from "@/hooks/useTheme"
import { PageAssistProvider } from "@/components/Common/PageAssistProvider"
import { useSplashScreen } from "@/hooks/useSplashScreen"
import { FontSizeProvider } from "@/context/FontSizeProvider"
import { getQueryClient } from "@/services/query-client"
import { SPLASH_TRIGGER_EVENT } from "@/services/splash-events"

type RouterComponent = React.ComponentType<{ children: React.ReactNode }>

type AppShellProps = {
  router: RouterComponent
  direction: "ltr" | "rtl"
  emptyDescription: string
  children: React.ReactNode
  extras?: React.ReactNode
  suspendWhenHidden?: boolean
  includeAntdApp?: boolean
}

const SplashOverlay = React.lazy(() =>
  import("@/components/Common/SplashScreen/SplashOverlay").catch((error) => {
    console.warn("Unable to load splash screen", error)
    return { default: () => null }
  })
)

const queryClient = getQueryClient()
const DevLocaleJsonDiagnostics = import.meta.env.DEV
  ? React.lazy(() =>
      import("@/components/Common/LocaleJsonDiagnostics").then((module) => ({
        default: module.LocaleJsonDiagnostics
      }))
    )
  : null

export const AppShell: React.FC<AppShellProps> = ({
  router: Router,
  direction,
  emptyDescription,
  children,
  extras,
  suspendWhenHidden = false,
  includeAntdApp = true
}) => {
  const { antdTheme } = useTheme()
  const splash = useSplashScreen()
  const portalRootRef = useRef<HTMLDivElement | null>(null)
  const getPopupContainer = React.useCallback(() => {
    if (typeof document === "undefined") return undefined
    return portalRootRef.current ?? document.body
  }, [])
  const [isVisible, setIsVisible] = useState(
    typeof document !== "undefined"
      ? document.visibilityState === "visible"
      : true
  )
  const [keepMountedWhileHidden, setKeepMountedWhileHidden] = useState(false)

  const hasOpenQuickIngestModal = React.useCallback(() => {
    if (typeof document === "undefined") return false
    return Boolean(
      document.querySelector(".quick-ingest-modal .ant-modal-content")
    )
  }, [])

  useEffect(() => {
    if (!suspendWhenHidden || typeof document === "undefined") return
    const handleVisibilityChange = () => {
      const visible = document.visibilityState === "visible"
      setIsVisible(visible)
      if (visible) {
        setKeepMountedWhileHidden(false)
        return
      }
      setKeepMountedWhileHidden(hasOpenQuickIngestModal())
    }

    document.addEventListener("visibilitychange", handleVisibilityChange)
    return () => {
      document.removeEventListener("visibilitychange", handleVisibilityChange)
    }
  }, [hasOpenQuickIngestModal, suspendWhenHidden])

  useEffect(() => {
    if (typeof window === "undefined") return
    const onSplashTrigger = (event: Event) => {
      const detail =
        event instanceof CustomEvent
          ? (event as CustomEvent<{ force?: boolean }>).detail
          : undefined
      splash.show({ force: detail?.force === true })
    }
    window.addEventListener(SPLASH_TRIGGER_EVENT, onSplashTrigger)
    return () =>
      window.removeEventListener(SPLASH_TRIGGER_EVENT, onSplashTrigger)
  }, [splash.show])

  const content = (
    <StyleProvider hashPriority="high">
      <QueryClientProvider client={queryClient}>
        <PageAssistProvider>
          <FontSizeProvider>
            {DevLocaleJsonDiagnostics ? (
              <React.Suspense fallback={null}>
                <DevLocaleJsonDiagnostics />
              </React.Suspense>
            ) : null}
            {suspendWhenHidden && !isVisible && !keepMountedWhileHidden
              ? null
              : children}
            {extras}
          </FontSizeProvider>
        </PageAssistProvider>
      </QueryClientProvider>
    </StyleProvider>
  )

  return (
    <Router>
      <ConfigProvider
        theme={antdTheme}
        getPopupContainer={getPopupContainer}
        renderEmpty={() => (
          <Empty
            styles={{ image: { height: 60 } }}
            description={emptyDescription}
          />
        )}
        direction={direction}>
        {includeAntdApp ? <AntdApp>{content}</AntdApp> : content}
        {splash.visible && splash.card ? (
          <React.Suspense fallback={null}>
            <SplashOverlay
              card={splash.card}
              message={splash.message}
              onDismiss={splash.dismiss}
            />
          </React.Suspense>
        ) : null}
        <div id="tldw-portal-root" ref={portalRootRef} />
      </ConfigProvider>
    </Router>
  )
}
