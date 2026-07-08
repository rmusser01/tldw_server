import React, { useEffect, useRef, useState } from "react"
import { App as AntdApp, ConfigProvider, Empty } from "antd"
import { StyleProvider } from "@ant-design/cssinjs"
import { QueryClientProvider } from "@tanstack/react-query"
import { useTranslation } from "react-i18next"
import { useTheme } from "@/hooks/useTheme"
import { PageAssistProvider } from "@/components/Common/PageAssistProvider"
import { FontSizeProvider } from "@/context/FontSizeProvider"
import { DemoModeProvider } from "@/context/demo-mode"
import { getQueryClient } from "@/services/query-client"
import { SplashOverlay } from "@/components/Common/SplashScreen"
import { NotificationToastBridge } from "@web/components/notifications/NotificationToastBridge"
import { useSplashScreen } from "@/hooks/useSplashScreen"
import { SPLASH_TRIGGER_EVENT } from "@/services/splash-events"
import { patchStaticAntdNotificationCompat } from "@/utils/antd-notification-compat"
import { ToastProvider } from "@web/components/ui/ToastProvider"

type AppProvidersProps = {
  children: React.ReactNode
  enableNotifications?: boolean
}

const queryClient = getQueryClient()
const EMPTY_STYLES = { image: { height: 60 } }
patchStaticAntdNotificationCompat()

export const AppProviders: React.FC<AppProvidersProps> = ({
  children,
  enableNotifications = true
}) => {
  const { antdTheme } = useTheme()
  const { i18n, t } = useTranslation("common")
  const splash = useSplashScreen()
  // SSR-safe: default to "ltr", update on client
  const [direction, setDirection] = useState<"ltr" | "rtl">("ltr")
  const portalRootRef = useRef<HTMLDivElement | null>(null)
  const renderEmpty = React.useCallback(
    () => (
      <Empty
        styles={EMPTY_STYLES}
        description={t("noData", { defaultValue: "No data" })}
      />
    ),
    [t]
  )

  const getPopupContainer = React.useCallback(
    (triggerNode?: HTMLElement) => {
      if (typeof document === "undefined") {
        return (triggerNode ?? ({} as HTMLElement))
      }
      return portalRootRef.current ?? triggerNode ?? document.body
    },
    []
  )

  useEffect(() => {
    if (typeof document === "undefined") return
    const language = i18n.resolvedLanguage || i18n.language
    document.documentElement.lang = language
    document.documentElement.dir = i18n.dir(language)
    setDirection(i18n.dir(language))
  }, [i18n, i18n.resolvedLanguage, i18n.language])

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
    return () => window.removeEventListener(SPLASH_TRIGGER_EVENT, onSplashTrigger)
  }, [splash.show])

  return (
    <StyleProvider hashPriority="high">
      <QueryClientProvider client={queryClient}>
        <DemoModeProvider>
          <PageAssistProvider>
            <FontSizeProvider>
              <ConfigProvider
                theme={antdTheme}
                getPopupContainer={getPopupContainer}
                renderEmpty={renderEmpty}
                direction={direction}
              >
                <ToastProvider>
                  {enableNotifications ? <NotificationToastBridge /> : null}
                  <AntdApp>{children}</AntdApp>
                  {splash.visible && splash.card && (
                    <SplashOverlay
                      card={splash.card}
                      message={splash.message}
                      onDismiss={splash.dismiss}
                    />
                  )}
                  <div id="tldw-portal-root" ref={portalRootRef} />
                </ToastProvider>
              </ConfigProvider>
            </FontSizeProvider>
          </PageAssistProvider>
        </DemoModeProvider>
      </QueryClientProvider>
    </StyleProvider>
  )
}
