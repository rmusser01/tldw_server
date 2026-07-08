import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

vi.mock("antd", () => ({
  App: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  ConfigProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  Empty: () => <div data-testid="empty-state" />
}))

vi.mock("@ant-design/cssinjs", () => ({
  StyleProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>
}))

vi.mock("@tanstack/react-query", () => ({
  QueryClientProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, options?: { defaultValue?: string }) =>
      options?.defaultValue ?? _key,
    i18n: {
      resolvedLanguage: "en",
      language: "en",
      dir: () => "ltr"
    }
  })
}))

vi.mock("@/hooks/useTheme", () => ({
  useTheme: () => ({ antdTheme: {} })
}))

vi.mock("@/components/Common/PageAssistProvider", () => ({
  PageAssistProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>
}))

vi.mock("@/context/FontSizeProvider", () => ({
  FontSizeProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>
}))

vi.mock("@/context/demo-mode", () => ({
  DemoModeProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>
}))

vi.mock("@/services/query-client", () => ({
  getQueryClient: () => ({})
}))

vi.mock("@/components/Common/SplashScreen", () => ({
  SplashOverlay: () => <div data-testid="splash-overlay" />
}))

vi.mock("@web/components/notifications/NotificationToastBridge", () => ({
  NotificationToastBridge: () => <div data-testid="notification-toast-bridge" />
}))

vi.mock("@/hooks/useSplashScreen", () => ({
  useSplashScreen: () => ({
    visible: false,
    card: null,
    message: null,
    dismiss: vi.fn(),
    show: vi.fn()
  })
}))

vi.mock("@/services/splash-events", () => ({
  SPLASH_TRIGGER_EVENT: "tldw:splash"
}))

vi.mock("@/utils/antd-notification-compat", () => ({
  patchStaticAntdNotificationCompat: vi.fn()
}))

vi.mock("@web/components/ui/ToastProvider", () => ({
  ToastProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>
}))

describe("AppProviders", () => {
  it("imports and renders with the web notification bridge path", async () => {
    const { AppProviders } = await import("@web/components/AppProviders")

    render(
      <AppProviders>
        <div>child content</div>
      </AppProviders>
    )

    expect(screen.getByText("child content")).toBeInTheDocument()
    expect(screen.getByTestId("notification-toast-bridge")).toBeInTheDocument()
  })

  it("keeps the notification bridge unmounted when notifications are disabled", async () => {
    const { AppProviders } = await import("@web/components/AppProviders")

    render(
      <AppProviders enableNotifications={false}>
        <div>child content</div>
      </AppProviders>
    )

    expect(screen.getByText("child content")).toBeInTheDocument()
    expect(screen.queryByTestId("notification-toast-bridge")).not.toBeInTheDocument()
  })
})
