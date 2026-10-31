// @vitest-environment jsdom

import React from "react"
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor
} from "@testing-library/react"
import { App, ConfigProvider, Form, Modal, type FormInstance } from "antd"
import { Storage } from "@plasmohq/storage"
import type { TFunction } from "i18next"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { TldwConfig } from "@/services/tldw/TldwApiClient"
import { invalidateRefreshSessionIfCurrent } from "@/services/tldw/single-user-credential"
import { TldwConnectionSettings } from "../TldwConnectionSettings"
import { useSettingsLoginStatus } from "../useSettingsLoginStatus"

// Match the WebUI storage boundary, including same-tab and cross-tab watches.
vi.mock("@plasmohq/storage", async () =>
  import("../../../../../../../tldw-frontend/extension/shims/plasmo-storage")
)

const signedIn: TldwConfig = {
  serverUrl: "https://settings.example.test",
  authMode: "multi-user",
  authSource: "manual",
  accessToken: "current-access",
  refreshToken: "current-refresh"
}
const t = ((_key: string, fallback: string) => fallback) as TFunction

describe("Settings auth-mode changes through the actual Form", () => {
  let currentForm: FormInstance
  let storage: Storage

  beforeEach(async () => {
    localStorage.clear()
    sessionStorage.clear()
    storage = new Storage({ area: "local" })
    await storage.set("tldwConfig", signedIn)
  })

  afterEach(() => {
    Modal.destroyAll()
    cleanup()
    vi.restoreAllMocks()
  })

  function Owner() {
    const [form] = Form.useForm()
    React.useEffect(() => {
      currentForm = form
    }, [form])
    const [mode, setMode] = React.useState<"single-user" | "multi-user">(
      "multi-user"
    )
    const login = useSettingsLoginStatus(form, true)
    return (
      <Form
        form={form}
        initialValues={signedIn}
        onValuesChange={() => {
          void login.refreshLoginStatus()
        }}
      >
        <output data-testid="login-status">
          {login.isLoggedIn ? "authenticated" : "unauthenticated"}
        </output>
        <TldwConnectionSettings
          t={t}
          form={form}
          configuredServerUrl={signedIn.serverUrl}
          authSource="manual"
          rememberApiKey
          setRememberApiKey={vi.fn()}
          onManualServerOriginChange={vi.fn()}
          authMode={mode}
          setAuthMode={setMode}
          isLoggedIn={login.isLoggedIn}
          setIsLoggedIn={login.setIsLoggedIn}
          refreshLoginStatus={login.refreshLoginStatus}
          loginMethod="password"
          setLoginMethod={vi.fn()}
          magicEmail=""
          setMagicEmail={vi.fn()}
          magicToken=""
          setMagicToken={vi.fn()}
          magicSent={false}
          setMagicSent={vi.fn()}
          magicSending={false}
          testingConnection={false}
          logoutLoading={false}
          connectionStatus={null}
          connectionDetail=""
          coreStatus="unknown"
          ragStatus="unknown"
          onTestConnection={vi.fn()}
          onLogin={vi.fn()}
          onSendMagicLink={vi.fn()}
          onVerifyMagicLink={vi.fn()}
          onLogout={vi.fn()}
          onGrantSiteAccess={vi.fn()}
          onOpenHealthDiagnostics={vi.fn()}
        />
      </Form>
    )
  }

  const startModeChange = async () => {
    render(<ConfigProvider theme={{ token: { colorPrimary: "#3267ab", motion: false } }}><App><Owner /></App></ConfigProvider>)
    await screen.findByText("Logged In")
    fireEvent.click(
      screen.getByRole("radio", { name: "Single User (API Key)" })
    )
    await screen.findByText("Login Required")
  }

  const closeConfirmation = async (action: "Cancel" | "Change mode") => {
    fireEvent.click(screen.getByRole("button", { name: action, exact: true }))
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "Change authentication mode?" })).not.toBeInTheDocument(), { timeout: 3000 })
  }

  it("opens its real confirmation in the current application context without a static modal warning", async () => {
    const errors = vi.spyOn(console, "error").mockImplementation(() => {})
    await startModeChange()
    expect(screen.getByRole("dialog", { name: "Change authentication mode?" })).toBeInTheDocument()
    expect(errors.mock.calls.flat().map(String).join("\n")).not.toMatch(/Static function can not consume context/)
    await closeConfirmation("Cancel")
  })

  it("restores the authenticated notice after cancelling the mode change", async () => {
    await startModeChange()
    await closeConfirmation("Cancel")
    await screen.findByText("Logged In")
    expect(currentForm.getFieldValue("authMode")).toBe("multi-user")
  })

  it("keeps a confirmed single-user draft unauthenticated by the saved multi-user token", async () => {
    await startModeChange()
    await closeConfirmation("Change mode")
    expect(currentForm.getFieldValue("authMode")).toBe("single-user")
    expect(screen.getByTestId("login-status")).toHaveTextContent(
      /^unauthenticated$/
    )
    expect(screen.queryByText("Logged In")).not.toBeInTheDocument()
  })

  it("revalidates a confirmed return to the saved multi-user mode", async () => {
    await startModeChange()
    await closeConfirmation("Change mode")
    // A key draft requires confirmation before switching back to multi-user.
    act(() => {
      currentForm.setFieldValue("apiKey", "unsaved-key")
    })
    fireEvent.click(screen.getByRole("radio", { name: "Multi User (Login)" }))
    await waitFor(() =>
      expect(screen.getByTestId("login-status")).toHaveTextContent(
        /^authenticated$/
      )
    )
    await closeConfirmation("Change mode")
    await screen.findByText("Logged In")
    expect(currentForm.getFieldValue("apiKey")).toBe("")
  })

  it("does not restore an invalidated session when the mode change is cancelled", async () => {
    await startModeChange()
    await act(async () => {
      await invalidateRefreshSessionIfCurrent(storage, signedIn)
    })
    await closeConfirmation("Cancel")
    await waitFor(() =>
      expect(currentForm.getFieldValue("authMode")).toBe("multi-user")
    )
    expect(screen.getByText("Login Required")).toBeInTheDocument()
    expect(screen.queryByText("Logged In")).not.toBeInTheDocument()
  })
})
