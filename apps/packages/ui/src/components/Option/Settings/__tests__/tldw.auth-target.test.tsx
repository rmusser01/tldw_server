import React from "react"
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react"
import { App, ConfigProvider } from "antd"
import { Storage } from "@plasmohq/storage"
import { afterEach, beforeEach, expect, it, vi } from "vitest"
import type { TldwConfig } from "@/services/tldw/TldwApiClient"

// Match the WebUI storage boundary, including same-tab and cross-tab watches.
vi.mock(
  "@plasmohq/storage",
  async () =>
    import("../../../../../../../tldw-frontend/extension/shims/plasmo-storage"),
)

const mocks = vi.hoisted(() => ({
  getConfig: vi.fn(),
  apiSend: vi.fn(),
  logout: vi.fn(),
  bgRequest: vi.fn(),
  updateConfig: vi.fn(),
  login: vi.fn(),
  requestMagicLink: vi.fn(),
  verifyMagicLink: vi.fn(),
}))
vi.mock("@/services/background-proxy", () => ({ bgRequest: mocks.bgRequest }))
vi.mock("@/services/tldw/TldwApiClient", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/services/tldw/TldwApiClient")>()),
  tldwClient: {
    getConfig: mocks.getConfig,
    initialize: vi.fn(),
    updateConfig: mocks.updateConfig,
    ragHealth: vi.fn(),
  },
}))
vi.mock("@/services/tldw/TldwAuth", () => ({
  tldwAuth: {
    logout: mocks.logout,
    login: mocks.login,
    requestMagicLink: mocks.requestMagicLink,
    verifyMagicLink: mocks.verifyMagicLink,
  },
}))
vi.mock("@/services/tldw-server", () => ({
  DEFAULT_TLDW_API_KEY: "default-key",
}))
vi.mock("@/services/api-send", () => ({ apiSend: mocks.apiSend }))
vi.mock("@/store/connection", () => ({
  useConnectionStore: { getState: () => ({ checkOnce: vi.fn() }) },
}))
vi.mock("@/components/Common/ServerOverviewHint", () => ({
  ServerOverviewHint: () => null,
}))
vi.mock("../server-health-probe", () => ({ probeServerHealth: vi.fn() }))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback: unknown) =>
      typeof fallback === "string" ? fallback : key,
  }),
}))
vi.mock("react-router-dom", () => ({
  Link: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  useNavigate: () => vi.fn(),
}))

import { TldwSettings } from "../tldw"

const target: TldwConfig = {
  serverUrl: "https://settings.example.test/base",
  authMode: "multi-user",
}
let storage: Storage
let saved: TldwConfig | null
beforeEach(async () => {
  localStorage.clear()
  sessionStorage.clear()
  vi.clearAllMocks()
  storage = new Storage({ area: "local" })
  saved = target
  await storage.set("tldwConfig", saved)
  mocks.getConfig.mockImplementation(async () => saved)
  mocks.updateConfig.mockImplementation(async (config) => {
    saved = { ...saved, ...config }
    await storage.set("tldwConfig", saved)
    window.dispatchEvent(new CustomEvent("tldw:config-updated"))
  })
  mocks.bgRequest.mockResolvedValue({ paths: {} })
  mocks.login.mockResolvedValue({})
  mocks.requestMagicLink.mockResolvedValue(undefined)
  mocks.verifyMagicLink.mockResolvedValue({})
  vi.spyOn(console, "error").mockImplementation(() => {})
})
afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})
const mount = () =>
  render(
    <ConfigProvider theme={{ token: { motion: false } }}>
      <App>
        <TldwSettings />
      </App>
    </ConfigProvider>,
  )
const fillCredentials = async () => {
  fireEvent.change(await screen.findByLabelText("Username"), {
    target: { value: "alice" },
  })
  fireEvent.change(screen.getByPlaceholderText("Enter password"), {
    target: { value: "synthetic-password" },
  })
}
const login = () => screen.getByRole("button", { name: "Login" })

it("blocks all three auth actions against an unsaved full base URL", async () => {
  mount()
  await fillCredentials()
  fireEvent.change(screen.getByLabelText("Server URL"), {
    target: { value: "https://settings.example.test/other" },
  })
  expect(login()).toBeDisabled()
  expect(
    screen.getByText("Save connection settings before signing in."),
  ).toBeInTheDocument()
  fireEvent.click(login())
  expect(mocks.login).not.toHaveBeenCalled()
  fireEvent.click(screen.getByText("Magic link", { exact: true }))
  expect(screen.getByRole("button", { name: "Send magic link" })).toBeDisabled()
  expect(screen.getByRole("button", { name: "Verify & Login" })).toBeDisabled()
})

it("enables sign-in only after an explicit successful Save", async () => {
  mount()
  await fillCredentials()
  const serverUrl = "https://other.example.test/prefix"
  fireEvent.change(screen.getByLabelText("Server URL"), {
    target: { value: serverUrl },
  })
  expect(login()).toBeDisabled()
  fireEvent.click(screen.getByRole("button", { name: "common:save" }))
  await waitFor(() => expect(login()).toBeEnabled())
  fireEvent.click(login())
  await waitFor(() =>
    expect(mocks.login).toHaveBeenCalledWith(
      { username: "alice", password: "synthetic-password" },
      expect.objectContaining({
        target: { serverUrl, authMode: "multi-user" },
        signal: expect.any(AbortSignal),
      }),
    ),
  )
})

it("keeps an unsaved target guarded when Save persistence fails", async () => {
  mocks.updateConfig.mockRejectedValue(new Error("storage unavailable"))
  mount()
  await fillCredentials()
  fireEvent.change(screen.getByLabelText("Server URL"), {
    target: { value: "https://other.example.test" },
  })
  fireEvent.click(screen.getByRole("button", { name: "common:save" }))
  await waitFor(() => expect(mocks.updateConfig).toHaveBeenCalled())
  expect(login()).toBeDisabled()
  expect(mocks.login).not.toHaveBeenCalled()
})

it("permits normalized saved targets without saving unrelated timeout drafts", async () => {
  mount()
  await fillCredentials()
  fireEvent.change(screen.getByLabelText("Server URL"), {
    target: { value: " https://settings.example.test/base/// " },
  })
  await waitFor(() => expect(login()).toBeEnabled())
  fireEvent.click(login())
  await waitFor(() => expect(mocks.login).toHaveBeenCalledOnce())
  expect(mocks.updateConfig).not.toHaveBeenCalled()
})

it("keeps the draft but blocks sign-in after a cross-tab target change", async () => {
  mount()
  await fillCredentials()
  await waitFor(() => expect(login()).toBeEnabled())
  await act(async () => {
    saved = { ...target, serverUrl: "https://other.example.test" }
    await storage.set("tldwConfig", saved)
  })
  expect(screen.getByLabelText("Server URL")).toHaveValue(target.serverUrl)
  await waitFor(() => expect(login()).toBeDisabled())
})

it.each(["unmount", "ABA"])(
  "invalidates a pending handler on %s before config validation finishes",
  async (change) => {
    const view = mount()
    await fillCredentials()
    await waitFor(() => expect(login()).toBeEnabled())
    let release!: (value: TldwConfig) => void
    mocks.getConfig.mockReturnValueOnce(
      new Promise((resolve) => {
        release = resolve
      }),
    )
    fireEvent.click(login())
    await waitFor(() => expect(release).toBeTypeOf("function"))
    if (change === "unmount") view.unmount()
    else {
      fireEvent.change(screen.getByLabelText("Server URL"), {
        target: { value: "https://other.example.test" },
      })
      fireEvent.change(screen.getByLabelText("Server URL"), {
        target: { value: target.serverUrl },
      })
    }
    await act(async () => {
      release(target)
    })
    expect(mocks.login).not.toHaveBeenCalled()
  },
)

it.each([null, { ...target, authMode: "single-user" as const }])(
  "blocks sign-in when the displayed multi-user mode has no matching saved configuration: %j",
  async (initial) => {
    saved = initial
    if (initial) await storage.set("tldwConfig", initial)
    else await storage.remove("tldwConfig")
    mount()
    fireEvent.click(
      await screen.findByText("Multi User (Login)", { exact: true }),
    )
    fireEvent.change(screen.getByLabelText("Server URL"), {
      target: { value: target.serverUrl },
    })
    await fillCredentials()
    expect(login()).toBeDisabled()
    expect(
      screen.getByText("Save connection settings before signing in."),
    ).toBeInTheDocument()
    expect(mocks.login).not.toHaveBeenCalled()
  },
)

it("fails closed if the fresh handler configuration read fails", async () => {
  mount()
  await fillCredentials()
  await waitFor(() => expect(login()).toBeEnabled())
  mocks.getConfig.mockRejectedValueOnce(new Error("storage unavailable"))
  fireEvent.click(login())
  await waitFor(() => expect(console.error).toHaveBeenCalled())
  expect(mocks.login).not.toHaveBeenCalled()
})

it.each([
  {
    button: "Send magic link",
    input: "you@company.com",
    value: "alice@example.test",
    mock: mocks.requestMagicLink,
  },
  {
    button: "Verify & Login",
    input: "Paste the token from your email",
    value: "synthetic-magic",
    mock: mocks.verifyMagicLink,
  },
])(
  "passes the visible saved target to $button",
  async ({ button, input, value, mock }) => {
    mount()
    await screen.findByLabelText("Username")
    fireEvent.click(screen.getByText("Magic link", { exact: true }))
    fireEvent.change(screen.getByPlaceholderText(input), { target: { value } })
    await waitFor(() =>
      expect(screen.getByRole("button", { name: button })).toBeEnabled(),
    )
    fireEvent.click(screen.getByRole("button", { name: button }))
    await waitFor(() =>
      expect(mock).toHaveBeenCalledWith(
        value,
        expect.objectContaining({ target, signal: expect.any(AbortSignal) }),
      ),
    )
  },
)

it("rejects a pending Form preflight after an account A→B→A boundary", async () => {
  mount()
  await fillCredentials()
  await waitFor(() => expect(login()).toBeEnabled())
  let release!: (config: TldwConfig) => void
  mocks.getConfig.mockReturnValueOnce(
    new Promise((resolve) => {
      release = resolve
    }),
  )
  fireEvent.click(login())
  await waitFor(() => expect(release).toBeTypeOf("function"))
  await act(async () => {
    window.dispatchEvent(
      new CustomEvent("tldw:config-updated", {
        detail: { authorityChanged: true },
      }),
    )
    window.dispatchEvent(
      new CustomEvent("tldw:config-updated", {
        detail: { authorityChanged: true },
      }),
    )
    release(target)
  })
  expect(mocks.login).not.toHaveBeenCalled()
})


it("UAT378 saves a multi-user target before credentials are entered and still validates Login", async () => {
  mount()
  const serverUrl = "https://new.example.test/base"
  fireEvent.change(await screen.findByLabelText("Server URL"), { target: { value: serverUrl } })
  fireEvent.click(screen.getByRole("button", { name: "common:save" }))
  await waitFor(() => expect(mocks.updateConfig).toHaveBeenCalledWith(expect.objectContaining({serverUrl, authMode: "multi-user"})))
  await waitFor(() => expect(login()).toBeEnabled())
  expect(screen.queryByText("Please enter your username")).toBeNull()
  expect(screen.queryByText("Please enter your password")).toBeNull()
  fireEvent.click(login())
  await screen.findByText("Please enter your username")
  await screen.findByText("Please enter your password")
  expect(mocks.login).not.toHaveBeenCalled()
})

it("UAT378 keeps invalid connection URLs from being saved", async () => {
  mount()
  fireEvent.change(await screen.findByLabelText("Server URL"), { target: { value: "" } })
  fireEvent.click(screen.getByRole("button", { name: "common:save" }))
  await screen.findByText("Please enter the server URL")
  expect(mocks.updateConfig).not.toHaveBeenCalled()
})


it("UAT378 submits the connection form with empty login credentials", async () => {
  mount()
  const serverUrl = "https://keyboard.example.test/base"
  const input = await screen.findByLabelText("Server URL")
  fireEvent.change(input, { target: { value: serverUrl } })
  fireEvent.submit(input.closest("form")!)
  await waitFor(() => expect(mocks.updateConfig).toHaveBeenCalledWith(expect.objectContaining({serverUrl})))
  await waitFor(() => expect(login()).toBeEnabled())
  expect(mocks.login).not.toHaveBeenCalled()
})


it("UAT379 leaves health unchecked until first login after saving a connection", async () => {
  mocks.apiSend.mockResolvedValue({ ok: false, status: 401, error: "Not authenticated" })
  mount()
  fireEvent.change(await screen.findByLabelText("Server URL"), { target: { value: "https://first-login.example.test" } })
  fireEvent.click(screen.getByRole("button", { name: "common:save" }))
  await waitFor(() => expect(login()).toBeEnabled())
  expect(screen.getByText("Login Required")).toBeInTheDocument()
  expect(screen.getByText("Core: not checked yet")).toBeInTheDocument()
  expect(mocks.apiSend).not.toHaveBeenCalled()
})
