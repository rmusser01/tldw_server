// @vitest-environment jsdom
/**
 * CC-03 (#3101): /chat used to open in a false "You're offline" state while
 * the first health check was still running, and Enter pressed then was
 * dropped without a word. While that first check runs, the composer now says
 * "Connecting…" and keeps the offline notice back until a check has actually
 * failed. Enter while not connected explains why nothing was sent and keeps
 * the draft (owner decision D7: no send outbox).
 */
import {
  act,
  createFakeTldwServer,
  renderPlayground,
  resetPlaygroundHarness,
  screen,
  visibleNotifications,
  waitFor
} from "./harness/playground-harness"
import { beforeEach, describe, expect, it } from "vitest"
import { useConnectionStore } from "@/store/connection"
import { ConnectionPhase, type ConnectionState } from "@/types/connection"

const setConnection = async (patch: Partial<ConnectionState>) => {
  await act(async () => {
    useConnectionStore.setState(({ state }) => ({ state: { ...state, ...patch } }))
  })
}

const FIRST_CHECK_IN_FLIGHT: Partial<ConnectionState> = {
  phase: ConnectionPhase.SEARCHING,
  isConnected: false,
  isChecking: true,
  lastCheckedAt: null,
  consecutiveFailures: 0,
  errorKind: "none"
}

const CHECK_FAILED: Partial<ConnectionState> = {
  phase: ConnectionPhase.ERROR,
  isConnected: false,
  isChecking: false,
  lastCheckedAt: Date.now(),
  consecutiveFailures: 1,
  errorKind: "unreachable"
}

describe("Playground while the server connection is pending", { timeout: 60_000 }, () => {
  beforeEach(async () => {
    await resetPlaygroundHarness()
  })

  it("says Connecting… rather than offline until the first check fails", async () => {
    const server = createFakeTldwServer()
    await renderPlayground({ server })

    await setConnection(FIRST_CHECK_IN_FLIGHT)

    expect(screen.queryByTestId("playground-composer-disconnected-notice")).toBeNull()
    expect(screen.queryByText(/You're offline/)).toBeNull()
    expect(screen.getByText("Connecting to your tldw server…")).toBeInTheDocument()
    expect(screen.getAllByRole("button", { name: /Connecting…/ }).length).toBeGreaterThan(0)
    expect(screen.queryAllByRole("button", { name: /Offline/ })).toHaveLength(0)

    await setConnection(CHECK_FAILED)

    expect(
      await screen.findByTestId("playground-composer-disconnected-notice")
    ).toBeInTheDocument()
    expect(screen.queryByText("Connecting to your tldw server…")).toBeNull()
    expect(screen.getAllByRole("button", { name: /Offline/ }).length).toBeGreaterThan(0)
  })

  it("explains why Enter did not send while connecting, and keeps the draft", async () => {
    const server = createFakeTldwServer()
    const view = await renderPlayground({ server })
    await setConnection(FIRST_CHECK_IN_FLIGHT)

    await view.user.click(view.composer)
    await view.user.type(view.composer, "hello there")
    await view.user.keyboard("{Enter}")

    await waitFor(() =>
      expect(visibleNotifications().join(" ")).toContain(
        "Still connecting to your tldw server"
      )
    )
    expect(view.composer).toHaveValue("hello there")
    expect(server.completionRequests()).toHaveLength(0)
  })

  it("explains why Enter did not send after a failed check", async () => {
    const server = createFakeTldwServer()
    const view = await renderPlayground({ server })
    await setConnection(CHECK_FAILED)

    await view.user.click(view.composer)
    await view.user.type(view.composer, "hello there")
    await view.user.keyboard("{Enter}")

    await waitFor(() =>
      expect(visibleNotifications().join(" ")).toContain(
        "Connect to your tldw server before sending"
      )
    )
    expect(view.composer).toHaveValue("hello there")
    expect(server.completionRequests()).toHaveLength(0)
  })
})
