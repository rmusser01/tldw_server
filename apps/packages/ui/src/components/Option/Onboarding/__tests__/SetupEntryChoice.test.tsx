// @vitest-environment jsdom
import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import type { FirstRunMetadata, FirstRunState } from "@/types/setup-onboarding"
import { SetupEntryChoice } from "../SetupEntryChoice"

const firstRunState = (
  status: FirstRunState["status"] = "not_started"
): FirstRunState => ({
  status,
  completed_steps: [],
  skipped_steps: [],
  step_data: {},
  first_chat: { completed: false },
  acknowledged_steps: [],
})

const firstRunMetadata = (
  overrides: Partial<FirstRunMetadata> = {}
): FirstRunMetadata => {
  const { connection, ...rest } = overrides

  return {
    auth_mode: "single_user",
    bundled_single_user_auth_available: true,
    manual_auth_required: false,
    setup_required: true,
    setup_completed: false,
    remote_setup_enabled: false,
    connection: {
      frontend_origin: null,
      api_origin: "http://127.0.0.1:8000",
      browser_access: "local",
      ...connection,
    },
    setup_paths: [],
    multi_user_exit: {
      guide_path: "/docs/multi-user",
    },
    ...rest,
  }
}

const renderChoice = (
  props: Partial<React.ComponentProps<typeof SetupEntryChoice>> = {}
) => {
  const onStartWebUiSetup = vi.fn()
  const onRefreshSetupState = vi.fn()

  render(
    <SetupEntryChoice
      state={firstRunState()}
      metadata={firstRunMetadata()}
      currentOrigin="http://127.0.0.1:8080"
      onStartWebUiSetup={onStartWebUiSetup}
      onRefreshSetupState={onRefreshSetupState}
      {...props}
    />
  )

  return { onStartWebUiSetup, onRefreshSetupState }
}

describe("SetupEntryChoice", () => {
  it("renders one page heading for choosing where to set up tldw", () => {
    renderChoice()

    const headings = screen.getAllByRole("heading", {
      level: 1,
      name: /choose where to set up tldw/i,
    })

    expect(headings).toHaveLength(1)
  })

  it("explains that the user is in the WebUI and API server setup opens separately", () => {
    renderChoice()

    expect(
      screen.getByText(/you are in the tldw webui setup/i)
    ).toBeInTheDocument()
    expect(
      screen.getByText(/api server setup opens separately/i)
    ).toBeInTheDocument()
  })

  it("calls onStartWebUiSetup when mutable setup state uses Set up in WebUI", () => {
    const { onStartWebUiSetup } = renderChoice({
      state: firstRunState("in_progress"),
    })

    fireEvent.click(screen.getByRole("button", { name: "Set up in WebUI" }))

    expect(onStartWebUiSetup).toHaveBeenCalledTimes(1)
  })

  it("disables Set up in WebUI and shows recovery copy when setup state is blocked", () => {
    renderChoice({
      state: firstRunState("blocked"),
    })

    expect(screen.getByRole("button", { name: "Set up in WebUI" })).toBeDisabled()
    expect(
      screen.getByText(/webui setup can continue after recovery/i)
    ).toBeInTheDocument()
  })

  it("renders Open API server setup as a safe new-tab anchor", () => {
    renderChoice()

    const link = screen.getByRole("link", {
      name: /open api server setup.*opens in a new tab/i,
    })

    expect(link).toHaveAttribute("href", "http://127.0.0.1:8000/setup")
    expect(link).toHaveAttribute("target", "_blank")
    expect(link).toHaveAttribute("rel", "noopener noreferrer")
    expect(screen.getByText("http://127.0.0.1:8000/setup")).toBeInTheDocument()
  })

  it("reveals I finished API server setup after the API setup link is clicked", () => {
    renderChoice()

    expect(
      screen.queryByRole("button", { name: "I finished API server setup" })
    ).not.toBeInTheDocument()

    fireEvent.click(
      screen.getByRole("link", {
        name: /open api server setup.*opens in a new tab/i,
      })
    )

    expect(
      screen.getByRole("button", { name: "I finished API server setup" })
    ).toBeInTheDocument()
  })

  it("shows fallback guidance without an authoritative API setup link and keeps refresh visible", () => {
    renderChoice({
      metadata: firstRunMetadata({
        connection: { api_origin: "http://app:8000" },
      }),
      configuredServerUrl: null,
    })

    expect(
      screen.queryByRole("link", { name: /open api server setup/i })
    ).not.toBeInTheDocument()
    expect(
      screen.getByText(/open the api server setup page on the machine running tldw/i)
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "I finished API server setup" })
    ).toBeInTheDocument()
  })

  it("calls onRefreshSetupState from I finished API server setup", () => {
    const { onRefreshSetupState } = renderChoice({
      metadata: firstRunMetadata({
        connection: { api_origin: "http://app:8000" },
      }),
      configuredServerUrl: null,
    })

    fireEvent.click(
      screen.getByRole("button", { name: "I finished API server setup" })
    )

    expect(onRefreshSetupState).toHaveBeenCalledTimes(1)
  })

  it("explains local API setup access when browser access is local", () => {
    renderChoice({
      metadata: firstRunMetadata({
        connection: { browser_access: "local" },
      }),
    })

    expect(
      screen.getByText(/api server setup should open locally/i)
    ).toBeInTheDocument()
  })

  it("warns that non-local API setup may require server-machine access when remote setup is disabled", () => {
    renderChoice({
      metadata: firstRunMetadata({
        remote_setup_enabled: false,
        connection: { browser_access: "remote" },
      }),
    })

    expect(
      screen.getByText(/opened on the server machine/i)
    ).toBeInTheDocument()
    expect(
      screen.getByText(/enabled for remote setup by the operator/i)
    ).toBeInTheDocument()
  })

  it("explains remote setup allowlist restrictions when remote setup is enabled", () => {
    renderChoice({
      metadata: firstRunMetadata({
        remote_setup_enabled: true,
        connection: { browser_access: "remote" },
      }),
    })

    expect(
      screen.getByText(/remote api setup access is enabled/i)
    ).toBeInTheDocument()
    expect(
      screen.getByText(/restricted by the server setup allowlist/i)
    ).toBeInTheDocument()
  })
})
