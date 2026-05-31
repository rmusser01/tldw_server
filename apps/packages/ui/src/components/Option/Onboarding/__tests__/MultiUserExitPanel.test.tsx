// @vitest-environment jsdom
import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { MultiUserExitPanel } from "../steps/MultiUserExitPanel"

describe("MultiUserExitPanel", () => {
  it("maps repo-relative docs paths to web links", () => {
    render(
      <MultiUserExitPanel
        metadata={{
          auth_mode: "multi_user",
          bundled_single_user_auth_available: false,
          manual_auth_required: true,
          setup_required: true,
          setup_completed: false,
          remote_setup_enabled: false,
          connection: { browser_access: "local" },
          setup_paths: [],
          multi_user_exit: {
            guide_path: "Docs/Getting_Started/Profile_Docker_Multi_User_Postgres.md",
            checklist_path: "Docs/User_Guides/Server/Multi-User_Deployment_Guide.md"
          }
        }}
        onBack={vi.fn()}
      />
    )

    expect(screen.getByRole("link", { name: /open guide/i })).toHaveAttribute(
      "href",
      "https://github.com/rmusser01/tldw_server/blob/main/Docs/Getting_Started/Profile_Docker_Multi_User_Postgres.md"
    )
    expect(screen.getByRole("link", { name: /open deployment checklist/i }))
      .toHaveAttribute(
        "href",
        "https://github.com/rmusser01/tldw_server/blob/main/Docs/User_Guides/Server/Multi-User_Deployment_Guide.md"
      )
  })
})
