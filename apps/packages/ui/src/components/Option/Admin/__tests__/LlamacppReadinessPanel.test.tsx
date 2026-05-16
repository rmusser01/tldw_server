import React from "react"
import { describe, expect, it } from "vitest"
import { render, screen } from "@testing-library/react"
import { LlamacppReadinessPanel } from "../LlamacppReadinessPanel"

describe("LlamacppReadinessPanel", () => {
  it("does not claim active handler is configured when the backend says it is not", () => {
    render(
      <LlamacppReadinessPanel
        config={{
          saved_config: {
            enabled: false,
            models_dir: "/srv/models",
            allowed_paths: [],
            registered_model_paths: []
          },
          active_config: {
            handler_configured: false
          },
          restart_required: true,
          restart_reasons: ["handler_not_configured"],
          env_overrides: {
            models_dir: true
          },
          warnings: []
        }}
      />
    )

    expect(screen.getByText("Saved disabled")).toBeTruthy()
    expect(screen.getByText("/srv/models")).toBeTruthy()
    expect(screen.getByText("Active handler not configured")).toBeTruthy()
    expect(screen.queryByText("Active handler configured")).toBeNull()
    expect(screen.getByText("API server restart required")).toBeTruthy()
    expect(screen.getByText("models_dir override")).toBeTruthy()
  })
})
