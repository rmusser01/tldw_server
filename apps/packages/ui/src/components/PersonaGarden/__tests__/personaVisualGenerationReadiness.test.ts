import { describe, expect, it } from "vitest"

import {
  classifyPersonaVisualGenerationReadiness,
  type PersonaVisualGenerationReadinessView
} from "../personaVisualGenerationReadiness"
import type { PersonaVisualGenerationReadinessResponse } from "@/types/persona-visuals"

const readiness = (
  overrides: Partial<PersonaVisualGenerationReadinessResponse> = {}
): PersonaVisualGenerationReadinessResponse => ({
  available: true,
  worker_enabled: true,
  queue: "generation",
  image_backend_available: true,
  default_backend: "openrouter",
  requested_backend: null,
  requested_backend_available: null,
  enabled_backends: ["openrouter", "novita"],
  reasons: [],
  ...overrides
})

describe("classifyPersonaVisualGenerationReadiness", () => {
  it("blocks queueing while readiness is loading", () => {
    const view = classifyPersonaVisualGenerationReadiness(null, "", {
      isLoading: true
    })

    expect(view).toMatchObject<Partial<PersonaVisualGenerationReadinessView>>({
      status: "loading",
      canQueue: false,
      blocking: true
    })
  })

  it("distinguishes Jobs worker unavailability from provider setup", () => {
    const view = classifyPersonaVisualGenerationReadiness(
      readiness({
        available: false,
        worker_enabled: false,
        reasons: ["jobs_worker_disabled"]
      }),
      ""
    )

    expect(view.status).toBe("jobs_unavailable")
    expect(view.canQueue).toBe(false)
    expect(view.blocking).toBe(true)
  })

  it("reports missing image providers separately from Jobs readiness", () => {
    const view = classifyPersonaVisualGenerationReadiness(
      readiness({
        available: false,
        image_backend_available: false,
        default_backend: null,
        enabled_backends: [],
        reasons: ["image_backend_unavailable"]
      }),
      ""
    )

    expect(view.status).toBe("image_provider_unavailable")
    expect(view.canQueue).toBe(false)
  })

  it("blocks unavailable typed backends before enqueue", () => {
    const view = classifyPersonaVisualGenerationReadiness(
      readiness(),
      "missing-provider"
    )

    expect(view.status).toBe("backend_unavailable")
    expect(view.selectedBackend).toBe("missing-provider")
    expect(view.canQueue).toBe(false)
  })

  it("blocks resolved backends whose adapter cannot be started", () => {
    const view = classifyPersonaVisualGenerationReadiness(
      readiness({
        available: false,
        image_backend_available: false,
        reasons: ["image_adapter_unavailable"]
      }),
      ""
    )

    expect(view.status).toBe("image_adapter_unavailable")
    expect(view.canQueue).toBe(false)
  })

  it("reports readiness dependency failures separately from provider setup", () => {
    const view = classifyPersonaVisualGenerationReadiness(
      readiness({
        available: false,
        image_backend_available: false,
        default_backend: null,
        enabled_backends: [],
        reasons: ["dependency_check_failed"]
      }),
      ""
    )

    expect(view.status).toBe("dependency_check_failed")
    expect(view.canQueue).toBe(false)
  })

  it("allows queueing when the user selects an enabled backend even without a default", () => {
    const base = readiness({
      available: false,
      image_backend_available: false,
      default_backend: null,
      enabled_backends: ["novita"],
      reasons: ["default_backend_unavailable"]
    })

    expect(classifyPersonaVisualGenerationReadiness(base, "")).toMatchObject({
      status: "default_backend_unavailable",
      canQueue: false
    })
    expect(classifyPersonaVisualGenerationReadiness(base, "novita")).toMatchObject({
      status: "ready",
      canQueue: true,
      selectedBackend: "novita"
    })
  })
})
