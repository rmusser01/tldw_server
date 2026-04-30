// @vitest-environment jsdom

import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, afterEach, describe, expect, it, vi } from "vitest"

import { AudioInstallerPanel } from "../AudioInstallerPanel"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn()
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string },
      maybeOptions?: { defaultValue?: string }
    ) => {
      if (typeof fallbackOrOptions === "string") {
        return fallbackOrOptions
      }
      if (
        fallbackOrOptions &&
        typeof fallbackOrOptions === "object" &&
        typeof fallbackOrOptions.defaultValue === "string"
      ) {
        return fallbackOrOptions.defaultValue
      }
      return maybeOptions?.defaultValue || key
    }
  })
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) =>
    (mocks.bgRequest as (...args: unknown[]) => unknown)(...args)
}))

vi.mock("@/services/tldw/path-utils", () => ({
  toAllowedPath: (path: string) => path
}))

vi.mock("@/components/Option/Admin/admin-error-utils", () => ({
  deriveAdminGuardFromError: (error: Error) => {
    if (String(error.message).includes("403")) return "forbidden"
    if (String(error.message).includes("404")) return "notFound"
    return null
  },
  sanitizeAdminErrorMessage: (error: Error, fallback: string) =>
    error?.message || fallback
}))

const createResponse = (data: unknown, status = 200) => ({
  ok: status >= 200 && status < 300,
  status,
  data,
  error: status >= 400 ? `Request failed: ${status}` : undefined,
  json: async () => data,
  text: async () => JSON.stringify(data ?? {})
})

const recommendationPayload = {
  machine_profile: {
    platform: "darwin",
    arch: "arm64",
    apple_silicon: true,
    cuda_available: false,
    free_disk_gb: 128,
    network_available_for_downloads: true
  },
  recommendations: [
    {
      bundle_id: "apple_silicon_local",
      resource_profile: "balanced",
      selection_key: "v2:apple_silicon_local:balanced",
      label: "Apple Silicon Local",
      bundle: {
        bundle_id: "apple_silicon_local",
        label: "Apple Silicon Local",
        description: "Local bundle for Apple Silicon machines.",
        default_resource_profile: "balanced",
        resource_profiles: {
          light: {
            profile_id: "light",
            label: "Light",
            description: "Lowest-footprint local speech profile.",
            estimated_disk_gb: 1,
            resource_class: "low"
          },
          balanced: {
            profile_id: "balanced",
            label: "Balanced",
            description: "Recommended Apple Silicon profile using MLX Parakeet for STT.",
            estimated_disk_gb: 3,
            resource_class: "medium"
          },
          performance: {
            profile_id: "performance",
            label: "Performance",
            description: "Higher-throughput Apple Silicon speech profile.",
            estimated_disk_gb: 4.5,
            resource_class: "high"
          }
        }
      },
      profile: {
        profile_id: "balanced",
        label: "Balanced",
        description: "Recommended Apple Silicon profile using MLX Parakeet for STT.",
        stt_plan: [{ engine: "nemo_parakeet_mlx", models: [] }],
        tts_plan: [{ engine: "kokoro", variants: [] }],
        estimated_disk_gb: 3,
        resource_class: "medium"
      }
    }
  ],
  excluded: [],
  catalog: []
}

const cpuKittenRecommendationPayload = {
  machine_profile: {
    platform: "linux",
    arch: "x86_64",
    apple_silicon: false,
    cuda_available: false,
    free_disk_gb: 64,
    network_available_for_downloads: true
  },
  recommendations: [
    {
      bundle_id: "cpu_local",
      resource_profile: "balanced",
      selection_key: "v2:cpu_local:balanced",
      label: "CPU Local",
      bundle: {
        bundle_id: "cpu_local",
        label: "CPU Local",
        description: "Local CPU-first speech bundle.",
        default_resource_profile: "balanced",
        resource_profiles: {
          light: {
            profile_id: "light",
            label: "Light",
            description: "Lowest-footprint local speech profile.",
            estimated_disk_gb: 1,
            resource_class: "low",
            default_tts_choice: "kokoro",
            tts_choices: [
              { choice_id: "kokoro", label: "Kokoro" },
              { choice_id: "kitten_tts", label: "KittenTTS" }
            ]
          },
          balanced: {
            profile_id: "balanced",
            label: "Balanced",
            description: "Balanced CPU local speech profile.",
            estimated_disk_gb: 2,
            resource_class: "medium",
            default_tts_choice: "kokoro",
            tts_choices: [
              { choice_id: "kokoro", label: "Kokoro" },
              { choice_id: "kitten_tts", label: "KittenTTS" }
            ]
          }
        }
      },
      profile: {
        profile_id: "balanced",
        label: "Balanced",
        description: "Balanced CPU local speech profile.",
        stt_plan: [{ engine: "faster_whisper", models: ["small"] }],
        tts_plan: [{ engine: "kokoro", variants: [] }],
        estimated_disk_gb: 2,
        resource_class: "medium",
        default_tts_choice: "kokoro",
        tts_choices: [
          { choice_id: "kokoro", label: "Kokoro" },
          { choice_id: "kitten_tts", label: "KittenTTS" }
        ]
      }
    }
  ],
  excluded: [],
  catalog: []
}

describe("AudioInstallerPanel", () => {
  beforeEach(() => {
    mocks.bgRequest.mockReset()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it("loads and shows the recommended audio bundle", async () => {
    mocks.bgRequest.mockImplementation((init: { path: string }) => {
      if (String(init.path).includes("/audio/recommendations")) {
        return Promise.resolve(createResponse(recommendationPayload))
      }
      if (String(init.path).includes("/install-status")) {
        return Promise.resolve(createResponse({ status: "idle", steps: [], errors: [] }))
      }
      throw new Error(`Unexpected request: ${init.path}`)
    })

    render(<AudioInstallerPanel />)

    expect(await screen.findByText("Recommended audio bundle")).toBeInTheDocument()
    expect(screen.getByText("Apple Silicon Local")).toBeInTheDocument()
    expect(screen.getByText("Recommended profile")).toBeInTheDocument()
    expect(screen.getByRole("radio", { name: "Balanced" })).toBeChecked()
    expect(screen.getByRole("button", { name: "Provision bundle" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Run verification" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Safe rerun" })).toBeInTheDocument()
  })

  it("deduplicates bundle options when recommendations contain multiple profiles for one bundle", async () => {
    mocks.bgRequest.mockImplementation((init: { path: string }) => {
      if (String(init.path).includes("/audio/recommendations")) {
        return Promise.resolve(
          createResponse({
            ...recommendationPayload,
            recommendations: [
              recommendationPayload.recommendations[0],
              {
                ...recommendationPayload.recommendations[0],
                resource_profile: "performance",
                profile: {
                  profile_id: "performance",
                  label: "Performance",
                  description: "Higher-throughput Apple Silicon speech profile.",
                  stt_plan: [{ engine: "nemo_parakeet_mlx", models: [] }],
                  tts_plan: [{ engine: "kokoro", variants: [] }],
                  estimated_disk_gb: 4.5,
                  resource_class: "high"
                }
              }
            ]
          })
        )
      }
      if (String(init.path).includes("/install-status")) {
        return Promise.resolve(createResponse({ status: "idle", steps: [], errors: [] }))
      }
      throw new Error(`Unexpected request: ${init.path}`)
    })

    render(<AudioInstallerPanel />)

    expect(await screen.findByText("Recommended audio bundle")).toBeInTheDocument()
    fireEvent.mouseDown(screen.getByRole("combobox"))
    expect(await screen.findAllByRole("option", { name: "Apple Silicon Local" })).toHaveLength(1)
  })
  it("provisions the selected bundle and polls install status while active", async () => {
    let installStatusCallCount = 0

    mocks.bgRequest.mockImplementation((init: { path: string; method?: string; body?: string }) => {
      const target = String(init.path)
      if (target.includes("/audio/recommendations")) {
        return Promise.resolve(createResponse(recommendationPayload))
      }
      if (target.includes("/install-status")) {
        installStatusCallCount += 1
        return Promise.resolve(
          createResponse(
            installStatusCallCount < 2
              ? {
                  status: "running",
                  steps: [{ name: "download", label: "Download assets", status: "running" }],
                  errors: []
                }
              : {
                  status: "completed",
                  steps: [{ name: "download", label: "Download assets", status: "completed" }],
                  errors: []
                }
          )
        )
      }
      if (target.includes("/audio/provision")) {
        expect(init?.method).toBe("POST")
        expect(JSON.parse(String(init?.body))).toEqual({
          bundle_id: "apple_silicon_local",
          resource_profile: "balanced",
          safe_rerun: false
        })
        return Promise.resolve(
          createResponse({
            status: "running",
            bundle_id: "apple_silicon_local",
            resource_profile: "balanced",
            steps: [{ name: "download", label: "Download assets", status: "running" }],
            errors: []
          })
        )
      }
      throw new Error(`Unexpected request: ${init.path}`)
    })

    render(<AudioInstallerPanel />)

    const provisionButton = await screen.findByRole("button", { name: "Provision bundle" })
    vi.useFakeTimers()
    await act(async () => {
      fireEvent.click(provisionButton)
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/setup/admin/audio/provision",
        method: "POST",
        timeoutMs: 30 * 60 * 1000
      })
    )

    await act(async () => {
      await Promise.resolve()
      await Promise.resolve()
    })

    await act(async () => {
      vi.advanceTimersByTime(3100)
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(
      mocks.bgRequest.mock.calls.filter(([init]) =>
        String(init?.path).includes("/api/v1/setup/admin/install-status")
      ).length
    ).toBeGreaterThanOrEqual(2)

    expect(screen.getByText("Install status")).toBeInTheDocument()
    expect(screen.getByText("completed")).toBeInTheDocument()
  })

  it("runs verification and shows remediation details", async () => {
    mocks.bgRequest.mockImplementation((init: { path: string; method?: string; body?: string }) => {
      const target = String(init.path)
      if (target.includes("/audio/recommendations")) {
        return Promise.resolve(createResponse(recommendationPayload))
      }
      if (target.includes("/install-status")) {
        return Promise.resolve(createResponse({ status: "idle", steps: [], errors: [] }))
      }
      if (target.includes("/audio/verify")) {
        expect(init?.method).toBe("POST")
        expect(JSON.parse(String(init?.body))).toEqual({
          bundle_id: "apple_silicon_local",
          resource_profile: "balanced"
        })
        return Promise.resolve(
          createResponse({
            status: "partial",
            bundle_id: "apple_silicon_local",
            selected_resource_profile: "balanced",
            targets_checked: ["stt_default", "tts_default"],
            remediation_items: [
              {
                code: "espeak_ng",
                action: "safe_rerun",
                message: "Install eSpeak NG before relying on Kokoro."
              }
            ]
          })
        )
      }
      throw new Error(`Unexpected request: ${init.path}`)
    })

    render(<AudioInstallerPanel />)

    fireEvent.click(await screen.findByRole("button", { name: "Run verification" }))

    await waitFor(() => {
      expect(mocks.bgRequest).toHaveBeenCalledWith(
        expect.objectContaining({
          path: "/api/v1/setup/admin/audio/verify",
          method: "POST"
        })
      )
    })

    expect(await screen.findByText("Verification result")).toBeInTheDocument()
    expect(await screen.findByText("partial")).toBeInTheDocument()
    expect(await screen.findByText("Install eSpeak NG before relying on Kokoro.")).toBeInTheDocument()
  })

  it("renders curated tts choices and submits the selected choice", async () => {
    mocks.bgRequest.mockImplementation((init: { path: string; method?: string; body?: string }) => {
      const target = String(init.path)
      if (target.includes("/audio/recommendations")) {
        return Promise.resolve(createResponse(cpuKittenRecommendationPayload))
      }
      if (target.includes("/install-status")) {
        return Promise.resolve(createResponse({ status: "idle", steps: [], errors: [] }))
      }
      if (target.includes("/audio/provision")) {
        expect(init?.method).toBe("POST")
        expect(JSON.parse(String(init?.body))).toEqual({
          bundle_id: "cpu_local",
          resource_profile: "balanced",
          tts_choice: "kitten_tts",
          safe_rerun: false
        })
        return Promise.resolve(
          createResponse({
            status: "completed",
            bundle_id: "cpu_local",
            resource_profile: "balanced",
            tts_choice: "kitten_tts",
            steps: [],
            errors: []
          })
        )
      }
      throw new Error(`Unexpected request: ${init.path}`)
    })

    render(<AudioInstallerPanel />)

    expect(await screen.findByText("Balanced CPU local speech profile.")).toBeInTheDocument()
    expect(screen.getByText("Curated TTS")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("radio", { name: "KittenTTS" }))
    fireEvent.click(screen.getByRole("button", { name: "Provision bundle" }))

    await waitFor(() => {
      expect(mocks.bgRequest).toHaveBeenCalledWith(
        expect.objectContaining({
          path: "/api/v1/setup/admin/audio/provision",
          method: "POST"
        })
      )
    })
  })

  it("preserves a curated tts choice across retry-triggered recommendation reloads", async () => {
    let verifyAttempts = 0

    mocks.bgRequest.mockImplementation((init: { path: string; method?: string; body?: string }) => {
      const target = String(init.path)
      if (target.includes("/audio/recommendations")) {
        return Promise.resolve(createResponse(cpuKittenRecommendationPayload))
      }
      if (target.includes("/install-status")) {
        return Promise.resolve(createResponse({ status: "idle", steps: [], errors: [] }))
      }
      if (target.includes("/audio/verify")) {
        verifyAttempts += 1
        if (verifyAttempts === 1) {
          return Promise.resolve(createResponse({ detail: "boom" }, 500))
        }
        return Promise.resolve(
          createResponse({
            status: "ready",
            bundle_id: "cpu_local",
            selected_resource_profile: "balanced",
            tts_choice: "kitten_tts",
            targets_checked: ["stt_default", "tts_default"],
            remediation_items: []
          })
        )
      }
      throw new Error(`Unexpected request: ${init.path}`)
    })

    render(<AudioInstallerPanel />)

    expect(await screen.findByText("Curated TTS")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("radio", { name: "KittenTTS" }))
    fireEvent.click(screen.getByRole("button", { name: "Run verification" }))

    expect(await screen.findByText("Installer request failed")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Retry" }))

    await waitFor(() => {
      expect(screen.getByRole("radio", { name: "KittenTTS" })).toBeChecked()
    })
  })

  it("shows an admin-only message when access is forbidden", async () => {
    mocks.bgRequest.mockImplementation((init: { path: string }) => {
      if (String(init.path).includes("/audio/recommendations")) {
        return Promise.resolve(createResponse({ detail: "forbidden" }, 403))
      }
      throw new Error(`Unexpected request: ${init.path}`)
    })

    render(<AudioInstallerPanel />)

    expect(
      await screen.findByText("Audio model installation requires server admin access.")
    ).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Provision bundle" })).not.toBeInTheDocument()
  })

  it("shows an unavailable message when the admin installer endpoint is missing", async () => {
    mocks.bgRequest.mockImplementation((init: { path: string }) => {
      if (String(init.path).includes("/audio/recommendations")) {
        return Promise.resolve(createResponse({ detail: "missing" }, 404))
      }
      throw new Error(`Unexpected request: ${init.path}`)
    })

    render(<AudioInstallerPanel />)

    expect(
      await screen.findByText("This server does not expose the admin audio installer yet.")
    ).toBeInTheDocument()
  })
})
