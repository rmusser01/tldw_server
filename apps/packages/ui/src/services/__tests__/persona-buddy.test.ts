import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  fetchWithAuth: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    fetchWithAuth: (...args: unknown[]) => mocks.fetchWithAuth(...args)
  }
}))

import {
  getBuddyPreferences,
  getPersonaBuddyPreferences,
  updateBuddyPreferences,
  updatePersonaBuddyPreferences
} from "../persona-buddy"

describe("persona buddy preference service", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("distinguishes a failed preference read from a missing stored row", async () => {
    mocks.fetchWithAuth.mockRejectedValueOnce(new Error("unauthorized"))

    await expect(getBuddyPreferences()).rejects.toThrow("unauthorized")

    mocks.fetchWithAuth.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({
        ambient_mode: "expressive",
        version: null,
        stored: false
      })
    })

    await expect(getBuddyPreferences()).resolves.toEqual({
      ambient_mode: "expressive",
      version: null,
      stored: false
    })
  })

  it("sends global preference versions without coercion", async () => {
    mocks.fetchWithAuth.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({
        ambient_mode: "roaming",
        version: 3,
        stored: true
      })
    })

    await expect(
      updateBuddyPreferences({ ambient_mode: "roaming", expected_version: 2 })
    ).resolves.toEqual({
      ambient_mode: "roaming",
      version: 3,
      stored: true
    })

    expect(mocks.fetchWithAuth).toHaveBeenCalledWith(
      "/api/v1/persona/buddy/preferences",
      expect.objectContaining({
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ambient_mode: "roaming", expected_version: 2 })
      })
    )
  })

  it("uses the version-checked per-Persona preference endpoint", async () => {
    mocks.fetchWithAuth.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({
        ambient_mode: "off",
        version: 8,
        stored: true
      })
    })

    await updatePersonaBuddyPreferences("persona/one", {
      ambient_mode: "off",
      expected_version: 7
    })

    expect(mocks.fetchWithAuth).toHaveBeenCalledWith(
      "/api/v1/persona/profiles/persona%2Fone/buddy/preferences",
      expect.objectContaining({
        method: "PATCH",
        body: JSON.stringify({ ambient_mode: "off", expected_version: 7 })
      })
    )
  })

  it("reads and clears a nullable per-Persona override", async () => {
    mocks.fetchWithAuth
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({ ambient_mode: null, version: 7, stored: false })
      })
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({ ambient_mode: null, version: 8, stored: false })
      })

    await expect(
      getPersonaBuddyPreferences("persona/one")
    ).resolves.toEqual({ ambient_mode: null, version: 7, stored: false })
    await updatePersonaBuddyPreferences("persona/one", {
      ambient_mode: null,
      expected_version: 7
    })

    expect(mocks.fetchWithAuth).toHaveBeenNthCalledWith(
      1,
      "/api/v1/persona/profiles/persona%2Fone/buddy/preferences",
      expect.objectContaining({ method: "GET" })
    )
    expect(mocks.fetchWithAuth).toHaveBeenNthCalledWith(
      2,
      "/api/v1/persona/profiles/persona%2Fone/buddy/preferences",
      expect.objectContaining({
        method: "PATCH",
        body: JSON.stringify({ ambient_mode: null, expected_version: 7 })
      })
    )
  })
})
