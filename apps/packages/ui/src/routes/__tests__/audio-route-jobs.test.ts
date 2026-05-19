import { describe, expect, it } from "vitest"
import { AUDIO_ROUTE_JOBS } from "../audio-route-jobs"

const routes = [
  "/audio",
  "/speech",
  "/stt",
  "/tts",
  "/audiobook-studio"
] as const

const findings = [
  "F2 support",
  "F9 support",
  "F15 support",
  "F18 support",
  "F19 support"
] as const

describe("audio route jobs", () => {
  it("covers every WP11A root audio route once", () => {
    expect(AUDIO_ROUTE_JOBS.map((job) => job.route).sort()).toEqual(
      Array.from(routes).sort()
    )
  })

  it("keeps route labels and primary jobs usable", () => {
    for (const job of AUDIO_ROUTE_JOBS) {
      expect(job.label).not.toHaveLength(0)
      expect(job.primaryJob).not.toHaveLength(0)
      expect(job.primaryActionLabel).not.toHaveLength(0)
      expect(job.canonicalComponent).not.toHaveLength(0)
    }
  })

  it("maps the audit findings into implementation coverage", () => {
    const covered = new Set(AUDIO_ROUTE_JOBS.flatMap((job) => job.findings))

    for (const finding of findings) {
      expect(covered.has(finding)).toBe(true)
    }
  })

  it("preserves canonical ownership for overlapping audio routes", () => {
    expect(AUDIO_ROUTE_JOBS).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          route: "/audio",
          concept: "audio_alias",
          routeOwner: "next_alias",
          canonicalComponent: "RouteRedirect:/speech",
          routeStatePolicy: "alias"
        }),
        expect.objectContaining({
          route: "/speech",
          concept: "speech_combined",
          routeOwner: "shared_route",
          canonicalComponent: "SpeechPlaygroundPage",
          routeStatePolicy: "ready_or_recoverable"
        }),
        expect.objectContaining({
          route: "/stt",
          concept: "stt",
          routeOwner: "shared_route",
          canonicalComponent: "SttPlaygroundPage",
          routeStatePolicy: "ready_or_recoverable"
        }),
        expect.objectContaining({
          route: "/tts",
          concept: "tts",
          routeOwner: "shared_route",
          canonicalComponent: "SpeechPlaygroundPage:listen",
          routeStatePolicy: "ready_or_recoverable"
        }),
        expect.objectContaining({
          route: "/audiobook-studio",
          concept: "audiobook",
          routeOwner: "shared_route",
          canonicalComponent: "AudiobookStudioPage",
          routeStatePolicy: "beta_ready_or_recoverable"
        })
      ])
    )
  })
})
