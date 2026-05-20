import { describe, expect, it } from "vitest"
import { AUDIO_ROUTE_FINDINGS, AUDIO_ROUTE_JOBS } from "../audio-route-jobs"
import { getRouteMetadata } from "../route-metadata"

const routes = [
  "/audio",
  "/speech",
  "/stt",
  "/tts",
  "/audiobook-studio"
] as const

describe("audio route jobs", () => {
  it("covers every WP11A root audio route once", () => {
    expect(AUDIO_ROUTE_JOBS.map((job) => job.route).sort()).toEqual(
      Array.from(routes).sort()
    )
  })

  it("keeps route labels aligned with route metadata", () => {
    for (const job of AUDIO_ROUTE_JOBS) {
      expect(job.copy.label.fallback).toBe(getRouteMetadata(job.route)?.label)
    }
  })

  it("keeps route copy behind stable translation keys and fallbacks", () => {
    for (const job of AUDIO_ROUTE_JOBS) {
      const copyPrefix = `routes.audio.${job.concept}`

      expect(job.copy.label.key).toBe(`${copyPrefix}.label`)
      expect(job.copy.primaryJob.key).toBe(`${copyPrefix}.primaryJob`)
      expect(job.copy.primaryActionLabel.key).toBe(
        `${copyPrefix}.primaryActionLabel`
      )
      expect(job.copy.label.fallback).not.toHaveLength(0)
      expect(job.copy.primaryJob.fallback).not.toHaveLength(0)
      expect(job.copy.primaryActionLabel.fallback).not.toHaveLength(0)
      expect(job.canonicalComponent).not.toHaveLength(0)
    }
  })

  it("maps the audit findings into implementation coverage", () => {
    const covered = new Set(AUDIO_ROUTE_JOBS.flatMap((job) => job.findings))

    for (const finding of AUDIO_ROUTE_FINDINGS) {
      expect(covered.has(finding)).toBe(true)
    }
  })

  it("preserves canonical ownership for overlapping audio routes", () => {
    expect(AUDIO_ROUTE_JOBS).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          route: "/audio",
          concept: "audio_alias",
          routeOwner: "shared_alias",
          canonicalComponent: "RouteAliasNavigate:/speech",
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
