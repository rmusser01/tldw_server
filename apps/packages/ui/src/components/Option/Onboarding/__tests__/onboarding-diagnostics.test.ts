import { describe, expect, it } from "vitest"
import type { TFunction } from "i18next"
import {
  buildReadinessDiagnostic,
  buildSetupDiagnostic,
} from "../onboarding-diagnostics"

const t = ((_: string, fallback: string) => fallback) as TFunction

describe("buildSetupDiagnostic", () => {
  it("maps invalid API keys to edit and retry actions", () => {
    const diagnostic = buildSetupDiagnostic("auth_invalid", t)

    expect(diagnostic).toMatchObject({
      title: "API key was not accepted",
      severity: "blocking",
      primaryAction: { id: "edit_api_key", label: "Edit API key" },
    })
    expect(diagnostic?.cause).toMatch(/authentication/i)
    expect(diagnostic?.secondaryActions.map((item) => item.id)).toEqual([
      "retry",
      "edit_server_url",
    ])
  })

  it("maps network categories to safe setup recovery actions", () => {
    expect(buildSetupDiagnostic("refused", t)?.primaryAction.id).toBe(
      "edit_server_url"
    )
    expect(buildSetupDiagnostic("dns_failed", t)?.primaryAction.id).toBe(
      "edit_server_url"
    )
    expect(buildSetupDiagnostic("timeout", t)?.primaryAction.id).toBe("retry")
    expect(buildSetupDiagnostic("cors_blocked", t)?.primaryAction.id).toBe(
      "open_setup"
    )
    expect(buildSetupDiagnostic("ssl_error", t)?.primaryAction.id).toBe(
      "edit_server_url"
    )
    expect(buildSetupDiagnostic("server_error", t)?.primaryAction.id).toBe(
      "retry"
    )
  })

  it("does not include raw exception details in primary diagnostic copy", () => {
    const unsafePattern =
      /traceback|stack trace|authorization|x-api-key|\/Users\/|sk-[A-Za-z0-9_-]+/i

    for (const kind of [
      "auth_invalid",
      "refused",
      "dns_failed",
      "timeout",
      "cors_blocked",
      "ssl_error",
      "server_error",
    ] as const) {
      const diagnostic = buildSetupDiagnostic(kind, t)
      const copy = [
        diagnostic?.title,
        diagnostic?.cause,
        diagnostic?.whyItMatters,
        diagnostic?.primaryAction.label,
        ...(diagnostic?.secondaryActions.map((item) => item.label) ?? []),
      ].join(" ")

      expect(copy).not.toMatch(unsafePattern)
    }
  })
})

describe("buildReadinessDiagnostic", () => {
  it("keeps restart and config-write issues actionable before first chat", () => {
    expect(buildReadinessDiagnostic("restart_needed", t)).toMatchObject({
      severity: "recoverable",
      primaryAction: { id: "open_setup" },
    })
    expect(buildReadinessDiagnostic("config_write_failed", t)).toMatchObject({
      severity: "blocking",
      primaryAction: { id: "open_setup" },
    })
  })

  it("treats optional RAG storage and audio lanes as deferrable warnings by default", () => {
    expect(buildReadinessDiagnostic("rag_storage_unavailable", t)).toMatchObject({
      severity: "warning",
      blockingFirstChat: false,
      primaryAction: { id: "continue_without_optional" },
    })
    expect(buildReadinessDiagnostic("audio_readiness_failed", t)).toMatchObject({
      severity: "warning",
      blockingFirstChat: false,
      primaryAction: { id: "continue_without_optional" },
    })
  })

  it("marks optional lanes blocking only after the user selects them", () => {
    expect(
      buildReadinessDiagnostic("rag_storage_unavailable", t, {
        selectedOptionalLane: true,
      })
    ).toMatchObject({
      severity: "recoverable",
      blockingFirstChat: true,
      primaryAction: { id: "open_rag_settings" },
    })
    expect(
      buildReadinessDiagnostic("audio_readiness_failed", t, {
        selectedOptionalLane: true,
      })
    ).toMatchObject({
      severity: "recoverable",
      blockingFirstChat: true,
      primaryAction: { id: "open_audio_setup" },
    })
  })

  it("maps install/download limitations to operator-safe recovery copy", () => {
    const unsafePattern =
      /traceback|stack trace|authorization|x-api-key|\/Users\/|sk-[A-Za-z0-9_-]+/i

    for (const kind of [
      "network_unavailable",
      "downloads_disabled",
      "package_installs_disabled",
    ] as const) {
      const diagnostic = buildReadinessDiagnostic(kind, t)
      const copy = [
        diagnostic?.title,
        diagnostic?.cause,
        diagnostic?.whyItMatters,
        diagnostic?.primaryAction.label,
        ...(diagnostic?.secondaryActions.map((item) => item.label) ?? []),
      ].join(" ")

      expect(diagnostic?.severity).toBe("recoverable")
      expect(copy).not.toMatch(unsafePattern)
    }
  })

  it("returns a safe fallback for unknown readiness issue kinds", () => {
    const diagnostic = buildReadinessDiagnostic(
      "unexpected_issue" as never,
      t
    )

    expect(diagnostic).toMatchObject({
      title: "System readiness issue",
      severity: "warning",
      blockingFirstChat: false,
      primaryAction: { id: "open_setup", label: "Open setup recovery" },
      secondaryActions: [],
    })
  })
})
