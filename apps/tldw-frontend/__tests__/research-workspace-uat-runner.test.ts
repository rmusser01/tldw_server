import { EventEmitter } from "node:events"
import os from "node:os"
import path from "node:path"
import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  spawn: vi.fn()
}))

vi.mock("node:child_process", () => ({
  default: { spawn: mocks.spawn },
  spawn: mocks.spawn
}))

import {
  buildRunnerConfig,
  classifyPlaywrightRun,
  createEvidence,
  runPlaywright,
} from "../scripts/research-workspace-uat-runner.mjs"

describe("research-workspace-uat-runner", () => {
  beforeEach(() => {
    mocks.spawn.mockReset()
  })

  it("defaults to localhost-safe startup and the focused Research Workspace specs", () => {
    const config = buildRunnerConfig({ argv: [], env: {} })

    expect(config.webUrl).toBe("http://localhost:8080")
    expect(config.webCommand).toBe("bun run dev -- -H 127.0.0.1 -p 8080")
    expect(config.apiUrl).toBe("http://127.0.0.1:8000")
    expect(config.project).toBe("chromium")
    expect(config.workers).toBe("1")
    expect(config.specs).toEqual([
      "e2e/workflows/research-workspace.spec.ts",
      "e2e/workflows/research-workspace.real-backend.spec.ts",
    ])
    expect(config.reportPath).toBe(
      "test-results/research-workspace-final-uat-report.json"
    )
    expect(config.evidencePath).toBe(
      "test-results/research-workspace-final-uat-evidence.json"
    )
  })

  it("classifies macOS Chromium Mach-port launch failures as environment blocked", () => {
    const classification = classifyPlaywrightRun({
      exitCode: 1,
      stdout: "",
      stderr:
        "browserType.launch: Target page, context or browser has been closed\nbootstrap_check_in com.microsoft.edgemac.MachPortRendezvousServer: Permission denied (1100)",
      report: null,
    })

    expect(classification.status).toBe("environment_blocked")
    expect(classification.failureScope).toBe("environment")
    expect(classification.reasons).toContain("macos_mach_port_denied")
  })

  it("classifies executed Playwright assertion failures as product failures", () => {
    const classification = classifyPlaywrightRun({
      exitCode: 1,
      stdout: "1 failed",
      stderr: "",
      report: {
        stats: {
          expected: 12,
          skipped: 0,
          unexpected: 1,
          flaky: 0,
        },
      },
    })

    expect(classification.status).toBe("product_failed")
    expect(classification.failureScope).toBe("product")
    expect(classification.reasons).toContain("unexpected_failures")
  })

  it("does not classify skipped or empty Playwright reports as product passes", () => {
    const skipped = classifyPlaywrightRun({
      exitCode: 0,
      stdout: "",
      stderr: "",
      report: {
        stats: {
          expected: 4,
          skipped: 1,
          unexpected: 0,
          flaky: 0,
        },
      },
    })
    const empty = classifyPlaywrightRun({
      exitCode: 0,
      stdout: "",
      stderr: "",
      report: {
        stats: {
          expected: 0,
          skipped: 0,
          unexpected: 0,
          flaky: 0,
        },
      },
    })

    expect(skipped.status).toBe("product_failed")
    expect(skipped.reasons).toContain("skipped_tests_present")
    expect(empty.status).toBe("environment_blocked")
    expect(empty.reasons).toContain("no_tests_executed")
  })

  it("classifies known environment-only skips as environment blocked", () => {
    const classification = classifyPlaywrightRun({
      exitCode: 0,
      stdout: "",
      stderr: "",
      report: {
        stats: {
          expected: 19,
          skipped: 2,
          unexpected: 0,
          flaky: 0,
        },
        suites: [
          {
            suites: [
              {
                specs: [
                  {
                    title:
                      "grounds live chat requests on the selected source",
                    tests: [
                      {
                        status: "skipped",
                        expectedStatus: "skipped",
                        annotations: [
                          {
                            type: "skip",
                            description:
                              "Skipping live generation workflow: server did not advertise a runnable chat model"
                          }
                        ],
                        results: []
                      }
                    ]
                  },
                  {
                    title:
                      "shows workspace-linked sandbox run in diagnostics when sandbox run API is available",
                    tests: [
                      {
                        status: "skipped",
                        expectedStatus: "skipped",
                        annotations: [
                          {
                            type: "skip",
                            description:
                              "POST /api/v1/sandbox/runs returned HTTP 404: {\"detail\":\"Not Found\"}"
                          }
                        ],
                        results: []
                      }
                    ]
                  }
                ]
              }
            ]
          }
        ]
      },
    })

    expect(classification.status).toBe("environment_blocked")
    expect(classification.failureScope).toBe("environment")
    expect(classification.reasons).toContain("environment_skips_present")
    expect(classification.reasons).toContain("no_runnable_chat_model")
    expect(classification.reasons).toContain("sandbox_run_api_unavailable")
  })

  it("writes evidence that separates environment failures from product results", () => {
    const config = buildRunnerConfig({ argv: [], env: {} })
    const classification = classifyPlaywrightRun({
      exitCode: 1,
      stdout: "",
      stderr:
        "bootstrap_check_in com.microsoft.edgemac.MachPortRendezvousServer: Permission denied (1100)",
      report: null,
    })

    const evidence = createEvidence({
      classification,
      config,
      exitCode: 1,
      finishedAt: "2026-06-26T07:05:00.000Z",
      startedAt: "2026-06-26T07:00:00.000Z",
    })

    expect(evidence.status).toBe("environment_blocked")
    expect(evidence.productPassed).toBe(false)
    expect(evidence.failureScope).toBe("environment")
    expect(evidence.requiredSetup).toContain("Local network permission")
    expect(evidence.requiredSetup).toContain(
      "Sandbox-capable backend profile with [API-Routes] stable_only = true and enable = sandbox for strict workspace sandbox diagnostics"
    )
    expect(evidence.fallback).toContain("explicit Chrome debugging endpoint")
    expect(evidence.fallback).toContain("chromium.connectOverCDP")
    expect(evidence.fallback).not.toMatch(/computer[ -]?control|CUA/i)
    expect(evidence.evidencePath).toContain(
      "test-results/research-workspace-final-uat-evidence.json"
    )
  })

  it("resolves with captured stderr when Playwright cannot be spawned", async () => {
    const child = new EventEmitter() as EventEmitter & {
      stdout: EventEmitter
      stderr: EventEmitter
    }
    child.stdout = new EventEmitter()
    child.stderr = new EventEmitter()
    mocks.spawn.mockReturnValue(child)

    const tempDir = path.join(os.tmpdir(), `rw-uat-${Date.now()}`)
    const config = buildRunnerConfig({
      argv: [
        "--report",
        path.join(tempDir, "report.json"),
        "--evidence",
        path.join(tempDir, "evidence.json")
      ],
      env: {}
    })

    const resultPromise = runPlaywright(config)
    child.emit("error", new Error("spawn bunx ENOENT"))

    await expect(resultPromise).resolves.toMatchObject({
      exitCode: 1,
      stderr: expect.stringContaining("spawn bunx ENOENT")
    })
  })
})
