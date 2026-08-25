import { describe, expect, it } from "vitest"
import {
  descendantUsage,
  evaluateRuntime,
  parseProcessTable,
} from "../dev-runtime-uat-lib.mjs"
import {
  buildRuntimeCommands,
  idleWaitDurations,
  parseArgs,
  resolveRuntimeEnvironment,
  runDevRuntimeUat,
  waitForStrictHttpOk,
} from "../dev-runtime-uat.mjs"

const gibibytes = (value: number) => value * 2 ** 30

describe("development runtime UAT evaluation", () => {
  it("sums only the complete descendant process tree", () => {
    const rows = parseProcessTable([
      "10 1 1024 1.0 bun run dev:webpack",
      "11 10 2048 2.0 next-server",
      "12 11 4096 3.0 next worker",
      "99 1 8192 4.0 unrelated process",
    ].join("\n"))

    expect(descendantUsage(rows, 10)).toEqual({
      rssBytes: 7 * 1024 * 1024,
      cpuPercent: 6,
      pids: [10, 11, 12],
    })
  })

  it("fails closed when the root process is absent from the process table", () => {
    const rows = parseProcessTable("10 1 1024 1.0 bun run dev")

    expect(() => descendantUsage(rows, 404)).toThrow("Root process 404 not found")
  })

  it("rejects malformed process rows instead of under-reporting usage", () => {
    expect(() => parseProcessTable("not a process row")).toThrow(
      "Malformed process table row 1",
    )
  })

  it("fails a responsive runtime whose warm-idle growth exceeds two GiB", () => {
    const result = evaluateRuntime([
      { phase: "post-traversal", rssBytes: gibibytes(10), responsive: true },
      { phase: "post-idle", rssBytes: gibibytes(13), responsive: true },
      { phase: "second-pass", rssBytes: gibibytes(13), responsive: true },
    ])

    expect(result).toEqual({ qualified: false, reasons: ["idle_rss_growth"] })
  })

  it("accepts exactly two GiB of idle growth when every required phase is responsive", () => {
    const result = evaluateRuntime([
      { phase: "post-traversal", rssBytes: gibibytes(10), responsive: true },
      { phase: "post-idle", rssBytes: gibibytes(12), responsive: true },
      { phase: "second-pass", rssBytes: gibibytes(12), responsive: true },
    ])

    expect(result).toEqual({ qualified: true, reasons: [] })
  })

  it("reports missing phases, unresponsiveness, and the sixteen-GiB boundary", () => {
    const result = evaluateRuntime([
      { phase: "post-traversal", rssBytes: gibibytes(16), responsive: false },
      { phase: "post-idle", rssBytes: gibibytes(16), responsive: true },
    ])

    expect(result).toEqual({
      qualified: false,
      reasons: ["unresponsive", "second_pass_missing", "rss_limit"],
    })
  })

  it("does not qualify a runtime without both warm-idle comparison samples", () => {
    const result = evaluateRuntime([
      { phase: "second-pass", rssBytes: gibibytes(8), responsive: true },
    ])

    expect(result).toEqual({
      qualified: false,
      reasons: ["post_traversal_missing", "post_idle_missing"],
    })
  })
})

describe("development runtime UAT command contract", () => {
  it("bounds every readiness request by the remaining advertised deadline", async () => {
    let nowMs = 0
    const probeTimeouts: number[] = []

    await expect(waitForStrictHttpOk("http://127.0.0.1:18180/api/v1/health", {
      timeoutMs: 450,
      intervalMs: 50,
      now: () => nowMs,
      probe: async (_url: string, { timeoutMs }: { timeoutMs: number }) => {
        probeTimeouts.push(timeoutMs)
        nowMs += timeoutMs
        throw new Error("probe deadline exceeded")
      },
      sleep: async (durationMs: number) => {
        nowMs += durationMs
      },
    })).rejects.toThrow("Timed out waiting for http://127.0.0.1:18180/api/v1/health")

    expect(probeTimeouts).toEqual([450])
  })

  it("parses one explicit bounded probe", () => {
    expect(parseArgs([
      "--bundler=turbopack",
      "--port=18182",
      "--warm-idle-ms=1200000",
      "--output=test-results/dev-runtime/turbopack.json",
    ])).toEqual({
      bundler: "turbopack",
      port: 18182,
      warmIdleMs: 1_200_000,
      output: "test-results/dev-runtime/turbopack.json",
      idleCheckIntervalMs: 30_000,
    })
  })

  it("rejects an unknown bundler before spawning a process", () => {
    expect(() => parseArgs([
      "--bundler=experimental",
      "--port=18182",
      "--warm-idle-ms=60000",
      "--output=report.json",
    ])).toThrow("--bundler must be webpack or turbopack")
  })

  it("rejects missing output and unbounded numeric options", () => {
    expect(() => parseArgs([
      "--bundler=webpack",
      "--port=18181",
      "--warm-idle-ms=60000",
    ])).toThrow("--output is required")
    expect(() => parseArgs([
      "--bundler=webpack",
      "--port=70000",
      "--warm-idle-ms=60000",
      "--output=report.json",
    ])).toThrow("--port must be an integer between 1 and 65535")
    expect(() => parseArgs([
      "--bundler=webpack",
      "--port=18181",
      "--warm-idle-ms=0",
      "--output=report.json",
    ])).toThrow("--warm-idle-ms must be a positive integer")
  })

  it("requires a real-backend environment with offline fallback disabled", () => {
    expect(resolveRuntimeEnvironment({
      NODE_ENV: "test",
      PATH: "/usr/bin",
      TLDW_E2E_ALLOW_OFFLINE: "0",
      TLDW_E2E_SERVER_URL: "http://127.0.0.1:18180/",
      TLDW_E2E_API_KEY: "synthetic-key",
    })).toEqual({
      apiKey: "synthetic-key",
      backendUrl: "http://127.0.0.1:18180",
    })

    expect(() => resolveRuntimeEnvironment({
      NODE_ENV: "test",
      TLDW_E2E_ALLOW_OFFLINE: "1",
      TLDW_E2E_SERVER_URL: "http://127.0.0.1:18180",
      TLDW_E2E_API_KEY: "synthetic-key",
    })).toThrow("TLDW_E2E_ALLOW_OFFLINE=0 is required")
  })

  it("rejects absent or non-HTTP backend coordinates", () => {
    expect(() => resolveRuntimeEnvironment({
      NODE_ENV: "test",
      TLDW_E2E_ALLOW_OFFLINE: "0",
      TLDW_E2E_API_KEY: "synthetic-key",
    })).toThrow("TLDW_E2E_SERVER_URL is required")
    expect(() => resolveRuntimeEnvironment({
      NODE_ENV: "test",
      TLDW_E2E_ALLOW_OFFLINE: "0",
      TLDW_E2E_SERVER_URL: "file:///tmp/backend",
      TLDW_E2E_API_KEY: "synthetic-key",
    })).toThrow("TLDW_E2E_SERVER_URL must use http or https")
    expect(() => resolveRuntimeEnvironment({
      NODE_ENV: "test",
      TLDW_E2E_ALLOW_OFFLINE: "0",
      TLDW_E2E_SERVER_URL: "http://127.0.0.1:18180",
    })).toThrow("TLDW_E2E_API_KEY is required")
  })

  it("constructs explicit frontend and no-autostart Playwright commands", () => {
    const commands = buildRuntimeCommands({
      bundler: "webpack",
      port: 18181,
      backendUrl: "http://127.0.0.1:18180",
      apiKey: "synthetic-key",
      baseEnv: {
        NODE_ENV: "test",
        PATH: "/usr/bin",
        OPENAI_API_KEY: "must-not-leak",
      },
      frontendRoot: "/repo/apps/tldw-frontend",
    })

    expect(commands.frontend).toMatchObject({
      command: "bun",
      args: ["run", "dev:webpack", "--", "-p", "18181"],
      cwd: "/repo/apps/tldw-frontend",
    })
    expect(commands.fullTraversal.args).toEqual([
      "run",
      "e2e:smoke:all-pages:gate",
    ])
    expect(commands.secondPass.args).toEqual([
      "x",
      "playwright",
      "test",
      "e2e/smoke/all-pages.spec.ts",
      "--reporter=line",
      "--grep",
      "Smoke Tests - Key Navigation Targets",
      "--workers=1",
    ])
    expect(commands.fullTraversal.env).toMatchObject({
      NEXT_PUBLIC_API_URL: "http://127.0.0.1:18180",
      NEXT_PUBLIC_X_API_KEY: "synthetic-key",
      TLDW_E2E_ALLOW_OFFLINE: "0",
      TLDW_E2E_API_KEY: "synthetic-key",
      TLDW_E2E_SERVER_URL: "http://127.0.0.1:18180",
      TLDW_SERVER_URL: "http://127.0.0.1:18180",
      TLDW_WEB_AUTOSTART: "false",
      TLDW_WEB_URL: "http://localhost:18181",
    })
    expect(commands.frontend.env).not.toHaveProperty("OPENAI_API_KEY")
    expect(commands.fullTraversal.env).not.toHaveProperty("OPENAI_API_KEY")
  })

  it("splits warm idle into health-check intervals without exceeding the duration", () => {
    expect(idleWaitDurations(65_000, 30_000)).toEqual([30_000, 30_000, 5_000])
  })
})

describe("development runtime UAT lifecycle", () => {
  const options = {
    bundler: "turbopack",
    port: 18182,
    warmIdleMs: 65_000,
    output: "/tmp/dev-runtime-report.json",
    idleCheckIntervalMs: 30_000,
  }
  const baseEnv = {
    NODE_ENV: "test",
    PATH: "/usr/bin",
    TLDW_E2E_ALLOW_OFFLINE: "0",
    TLDW_E2E_SERVER_URL: "http://127.0.0.1:18180",
    TLDW_E2E_API_KEY: "synthetic-key",
  }

  function createOperations({
    traversalExitCode = 0,
    probeFailureAt = 0,
    rssKibibytesBySample = [8_388_608],
  } = {}) {
    const events: string[] = []
    let writtenReport = ""
    let probeCount = 0
    let processSampleCount = 0
    const processRecord = { pid: 10 }
    return {
      events,
      processRecord,
      get writtenReport() {
        return writtenReport
      },
      operations: {
        clearBuildOutput: (frontendRoot: string) => {
          events.push(`clear-build:${frontendRoot}/.next`)
        },
        now: () => Date.UTC(2026, 7, 25, 16, 0, events.length),
        probeHttpOk: async (url: string) => {
          probeCount += 1
          events.push(`probe:${url}`)
          if (probeCount === probeFailureAt) throw new Error("probe deadline exceeded")
          return { status: 200 }
        },
        readProcessTable: async () => {
          events.push("process-table")
          const rssKibibytes = rssKibibytesBySample[
            Math.min(processSampleCount, rssKibibytesBySample.length - 1)
          ]
          processSampleCount += 1
          return `10 1 ${rssKibibytes} 1.0 bun run dev:turbopack\n11 10 1024 0.5 next-server`
        },
        runCommand: async (command: { name: string }) => {
          events.push(`run:${command.name}`)
          return {
            code: command.name === "all-pages-traversal" ? traversalExitCode : 0,
            signal: null,
          }
        },
        sleep: async (durationMs: number) => {
          events.push(`sleep:${durationMs}`)
        },
        spawnLoggedProcess: (command: { name: string }) => {
          events.push(`spawn:${command.name}`)
          return processRecord
        },
        stopProcessTree: async (record: unknown) => {
          expect(record).toBe(processRecord)
          events.push("stop")
        },
        waitForHttpOk: async (url: string) => {
          events.push(`http:${url}`)
          return { status: 200 }
        },
        writeReport: (_path: string, content: string) => {
          events.push("write-report")
          writtenReport = content
        },
      },
    }
  }

  it("records every required phase and tears down only its spawned frontend", async () => {
    const harness = createOperations()

    const report = await runDevRuntimeUat({
      options,
      baseEnv,
      frontendRoot: "/repo/apps/tldw-frontend",
      operations: harness.operations,
    })

    expect(report.status).toBe("qualified")
    expect(harness.events[0]).toBe("clear-build:/repo/apps/tldw-frontend/.next")
    expect(report.samples.map((sample: { phase: string }) => sample.phase)).toEqual([
      "initial",
      "post-traversal",
      "warm-idle-1",
      "warm-idle-2",
      "post-idle",
      "second-pass",
    ])
    expect(harness.events.filter((event) => event.startsWith("run:"))).toEqual([
      "run:all-pages-traversal",
      "run:critical-route-second-pass",
    ])
    expect(harness.events.filter((event) => event.startsWith("sleep:"))).toEqual([
      "sleep:30000",
      "sleep:30000",
      "sleep:5000",
    ])
    expect(harness.events.slice(-2)).toEqual(["write-report", "stop"])
    expect(JSON.parse(harness.writtenReport)).toEqual(report)
    expect(harness.writtenReport).not.toContain("synthetic-key")
  })

  it("writes a failed report and tears down after traversal failure", async () => {
    const harness = createOperations({ traversalExitCode: 7 })

    const report = await runDevRuntimeUat({
      options,
      baseEnv,
      frontendRoot: "/repo/apps/tldw-frontend",
      operations: harness.operations,
    })

    expect(report.status).toBe("failed")
    expect(report.failure).toBe("all-pages-traversal exited with code 7")
    expect(report.evaluation.qualified).toBe(false)
    expect(harness.events).not.toContain("run:critical-route-second-pass")
    expect(harness.events.slice(-2)).toEqual(["write-report", "stop"])
  })

  it("disqualifies a runtime when one strict warm-idle health probe misses its deadline", async () => {
    const harness = createOperations({ probeFailureAt: 3 })

    const report = await runDevRuntimeUat({
      options,
      baseEnv,
      frontendRoot: "/repo/apps/tldw-frontend",
      operations: harness.operations,
    })

    expect(report.status).toBe("failed")
    expect(report.evaluation.reasons).toContain("unresponsive")
    expect(report.failure).toContain("warm-idle-1 became unresponsive")
    expect(harness.events).not.toContain("run:critical-route-second-pass")
    expect(report.samples.find(
      (sample: { phase: string }) => sample.phase === "warm-idle-1",
    )).toMatchObject({
      responsive: false,
      healthError: "probe deadline exceeded",
    })
  })

  it("stops immediately after an irreversible RSS guardrail failure", async () => {
    const harness = createOperations({
      rssKibibytesBySample: [8_388_608, 16_777_216],
    })

    const report = await runDevRuntimeUat({
      options,
      baseEnv,
      frontendRoot: "/repo/apps/tldw-frontend",
      operations: harness.operations,
    })

    expect(report.status).toBe("failed")
    expect(report.evaluation.reasons).toContain("rss_limit")
    expect(report.failure).toContain("post-traversal reached the RSS guardrail")
    expect(harness.events).not.toContain("sleep:30000")
    expect(harness.events).not.toContain("run:critical-route-second-pass")
    expect(harness.events.slice(-2)).toEqual(["write-report", "stop"])
  })
})
