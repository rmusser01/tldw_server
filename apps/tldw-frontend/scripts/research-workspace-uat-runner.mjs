#!/usr/bin/env node

import { spawn } from "node:child_process"
import fs from "node:fs"
import path from "node:path"
import { pathToFileURL } from "node:url"

export const ENVIRONMENT_BLOCKED_EXIT_CODE = 75

const DEFAULT_SPECS = [
  "e2e/workflows/research-workspace.spec.ts",
  "e2e/workflows/research-workspace.real-backend.spec.ts",
]

const ENVIRONMENT_FAILURE_PATTERNS = [
  {
    code: "macos_mach_port_denied",
    pattern:
      /bootstrap_check_in[\s\S]*(?:MachPortRendezvousServer[\s\S]*)?Permission denied \(1100\)|MachPortRendezvousServer/i,
  },
  {
    code: "localhost_bind_denied",
    pattern: /(?:listen\s+)?EPERM[\s\S]*(?:0\.0\.0\.0|127\.0\.0\.1|localhost)[\s\S]*:8080/i,
  },
  {
    code: "browser_launch_failed",
    pattern:
      /browserType\.launch|Failed to launch browser|Executable doesn't exist|Target page, context or browser has been closed/i,
  },
]

const ENVIRONMENT_SKIP_PATTERNS = [
  {
    code: "no_runnable_chat_model",
    pattern:
      /server did not advertise a runnable chat model|No LLM models available/i,
  },
  {
    code: "sandbox_run_api_unavailable",
    pattern:
      /sandbox run API unavailable|POST\s+\/api\/v1\/sandbox\/runs\s+returned\s+HTTP\s+404/i,
  },
  {
    code: "server_unavailable",
    pattern: /Server is not available/i,
  },
]

const readJsonIfPresent = (filePath) => {
  if (!filePath || !fs.existsSync(filePath)) return null
  return JSON.parse(fs.readFileSync(filePath, "utf8"))
}

const resolveOutputPath = (filePath) =>
  path.isAbsolute(filePath) ? filePath : path.resolve(process.cwd(), filePath)

const splitList = (value) =>
  String(value || "")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean)

const parseArgs = (argv) => {
  const options = {
    passthroughArgs: [],
    specs: [],
  }
  let passthrough = false

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index]
    if (passthrough) {
      options.passthroughArgs.push(arg)
      continue
    }
    if (arg === "--") {
      passthrough = true
      continue
    }
    if (arg === "--help" || arg === "-h") {
      options.help = true
      continue
    }
    if (arg === "--spec") {
      const next = argv[index + 1]
      if (next) {
        options.specs.push(next)
        index += 1
      }
      continue
    }
    if (arg.startsWith("--spec=")) {
      options.specs.push(arg.slice("--spec=".length))
      continue
    }
    const valueOption = [
      "--web-url",
      "--web-cmd",
      "--api-url",
      "--project",
      "--workers",
      "--report",
      "--evidence",
      "--grep",
    ].find((name) => arg === name || arg.startsWith(`${name}=`))
    if (valueOption) {
      const key = valueOption.slice(2).replace(/-([a-z])/g, (_, char) =>
        char.toUpperCase()
      )
      if (arg.includes("=")) {
        options[key] = arg.slice(valueOption.length + 1)
      } else {
        const next = argv[index + 1]
        if (next) {
          options[key] = next
          index += 1
        }
      }
      continue
    }
    if (arg === "--headed") {
      options.headed = true
      continue
    }
    if (arg === "--no-autostart") {
      options.noAutostart = true
      continue
    }
    options.passthroughArgs.push(arg)
  }

  return options
}

export const buildRunnerConfig = ({ argv = process.argv.slice(2), env = process.env } = {}) => {
  const options = parseArgs(argv)
  const envSpecs = splitList(env.TLDW_RESEARCH_WORKSPACE_UAT_SPECS)
  const specs = options.specs.length > 0 ? options.specs : envSpecs.length > 0 ? envSpecs : DEFAULT_SPECS
  const webUrl = options.webUrl || env.TLDW_WEB_URL || "http://localhost:8080"
  const webCommand =
    options.webCmd ||
    env.TLDW_WEB_CMD ||
    "bun run dev -- -H 127.0.0.1 -p 8080"
  const apiUrl =
    options.apiUrl ||
    env.NEXT_PUBLIC_API_URL ||
    env.TLDW_SERVER_URL ||
    env.TLDW_E2E_SERVER_URL ||
    "http://127.0.0.1:8000"

  return {
    apiUrl,
    evidencePath:
      options.evidence ||
      env.TLDW_RESEARCH_WORKSPACE_UAT_EVIDENCE ||
      "test-results/research-workspace-final-uat-evidence.json",
    grep: options.grep || env.TLDW_RESEARCH_WORKSPACE_UAT_GREP || "",
    headless: !options.headed,
    help: Boolean(options.help),
    passthroughArgs: options.passthroughArgs,
    project: options.project || env.TLDW_RESEARCH_WORKSPACE_UAT_PROJECT || "chromium",
    reportPath:
      options.report ||
      env.TLDW_RESEARCH_WORKSPACE_UAT_REPORT ||
      "test-results/research-workspace-final-uat-report.json",
    shouldAutostart:
      options.noAutostart || env.TLDW_WEB_AUTOSTART === "false" ? "false" : "true",
    specs,
    webCommand,
    webUrl,
    workers: options.workers || env.TLDW_RESEARCH_WORKSPACE_UAT_WORKERS || "1",
  }
}

export const buildPlaywrightArgs = (config) => {
  const args = [
    "playwright",
    "test",
    ...config.specs,
    "--project",
    config.project,
    "--reporter=json",
    "--workers",
    config.workers,
  ]

  if (config.grep) {
    args.push("--grep", config.grep)
  }
  if (!config.headless) {
    args.push("--headed")
  }
  args.push(...config.passthroughArgs)
  return args
}

const getReportStats = (report) => {
  const stats = report?.stats || {}
  const expected = Number(stats.expected || 0)
  const skipped = Number(stats.skipped || 0)
  const unexpected = Number(stats.unexpected || 0)
  const flaky = Number(stats.flaky || 0)
  return {
    executed: expected + skipped + unexpected + flaky,
    expected,
    flaky,
    skipped,
    unexpected,
  }
}

const collectSkippedTests = (report) => {
  const skippedTests = []

  const visitSuite = (suite) => {
    for (const spec of suite?.specs || []) {
      for (const test of spec.tests || []) {
        const isSkipped =
          test?.status === "skipped" ||
          test?.expectedStatus === "skipped" ||
          (test?.results || []).some((result) => result?.status === "skipped")
        if (!isSkipped) continue

        const annotations = [
          ...(test.annotations || []),
          ...(test.results || []).flatMap((result) => result?.annotations || []),
        ]
        skippedTests.push({
          annotations,
          title: spec.title || "",
        })
      }
    }

    for (const childSuite of suite?.suites || []) {
      visitSuite(childSuite)
    }
  }

  for (const suite of report?.suites || []) {
    visitSuite(suite)
  }

  return skippedTests
}

const getEnvironmentSkipReasons = ({ report, stats }) => {
  const skippedTests = collectSkippedTests(report)
  if (skippedTests.length === 0 || skippedTests.length !== stats.skipped) {
    return null
  }

  const reasonCodes = new Set()
  for (const skippedTest of skippedTests) {
    const description = skippedTest.annotations
      .map((annotation) => annotation?.description || "")
      .filter(Boolean)
      .join("\n")
    const matchingReasons = ENVIRONMENT_SKIP_PATTERNS.filter(({ pattern }) =>
      pattern.test(description)
    )
    if (matchingReasons.length === 0) {
      return null
    }
    for (const reason of matchingReasons) {
      reasonCodes.add(reason.code)
    }
  }

  return ["environment_skips_present", ...reasonCodes]
}

export const classifyPlaywrightRun = ({ exitCode, stdout = "", stderr = "", report }) => {
  const combinedOutput = `${stdout}\n${stderr}`
  const environmentReasons = ENVIRONMENT_FAILURE_PATTERNS.filter(({ pattern }) =>
    pattern.test(combinedOutput)
  ).map(({ code }) => code)

  if (environmentReasons.length > 0) {
    return {
      failureScope: "environment",
      reasons: environmentReasons,
      status: "environment_blocked",
    }
  }

  if (report) {
    const stats = getReportStats(report)
    const reasons = []
    if (stats.executed <= 0) reasons.push("no_tests_executed")
    if (stats.skipped > 0) reasons.push("skipped_tests_present")
    if (stats.unexpected > 0) reasons.push("unexpected_failures")
    if (stats.flaky > 0) reasons.push("flaky_tests_present")

    if (reasons.includes("no_tests_executed")) {
      return {
        failureScope: "environment",
        reasons,
        status: "environment_blocked",
      }
    }
    if (
      stats.skipped > 0 &&
      stats.unexpected === 0 &&
      stats.flaky === 0 &&
      exitCode === 0
    ) {
      const environmentSkipReasons = getEnvironmentSkipReasons({ report, stats })
      if (environmentSkipReasons) {
        return {
          failureScope: "environment",
          reasons: environmentSkipReasons,
          status: "environment_blocked",
        }
      }
    }
    if (reasons.length > 0 || exitCode !== 0) {
      return {
        failureScope: "product",
        reasons: reasons.length > 0 ? reasons : ["playwright_nonzero_exit"],
        status: "product_failed",
      }
    }
    return {
      failureScope: "none",
      reasons: [],
      status: "passed",
    }
  }

  if (exitCode === 0) {
    return {
      failureScope: "environment",
      reasons: ["missing_json_report"],
      status: "environment_blocked",
    }
  }

  return {
    failureScope: "environment",
    reasons: ["unclassified_playwright_failure"],
    status: "environment_blocked",
  }
}

export const createEvidence = ({
  classification,
  config,
  exitCode,
  finishedAt,
  report = null,
  startedAt,
}) => ({
  command: {
    args: buildPlaywrightArgs(config),
    executable: process.platform === "win32" ? "bunx.cmd" : "bunx",
  },
  environment: {
    apiUrl: config.apiUrl,
    webAutostart: config.shouldAutostart,
    webCommand: config.webCommand,
    webUrl: config.webUrl,
  },
  fallback:
    "If standalone Playwright is environment_blocked, launch Chrome with an explicit Chrome debugging endpoint and use chromium.connectOverCDP; attach screenshots, console errors, network failures, and timing notes to the UAT matrix.",
  failureScope: classification.failureScope,
  finishedAt,
  playwrightExitCode: exitCode,
  productPassed: classification.status === "passed",
  reasons: classification.reasons,
  evidencePath: resolveOutputPath(config.evidencePath),
  reportPath: resolveOutputPath(config.reportPath),
  requiredSetup: [
    "Local network permission",
    "Local network permission for 127.0.0.1 WebUI/backend access",
    "WebUI autostart command binds to 127.0.0.1 instead of 0.0.0.0",
    "Browser launch permissions for the selected Playwright channel",
    "Backend running at the configured API URL for real-backend specs",
    "Sandbox-capable backend profile with [API-Routes] stable_only = true and enable = sandbox for strict workspace sandbox diagnostics",
  ],
  specs: config.specs,
  startedAt,
  status: classification.status,
  stats: report ? getReportStats(report) : null,
})

const printUsage = () => {
  console.log(`Research Workspace final UAT runner

Usage:
  node scripts/research-workspace-uat-runner.mjs [options] [-- extra Playwright args]

Options:
  --spec <path>       Add a spec path. Repeat to override the default focused specs.
  --web-url <url>     WebUI URL. Default: http://localhost:8080
  --web-cmd <cmd>     WebUI start command. Default: bun run dev -- -H 127.0.0.1 -p 8080
  --api-url <url>     Backend API URL. Default: http://127.0.0.1:8000
  --project <name>    Playwright project. Default: chromium
  --workers <n>       Playwright workers. Default: 1
  --report <path>     Playwright JSON report path.
  --evidence <path>   UAT evidence JSON path.
  --grep <pattern>    Optional Playwright grep.
  --headed            Run headed.
  --no-autostart      Use an already-running WebUI.

Environment:
  TLDW_WEB_URL, TLDW_WEB_CMD, NEXT_PUBLIC_API_URL, TLDW_E2E_SERVER_URL
  TLDW_RESEARCH_WORKSPACE_UAT_SPECS, TLDW_RESEARCH_WORKSPACE_UAT_REPORT
  TLDW_RESEARCH_WORKSPACE_UAT_EVIDENCE, TLDW_RESEARCH_WORKSPACE_UAT_GREP
`)
}

export const runPlaywright = (config) =>
  new Promise((resolve) => {
    const reportPath = resolveOutputPath(config.reportPath)
    const evidencePath = resolveOutputPath(config.evidencePath)
    const reportDir = path.dirname(reportPath)
    const evidenceDir = path.dirname(evidencePath)
    fs.mkdirSync(reportDir, { recursive: true })
    fs.mkdirSync(evidenceDir, { recursive: true })

    const executable = process.platform === "win32" ? "bunx.cmd" : "bunx"
    const args = buildPlaywrightArgs(config)
    const child = spawn(executable, args, {
      cwd: process.cwd(),
      env: {
        ...process.env,
        NEXT_PUBLIC_API_URL: config.apiUrl,
        PLAYWRIGHT_JSON_OUTPUT_NAME: reportPath,
        TLDW_E2E_SERVER_URL: config.apiUrl,
        TLDW_WEB_AUTOSTART: config.shouldAutostart,
        TLDW_WEB_CMD: config.webCommand,
        TLDW_WEB_URL: config.webUrl,
      },
      stdio: ["ignore", "pipe", "pipe"],
    })

    let stdout = ""
    let stderr = ""
    let settled = false
    const finish = (result) => {
      if (settled) return
      settled = true
      resolve(result)
    }

    child.stdout?.on("data", (chunk) => {
      const text = String(chunk)
      stdout += text
      process.stdout.write(text)
    })
    child.stderr?.on("data", (chunk) => {
      const text = String(chunk)
      stderr += text
      process.stderr.write(text)
    })
    child.on("error", (error) => {
      const message = error instanceof Error ? error.message : String(error)
      stderr += `\n[research-workspace-uat] Failed to start Playwright: ${message}\n`
      finish({
        exitCode: 1,
        stderr,
        stdout,
      })
    })
    child.on("close", (code) => {
      finish({
        exitCode: typeof code === "number" ? code : 1,
        stderr,
        stdout,
      })
    })
  })

export const main = async ({ argv = process.argv.slice(2), env = process.env } = {}) => {
  const config = buildRunnerConfig({ argv, env })
  if (config.help) {
    printUsage()
    return 0
  }

  const startedAt = new Date().toISOString()
  const result = await runPlaywright(config)
  const finishedAt = new Date().toISOString()
  const reportPath = resolveOutputPath(config.reportPath)
  const evidencePath = resolveOutputPath(config.evidencePath)
  let report = null
  try {
    report = readJsonIfPresent(reportPath)
  } catch (error) {
    result.stderr += `\n[research-workspace-uat] Unable to parse JSON report: ${
      error instanceof Error ? error.message : String(error)
    }\n`
  }

  const classification = classifyPlaywrightRun({
    exitCode: result.exitCode,
    report,
    stderr: result.stderr,
    stdout: result.stdout,
  })
  const evidence = createEvidence({
    classification,
    config,
    exitCode: result.exitCode,
    finishedAt,
    report,
    startedAt,
  })
  fs.writeFileSync(evidencePath, `${JSON.stringify(evidence, null, 2)}\n`)

  console.log(
    `[research-workspace-uat] status=${classification.status} scope=${classification.failureScope} reasons=${classification.reasons.join(",") || "none"} evidence=${evidencePath}`
  )

  if (classification.status === "passed") return 0
  if (classification.status === "environment_blocked") {
    return ENVIRONMENT_BLOCKED_EXIT_CODE
  }
  return result.exitCode || 1
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().then((exitCode) => {
    process.exit(exitCode)
  })
}
