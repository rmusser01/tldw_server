#!/usr/bin/env node

import { spawn } from "node:child_process"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"

import { reservePorts } from "./onboarding-uat/ports.mjs"

const frontendRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..")
const ports = await reservePorts(["api", "hostile", "web"])
const command = process.platform === "win32" ? "playwright.cmd" : "playwright"
const child = spawn(
  command,
  ["test", "e2e/single-user-cookie-lifecycle.spec.ts", "--reporter=line"],
  {
    cwd: frontendRoot,
    env: {
      ...process.env,
      TLDW_COOKIE_LIFECYCLE: "1",
      TLDW_COOKIE_LIFECYCLE_API_PORT: String(ports.api),
      TLDW_COOKIE_LIFECYCLE_API_URL: `http://127.0.0.1:${ports.api}`,
      TLDW_COOKIE_LIFECYCLE_HOSTILE_PORT: String(ports.hostile),
      TLDW_WEB_URL: `http://localhost:${ports.web}`,
      TLDW_WEB_CMD: `bun run dev -- -p ${ports.web}`,
    },
    stdio: "inherit",
  }
)

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => child.kill(signal))
}

child.once("error", (error) => {
  console.error(error)
  process.exitCode = 1
})
child.once("exit", (code, signal) => {
  process.exitCode = code ?? (signal ? 1 : 0)
})
