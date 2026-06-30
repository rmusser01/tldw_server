import { spawn } from "node:child_process"
import { mkdirSync, appendFileSync } from "node:fs"
import path from "node:path"
import { redactText } from "./artifacts.mjs"

function appendRedacted(logPath, chunk) {
  mkdirSync(path.dirname(logPath), { recursive: true })
  appendFileSync(logPath, redactText(chunk), "utf8")
}

export function spawnLoggedProcess({ name, command, args = [], cwd, env, logPath }) {
  if (!name || !command || !logPath) {
    throw new Error("spawnLoggedProcess requires name, command, and logPath")
  }

  const child = spawn(command, args, {
    cwd,
    env,
    stdio: ["ignore", "pipe", "pipe"],
    detached: process.platform !== "win32",
  })

  appendRedacted(logPath, `[${name}] started pid=${child.pid ?? "unknown"}\n`)
  child.stdout?.on("data", (chunk) => appendRedacted(logPath, chunk))
  child.stderr?.on("data", (chunk) => appendRedacted(logPath, chunk))
  child.once("error", (error) => {
    appendRedacted(logPath, `[${name}] error: ${error.message}\n`)
  })
  child.once("exit", (code, signal) => {
    appendRedacted(logPath, `[${name}] exited code=${code} signal=${signal}\n`)
  })

  return {
    name,
    command,
    args,
    cwd,
    logPath,
    child,
    pid: child.pid,
  }
}

export async function waitForHttpOk(url, { headers, timeoutMs = 30_000, intervalMs = 250 } = {}) {
  const startedAt = Date.now()
  let lastError

  while (Date.now() - startedAt <= timeoutMs) {
    try {
      const response = await fetch(url, { headers, redirect: "manual" })
      if (response.status >= 200 && response.status < 300) {
        return response
      }
      lastError = new Error(`HTTP ${response.status} from ${url}`)
    } catch (error) {
      lastError = error
    }

    await new Promise((resolve) => setTimeout(resolve, intervalMs))
  }

  throw new Error(`Timed out waiting for ${url}: ${lastError?.message ?? "no response"}`)
}

function childFromRecord(childOrRecord) {
  return childOrRecord?.child ?? childOrRecord
}

function childHasExited(child) {
  return child?.exitCode !== null || child?.signalCode !== null
}

function signalChild(child, signal) {
  if (!child?.pid || childHasExited(child)) {
    return
  }

  try {
    if (process.platform !== "win32" && child.spawnfile) {
      process.kill(-child.pid, signal)
    } else {
      child.kill(signal)
    }
  } catch (error) {
    if (error?.code !== "ESRCH") {
      throw error
    }
  }
}

export async function stopProcessTree(childOrRecord, { timeoutMs = 5_000 } = {}) {
  const child = childFromRecord(childOrRecord)
  if (!child?.pid || childHasExited(child)) {
    return
  }

  await new Promise((resolve) => {
    let settled = false
    let termTimer
    let killFallbackTimer
    const finish = () => {
      if (!settled) {
        settled = true
        if (termTimer) {
          clearTimeout(termTimer)
        }
        if (killFallbackTimer) {
          clearTimeout(killFallbackTimer)
        }
        resolve()
      }
    }

    child.once("exit", finish)
    signalChild(child, "SIGTERM")
    if (childHasExited(child)) {
      finish()
      return
    }

    termTimer = setTimeout(() => {
      if (childHasExited(child)) {
        finish()
        return
      }
      if (!settled && !childHasExited(child)) {
        signalChild(child, "SIGKILL")
        if (childHasExited(child)) {
          finish()
          return
        }
        killFallbackTimer = setTimeout(finish, Math.max(1000, timeoutMs))
        killFallbackTimer.unref?.()
      }
    }, timeoutMs).unref()
    termTimer.unref?.()
  })
}
