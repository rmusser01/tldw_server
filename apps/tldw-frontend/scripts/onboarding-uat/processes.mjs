import { spawn } from "node:child_process"
import { mkdirSync, appendFileSync } from "node:fs"
import path from "node:path"
import { redactText } from "./artifacts.mjs"

function appendRedacted(logPath, chunk, loggingErrors) {
  try {
    mkdirSync(path.dirname(logPath), { recursive: true })
    appendFileSync(logPath, redactText(chunk), "utf8")
  } catch (error) {
    if (loggingErrors.length === 0) loggingErrors.push(error)
  }
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

  const loggingErrors = []
  appendRedacted(logPath, `[${name}] started pid=${child.pid ?? "unknown"}\n`, loggingErrors)
  child.stdout?.on("data", (chunk) => appendRedacted(logPath, chunk, loggingErrors))
  child.stderr?.on("data", (chunk) => appendRedacted(logPath, chunk, loggingErrors))
  child.once("error", (error) => {
    appendRedacted(logPath, `[${name}] error: ${error.message}\n`, loggingErrors)
  })
  child.once("exit", (code, signal) => {
    appendRedacted(logPath, `[${name}] exited code=${code} signal=${signal}\n`, loggingErrors)
  })

  return {
    name,
    command,
    args,
    cwd,
    logPath,
    child,
    pid: child.pid,
    loggingErrors,
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

function signalChild(child, signal, platform = process.platform) {
  if (!child?.pid || childHasExited(child)) {
    return
  }

  try {
    if (platform !== "win32" && child.spawnfile) {
      process.kill(-child.pid, signal)
    } else {
      child.kill(signal)
    }
  } catch (error) {
    if (error?.code !== "ESRCH") {
      if (error?.code === "EPERM" && child.spawnfile) {
        child.kill(signal)
      } else {
        throw error
      }
    }
  }
}

function defaultTaskkill(pid, timeoutMs) {
  return new Promise((resolve, reject) => {
    const taskkill = spawn("taskkill.exe", ["/PID", String(pid), "/T", "/F"], { stdio: "ignore", windowsHide: true })
    let settled = false
    const finish = (error) => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      if (error) reject(error)
      else resolve()
    }
    const timer = setTimeout(() => { try { taskkill.kill() } catch {} ; finish(new Error(`taskkill.exe timed out for PID ${pid}`)) }, timeoutMs)
    taskkill.once("error", finish)
    taskkill.once("close", (code) => finish(code === 0 ? undefined : new Error(`taskkill.exe failed for PID ${pid} with exit code ${code}`)))
  })
}

export async function stopProcessTree(childOrRecord, { timeoutMs = 5_000, platform = process.platform, taskkill = defaultTaskkill } = {}) {
  const child = childFromRecord(childOrRecord)
  if (!child?.pid || childHasExited(child)) {
    return
  }

  if (platform === "win32") {
    await taskkill(child.pid, timeoutMs)
    if (childHasExited(child)) return
    await new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error(`Timed out waiting for process ${child.pid} to exit`)), timeoutMs)
      child.once("exit", () => { clearTimeout(timer); resolve() })
    })
    return
  }

  await new Promise((resolve, reject) => {
    let settled = false
    let termTimer
    let killFallbackTimer
    const finish = (error) => {
      if (!settled) {
        settled = true
        if (termTimer) {
          clearTimeout(termTimer)
        }
        if (killFallbackTimer) {
          clearTimeout(killFallbackTimer)
        }
        if (error) reject(error)
        else resolve()
      }
    }

    child.once("exit", () => finish())
    try { signalChild(child, "SIGTERM", platform) } catch (error) { finish(error); return }
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
        try { signalChild(child, "SIGKILL", platform) } catch (error) { finish(error); return }
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
