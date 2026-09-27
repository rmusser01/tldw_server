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

/**
 * @param {string} url
 * @param {{ headers?: HeadersInit, timeoutMs?: number, intervalMs?: number, signal?: AbortSignal }} [options]
 */
export async function waitForHttpOk(url, {
  headers,
  timeoutMs = 30_000,
  intervalMs = 250,
  signal,
} = {}) {
  const startedAt = Date.now()
  let lastError

  while (Date.now() - startedAt <= timeoutMs) {
    if (signal?.aborted) {
      throw signal.reason instanceof Error
        ? signal.reason
        : new Error("HTTP readiness wait aborted")
    }
    try {
      const response = await fetch(url, { headers, redirect: "manual", signal })
      if (response.status >= 200 && response.status < 300) {
        return response
      }
      lastError = new Error(`HTTP ${response.status} from ${url}`)
    } catch (error) {
      if (signal?.aborted) {
        throw signal.reason instanceof Error
          ? signal.reason
          : new Error("HTTP readiness wait aborted")
      }
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

function processTargetExists(target, killProcess) {
  try {
    killProcess(target, 0)
    return true
  } catch (error) {
    if (error?.code === "ESRCH") {
      return false
    }
    throw error
  }
}

async function waitForProcessTargetExit(
  target,
  { timeoutMs, killProcess, wait },
) {
  const deadline = Date.now() + Math.max(0, timeoutMs)
  while (processTargetExists(target, killProcess)) {
    const remainingMs = deadline - Date.now()
    if (remainingMs <= 0) {
      return false
    }
    await wait(Math.min(25, remainingMs))
  }
  return true
}

async function stopDetachedPosixGroup(
  child,
  { timeoutMs, killProcess, wait },
) {
  const target = -child.pid
  try {
    killProcess(target, "SIGTERM")
  } catch (error) {
    if (error?.code === "ESRCH") return true
    if (error?.code === "EPERM") return false
    throw error
  }
  try {
    if (await waitForProcessTargetExit(target, { timeoutMs, killProcess, wait })) {
      return true
    }
  } catch (error) {
    if (error?.code === "EPERM") return false
    throw error
  }

  try {
    killProcess(target, "SIGKILL")
  } catch (error) {
    if (error?.code === "ESRCH") return true
    if (error?.code === "EPERM") return false
    throw error
  }
  const killTimeoutMs = Math.max(1000, timeoutMs)
  let exited
  try {
    exited = await waitForProcessTargetExit(target, { timeoutMs: killTimeoutMs, killProcess, wait })
  } catch (error) {
    if (error?.code === "EPERM") return false
    throw error
  }
  if (!exited) {
    throw new Error(`Timed out waiting for process group ${target} to exit`)
  }
  return true
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

export async function stopProcessTree(childOrRecord, {
  timeoutMs = 5_000,
  platform = process.platform,
  taskkill = defaultTaskkill,
  killProcess = process.kill.bind(process),
  wait = (delayMs) => new Promise((resolve) => setTimeout(resolve, delayMs)),
} = {}) {
  const child = childFromRecord(childOrRecord)
  if (!child?.pid) {
    return
  }

  if (platform === "win32") {
    if (childHasExited(child)) return
    await taskkill(child.pid, timeoutMs)
    if (childHasExited(child)) return
    await new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error(`Timed out waiting for process ${child.pid} to exit`)), timeoutMs)
      child.once("exit", () => { clearTimeout(timer); resolve() })
    })
    return
  }

  if (child.spawnfile) {
    const groupHandled = await stopDetachedPosixGroup(child, { timeoutMs, killProcess, wait })
    if (groupHandled) return
  }

  if (childHasExited(child)) {
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
