export type AudioErrorCategory =
  | "missing_credentials"
  | "missing_model"
  | "engine_unavailable"
  | "unsupported_capability"
  | "microphone_blocked"
  | "capture_busy"
  | "network"
  | "timeout"
  | "unknown"

export type AudioErrorClassification = {
  category: AudioErrorCategory
  title: string
  recovery: string
  settingsHref?: "/settings/speech"
  debugMessage: string
}

const SECRET_PATTERN =
  /\b(?:sk|pk|rk|xai|hf|ghp|gho|ghu|github_pat|tldw)[A-Za-z0-9_\-]{6,}\b/g

function stringifyError(error: unknown): string {
  if (error instanceof Error) return `${error.name}: ${error.message}`
  if (typeof error === "string") return error
  if (error && typeof error === "object") {
    const parts: string[] = []
    const record = error as Record<string, unknown>
    if (typeof record.name === "string") parts.push(record.name)
    if (typeof record.code === "string") parts.push(record.code)
    if (typeof record.message === "string") parts.push(record.message)
    const status = (record.response as { status?: unknown } | undefined)?.status
    if (typeof status === "number" || typeof status === "string") {
      parts.push(`status ${status}`)
    }
    if (parts.length > 0) return parts.join(": ")
  }
  return String(error ?? "")
}

function sanitizeDebugMessage(value: string): string {
  return value.replace(SECRET_PATTERN, "[redacted]")
}

function containsSecret(value: string): boolean {
  return new RegExp(SECRET_PATTERN.source).test(value)
}

function includesAny(value: string, needles: string[]): boolean {
  return needles.some((needle) => value.includes(needle))
}

export function classifyAudioError(error: unknown): AudioErrorClassification {
  const rawMessage = stringifyError(error)
  const debugMessage = sanitizeDebugMessage(rawMessage || "Unknown audio error")
  const lower = rawMessage.toLowerCase()

  if (
    includesAny(lower, [
      "notallowederror",
      "permission denied",
      "microphone permission",
      "permission dismissed",
      "permission blocked"
    ])
  ) {
    return {
      category: "microphone_blocked",
      title: "Microphone access is blocked",
      recovery: "Allow microphone access in your browser permission settings, then try recording again.",
      debugMessage
    }
  }

  const captureBusyMatch = rawMessage.match(
    /audio capture is already active for ([\w:-]+)/i
  )
  if (captureBusyMatch) {
    return {
      category: "capture_busy",
      title: "Audio capture is already active",
      recovery: `Audio capture is already active for ${captureBusyMatch[1]}`,
      debugMessage
    }
  }

  if (
    includesAny(lower, [
      "timeout",
      "timed out",
      "err_timed_out",
      "stream idle timeout"
    ])
  ) {
    return {
      category: "timeout",
      title: "Audio request timed out",
      recovery: "Retry once, then check server health or use a smaller audio/text sample.",
      debugMessage
    }
  }

  if (
    includesAny(lower, [
      "model not found",
      "no such model",
      "unknown model",
      "model is not available",
      "model unavailable",
      "install required dependencies",
      "not downloaded"
    ]) ||
    /model .+not found/.test(lower)
  ) {
    return {
      category: "missing_model",
      title: "Model is not available",
      recovery: "Install or select an available model in the Audio Setup Guide.",
      debugMessage
    }
  }

  if (
    includesAny(lower, [
      "401",
      "403",
      "unauthorized",
      "forbidden",
      "api key",
      "authentication token",
      "missing authentication",
      "missing elevenlabs configuration",
      "credential"
    ]) ||
    containsSecret(rawMessage)
  ) {
    return {
      category: "missing_credentials",
      title: "Credentials need attention",
      recovery: "Open Settings -> Speech, check the selected provider credentials, and retry.",
      settingsHref: "/settings/speech",
      debugMessage
    }
  }

  if (
    includesAny(lower, [
      "failed to fetch",
      "networkerror",
      "err_network",
      "enotfound",
      "econnrefused",
      "connection refused",
      "server not configured"
    ])
  ) {
    return {
      category: "network",
      title: "Audio service is unreachable",
      recovery: "Check the tldw server connection, API key, and network status, then retry.",
      debugMessage
    }
  }

  if (
    includesAny(lower, [
      "engine unavailable",
      "audio/speech api not detected",
      "no audio artifact",
      "no transcription models available",
      "ffmpeg",
      "cuda"
    ])
  ) {
    return {
      category: "engine_unavailable",
      title: "Audio engine is unavailable",
      recovery: "Check the Audio Setup Guide and server logs, then retry after the engine is ready.",
      debugMessage
    }
  }

  if (
    includesAny(lower, [
      "unsupported",
      "not supported",
      "unsupported capability"
    ])
  ) {
    return {
      category: "unsupported_capability",
      title: "This audio option is not supported",
      recovery: "Change the selected model, format, or provider and try again.",
      debugMessage
    }
  }

  return {
    category: "unknown",
    title: "Audio request failed",
    recovery: "Retry the request. If it fails again, check server health and provider settings.",
    debugMessage
  }
}
