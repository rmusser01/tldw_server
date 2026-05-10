import type { PersonaVisualDiagnostic } from "@/components/Common/PersonaBuddy/personaVisualDiagnostics"

export type PersonaBuddyDiagnosticState =
  | "healthy"
  | "unavailable"
  | "degraded"
  | "recovering"

export type PersonaBuddyProfileState = "idle" | "loading" | "loaded" | "error"

export type PersonaBuddyDiagnosticRow = {
  label: string
  value: string
  state?: PersonaBuddyDiagnosticState
  detail?: string
}

export type PersonaBuddyDiagnostics = {
  state: PersonaBuddyDiagnosticState
  title: string
  message: string
  rows: PersonaBuddyDiagnosticRow[]
}

export type PersonaBuddyDiagnosticsInput = {
  selectedPersona?: {
    id?: string | null
    name?: string | null
  } | null
  profileState?: PersonaBuddyProfileState
  profileError?: string | null
  buddySummary?:
    | string
    | {
        has_buddy?: boolean | null
        persona_name?: string | null
        role_summary?: string | null
      }
    | null
  capabilities?: {
    hasPersona?: boolean | null
    hasMcp?: boolean | null
  } | null
  capabilitiesLoading?: boolean
  liveSession?: {
    connected?: boolean
    connecting?: boolean
    sessionId?: string | null
    error?: string | null
    lastEvent?: string | null
  } | null
  liveVoice?: {
    state?: string | null
    recoveryMode?: string | null
    warning?: string | null
    activeToolStatus?: string | null
    textOnlyDueToTtsFailure?: boolean
    manualModeRequired?: boolean
  } | null
  wake?: {
    armed?: boolean
    detectorState?: string | null
    warning?: string | null
    triggerPhrases?: string[] | null
    behavior?: string | null
  } | null
  visual?: {
    packId?: string | null
    packTitle?: string | null
    packLoadStatus?: "idle" | "loading" | "loaded" | "error"
    visualState?: string | null
    diagnostic?: PersonaVisualDiagnostic | null
  } | null
}

const stateRank: Record<PersonaBuddyDiagnosticState, number> = {
  healthy: 0,
  degraded: 1,
  recovering: 2,
  unavailable: 3
}

const worseState = (
  current: PersonaBuddyDiagnosticState,
  candidate: PersonaBuddyDiagnosticState
): PersonaBuddyDiagnosticState =>
  stateRank[candidate] > stateRank[current] ? candidate : current

const titleCase = (value: string | null | undefined, fallback = "Unknown") => {
  const normalized = String(value || "").trim()
  if (!normalized) return fallback
  return normalized
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase())
}

const summarizeBuddy = (
  buddySummary: PersonaBuddyDiagnosticsInput["buddySummary"]
): PersonaBuddyDiagnosticRow => {
  if (typeof buddySummary === "string" && buddySummary.trim()) {
    return {
      label: "Buddy",
      value: "Ready",
      state: "healthy",
      detail: buddySummary.trim()
    }
  }

  if (buddySummary && typeof buddySummary === "object") {
    const roleSummary = String(buddySummary.role_summary || "").trim()
    if (buddySummary.has_buddy || roleSummary) {
      return {
        label: "Buddy",
        value: "Ready",
        state: "healthy",
        detail: roleSummary || "Buddy summary is attached to this persona."
      }
    }
  }

  return {
    label: "Buddy",
    value: "Dormant",
    state: "degraded",
    detail: "No Buddy summary is attached to the selected persona."
  }
}

const summarizeVisual = (
  visual: PersonaBuddyDiagnosticsInput["visual"]
): PersonaBuddyDiagnosticRow => {
  const diagnostic = visual?.diagnostic
  const renderState = visual?.visualState
    ? `Render state: ${titleCase(visual.visualState)}`
    : undefined
  if (diagnostic?.code === "no_active_pack") {
    return {
      label: "Visual pack",
      value: "Default Buddy fallback",
      state: "healthy",
      detail: [diagnostic.message, renderState].filter(Boolean).join(" - ")
    }
  }

  if (diagnostic) {
    return {
      label: "Visual pack",
      value: diagnostic.title || titleCase(diagnostic.code),
      state: diagnostic.severity === "info" ? "healthy" : "degraded",
      detail: [diagnostic.message, renderState].filter(Boolean).join(" - ")
    }
  }

  if (visual?.packLoadStatus === "loading") {
    return {
      label: "Visual pack",
      value: "Loading",
      state: "recovering",
      detail: renderState
    }
  }

  if (visual?.packLoadStatus === "error") {
    return {
      label: "Visual pack",
      value: "Load failed",
      state: "degraded",
      detail: ["The active visual pack could not be loaded.", renderState]
        .filter(Boolean)
        .join(" - ")
    }
  }

  const activePack = visual?.packTitle || visual?.packId
  return {
    label: "Visual pack",
    value: activePack ? String(activePack) : "Default Buddy fallback",
    state: "healthy",
    detail: renderState
  }
}

const summarizeLiveSession = (
  liveSession: PersonaBuddyDiagnosticsInput["liveSession"]
): PersonaBuddyDiagnosticRow => {
  if (liveSession?.connecting) {
    return {
      label: "Live session",
      value: "Reconnecting",
      state: "recovering",
      detail: liveSession.sessionId ? `Session ${liveSession.sessionId}` : undefined
    }
  }

  if (liveSession?.error) {
    return {
      label: "Live session",
      value: "Connection issue",
      state: "degraded",
      detail: liveSession.error
    }
  }

  if (liveSession?.connected) {
    return {
      label: "Live session",
      value: "Connected",
      state: "healthy",
      detail: liveSession.sessionId ? `Session ${liveSession.sessionId}` : undefined
    }
  }

  return {
    label: "Live session",
    value: "Ready to connect",
    state: "healthy"
  }
}

const summarizeLiveVoice = (
  liveVoice: PersonaBuddyDiagnosticsInput["liveVoice"]
): PersonaBuddyDiagnosticRow => {
  const recoveryMode = String(liveVoice?.recoveryMode || "none").trim()
  if (recoveryMode && recoveryMode !== "none") {
    return {
      label: "Live voice",
      value: "Recovering",
      state: "recovering",
      detail: titleCase(recoveryMode)
    }
  }

  if (
    liveVoice?.warning ||
    liveVoice?.textOnlyDueToTtsFailure ||
    liveVoice?.manualModeRequired
  ) {
    return {
      label: "Live voice",
      value: liveVoice.textOnlyDueToTtsFailure ? "Text-only fallback" : "Limited",
      state: "degraded",
      detail: liveVoice.warning || "Manual voice controls are required for this session."
    }
  }

  return {
    label: "Live voice",
    value: titleCase(liveVoice?.state, "Idle"),
    state: "healthy",
    detail: liveVoice?.activeToolStatus || undefined
  }
}

const summarizeWake = (
  wake: PersonaBuddyDiagnosticsInput["wake"]
): PersonaBuddyDiagnosticRow => {
  const detectorState = String(wake?.detectorState || "idle").trim()
  if (wake?.warning || detectorState === "unavailable" || detectorState === "error") {
    return {
      label: "Wake",
      value: "Limited",
      state: "degraded",
      detail: wake?.warning || `Wake detector state: ${titleCase(detectorState)}`
    }
  }

  const phrases = wake?.triggerPhrases?.filter((phrase) => phrase.trim())
  return {
    label: "Wake",
    value: wake?.armed ? "Armed" : "Not armed",
    state: "healthy",
    detail:
      phrases && phrases.length > 0
        ? `Listening for ${phrases.join(", ")}`
        : wake?.behavior
          ? `Behavior: ${titleCase(wake.behavior)}`
          : undefined
  }
}

const summarizeMcp = (
  capabilities: PersonaBuddyDiagnosticsInput["capabilities"]
): PersonaBuddyDiagnosticRow => {
  if (capabilities?.hasMcp === false) {
    return {
      label: "MCP persona_visuals",
      value: "Transport unavailable",
      state: "degraded",
      detail: "MCP is not advertised by the current server capabilities."
    }
  }

  if (capabilities?.hasMcp === true) {
    return {
      label: "MCP persona_visuals",
      value: "Transport ready",
      state: "healthy",
      detail: "Tool-level readiness is not checked by this client state."
    }
  }

  return {
    label: "MCP persona_visuals",
    value: "Unknown",
    state: "healthy",
    detail: "Server capability discovery has not reported MCP readiness."
  }
}

const deriveTitleAndMessage = (
  state: PersonaBuddyDiagnosticState,
  rows: PersonaBuddyDiagnosticRow[]
) => {
  if (state === "unavailable") {
    const unavailable = rows.find((row) => row.state === "unavailable")
    return {
      title: "Persona Buddy unavailable",
      message:
        unavailable?.detail ||
        "Persona support is not available from the current server state."
    }
  }
  if (state === "recovering") {
    return {
      title: "Persona Buddy recovering",
      message:
        "Live session or voice state is reconnecting. Existing controls remain available."
    }
  }
  if (state === "degraded") {
    return {
      title: "Persona Buddy degraded",
      message:
        "Core controls are available, but one or more assistant surfaces need attention."
    }
  }
  return {
    title: "Persona Buddy ready",
    message: "Persona Live, Buddy, wake, MCP, and visual state are ready."
  }
}

export const buildPersonaBuddyDiagnostics = ({
  selectedPersona = null,
  profileState = "idle",
  profileError = null,
  buddySummary = null,
  capabilities = null,
  capabilitiesLoading = false,
  liveSession = null,
  liveVoice = null,
  wake = null,
  visual = null
}: PersonaBuddyDiagnosticsInput): PersonaBuddyDiagnostics => {
  let state: PersonaBuddyDiagnosticState = "healthy"
  const rows: PersonaBuddyDiagnosticRow[] = []

  const personaName = selectedPersona?.name || selectedPersona?.id
  const personaRow: PersonaBuddyDiagnosticRow = {
    label: "Persona",
    value: personaName ? String(personaName) : "Not selected",
    state: personaName ? "healthy" : "unavailable",
    detail: selectedPersona?.id ? `ID: ${selectedPersona.id}` : "Select a persona to use Buddy."
  }
  rows.push(personaRow)
  state = worseState(state, personaRow.state || "healthy")

  if (capabilitiesLoading) {
    const row: PersonaBuddyDiagnosticRow = {
      label: "Server capability",
      value: "Checking",
      state: "recovering",
      detail: "Waiting for server capability discovery."
    }
    rows.push(row)
    state = worseState(state, row.state || "healthy")
  } else if (!capabilities || capabilities.hasPersona === false) {
    const row: PersonaBuddyDiagnosticRow = {
      label: "Server capability",
      value: "Persona unavailable",
      state: "unavailable",
      detail: "The current server does not advertise Persona support."
    }
    rows.push(row)
    state = worseState(state, row.state || "healthy")
  } else {
    rows.push({
      label: "Server capability",
      value: "Persona ready",
      state: "healthy"
    })
  }

  const profileRow: PersonaBuddyDiagnosticRow =
    profileState === "loading"
      ? {
          label: "Profile",
          value: "Loading",
          state: "recovering"
        }
      : profileState === "error"
        ? {
            label: "Profile",
            value: "Load failed",
            state: "degraded",
            detail: profileError || "The selected persona profile could not be loaded."
          }
        : profileState === "loaded"
          ? {
              label: "Profile",
              value: "Loaded",
              state: "healthy"
            }
          : {
              label: "Profile",
              value: "Unknown",
              state: "healthy"
            }
  rows.push(profileRow)
  state = worseState(state, profileRow.state || "healthy")

  const computedRows = [
    summarizeBuddy(buddySummary),
    summarizeVisual(visual),
    summarizeLiveSession(liveSession),
    summarizeLiveVoice(liveVoice),
    summarizeWake(wake),
    summarizeMcp(capabilities)
  ]

  for (const row of computedRows) {
    rows.push(row)
    state = worseState(state, row.state || "healthy")
  }

  const { title, message } = deriveTitleAndMessage(state, rows)
  return { state, title, message, rows }
}
