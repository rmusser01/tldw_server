export const EMOTE_EVENT_LIMIT = 5
export const CHARACTER_EMOTE_STATE_PATTERN = /^[a-z0-9][a-z0-9_-]{0,39}$/

export type CharacterEmoteEvent = {
  state: string
  at_char: number
}

export type CharacterEmoteParseResult = {
  cleanText: string
  events: CharacterEmoteEvent[]
}

type StreamParseResult = {
  visibleText: string
  events: CharacterEmoteEvent[]
}

const DIRECTIVE_PATTERN = /^emote:(.*)$/i

export const normalizeCharacterEmoteState = (value: unknown): string | null => {
  if (typeof value !== "string") return null
  const normalized = value.trim().toLowerCase().replace(/\s+/g, "-")
  return CHARACTER_EMOTE_STATE_PATTERN.test(normalized) ? normalized : null
}

const parseDirectiveState = (line: string): string | null | undefined => {
  const match = line.trim().match(DIRECTIVE_PATTERN)
  return match ? normalizeCharacterEmoteState(match[1]) : undefined
}

const isFenceLine = (line: string): boolean => line.trim().startsWith("```")

export const parseCharacterEmoteDirectives = (
  input: string
): CharacterEmoteParseResult => {
  let cleanText = ""
  const events: CharacterEmoteEvent[] = []
  let inFence = false
  let lastState: string | null = null
  let index = 0

  while (index < input.length) {
    const newlineIndex = input.indexOf("\n", index)
    const hasNewline = newlineIndex !== -1
    const line = hasNewline ? input.slice(index, newlineIndex) : input.slice(index)
    const separator = hasNewline ? "\n" : ""
    index = hasNewline ? newlineIndex + 1 : input.length

    if (isFenceLine(line)) {
      cleanText += line + separator
      inFence = !inFence
      continue
    }

    if (!inFence) {
      const state = parseDirectiveState(line)
      if (state !== undefined) {
        if (state && state !== lastState && events.length < EMOTE_EVENT_LIMIT) {
          events.push({ state, at_char: cleanText.length })
          lastState = state
        }
        continue
      }
    }

    cleanText += line + separator
  }

  return { cleanText, events }
}

export const isValidCharacterEmoteEvent = (value: unknown): boolean => {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false
  const event = value as Partial<CharacterEmoteEvent>
  return (
    typeof event.state === "string" &&
    CHARACTER_EMOTE_STATE_PATTERN.test(event.state) &&
    Number.isInteger(event.at_char) &&
    Number(event.at_char) >= 0
  )
}

export const createCharacterEmoteStreamParser = () => {
  let buffer = ""
  let cleanLength = 0
  let inFence = false
  let lineMode: "maybe" | "ordinary" = "maybe"
  let lastState: string | null = null
  let eventCount = 0

  const couldBeControlLine = (line: string): boolean => {
    const trimmed = line.trimStart()
    if (!trimmed) return true
    if ("```".startsWith(trimmed) || trimmed.startsWith("```")) return true
    if (inFence) return false
    const lower = trimmed.toLowerCase()
    return "emote:".startsWith(lower) || lower.startsWith("emote:")
  }

  const acceptEvent = (state: string, events: CharacterEmoteEvent[]) => {
    if (state === lastState || eventCount >= EMOTE_EVENT_LIMIT) return
    events.push({ state, at_char: cleanLength })
    lastState = state
    eventCount += 1
  }

  const parseBufferedLine = (rawLine: string): StreamParseResult => {
    const hasNewline = rawLine.endsWith("\n")
    const line = hasNewline ? rawLine.slice(0, -1) : rawLine
    const separator = hasNewline ? "\n" : ""
    const events: CharacterEmoteEvent[] = []

    if (isFenceLine(line)) {
      const visibleText = line + separator
      cleanLength += visibleText.length
      inFence = !inFence
      return { visibleText, events }
    }

    if (!inFence) {
      const state = parseDirectiveState(line)
      if (state !== undefined) {
        if (state) acceptEvent(state, events)
        return { visibleText: "", events }
      }
    }

    const visibleText = line + separator
    cleanLength += visibleText.length
    return { visibleText, events }
  }

  const push = (chunk: string): StreamParseResult => {
    let visibleText = ""
    const events: CharacterEmoteEvent[] = []

    for (const char of chunk) {
      if (lineMode === "ordinary") {
        visibleText += char
        cleanLength += char.length
        if (char === "\n") lineMode = "maybe"
        continue
      }

      buffer += char
      if (char === "\n") {
        const parsed = parseBufferedLine(buffer)
        visibleText += parsed.visibleText
        events.push(...parsed.events)
        buffer = ""
        continue
      }

      if (!couldBeControlLine(buffer)) {
        visibleText += buffer
        cleanLength += buffer.length
        buffer = ""
        lineMode = "ordinary"
      }
    }

    return { visibleText, events }
  }

  const flush = (): StreamParseResult => {
    if (!buffer) return { visibleText: "", events: [] }
    const parsed = parseBufferedLine(buffer)
    buffer = ""
    lineMode = "maybe"
    return parsed
  }

  return { push, flush }
}
