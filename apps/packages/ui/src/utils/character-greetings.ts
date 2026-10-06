export type CharacterGreetingFields = {
  greeting?: unknown
  first_message?: unknown
  firstMessage?: unknown
  greet?: unknown
  alternate_greetings?: unknown
  alternateGreetings?: unknown
}

/** Exact server creation input; display filtering must never change its raw index or text. */
export type GreetingSeed =
  | { greetingStrategy: "default"; rawText: string }
  | { greetingStrategy: "alternate_index"; alternateIndex: number; rawText: string }

export type GreetingOption = {
  id: string
  text: string
  index: number
  sourceKey?: string
  sourceLabel?: string
  seed?: GreetingSeed
}

export type GreetingSelectionFallback = "none" | "first" | "random"

export type ResolveGreetingSelectionArgs = {
  options: GreetingOption[]
  greetingSelectionId?: string | null
  greetingsChecksum?: string | null
  useCharacterDefault?: boolean
  fallback?: GreetingSelectionFallback
}

export type ResolvedGreetingSelection = {
  option: GreetingOption | null
  checksum: string | null
  isStale: boolean
}

export type GreetingEntry = {
  text: string
  sourceKey: string
  sourceLabel: string
  seed?: GreetingSeed
}

const normalizeStringEntries = (value: unknown[]): string[] =>
  value
    .map((entry) => (typeof entry === "string" ? entry.trim() : ""))
    .filter((entry) => entry.length > 0)

const tryParseStringifiedGreetingList = (value: string): string[] | null => {
  try {
    const parsed = JSON.parse(value)
    if (Array.isArray(parsed)) {
      return normalizeStringEntries(parsed)
    }
    if (typeof parsed === "string") {
      const nestedParsed = JSON.parse(parsed)
      if (Array.isArray(nestedParsed)) {
        return normalizeStringEntries(nestedParsed)
      }
    }
  } catch {
    return null
  }
  return null
}

export const normalizeGreetingValue = (value: unknown): string[] => {
  if (!value) return []
  if (Array.isArray(value)) {
    return normalizeStringEntries(value)
  }
  if (typeof value === "string") {
    const trimmed = value.trim()
    if (trimmed.length === 0) return []
    const parsedList = tryParseStringifiedGreetingList(trimmed)
    if (parsedList) return parsedList
    return [trimmed]
  }
  return []
}

const GREETING_SOURCES: Array<{ key: keyof CharacterGreetingFields; label: string }> =
  [
    { key: "greeting", label: "Greeting" },
    { key: "first_message", label: "First message" },
    { key: "firstMessage", label: "First message" },
    { key: "greet", label: "Greeting" },
    { key: "alternate_greetings", label: "Alternate greeting" },
    { key: "alternateGreetings", label: "Alternate greeting" }
  ]

export const collectGreetings = (
  character: CharacterGreetingFields | null | undefined
): string[] => {
  const greetings = [
    ...normalizeGreetingValue(character?.greeting),
    ...normalizeGreetingValue(character?.first_message),
    ...normalizeGreetingValue(character?.firstMessage),
    ...normalizeGreetingValue(character?.greet),
    ...normalizeGreetingValue(character?.alternate_greetings),
    ...normalizeGreetingValue(character?.alternateGreetings)
  ]
  return Array.from(new Set(greetings))
}

export const collectGreetingEntries = (
  character: CharacterGreetingFields | null | undefined
): GreetingEntry[] => {
  if (!character) return []
  // The backend selects first_message verbatim or one raw alternate_greetings
  // list position. Compatibility aliases and nested JSON are display-only.
  const seeds = new Map<string, GreetingSeed>()
  if (typeof character.first_message === "string" && character.first_message.trim())
    seeds.set(character.first_message.trim(), { greetingStrategy: "default", rawText: character.first_message })
  let alternates = character.alternate_greetings
  if (typeof alternates === "string") {
    try { alternates = JSON.parse(alternates) } catch { alternates = null }
  }
  if (Array.isArray(alternates)) alternates.forEach((rawText, alternateIndex) => {
    if (typeof rawText === "string" && rawText.trim() && !seeds.has(rawText.trim()))
      seeds.set(rawText.trim(), { greetingStrategy: "alternate_index", alternateIndex, rawText })
  })
  const entries: GreetingEntry[] = []
  GREETING_SOURCES.forEach(({ key, label }) => {
    const values = normalizeGreetingValue(
      character[key as keyof CharacterGreetingFields]
    )
    values.forEach((text) => {
      entries.push({
        text,
        sourceKey: String(key),
        sourceLabel: label,
        ...(seeds.has(text) ? { seed: seeds.get(text) } : {})
      })
    })
  })

  const seen = new Set<string>()
  const deduped: GreetingEntry[] = []
  entries.forEach((entry) => {
    if (seen.has(entry.text)) return
    seen.add(entry.text)
    deduped.push(entry)
  })

  return deduped
}

const hashString = (value: string): string => {
  let hash = 2166136261
  for (let i = 0; i < value.length; i += 1) {
    hash ^= value.charCodeAt(i)
    hash +=
      (hash << 1) +
      (hash << 4) +
      (hash << 7) +
      (hash << 8) +
      (hash << 24)
  }
  return (hash >>> 0).toString(36)
}

export const buildGreetingId = (text: string, index: number): string =>
  `greeting:${index}:${hashString(text)}`

export const buildGreetingOptions = (greetings: string[]): GreetingOption[] =>
  greetings.map((text, index) => ({
    id: buildGreetingId(text, index),
    text,
    index
  }))

export const buildGreetingOptionsFromEntries = (
  entries: GreetingEntry[]
): GreetingOption[] =>
  entries.map((entry, index) => ({
    id: buildGreetingId(entry.text, index),
    text: entry.text,
    index,
    sourceKey: entry.sourceKey,
    sourceLabel: entry.sourceLabel,
    ...(entry.seed ? { seed: entry.seed } : {})
  }))

export const buildGreetingsChecksumFromOptions = (
  options: GreetingOption[]
): string =>
  hashString(
    options
      .map((option) => `${option.id}:${option.text}`)
      .join("\u001f")
  )

export const buildGreetingsChecksum = (greetings: string[]): string =>
  buildGreetingsChecksumFromOptions(buildGreetingOptions(greetings))

export const isGreetingMessageType = (messageType?: string | null): boolean =>
  messageType === "character:greeting" || messageType === "greeting"

export const parseGreetingSelectionIndex = (
  selectionId: unknown
): number | null => {
  if (typeof selectionId !== "string") return null
  const parts = selectionId.trim().split(":")
  if (parts.length < 3 || parts[0] !== "greeting") return null
  const index = Number(parts[1])
  if (!Number.isInteger(index) || index < 0) return null
  return index
}

export const resolveGreetingSelection = ({
  options,
  greetingSelectionId,
  greetingsChecksum,
  useCharacterDefault = false,
  fallback = "first"
}: ResolveGreetingSelectionArgs): ResolvedGreetingSelection => {
  const checksum =
    options.length > 0 ? buildGreetingsChecksumFromOptions(options) : null
  const storedSelectionId =
    typeof greetingSelectionId === "string" ? greetingSelectionId : null
  const storedChecksum =
    typeof greetingsChecksum === "string" ? greetingsChecksum : null
  const isStale =
    Boolean(storedChecksum) && Boolean(checksum)
      ? storedChecksum !== checksum
      : false

  let option =
    !isStale && storedSelectionId
      ? options.find((candidate) => candidate.id === storedSelectionId)
      : undefined
  if (!option && !isStale && storedSelectionId) {
    const selectedIndex = parseGreetingSelectionIndex(storedSelectionId)
    if (selectedIndex != null) {
      option = options[selectedIndex]
    }
  }

  if (!option) {
    if (useCharacterDefault) {
      option = options[0]
    } else if (fallback === "first") {
      option = options[0]
    } else if (fallback === "random" && options.length > 0) {
      option = options[Math.floor(Math.random() * options.length)]
    }
  }

  return {
    option: option ?? null,
    checksum,
    isStale
  }
}

export const pickGreeting = (greetings: string[]): string => {
  if (greetings.length === 0) return ""
  if (greetings.length === 1) return greetings[0]
  const index = Math.floor(Math.random() * greetings.length)
  return greetings[index]
}

/** Display classification only: canonical H1 rows and provider requests stay raw. */
export const resolveCharacterMessageType = (
  message: { messageType?: string; message_type?: string; isBot?: boolean; role?: string;
    parentMessageId?: string | null; id?: string; serverMessageId?: string },
  index: number,
  nativeCharacter: boolean
): string | undefined => message.messageType ?? message.message_type ?? (
  nativeCharacter && index === 0 && (message.role === "assistant" || message.isBot) &&
    message.parentMessageId == null && (message.serverMessageId?.trim() || message.id?.trim()) ? "character:greeting" : undefined
)
