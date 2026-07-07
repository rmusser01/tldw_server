import { createImageDataUrl } from "@/utils/image-utils"
import { normalizeCharacterEmoteState } from "./character-emotes"

const isPlainObject = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value)

const normalizeText = (value: unknown): string =>
  typeof value === "string" ? value.trim().toLowerCase() : ""

export const CHARACTER_MOOD_OPTIONS = [
  { key: "neutral", label: "Neutral" },
  { key: "happy", label: "Happy" },
  { key: "excited", label: "Excited" },
  { key: "sad", label: "Sad" },
  { key: "angry", label: "Angry" },
  { key: "thinking", label: "Thinking" },
  { key: "confused", label: "Confused" },
  { key: "surprised", label: "Surprised" }
] as const

export type CharacterMoodLabel = (typeof CHARACTER_MOOD_OPTIONS)[number]["key"]

export type CharacterMoodDetection = {
  label: CharacterMoodLabel
  confidence: number
  topic: string | null
}

export type CharacterMoodImages = Record<string, string>

type CharacterWithPortraitData = {
  avatar_url?: string | null
  image_base64?: string | null
  extensions?: unknown
} | null | undefined

const MOOD_ALIASES: Record<string, CharacterMoodLabel> = {
  neutral: "neutral",
  calm: "neutral",
  normal: "neutral",
  default: "neutral",
  happy: "happy",
  joy: "happy",
  joyful: "happy",
  cheerful: "happy",
  excited: "excited",
  hype: "excited",
  energetic: "excited",
  thrilled: "excited",
  sad: "sad",
  upset: "sad",
  sorrowful: "sad",
  unhappy: "sad",
  angry: "angry",
  mad: "angry",
  annoyed: "angry",
  furious: "angry",
  thinking: "thinking",
  thoughtful: "thinking",
  pondering: "thinking",
  reflective: "thinking",
  confused: "confused",
  unsure: "confused",
  uncertain: "confused",
  puzzled: "confused",
  surprised: "surprised",
  shocked: "surprised",
  astonished: "surprised",
  amazed: "surprised"
}

const TOPIC_STOPWORDS = new Set([
  "about",
  "after",
  "again",
  "also",
  "because",
  "before",
  "between",
  "could",
  "great",
  "hello",
  "please",
  "should",
  "thanks",
  "there",
  "their",
  "these",
  "those",
  "through",
  "would",
  "while",
  "which",
  "where",
  "when",
  "what",
  "your",
  "yours",
  "have",
  "with",
  "this",
  "that",
  "from",
  "they",
  "them",
  "been",
  "into",
  "then",
  "than",
  "just",
  "dont",
  "cant",
  "wont",
  "lets"
])

const MOOD_PATTERNS: Record<Exclude<CharacterMoodLabel, "neutral">, RegExp[]> = {
  happy: [
    /\b(happy|glad|joy|cheerful|delighted|nice|great|awesome|lovely)\b/g,
    /\b(thank you|thanks|appreciate it)\b/g
  ],
  excited: [
    /\b(excited|amazing|incredible|fantastic|let'?s go|hyped|thrilled)\b/g,
    /!{1,}/g
  ],
  sad: [
    /\b(sad|sorry|apolog(?:y|ize)|unfortunately|regret|upset)\b/g,
    /\b(i'?m sorry|i am sorry)\b/g
  ],
  angry: [
    /\b(angry|mad|furious|annoyed|frustrated|rage|outrage)\b/g,
    /\b(hate|ridiculous|unacceptable)\b/g
  ],
  thinking: [
    /\b(think|consider|analy(?:ze|sis)|reason|step by step|let'?s break)\b/g,
    /\b(maybe|perhaps|possibly)\b/g
  ],
  confused: [
    /\b(confused|unclear|unsure|uncertain|puzzled|don'?t understand)\b/g,
    /\b(not sure|hard to tell)\b/g
  ],
  surprised: [
    /\b(surprised|unexpected|whoa|wow|didn'?t expect|astonished|shocked)\b/g,
    /\?{2,}/g
  ]
}

const countPatternHits = (text: string, pattern: RegExp): number => {
  if (!text) return 0
  const matches = text.match(pattern)
  return matches ? matches.length : 0
}

const normalizeMoodImageSource = (value: unknown): string | null => {
  if (typeof value !== "string") return null
  const trimmed = value.trim()
  if (!trimmed) return null
  if (
    trimmed.startsWith("data:image/") ||
    trimmed.startsWith("http://") ||
    trimmed.startsWith("https://")
  ) {
    return trimmed
  }
  return createImageDataUrl(trimmed)
}

const clampConfidence = (value: number): number =>
  Math.max(0.35, Math.min(0.98, value))

const extractTopic = (assistantText: string, userText?: string | null): string | null => {
  const combined = `${userText || ""} ${assistantText}`.toLowerCase()
  const cleaned = combined.replace(/[^a-z0-9\s]/g, " ")
  const words = cleaned
    .split(/\s+/)
    .map((token) => token.trim())
    .filter((token) => token.length >= 4 && !TOPIC_STOPWORDS.has(token))

  if (words.length === 0) return null

  const counts = new Map<string, number>()
  for (const word of words) {
    counts.set(word, (counts.get(word) || 0) + 1)
  }

  let winner = ""
  let winnerCount = 0
  counts.forEach((count, word) => {
    if (count > winnerCount) {
      winner = word
      winnerCount = count
    }
  })

  if (!winner) return null
  return winner.slice(0, 40)
}

export const normalizeCharacterMoodLabel = (
  value: unknown
): CharacterMoodLabel | null => {
  const normalized = normalizeText(value)
  if (!normalized) return null
  return MOOD_ALIASES[normalized] || null
}

export const parseCharacterExtensions = (
  value: unknown
): Record<string, unknown> => {
  if (!value) return {}
  if (isPlainObject(value)) return { ...value }
  if (typeof value === "string") {
    const trimmed = value.trim()
    if (!trimmed) return {}
    try {
      const parsed = JSON.parse(trimmed)
      return isPlainObject(parsed) ? { ...parsed } : {}
    } catch {
      return {}
    }
  }
  return {}
}

export const getCharacterMoodImagesFromExtensions = (
  extensions: unknown
): CharacterMoodImages => {
  const parsed = parseCharacterExtensions(extensions)
  const tldw = isPlainObject(parsed.tldw) ? parsed.tldw : null

  const candidates = [
    tldw?.mood_images,
    tldw?.moodImages,
    parsed.mood_images,
    parsed.moodImages
  ]

  const source = candidates.find((candidate) => isPlainObject(candidate))
  if (!isPlainObject(source)) return {}

  const result: CharacterMoodImages = {}
  Object.entries(source).forEach(([rawMood, rawImage]) => {
    const moodLabel = normalizeCharacterEmoteState(rawMood)
    if (!moodLabel) return
    const normalizedImage = normalizeMoodImageSource(rawImage)
    if (!normalizedImage) return
    result[moodLabel] = normalizedImage
  })

  return result
}

export const mergeCharacterMoodImagesIntoExtensions = (
  extensions: unknown,
  moodImages: CharacterMoodImages
): Record<string, unknown> => {
  const parsed = parseCharacterExtensions(extensions)
  const tldw = isPlainObject(parsed.tldw) ? { ...parsed.tldw } : {}

  const normalizedMap: CharacterMoodImages = {}
  Object.entries(moodImages || {}).forEach(([rawMood, rawImage]) => {
    const moodLabel = normalizeCharacterEmoteState(rawMood)
    if (!moodLabel) return
    const normalizedImage = normalizeMoodImageSource(rawImage)
    if (!normalizedImage) return
    normalizedMap[moodLabel] = normalizedImage
  })

  if (Object.keys(normalizedMap).length > 0) {
    tldw.mood_images = normalizedMap
  } else {
    delete tldw.mood_images
  }
  delete tldw.moodImages

  if (Object.keys(tldw).length > 0) {
    parsed.tldw = tldw
  } else {
    delete parsed.tldw
  }

  delete parsed.mood_images
  delete parsed.moodImages

  return parsed
}

export const upsertCharacterMoodImage = (
  extensions: unknown,
  moodLabel: unknown,
  imageSource: unknown
): Record<string, unknown> => {
  const normalizedMood = normalizeCharacterMoodLabel(moodLabel)
  const normalizedImage = normalizeMoodImageSource(imageSource)
  const existing = getCharacterMoodImagesFromExtensions(extensions)

  if (!normalizedMood || !normalizedImage) {
    return mergeCharacterMoodImagesIntoExtensions(extensions, existing)
  }

  return mergeCharacterMoodImagesIntoExtensions(extensions, {
    ...existing,
    [normalizedMood]: normalizedImage
  })
}

export const removeCharacterMoodImage = (
  extensions: unknown,
  moodLabel: unknown
): Record<string, unknown> => {
  const normalizedMood = normalizeCharacterMoodLabel(moodLabel)
  if (!normalizedMood) {
    return parseCharacterExtensions(extensions)
  }

  const existing = getCharacterMoodImagesFromExtensions(extensions)
  const next = { ...existing }
  delete next[normalizedMood]
  return mergeCharacterMoodImagesIntoExtensions(extensions, next)
}

export const resolveCharacterBaseAvatarUrl = (
  character: CharacterWithPortraitData
): string => {
  if (!character) return ""

  if (typeof character.avatar_url === "string" && character.avatar_url.trim()) {
    return character.avatar_url.trim()
  }

  if (typeof character.image_base64 === "string" && character.image_base64.trim()) {
    return createImageDataUrl(character.image_base64.trim()) || ""
  }

  return ""
}

export const resolveCharacterMoodImageUrl = (
  character: CharacterWithPortraitData,
  moodLabel: unknown
): string => {
  if (!character) return ""

  const normalizedMood = normalizeCharacterEmoteState(moodLabel)
  const moodImages = getCharacterMoodImagesFromExtensions(character.extensions)
  if (normalizedMood && typeof moodImages[normalizedMood] === "string") {
    return moodImages[normalizedMood]
  }

  const legacyMood = normalizeCharacterMoodLabel(moodLabel)
  if (!legacyMood || legacyMood === normalizedMood) return ""

  const image = moodImages[legacyMood]
  return typeof image === "string" ? image : ""
}

export const detectCharacterMood = (params: {
  assistantText: string
  userText?: string | null
}): CharacterMoodDetection => {
  const assistantText = normalizeText(params.assistantText)
  const userText = normalizeText(params.userText)
  const combined = `${assistantText} ${userText}`.trim()

  if (!combined) {
    return {
      label: "neutral",
      confidence: 0.4,
      topic: null
    }
  }

  const scores: Record<CharacterMoodLabel, number> = {
    neutral: 0,
    happy: 0,
    excited: 0,
    sad: 0,
    angry: 0,
    thinking: 0,
    confused: 0,
    surprised: 0
  }

  ;(Object.keys(MOOD_PATTERNS) as Array<Exclude<CharacterMoodLabel, "neutral">>).forEach(
    (label) => {
      for (const pattern of MOOD_PATTERNS[label]) {
        scores[label] += countPatternHits(combined, pattern)
      }
    }
  )

  const questionMarks = (params.assistantText.match(/\?/g) || []).length
  if (questionMarks > 0) {
    scores.thinking += 0.35 * questionMarks
  }

  const exclamationMarks = (params.assistantText.match(/!/g) || []).length
  if (exclamationMarks > 1) {
    scores.excited += 0.2 * exclamationMarks
  }

  const ranked = (Object.entries(scores) as Array<[CharacterMoodLabel, number]>).sort(
    (a, b) => b[1] - a[1]
  )

  const [topLabel, topScore] = ranked[0]
  const secondScore = ranked[1]?.[1] ?? 0

  let label: CharacterMoodLabel = topLabel
  let confidence = clampConfidence(0.42 + topScore * 0.12 + (topScore - secondScore) * 0.05)

  if (topScore < 0.85) {
    label = "neutral"
    confidence = clampConfidence(0.5 - Math.max(0, topScore) * 0.06)
  }

  if (label === "neutral") {
    confidence = Math.min(confidence, 0.72)
  }

  return {
    label,
    confidence: Number(confidence.toFixed(2)),
    topic: extractTopic(params.assistantText, params.userText)
  }
}
