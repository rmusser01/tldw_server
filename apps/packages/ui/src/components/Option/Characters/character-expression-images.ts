import {
  createAvatarValue,
  extractAvatarValues,
  type AvatarFieldValue
} from "./AvatarField"
import {
  getCharacterMoodImagesFromExtensions,
  type CharacterMoodImages
} from "@/utils/character-mood"
import { normalizeCharacterEmoteState } from "@/utils/character-emotes"
import { createImageDataUrl } from "@/utils/image-utils"

export const EXPRESSION_IMAGE_STARTER_STATES = [
  "neutral",
  "happy",
  "sad",
  "angry",
  "thinking",
  "surprised"
] as const
const EXPRESSION_IMAGE_STARTER_STATE_SET = new Set<string>(
  EXPRESSION_IMAGE_STARTER_STATES
)

export type ExpressionImageRow = {
  id: string
  state: string
  image: AvatarFieldValue
  starter: boolean
}

export type ExpressionImageRowErrorReason =
  | "invalid-state"
  | "invalid-image"
  | "duplicate"
  | "missing-state"
  | "missing-image"

export type ExpressionImageRowError = {
  id: string
  reason: ExpressionImageRowErrorReason
}

let customExpressionRowId = 0

const hasExpressionImage = (image: AvatarFieldValue): boolean => {
  const values = extractAvatarValues(image)
  return Boolean(values.avatar_url || values.image_base64)
}

const expressionImageSource = (image: AvatarFieldValue): string | null => {
  const values = extractAvatarValues(image)
  const source = values.avatar_url || values.image_base64 || null
  if (typeof source !== "string") return null
  const trimmed = source.trim()
  if (!trimmed) return null
  if (trimmed.startsWith("data:image/")) {
    return createImageDataUrl(trimmed)
  }
  if (trimmed.startsWith("http://") || trimmed.startsWith("https://")) {
    return trimmed
  }
  return createImageDataUrl(trimmed)
}

export const expressionRowsFromExtensions = (
  extensions: unknown
): ExpressionImageRow[] => {
  const moodImages = getCharacterMoodImagesFromExtensions(extensions)
  const starterStates = new Set<string>(EXPRESSION_IMAGE_STARTER_STATES)
  const rows: ExpressionImageRow[] = EXPRESSION_IMAGE_STARTER_STATES.map((state) => ({
    id: state,
    state,
    image: createAvatarValue(moodImages[state]),
    starter: true
  }))

  Object.entries(moodImages).forEach(([state, image]) => {
    if (starterStates.has(state)) return
    rows.push({
      id: state,
      state,
      image: createAvatarValue(image),
      starter: false
    })
  })

  return rows
}

export const createEmptyCustomExpressionRow = (): ExpressionImageRow => ({
  id: `custom-expression-${customExpressionRowId++}`,
  state: "",
  image: createAvatarValue(),
  starter: false
})

export const normalizeExpressionImageRows = (
  rows: ExpressionImageRow[]
): { rows: ExpressionImageRow[]; errors: ExpressionImageRowError[] } => {
  const normalizedRows: ExpressionImageRow[] = []
  const errors: ExpressionImageRowError[] = []
  const seenStates = new Set<string>()

  rows.forEach((row) => {
    const rawState = row.state.trim()
    const normalizedState = normalizeCharacterEmoteState(row.state)
    const hasImage = hasExpressionImage(row.image)
    const imageSource = expressionImageSource(row.image)
    const isStarterSuggestion =
      row.starter &&
      (!rawState ||
        (normalizedState !== null &&
          EXPRESSION_IMAGE_STARTER_STATE_SET.has(normalizedState)))

    if (!rawState) {
      errors.push({ id: row.id, reason: "missing-state" })
    } else if (!normalizedState) {
      errors.push({ id: row.id, reason: "invalid-state" })
    }

    if (hasImage && !imageSource) {
      errors.push({ id: row.id, reason: "invalid-image" })
    }

    if (!hasImage && !isStarterSuggestion) {
      errors.push({ id: row.id, reason: "missing-image" })
    }

    if (!normalizedState || !imageSource) return

    if (seenStates.has(normalizedState)) {
      errors.push({ id: row.id, reason: "duplicate" })
      return
    }

    seenStates.add(normalizedState)

    if (hasImage) {
      normalizedRows.push({
        ...row,
        state: normalizedState
      })
    }
  })

  return { rows: normalizedRows, errors }
}

export const expressionRowsToMoodImages = (
  rows: ExpressionImageRow[]
): CharacterMoodImages => {
  const moodImages: CharacterMoodImages = {}

  rows.forEach((row) => {
    const state = normalizeCharacterEmoteState(row.state)
    const image = expressionImageSource(row.image)
    if (state && image) {
      moodImages[state] = image
    }
  })

  return moodImages
}
