import type { FlashcardCreate, FlashcardTemplate } from "@/services/flashcards"

const PLACEHOLDER_TOKEN_RE = /\{\{\s*([^\s{}]+)\s*\}\}/g
const CLOZE_TOKEN_SEPARATOR = "::"

type FlashcardTemplateDraftDefaults = Pick<FlashcardCreate, "deck_id" | "tags">

const normalizePlaceholderValue = (value: string | null | undefined): string => value?.trim() ?? ""
const isClozeToken = (token: string): boolean => token.includes(CLOZE_TOKEN_SEPARATOR)

export function getFlashcardTemplatePlaceholderDefaults(
  template: FlashcardTemplate
): Record<string, string> {
  return Object.fromEntries(
    template.placeholder_definitions.map((definition) => [
      definition.key,
      definition.default_value ?? ""
    ])
  )
}

export function materializeFlashcardTemplateDraft(
  template: FlashcardTemplate,
  values: Record<string, string | null | undefined>,
  defaults?: Partial<FlashcardTemplateDraftDefaults>
): Pick<FlashcardCreate, "deck_id" | "tags" | "model_type" | "front" | "back" | "notes" | "extra"> {
  const resolvedValues = Object.fromEntries(
    template.placeholder_definitions.map((definition) => {
      const explicitValue = normalizePlaceholderValue(values[definition.key])
      const defaultValue = normalizePlaceholderValue(definition.default_value)
      const finalValue = explicitValue || defaultValue

      if (definition.required && finalValue.length === 0) {
        throw new Error(`Missing required placeholder value: ${definition.key}`)
      }

      return [definition.key, finalValue]
    })
  )

  const resolveTemplateText = (text: string | null | undefined): string | null => {
    if (text == null) {
      return null
    }
    return text.replace(PLACEHOLDER_TOKEN_RE, (match, token: string) => {
      if (isClozeToken(token)) {
        return match
      }
      if (!(token in resolvedValues)) {
        throw new Error(`Unknown placeholder token: ${token}`)
      }
      return resolvedValues[token] ?? ""
    })
  }

  return {
    deck_id: defaults?.deck_id ?? undefined,
    tags: defaults?.tags ? [...defaults.tags] : undefined,
    model_type: template.model_type,
    front: resolveTemplateText(template.front_template) ?? "",
    back: resolveTemplateText(template.back_template) ?? "",
    notes: resolveTemplateText(template.notes_template),
    extra: resolveTemplateText(template.extra_template)
  }
}
