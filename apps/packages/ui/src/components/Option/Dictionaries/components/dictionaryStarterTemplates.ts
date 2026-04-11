export type DictionaryStarterTemplateId =
  | "medical_abbreviations"
  | "chat_speak_translator"
  | "custom_terminology"
  | "character_speech_patterns"

type DictionaryStarterEntry = {
  pattern: string
  replacement: string
  type: "literal" | "regex"
  group?: string
  enabled?: boolean
  case_sensitive?: boolean
}

export type DictionaryStarterTemplate = {
  id: DictionaryStarterTemplateId
  label: string
  description: string
  entries: DictionaryStarterEntry[]
}

export const DICTIONARY_STARTER_TEMPLATES: DictionaryStarterTemplate[] = [
  {
    id: "medical_abbreviations",
    label: "Medical Abbreviations",
    description: "Expand common clinical shorthand into patient-friendly terms.",
    entries: [
      {
        pattern: "BP",
        replacement: "blood pressure",
        type: "literal",
        group: "medical",
        enabled: true,
        case_sensitive: false,
      },
      {
        pattern: "HR",
        replacement: "heart rate",
        type: "literal",
        group: "medical",
        enabled: true,
        case_sensitive: false,
      },
      {
        pattern: "PRN",
        replacement: "as needed",
        type: "literal",
        group: "medical",
        enabled: true,
        case_sensitive: false,
      },
    ],
  },
  {
    id: "chat_speak_translator",
    label: "Chat Speak Translator",
    description: "Convert informal shorthand into clearer written language.",
    entries: [
      {
        pattern: "brb",
        replacement: "be right back",
        type: "literal",
        group: "chat_speak",
        enabled: true,
        case_sensitive: false,
      },
      {
        pattern: "idk",
        replacement: "I do not know",
        type: "literal",
        group: "chat_speak",
        enabled: true,
        case_sensitive: false,
      },
      {
        pattern: "imo",
        replacement: "in my opinion",
        type: "literal",
        group: "chat_speak",
        enabled: true,
        case_sensitive: false,
      },
    ],
  },
  {
    id: "custom_terminology",
    label: "Custom Terminology",
    description: "Seed a glossary for common business and project abbreviations.",
    entries: [
      {
        pattern: "KPI",
        replacement: "key performance indicator",
        type: "literal",
        group: "terminology",
        enabled: true,
        case_sensitive: false,
      },
      {
        pattern: "SLA",
        replacement: "service level agreement",
        type: "literal",
        group: "terminology",
        enabled: true,
        case_sensitive: false,
      },
      {
        pattern: "ETA",
        replacement: "estimated time of arrival",
        type: "literal",
        group: "terminology",
        enabled: true,
        case_sensitive: false,
      },
    ],
  },
  {
    id: "character_speech_patterns",
    label: "Character Speech Patterns",
    description: "Add accent, dialect, or speech quirks for roleplay characters.",
    entries: [
      {
        pattern: "you",
        replacement: "ye",
        type: "literal",
        group: "dialect",
        enabled: true,
        case_sensitive: false,
      },
      {
        pattern: "the",
        replacement: "th'",
        type: "literal",
        group: "dialect",
        enabled: true,
        case_sensitive: false,
      },
      {
        pattern: "going to",
        replacement: "gonna",
        type: "literal",
        group: "slang",
        enabled: true,
        case_sensitive: false,
      },
    ],
  },
]

export function getDictionaryStarterTemplate(
  templateId: unknown
): DictionaryStarterTemplate | null {
  if (typeof templateId !== "string") return null
  return (
    DICTIONARY_STARTER_TEMPLATES.find((template) => template.id === templateId) ??
    null
  )
}

