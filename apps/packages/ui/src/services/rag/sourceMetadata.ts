import type { RagSource } from "./unified-rag"

type RagSourceMetadata = {
  label: string
  description: string
  translationKey: string
}

export const ALL_RAG_SOURCES: RagSource[] = [
  "media_db",
  "notes",
  "characters",
  "chats",
  "kanban",
]

const RAG_SOURCE_METADATA: Record<RagSource, RagSourceMetadata> = {
  media_db: {
    label: "Documents & Media",
    description: "Uploaded files, transcripts, and web pages",
    translationKey: "sidepanel:rag.sources.media",
  },
  notes: {
    label: "Notes",
    description: "Your personal notes and clips",
    translationKey: "sidepanel:rag.sources.notes",
  },
  characters: {
    label: "Story Characters",
    description: "Character cards and persona definitions",
    translationKey: "sidepanel:rag.sources.characters",
  },
  chats: {
    label: "Conversations",
    description: "Previous chat conversations",
    translationKey: "sidepanel:rag.sources.chats",
  },
  kanban: {
    label: "Task Boards",
    description: "Kanban board items and tasks",
    translationKey: "sidepanel:rag.sources.kanban",
  },
}

const RAG_SOURCE_VALUES = new Set<RagSource>(ALL_RAG_SOURCES)

export function isRagSource(value: unknown): value is RagSource {
  return typeof value === "string" && RAG_SOURCE_VALUES.has(value as RagSource)
}

export function getRagSourceLabel(source: RagSource): string {
  return RAG_SOURCE_METADATA[source].label
}

export function getRagSourceDescription(source: RagSource): string {
  return RAG_SOURCE_METADATA[source].description
}

export function getRagSourceTranslationKey(source: RagSource): string {
  return RAG_SOURCE_METADATA[source].translationKey
}

export function getRagSourceOptions(
  translate?: (key: string, fallback: string) => string
): Array<{ value: RagSource; label: string; translationKey: string }> {
  return ALL_RAG_SOURCES.map((source) => {
    const metadata = RAG_SOURCE_METADATA[source]
    return {
      value: source,
      label: translate
        ? translate(metadata.translationKey, metadata.label)
        : metadata.label,
      translationKey: metadata.translationKey,
    }
  })
}
