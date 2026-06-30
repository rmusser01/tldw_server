export type BuddyBuilderTranslate = (
  key: string,
  options?: Record<string, unknown>
) => string

export const NATIVE_PERSONA_VISUAL_PACK_EXTENSION = ".tldw-persona-vpack"
export const CODEX_PET_ARCHIVE_EXTENSION = ".zip"

export const BUDDY_IMPORT_ARCHIVE_EXTENSIONS = [
  NATIVE_PERSONA_VISUAL_PACK_EXTENSION,
  CODEX_PET_ARCHIVE_EXTENSION
] as const

export const BUDDY_IMPORT_ARCHIVE_MIME_TYPES = new Set([
  "application/octet-stream",
  "application/vnd.tldw.persona.visual-pack+zip",
  "application/x-zip-compressed",
  "application/zip"
])

export const BUDDY_IMPORT_ARCHIVE_ACCEPT = [
  ...BUDDY_IMPORT_ARCHIVE_EXTENSIONS,
  ...BUDDY_IMPORT_ARCHIVE_MIME_TYPES
].join(",")

export const hasBuddyImportArchiveExtension = (file: File | null): boolean => {
  if (!file) return true
  const fileName = file.name.toLowerCase()
  return BUDDY_IMPORT_ARCHIVE_EXTENSIONS.some((extension) =>
    fileName.endsWith(extension)
  )
}

export const hasBuddyImportArchiveMediaType = (file: File | null): boolean => {
  if (!file) return true
  const mediaType = file.type.trim().toLowerCase()
  return !mediaType || BUDDY_IMPORT_ARCHIVE_MIME_TYPES.has(mediaType)
}

export const isBuddyImportArchiveFile = (file: File | null): boolean =>
  hasBuddyImportArchiveExtension(file) && hasBuddyImportArchiveMediaType(file)

export const getBuddyImportArchiveFileError = (
  file: File | null,
  t: BuddyBuilderTranslate
): string | null => {
  if (isBuddyImportArchiveFile(file)) return null
  if (!hasBuddyImportArchiveExtension(file)) {
    return t("sidepanel:personaGarden.visuals.builder.importUnsupportedExtension", {
      defaultValue:
        "Choose a .tldw-persona-vpack or Codex/Petdex .zip archive."
    })
  }
  return t("sidepanel:personaGarden.visuals.builder.importUnsupportedMimeType", {
    defaultValue:
      "Choose a Persona Visual or Codex/Petdex archive with a supported zip media type."
  })
}
