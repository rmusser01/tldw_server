const DISPLAY_NAME_PLACEHOLDER_REGEX =
  /\{\{\s*(user|username|user_name|you|me|char|character)\s*\}\}/gi

export const replaceUserDisplayNamePlaceholders = (
  content: string,
  userDisplayName: string,
  characterDisplayName = ""
): string => {
  const user = userDisplayName.trim()
  const character = characterDisplayName.trim()
  return content.replace(DISPLAY_NAME_PLACEHOLDER_REGEX, (placeholder, kind: string) =>
    (["char", "character"].includes(kind.toLowerCase()) ? character : user) || placeholder
  )
}
