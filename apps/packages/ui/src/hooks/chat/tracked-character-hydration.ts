type TrackedCharacterCandidate = {
  id?: string | number | null
  name?: string | null
}

type CharacterLoader = (
  id: string | number
) => Promise<Record<string, unknown> | null | undefined>

const hasAuthoritativeCharacterName = (
  character: TrackedCharacterCandidate
): boolean => {
  const name = typeof character.name === "string" ? character.name.trim() : ""
  return Boolean(name) && name.toLowerCase() !== "assistant"
}

export const hydrateTrackedCharacterForSend = async <
  T extends TrackedCharacterCandidate
>(candidate: T, loadCharacter: CharacterLoader): Promise<T> => {
  if (candidate.id == null || hasAuthoritativeCharacterName(candidate)) {
    return candidate
  }

  try {
    const loaded = await loadCharacter(candidate.id)
    if (!loaded || typeof loaded !== "object") return candidate
    return {
      ...candidate,
      ...loaded,
      id: candidate.id
    } as T
  } catch {
    return candidate
  }
}
