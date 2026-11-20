import type { NotesTagMatch } from "./hooks/useNotesListManagement"

/**
 * The tag part of a notes search (NL-12 tag-match follow-up, #3101).
 *
 * The reader's any/all toggle applies to the tags the reader picked. Tags a
 * view implies (the Inbox's "captured", a saved filter's tags) always match:
 * "required AND (any of the picked)". A picked tag that is also required
 * meets the "any" group by itself. Required tags go out as `required_tokens`
 * only when "any" covers several picked tags; otherwise every tag must match
 * anyway, so they stay plain `tokens` with `tokens_match=all`, which older
 * servers understand too.
 */

const normalizeTokens = (tokens: string[]): string[] => {
  const out: string[] = []
  for (const token of tokens) {
    const normalized = String(token || "").trim().toLowerCase()
    if (normalized && !out.includes(normalized)) out.push(normalized)
  }
  return out
}

/** The any/all toggle shows for two or more picked tags, required ones included. */
export const tagMatchToggleApplies = (picked: string[]): boolean =>
  normalizeTokens(picked).length > 1

export const buildNotesTagSearchParams = (
  params: URLSearchParams,
  {
    picked,
    required,
    match
  }: { picked: string[]; required: string[]; match: NotesTagMatch }
): void => {
  const requiredTokens = normalizeTokens(required)
  const pickedTokens = normalizeTokens(picked)
  const anyOfSeveral = match === "any" && pickedTokens.length > 1
  const anyMetByRequired =
    anyOfSeveral && pickedTokens.some((token) => requiredTokens.includes(token))
  const tokens = anyMetByRequired
    ? requiredTokens
    : anyOfSeveral
      ? pickedTokens
      : normalizeTokens([...requiredTokens, ...pickedTokens])
  tokens.forEach((token) => params.append("tokens", token))
  const anyOverPicked = anyOfSeveral && !anyMetByRequired
  if (tokens.length > 1) params.set("tokens_match", anyOverPicked ? "any" : "all")
  if (anyOverPicked) requiredTokens.forEach((token) => params.append("required_tokens", token))
}
