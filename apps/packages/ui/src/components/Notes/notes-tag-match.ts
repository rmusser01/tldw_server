import type { NotesTagMatch } from "./hooks/useNotesListManagement"

/**
 * The tag part of a notes search (NL-12 tag-match follow-up, #3101).
 *
 * The reader's any/all toggle applies to the tags the reader picked. Tags a
 * view implies (the Inbox's "captured") always match: "captured AND (a OR b)".
 * Those go out as `required_tokens` only when "any" covers several picked
 * tags; otherwise every tag must match anyway, so they stay plain `tokens`
 * with `tokens_match=all`, which older servers understand too.
 */

const normalizeTokens = (tokens: string[]): string[] => {
  const out: string[] = []
  for (const token of tokens) {
    const normalized = String(token || "").trim().toLowerCase()
    if (normalized && !out.includes(normalized)) out.push(normalized)
  }
  return out
}

/** The picked tags the any/all toggle applies to: not the ones a view already requires. */
export const pickedTagMatchTokens = (picked: string[], required: string[]): string[] => {
  const requiredTokens = normalizeTokens(required)
  return normalizeTokens(picked).filter((token) => !requiredTokens.includes(token))
}

export const buildNotesTagSearchParams = (
  params: URLSearchParams,
  {
    picked,
    required,
    match
  }: { picked: string[]; required: string[]; match: NotesTagMatch }
): void => {
  const requiredTokens = normalizeTokens(required)
  const pickedTokens = pickedTagMatchTokens(picked, requiredTokens)
  const anyOfSeveral = match === "any" && pickedTokens.length > 1
  const tokens = anyOfSeveral ? pickedTokens : [...requiredTokens, ...pickedTokens]
  tokens.forEach((token) => params.append("tokens", token))
  if (tokens.length > 1) params.set("tokens_match", anyOfSeveral ? "any" : "all")
  if (anyOfSeveral) requiredTokens.forEach((token) => params.append("required_tokens", token))
}
