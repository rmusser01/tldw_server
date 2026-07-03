import { normalizeVisualIdentityExpressionKey } from "./visual-identity-expressions"

export interface VisualIdentityEmoteCommand {
  expressionKey: string
  rawExpression: string
}

const EMOTE_COMMAND_PATTERN = /^\/emote(?:\s+(.+))?$/i

export const parseVisualIdentityEmoteCommand = (
  input: string
): VisualIdentityEmoteCommand | null => {
  const match = input.trim().match(EMOTE_COMMAND_PATTERN)
  if (!match) return null

  const rawExpression = (match[1] || "").trim()
  if (!rawExpression) return null

  const expressionKey = normalizeVisualIdentityExpressionKey(rawExpression)
  if (!expressionKey) return null

  return {
    expressionKey,
    rawExpression
  }
}
