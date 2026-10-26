import type { CSSProperties } from "react"

/** The name a reply carries when nothing better is known about who wrote it. */
export const GENERIC_ASSISTANT_NAME = "Assistant"

/**
 * The name shown in an assistant reply's header.
 *
 * A reply reloaded from the server in a chat without a character or persona
 * is named "Assistant". When the server recorded which model wrote it
 * (`metadata_extra.model_id`), name it after that model, as a freshly sent
 * reply is. A character or persona name always wins.
 */
export const resolveAssistantDisplayName = (
  name: string | null | undefined,
  savedModel: string | null | undefined
): string | undefined => {
  const trimmed = typeof name === "string" ? name.trim() : ""
  const model = typeof savedModel === "string" ? savedModel.trim() : ""
  if (model && (!trimmed || trimmed === GENERIC_ASSISTANT_NAME)) return model
  return name ?? undefined
}

const USER_TURN_TINT = "rgb(var(--color-primary) / 0.06)"

/**
 * A faint primary wash layered over the user turn's own surface fill, so the
 * user's messages read apart from replies. It is a background image, so the
 * fill's chat-message opacity (for background images) still applies.
 */
export const USER_TURN_TINT_STYLE: CSSProperties = {
  backgroundImage: `linear-gradient(${USER_TURN_TINT}, ${USER_TURN_TINT})`
}

/**
 * Width of the user's chat bubble. Casual mode caps it at three quarters of
 * the row on wider screens so it reads as the user's side of the
 * conversation; Pro keeps the denser, wider layout.
 */
export const userBubbleWidthClass = (isProMode: boolean): string =>
  isProMode
    ? "max-w-[calc(100%-1.75rem)]"
    : "max-w-[calc(100%-1.75rem)] sm:max-w-[75%]"
