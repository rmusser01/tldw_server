import type { HistoryTurnRecovery } from "@/db/dexie/types"

/**
 * Review copy for a retained native Continue (CM-01, #3106). Continue reuses
 * the saved question like Regenerate, but the user asked to continue a reply,
 * so the review says so instead of talking about a new reply.
 */
export function ContinuedReplyRecovery({
  turn,
  text
}: {
  turn: HistoryTurnRecovery
  text: (key: string, fallback: string) => string
}) {
  const [title, message] =
    turn.outcome === "complete" && turn.settled_message_id
      ? [
          text("savedContinuedReplyTitle", "Continued reply saved"),
          text(
            "savedContinuedReplyMessage",
            "The continued reply is saved as an alternative to the original reply, which is kept. Open the saved reply when you are ready."
          )
        ]
      : turn.state === "refused"
        ? [
            text("continuationRefusedTitle", "Continued reply refused"),
            text(
              "continuationRefusedMessage",
              "This saved chat uses unsupported context. Your original exchange and draft are kept. Review the saved setup before continuing again."
            )
          ]
        : [
            text("continuationUnknownTitle", "Continued reply unconfirmed"),
            text(
              "continuationUnknownMessage",
              "Your original exchange is kept. Any new text that arrived is retained here; the server may also have saved the continued reply. Reload this chat before continuing again. Recovery will not send it again automatically."
            )
          ]
  return (
    <>
      <p className="font-medium">{title}</p>
      <p>{message}</p>
    </>
  )
}
