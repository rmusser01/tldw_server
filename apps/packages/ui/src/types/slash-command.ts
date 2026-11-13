/** One entry in a composer's slash-command menu. */
export type SlashCommandItem = {
  id: string
  command: string
  label: string
  description?: string
  keywords?: string[]
  /** Runs the command; a standalone command receives the text typed after it. */
  action?: (argument?: string) => void
  insertText?: string
  /** Where the command came from: the composer, or `/api/v1/chat/commands`. */
  source?: "server" | "local"
  /**
   * Runs on its own and never sends: the chat's saved draft is cleared first,
   * and the text after the command stays in the composer (`/new`) or goes to
   * the command (`consumesArgument`, `/prompt`).
   */
  standalone?: boolean
  /** A standalone command that takes the text after it; the composer is left empty. */
  consumesArgument?: boolean
}
