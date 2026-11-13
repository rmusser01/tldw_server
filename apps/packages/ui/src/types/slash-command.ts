/** One entry in a composer's slash-command menu. */
export type SlashCommandItem = {
  id: string
  command: string
  label: string
  description?: string
  keywords?: string[]
  action?: () => void
  insertText?: string
  /** Where the command came from: the composer, or `/api/v1/chat/commands`. */
  source?: "server" | "local"
  /** Runs on its own: a submitted draft's text after it is not sent. */
  standalone?: boolean
}
