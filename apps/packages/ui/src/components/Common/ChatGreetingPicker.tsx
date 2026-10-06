import React from "react"
import { RefreshCcw } from "lucide-react"
import { Select } from "antd"
import { useStorage } from "@plasmohq/storage/hook"
import { useTranslation } from "react-i18next"
import type { Character } from "@/types/character"
import type { Message } from "@/store/option/types"
import type { AssistantSelectionMode } from "@/types/assistant-selection"
import type { ChatScope } from "@/types/chat-scope"
import { useCharacterGreeting } from "@/hooks/useCharacterGreeting"
import { buildGreetingOptionsFromEntries, collectGreetingEntries, type GreetingOption } from "@/utils/character-greetings"
import { replaceUserDisplayNamePlaceholders } from "@/utils/chat-display-name"

type Props = {
  selectedCharacter: Character | null
  selectedCharacterMode: AssistantSelectionMode | null
  messages: Message[]
  historyId: string | null
  serverChatId: string | null
  ready: boolean
  scope?: ChatScope
  viewKey?: string
  inputRevision?: number
  disabled?: boolean
  blockedReason?: string | null
  onStartGreeting?: (option: GreetingOption, isCurrent: () => boolean) => Promise<void>
  onStartWithMessage: () => void
  className?: string
}

/** The opening is a preview until a deliberate, owner-verified creation. */
export const ChatGreetingPicker: React.FC<Props> = (props) => {
  const { t } = useTranslation(["sidepanel", "common"])
  const [userDisplayName] = useStorage("chatUserDisplayName", "")
  const preview = useCharacterGreeting({ playgroundReady: props.ready,
    selectedCharacter: props.selectedCharacter, selectedCharacterMode: props.selectedCharacterMode,
    serverChatId: props.serverChatId, historyId: props.historyId, messagesLength: props.messages.length,
    scope: props.scope, viewKey: props.viewKey })
  const options = React.useMemo(() => buildGreetingOptionsFromEntries(collectGreetingEntries(preview.character)), [preview.character])
  const [choice, setChoice] = React.useState<{ characterId: string; id: string } | null>(null)
  const option = (choice?.characterId === String(preview.character?.id) && options.find(entry => entry.id === choice.id)) || options[0]
  const [activity, setActivity] = React.useState<{ current: () => boolean } | null>(null)
  const [error, setError] = React.useState<{ current: () => boolean; text: string } | null>(null)
  const active = React.useRef({ character: JSON.stringify(props.selectedCharacter), choice, viewKey: props.viewKey, inputRevision: props.inputRevision, mounted: true })
  active.current.character = JSON.stringify(props.selectedCharacter)
  active.current.choice = choice
  active.current.viewKey = props.viewKey
  active.current.inputRevision = props.inputRevision
  const starting = activity?.current() === true
  const inFlight = React.useRef<{ current: () => boolean } | null>(null)
  React.useEffect(() => {
    const view = active.current
    view.mounted = true
    return () => { view.mounted = false }
  }, [])
  const start = async () => {
    if (!option?.seed || !props.onStartGreeting || props.disabled || props.blockedReason || preview.loading || preview.unavailable || inFlight.current?.current()) return
    setError(null)
    const captured = { ...active.current }
    const current = () => active.current.mounted && active.current.character === captured.character &&
      active.current.choice === captured.choice && active.current.viewKey === captured.viewKey && active.current.inputRevision === captured.inputRevision
    const operation = { current }
    inFlight.current = operation
    setActivity(operation)
    try { await props.onStartGreeting(option, current) }
    catch (cause) {
      if (current()) setError({ current, text: cause instanceof Error ? cause.message : t("sidepanel:greetingPicker.startFailed", "Could not save the opening. Your draft is still here. Review the connection and retained drafts before starting again.") })
    } finally {
      if (inFlight.current === operation) {
        inFlight.current = null
        if (active.current.mounted) setActivity(null)
      }
    }
  }
  if (!preview.fresh) return null
  const busy = starting || props.disabled
  const text = option ? replaceUserDisplayNamePlaceholders(option.text, userDisplayName, preview.character?.name) : ""
  const name = preview.character?.name || t("sidepanel:greetingPicker.character", "AI character")
  return <section aria-label={t("sidepanel:greetingPicker.opening", "{{name}} opening", { name })}
    className={`w-full min-w-0 rounded-xl border border-border bg-surface p-3 text-sm text-text ${props.className || ""}`}>
    <h2 className="font-semibold">{name}</h2>
    <p className="mt-1 text-xs text-text-muted">{t("sidepanel:greetingPicker.previewHelp", "Choose who begins. Previewing an opening does not add a message.")}</p>
    <div className="mt-3 flex flex-wrap gap-2">
      {option && <button type="button" disabled={busy || Boolean(props.blockedReason) || preview.loading || preview.unavailable || !option.seed || !props.onStartGreeting}
        onClick={() => void start()} className="min-h-9 rounded-md bg-primary px-3 py-2 text-sm text-primary-foreground focus-visible:outline focus-visible:outline-2 disabled:opacity-60">
        {starting ? t("sidepanel:greetingPicker.starting", "Saving opening…") : t("sidepanel:greetingPicker.start", "Start with this greeting")}
      </button>}
      <button type="button" disabled={busy} onClick={props.onStartWithMessage}
        className="min-h-9 rounded-md border border-border px-3 py-2 text-sm focus-visible:outline focus-visible:outline-2 disabled:opacity-60">
        {option ? t("sidepanel:greetingPicker.ownMessage", "Start with your message") : t("sidepanel:greetingPicker.firstMessage", "Write the first message")}
      </button>
    </div>
    {props.blockedReason && <p role="status" className="mt-2 text-xs text-text-muted">{props.blockedReason}</p>}
    {preview.loading && <p role="status" className="mt-2 text-xs text-text-muted">{t("sidepanel:greetingPicker.loading", "Loading character opening…")}</p>}
    {preview.unavailable && <div role="status" className="mt-2 text-xs text-text-muted">
      <p>{t("sidepanel:greetingPicker.unavailable", "Could not refresh this card. Review the connection before saving an opening.")}</p>
      <button type="button" disabled={busy} onClick={() => { setError(null); preview.retry() }} className="mt-2 min-h-9 rounded-md border border-border px-3 py-2 focus-visible:outline focus-visible:outline-2">{t("sidepanel:greetingPicker.retry", "Retry loading opening")}</button>
    </div>}
    {option && !option.seed && <p className="mt-2 text-xs text-text-muted">{t("sidepanel:greetingPicker.unsupported", "This opening cannot be saved with the server’s greeting options. Write your message or review the character card.")}</p>}
    {text && <div className="mt-3 max-h-48 overflow-auto whitespace-pre-wrap break-words rounded-lg bg-surface2 p-3 text-sm">{text}</div>}
    {options.length > 1 && <details className="mt-3 text-xs">
      <summary className="cursor-pointer">{t("sidepanel:greetingPicker.alternates", "Alternate greetings")}</summary>
      <div className="mt-2 flex min-w-0 items-center gap-2">
        <Select aria-label={t("sidepanel:greetingPicker.choose", "Choose greeting")} value={option?.id} disabled={busy}
          className="min-w-0 flex-1" options={options.map(entry => ({ value: entry.id, label: entry.text, title: entry.text }))}
          onChange={(id: string) => { setChoice({ characterId: String(preview.character?.id), id }); setError(null) }} />
        <button type="button" disabled={busy} className="min-h-9 rounded-md border border-border px-2 focus-visible:outline focus-visible:outline-2"
          aria-label={t("sidepanel:greetingPicker.reroll", "Reroll")}
          onClick={() => { const rest = options.filter(entry => entry !== option); const next = rest[Math.floor(Math.random() * rest.length)]; setChoice({ characterId: String(preview.character?.id), id: next.id }); setError(null) }}><RefreshCcw className="h-4 w-4" /></button>
      </div>
    </details>}
    {starting && <p role="status" className="mt-2 text-xs">{t("sidepanel:greetingPicker.startingHelp", "Saving the character’s opening. Your message has not been sent.")}</p>}
    {error?.current() && <p role="alert" className="mt-2 text-xs text-danger">{error.text}</p>}
  </section>
}
