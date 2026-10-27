import { act, cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { Character } from "@/types/character"
import { ChatGreetingPicker } from "../ChatGreetingPicker"
const state = vi.hoisted(() => ({ displayName: "Viewer", loading: false, unavailable: false, retry: vi.fn() }))
vi.mock("react-i18next", () => ({ useTranslation: () => ({ t: (_key: string, fallback: string, values?: { name?: string }) => values?.name ? fallback.replace("{{name}}", values.name) : fallback }) }))
vi.mock("@plasmohq/storage/hook", () => ({ useStorage: () => [state.displayName] }))
vi.mock("@/hooks/useCharacterGreeting", () => ({ useCharacterGreeting: (props: { selectedCharacter: Character | null; selectedCharacterMode: string | null; serverChatId: string | null; historyId: string | null; messagesLength: number; playgroundReady: boolean }) => ({
  character: props.selectedCharacter, loading: state.loading, unavailable: state.unavailable, retry: state.retry,
  fresh: props.playgroundReady && props.selectedCharacterMode === "tracked" && !!props.selectedCharacter?.id && !props.serverChatId && !props.historyId && props.messagesLength === 0
}) }))
vi.mock("antd", () => ({ Select: ({ value, onChange, options, disabled, ...props }: { value: string; onChange: (value: string) => void; options: { value: string; label: string; title?: string }[]; disabled: boolean; "aria-label"?: string }) => <select aria-label={props["aria-label"]} value={value} onChange={event => onChange(event.target.value)} disabled={disabled}>{options.map(option => <option key={option.value} value={option.value} title={option.title}>{option.label}</option>)}</select> }))
const card = { id: "card-1", name: "Guide", first_message: "{{user}} meets {{char}}", alternate_greetings: ["Alternate greeting"] } as Character
const props = () => ({ selectedCharacter: card, selectedCharacterMode: "tracked" as const, messages: [], historyId: null, serverChatId: null, ready: true, viewKey: "fresh-1", onStartGreeting: vi.fn().mockResolvedValue(undefined), onStartWithMessage: vi.fn() })
afterEach(cleanup)
beforeEach(() => { state.displayName = "Viewer"; state.loading = false; state.unavailable = false; state.retry.mockClear() })

describe("explicit character opening", () => {
  it("exposes explicit opening choices before committing a greeting", () => {
    const input = props(); render(<ChatGreetingPicker {...input} />)
    expect(screen.getByRole("button", { name: "Start with this greeting" })).toBeEnabled()
    fireEvent.click(screen.getByRole("button", { name: "Start with your message" }))
    expect(input.onStartWithMessage).toHaveBeenCalledOnce()
    expect(input.onStartGreeting).not.toHaveBeenCalled()
  })
  it.each(["Guide", ""])("renders distinct user and character identities (%s)", name => {
    render(<ChatGreetingPicker {...props()} selectedCharacter={{ ...card, name }} />)
    expect(screen.getByText(`Viewer meets ${name || "{{char}}"}`, { selector: "div" })).toBeInTheDocument()
  })
  it.each(["", " \t "])("renders the default user name in previews and alternate labels/titles (%j)", displayName => {
    state.displayName = displayName
    render(<ChatGreetingPicker {...props()} selectedCharacter={{ ...card, alternate_greetings: ["{{username}} greets {{character}}"] }} />)
    expect(screen.getByText("User meets Guide", { selector: "div" })).toBeInTheDocument()
    fireEvent.click(screen.getByText("Alternate greetings"))
    const alternate = screen.getByRole("option", { name: "User greets Guide" })
    expect(alternate).toHaveAttribute("title", "User greets Guide")
  })
  it("updates alternate display names without changing the chosen raw index, seed or option id", async () => {
    const raw = "  {{ you }} meets {{ CHARACTER }}\n  exact $&  "
    const input = { ...props(), selectedCharacter: { ...card, alternate_greetings: ["", raw] } }
    const view = render(<ChatGreetingPicker {...input} />)
    fireEvent.click(screen.getByText("Alternate greetings"))
    const alternate = screen.getByRole("option", { name: "Viewer meets Guide exact $&" }) as HTMLOptionElement
    const id = alternate.value
    fireEvent.change(screen.getByRole("combobox"), { target: { value: id } })
    state.displayName = "  Viewer $&  "
    view.rerender(<ChatGreetingPicker {...input} selectedCharacter={{ ...input.selectedCharacter, name: "Guide $'" }} />)
    const updated = screen.getByRole("option", { name: "Viewer $& meets Guide $' exact $&" })
    expect(updated).toHaveAttribute("title", "Viewer $& meets Guide $'\n  exact $&")
    expect(updated).toHaveValue(id)
    expect(screen.getByRole("combobox")).toHaveValue(id)
    await act(async () => fireEvent.click(screen.getByRole("button", { name: "Start with this greeting" })))
    expect(input.onStartGreeting).toHaveBeenCalledWith(expect.objectContaining({
      id, text: raw.trim(), seed: { greetingStrategy: "alternate_index", alternateIndex: 1, rawText: raw }
    }), expect.any(Function))
    expect(input.selectedCharacter.alternate_greetings[1]).toBe(raw)
  })
  it("alternate selection and reroll only change preview until explicit start", async () => {
    const input = props(); render(<ChatGreetingPicker {...input} />)
    fireEvent.click(screen.getByText("Alternate greetings"))
    const select = screen.getByRole("combobox", { name: "Choose greeting" })
    const option = screen.getByRole("option", { name: "Alternate greeting" }) as HTMLOptionElement
    fireEvent.change(select, { target: { value: option.value } })
    expect(input.onStartGreeting).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole("button", { name: "Reroll" }))
    expect(input.onStartGreeting).not.toHaveBeenCalled()
    await act(async () => fireEvent.click(screen.getByRole("button", { name: "Start with this greeting" })))
    expect(input.onStartGreeting).toHaveBeenCalledWith(expect.objectContaining({ seed: { greetingStrategy: "default", rawText: card.first_message } }), expect.any(Function))
  })
  it("passes the exact raw alternate index zero without a second random selection", async () => {
    const input = props(); render(<ChatGreetingPicker {...input} />)
    fireEvent.click(screen.getByText("Alternate greetings"))
    fireEvent.change(screen.getByRole("combobox"), { target: { value: (screen.getByRole("option", { name: "Alternate greeting" }) as HTMLOptionElement).value } })
    await act(async () => fireEvent.click(screen.getByRole("button", { name: "Start with this greeting" })))
    expect(input.onStartGreeting).toHaveBeenCalledWith(expect.objectContaining({ seed: { greetingStrategy: "alternate_index", alternateIndex: 0, rawText: "Alternate greeting" } }), expect.any(Function))
  })
  it("offers a first-message action when the card has no opening", () => {
    const input = props(); render(<ChatGreetingPicker {...input} selectedCharacter={{ id: "empty", name: "Guide" } as Character} />)
    fireEvent.click(screen.getByRole("button", { name: "Write the first message" }))
    expect(input.onStartWithMessage).toHaveBeenCalledOnce()
    expect(screen.queryByRole("button", { name: "Start with this greeting" })).not.toBeInTheDocument()
  })
  it("explains a display-only alias and never appends it to an existing chat", () => {
    const input = props(); render(<ChatGreetingPicker {...input} selectedCharacter={{ id: "alias", name: "Guide", greeting: "Alias opening" } as Character} />)
    expect(screen.getByRole("button", { name: "Start with this greeting" })).toBeDisabled()
    expect(screen.getByText(/cannot be saved with the server/)).toBeInTheDocument()
    expect(input.onStartGreeting).not.toHaveBeenCalled()
  })
  it.each([{ serverChatId: "saved" }, { historyId: "local-saved" }, { selectedCharacterMode: "overlay" as const }])("does not expose startup for existing/incompatible state %j", patch => {
    render(<ChatGreetingPicker {...props()} {...patch} />)
    expect(screen.queryByRole("button", { name: "Start with this greeting" })).not.toBeInTheDocument()
  })
  it.each(["loading", "unavailable"] as const)("keeps own-message choice while %s prevents an unverified opening", key => {
    state[key] = true; render(<ChatGreetingPicker {...props()} />)
    expect(screen.getByRole("button", { name: "Start with this greeting" })).toBeDisabled()
    expect(screen.getByRole("button", { name: "Start with your message" })).toBeEnabled()
    expect(screen.getByRole("status")).toBeInTheDocument()
  })
  it("bounds long preview after the primary choices", () => {
    render(<ChatGreetingPicker {...props()} selectedCharacter={{ ...card, first_message: "Long opening ".repeat(500) }} />)
    const preview = screen.getByText(/^Long opening/, { selector: "div" })
    expect(preview.className).toContain("max-h-48")
    expect(screen.getByRole("button", { name: "Start with this greeting" }).compareDocumentPosition(preview) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
  })
  it("blocks rapid repeat and the competing own-message action during startup", async () => {
    let finish!: () => void
    const input = props(); input.onStartGreeting.mockReturnValue(new Promise<void>(done => { finish = done }))
    render(<ChatGreetingPicker {...input} />)
    fireEvent.click(screen.getByRole("button", { name: "Start with this greeting" }))
    fireEvent.click(screen.getByRole("button", { name: "Saving opening…" }))
    expect(input.onStartGreeting).toHaveBeenCalledOnce()
    expect(screen.getByRole("button", { name: "Start with your message" })).toBeDisabled()
    await act(async () => finish())
  })
})

it("offers a working Retry without losing the own-message action", () => {
  state.unavailable = true
  render(<ChatGreetingPicker {...props()} />)
  fireEvent.click(screen.getByRole("button", { name: "Retry loading opening" }))
  expect(state.retry).toHaveBeenCalledOnce()
  expect(screen.getByRole("button", { name: "Start with your message" })).toBeEnabled()
})
it("explains unresolved creation and leaves own-message focus available", () => {
  render(<ChatGreetingPicker {...props()} blockedReason="Review the unconfirmed chat first." />)
  expect(screen.getByRole("button", { name: "Start with this greeting" })).toBeDisabled()
  expect(screen.getByText("Review the unconfirmed chat first.")).toBeInTheDocument()
  expect(screen.getByRole("button", { name: "Start with your message" })).toBeEnabled()
})
it("retires old busy state on New and an old completion cannot clear the newer operation", async () => {
  const releases: (() => void)[] = []
  const input = props()
  input.onStartGreeting.mockImplementation(() => new Promise<void>(done => releases.push(done)))
  const view = render(<ChatGreetingPicker {...input} />)
  fireEvent.click(screen.getByRole("button", { name: "Start with this greeting" }))
  view.rerender(<ChatGreetingPicker {...input} viewKey="fresh-2" />)
  expect(screen.getByRole("button", { name: "Start with this greeting" })).toBeEnabled()
  fireEvent.click(screen.getByRole("button", { name: "Start with this greeting" }))
  await act(async () => releases[0]())
  expect(screen.getByRole("button", { name: "Saving opening…" })).toBeDisabled()
  expect(input.onStartGreeting).toHaveBeenCalledTimes(2)
  await act(async () => releases[1]())
})

it("retires picker activity on an input revision without changing the preview or sending input", async () => {
  let finish!: () => void
  const input = props(); input.onStartGreeting.mockReturnValue(new Promise<void>(done => { finish = done }))
  const view = render(<ChatGreetingPicker {...input} inputRevision={0} />)
  fireEvent.click(screen.getByRole("button", { name: "Start with this greeting" }))
  view.rerender(<ChatGreetingPicker {...input} inputRevision={1} />)
  expect(screen.getByRole("button", { name: "Start with this greeting" })).toBeEnabled()
  expect(screen.getByRole("button", { name: "Start with your message" })).toBeEnabled()
  expect(screen.getByText("Viewer meets Guide", { selector: "div" })).toBeInTheDocument()
  await act(async () => finish())
})
