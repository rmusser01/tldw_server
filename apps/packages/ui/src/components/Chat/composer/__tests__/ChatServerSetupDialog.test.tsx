import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { App, ConfigProvider } from "antd"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { ChatServerSetupDialog } from "../ChatServerSetupDialog"
import { useSimpleForm } from "@/hooks/useSimpleForm"

const leaf = vi.hoisted(() => ({ mounted: vi.fn(), unmounted: vi.fn() }))
vi.mock("@/components/Option/Settings/tldw", () => ({
  TldwSettings: ({ onDirtyChange, onOpenDiagnostics, onSavingChange }: { onDirtyChange?: (value: boolean) => void; onOpenDiagnostics?: () => void; onSavingChange?: (value: boolean) => void }) => {
    React.useEffect(() => { leaf.mounted(); return () => { leaf.unmounted() } }, [])
    return <><input aria-label="Setup Server URL" onChange={() => onDirtyChange?.(true)} />
      <button onClick={onOpenDiagnostics}>Health</button><button onClick={() => onSavingChange?.(true)}>Hold Save</button><button onClick={() => onSavingChange?.(false)}>Finish Save</button></>
  }
}))
vi.mock("react-i18next", () => ({ useTranslation: () => ({ t: (_: string, fallback: string) => fallback }) }))

const Harness = ({ removeOpenerOnOpen = false } = {}) => {
  const [open, setOpen] = React.useState(false)
  const form = useSimpleForm({ initialValues: { message: "FIRST IDEA\nSECOND LINE", image: "ORIGINAL IMAGE" } })
  return <ConfigProvider theme={{ token: { motion: false } }}><App><main id="main-content" tabIndex={-1}>
    {(!open || !removeOpenerOnOpen) && <button onClick={() => setOpen(true)}>Open server setup</button>}
    <textarea aria-label="Source message" {...form.getInputProps("message")} />
    <input aria-label="Source image" {...form.getInputProps("image")} />
    <ChatServerSetupDialog open={open} onClose={() => setOpen(false)} />
  </main></App></ConfigProvider>
}
// AntD uses one fixed aria ID in test mode. Its development behavior exercises real stacked-dialog labels.
beforeEach(() => { vi.stubEnv("NODE_ENV", "development") })
afterEach(() => { vi.restoreAllMocks(); vi.unstubAllEnvs() })

describe("existing composer lifetime through setup dismissal", () => {
  it("keeps the same local form and exact input when returning to Chat", async () => {
    const user = userEvent.setup()
    render(<Harness />)
    const text = screen.getByRole("textbox", { name: "Source message" })
    const image = screen.getByRole("textbox", { name: "Source image" })
    const opener = screen.getByRole("button", { name: "Open server setup" })
    expect(screen.queryByRole("textbox", { name: "Setup Server URL" })).not.toBeInTheDocument()
    await user.click(opener)
    expect(screen.getByRole("dialog", { name: "Server settings" })).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Return to chat" }))
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    expect(screen.getByRole("textbox", { name: "Source message" })).toBe(text)
    expect(text).toHaveValue("FIRST IDEA\nSECOND LINE")
    expect(image).toHaveValue("ORIGINAL IMAGE")
    await waitFor(() => expect(opener).toHaveFocus())
  })
  it("keeps edited settings when dirty dismissal is declined", async () => {
    const user = userEvent.setup()
    render(<Harness />)
    await user.click(screen.getByRole("button", { name: "Open server setup" }))
    const settings = screen.getByRole("textbox", { name: "Setup Server URL" })
    fireEvent.change(settings, { target: { value: "https://unsaved.example.test" } })
    await user.click(screen.getByRole("button", { name: "Return to chat" }))
    expect(screen.getByRole("dialog", { name: "Discard connection changes?" })).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Keep editing" }))
    expect(screen.getByRole("dialog", { name: "Server settings" })).toBeInTheDocument()
    expect(settings).toHaveValue("https://unsaved.example.test")
    await user.click(screen.getByRole("button", { name: "Return to chat" }))
    await user.click(screen.getByRole("button", { name: "Discard changes" }))
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    expect(screen.getByRole("textbox", { name: "Source message" })).toHaveValue("FIRST IDEA\nSECOND LINE")
  })
  it("returns focus to the Chat main region when the initiating control disappears", async () => {
    const user = userEvent.setup()
    render(<Harness removeOpenerOnOpen />)
    await user.click(screen.getByRole("button", { name: "Open server setup" }))
    await user.click(screen.getByRole("button", { name: "Return to chat" }))
    await waitFor(() => expect(screen.getByRole("main")).toHaveFocus())
    expect(screen.getByRole("textbox", { name: "Source message" })).toHaveValue("FIRST IDEA\nSECOND LINE")
  })
  it("keeps the dialog open until an in-progress Save finishes", async () => {
    const user = userEvent.setup()
    render(<Harness />)
    await user.click(screen.getByRole("button", { name: "Open server setup" }))
    await user.click(screen.getByRole("button", { name: "Hold Save" }))
    expect(screen.getByRole("button", { name: "Return to chat" })).toBeDisabled()
    expect(screen.queryByRole("button", { name: "Close" })).not.toBeInTheDocument()
    await user.keyboard("{Escape}")
    expect(screen.getByRole("dialog", { name: "Server settings" })).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Finish Save" }))
    await user.click(screen.getByRole("button", { name: "Return to chat" }))
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    expect(screen.getByRole("textbox", { name: "Source message" })).toHaveValue("FIRST IDEA\nSECOND LINE")
  })
  it("opens diagnostics separately without removing the composer", async () => {
    const user = userEvent.setup()
    const open = vi.spyOn(window, "open").mockReturnValue(null)
    render(<Harness />)
    await user.click(screen.getByRole("button", { name: "Open server setup" }))
    await user.click(screen.getByRole("button", { name: "Health" }))
    await act(async () => {})
    expect(open).toHaveBeenCalledWith(expect.stringContaining("/settings/health"), "_blank", "noopener")
    expect(screen.getByRole("textbox", { name: "Source message" })).toHaveValue("FIRST IDEA\nSECOND LINE")
    expect(screen.getByRole("dialog", { name: "Server settings" })).toBeInTheDocument()
  })
})
