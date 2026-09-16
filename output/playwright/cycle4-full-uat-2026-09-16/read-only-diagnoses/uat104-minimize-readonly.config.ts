import base from "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/vitest.config"
const ui = "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui"
const target = ui + "/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx"
const probe = `
  it.each(["processing-button", "close-confirmation"])("UAT104 real minimize boundary: %s", async (via) => {
    useQuickIngestSessionStore.getState().createDraftSession({
      ...createEmptyQuickIngestSession(), lifecycle: "processing", currentStep: 4,
      queueItems: [{ id: "uat104", kind: "url", url: "https://example.com/uat104", detectedType: "web", icon: "Globe", fileSize: 0, validation: { valid: true } } as any],
      processingState: { status: "running", perItemProgress: [{ id: "uat104", status: "processing", progress: 0 }], elapsed: 1, estimatedRemaining: 0 },
      tracking: { mode: "extension-runtime", sessionId: "uat104-runtime", itemIds: ["uat104"], startedAt: Date.now() } as any,
    })
    const originalId = useQuickIngestSessionStore.getState().session!.id
    const view = render(<SessionBackedQuickIngestModal />)
    try {
      expect(await screen.findByRole("button", { name: "Minimize to Background" })).toBeInTheDocument()
      if (via === "processing-button") {
        await userEvent.click(screen.getByRole("button", { name: "Minimize to Background" }))
      } else {
        await userEvent.click(screen.getByRole("button", { name: "Close", exact: true }))
        const { Modal } = await import("antd")
        const options = vi.mocked(Modal.confirm).mock.calls.at(-1)?.[0]
        await act(async () => { await options?.onOk?.() })
      }
      await act(async () => { await Promise.resolve() })
      const observed = { visibility: useQuickIngestSessionStore.getState().session?.visibility, dialogs: screen.queryAllByRole("dialog").length, lifecycle: useQuickIngestSessionStore.getState().session?.lifecycle, lastModalOpen: mocks.modalProps.at(-1)?.open }
      console.log("UAT104_BOUNDARY", via, JSON.stringify(observed))
      expect(observed.visibility).toBe("hidden")
      expect(observed.dialogs).toBe(0)
      expect(useQuickIngestSessionStore.getState().session?.id).toBe(originalId)
      expect(mocks.cancelQuickIngestSession).not.toHaveBeenCalled()
      expect(mocks.startQuickIngestSession).not.toHaveBeenCalled()
      await act(async () => { useQuickIngestSessionStore.getState().showSession() })
      expect(await screen.findByRole("button", { name: "Minimize to Background" })).toBeInTheDocument()
      expect(useQuickIngestSessionStore.getState().session?.id).toBe(originalId)
    } finally { view.unmount() }
  })
`
export default { ...base, plugins: [{ name: "uat104-readonly-real-processing", enforce: "pre" as const, transform(code: string, id: string) {
  if (id !== target) return
  const start = code.indexOf('vi.mock("@/components/Common/QuickIngest/ProcessingStep",')
  const end = code.indexOf('vi.mock("@/components/Common/QuickIngest/WizardResultsStep",', start)
  if (start < 0 || end < 0) throw new Error("Existing ProcessingStep mock not found")
  code = code.slice(0,start) + code.slice(end)
  code = code.replace(/vi\.mock\("@\/components\/Common\/QuickIngest\/FloatingProgressWidget", \(\) => \(\{[\s\S]*?\}\)\)\n/, "")
  const marker = 'describe("QuickIngestWizardModal session runtime", () => {'
  if (!code.includes(marker)) throw new Error("Existing suite not found")
  return { code: code.replace(marker, marker + probe), map:null }
}}], test: { ...base.test, setupFiles: [ui + "/vitest.setup.ts"], include: [target], testNamePattern: "UAT104 real minimize boundary" } }
