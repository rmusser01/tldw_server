import { defineContentScript } from "wxt/utils/define-content-script"

type CopilotPopupEntrypoint = {
  main: (ctx: unknown) => unknown
}

export default defineContentScript({
  matches: ["http://*/*", "https://*/*"],
  allFrames: true,
  async main(ctx: unknown) {
    try {
      const entrypoint = (await import(
        "@tldw/ui/entries/copilot-popup.content"
      )) as { default: CopilotPopupEntrypoint }
      return entrypoint.default.main(ctx)
    } catch (error) {
      throw new Error("Failed to load copilot popup content entrypoint", {
        cause: error,
      })
    }
  }
})
