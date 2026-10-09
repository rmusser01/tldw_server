import { defineUnlistedScript } from "wxt/utils/define-unlisted-script"
import { registerCopilotPopupHandler } from "@tldw/ui/entries/copilot-popup.content"

/**
 * Web-accessible lazily-loaded chunk carrying the actual copilot popup
 * implementation (chat streaming, popup DOM, selection replacement). The
 * every-page copilot content script is a minimal stub that `import()`s this
 * chunk on the first `tldw:popup:open` message, so pages only pay for the
 * popup when it is actually used.
 *
 * Loaded via `import(browser.runtime.getURL("copilot-popup-main.js"))` — see
 * `extension/entrypoints/copilot-popup.content.tsx`.
 */
export default defineUnlistedScript(() => {
  registerCopilotPopupHandler()
})
