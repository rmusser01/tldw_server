import { browser } from "wxt/browser"
import { TldwChatService } from "@/services/tldw/TldwChat"
import {
  captureSelectionContext,
  isSelectionTargetValid,
  replaceSelectionTarget,
  type SelectionContext
} from "@/utils/selection-replace"

const POPUP_HOST_ID = "tldw-copilot-popup-host"
const POPUP_Z_INDEX = 2147483647
const POPUP_WIDTH = 380
const POPUP_MARGIN = 12
const POPUP_OFFSET = 10

type PopupState = "idle" | "streaming" | "done" | "error"

type PopupInstance = {
  close: () => void
  setState: (state: PopupState, message?: string) => void
  setResponseText: (text: string) => void
  setPreviewText: (text: string) => void
  getPreviewText: () => string
  setReplaceEnabled: (enabled: boolean) => void
  schedulePositionUpdate: () => void
}

type PopupPayload = {
  selectionText?: string
  pageUrl?: string
  pageTitle?: string
  frameId?: number
}

const chatService = new TldwChatService()
let activePopup: PopupInstance | null = null
let streamCancelled = false

const getMessage = (
  key:
    | "contextCopilotPopup"
    | "popupStop"
    | "popupOpenSidepanel"
    | "popupCopy"
    | "popupReplace"
    | "popupCancel"
    | "popupClose"
    | "popupNoSelection"
    | "popupNoModel"
    | "popupStreaming"
    | "popupError"
    | "popupPreviewLabel"
    | "popupResponseLabel",
  fallback: string
) => {
  try {
    const msg = browser.i18n?.getMessage(
      key as Parameters<typeof browser.i18n.getMessage>[0]
    )
    return msg || fallback
  } catch {
    return fallback
  }
}

const buildPrompt = (selection: string) =>
  `Respond helpfully to the selected text:\n\n${selection}`

const createPopup = (
  context: SelectionContext,
  actions: {
    onClose: () => void
    onStop: () => void
    onCopy: () => void
    onOpenSidepanel: () => void
    onReplace: () => void
    onCancel: () => void
  }
): PopupInstance => {
  const host = document.createElement("div")
  host.id = POPUP_HOST_ID
  host.style.position = "fixed"
  host.style.inset = "0"
  host.style.zIndex = POPUP_Z_INDEX.toString()
  host.style.pointerEvents = "none"

  const shadow = host.attachShadow({ mode: "open" })
  shadow.innerHTML = `
    <style>
      :host {
        all: initial;
        --popup-bg: 15 23 42;
        --popup-surface: 30 41 59;
        --popup-text: 226 232 240;
        --popup-text-strong: 248 250 252;
        --popup-text-muted: 148 163 184;
        --popup-accent: 56 189 248;
        --popup-danger: 252 165 165;
      }
      * { box-sizing: border-box; }
      .tldw-popup {
        position: fixed;
        width: ${POPUP_WIDTH}px;
        background: rgb(var(--popup-bg));
        color: rgb(var(--popup-text));
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 14px;
        box-shadow: 0 20px 50px rgba(15, 23, 42, 0.4);
        font-family: "Space Grotesk", "Segoe UI", "Helvetica Neue", sans-serif;
        pointer-events: auto;
        overflow: hidden;
      }
      .tldw-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 12px 6px 12px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.12);
      }
      .tldw-title {
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.02em;
        color: rgb(var(--popup-text-strong));
      }
      .tldw-icon-btn {
        border: none;
        background: transparent;
        color: rgb(var(--popup-text-muted));
        font-size: 14px;
        cursor: pointer;
        padding: 4px 6px;
      }
      .tldw-body {
        padding: 10px 12px;
        display: flex;
        flex-direction: column;
        gap: 10px;
      }
      .tldw-label {
        font-size: 11px;
        font-weight: 600;
        color: rgb(var(--popup-text-muted));
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }
      .tldw-output {
        max-height: 180px;
        overflow-y: auto;
        font-size: 13px;
        line-height: 1.5;
        white-space: pre-wrap;
        color: rgb(var(--popup-text));
      }
      .tldw-preview {
        width: 100%;
        min-height: 90px;
        max-height: 160px;
        resize: vertical;
        border-radius: 10px;
        border: 1px solid rgba(148, 163, 184, 0.2);
        background: rgba(15, 23, 42, 0.7);
        color: rgb(var(--popup-text));
        padding: 8px 10px;
        font-size: 13px;
        line-height: 1.5;
      }
      .tldw-status {
        font-size: 12px;
        color: rgb(var(--popup-text-muted));
        margin-right: auto;
      }
      .tldw-error {
        font-size: 12px;
        color: rgb(var(--popup-danger));
      }
      .tldw-footer {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 10px 12px 12px;
        border-top: 1px solid rgba(148, 163, 184, 0.12);
      }
      .tldw-actions {
        display: flex;
        align-items: center;
        gap: 6px;
        flex-wrap: wrap;
        justify-content: flex-end;
      }
      .tldw-btn {
        border: 1px solid rgba(148, 163, 184, 0.2);
        background: rgb(var(--popup-surface) / 0.6);
        color: rgb(var(--popup-text));
        font-size: 12px;
        font-weight: 600;
        padding: 6px 10px;
        border-radius: 999px;
        cursor: pointer;
      }
      .tldw-btn.primary {
        background: rgb(var(--popup-accent));
        color: rgb(var(--popup-bg));
        border-color: transparent;
      }
      .tldw-btn.ghost {
        background: transparent;
      }
      .tldw-hidden { display: none !important; }
    </style>
    <div class="tldw-popup" role="dialog" aria-live="polite">
      <div class="tldw-header">
        <span class="tldw-title">tldw</span>
        <button class="tldw-icon-btn" data-action="close" aria-label="Close">×</button>
      </div>
      <div class="tldw-body">
        <div class="tldw-section">
          <div class="tldw-label" data-role="response-label"></div>
          <div class="tldw-output" data-role="response"></div>
        </div>
        <div class="tldw-section" data-role="preview-section">
          <div class="tldw-label" data-role="preview-label"></div>
          <textarea class="tldw-preview" data-role="preview"></textarea>
        </div>
        <div class="tldw-error tldw-hidden" data-role="error"></div>
      </div>
      <div class="tldw-footer">
        <div class="tldw-status" data-role="status"></div>
        <div class="tldw-actions">
          <button class="tldw-btn" data-action="stop"></button>
          <button class="tldw-btn" data-action="copy"></button>
          <button class="tldw-btn ghost" data-action="open-sidepanel"></button>
          <button class="tldw-btn primary" data-action="replace"></button>
          <button class="tldw-btn" data-action="cancel"></button>
        </div>
      </div>
    </div>
  `

  document.documentElement.appendChild(host)

  const popup = shadow.querySelector(".tldw-popup") as HTMLDivElement
  const responseLabel = shadow.querySelector(
    "[data-role='response-label']"
  ) as HTMLDivElement
  const previewLabel = shadow.querySelector(
    "[data-role='preview-label']"
  ) as HTMLDivElement
  const responseEl = shadow.querySelector(
    "[data-role='response']"
  ) as HTMLDivElement
  const previewSection = shadow.querySelector(
    "[data-role='preview-section']"
  ) as HTMLDivElement
  const previewEl = shadow.querySelector(
    "[data-role='preview']"
  ) as HTMLTextAreaElement
  const statusEl = shadow.querySelector(
    "[data-role='status']"
  ) as HTMLDivElement
  const errorEl = shadow.querySelector(
    "[data-role='error']"
  ) as HTMLDivElement

  const stopBtn = shadow.querySelector("[data-action='stop']") as HTMLButtonElement
  const copyBtn = shadow.querySelector("[data-action='copy']") as HTMLButtonElement
  const openSidepanelBtn = shadow.querySelector(
    "[data-action='open-sidepanel']"
  ) as HTMLButtonElement
  const replaceBtn = shadow.querySelector(
    "[data-action='replace']"
  ) as HTMLButtonElement
  const cancelBtn = shadow.querySelector(
    "[data-action='cancel']"
  ) as HTMLButtonElement
  const closeBtn = shadow.querySelector("[data-action='close']") as HTMLButtonElement

  responseLabel.textContent = getMessage("popupResponseLabel", "Response")
  previewLabel.textContent = getMessage("popupPreviewLabel", "Preview")
  stopBtn.textContent = getMessage("popupStop", "Stop")
  copyBtn.textContent = getMessage("popupCopy", "Copy")
  openSidepanelBtn.textContent = getMessage("popupOpenSidepanel", "Open sidepanel")
  replaceBtn.textContent = getMessage("popupReplace", "Replace selection")
  cancelBtn.textContent = getMessage("popupCancel", "Cancel")
  closeBtn.setAttribute("aria-label", getMessage("popupClose", "Close"))

  let state: PopupState = "idle"
  let rafId: number | null = null
  let lastRect: DOMRect | null = null
  const anchorRange = context.anchorRange

  const updatePosition = () => {
    if (!popup) return
    let rect: DOMRect | null = null
    if (anchorRange) {
      rect = anchorRange.getBoundingClientRect()
      if (!rect || (rect.width === 0 && rect.height === 0)) {
        const rects = anchorRange.getClientRects()
        if (rects.length > 0) rect = rects[0]
      }
    }
    if (!rect && lastRect) rect = lastRect
    if (!rect || (!rect.width && !rect.height)) {
      rect = new DOMRect(
        window.innerWidth / 2,
        window.innerHeight / 2,
        0,
        0
      )
    }
    lastRect = rect

    const popupRect = popup.getBoundingClientRect()
    const width = popupRect.width || POPUP_WIDTH
    const height = popupRect.height || 240

    let left = rect.left
    let top = rect.bottom + POPUP_OFFSET

    if (left + width > window.innerWidth - POPUP_MARGIN) {
      left = window.innerWidth - width - POPUP_MARGIN
    }
    if (left < POPUP_MARGIN) left = POPUP_MARGIN

    if (top + height > window.innerHeight - POPUP_MARGIN) {
      const fallbackTop = rect.top - height - POPUP_OFFSET
      if (fallbackTop >= POPUP_MARGIN) {
        top = fallbackTop
      } else {
        top = window.innerHeight - height - POPUP_MARGIN
      }
    }
    if (top < POPUP_MARGIN) top = POPUP_MARGIN

    popup.style.left = `${Math.round(left)}px`
    popup.style.top = `${Math.round(top)}px`
  }

  const schedulePositionUpdate = () => {
    if (rafId) return
    rafId = window.requestAnimationFrame(() => {
      rafId = null
      updatePosition()
    })
  }

  const updateReplaceVisibility = () => {
    const canReplace = isSelectionTargetValid(context.target)
    previewSection.classList.toggle("tldw-hidden", !canReplace)
    replaceBtn.disabled = !canReplace
    replaceBtn.classList.toggle("tldw-hidden", !canReplace || state !== "done")
    cancelBtn.classList.toggle("tldw-hidden", !canReplace || state !== "done")
  }

  const setState = (next: PopupState, message?: string) => {
    state = next
    errorEl.classList.add("tldw-hidden")
    statusEl.textContent = ""

    stopBtn.classList.toggle("tldw-hidden", state !== "streaming")
    copyBtn.classList.toggle("tldw-hidden", state !== "done")
    openSidepanelBtn.classList.toggle("tldw-hidden", false)

    if (state === "streaming") {
      statusEl.textContent = getMessage("popupStreaming", "Streaming…")
      previewSection.classList.add("tldw-hidden")
    }

    if (state === "done") {
      updateReplaceVisibility()
    }

    if (state === "error") {
      errorEl.textContent = message || getMessage("popupError", "Something went wrong. Please try again.")
      errorEl.classList.remove("tldw-hidden")
      previewSection.classList.add("tldw-hidden")
      copyBtn.classList.add("tldw-hidden")
      stopBtn.classList.add("tldw-hidden")
      replaceBtn.classList.add("tldw-hidden")
      cancelBtn.classList.add("tldw-hidden")
    }

    schedulePositionUpdate()
  }

  const setResponseText = (text: string) => {
    responseEl.textContent = text
    schedulePositionUpdate()
  }

  const setPreviewText = (text: string) => {
    previewEl.value = text
  }

  const getPreviewText = () => previewEl.value

  const setReplaceEnabled = (enabled: boolean) => {
    previewSection.classList.toggle("tldw-hidden", !enabled)
    replaceBtn.disabled = !enabled
    replaceBtn.classList.toggle("tldw-hidden", !enabled || state !== "done")
    cancelBtn.classList.toggle("tldw-hidden", !enabled || state !== "done")
  }

  stopBtn.addEventListener("click", actions.onStop)
  copyBtn.addEventListener("click", actions.onCopy)
  openSidepanelBtn.addEventListener("click", actions.onOpenSidepanel)
  replaceBtn.addEventListener("click", actions.onReplace)
  cancelBtn.addEventListener("click", actions.onCancel)
  closeBtn.addEventListener("click", actions.onClose)

  const onDocumentClick = (event: MouseEvent) => {
    const path = event.composedPath()
    if (path.includes(popup)) return
    actions.onClose()
  }

  const onKeyDown = (event: KeyboardEvent) => {
    if (event.key === "Escape") {
      event.stopPropagation()
      actions.onClose()
    }
  }

  document.addEventListener("mousedown", onDocumentClick, true)
  window.addEventListener("keydown", onKeyDown, true)
  window.addEventListener("scroll", schedulePositionUpdate, true)
  window.addEventListener("resize", schedulePositionUpdate)

  schedulePositionUpdate()

  const close = () => {
    document.removeEventListener("mousedown", onDocumentClick, true)
    window.removeEventListener("keydown", onKeyDown, true)
    window.removeEventListener("scroll", schedulePositionUpdate, true)
    window.removeEventListener("resize", schedulePositionUpdate)
    if (rafId) {
      window.cancelAnimationFrame(rafId)
      rafId = null
    }
    host.remove()
  }

  return {
    close,
    setState,
    setResponseText,
    setPreviewText,
    getPreviewText,
    setReplaceEnabled,
    schedulePositionUpdate
  }
}

const openSidepanel = async () => {
  try {
    await browser.runtime.sendMessage({ type: "sidepanel" })
  } catch {
    // ignore errors
  }
}

const resolveSelectedModel = async (): Promise<string> => {
  const parseValue = (value: unknown) => {
    if (typeof value !== "string") return ""
    try {
      const parsed = JSON.parse(value)
      return typeof parsed === "string" ? parsed : value
    } catch {
      return value
    }
  }

  try {
    const syncArea = browser.storage?.sync
    if (syncArea?.get) {
      const data = await syncArea.get("selectedModel")
      const candidate = parseValue(data?.selectedModel)
      if (candidate) return candidate
    }
  } catch {
    // ignore sync read errors
  }

  try {
    const localArea = browser.storage?.local
    if (localArea?.get) {
      const data = await localArea.get("selectedModel")
      return parseValue(data?.selectedModel)
    }
  } catch {
    // ignore local read errors
  }

  return ""
}

const handlePopupOpen = async (
  payload?: PopupPayload,
  capturedContext?: SelectionContext | null
) => {
  if (activePopup) {
    activePopup.close()
    activePopup = null
  }

  let responseText = ""
  const context =
    capturedContext === undefined
      ? captureSelectionContext(payload?.selectionText)
      : capturedContext
  if (!context) {
    const fallback = createPopup(
      {
        selectionText: "",
        anchorRange: null,
        target: null
      },
      {
        onClose: () => {
          activePopup?.close()
          activePopup = null
        },
        onStop: () => {},
        onCopy: () => {},
        onOpenSidepanel: openSidepanel,
        onReplace: () => {},
        onCancel: () => {
          activePopup?.close()
          activePopup = null
        }
      }
    )
    fallback.setState(
      "error",
      getMessage("popupNoSelection", "No selection found.")
    )
    activePopup = fallback
    return
  }

  const onClose = () => {
    streamCancelled = true
    chatService.cancelStream()
    activePopup?.close()
    activePopup = null
  }

  const onStop = () => {
    streamCancelled = true
    chatService.cancelStream()
    popup.setState("done")
    popup.setResponseText(responseText)
    popup.setPreviewText(responseText)
    popup.setReplaceEnabled(isSelectionTargetValid(context.target))
  }

  const onCopy = async () => {
    const text = responseText
    if (!text) return
    try {
      await navigator.clipboard.writeText(text)
    } catch {
      // ignore copy errors
    }
  }

  const onReplace = () => {
    if (!context.target || !isSelectionTargetValid(context.target)) return
    const replacement = popup.getPreviewText()
    const ok = replaceSelectionTarget(context.target, replacement)
    if (ok) {
      onClose()
    }
  }

  const onCancel = () => {
    onClose()
  }

  const popup = createPopup(context, {
    onClose,
    onStop,
    onCopy,
    onOpenSidepanel: openSidepanel,
    onReplace,
    onCancel
  })
  activePopup = popup

  popup.setState("idle")

  const model = (await resolveSelectedModel()).trim()
  if (!model) {
    popup.setState("error", getMessage("popupNoModel", "Select a model to continue."))
    return
  }

  streamCancelled = false
  popup.setState("streaming")

  try {
    const messages = [{ role: "user" as const, content: buildPrompt(context.selectionText) }]
    for await (const chunk of chatService.streamMessage(messages, { model, stream: true })) {
      if (streamCancelled) break
      responseText += chunk
      popup.setResponseText(responseText)
    }
  } catch (error) {
    if (!streamCancelled) {
      popup.setState("error")
      return
    }
  }

  if (!streamCancelled) {
    popup.setState("done")
    popup.setResponseText(responseText)
    popup.setPreviewText(responseText)
    popup.setReplaceEnabled(isSelectionTargetValid(context.target))
  }
}

/**
 * Publish the popup handler on globalThis so a tiny every-page content
 * script stub can lazily `import()` a web-accessible chunk containing this
 * module and delegate `tldw:popup:open` messages to it.
 */
export const COPILOT_POPUP_HANDLE_KEY = "__tldwCopilotPopupHandle"

export const registerCopilotPopupHandler = (): void => {
  ;(globalThis as Record<string, unknown>)[COPILOT_POPUP_HANDLE_KEY] =
    handlePopupOpen
}

export const getCopilotPopupHandler = (): typeof handlePopupOpen | undefined => {
  const handle = (globalThis as Record<string, unknown>)[
    COPILOT_POPUP_HANDLE_KEY
  ]
  return typeof handle === "function"
    ? (handle as typeof handlePopupOpen)
    : undefined
}

export default defineContentScript({
  matches: ["http://*/*", "https://*/*"],
  main() {
    try {
      ;(window as any).__tldwCopilotPopupReady = true
      document.documentElement.dataset.tldwCopilotPopupReady = "true"
    } catch {
      // ignore readiness flag failures
    }
    registerCopilotPopupHandler()
    browser.runtime.onMessage.addListener((message: any) => {
      if (message?.type !== "tldw:popup:open") return
      void handlePopupOpen(message.payload || {})
      return Promise.resolve({ ok: true })
    })
  }
})
