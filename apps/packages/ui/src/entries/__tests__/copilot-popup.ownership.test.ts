import { readFileSync } from "node:fs"
import { resolve } from "node:path"
import { createContext, runInContext } from "node:vm"
import { JSDOM } from "jsdom"
import ts from "typescript"
import { afterEach, describe, expect, it, vi } from "vitest"
import type { SelectionTarget } from "../../utils/selection-replace"

type FiniteExports = {
  default: { main: () => void }
  registerCopilotPopupHandler: () => void
  detectSelectionTarget: (selection: Selection | null) => SelectionTarget | null
  isSelectionTargetValid: (target: SelectionTarget | null) => boolean
  replaceSelectionTarget: (
    target: SelectionTarget,
    replacement: string
  ) => boolean
}

const repo = resolve(import.meta.dirname, "../../../../../..")
const realms: ReturnType<typeof createRealm>[] = []

const createRealm = (deferredImport = false) => {
  const dom = new JSDOM("<!doctype html><html><body></body></html>", {
    url: "https://example.test/",
    pretendToBeVisual: true
  })
  const window = dom.window
  // jsdom lacks editing/layout APIs; DOM selections and writes remain real.
  Object.defineProperty(window.HTMLElement.prototype, "isContentEditable", {
    get() {
      return (
        this.closest("[contenteditable]")?.getAttribute("contenteditable") !==
          "false" && Boolean(this.closest("[contenteditable]"))
      )
    }
  })
  window.Range.prototype.getBoundingClientRect = () =>
    new window.DOMRect(20, 20, 40, 20)
  const listeners: Array<(message: unknown) => unknown> = []
  const prompts: string[] = []
  const chat = {
    cancelStream: vi.fn(),
    async *streamMessage(messages: Array<{ content: string }>) {
      prompts.push(messages[0].content)
      yield "replacement"
    }
  }
  let releaseImport!: () => void
  const importGate = deferredImport
    ? new Promise<void>((resolve) => {
        releaseImport = resolve
      })
    : Promise.resolve()
  const browser = {
    runtime: {
      getURL: () => "finite:copilot-popup-main.js",
      onMessage: {
        addListener: (listener: (message: unknown) => unknown) =>
          listeners.push(listener)
      },
      sendMessage: vi.fn(async () => undefined)
    },
    storage: {
      sync: { get: vi.fn(async () => ({ selectedModel: "mock-model" })) }
    },
    i18n: { getMessage: () => "" }
  }
  const context = createContext({
    window,
    document: window.document,
    navigator: window.navigator,
    HTMLElement: window.HTMLElement,
    HTMLInputElement: window.HTMLInputElement,
    HTMLTextAreaElement: window.HTMLTextAreaElement,
    DOMRect: window.DOMRect,
    defineContentScript: (entry: unknown) => entry,
    console
  })
  const loaded = new Map<string, FiniteExports>()
  const load = (path: string): FiniteExports => {
    if (loaded.has(path)) return loaded.get(path)
    const exports = {} as FiniteExports
    loaded.set(path, exports)
    const compiled = ts.transpileModule(
      readFileSync(resolve(repo, path), "utf8"),
      {
        compilerOptions: {
          module: ts.ModuleKind.CommonJS,
          target: ts.ScriptTarget.ES2022
        }
      }
    ).outputText
    const require = (id: string): unknown => {
      if (id === "wxt/browser") return { browser }
      if (id === "wxt/utils/define-content-script")
        return { defineContentScript: (entry: unknown) => entry }
      if (id === "@/services/tldw/TldwChat")
        return {
          TldwChatService: class {
            cancelStream = chat.cancelStream
            streamMessage = chat.streamMessage
          }
        }
      if (
        id === "@/utils/selection-replace" ||
        id.endsWith("/utils/selection-replace")
      ) {
        return load("apps/packages/ui/src/utils/selection-replace.ts")
      }
      if (id === "finite:copilot-popup-main.js") {
        return importGate.then(() => {
          const main = load(
            "apps/packages/ui/src/entries/copilot-popup.content.tsx"
          )
          main.registerCopilotPopupHandler()
          return main
        })
      }
      throw new Error(`Unexpected module in finite Copilot realm: ${id}`)
    }
    runInContext(`(function(require, exports) { ${compiled}\n})`, context)(
      require,
      exports
    )
    return exports
  }
  const installStub = () =>
    load("apps/extension/entrypoints/copilot-popup.content.tsx").default.main()
  const receive = (selectionText: string) => {
    for (const listener of listeners)
      listener({ type: "tldw:popup:open", payload: { selectionText } })
  }
  const selectEditable = (text: string) => {
    const root = window.document.createElement("div")
    root.setAttribute("contenteditable", "true")
    root.textContent = text
    window.document.body.appendChild(root)
    const range = window.document.createRange()
    range.selectNodeContents(root)
    window.getSelection()!.removeAllRanges()
    window.getSelection()!.addRange(range)
    return root
  }
  const shadow = () =>
    window.document.getElementById("tldw-copilot-popup-host")?.shadowRoot
  const replace = () =>
    (
      shadow()!.querySelector("[data-action='replace']") as HTMLButtonElement
    ).click()
  const result = {
    window,
    context,
    browser,
    load,
    installStub,
    listeners,
    prompts,
    receive,
    releaseImport,
    selectEditable,
    shadow,
    replace,
    close: () => {
      ;(
        shadow()?.querySelector(
          "[data-action='close']"
        ) as HTMLButtonElement | null
      )?.click()
      window.close()
    }
  }
  realms.push(result)
  return result
}

afterEach(() => {
  for (const realm of realms.splice(0)) realm.close()
})

const settlePopup = async (realm: ReturnType<typeof createRealm>) => {
  await vi.waitFor(() => {
    expect(
      realm.shadow()?.querySelector("[data-role='response']")?.textContent
    ).toBe("replacement")
    expect(
      realm
        .shadow()
        ?.querySelector("[data-action='replace']")
        ?.classList.contains("tldw-hidden")
    ).toBe(false)
  })
}

describe("Copilot receipt-time selection authority", () => {
  it.each(["readOnly", "disabled"] as const)(
    "rejects a captured input that becomes %s before replacement",
    (property) => {
      const realm = createRealm()
      const input = realm.window.document.createElement("textarea")
      input.value = "selection A"
      realm.window.document.body.appendChild(input)
      input.focus()
      input.setSelectionRange(0, 11)
      const helper = realm.load(
        "apps/packages/ui/src/utils/selection-replace.ts"
      )
      const target = helper.detectSelectionTarget(realm.window.getSelection())
      expect(helper.isSelectionTargetValid(target)).toBe(true)
      input[property] = true
      expect(helper.replaceSelectionTarget(target!, "replacement")).toBe(false)
      expect(input.value).toBe("selection A")
    }
  )

  it("rejects a captured contenteditable that becomes noneditable", () => {
    const realm = createRealm()
    const root = realm.selectEditable("selection A")
    const helper = realm.load("apps/packages/ui/src/utils/selection-replace.ts")
    const target = helper.detectSelectionTarget(realm.window.getSelection())
    expect(helper.isSelectionTargetValid(target)).toBe(true)
    root.setAttribute("contenteditable", "false")
    expect(helper.replaceSelectionTarget(target!, "replacement")).toBe(false)
    expect(root.textContent).toBe("selection A")
  })

  it("rejects a captured range that collapses after its selected DOM is removed", () => {
    const realm = createRealm()
    const root = realm.selectEditable("selection A")
    const helper = realm.load("apps/packages/ui/src/utils/selection-replace.ts")
    const target = helper.detectSelectionTarget(realm.window.getSelection())
    expect(helper.isSelectionTargetValid(target)).toBe(true)
    root.replaceChildren()
    expect(helper.replaceSelectionTarget(target!, "replacement")).toBe(false)
    expect(root.textContent).toBe("")
  })

  it("keeps editable selection A when selection changes to B during the first import", async () => {
    const realm = createRealm(true)
    realm.installStub()
    const a = realm.selectEditable("selection A")
    realm.receive("menu fallback")
    const b = realm.selectEditable("selection B")
    realm.releaseImport()
    await settlePopup(realm)
    expect(realm.prompts).toEqual([
      "Respond helpfully to the selected text:\n\nselection A"
    ])
    realm.replace()
    expect(a.textContent).toBe("replacement")
    expect(b.textContent).toBe("selection B")
  })

  it("keeps input offsets and the original field across the deferred import", async () => {
    const realm = createRealm(true)
    realm.installStub()
    const a = realm.window.document.createElement("textarea")
    a.value = "before selected after"
    realm.window.document.body.appendChild(a)
    a.focus()
    a.setSelectionRange(7, 15)
    realm.receive("selected")
    const b = realm.window.document.createElement("textarea")
    b.value = "another field"
    realm.window.document.body.appendChild(b)
    b.focus()
    b.setSelectionRange(0, 7)
    a.setSelectionRange(0, 6)
    realm.releaseImport()
    await settlePopup(realm)
    realm.replace()
    expect(a.value).toBe("before replacement after")
    expect(b.value).toBe("another field")
  })

  it.each(["Hello Alice", "Other world"])(
    "does not overwrite changed textarea value %s after the deferred import",
    async (changedValue) => {
      const realm = createRealm(true)
      realm.installStub()
      const input = realm.window.document.createElement("textarea")
      input.value = "Hello world"
      realm.window.document.body.appendChild(input)
      input.focus()
      input.setSelectionRange(6, 11)
      realm.receive("world")
      input.value = changedValue
      input.setSelectionRange(11, 11)
      realm.releaseImport()
      await vi.waitFor(() =>
        expect(
          realm.shadow()?.querySelector("[data-role='response']")?.textContent
        ).toBe("replacement")
      )
      expect(realm.prompts).toEqual([
        "Respond helpfully to the selected text:\n\nworld"
      ])
      realm.replace()
      expect(input.value).toBe(changedValue)
      expect(
        (
          realm
            .shadow()!
            .querySelector("[data-action='replace']") as HTMLButtonElement
        ).disabled
      ).toBe(true)
    }
  )

  it("rechecks the original textarea value at the final replacement click", async () => {
    const realm = createRealm(true)
    realm.installStub()
    const input = realm.window.document.createElement("textarea")
    input.value = "Hello world"
    realm.window.document.body.appendChild(input)
    input.focus()
    input.setSelectionRange(6, 11)
    realm.receive("world")
    realm.releaseImport()
    await settlePopup(realm)
    input.value = "Hello Alice"
    input.setSelectionRange(11, 11)
    realm.replace()
    expect(input.value).toBe("Hello Alice")
    expect(realm.prompts).toEqual([
      "Respond helpfully to the selected text:\n\nworld"
    ])
  })

  it("does not overwrite textarea edits made by a focus handler before the write", async () => {
    const realm = createRealm(true)
    realm.installStub()
    const input = realm.window.document.createElement("textarea")
    input.value = "Hello world"
    realm.window.document.body.appendChild(input)
    input.focus()
    input.setSelectionRange(6, 11)
    realm.receive("world")
    input.addEventListener("focus", () => {
      input.value = "Hello Alice"
    })
    const other = realm.window.document.createElement("textarea")
    realm.window.document.body.appendChild(other)
    other.focus()
    realm.releaseImport()
    await settlePopup(realm)
    realm.replace()
    expect(input.value).toBe("Hello Alice")
  })

  it.each(["text", "insert"] as const)(
    "does not delete later contenteditable %s edits through a live captured range",
    async (mutation) => {
      const realm = createRealm(true)
      realm.installStub()
      const root = realm.selectEditable("world")
      realm.receive("menu fallback")
      if (mutation === "text") root.firstChild!.nodeValue = "Alice"
      else
        root.insertBefore(
          realm.window.document.createTextNode("new "),
          root.firstChild
        )
      const changedText = mutation === "text" ? "Alice" : "new world"
      expect(realm.window.getSelection()!.toString()).toBe(changedText)
      realm.releaseImport()
      await vi.waitFor(() =>
        expect(
          realm.shadow()?.querySelector("[data-role='response']")?.textContent
        ).toBe("replacement")
      )
      expect(realm.prompts).toEqual([
        "Respond helpfully to the selected text:\n\nworld"
      ])
      realm.replace()
      expect(root.textContent).toBe(changedText)
    }
  )

  it("keeps untrimmed contenteditable range authority while trimming the prompt", async () => {
    const realm = createRealm(true)
    realm.installStub()
    const root = realm.selectEditable(" world ")
    realm.receive("menu fallback")
    realm.releaseImport()
    await settlePopup(realm)
    expect(realm.prompts).toEqual([
      "Respond helpfully to the selected text:\n\nworld"
    ])
    realm.replace()
    expect(root.textContent).toBe("replacement")
  })

  it("does not delete contenteditable edits made by a focus handler before the write", async () => {
    const realm = createRealm(true)
    realm.installStub()
    const root = realm.selectEditable("world")
    root.tabIndex = 0
    realm.receive("world")
    root.addEventListener("focus", () => {
      root.firstChild!.nodeValue = "Alice"
    })
    realm.releaseImport()
    await settlePopup(realm)
    realm.replace()
    expect(root.textContent).toBe("Alice")
  })

  it("retains unchanged selection replacement through the direct registered handler", async () => {
    const realm = createRealm()
    const root = realm.selectEditable("direct selection")
    realm
      .load("apps/packages/ui/src/entries/copilot-popup.content.tsx")
      .registerCopilotPopupHandler()
    const handle = runInContext(
      "__tldwCopilotPopupHandle",
      realm.context
    ) as (payload: { selectionText: string }) => Promise<void>
    await handle({ selectionText: "menu fallback" })
    await settlePopup(realm)
    expect(realm.prompts).toEqual([
      "Respond helpfully to the selected text:\n\ndirect selection"
    ])
    realm.replace()
    expect(root.textContent).toBe("replacement")
  })

  it("does not invent a later selection when receipt had no selection", async () => {
    const realm = createRealm(true)
    realm.installStub()
    realm.receive("")
    realm.selectEditable("later selection")
    realm.releaseImport()
    await vi.waitFor(() =>
      expect(
        realm.shadow()?.querySelector("[data-role='error']")?.textContent
      ).toBe("No selection found.")
    )
    expect(realm.prompts).toEqual([])
  })

  it("does not redirect replacement after the original target is detached", async () => {
    const realm = createRealm(true)
    realm.installStub()
    const a = realm.selectEditable("selection A")
    realm.receive("selection A")
    a.remove()
    const b = realm.selectEditable("selection B")
    realm.releaseImport()
    await vi.waitFor(() =>
      expect(
        realm.shadow()?.querySelector("[data-role='response']")?.textContent
      ).toBe("replacement")
    )
    realm.replace()
    expect(b.textContent).toBe("selection B")
  })

  it("rejects replacement when the captured range has moved outside its editable root", async () => {
    const realm = createRealm(true)
    realm.installStub()
    const a = realm.selectEditable("selection A")
    realm.receive("selection A")
    const outside = realm.window.document.createElement("div")
    realm.window.document.body.appendChild(outside)
    outside.appendChild(a.firstChild!)
    realm.window.getSelection()!.removeAllRanges()
    realm.releaseImport()
    await vi.waitFor(() =>
      expect(
        realm.shadow()?.querySelector("[data-role='response']")?.textContent
      ).toBe("replacement")
    )
    realm.replace()
    expect(a.textContent).toBe("")
    expect(outside.textContent).toBe("selection A")
  })

  it("retains payload fallback and does not replace noneditable page text", async () => {
    const realm = createRealm(true)
    realm.installStub()
    realm.receive("fallback text")
    const b = realm.selectEditable("later editable selection")
    realm.releaseImport()
    await vi.waitFor(() =>
      expect(
        realm.shadow()?.querySelector("[data-role='response']")?.textContent
      ).toBe("replacement")
    )
    expect(realm.prompts).toEqual([
      "Respond helpfully to the selected text:\n\nfallback text"
    ])
    realm.replace()
    expect(b.textContent).toBe("later editable selection")
  })
})

const createBackground = (
  frames: Map<number, ReturnType<typeof createRealm>>
) => {
  let onClick!: (info: unknown, tab: unknown) => Promise<void>
  const injected: unknown[] = []
  const deliveries: number[] = []
  const notifications: unknown[][] = []
  const browser = {
    runtime: {
      onMessage: { addListener() {} },
      onConnect: { addListener() {} }
    },
    storage: { onChanged: { addListener() {} } },
    contextMenus: {
      onClicked: {
        addListener: (listener: typeof onClick) => {
          onClick = listener
        }
      }
    },
    tabs: {
      async sendMessage(
        _tabId: number,
        message: { payload: { selectionText: string } },
        options?: { frameId: number }
      ) {
        const ids = options ? [options.frameId] : [...frames.keys()]
        const receivers = ids.filter((id) => frames.get(id)?.listeners.length)
        if (!receivers.length)
          throw new Error(
            "Could not establish connection. Receiving end does not exist."
          )
        for (const id of receivers) {
          deliveries.push(id)
          frames.get(id)!.receive(message.payload.selectionText)
        }
        return { ok: true }
      }
    },
    scripting: {
      async executeScript(options: {
        target: { tabId: number; frameIds: number[] }
        files: string[]
      }) {
        injected.push(options)
        for (const id of options.target.frameIds) frames.get(id)!.installStub()
      }
    },
    i18n: { getMessage: (key: string) => key }
  }
  const dependencies: Record<string, unknown> = {
    "wxt/browser": { browser },
    "@/utils/safe-storage": {
      createSafeStorage: () => ({ get: async () => undefined })
    },
    "@/services/background-helpers": {
      notify: (...args: unknown[]) => notifications.push(args)
    },
    "@/services/recipe-persistence-registry": {
      RecipePersistenceRegistry: class {}
    },
    "@/entries/shared/quick-ingest-session-runtime": {
      createQuickIngestSessionRuntime: () => ({})
    },
    "@/entries/background-session-store": {
      getSessionStorageArea: () => undefined,
      createSerializedSessionStateWriter: () => () => {}
    },
    "@/entries/shared/background-init": {
      initBackground: async () => {
        throw new Error("Finite fixture skips unrelated initialization")
      }
    }
  }
  const context = createContext({
    defineBackground: (entry: unknown) => entry,
    crypto: { randomUUID: () => "finite-id" },
    console: { debug() {}, error() {} }
  })
  const exports = {} as { default: { main: () => void } }
  const compiled = ts.transpileModule(
    readFileSync(
      resolve(repo, "apps/packages/ui/src/entries/background.ts"),
      "utf8"
    ),
    {
      compilerOptions: {
        module: ts.ModuleKind.CommonJS,
        target: ts.ScriptTarget.ES2022
      }
    }
  ).outputText
  runInContext(`(function(require, exports) { ${compiled}\n})`, context)(
    (id: string) => dependencies[id] ?? {},
    exports
  )
  exports.default.main()
  return { onClick, injected, deliveries, notifications, browser }
}

describe("Copilot originating-frame background dispatch", () => {
  it("delivers only to the originating iframe when both frames have a receiver", async () => {
    const top = createRealm()
    top.installStub()
    const topSelection = top.selectEditable("retained top selection")
    const frame = createRealm()
    frame.installStub()
    const target = frame.selectEditable("iframe selection")
    const bg = createBackground(
      new Map([
        [0, top],
        [7, frame]
      ])
    )
    await bg.onClick(
      {
        menuItemId: "contextual-popup-pa",
        frameId: 7,
        selectionText: "iframe selection"
      },
      { id: 42 }
    )
    await settlePopup(frame)
    frame.replace()
    expect(bg.deliveries).toEqual([7])
    expect(bg.injected).toEqual([])
    expect(top.prompts).toEqual([])
    expect(topSelection.textContent).toBe("retained top selection")
    expect(target.textContent).toBe("replacement")
  })

  it("lazily injects only the minimal stub into the absent origin iframe then retries there", async () => {
    const top = createRealm()
    top.installStub()
    const frame = createRealm()
    const target = frame.selectEditable("iframe selection")
    const bg = createBackground(
      new Map([
        [0, top],
        [7, frame]
      ])
    )
    await bg.onClick(
      {
        menuItemId: "contextual-popup-pa",
        frameId: 7,
        selectionText: "iframe selection"
      },
      { id: 42 }
    )
    await settlePopup(frame)
    frame.replace()
    expect(bg.injected).toEqual([
      {
        target: { tabId: 42, frameIds: [7] },
        files: ["content-scripts/copilot-popup.js"]
      }
    ])
    expect(bg.deliveries).toEqual([7])
    expect(top.prompts).toEqual([])
    expect(target.textContent).toBe("replacement")
  })

  it("keeps ordinary top-frame delivery targeted without injecting a stub", async () => {
    const top = createRealm()
    top.installStub()
    top.selectEditable("top selection")
    const bg = createBackground(new Map([[0, top]]))
    await bg.onClick(
      { menuItemId: "contextual-popup-pa", selectionText: "top selection" },
      { id: 42 }
    )
    await settlePopup(top)
    expect(bg.deliveries).toEqual([0])
    expect(bg.injected).toEqual([])
  })

  it("preserves delivery failure notification if bounded injection is denied", async () => {
    const frame = createRealm()
    const bg = createBackground(new Map([[7, frame]]))
    bg.browser.scripting.executeScript = async () => {
      throw new Error("Cannot access this frame")
    }
    await bg.onClick(
      {
        menuItemId: "contextual-popup-pa",
        frameId: 7,
        selectionText: "iframe selection"
      },
      { id: 42 }
    )
    expect(bg.deliveries).toEqual([])
    expect(bg.notifications).toEqual([
      ["contextCopilotPopup", "contextCopilotPopupDeliveryFailed"]
    ])
  })

  it("does not inject another receiver for an error other than an absent stub", async () => {
    const top = createRealm()
    top.installStub()
    const bg = createBackground(new Map([[0, top]]))
    bg.browser.tabs.sendMessage = async () => {
      throw new Error("The message port closed before a response was received.")
    }
    await bg.onClick(
      { menuItemId: "contextual-popup-pa", selectionText: "top selection" },
      { id: 42 }
    )
    expect(bg.injected).toEqual([])
    expect(bg.notifications).toEqual([
      ["contextCopilotPopup", "contextCopilotPopupDeliveryFailed"]
    ])
  })
})
