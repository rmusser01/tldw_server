import { browser } from "wxt/browser"

type ChromeGlobal = typeof globalThis & { chrome?: typeof chrome }

const getChrome = () => (globalThis as ChromeGlobal).chrome

export const isSidepanelSupported = (): boolean => {
  const chromeGlobal = getChrome()
  if (chromeGlobal?.sidePanel?.open) return true
  const sidebar: any = (browser as any)?.sidebarAction
  return Boolean(sidebar?.open || sidebar?.toggle)
}

export const openSidepanel = async (tabId?: number): Promise<void> => {
  const chromeGlobal = getChrome()
  if (chromeGlobal?.sidePanel?.open) {
    const enableAndOpen = (id?: number): Promise<void> => {
      if (!id) throw new Error("Cannot identify the active browser tab")
      // Start both browser calls during the original gesture; observe their acknowledgments.
      const configured = chromeGlobal.sidePanel.setOptions?.({
        tabId: id,
        path: "sidepanel.html",
        enabled: true,
      })
      const opened = chromeGlobal.sidePanel.open({ tabId: id })
      return Promise.all([configured, opened]).then(() => undefined)
    }
    if (tabId) return enableAndOpen(tabId)
    return new Promise<void>((resolve, reject) => {
      if (!chromeGlobal.tabs?.query) {
        reject(new Error("Cannot identify the active browser tab"))
        return
      }
      chromeGlobal.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        try {
          enableAndOpen(tabs?.[0]?.id).then(resolve, reject)
        } catch (error) {
          reject(error)
        }
      })
    })
  }

  const sidebar: any = (browser as any)?.sidebarAction
  if (sidebar?.open) return sidebar.open()
  if (sidebar?.toggle) return sidebar.toggle()
  throw new Error("Sidebar is unavailable in this browser")
}

export const openSidepanelForActiveTab = async (): Promise<void> =>
  openSidepanel()
