import { afterAll, vi } from "vitest"
import { browser } from "../../../../../../tldw-frontend/extension/shims/wxt-browser"

// Install the API before source imports construct real Plasmo storage instances.
vi.stubGlobal("chrome", { storage: browser.storage })
afterAll(() => vi.unstubAllGlobals())
