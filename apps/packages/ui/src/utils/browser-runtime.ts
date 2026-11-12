export const getBrowserRuntime = () => {
  try {
    const globalScope = globalThis as typeof globalThis & {
      browser?: typeof chrome
      chrome?: typeof chrome
    }

    if (globalScope.browser?.runtime) {
      return globalScope.browser.runtime
    }

    if (globalScope.chrome?.runtime) {
      return globalScope.chrome.runtime
    }
  } catch {
    // ignore runtime detection errors
  }

  return null
}

export const isExtensionRuntime = (runtime = getBrowserRuntime()): boolean =>
  Boolean(runtime?.id)

export const resolveOptionsHref = (path: string) => {
  try {
    const runtime = getBrowserRuntime()
    if (isExtensionRuntime(runtime) && runtime?.getURL) {
      return runtime.getURL(`/options.html#${path}`)
    }
  } catch {
    // Fall back to the hosted WebUI path when extension APIs are unavailable.
  }

  return path
}
