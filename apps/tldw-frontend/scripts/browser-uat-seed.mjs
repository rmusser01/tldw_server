/** Seed a disposable browser context for manual-key UAT. Keep this callback
 * self-contained because Playwright serializes it into the target page.
 */
export function seedManualUatBrowser({ webUrl, serverUrl, apiKey, legacyBootstrap = false }) {
  // Init scripts also execute in new frames and after cross-origin navigation.
  // Only the explicitly selected WebUI may receive the operator's credential.
  const target = new URL(webUrl)
  if (!["http:", "https:"].includes(target.protocol)) return
  if (location.origin !== target.origin) return
  const config = { serverUrl, authMode: "single-user", apiKey }
  localStorage.setItem("tldwConfig", JSON.stringify(config))
  localStorage.setItem("isMigrated", "true")
  if (!legacyBootstrap) return
  localStorage.setItem("serverUrl", serverUrl)
  localStorage.setItem("tldwServerUrl", serverUrl)
  localStorage.setItem("tldw-api-host", serverUrl)
  localStorage.setItem("authMode", "single-user")
  localStorage.setItem("apiKey", apiKey)
  localStorage.setItem("__tldw_first_run_complete", "true")
  localStorage.setItem("assistant_setup_dismissed", "true")
}
