export function resolveExtensionHeadlessMode(): boolean {
  const explicitHeadless = String(process.env.TLDW_E2E_EXTENSION_HEADLESS || "")
    .trim()
    .toLowerCase()
  if (explicitHeadless) {
    return !["0", "false", "no", "off"].includes(explicitHeadless)
  }

  return !!process.env.CI
}
