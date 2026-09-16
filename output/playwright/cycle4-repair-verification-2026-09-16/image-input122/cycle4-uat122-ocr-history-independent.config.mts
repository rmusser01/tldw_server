import base from "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/vitest.config.ts"
const probe = "/private/tmp/cycle4-uat122-ocr-history-independent.test.tsx"
const original = "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/hooks/chat/__tests__/useChatActions.saved-normal.integration.test.tsx"
export default {
  ...base,
  plugins: [{ name: "private-probe-dependencies", enforce: "pre", async resolveId(source, importer) {
    if (importer === probe && !source.startsWith(".") && !source.startsWith("/") && source !== "vitest")
      return this.resolve(source, original, { skipSelf: true })
  } }],
  resolve: { ...base.resolve, alias: { ...base.resolve.alias, "pa-tesseract.js": "/private/tmp/cycle4-uat122-ocr-unused.ts" } },
  root: "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui",
  test: { ...base.test, include: [probe] }
}
