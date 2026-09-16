import base from "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/vitest.config.ts"
export default {
  ...base,
  resolve: { ...base.resolve, alias: { ...base.resolve.alias, "pa-tesseract.js": "/private/tmp/cycle4-uat122-ocr-unused.ts" } },
  root: "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui",
  test: { ...base.test, include: ["/private/tmp/cycle4-uat122-formatter-probe.test.ts"] }
}
