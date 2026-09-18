import path from "path"
import { defineConfig } from "vitest/config"

const repoRoot = "/Users/macbook-dev/Documents/GitHub/tldw_server2"

export default defineConfig({
  resolve: {
    alias: {
      "@": path.join(repoRoot, "apps/packages/ui/src"),
      "~": path.join(repoRoot, "apps/packages/ui/src"),
      "pa-tesseract.js": path.join(
        repoRoot,
        "apps/tldw-frontend/node_modules/pa-tesseract.js"
      )
    }
  },
  test: {
    environment: "jsdom",
    setupFiles: [path.join(repoRoot, "apps/packages/ui/vitest.setup.ts")],
    include: [
      path.join(
        repoRoot,
        ".tmp/uat-repairs-231-246/worldbook274/baseline-source/__tests__/worldBookEntryUtils.baseline.test.ts"
      )
    ]
  }
})
