import { readFileSync } from "node:fs"
import base from "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/tldw-frontend/vitest.config"
const root = "/Users/macbook-dev/Documents/GitHub/tldw_server2"
export default {
  ...base,
  plugins: [
    ...(base.plugins ?? []),
    {
      name: "original-settings-form-lifetime",
      enforce: "pre" as const,
      load(id: string) {
        if (id === root + "/apps/packages/ui/src/components/Option/Settings/tldw.tsx") {
          return readFileSync("/private/tmp/uat096-settings-before.tsx", "utf8")
        }
      }
    }
  ],
  test: {
    ...base.test,
    setupFiles: [root + "/apps/tldw-frontend/vitest.setup.ts"],
    include: [root + "/apps/packages/ui/src/components/Option/Settings/__tests__/tldw.form-lifecycle.test.tsx"]
  }
}
