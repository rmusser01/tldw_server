import { readFileSync } from "node:fs"
import base from "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/tldw-frontend/vitest.config"
const root = "/Users/macbook-dev/Documents/GitHub/tldw_server2"
const originals = new Map([
  [root + "/apps/packages/ui/src/components/Common/ChatSidebar.tsx", "/private/tmp/uat097-before-ChatSidebar.tsx"],
  [root + "/apps/packages/ui/src/components/Option/Playground/ComposerToolbar.tsx", "/private/tmp/uat097-before-ComposerToolbar.tsx"]
])
export default {
  ...base,
  plugins: [
    ...(base.plugins ?? []),
    {
      name: "original-chat-layout",
      enforce: "pre" as const,
      load(id: string) {
        const original = originals.get(id)
        if (original) return readFileSync(original, "utf8")
      }
    }
  ],
  test: {
    ...base.test,
    setupFiles: [root + "/apps/tldw-frontend/vitest.setup.ts"],
    include: [
      root + "/apps/packages/ui/src/components/Common/ChatSidebar/__tests__/ChatSidebar.tools-first.test.tsx",
      root + "/apps/packages/ui/src/components/Option/Playground/__tests__/ComposerToolbar.test.tsx"
    ]
  }
}
