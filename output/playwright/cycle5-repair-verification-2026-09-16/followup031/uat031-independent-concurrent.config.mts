import { readFileSync } from "node:fs"
import config from "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/tldw-frontend/vitest.config.ts"

const target = "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/hooks/chat/__tests__/useChatActions.character.integration.test.tsx"
const source = readFileSync(target, "utf8")
const sequential = '    await act(async () => { await result.current.regenerateLastMessage() })\n    await act(async () => { await result.current.regenerateLastMessage() })'
if (!source.includes(sequential)) throw new Error("Expected sequential retry fixture missing")
const concurrent = source.replace(sequential, '    await act(async () => { await Promise.all([result.current.regenerateLastMessage(), result.current.regenerateLastMessage()]) })')

export default {
  ...config,
  cacheDir: "/private/tmp/uat031-independent-concurrent-vite-cache",
  plugins: [
    {
      name: "uat031-read-only-overlapping-failed-retries",
      enforce: "pre",
      load(id) {
        if (id.split("?")[0] === target) return concurrent
      }
    },
    ...(config.plugins ?? [])
  ]
}
