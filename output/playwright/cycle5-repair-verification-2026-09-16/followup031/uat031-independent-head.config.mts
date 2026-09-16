import { execFileSync } from "node:child_process"
import config from "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/tldw-frontend/vitest.config.ts"

const repo = "/Users/macbook-dev/Documents/GitHub/tldw_server2"
const target = "apps/packages/ui/src/hooks/chat/useChatActions.ts"
const headSource = execFileSync("git", ["show", `HEAD:${target}`], {
  cwd: repo,
  encoding: "utf8"
})

export default {
  ...config,
  cacheDir: "/private/tmp/uat031-independent-head-vite-cache",
  plugins: [
    {
      name: "uat031-read-only-head-implementation",
      enforce: "pre",
      load(id) {
        if (id.split("?")[0] === `${repo}/${target}`) return headSource
      }
    },
    ...(config.plugins ?? [])
  ]
}
