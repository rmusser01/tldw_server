import original from "/private/tmp/uat093-web-storage-cross-tab-event.config"
import { readFileSync } from "node:fs"
const target = "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Option/Playground/__tests__/Playground.coordinator.integration.test.tsx"
// Replay unchanged original behavior probes using the retained matching fixture.
// Prior hypothetical production transforms are omitted: all production is current.
const probes = original.plugins.filter(plugin => [
  "uat093-real-web-storage",
  "isolate-web-storage-probe-and-record-mismatch",
  "uat093-actual-cross-tab-storage-control",
  "uat093-vitest-storage-event-adapter"
].includes(plugin.name)).map(plugin => ({...plugin, transform(code:string,id:string) {
  if(id!==target) return
  return plugin.transform(code,id)
}}))
export default {...original, plugins:[{
  name:"retained-coordinator-fixture", enforce:"pre" as const,
  transform(_code:string,id:string){if(id===target)return {code:readFileSync("/private/tmp/uat093-before-ownership-coordinator.tsx","utf8"),map:null}}
},...probes]}
