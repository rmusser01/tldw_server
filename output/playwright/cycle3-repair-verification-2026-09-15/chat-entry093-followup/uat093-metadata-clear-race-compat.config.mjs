import original from "/private/tmp/uat093-metadata-clear-race.config.ts"
import fs from "node:fs"
const target = "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Option/Playground/__tests__/Playground.coordinator.integration.test.tsx"
export default { ...original, plugins: [
 { name: "retained-original-fixture", enforce: "pre", transform(code, id) { if(id===target) return {code:fs.readFileSync("/private/tmp/uat093-prior-coordinator-fixture.tsx","utf8"), map:null} } },
 ...original.plugins,
 { name: "new-selection-revision-mock-contract", enforce: "pre", transform(code, id) { if(id!==target) return; return {code:code.replace('const probeSetAssistant = async', 'let probeSelectionRevision = 0\nconst probeSetAssistant = async').replace('useStoreMessageOption.setState({ _probeAssistant: selection } as any)', 'probeSelectionRevision++; useStoreMessageOption.setState({ _probeAssistant: selection } as any)').replace('useSelectedAssistant: () => [useStoreMessageOption', 'getSelectedAssistantOperationRevision: () => probeSelectionRevision, waitForSelectedAssistantCommit: async () => undefined, useSelectedAssistant: () => [useStoreMessageOption'),map:null} } }
] }
