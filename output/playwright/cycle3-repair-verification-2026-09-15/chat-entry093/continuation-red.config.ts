import base from "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/vitest.config"
const ui = "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui"
export default { ...base, plugins: [{ name: "uat093-prior-continuation", enforce: "pre" as const, transform(code: string, id: string) {
  if (id !== ui + "/src/components/Option/Playground/Playground.tsx") return
  return { code: code.replace('if (await loadLocalConversation(returnHistoryIdFromSettings) === false) return;', 'await loadLocalConversation(returnHistoryIdFromSettings);'), map: null }
} }], test: { ...base.test, setupFiles: [ui + "/vitest.setup.ts"], include: [ui + "/src/components/Option/Playground/__tests__/Playground.coordinator.integration.test.tsx"] } }
