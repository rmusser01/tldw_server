import original from "/private/tmp/uat093-concurrent-metadata.config.ts"
const target = "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Option/Playground/__tests__/Playground.coordinator.integration.test.tsx"
export default { ...original, plugins: [
  { name: "original-concurrent-probe-form-stub-compatibility", enforce: "pre", transform(code, id) {
    if (id !== target) return
    const current = 'PlaygroundForm: () => <div data-testid="playground-form">{realLoader.additionalLoader ? <AdditionalServerLoader /> : null}</div>'
    if (!code.includes(current)) throw new Error("Expected current coordinator form fixture")
    return {code:code.replace(current, 'PlaygroundForm: () => <div data-testid="playground-form" />'), map:null}
  } },
  ...original.plugins
] }
