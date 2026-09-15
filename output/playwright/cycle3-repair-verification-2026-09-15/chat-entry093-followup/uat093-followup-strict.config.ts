import base from "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/vitest.config"
const ui = "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui"
const target = ui + "/src/components/Option/Playground/__tests__/Playground.coordinator.integration.test.tsx"
export default {
  ...base,
  plugins: [{
    name: "independent-strict-selection-boundaries",
    enforce: "pre" as const,
    transform(code: string, id: string) {
      if (id !== target) return
      const needle = "render(<Playground />)"
      if (!code.includes(needle)) throw new Error("Strict-mode probe fixture changed")
      return { code: code.replaceAll(needle, "render(<React.StrictMode><Playground /></React.StrictMode>)"), map: null }
    }
  }],
  test: { ...base.test, setupFiles: [ui + "/vitest.setup.ts"], include: [target] }
}
