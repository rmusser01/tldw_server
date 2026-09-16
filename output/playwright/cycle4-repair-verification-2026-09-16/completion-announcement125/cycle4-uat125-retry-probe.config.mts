import base from "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/vitest.config.ts"
const target = "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Common/Playground/__tests__/Message.error-recovery.integration.test.tsx"
export default {
  ...base,
  root: "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui",
  plugins: [{ name: "private-uat125-retry-probe", enforce: "pre", transform(source, id) {
    if (id.split("?")[0] !== target) return
    const insertion = source.lastIndexOf("\n})")
    if (insertion < 0) throw new Error("test suite anchor missing")
    return source.slice(0, insertion) + `
      it("independent does not announce completion when Retry starts immediately after an error", () => {
        const view = render(<PlaygroundMessage {...baseProps} isStreaming />)
        decodeChatErrorPayloadMock.mockReturnValue({ summary: "Failed", hint: "Retry", detail: "" })
        view.rerender(<PlaygroundMessage {...baseProps} message="failed response" />)
        expect(screen.queryByText("Response complete")).not.toBeInTheDocument()
        decodeChatErrorPayloadMock.mockReturnValue(null)
        view.rerender(<PlaygroundMessage {...baseProps} message="▋" isStreaming />)
        expect(screen.queryByText("Response complete")).not.toBeInTheDocument()
      })
    ` + source.slice(insertion)
  } }],
  test: { ...base.test, include: [target] }
}
