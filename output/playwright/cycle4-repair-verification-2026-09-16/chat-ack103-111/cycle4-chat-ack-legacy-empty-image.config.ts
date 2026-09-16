import base from "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/vitest.config";
export default { ...base, plugins: [{ name: "independent-legacy-empty-image", enforce: "pre", transform(code, id) {
  if (id !== "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/db/dexie/__tests__/server-chat-mirror.test.ts") return;
  const end = code.lastIndexOf("})");
  return { code: code.slice(0, end) + `
  it("independent legacy text-only save shape retains its anchored canonical identity", async () => {
    state.histories.set("alice", history("alice", "A"));
    // saveMessageOnSuccess/saveMessageOnError persist [image], including [""]
    // for a text-only composer. The real server adapter supplies [] here.
    state.messages.set("local-q", { ...row("local-q", "alice", "Repeat"), images: [""] });
    state.messages.set("local-a", { ...row("local-a", "alice", "Reply", "answer"), role: "assistant", parent_message_id: "local-q" });
    state.messages.set("draft", { ...row("draft", "alice", "Repeat"), images: [""] });
    const remote = [incoming("question", "Repeat"), { ...incoming("answer", "Reply"), isBot: true, role: "assistant" }];
    await reconcileServerChatMirror({ historyId: "alice", chatId: "chat-1", ownerKey: "A", messages: remote });
    expect([...state.messages.values()].map(r => [r.id, r.serverMessageId])).toEqual([
      ["local-q", "question"], ["local-a", "answer"], ["draft", undefined]
    ]);
  });
` + code.slice(end), map: null };
} }], test: { ...base.test, include: ["/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/db/dexie/__tests__/server-chat-mirror.test.ts"], setupFiles: ["/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/vitest.setup.ts"] } };
