import base from "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/vitest.config";
export default { ...base, plugins: [{ name: "independent-character-fallback-ack", enforce: "pre", transform(code, id) {
  if (id !== "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/hooks/chat/__tests__/useChatActions.character.integration.test.tsx") return;
  const end = code.lastIndexOf("})");
  return { code: code.slice(0,end) + `
  it.each(["fallback", "degraded"])("independent character fallback forwards its confirmed assistant ID to local persistence (%s)", async (outcome) => {
    persistCharacterCompletionMock.mockRejectedValueOnce(outcome === "fallback" ? new Error("Persist endpoint unavailable") : Object.assign(new Error("Saved with validation warning"), { status: 503, detail: { code: "persist_validation_degraded", saved: true, assistant_message_id: "fallback-assistant" } }));
    addChatMessageMock.mockResolvedValueOnce({ id: "user-server-1", version: 1 });
    if (outcome === "fallback") addChatMessageMock.mockResolvedValueOnce({ id: "fallback-assistant", version: 1 });
    const options = createHookOptions();
    const { result } = renderHook(() => useChatActions(options as any));
    await act(async () => { await result.current.onSubmit({message:"Question", image:""}); });
    expect(addChatMessageMock).toHaveBeenCalledTimes(outcome === "fallback" ? 2 : 1);
    expect(saveLocalSuccessMock).toHaveBeenCalledWith(expect.objectContaining({
      userServerMessageId:"user-server-1", assistantServerMessageId:"fallback-assistant", serverMessagesAlreadyPersisted:true
    }));
  });
` + code.slice(end), map:null };
} }], test: { ...base.test, include: ["/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/hooks/chat/__tests__/useChatActions.character.integration.test.tsx"], setupFiles: ["/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/vitest.setup.ts"] } };
