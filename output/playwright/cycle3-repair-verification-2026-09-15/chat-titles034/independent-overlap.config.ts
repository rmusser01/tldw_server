import base from "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/tldw-frontend/vitest.config"
const web = "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/tldw-frontend"
const target = web + "/__tests__/pages/chat-title.integration.test.tsx"
const probe = `
  it('independent: does not discard a second rename while the first save is pending', async () => {
    const first = deferred<{ id: string; title: string; version: number }>();
    fixture.updateChat.mockReturnValueOnce(first.promise);
    change({ serverChatId: 'server-a', serverChatTitle: 'Cedar' });
    render(<Header />);
    fireEvent.click(await screen.findByRole('button', { name: 'Cedar', exact: true }));
    let input = screen.getByRole('textbox', { name: 'Rename conversation' });
    fireEvent.change(input, { target: { value: 'First saved title' } });
    fireEvent.keyDown(input, { key: 'Enter' });
    await waitFor(() => expect(fixture.updateChat).toHaveBeenCalledTimes(1));
    const nextEdit = screen.getByRole('button', { name: 'Cedar', exact: true });
    if (nextEdit.hasAttribute('disabled') || nextEdit.getAttribute('aria-disabled') === 'true') {
      await act(async () => { first.resolve({ id: 'server-a', title: 'First saved title', version: 2 }); await first.promise; });
      return;
    }
    fireEvent.click(nextEdit);
    input = screen.getByRole('textbox', { name: 'Rename conversation' });
    fireEvent.change(input, { target: { value: 'My second title' } });
    fireEvent.keyDown(input, { key: 'Enter' });
    await act(async () => { first.resolve({ id: 'server-a', title: 'First saved title', version: 2 }); await first.promise; });
    await waitFor(() => expect(fixture.rows.get('local-a')).not.toBe('Local Cedar'));
    const retainedDraft = (screen.queryByRole('textbox', { name: 'Rename conversation' }) as HTMLInputElement | null)?.value;
    expect(fixture.rows.get('local-a') === 'My second title' || retainedDraft === 'My second title').toBe(true);
  });
`
export default { ...base, plugins: [...(Array.isArray(base.plugins) ? base.plugins : []), { name: "independent-title-overlap", enforce: "pre" as const, transform(code: string, id: string) { if (id !== target) return; const end = code.lastIndexOf("});"); return { code: code.slice(0, end) + probe + code.slice(end), map: null } } }], test: { ...base.test, setupFiles: [web + "/vitest.setup.ts"], include: [target] } }
