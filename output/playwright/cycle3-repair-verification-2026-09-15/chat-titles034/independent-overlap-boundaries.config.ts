import base from "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/tldw-frontend/vitest.config"
const web = "/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/tldw-frontend"
const target = web + "/__tests__/pages/chat-title.integration.test.tsx"
const probe = `
  it.each(['failure', 'principal'] as const)('independent: retains or masks the newer draft after pending %s', async outcome => {
    let resolve!: (value: { id: string; title: string; version: number }) => void;
    let reject!: (error: Error) => void;
    const pending = new Promise<{ id: string; title: string; version: number }>((yes, no) => { resolve = yes; reject = no; });
    fixture.updateChat.mockReturnValueOnce(pending);
    const error = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    change({ serverChatId: 'server-a', serverChatTitle: 'Cedar', serverChatVersion: 1 });
    render(<Header />);
    fireEvent.click(await screen.findByRole('button', { name: 'Cedar', exact: true }));
    let input = screen.getByRole('textbox', { name: 'Rename conversation' });
    fireEvent.change(input, { target: { value: 'First title' } });
    fireEvent.keyDown(input, { key: 'Enter' });
    await waitFor(() => expect(fixture.updateChat).toHaveBeenCalledTimes(1));
    fireEvent.click(screen.getByRole('button', { name: 'Cedar', exact: true }));
    input = screen.getByRole('textbox', { name: 'Rename conversation' });
    fireEvent.change(input, { target: { value: 'Latest private draft' } });
    fireEvent.keyDown(input, { key: 'Enter' });
    if (outcome === 'principal') act(() => { window.dispatchEvent(new CustomEvent('tldw:auth-principal-changed', { detail: { kind: 'logout' } })); });
    await act(async () => {
      if (outcome === 'failure') reject(new Error('First save failed'));
      else resolve({ id: 'server-a', title: 'First title', version: 7 });
      await pending.catch(() => undefined);
    });
    if (outcome === 'failure') {
      await waitFor(() => expect(error).toHaveBeenCalled());
      expect(screen.getByRole('textbox', { name: 'Rename conversation' })).toHaveValue('Latest private draft');
      fireEvent.keyDown(input, { key: 'Enter' });
      await waitFor(() => expect(fixture.rows.get('local-a')).toBe('Latest private draft'));
      expect(fixture.updateChat).toHaveBeenLastCalledWith('server-a', { title: 'Latest private draft' }, expect.objectContaining({ expectedVersion: 1 }));
    } else {
      expect(screen.queryByRole('textbox', { name: 'Rename conversation' })).toBeNull();
      expect(fixture.rows.get('local-a')).toBe('Local Cedar');
      expect(useStoreMessageOption.getState().serverChatTitle).toBe('Cedar');
    }
  });
`
export default { ...base, plugins: [...(Array.isArray(base.plugins) ? base.plugins : []), { name: "independent-title-overlap-boundaries", enforce: "pre" as const, transform(code: string, id: string) { if (id !== target) return; const end = code.lastIndexOf("});"); return { code: code.slice(0, end) + probe + code.slice(end), map: null } } }], test: { ...base.test, setupFiles: [web + "/vitest.setup.ts"], include: [target] } }
