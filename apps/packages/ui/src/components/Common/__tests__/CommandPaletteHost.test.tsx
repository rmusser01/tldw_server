import React from 'react';
import { Drawer } from 'antd';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';

import { CommandPaletteHost } from "../CommandPaletteHost"
import { isMac } from "@/hooks/useKeyboardShortcuts"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, defaultValue?: string) => defaultValue ?? key
  })
}))

vi.mock("@/hooks/keyboard/useShortcutConfig", () => ({
  useShortcutConfig: () => ({
    shortcuts: {},
    updateShortcut: vi.fn(),
    resetShortcuts: vi.fn(),
    resetShortcut: vi.fn()
  })
}))

vi.mock("@/components/Option/Prompt/usePromptPaletteCommands", () => ({
  usePromptPaletteCommands: () => []
}))

describe("CommandPaletteHost", () => {
  beforeEach(() => {
    HTMLElement.prototype.scrollIntoView = vi.fn()
  })

  it("lazy-mounts and opens the palette in response to the global open event", async () => {
    render(
      <MemoryRouter>
        <CommandPaletteHost />
      </MemoryRouter>
    )

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()

    window.dispatchEvent(new CustomEvent("tldw:open-command-palette"))

    expect(await screen.findByRole("dialog")).toBeInTheDocument()
  })

  it("blocks the Cmd/Ctrl+K opener on the research workspace route while still honoring the event API", async () => {
    render(
      <MemoryRouter initialEntries={["/research-workspace"]}>
        <CommandPaletteHost />
      </MemoryRouter>
    )

    fireEvent.keyDown(
      document,
      isMac ? { key: "k", metaKey: true } : { key: "k", ctrlKey: true }
    )

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()

    window.dispatchEvent(new CustomEvent("tldw:open-command-palette"))

    expect(await screen.findByRole("dialog")).toBeInTheDocument()
  })

  it("opens the palette with Control+K on standard routes", async () => {
    render(
      <MemoryRouter>
        <CommandPaletteHost />
      </MemoryRouter>
    )

    fireEvent.keyDown(document, { key: "k", ctrlKey: true })

    expect(await screen.findByRole("dialog")).toBeInTheDocument()
  })

  it("opens the palette even when an input is focused", async () => {
    render(
      <MemoryRouter>
        <div>
          <input aria-label="composer" />
          <CommandPaletteHost />
        </div>
      </MemoryRouter>
    )

    const input = screen.getByRole("textbox", { name: "composer" })
    input.focus()
    fireEvent.keyDown(input, { key: "k", ctrlKey: true })

    expect(await screen.findByRole("dialog")).toBeInTheDocument()
  })

  it("opens the palette even when a focused control stops keydown propagation", async () => {
    render(
      <MemoryRouter>
        <div>
          <input
            aria-label="propagation-blocker"
            onKeyDown={(event) => {
              event.stopPropagation()
            }}
          />
          <CommandPaletteHost />
        </div>
      </MemoryRouter>
    )

    const input = screen.getByRole("textbox", { name: "propagation-blocker" })
    input.focus()
    fireEvent.keyDown(input, { key: "k", ctrlKey: true })

    expect(await screen.findByRole("dialog")).toBeInTheDocument()
  })
})

describe('Compact drawer admission', () => {
  it('retires an already pending palette load when Chats opens, and allows a fresh request after dismissal', async () => {
    let release!: (module: typeof import('../CommandPalette')) => void;
    const pending = new Promise<typeof import('../CommandPalette')>((resolve) => {
      release = resolve;
    });
    const loadPalette = vi.fn(() => pending);
    const view = render(
      <MemoryRouter>
        <CommandPaletteHost loadPalette={loadPalette} />
      </MemoryRouter>
    );
    await act(async () => {
      window.dispatchEvent(new CustomEvent('tldw:open-command-palette'));
    });
    expect(loadPalette).toHaveBeenCalledTimes(1);
    view.rerender(
      <MemoryRouter>
        <CommandPaletteHost enabled={false} loadPalette={loadPalette} />
      </MemoryRouter>
    );
    await act(async () => {
      release(await import('../CommandPalette'));
      await pending;
    });
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    view.rerender(
      <MemoryRouter>
        <CommandPaletteHost loadPalette={loadPalette} />
      </MemoryRouter>
    );
    await act(async () => {
      window.dispatchEvent(new CustomEvent('tldw:open-command-palette'));
    });
    expect(await screen.findByRole('dialog', { name: 'Search commands' })).toBeInTheDocument();
    expect(loadPalette).toHaveBeenCalledTimes(2);
  });

  it('keeps the real palette closed under the real Drawer for keyboard and explicit event, then opens after dismissal', async () => {
    const loadPalette = vi.fn(() => import('../CommandPalette'));
    const Shell = () => {
      const [open, setOpen] = React.useState(true);
      return (
        <MemoryRouter>
          {open && (
            <Drawer open title="Chats" onClose={() => setOpen(false)}>
              <button>Drawer control</button>
            </Drawer>
          )}
          <CommandPaletteHost enabled={!open} loadPalette={loadPalette} />
        </MemoryRouter>
      );
    };
    render(<Shell />);
    const drawer = await screen.findByRole('dialog', { name: 'Chats' });
    fireEvent.keyDown(drawer, { key: 'k', ctrlKey: true });
    fireEvent.keyDown(drawer, { key: 'k', metaKey: true });
    await act(async () => {
      window.dispatchEvent(new CustomEvent('tldw:open-command-palette'));
    });
    expect(loadPalette).not.toHaveBeenCalled();
    expect(screen.getAllByRole('dialog')).toEqual([drawer]);
    fireEvent.keyDown(drawer, { key: 'Escape', code: 'Escape' });
    expect(screen.queryByRole('dialog', { name: 'Chats' })).not.toBeInTheDocument();
    window.dispatchEvent(new CustomEvent('tldw:open-command-palette'));
    expect(await screen.findByRole('dialog', { name: 'Search commands' })).toBeInTheDocument();
  });
});
