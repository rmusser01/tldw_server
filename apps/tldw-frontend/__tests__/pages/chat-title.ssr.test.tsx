// @vitest-environment node
import React from 'react';
import { renderToString } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';
import { HeadManagerContext } from 'next/dist/shared/lib/head-manager-context.shared-runtime';
import ChatPage from '@web/pages/chat';

describe('Chat page server rendering', () => {
  it('imports the real shared title hook and emits its fallback without browser globals', () => {
    expect(typeof window).toBe('undefined');
    const updateHead = vi.fn();
    renderToString(
      <HeadManagerContext.Provider value={{ updateHead, mountedInstances: new Set() }}>
        <ChatPage />
      </HeadManagerContext.Provider>
    );
    expect(updateHead.mock.calls.at(-1)?.[0]).toEqual(
      expect.arrayContaining([expect.objectContaining({ type: 'title', props: { children: 'Chat | tldw' } })])
    );
  });
});
