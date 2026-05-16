// @vitest-environment jsdom
import React from 'react';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { PlaygroundCollapsedCompositionSummary } from '../PlaygroundCollapsedCompositionSummary';
import type { PlaygroundCompositionPreviewSummary } from '../playground-composition-preview';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback,
  }),
}));

const summary = (): PlaygroundCompositionPreviewSummary => ({
  overallState: 'ready',
  settingsScopeLabel: 'openai:gpt-4.1-mini',
  entries: [
    {
      id: 'prompt',
      kind: 'prompt',
      label: 'Prompt',
      title: 'Research brief',
      detail: 'System prompt',
      state: 'active',
    },
    {
      id: 'assistant',
      kind: 'assistant',
      label: 'Assistant',
      title: 'Research Persona',
      detail: 'Persona selected',
      state: 'active',
    },
    {
      id: 'model',
      kind: 'model',
      label: 'Model',
      title: 'openai:gpt-4.1-mini',
      detail: 'openai',
      state: 'active',
    },
    {
      id: 'context',
      kind: 'context',
      label: 'Context',
      title: '2 active sources',
      detail: '2 configured sources',
      state: 'active',
    },
    {
      id: 'tools',
      kind: 'tools',
      label: 'MCP tools',
      title: 'MCP tools',
      detail: '2 chat tools available',
      state: 'active',
    },
  ],
  contextStack: [],
  footprint: {
    providerMessageCount: 1,
    previewSectionCount: 1,
    contextPieceCount: 2,
    warningCount: 0,
    readiness: 'ready',
  },
});

describe('PlaygroundCollapsedCompositionSummary', () => {
  it('keeps active composition state visible when a rail is collapsed', () => {
    const restoreContext = vi.fn();

    render(
      <PlaygroundCollapsedCompositionSummary
        summary={summary()}
        contextRailVisible={false}
        runtimeRailVisible
        onRestoreContextRail={restoreContext}
      />
    );

    const region = screen.getByRole('region', {
      name: 'Collapsed cockpit summary',
    });
    expect(within(region).getByText('Context hidden')).toBeInTheDocument();
    expect(within(region).getByLabelText('Model: openai:gpt-4.1-mini. openai')).toBeInTheDocument();
    expect(
      within(region).getByLabelText('Prompt: Research brief. System prompt')
    ).toBeInTheDocument();
    expect(
      within(region).getByLabelText('Assistant: Research Persona. Persona selected')
    ).toBeInTheDocument();
    expect(
      within(region).getByLabelText('Context: 2 active sources. 2 configured sources')
    ).toBeInTheDocument();
    expect(
      within(region).getByLabelText('MCP tools: MCP tools. 2 chat tools available')
    ).toBeInTheDocument();

    fireEvent.click(within(region).getByRole('button', { name: 'Restore context rail' }));
    expect(restoreContext).toHaveBeenCalledTimes(1);
  });

  it('does not duplicate rail state while both cockpit rails are visible', () => {
    const { container } = render(
      <PlaygroundCollapsedCompositionSummary
        summary={summary()}
        contextRailVisible
        runtimeRailVisible
      />
    );

    expect(container).toBeEmptyDOMElement();
  });
});
