import React from 'react';
import { ConfigProvider } from 'antd';
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { WorkspaceTour } from '../WorkspaceTips';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (_key: string, fallback: string) => fallback }),
}));

const key = 'dw-tips-tour-completed';
const view = (paused: boolean) => (
  <ConfigProvider theme={{ token: { motion: false } }}>
    <WorkspaceTour paused={paused} />
  </ConfigProvider>
);
const tick = () =>
  act(async () => {
    await vi.advanceTimersByTimeAsync(500);
  });

beforeEach(() => {
  localStorage.removeItem(key);
  vi.useFakeTimers();
});
afterEach(() => {
  cleanup();
  vi.clearAllTimers();
  vi.useRealTimers();
  localStorage.removeItem(key);
});

it('does not start its delayed mask over a dialog and resumes after the dialog closes', async () => {
  const screenView = render(view(false));
  await act(async () => {
    await vi.advanceTimersByTimeAsync(250);
  });
  screenView.rerender(view(true));
  await tick();
  expect(screen.queryByText('Welcome to Document Workspace')).not.toBeInTheDocument();
  expect(localStorage.getItem(key)).toBeNull();
  screenView.rerender(view(false));
  await tick();
  expect(screen.getByText('Welcome to Document Workspace')).toBeVisible();
  fireEvent.click(screen.getByRole('button', { name: 'Close' }));
  expect(localStorage.getItem(key)).toBe('true');
  screenView.unmount();
  render(view(false));
  await tick();
  expect(screen.queryByText('Welcome to Document Workspace')).not.toBeInTheDocument();
});

it('suspends an already-open tour without completing it', async () => {
  const screenView = render(view(false));
  await tick();
  expect(screen.getByText('Welcome to Document Workspace')).toBeVisible();
  screenView.rerender(view(true));
  expect(screen.queryByText('Welcome to Document Workspace')).not.toBeInTheDocument();
  expect(localStorage.getItem(key)).toBeNull();
  screenView.rerender(view(false));
  await tick();
  expect(screen.getByText('Welcome to Document Workspace')).toBeVisible();
});

it('keeps a resumed tour dismissed when Close precedes its delayed timer', async () => {
  const screenView = render(view(false));
  await tick();
  screenView.rerender(view(true));
  screenView.rerender(view(false));
  fireEvent.click(screen.getByRole('button', { name: 'Close' }));
  await tick();
  expect(screen.queryByText('Welcome to Document Workspace')).not.toBeInTheDocument();
  expect(localStorage.getItem(key)).toBe('true');
});
