import { fireEvent, render } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { useKeyboardShortcuts } from './use-keyboard-shortcuts';
import { useSensitiveNavigationGuard } from './use-sensitive-navigation-guard';

const mocks = vi.hoisted(() => ({
  push: vi.fn(),
}));

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: mocks.push }),
}));

const Harness = ({ onBlocked }: { onBlocked: () => void }) => {
  useSensitiveNavigationGuard(true, onBlocked);
  useKeyboardShortcuts();
  return null;
};

describe('keyboard shortcut navigation', () => {
  beforeEach(() => {
    mocks.push.mockReset();
  });

  it('does not start a programmatic route transition while sensitive state is active', () => {
    const onBlocked = vi.fn();
    render(<Harness onBlocked={onBlocked} />);

    fireEvent.keyDown(window, { key: 'g' });
    fireEvent.keyDown(window, { key: 'u' });

    expect(mocks.push).not.toHaveBeenCalled();
    expect(onBlocked).toHaveBeenCalledOnce();
  });
});
