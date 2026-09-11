/* @vitest-environment jsdom */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import GlobalError from '../global-error';

vi.mock('@sentry/nextjs', () => ({ captureException: vi.fn() }));

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe('GlobalError', () => {
  it('adds dedicated dark-mode style hooks for the retry button and icon container', () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    const reset = vi.fn();
    render(
      <GlobalError
        error={new Error('Boom')}
        reset={reset}
      />,
      { container: document }
    );

    const retryButton = screen.getByRole('button', { name: 'Try Again' });
    expect(retryButton.classList.contains('ge-btn-primary')).toBe(true);
    expect(document.querySelector('.ge-icon')).not.toBeNull();

    const rules = Array.from(document.querySelector('style')?.sheet?.cssRules ?? []);
    const darkMode = rules.find((rule) => rule.type === CSSRule.MEDIA_RULE) as CSSMediaRule;
    expect(darkMode.conditionText).toBe('(prefers-color-scheme:dark)');
    const darkStyles = Array.from(darkMode.cssRules) as CSSStyleRule[];
    expect(darkStyles.find((rule) => rule.selectorText === '.ge-btn-primary')?.style.background)
      .toBe('rgb(59, 130, 246)');
    expect(darkStyles.find((rule) => rule.selectorText === '.ge-icon')?.style.background)
      .toBe('rgb(91, 33, 33)');

    fireEvent.click(retryButton);
    expect(reset).toHaveBeenCalledTimes(1);
  });
});
