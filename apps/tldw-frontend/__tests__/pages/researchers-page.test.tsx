import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import ResearchersPage from '@web/pages/for/researchers';

vi.mock('next/head', () => ({
  default: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

vi.mock('next/link', () => ({
  default: ({
    href,
    children,
    ...rest
  }: {
    href: string;
    children: React.ReactNode;
  }) => (
    <a href={href} {...rest}>
      {children}
    </a>
  ),
}));

describe('ResearchersPage CTA routing', () => {
  it('keeps docs CTAs in docs while sending the trial CTA to deep research', () => {
    render(<ResearchersPage />);

    expect(screen.getByRole('link', { name: 'Start Self-Hosting Free' })).toHaveAttribute(
      'href',
      '/docs/self-hosting'
    );
    expect(screen.getByRole('link', { name: 'View Documentation' })).toHaveAttribute(
      'href',
      '/docs'
    );
    expect(screen.getByRole('link', { name: 'Download Free' })).toHaveAttribute(
      'href',
      '/docs/self-hosting'
    );
    expect(screen.getByRole('link', { name: 'Read the Docs' })).toHaveAttribute(
      'href',
      '/docs'
    );
  });
});
