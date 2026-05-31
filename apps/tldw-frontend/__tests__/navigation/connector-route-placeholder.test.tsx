import React from 'react';
import { render, screen, within } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { ConnectorRoutePlaceholder } from '@web/components/navigation/ConnectorRoutePlaceholder';

vi.mock('next/link', () => ({
  default: ({ href, children, ...rest }: { href: string; children: React.ReactNode }) => (
    <a href={href} {...rest}>
      {children}
    </a>
  ),
}));

describe('ConnectorRoutePlaceholder', () => {
  it('renders connector routes as placeholders with one primary action', () => {
    render(<ConnectorRoutePlaceholder route="/connectors/jobs" />);

    expect(screen.getByText('Connector placeholder')).toBeVisible();
    expect(screen.getByRole('heading', { name: 'Connector Jobs' })).toBeVisible();
    expect(
      screen.getByText(/Connector job orchestration is not active in this build/i)
    ).toBeVisible();
    expect(screen.queryByText(/Coming Soon/i)).not.toBeInTheDocument();

    const primaryActions = screen.getAllByTestId('connector-placeholder-primary');
    expect(primaryActions).toHaveLength(1);
    expect(primaryActions[0]).toHaveAttribute('href', '/scheduled-tasks');
    expect(primaryActions[0]).toHaveTextContent('Open Scheduled Tasks');
  });

  it('points connector child routes to currently supported alternatives', () => {
    render(<ConnectorRoutePlaceholder route="/connectors/sources" />);

    expect(screen.getByRole('heading', { name: 'Connector Sources' })).toBeVisible();
    expect(
      screen.getByText(/Source-specific connector workflows are not active in this build/i)
    ).toBeVisible();

    const alternatives = screen.getByTestId('connector-placeholder-alternatives');
    const links = within(alternatives).getAllByRole('link');
    const destinations = links.map((link) => link.getAttribute('href'));

    expect(destinations).toEqual(['/integrations', '/settings']);
    expect(destinations).not.toContain('/sources');
    expect(screen.queryByText(/browse connector catalog/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/run connector jobs/i)).not.toBeInTheDocument();
  });

  it('falls back to the connector hub for unknown connector placeholder paths', () => {
    render(<ConnectorRoutePlaceholder route="/connectors/unknown" />);

    expect(screen.getByRole('heading', { name: 'Connectors' })).toBeVisible();
    expect(screen.getByTestId('connector-placeholder-primary')).toHaveAttribute(
      'href',
      '/settings'
    );
  });
});
