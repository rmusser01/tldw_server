/* @vitest-environment jsdom */
import { useState } from 'react';
import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { Button } from './button';
import {
  PrivilegedActionDialogProvider,
  usePrivilegedActionDialog,
} from './privileged-action-dialog';

function PrivilegedActionHarness() {
  const prompt = usePrivilegedActionDialog();
  const [result, setResult] = useState('idle');

  return (
    <div>
      <Button
        onClick={async () => {
          const approval = await prompt({
            title: 'Deactivate user',
            message: 'Support action',
            confirmText: 'Continue',
            requirePassword: false,
          });
          setResult(approval ? `${approval.reason}|${approval.adminPassword}` : 'cancelled');
        }}
      >
        Open dialog
      </Button>
      <p data-testid="privileged-action-result">{result}</p>
    </div>
  );
}

function ConfirmationOnlyHarness() {
  const prompt = usePrivilegedActionDialog();
  const [result, setResult] = useState('idle');

  return (
    <div>
      <Button
        onClick={async () => {
          const approval = await prompt({
            title: 'Rotate webhook secret',
            message: 'Review the current webhook revision before continuing.',
            confirmText: 'Rotate secret',
            confirmationOnly: true,
          });
          setResult(approval ? `${approval.reason}|${approval.adminPassword}` : 'cancelled');
        }}
      >
        Open confirmation
      </Button>
      <p data-testid="confirmation-only-result">{result}</p>
    </div>
  );
}

afterEach(() => {
  cleanup();
});

describe('usePrivilegedActionDialog', () => {
  it('allows single-user approval without entering a password when password reauth is disabled', async () => {
    render(
      <PrivilegedActionDialogProvider>
        <PrivilegedActionHarness />
      </PrivilegedActionDialogProvider>
    );

    fireEvent.click(screen.getByRole('button', { name: 'Open dialog' }));
    fireEvent.change(screen.getByLabelText('Reason'), {
      target: { value: 'Customer requested account restore' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));

    await waitFor(() => {
      expect(screen.getByTestId('privileged-action-result').textContent).toBe(
        'Customer requested account restore|'
      );
    });
  });

  it('supports honest confirmation without collecting unused audit or reauthentication fields', async () => {
    render(
      <PrivilegedActionDialogProvider>
        <ConfirmationOnlyHarness />
      </PrivilegedActionDialogProvider>
    );

    fireEvent.click(screen.getByRole('button', { name: 'Open confirmation' }));

    expect(screen.queryByLabelText('Reason')).not.toBeInTheDocument();
    expect(screen.queryByLabelText('Current password')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Rotate secret' }));

    await waitFor(() => {
      expect(screen.getByTestId('confirmation-only-result').textContent).toBe('|');
    });
  });
});
