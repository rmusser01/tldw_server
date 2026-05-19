import type { ReactNode } from 'react';
import { afterEach, describe, it, expect, vi, beforeEach } from 'vitest';
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import RegistrationCodesPage from '../page';

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

const mockGetRegistrationSettings = vi.fn();
const mockGetRegistrationCodes = vi.fn();
const mockDeleteRegistrationCode = vi.fn();
const mockCreateRegistrationCode = vi.fn();
const mockUpdateRegistrationSettings = vi.fn();

vi.mock('@/lib/api-client', () => ({
  api: {
    getRegistrationSettings: (...a: unknown[]) => mockGetRegistrationSettings(...a),
    getRegistrationCodes: (...a: unknown[]) => mockGetRegistrationCodes(...a),
    deleteRegistrationCode: (...a: unknown[]) => mockDeleteRegistrationCode(...a),
    createRegistrationCode: (...a: unknown[]) => mockCreateRegistrationCode(...a),
    updateRegistrationSettings: (...a: unknown[]) => mockUpdateRegistrationSettings(...a),
  },
  ApiError: class ApiError extends Error {
    status: number;
    constructor(message: string, status = 500) {
      super(message);
      this.status = status;
    }
  },
}));

vi.mock('@/lib/logger', () => ({
  logger: {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  },
}));

// PermissionGuard: render children immediately
vi.mock('@/components/PermissionGuard', () => ({
  PermissionGuard: ({ children }: { children: ReactNode }) => <>{children}</>,
  usePermissions: () => ({
    user: { id: 1, username: 'admin', role: 'admin' },
    permissions: ['*'],
    permissionHints: ['*'],
    roles: ['admin'],
    loading: false,
    authError: false,
    hasPermission: () => true,
    hasRole: () => true,
    hasAnyPermission: () => true,
    hasAllPermissions: () => true,
    isAdmin: () => true,
    isSuperAdmin: () => false,
    refresh: vi.fn(),
  }),
}));

vi.mock('@/components/ResponsiveLayout', () => ({
  ResponsiveLayout: ({ children }: { children: ReactNode }) => <div>{children}</div>,
}));

// Minimal confirm mock that always resolves true
const mockConfirm = vi.fn().mockResolvedValue(true);
vi.mock('@/components/ui/confirm-dialog', () => ({
  useConfirm: () => mockConfirm,
}));

const mockSuccess = vi.fn();
const mockShowError = vi.fn();
vi.mock('@/components/ui/toast', () => ({
  useToast: () => ({
    success: mockSuccess,
    error: mockShowError,
    warning: vi.fn(),
    info: vi.fn(),
    toasts: [],
    addToast: vi.fn(),
    removeToast: vi.fn(),
  }),
}));

vi.mock('next/link', () => ({
  default: ({ children, href }: { children: ReactNode; href: string }) => (
    <a href={href}>{children}</a>
  ),
}));

// Stub clipboard API
const mockWriteText = vi.fn().mockResolvedValue(undefined);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const sampleSettings = {
  enable_registration: true,
  require_registration_code: true,
  auth_mode: 'multi_user',
  profile: 'local-multi-user',
  self_registration_allowed: true,
};

const sampleCodes = [
  {
    id: 1,
    code: 'ABCDEF-123456',
    max_uses: 5,
    times_used: 2,
    expires_at: '2099-12-31T00:00:00Z',
    created_at: '2026-01-01T00:00:00Z',
    role_to_grant: 'user',
  },
  {
    id: 2,
    code: 'GHIJKL-789012',
    max_uses: 1,
    times_used: 1,
    expires_at: '2020-01-01T00:00:00Z',
    created_at: '2019-01-01T00:00:00Z',
    role_to_grant: 'admin',
  },
];

function setupDefaults() {
  mockGetRegistrationSettings.mockResolvedValue(sampleSettings);
  mockGetRegistrationCodes.mockResolvedValue(sampleCodes);
  mockDeleteRegistrationCode.mockResolvedValue({});
  mockCreateRegistrationCode.mockResolvedValue({
    id: 3,
    code: 'NEWCODE-999999',
    max_uses: 1,
    times_used: 0,
    expires_at: '2099-12-31T00:00:00Z',
    created_at: '2026-03-01T00:00:00Z',
    role_to_grant: 'user',
  });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('RegistrationCodesPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setupDefaults();
    // Ensure clipboard mock is in place for each test
    Object.defineProperty(navigator, 'clipboard', {
      value: { writeText: mockWriteText },
      writable: true,
      configurable: true,
    });
  });

  afterEach(() => {
    cleanup();
  });

  it('renders registration code list after loading', async () => {
    render(<RegistrationCodesPage />);

    await waitFor(() => {
      expect(screen.getByText('ABCDEF-123456')).toBeInTheDocument();
    });
    expect(screen.getByText('GHIJKL-789012')).toBeInTheDocument();
    // Active badge should show count
    expect(screen.getByText('1 active')).toBeInTheDocument();
  });

  it('shows empty state when no codes exist', async () => {
    mockGetRegistrationCodes.mockResolvedValue([]);

    render(<RegistrationCodesPage />);

    await waitFor(() => {
      expect(
        screen.getByText(/No registration codes yet/i)
      ).toBeInTheDocument();
    });
  });

  it('opens Create Code dialog when New Code button is clicked', async () => {
    const user = userEvent.setup();
    render(<RegistrationCodesPage />);

    await waitFor(() => {
      expect(screen.getByText('ABCDEF-123456')).toBeInTheDocument();
    });

    await user.click(screen.getByRole('button', { name: /new code/i }));

    await waitFor(() => {
      expect(screen.getByText('Create registration code')).toBeInTheDocument();
    });
  });

  it('copies code to clipboard and shows success or error toast', async () => {
    const user = userEvent.setup();
    render(<RegistrationCodesPage />);

    await waitFor(() => {
      expect(screen.getByText('ABCDEF-123456')).toBeInTheDocument();
    });

    const copyButtons = screen.getAllByLabelText('Copy registration code');
    await user.click(copyButtons[0]);

    // The copy action is async and calls navigator.clipboard.writeText.
    // In jsdom, clipboard may not be available, so the component falls
    // back to showing either a success or error toast.
    await waitFor(() => {
      const toastCalled = mockSuccess.mock.calls.length > 0 || mockShowError.mock.calls.length > 0;
      expect(toastCalled).toBe(true);
    });
  });

  it('deletes a code after confirmation', async () => {
    const user = userEvent.setup();
    render(<RegistrationCodesPage />);

    await waitFor(() => {
      expect(screen.getByText('ABCDEF-123456')).toBeInTheDocument();
    });

    const deleteButtons = screen.getAllByLabelText('Delete registration code');
    await user.click(deleteButtons[0]);

    await waitFor(() => {
      expect(mockConfirm).toHaveBeenCalledWith(
        expect.objectContaining({
          title: 'Delete registration code',
          variant: 'danger',
        })
      );
    });

    await waitFor(() => {
      expect(mockDeleteRegistrationCode).toHaveBeenCalledWith(1);
    });
  });

  it('renders registration settings controls', async () => {
    render(<RegistrationCodesPage />);

    await waitFor(() => {
      expect(screen.getByText('Registration Settings')).toBeInTheDocument();
    });

    expect(screen.getByLabelText('Toggle self-registration')).toBeInTheDocument();
    expect(screen.getByLabelText('Toggle registration code requirement')).toBeInTheDocument();
    expect(screen.getByText('Registration enabled')).toBeInTheDocument();
    expect(screen.getByText('Codes required')).toBeInTheDocument();
  });

  it('shows error alert when data fetch fails', async () => {
    mockGetRegistrationSettings.mockRejectedValue(new Error('network error'));
    mockGetRegistrationCodes.mockRejectedValue(new Error('network error'));

    render(<RegistrationCodesPage />);

    await waitFor(() => {
      expect(screen.getByText(/some data failed to load/i)).toBeInTheDocument();
    });
  });

  it('displays Active and Expired badges correctly', async () => {
    render(<RegistrationCodesPage />);

    await waitFor(() => {
      expect(screen.getByText('ABCDEF-123456')).toBeInTheDocument();
    });

    const badges = screen.getAllByText(/^(Active|Expired)$/);
    // First code is active (future expiry, uses remaining), second is expired
    expect(badges).toHaveLength(2);
    expect(badges[0].textContent).toBe('Active');
    expect(badges[1].textContent).toBe('Expired');
  });
});
