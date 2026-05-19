'use client';

import { createContext, Suspense, useCallback, useContext, useEffect, useState, ReactNode } from 'react';
import { usePathname, useRouter, useSearchParams } from 'next/navigation';
import { api, ApiError } from '@/lib/api-client';
import { hasStoredAuth, subscribeAuthChange } from '@/lib/auth';
import { resolveUnauthenticatedRouteState } from '@/lib/auth-navigation';
import { User } from '@/types';
import { getRoleRank, hasRoleAccess, isAdminRole, isMemberRole, isSuperAdminRole } from '@/lib/roles';
import { logger } from '@/lib/logger';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { CardSkeleton } from '@/components/ui/skeleton';

// UI gating only; backend must enforce authorization and never trust client permissions.
interface PermissionContextType {
  user: User | null;
  permissions: string[];
  permissionHints: string[];
  roles: string[];
  loading: boolean;
  authError: boolean;
  hasPermission: (permission: string | string[]) => boolean;
  hasRole: (role: string | string[]) => boolean;
  hasAnyPermission: (permissions: string[]) => boolean;
  hasAllPermissions: (permissions: string[]) => boolean;
  isAdmin: () => boolean;
  isSuperAdmin: () => boolean;
  refresh: () => Promise<void>;
}

const PermissionContext = createContext<PermissionContextType>({
  user: null,
  permissions: [],
  permissionHints: [],
  roles: [],
  loading: true,
  authError: false,
  hasPermission: () => false,
  hasRole: () => false,
  hasAnyPermission: () => false,
  hasAllPermissions: () => false,
  isAdmin: () => false,
  isSuperAdmin: () => false,
  refresh: async () => {},
});

export function usePermissions() {
  return useContext(PermissionContext);
}

const normalizeRoleValue = (role: unknown): string =>
  typeof role === 'string' ? role.trim().toLowerCase() : '';

const normalizeRoles = (userData: User): string[] => {
  const extraRoles = Array.isArray(userData.roles) ? userData.roles : [];
  const combined = [userData.role, ...extraRoles]
    .map((role) => normalizeRoleValue(role))
    .filter((role) => role.length > 0);
  return Array.from(new Set(combined));
};

const normalizeRoleInput = (role: string | string[]): string[] => {
  const roles = Array.isArray(role) ? role : [role];
  return roles.map((entry) => normalizeRoleValue(entry)).filter((entry) => entry.length > 0);
};

const getHighestRole = (roles: string[]): string | undefined => {
  let bestRole: string | undefined;
  let bestRank = -1;
  for (const role of roles) {
    const rank = getRoleRank(role);
    if (rank !== undefined && rank > bestRank) {
      bestRank = rank;
      bestRole = role;
    }
  }
  return bestRole;
};

interface PermissionProviderProps {
  children: ReactNode;
}

export function PermissionProvider({ children }: PermissionProviderProps) {
  const [user, setUser] = useState<User | null>(null);
  const [permissions, setPermissions] = useState<string[]>([]);
  const [permissionHints, setPermissionHints] = useState<string[]>([]);
  const [roles, setRoles] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [authError, setAuthError] = useState(false);

  const clearAuthState = useCallback((didAuthFail: boolean) => {
    setUser(null);
    setPermissions([]);
    setPermissionHints([]);
    setRoles([]);
    setAuthError(didAuthFail);
    setLoading(false);
  }, []);

  const loadUserPermissions = useCallback(async () => {
    try {
      setLoading(true);
      setAuthError(false);
      const userData = await api.getCurrentUser();
      setUser(userData);

      // UI-only role hints. Backend authorization must enforce real permissions.
      const userRoles = normalizeRoles(userData);
      setRoles(userRoles);
      const userRole = getHighestRole(userRoles) ?? userRoles[0] ?? '';

      const derivedPermissions: string[] = [];

      if (isSuperAdminRole(userRole)) {
        derivedPermissions.push('*'); // Wildcard for all permissions
      } else if (isAdminRole(userRole)) {
        derivedPermissions.push(
          'read:users', 'write:users',
          'read:orgs', 'write:orgs',
          'read:teams', 'write:teams',
          'read:api_keys', 'write:api_keys',
          'read:audit', 'read:config'
        );
      } else if (isMemberRole(userRole)) {
        derivedPermissions.push('read:users', 'read:orgs', 'read:teams');
      }

      setPermissionHints(derivedPermissions);

      // NOTE: UI gating only. Backend must enforce authorization.
      try {
        const effective = await api.getUserEffectivePermissions(userData.id.toString());
        const serverPermissions = Array.isArray(effective?.permissions) ? effective.permissions : [];
        setPermissions(serverPermissions);
      } catch (error) {
        logger.error('Failed to load server permissions', { component: 'PermissionProvider', error: error instanceof Error ? error.message : String(error) });
        setPermissions([]);
      }
    } catch (error) {
      logger.error('Failed to load user permissions', { component: 'PermissionProvider', error: error instanceof Error ? error.message : String(error) });
      // Only set authError for 401/403 to trigger login redirect.
      // Other errors (network, 5xx, etc.) should show loading state, not redirect.
      const status = error instanceof ApiError
        ? error.status
        : (error as { response?: { status?: number } })?.response?.status;
      setAuthError(status === 401 || status === 403);
      setUser(null);
      setPermissions([]);
      setPermissionHints([]);
      setRoles([]);
    } finally {
      setLoading(false);
    }
  }, []);

  const refreshPermissions = useCallback(async () => {
    if (!hasStoredAuth()) {
      clearAuthState(true);
      return;
    }
    await loadUserPermissions();
  }, [clearAuthState, loadUserPermissions]);

  useEffect(() => {
    void refreshPermissions();
  }, [refreshPermissions]);

  useEffect(() => {
    return subscribeAuthChange(() => {
      void refreshPermissions();
    });
  }, [refreshPermissions]);

  const hasPermission = (permission: string | string[]): boolean => {
    if (permissions.includes('*')) return true;

    const perms = Array.isArray(permission) ? permission : [permission];
    return perms.some((p) => permissions.includes(p));
  };

  const hasRole = (role: string | string[]): boolean => {
    const requiredRoles = normalizeRoleInput(role);
    if (requiredRoles.length === 0 || roles.length === 0) {
      return false;
    }
    return roles.some((userRole) => hasRoleAccess(userRole, requiredRoles));
  };

  const hasAnyPermission = (perms: string[]): boolean => {
    if (permissions.includes('*')) return true;
    return perms.some((p) => permissions.includes(p));
  };

  const hasAllPermissions = (perms: string[]): boolean => {
    if (permissions.includes('*')) return true;
    return perms.every((p) => permissions.includes(p));
  };

  const isAdmin = (): boolean => roles.some((role) => isAdminRole(role));

  const isSuperAdmin = (): boolean => roles.some((role) => isSuperAdminRole(role));

  return (
    <PermissionContext.Provider
      value={{
        user,
        permissions,
        permissionHints,
        roles,
        loading,
        authError,
        hasPermission,
        hasRole,
        hasAnyPermission,
        hasAllPermissions,
        isAdmin,
        isSuperAdmin,
        refresh: loadUserPermissions,
      }}
    >
      {children}
    </PermissionContext.Provider>
  );
}

// Component to conditionally render based on permissions
interface PermissionGuardProps {
  children: ReactNode;
  permission?: string | string[];
  permissions?: string[];
  requireAll?: boolean;
  role?: string | string[];
  fallback?: ReactNode;
  showLoading?: boolean;
  requireAuth?: boolean;
  variant?: 'inline' | 'route';
}

type GuardDecision = 'loading' | 'unauthenticated' | 'denied' | 'allowed';

const usePermissionGuardDecision = ({
  permission,
  permissions: requiredPermissions,
  requireAll,
  role,
  requireAuth,
}: PermissionGuardProps) => {
  const {
    hasPermission,
    hasAnyPermission,
    hasAllPermissions,
    hasRole,
    loading,
    user,
    authError,
  } = usePermissions();

  if (loading) {
    return { decision: 'loading' as GuardDecision, user, authError };
  }

  if ((requireAuth ?? false) && !user) {
    return { decision: 'unauthenticated' as GuardDecision, user, authError };
  }

  if (role && !hasRole(role)) {
    return { decision: 'denied' as GuardDecision, user, authError };
  }

  if (permission && !hasPermission(permission)) {
    return { decision: 'denied' as GuardDecision, user, authError };
  }

  if (requiredPermissions && requiredPermissions.length > 0) {
    const requireAllPermissions = requireAll ?? false;
    const hasAccess = requireAllPermissions
      ? hasAllPermissions(requiredPermissions)
      : hasAnyPermission(requiredPermissions);
    if (!hasAccess) {
      return { decision: 'denied' as GuardDecision, user, authError };
    }
  }

  return { decision: 'allowed' as GuardDecision, user, authError };
};

const renderRouteLoading = () => (
  <div className="flex h-screen items-center justify-center" role="status" aria-live="polite" aria-atomic="true">
    <span className="sr-only">Loading protected page</span>
    <CardSkeleton />
  </div>
);

const renderRouteDenied = () => (
  <div className="flex h-screen items-center justify-center p-8">
    <Alert variant="destructive" className="max-w-md">
      <AlertDescription>
        You do not have permission to access this page.
      </AlertDescription>
    </Alert>
  </div>
);

const renderRouteSessionUnavailable = (onRetry: () => void, onLogin: () => void) => (
  <div className="flex h-screen items-center justify-center p-8">
    <Alert className="max-w-md">
      <AlertDescription className="space-y-4">
        <p>Unable to verify your session right now. Check your connection and try again.</p>
        <div className="flex flex-wrap gap-2">
          <Button type="button" variant="outline" onClick={onRetry}>
            Retry
          </Button>
          <Button type="button" onClick={onLogin}>
            Sign in again
          </Button>
        </div>
      </AlertDescription>
    </Alert>
  </div>
);

function RoutePermissionGuard(props: PermissionGuardProps) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const { decision, user, authError } = usePermissionGuardDecision(props);
  const requireAuth = props.requireAuth ?? false;
  const shouldAuthRedirect = requireAuth && !user && authError;
  const search = searchParams?.toString();
  const redirectTo = search ? `${pathname}?${search}` : pathname;

  useEffect(() => {
    if (!shouldAuthRedirect) {
      return;
    }
    router.replace(`/login?redirectTo=${encodeURIComponent(redirectTo)}`);
  }, [router, shouldAuthRedirect, redirectTo]);

  if (decision === 'loading') {
    return renderRouteLoading();
  }

  if (decision === 'unauthenticated') {
    const routeState = resolveUnauthenticatedRouteState(authError);
    if (routeState === 'redirect_to_login') {
      return (
        <div className="flex h-screen items-center justify-center">
          <div className="text-muted-foreground">Redirecting to login...</div>
        </div>
      );
    }
    return renderRouteSessionUnavailable(
      () => {
        if (typeof window !== 'undefined') {
          window.location.reload();
        }
      },
      () => router.replace(`/login?redirectTo=${encodeURIComponent(redirectTo)}`)
    );
  }

  if (decision === 'denied') {
    return renderRouteDenied();
  }

  return <>{props.children}</>;
}

function InlinePermissionGuard(props: PermissionGuardProps) {
  const { decision } = usePermissionGuardDecision(props);

  if (decision === 'loading') {
    if (props.showLoading) {
      return <div className="animate-pulse bg-muted h-8 rounded" />;
    }
    return null;
  }

  if (decision === 'unauthenticated' || decision === 'denied') {
    return <>{props.fallback ?? null}</>;
  }

  return <>{props.children}</>;
}

export function PermissionGuard({
  children,
  permission,
  permissions: requiredPermissions,
  requireAll = false,
  role,
  fallback = null,
  showLoading = false,
  requireAuth = false,
  variant = 'inline',
}: PermissionGuardProps) {
  const resolvedProps: PermissionGuardProps = {
    children,
    permission,
    permissions: requiredPermissions,
    requireAll,
    role,
    fallback,
    showLoading,
    requireAuth,
    variant,
  };

  if (variant === 'route') {
    // Wrap in Suspense for useSearchParams (Next.js 15 requirement)
    return (
      <Suspense fallback={renderRouteLoading()}>
        <RoutePermissionGuard {...resolvedProps} />
      </Suspense>
    );
  }

  return <InlinePermissionGuard {...resolvedProps} />;
}

// Higher-order component for protecting entire pages
interface WithPermissionOptions {
  permission?: string | string[];
  role?: string | string[];
  redirectTo?: string;
}

export function withPermission<P extends object>(
  WrappedComponent: React.ComponentType<P>,
  options: WithPermissionOptions
) {
  return function PermissionProtectedComponent(props: P) {
    const router = useRouter();
    const { hasPermission, hasRole, loading } = usePermissions();
    const [mounted, setMounted] = useState(false);

    useEffect(() => {
      setMounted(true);
    }, []);

    if (!mounted || loading) {
      return (
        <div className="flex items-center justify-center min-h-screen">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary" />
        </div>
      );
    }

    const hasAccess =
      (!options.permission || hasPermission(options.permission)) &&
      (!options.role || hasRole(options.role));

    if (!hasAccess) {
      if (options.redirectTo) {
        router.push(options.redirectTo);
        return null;
      }

      return (
        <div className="flex flex-col items-center justify-center min-h-screen">
          <h1 className="text-2xl font-bold mb-4">Access Denied</h1>
          <p className="text-muted-foreground">
            You don&apos;t have permission to access this page.
          </p>
        </div>
      );
    }

    return <WrappedComponent {...props} />;
  };
}

// Utility component for admin-only content
interface AdminOnlyProps {
  children: ReactNode;
  fallback?: ReactNode;
}

export function AdminOnly({ children, fallback = null }: AdminOnlyProps) {
  return (
    <PermissionGuard role="admin" fallback={fallback}>
      {children}
    </PermissionGuard>
  );
}

// Utility component for super-admin-only content
export function SuperAdminOnly({ children, fallback = null }: AdminOnlyProps) {
  return (
    <PermissionGuard role="super_admin" fallback={fallback}>
      {children}
    </PermissionGuard>
  );
}

export default PermissionGuard;
