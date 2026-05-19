'use client';

import { createContext, useCallback, useContext, useEffect, useState, ReactNode } from 'react';
import { Building2, ChevronDown, Check } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { api } from '@/lib/api-client';
import { Organization } from '@/types';
import { usePermissions } from '@/components/PermissionGuard';
import { getScopedItem, setScopedItem, removeScopedItem } from '@/lib/scoped-storage';
import { logger } from '@/lib/logger';

interface OrgContextType {
  organizations: Organization[];
  selectedOrg: Organization | null;
  setSelectedOrg: (org: Organization | null) => void;
  loading: boolean;
  isOrgScoped: boolean;
  refresh: () => Promise<void>;
}

const OrgContext = createContext<OrgContextType>({
  organizations: [],
  selectedOrg: null,
  setSelectedOrg: () => {},
  loading: true,
  isOrgScoped: false,
  refresh: async () => {},
});

const ORG_SELECTION_STORAGE_KEY = 'selectedOrgId';

export function useOrgContext() {
  return useContext(OrgContext);
}

interface OrgContextProviderProps {
  children: ReactNode;
}

export function OrgContextProvider({ children }: OrgContextProviderProps) {
  const [organizations, setOrganizations] = useState<Organization[]>([]);
  const [selectedOrg, setSelectedOrg] = useState<Organization | null>(null);
  const [loading, setLoading] = useState(true);
  const { isSuperAdmin, user, loading: permLoading } = usePermissions();

  // Org-scoped means the user is NOT a super admin but might be an org admin
  const isOrgScoped = !permLoading && !isSuperAdmin() && user !== null;

  const handleSetSelectedOrg = useCallback((org: Organization | null) => {
    setSelectedOrg(org);
    try {
      if (org) {
        setScopedItem(ORG_SELECTION_STORAGE_KEY, String(org.id));
      } else {
        removeScopedItem(ORG_SELECTION_STORAGE_KEY);
      }
    } catch (error) {
      logger.warn('Failed to persist org selection', { component: 'OrgContextProvider', error: error instanceof Error ? error.message : String(error) });
    }
  }, []);

  const loadOrganizations = useCallback(async () => {
    try {
      setLoading(true);
      const orgs = await api.getOrganizations();
      const orgList = Array.isArray(orgs) ? orgs : [];
      setOrganizations(orgList);
    } catch (error) {
      logger.error('Failed to load organizations', { component: 'OrgContextProvider', error: error instanceof Error ? error.message : String(error) });
      setOrganizations([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (organizations.length === 0 || selectedOrg) {
      return;
    }
    let storedId: number | null = null;
    try {
      const stored = getScopedItem(ORG_SELECTION_STORAGE_KEY);
      if (stored) {
        const parsed = Number.parseInt(stored, 10);
        if (!Number.isNaN(parsed)) {
          storedId = parsed;
        }
      }
    } catch (error) {
      logger.warn('Failed to load persisted org selection', { component: 'OrgContextProvider', error: error instanceof Error ? error.message : String(error) });
    }

    if (storedId !== null) {
      const storedOrg = organizations.find((org) => org.id === storedId);
      if (storedOrg) {
        handleSetSelectedOrg(storedOrg);
        return;
      }
      try {
        removeScopedItem(ORG_SELECTION_STORAGE_KEY);
      } catch (error) {
        logger.warn('Failed to clear invalid org selection', { component: 'OrgContextProvider', error: error instanceof Error ? error.message : String(error) });
      }
    }

    if (isOrgScoped) {
      handleSetSelectedOrg(organizations[0]);
    }
  }, [handleSetSelectedOrg, isOrgScoped, organizations, selectedOrg]);

  useEffect(() => {
    if (permLoading) {
      return;
    }
    if (!user) {
      setOrganizations([]);
      setSelectedOrg(null);
      setLoading(false);
      return;
    }
    void loadOrganizations();
  }, [loadOrganizations, permLoading, user]);

  return (
    <OrgContext.Provider
      value={{
        organizations,
        selectedOrg,
        setSelectedOrg: handleSetSelectedOrg,
        loading,
        isOrgScoped,
        refresh: loadOrganizations,
      }}
    >
      {children}
    </OrgContext.Provider>
  );
}

// Visual component for switching org context
interface OrgContextSwitcherProps {
  className?: string;
}

export function OrgContextSwitcher({ className = '' }: OrgContextSwitcherProps) {
  const { organizations, selectedOrg, setSelectedOrg, loading } = useOrgContext();
  const { isSuperAdmin } = usePermissions();

  // Org-scoped (non-super-admin) users see a read-only badge showing their org
  if (!isSuperAdmin()) {
    if (loading) {
      return (
        <div className={`flex min-w-0 items-center gap-2 px-3 py-2 ${className}`}>
          <Building2 className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm text-muted-foreground">Loading...</span>
        </div>
      );
    }
    if (!selectedOrg) {
      return null;
    }
    return (
      <div className={`flex min-w-0 items-center gap-2 px-3 py-2 ${className}`} data-testid="org-badge">
        <Building2 className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
        <Badge variant="secondary" className="cursor-default hover:bg-secondary truncate max-w-[140px]" title={selectedOrg.name}>
          {selectedOrg.name}
        </Badge>
      </div>
    );
  }

  // Super admins see all orgs and can choose to scope to one
  if (loading) {
    return (
      <div className={`flex items-center gap-2 px-3 py-2 ${className}`}>
        <Building2 className="h-4 w-4 text-muted-foreground" />
        <span className="text-sm text-muted-foreground">Loading...</span>
      </div>
    );
  }

  if (organizations.length === 0) {
    return null;
  }

  return (
    <div className={className}>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="ghost" size="sm" className="group w-full justify-between">
            <div className="flex items-center gap-2">
              <Building2 className="h-4 w-4" />
              <span className="truncate max-w-[120px]">
                {selectedOrg ? selectedOrg.name : 'All Organizations'}
              </span>
            </div>
            <ChevronDown className="h-4 w-4 transition-transform group-data-[state=open]:rotate-180" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start" className="min-w-[200px]">
          {isSuperAdmin() && (
            <DropdownMenuItem
              className="flex items-center gap-2"
              onSelect={() => setSelectedOrg(null)}
            >
              <Building2 className="h-4 w-4" />
              <span className="flex-1">All Organizations</span>
              {!selectedOrg && <Check className="h-4 w-4 text-primary" />}
            </DropdownMenuItem>
          )}

          {isSuperAdmin() && organizations.length > 0 && <DropdownMenuSeparator />}

          {organizations.map((org) => (
            <DropdownMenuItem
              key={org.id}
              className="flex items-center gap-2"
              onSelect={() => setSelectedOrg(org)}
            >
              <Building2 className="h-4 w-4" />
              <span className="flex-1 truncate">{org.name}</span>
              {selectedOrg?.id === org.id && <Check className="h-4 w-4 text-primary" />}
            </DropdownMenuItem>
          ))}
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  );
}

// Helper hook to get filtered data based on org context
export function useOrgFilteredData<T extends { org_id?: number }>(data: T[]): T[] {
  const { selectedOrg } = useOrgContext();

  if (!selectedOrg) {
    return data;
  }

  return data.filter((item) => item.org_id === selectedOrg.id);
}

// Banner component showing the active org context near page headers
export function OrgContextBanner() {
  const { selectedOrg } = useOrgContext();

  if (!selectedOrg) {
    return null;
  }

  return (
    <div
      data-testid="org-context-banner"
      className="inline-flex items-center gap-1.5 rounded-md border border-border bg-muted/50 px-2.5 py-1 text-xs text-muted-foreground"
    >
      <Building2 className="h-3 w-3" aria-hidden="true" />
      <span>Viewing: <span className="font-medium text-foreground">{selectedOrg.name}</span></span>
    </div>
  );
}

export default OrgContextSwitcher;
