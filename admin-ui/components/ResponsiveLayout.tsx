'use client';

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  ReactNode,
} from 'react';
import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import { LogOut, Menu, Search, X } from 'lucide-react';
import { isSingleUserMode, logout } from '@/lib/auth';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { cn } from '@/lib/utils';
import { logger } from '@/lib/logger';
import { ThemeToggle } from '@/components/ThemeToggle';
import { Breadcrumbs } from '@/components/Breadcrumbs';
import { OrgContextSwitcher, OrgContextBanner } from '@/components/OrgContextSwitcher';
import { usePermissions } from '@/components/PermissionGuard';
import { useToast } from '@/components/ui/toast';
import { isBillingEnabled } from '@/lib/billing';
import {
  buildBreadcrumbs,
  getPageTitleForPath,
  matchesNavigationQuery,
  navigationSections,
  type NavigationItem,
} from '@/lib/navigation';

// Mobile menu context
interface MobileMenuContextType {
  isOpen: boolean;
  open: () => void;
  close: () => void;
  toggle: () => void;
}

const MobileMenuContext = createContext<MobileMenuContextType>({
  isOpen: false,
  open: () => {},
  close: () => {},
  toggle: () => {},
});

const SHORTCUT_BANNER_STORAGE_KEY_PREFIX = 'admin_shortcuts_tip_dismissed_v1';

export function useMobileMenu() {
  return useContext(MobileMenuContext);
}

// Mobile header with hamburger menu
function MobileHeader() {
  const { toggle, isOpen } = useMobileMenu();

  return (
    <div className="lg:hidden fixed top-0 left-0 right-0 z-40 flex h-14 items-center justify-between border-b bg-card px-4">
      <Button
        variant="ghost"
        size="icon"
        onClick={toggle}
        aria-label={isOpen ? 'Close navigation menu' : 'Open navigation menu'}
        aria-expanded={isOpen}
        aria-controls="mobile-navigation"
      >
        <Menu className="h-5 w-5" aria-hidden="true" />
      </Button>
      <h1 className="text-lg font-bold">tldw Admin</h1>
      <ThemeToggle />
    </div>
  );
}

// Sidebar content (shared between mobile and desktop)
function SidebarContent({ onNavigate }: { onNavigate?: () => void }) {
  const pathname = usePathname();
  const router = useRouter();
  const { user, hasPermission, hasRole, loading: permLoading, refresh } = usePermissions();
  const { error: showError } = useToast();
  const [navQuery, setNavQuery] = useState('');
  const searchInputId = useId();

  const handleLogout = async () => {
    try {
      await logout();
      await refresh();
      router.push('/login');
    } catch (error) {
      logger.error('Logout failed', { component: 'SidebarContent', error: error instanceof Error ? error.message : String(error) });
      showError('Logout failed', 'Please try again.');
    }
  };

  const handleNavClick = () => {
    if (onNavigate) onNavigate();
  };

  // Filter items based on permissions and billing
  const isItemVisible = useCallback((item: NavigationItem) => {
    // Filter out billing-only items when billing is disabled
    if (item.billingOnly && !isBillingEnabled()) return false;
    if (!item.permission && !item.role) return true;
    if (permLoading) return false;
    if (item.href === '/debug' && isSingleUserMode() && hasRole('admin')) return true;
    if (item.permission && hasPermission(item.permission)) return true;
    if (item.role && hasRole(item.role)) return true;
    return false;
  }, [hasPermission, hasRole, permLoading]);

  // Get visible sections with visible items
  const visibleSections = useMemo(
    () =>
      navigationSections
        .map((section) => ({
          ...section,
          items: section.items.filter(
            (item) => isItemVisible(item) && matchesNavigationQuery(item, section.title, navQuery)
          ),
        }))
        .filter((section) => section.items.length > 0),
    [isItemVisible, navQuery]
  );

  return (
    <>
      {/* Header - hidden on mobile since we have MobileHeader */}
      <div className="hidden lg:flex h-16 items-center justify-between border-b px-6">
        <h1 className="text-xl font-bold">tldw Admin</h1>
        <ThemeToggle />
      </div>

      {/* User info */}
      {user && (
        <div className="border-b px-4 py-3">
          <p className="text-sm font-medium truncate">{user.username || user.email}</p>
          <p className="text-xs text-muted-foreground capitalize">{user.role}</p>
        </div>
      )}

      {/* Org Context Switcher */}
      <div className="border-b px-3 py-2">
        <OrgContextSwitcher />
      </div>

      <div className="border-b px-3 py-2">
        <label htmlFor={searchInputId} className="sr-only">
          Find navigation page
        </label>
        <div className="relative">
          <Search className="absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" aria-hidden="true" />
          <Input
            id={searchInputId}
            value={navQuery}
            onChange={(event) => setNavQuery(event.target.value)}
            placeholder="Find page..."
            className="h-9 pl-8"
            aria-label="Find navigation page"
          />
        </div>
      </div>

      {/* Navigation with sections */}
      <nav className="flex-1 px-3 py-4 overflow-y-auto" aria-label="Main navigation">
        {visibleSections.length === 0 ? (
          <p className="px-3 py-2 text-sm text-muted-foreground">
            No navigation matches your search.
          </p>
        ) : (
          visibleSections.map((section, sectionIndex) => (
            <div key={section.title} className={sectionIndex > 0 ? 'mt-6' : ''}>
              <h3 className="px-3 mb-2 text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                {section.title}
              </h3>
              <div className="space-y-1">
                {section.items.map((item) => {
                  const Icon = item.icon;
                  const isActive = pathname === item.href ||
                    (item.href !== '/' && pathname.startsWith(item.href));

                  return (
                    <Link
                      key={item.name}
                      href={item.href}
                      onClick={handleNavClick}
                      className={cn(
                        'flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors',
                        isActive
                          ? 'bg-primary/10 text-primary border border-primary/20'
                          : 'text-foreground hover:bg-muted'
                      )}
                      aria-current={isActive ? 'page' : undefined}
                    >
                      <Icon className="h-5 w-5 flex-shrink-0" aria-hidden="true" />
                      <span className="truncate">{item.name}</span>
                    </Link>
                  );
                })}
              </div>
            </div>
          ))
        )}
      </nav>

      {/* Logout */}
      <div className="border-t px-3 py-4">
        <p className="mb-3 text-xs text-muted-foreground" data-testid="sidebar-shortcuts-hint">
          Press ? for shortcuts
        </p>
        <Button
          variant="outline"
          className="w-full justify-start gap-3"
          onClick={handleLogout}
        >
          <LogOut className="h-5 w-5" />
          Logout
        </Button>
      </div>
    </>
  );
}

// Desktop sidebar
function DesktopSidebar() {
  return (
    <div className="hidden lg:flex h-screen w-64 flex-col bg-card border-r flex-shrink-0">
      <SidebarContent />
    </div>
  );
}

// Mobile sidebar (slide-out drawer)
function MobileSidebar() {
  const { isOpen, close } = useMobileMenu();
  const closeButtonRef = useRef<HTMLButtonElement | null>(null);
  const drawerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!isOpen) {
      return undefined;
    }

    const previousFocus = document.activeElement instanceof HTMLElement
      ? document.activeElement
      : null;
    const focusableSelector = [
      'a[href]',
      'button:not([disabled])',
      'textarea:not([disabled])',
      'input:not([disabled])',
      'select:not([disabled])',
      '[tabindex]:not([tabindex="-1"])',
    ].join(',');
    const getFocusableElements = () =>
      Array.from(drawerRef.current?.querySelectorAll<HTMLElement>(focusableSelector) ?? []);

    const focusTimer = window.setTimeout(() => {
      const nextFocus = closeButtonRef.current ?? getFocusableElements()[0] ?? drawerRef.current;
      nextFocus?.focus();
    }, 0);

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        close();
        return;
      }
      if (event.key !== 'Tab') return;

      const focusableElements = getFocusableElements();
      if (focusableElements.length === 0) {
        event.preventDefault();
        drawerRef.current?.focus();
        return;
      }

      const firstElement = focusableElements[0];
      const lastElement = focusableElements[focusableElements.length - 1];
      const activeElement = document.activeElement as HTMLElement | null;

      if (event.shiftKey && activeElement === firstElement) {
        event.preventDefault();
        lastElement.focus();
        return;
      }
      if (!event.shiftKey && activeElement === lastElement) {
        event.preventDefault();
        firstElement.focus();
      }
    };

    document.addEventListener('keydown', handleKeyDown);

    return () => {
      window.clearTimeout(focusTimer);
      document.removeEventListener('keydown', handleKeyDown);
      previousFocus?.focus();
    };
  }, [close, isOpen]);

  if (!isOpen) {
    return null;
  }

  return (
    <>
      {/* Backdrop */}
      <div
        className="lg:hidden fixed inset-0 z-40 bg-black/50"
        onClick={close}
        aria-hidden="true"
      />

      {/* Drawer */}
      <div
        id="mobile-navigation"
        ref={drawerRef}
        className="lg:hidden fixed inset-y-0 left-0 z-50 w-64 bg-card border-r"
        role="dialog"
        aria-modal="true"
        aria-label="Main navigation"
        tabIndex={-1}
      >
        {/* Close button */}
        <div className="flex h-14 items-center justify-between border-b px-4">
          <h1 className="text-lg font-bold">tldw Admin</h1>
          <Button
            variant="ghost"
            size="icon"
            onClick={close}
            aria-label="Close navigation menu"
            ref={closeButtonRef}
          >
            <X className="h-5 w-5" aria-hidden="true" />
          </Button>
        </div>

        <div className="flex flex-col h-[calc(100%-3.5rem)]">
          <SidebarContent onNavigate={close} />
        </div>
      </div>
    </>
  );
}

// Main responsive layout component
interface ResponsiveLayoutProps {
  children: ReactNode;
}

export function ResponsiveLayout({ children }: ResponsiveLayoutProps) {
  const [isOpen, setIsOpen] = useState(false);
  const pathname = usePathname();
  const { user, loading: permissionsLoading } = usePermissions();
  const prevPathnameRef = useRef(pathname);
  const focusedPathnameRef = useRef(pathname);
  const [dismissedShortcutBannerKeys, setDismissedShortcutBannerKeys] = useState<Set<string>>(new Set());
  const openMenu = useCallback(() => setIsOpen(true), []);
  const closeMenu = useCallback(() => setIsOpen(false), []);
  const toggleMenu = useCallback(() => setIsOpen((prev) => !prev), []);
  const breadcrumbs = useMemo(() => buildBreadcrumbs(pathname || '/'), [pathname]);
  const showBreadcrumbs = breadcrumbs.length >= 3;
  const shortcutBannerStorageKey = user ? `${SHORTCUT_BANNER_STORAGE_KEY_PREFIX}:${user.id}` : null;
  const showShortcutBanner = useMemo(() => {
    if (permissionsLoading || !shortcutBannerStorageKey) return false;
    if (dismissedShortcutBannerKeys.has(shortcutBannerStorageKey)) return false;
    try {
      return window.localStorage.getItem(shortcutBannerStorageKey) !== '1';
    } catch {
      return true;
    }
  }, [dismissedShortcutBannerKeys, permissionsLoading, shortcutBannerStorageKey]);

  useEffect(() => {
    if (typeof document === 'undefined') return;
    document.title = getPageTitleForPath(pathname || '/');
  }, [pathname]);

  useEffect(() => {
    if (!pathname || focusedPathnameRef.current === pathname) {
      return undefined;
    }
    focusedPathnameRef.current = pathname;
    const timeoutId = window.setTimeout(() => {
      const main = document.getElementById('main-content');
      if (main instanceof HTMLElement) {
        main.focus();
      }
    }, 0);
    return () => window.clearTimeout(timeoutId);
  }, [pathname]);

  // Close mobile menu on route change
  useEffect(() => {
    if (!isOpen) {
      prevPathnameRef.current = pathname;
      return undefined;
    }

    if (prevPathnameRef.current === pathname) {
      return undefined;
    }

    const timeoutId = window.setTimeout(() => {
      setIsOpen(false);
    }, 0);

    prevPathnameRef.current = pathname;

    return () => window.clearTimeout(timeoutId);
  }, [isOpen, pathname]);

  // Prevent body scroll when mobile menu is open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => {
      document.body.style.overflow = '';
    };
  }, [isOpen]);

  const dismissShortcutBanner = () => {
    if (!shortcutBannerStorageKey) {
      return;
    }
    try {
      window.localStorage.setItem(shortcutBannerStorageKey, '1');
    } catch {
      // noop: banner dismissal is best-effort persistence.
    }
    setDismissedShortcutBannerKeys((prev) => {
      const next = new Set(prev);
      next.add(shortcutBannerStorageKey);
      return next;
    });
  };

  return (
    <MobileMenuContext.Provider
      value={{
        isOpen,
        open: openMenu,
        close: closeMenu,
        toggle: toggleMenu,
      }}
    >
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded-md focus:border focus:border-primary focus:bg-background focus:px-4 focus:py-2 focus:text-foreground"
      >
        Skip to main content
      </a>
      <div className="flex h-screen bg-background">
        {/* Desktop sidebar */}
        <DesktopSidebar />

        {/* Mobile sidebar */}
        <MobileSidebar />

        {/* Mobile header */}
        <MobileHeader />

        {/* Main content */}
        <main id="main-content" tabIndex={-1} className="flex-1 overflow-y-auto pt-14 lg:pt-0">
          {(showShortcutBanner || showBreadcrumbs) && (
            <div className="space-y-3 px-4 pt-3 lg:px-8 lg:pt-5">
              {showShortcutBanner && (
                <Alert data-testid="shortcuts-tip-banner">
                  <AlertDescription className="flex flex-wrap items-center justify-between gap-2">
                    <span>
                      Tip: Use keyboard shortcuts for faster navigation. Press Shift+? for help.
                    </span>
                    <Button variant="ghost" size="sm" onClick={dismissShortcutBanner}>
                      Dismiss
                    </Button>
                  </AlertDescription>
                </Alert>
              )}
              {showBreadcrumbs && <Breadcrumbs />}
            </div>
          )}
          <div className="px-4 pt-2 lg:px-8 empty:hidden">
            <OrgContextBanner />
          </div>
          {children}
        </main>
      </div>
    </MobileMenuContext.Provider>
  );
}

export default ResponsiveLayout;
