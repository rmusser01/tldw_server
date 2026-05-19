'use client';

import { useEffect, useCallback, useMemo, useState } from 'react';
import { useRouter } from 'next/navigation';

interface ShortcutConfig {
  key: string;
  ctrlKey?: boolean;
  metaKey?: boolean;
  shiftKey?: boolean;
  altKey?: boolean;
  action: () => void;
  description: string;
  category: string;
}

export const SHORTCUTS_HELP_EVENT = 'admin:shortcuts:open';

export function openShortcutsHelp(): void {
  if (typeof window === 'undefined') return;
  window.dispatchEvent(new CustomEvent(SHORTCUTS_HELP_EVENT));
}

// Global keyboard shortcuts
export function useKeyboardShortcuts() {
  const router = useRouter();
  const [isHelpOpen, setIsHelpOpen] = useState(false);

  // Define shortcuts
  const shortcuts = useMemo<ShortcutConfig[]>(() => ([
    // Navigation shortcuts (g + key pattern)
    {
      key: 'g h',
      action: () => router.push('/'),
      description: 'Go to Dashboard',
      category: 'Navigation',
    },
    {
      key: 'g u',
      action: () => router.push('/users'),
      description: 'Go to Users',
      category: 'Navigation',
    },
    {
      key: 'g o',
      action: () => router.push('/organizations'),
      description: 'Go to Organizations',
      category: 'Navigation',
    },
    {
      key: 'g t',
      action: () => router.push('/teams'),
      description: 'Go to Teams',
      category: 'Navigation',
    },
    {
      key: 'g r',
      action: () => router.push('/roles'),
      description: 'Go to Roles',
      category: 'Navigation',
    },
    {
      key: 'g a',
      action: () => router.push('/audit'),
      description: 'Go to Audit Logs',
      category: 'Navigation',
    },
    {
      key: 'g m',
      action: () => router.push('/monitoring'),
      description: 'Go to Monitoring',
      category: 'Navigation',
    },
    {
      key: 'g p',
      action: () => router.push('/providers'),
      description: 'Go to Providers',
      category: 'Navigation',
    },
    {
      key: 'g c',
      action: () => router.push('/config'),
      description: 'Go to Configuration',
      category: 'Navigation',
    },
    {
      key: 'g s',
      action: () => router.push('/security'),
      description: 'Go to Security',
      category: 'Navigation',
    },
    {
      key: 'g b',
      action: () => router.push('/budgets'),
      description: 'Go to Budgets',
      category: 'Navigation',
    },
    {
      key: 'g j',
      action: () => router.push('/jobs'),
      description: 'Go to Jobs',
      category: 'Navigation',
    },
    {
      key: 'g i',
      action: () => router.push('/incidents'),
      description: 'Go to Incidents',
      category: 'Navigation',
    },
    {
      key: 'g d',
      action: () => router.push('/data-ops'),
      description: 'Go to Data Ops',
      category: 'Navigation',
    },
    {
      key: 'g k',
      action: () => router.push('/api-keys'),
      description: 'Go to API Keys',
      category: 'Navigation',
    },
    {
      key: 'g l',
      action: () => router.push('/logs'),
      description: 'Go to Logs',
      category: 'Navigation',
    },
    // Help
    {
      key: '?',
      shiftKey: true,
      action: () => setIsHelpOpen(true),
      description: 'Show keyboard shortcuts',
      category: 'General',
    },
    {
      key: 'Escape',
      action: () => setIsHelpOpen(false),
      description: 'Close dialogs',
      category: 'General',
    },
  ]), [router]);

  // Track for "g" prefix shortcuts
  const [pendingPrefix, setPendingPrefix] = useState<string | null>(null);
  const [prefixTimeout, setPrefixTimeout] = useState<NodeJS.Timeout | null>(null);

  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      // Don't trigger shortcuts when typing in inputs
      const target = event.target as HTMLElement;
      if (
        target.tagName === 'INPUT' ||
        target.tagName === 'TEXTAREA' ||
        target.tagName === 'SELECT' ||
        target.isContentEditable
      ) {
        return;
      }

      const key = event.key.toLowerCase();

      // Handle escape for closing modals
      if (key === 'escape') {
        setIsHelpOpen(false);
        setPendingPrefix(null);
        return;
      }

      // Handle ? for help (requires shift)
      if (event.key === '?' && event.shiftKey) {
        event.preventDefault();
        setIsHelpOpen(true);
        return;
      }

      // Handle "g" prefix for navigation
      // Don't preventDefault on the initial chord key — screen readers use single-letter keys
      if (key === 'g' && !pendingPrefix && !event.ctrlKey && !event.metaKey) {
        setPendingPrefix('g');

        // Clear prefix after 1 second
        if (prefixTimeout) clearTimeout(prefixTimeout);
        const timeout = setTimeout(() => {
          setPendingPrefix(null);
        }, 1000);
        setPrefixTimeout(timeout);
        return;
      }

      // Handle second key after "g" prefix
      if (pendingPrefix === 'g') {
        const fullKey = `g ${key}`;
        const shortcut = shortcuts.find((s) => s.key === fullKey);

        if (shortcut) {
          event.preventDefault();
          shortcut.action();
        }

        setPendingPrefix(null);
        if (prefixTimeout) clearTimeout(prefixTimeout);
        return;
      }

      // Handle single-key shortcuts
      const shortcut = shortcuts.find((s) => {
        if (s.key.includes(' ')) return false; // Skip multi-key shortcuts
        return (
          s.key.toLowerCase() === key &&
          !!s.ctrlKey === event.ctrlKey &&
          !!s.metaKey === event.metaKey &&
          !!s.shiftKey === event.shiftKey &&
          !!s.altKey === event.altKey
        );
      });

      if (shortcut) {
        event.preventDefault();
        shortcut.action();
      }
    },
    [pendingPrefix, prefixTimeout, shortcuts]
  );

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      if (prefixTimeout) clearTimeout(prefixTimeout);
    };
  }, [handleKeyDown, prefixTimeout]);

  useEffect(() => {
    const handleHelpOpen = () => setIsHelpOpen(true);
    window.addEventListener(SHORTCUTS_HELP_EVENT, handleHelpOpen);
    return () => window.removeEventListener(SHORTCUTS_HELP_EVENT, handleHelpOpen);
  }, []);

  return {
    isHelpOpen,
    setIsHelpOpen,
    shortcuts,
    pendingPrefix,
  };
}

// Get shortcut groups for display
export function getShortcutGroups(shortcuts: ShortcutConfig[]) {
  const groups: Record<string, ShortcutConfig[]> = {};

  for (const shortcut of shortcuts) {
    if (!groups[shortcut.category]) {
      groups[shortcut.category] = [];
    }
    groups[shortcut.category].push(shortcut);
  }

  return groups;
}

// Format shortcut key for display
export function formatShortcutKey(shortcut: ShortcutConfig): string {
  const parts: string[] = [];

  if (shortcut.ctrlKey) parts.push('Ctrl');
  if (shortcut.metaKey) parts.push('⌘');
  if (shortcut.altKey) parts.push('Alt');
  if (shortcut.shiftKey) parts.push('Shift');

  // Format the key
  let key = shortcut.key;
  if (key.includes(' ')) {
    // Multi-key shortcut like "g h"
    key = key.split(' ').map(k => k.toUpperCase()).join(' then ');
  } else {
    key = key.toUpperCase();
  }

  parts.push(key);

  return parts.join(' + ');
}
