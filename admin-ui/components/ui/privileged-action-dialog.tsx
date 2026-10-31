'use client';

import { createContext, ReactNode, useCallback, useContext, useState } from 'react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';

export interface PrivilegedActionOptions {
  title: string;
  message: string;
  confirmText?: string;
  cancelText?: string;
  requirePassword?: boolean;
  confirmationOnly?: boolean;
}

export interface PrivilegedActionResult {
  reason: string;
  adminPassword: string;
}

interface PrivilegedActionContextType {
  prompt: (options: PrivilegedActionOptions) => Promise<PrivilegedActionResult | null>;
}

const PrivilegedActionContext = createContext<PrivilegedActionContextType | null>(null);

export function usePrivilegedActionDialog() {
  const context = useContext(PrivilegedActionContext);
  if (!context) {
    throw new Error('usePrivilegedActionDialog must be used within a PrivilegedActionDialogProvider');
  }
  return context.prompt;
}

interface ProviderProps {
  children: ReactNode;
}

export function PrivilegedActionDialogProvider({ children }: ProviderProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [options, setOptions] = useState<PrivilegedActionOptions | null>(null);
  const [reason, setReason] = useState('');
  const [adminPassword, setAdminPassword] = useState('');
  const [resolveRef, setResolveRef] = useState<((value: PrivilegedActionResult | null) => void) | null>(null);

  const closeDialog = useCallback((value: PrivilegedActionResult | null) => {
    setIsOpen(false);
    setReason('');
    setAdminPassword('');
    resolveRef?.(value);
    setResolveRef(null);
  }, [resolveRef]);

  const prompt = useCallback((nextOptions: PrivilegedActionOptions) => {
    setOptions(nextOptions);
    setReason('');
    setAdminPassword('');
    setIsOpen(true);

    return new Promise<PrivilegedActionResult | null>((resolve) => {
      setResolveRef(() => resolve);
    });
  }, []);

  const confirmationOnly = options?.confirmationOnly ?? false;
  const requiresPassword = !confirmationOnly && (options?.requirePassword ?? true);
  const canSubmit = confirmationOnly
    || (reason.trim().length >= 8 && (!requiresPassword || adminPassword.trim().length >= 8));

  return (
    <PrivilegedActionContext.Provider value={{ prompt }}>
      {children}

      <Dialog open={isOpen} onOpenChange={(open) => !open && closeDialog(null)}>
        <DialogContent className="sm:max-w-lg">
          <DialogHeader>
            <DialogTitle>{options?.title || 'Privileged action'}</DialogTitle>
            <DialogDescription>{options?.message}</DialogDescription>
          </DialogHeader>

          {!confirmationOnly && (
            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="privileged-action-reason">Reason</Label>
                <textarea
                  id="privileged-action-reason"
                  className="min-h-24 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                  value={reason}
                  onChange={(event) => setReason(event.target.value)}
                  placeholder="Describe why this action is necessary"
                />
                <p className="text-xs text-muted-foreground">
                  Required for auditability. Use at least 8 characters.
                </p>
              </div>

              {requiresPassword ? (
                <div className="space-y-2">
                  <Label htmlFor="privileged-action-password">Current password</Label>
                  <Input
                    id="privileged-action-password"
                    type="password"
                    value={adminPassword}
                    onChange={(event) => setAdminPassword(event.target.value)}
                    placeholder="Re-enter your password"
                    autoComplete="current-password"
                  />
                  <p className="text-xs text-muted-foreground">
                    Required to reauthenticate before this high-risk action.
                  </p>
                </div>
              ) : (
                <p className="text-xs text-muted-foreground">
                  Single-user mode requires an audit reason but does not prompt for password reauthentication.
                </p>
              )}
            </div>
          )}

          <DialogFooter className="gap-2 sm:gap-0">
            <Button variant="outline" onClick={() => closeDialog(null)}>
              {options?.cancelText || 'Cancel'}
            </Button>
            <Button
              onClick={() => closeDialog({ reason: reason.trim(), adminPassword: adminPassword.trim() })}
              disabled={!canSubmit}
            >
              {options?.confirmText || 'Confirm'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </PrivilegedActionContext.Provider>
  );
}
