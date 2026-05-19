'use client';

import { useState, createContext, useContext, ReactNode, useCallback } from 'react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import type { ButtonProps } from '@/components/ui/button';
import { AlertTriangle, Trash2, RefreshCw, UserMinus, Key } from 'lucide-react';

type ConfirmVariant = 'danger' | 'warning' | 'default';
type ButtonVariant = NonNullable<ButtonProps['variant']>;

interface ConfirmOptions {
  title: string;
  message: string;
  confirmText?: string;
  cancelText?: string;
  variant?: ConfirmVariant;
  icon?: 'delete' | 'warning' | 'rotate' | 'remove-user' | 'key';
}

interface ConfirmContextType {
  confirm: (options: ConfirmOptions) => Promise<boolean>;
}

const ConfirmContext = createContext<ConfirmContextType | null>(null);

export function useConfirm() {
  const context = useContext(ConfirmContext);
  if (!context) {
    throw new Error('useConfirm must be used within a ConfirmProvider');
  }
  return context.confirm;
}

interface ConfirmProviderProps {
  children: ReactNode;
}

export function ConfirmProvider({ children }: ConfirmProviderProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [options, setOptions] = useState<ConfirmOptions | null>(null);
  const [resolveRef, setResolveRef] = useState<((value: boolean) => void) | null>(null);

  const confirm = useCallback((opts: ConfirmOptions): Promise<boolean> => {
    setOptions(opts);
    setIsOpen(true);

    return new Promise<boolean>((resolve) => {
      setResolveRef(() => resolve);
    });
  }, []);

  const handleConfirm = () => {
    setIsOpen(false);
    resolveRef?.(true);
  };

  const handleCancel = () => {
    setIsOpen(false);
    resolveRef?.(false);
  };

  const getIcon = () => {
    if (!options?.icon) return null;

    const iconClass = 'h-6 w-6';
    switch (options.icon) {
      case 'delete':
        return <Trash2 className={`${iconClass} text-destructive`} />;
      case 'warning':
        return <AlertTriangle className={`${iconClass} text-yellow-500`} />;
      case 'rotate':
        return <RefreshCw className={`${iconClass} text-primary`} />;
      case 'remove-user':
        return <UserMinus className={`${iconClass} text-destructive`} />;
      case 'key':
        return <Key className={`${iconClass} text-primary`} />;
      default:
        return null;
    }
  };

  const getButtonVariant = (): ButtonVariant => {
    switch (options?.variant) {
      case 'danger':
        return 'destructive';
      case 'warning':
        return 'default';
      default:
        return 'default';
    }
  };

  return (
    <ConfirmContext.Provider value={{ confirm }}>
      {children}

      <Dialog open={isOpen} onOpenChange={(open) => !open && handleCancel()}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <div className="flex items-center gap-3">
              {getIcon()}
              <DialogTitle>{options?.title || 'Confirm'}</DialogTitle>
            </div>
            <DialogDescription className="pt-2">
              {options?.message}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="gap-2 sm:gap-0">
            <Button variant="outline" onClick={handleCancel}>
              {options?.cancelText || 'Cancel'}
            </Button>
            <Button variant={getButtonVariant()} onClick={handleConfirm}>
              {options?.confirmText || 'Confirm'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </ConfirmContext.Provider>
  );
}

// Note: The deprecated ConfirmDialog component has been removed.
// All consumers have been migrated to useConfirm() via ConfirmProvider.
