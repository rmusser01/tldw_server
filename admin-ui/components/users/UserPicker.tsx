'use client';

import { useEffect, useMemo, useState } from 'react';
import { Search, User as UserIcon, X } from 'lucide-react';
import { api } from '@/lib/api-client';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import type { User } from '@/types';
import { logger } from '@/lib/logger';

type UserPickerProps = {
  label?: string;
  placeholder?: string;
  value?: User | null;
  helperText?: string;
  disabled?: boolean;
  onSelect: (user: User) => void;
  onClear?: () => void;
};

const MIN_QUERY_LENGTH = 2;
const SEARCH_DEBOUNCE_MS = 300;

export function UserPicker({
  label = 'User',
  placeholder = 'Search by username or email',
  value,
  helperText,
  disabled,
  onSelect,
  onClear,
}: UserPickerProps) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<User[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const showResults = useMemo(() => {
    return query.trim().length >= MIN_QUERY_LENGTH;
  }, [query]);

  useEffect(() => {
    if (disabled) {
      return;
    }
    const trimmed = query.trim();
    if (trimmed.length < MIN_QUERY_LENGTH) {
      setResults([]);
      setError('');
      return;
    }

    const handle = window.setTimeout(async () => {
      try {
        setLoading(true);
        setError('');
        const data = await api.getUsers({ search: trimmed, limit: '20' });
        setResults(data);
      } catch (err: unknown) {
        logger.error('Failed to search users', { component: 'UserPicker', error: err instanceof Error ? err.message : String(err) });
        setResults([]);
        setError(err instanceof Error ? err.message : 'Failed to search users');
      } finally {
        setLoading(false);
      }
    }, SEARCH_DEBOUNCE_MS);

    return () => window.clearTimeout(handle);
  }, [disabled, query]);

  useEffect(() => {
    if (value) {
      setQuery(value.username || value.email || '');
    }
  }, [value]);

  return (
    <div className="space-y-2">
      <Label>{label}</Label>
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
        <Input
          value={query}
          placeholder={placeholder}
          onChange={(event) => setQuery(event.target.value)}
          disabled={disabled}
          className="pl-9"
        />
        {value && onClear && (
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="absolute right-2 top-1/2 -translate-y-1/2 h-7 w-7"
            aria-label="Clear selected user"
            title="Clear selected user"
            onClick={() => {
              setQuery('');
              setResults([]);
              onClear();
            }}
          >
            <X className="h-4 w-4" />
          </Button>
        )}
      </div>
      {helperText && <p className="text-xs text-muted-foreground">{helperText}</p>}

      {loading && (
        <div className="text-xs text-muted-foreground">Searching...</div>
      )}
      {error && (
        <div className="text-xs text-destructive">{error}</div>
      )}

      {showResults && (
        <div className="rounded-md border bg-background shadow-sm">
          {results.map((user) => (
            <button
              key={user.id}
              type="button"
              onClick={() => onSelect(user)}
              className="flex w-full items-center justify-between gap-3 px-3 py-2 text-left text-sm hover:bg-muted"
            >
              <div className="flex items-center gap-2 truncate">
                <UserIcon className="h-4 w-4 text-muted-foreground" />
                <span className="font-medium truncate">{user.username || user.email}</span>
                {user.email && user.username && (
                  <span className="text-xs text-muted-foreground truncate">{user.email}</span>
                )}
              </div>
              <Badge variant={user.is_active ? 'default' : 'secondary'}>
                {user.is_active ? 'Active' : 'Inactive'}
              </Badge>
            </button>
          ))}
          {results.length === 0 && !loading && !error && (
            <div className="px-3 py-2 text-xs text-muted-foreground">No matches</div>
          )}
        </div>
      )}

      {value && (
        <div className="rounded-md border border-dashed px-3 py-2 text-xs text-muted-foreground">
          Selected user: <span className="font-medium text-foreground">{value.username || value.email}</span>
          {' '}({value.id})
        </div>
      )}
    </div>
  );
}
