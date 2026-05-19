'use client';

import { useCallback, useRef, useState } from 'react';
import { X } from 'lucide-react';
import { cn } from '@/lib/utils';

export interface TagInputProps {
  /** Current comma-separated string value (kept in sync with parent form state). */
  value: string;
  /** Called with the updated comma-separated string whenever tags change. */
  onChange: (value: string) => void;
  placeholder?: string;
  id?: string;
  className?: string;
  /** If true, the component is read-only. */
  disabled?: boolean;
}

function parseTags(raw: string): string[] {
  return raw
    .split(',')
    .map((t) => t.trim())
    .filter(Boolean);
}

function joinTags(tags: string[]): string {
  return tags.join(', ');
}

/**
 * A text input that converts comma-separated values into visual tag chips.
 *
 * Type a value and press Enter or comma to add it as a tag.
 * Click the X on a tag to remove it.  The component stores and
 * emits a plain comma-separated string so it can be used as a
 * drop-in replacement for a regular `<Input>`.
 */
export function TagInput({
  value,
  onChange,
  placeholder = 'Type and press Enter',
  id,
  className,
  disabled = false,
}: TagInputProps) {
  const tags = parseTags(value);
  const [draft, setDraft] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  const addTag = useCallback(
    (raw: string) => {
      const tag = raw.trim();
      if (!tag) return;
      // Avoid duplicates
      if (tags.includes(tag)) {
        setDraft('');
        return;
      }
      onChange(joinTags([...tags, tag]));
      setDraft('');
    },
    [tags, onChange],
  );

  const removeTag = useCallback(
    (index: number) => {
      const next = tags.filter((_, i) => i !== index);
      onChange(joinTags(next));
    },
    [tags, onChange],
  );

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' || e.key === ',') {
      e.preventDefault();
      addTag(draft);
    }
    if (e.key === 'Backspace' && draft === '' && tags.length > 0) {
      removeTag(tags.length - 1);
    }
  };

  const handlePaste = (e: React.ClipboardEvent<HTMLInputElement>) => {
    const text = e.clipboardData.getData('text');
    if (text.includes(',')) {
      e.preventDefault();
      const pasted = parseTags(text);
      const unique = pasted.filter((t) => !tags.includes(t));
      if (unique.length > 0) {
        onChange(joinTags([...tags, ...unique]));
      }
    }
  };

  return (
    <div
      className={cn(
        'flex min-h-[2.5rem] w-full flex-wrap items-center gap-1.5 rounded-md border border-input bg-background px-2 py-1.5 text-sm ring-offset-background focus-within:ring-2 focus-within:ring-ring focus-within:ring-offset-2',
        disabled && 'cursor-not-allowed opacity-50',
        className,
      )}
      onClick={() => inputRef.current?.focus()}
      role="group"
      aria-label="Tag input"
    >
      {tags.map((tag, index) => (
        <span
          key={`${tag}-${index}`}
          className="inline-flex items-center gap-1 rounded-full border bg-secondary px-2 py-0.5 text-xs font-medium text-secondary-foreground"
        >
          {tag}
          {!disabled && (
            <button
              type="button"
              aria-label={`Remove ${tag}`}
              className="ml-0.5 rounded-full hover:bg-muted-foreground/20 focus:outline-none focus:ring-1 focus:ring-ring"
              onClick={(e) => {
                e.stopPropagation();
                removeTag(index);
              }}
            >
              <X className="h-3 w-3" />
            </button>
          )}
        </span>
      ))}
      <input
        ref={inputRef}
        id={id}
        type="text"
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={handleKeyDown}
        onPaste={handlePaste}
        onBlur={() => addTag(draft)}
        placeholder={tags.length === 0 ? placeholder : ''}
        disabled={disabled}
        className="min-w-[80px] flex-1 border-0 bg-transparent p-0 text-sm outline-none placeholder:text-muted-foreground focus:ring-0"
        aria-label={placeholder}
      />
    </div>
  );
}
