'use client';

import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Download, FileSpreadsheet, FileJson, ChevronDown } from 'lucide-react';
import { ExportFormat } from '@/lib/export';

interface ExportMenuProps {
  onExport: (format: ExportFormat) => void;
  disabled?: boolean;
  label?: string;
}

export function ExportMenu({ onExport, disabled = false, label = 'Export' }: ExportMenuProps) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="outline"
          disabled={disabled}
          className="gap-2"
          aria-label={label}
        >
          <Download className="h-4 w-4" />
          {label}
          <ChevronDown className="h-4 w-4" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-48">
        <DropdownMenuItem
          onSelect={() => onExport('csv')}
          className="flex items-center gap-2 px-3 py-2 cursor-pointer"
        >
          <FileSpreadsheet className="h-4 w-4 text-green-600" />
          <div>
            <div className="font-medium">Export as CSV</div>
            <div className="text-xs text-muted-foreground">Spreadsheet format</div>
          </div>
        </DropdownMenuItem>
        <DropdownMenuItem
          onSelect={() => onExport('json')}
          className="flex items-center gap-2 px-3 py-2 cursor-pointer"
        >
          <FileJson className="h-4 w-4 text-blue-600" />
          <div>
            <div className="font-medium">Export as JSON</div>
            <div className="text-xs text-muted-foreground">Structured data format</div>
          </div>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
