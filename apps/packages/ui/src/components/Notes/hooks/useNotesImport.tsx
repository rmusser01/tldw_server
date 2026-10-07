import React from 'react'
import type { MessageInstance } from 'antd/es/message/interface'
import { bgRequest } from '@/services/background-proxy'
import type {
  ImportDuplicateStrategy,
  PendingImportFile,
  NotesImportResponsePayload,
  ImportFormat,
} from '../notes-manager-types'
import {
  detectImportFormatFromFileName,
  detectDeclaredPortableImport,
  estimateDetectedNotesFromImportContent,
} from '../notes-manager-utils'

export interface UseNotesImportDeps {
  isOnline: boolean
  message: MessageInstance
  t: (key: string, opts?: Record<string, any>) => string
  listMode: 'active' | 'trash'
  refetch: () => Promise<any>
}

export function useNotesImport(deps: UseNotesImportDeps) {
  const {
    isOnline,
    message,
    t,
    listMode,
    refetch,
  } = deps

  const [importModalOpen, setImportModalOpen] = React.useState(false)
  const [importSubmitting, setImportSubmitting] = React.useState(false)
  const [importDuplicateStrategy, setImportDuplicateStrategy] =
    React.useState<ImportDuplicateStrategy>('create_copy')
  const [pendingImportFiles, setPendingImportFiles] = React.useState<PendingImportFile[]>([])
  const importInputRef = React.useRef<HTMLInputElement | null>(null)

  const clearImportSelection = React.useCallback(() => {
    setPendingImportFiles([])
    if (importInputRef.current) {
      importInputRef.current.value = ''
    }
  }, [])

  const openImportPicker = React.useCallback(() => {
    if (!isOnline || listMode !== 'active') return
    importInputRef.current?.click()
  }, [isOnline, listMode])

  const prepareImportFiles = React.useCallback(
    async (selectedFiles: File[], sourceMode = false) => {
      const loaded = await Promise.all(
        selectedFiles.map(async (file): Promise<PendingImportFile> => {
          const initialFormat = detectImportFormatFromFileName(file.name, sourceMode);
          try {
            // File.text() consumes a UTF-8 BOM; decode source bytes without removing it.
            const content = new TextDecoder('utf-8', {
              fatal: true,
              ignoreBOM: initialFormat === 'source',
            }).decode(await file.arrayBuffer());
            const format: ImportFormat = detectDeclaredPortableImport(initialFormat, content);
            const detectedNotes = estimateDetectedNotesFromImportContent(format, content);
            const parseError =
              detectedNotes === 0
                ? format === 'json'
                  ? 'Could not parse notes from this JSON file.'
                  : 'Could not read notes from this file.'
                : null;
            return { fileName: file.name, format, content, detectedNotes, parseError };
          } catch {
            return {
              fileName: file.name,
              format: initialFormat,
              content: '',
              detectedNotes: 0,
              parseError: 'Could not read UTF-8 source from this file.',
            };
          }
        })
      );
      setPendingImportFiles(loaded);
      setImportModalOpen(true);
    },
    []
  );
  const handleImportInputChange = React.useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      const selectedFiles = Array.from(event.target.files || [])
      if (selectedFiles.length === 0) {
        return
      }
      await prepareImportFiles(selectedFiles)
    },
    [prepareImportFiles]
  )

  const closeImportModal = React.useCallback(() => {
    setImportModalOpen(false)
    clearImportSelection()
  }, [clearImportSelection])

  const confirmImport = React.useCallback(async () => {
    if (pendingImportFiles.length === 0) {
      message.warning('Select at least one import file')
      return
    }
    if (pendingImportFiles.some((entry) => entry.parseError)) {
      message.warning(
        t('option:notesSearch.importParseErrorsBlockSubmit', {
          defaultValue: 'Fix or reselect files with import errors before importing.'
        })
      )
      return
    }
    const importItems = pendingImportFiles
      .map((entry) => ({
        file_name: entry.fileName,
        format: entry.format,
        content: entry.content
      }))
    if (importItems.length === 0) {
      message.warning('Selected files are empty')
      return
    }

    setImportSubmitting(true)
    try {
      const response = await bgRequest<NotesImportResponsePayload>({
        path: '/api/v1/notes/import' as any,
        method: 'POST' as any,
        headers: { 'Content-Type': 'application/json' },
        body: {
          duplicate_strategy: importDuplicateStrategy,
          items: importItems
        }
      })

      const createdCount = Number(response?.created_count || 0)
      const updatedCount = Number(response?.updated_count || 0)
      const skippedCount = Number(response?.skipped_count || 0)
      const failedCount = Number(response?.failed_count || 0)

      if (failedCount > 0 || skippedCount > 0) {
        message.warning(
          `Import completed with partial results: ${createdCount} created, ${updatedCount} updated, ${skippedCount} skipped, ${failedCount} failed.`
        )
      } else {
        message.success(`Imported ${createdCount + updatedCount} note${createdCount + updatedCount === 1 ? '' : 's'}.`)
      }

      closeImportModal()
      await refetch()
    } catch (error: any) {
      message.error(String(error?.message || 'Import failed'))
    } finally {
      setImportSubmitting(false)
    }
  }, [
    closeImportModal,
    importDuplicateStrategy,
    message,
    pendingImportFiles,
    refetch,
    t
  ])

  return {
    importModalOpen,
    importSubmitting,
    importDuplicateStrategy, setImportDuplicateStrategy,
    pendingImportFiles,
    importInputRef,
    openImportPicker,
    handleImportInputChange,
    prepareImportFiles,
    closeImportModal,
    confirmImport,
  }
}
