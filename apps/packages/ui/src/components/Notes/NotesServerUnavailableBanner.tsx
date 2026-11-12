import React from 'react'
import { useTranslation } from 'react-i18next'

/**
 * Editor notice shown while the notes server cannot be reached (NL-16).
 *
 * The editor stays usable: saves that fail because the server is down keep
 * the draft on this device and send it once the server answers (NS-05).
 */
const NotesServerUnavailableBanner: React.FC = () => {
  const { t } = useTranslation(['option'])
  return (
    <div
      className="mb-3 rounded border border-warn/40 bg-warn/10 px-3 py-2 text-[12px] text-text"
      role="status"
      aria-live="polite"
      data-testid="notes-server-unavailable-banner"
    >
      <div className="font-medium">
        {t('option:notesSearch.serverUnavailableTitle', {
          defaultValue: 'Notes server unavailable'
        })}
      </div>
      <div className="mt-0.5 text-text-muted">
        {t('option:notesSearch.serverUnavailableDescription', {
          defaultValue:
            'You can keep writing. Your edits are kept on this device and saved when the server is back.'
        })}
      </div>
    </div>
  )
}

export default NotesServerUnavailableBanner
