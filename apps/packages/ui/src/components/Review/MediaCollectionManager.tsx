import { useState } from 'react'
import { Modal } from 'antd'
import { useTranslation } from 'react-i18next'
import type { useMediaSelection } from './hooks/useMediaSelection'

type Selection = ReturnType<typeof useMediaSelection>
type CheckedMember = { id: string; available: boolean; title?: string }

export function MediaCollectionManager({ selection }: { selection: Selection }) {
  const { t } = useTranslation('review')
  const [open, setOpen] = useState(false)
  const [id, setId] = useState('')
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [query, setQuery] = useState('')
  const [checked, setChecked] = useState<CheckedMember[]>([])
  const [busy, setBusy] = useState(false)
  const [confirmDelete, setConfirmDelete] = useState(false)
  const collection = selection.mediaCollections.find(row => row.id === id)
  const load = (nextId: string) => {
    const next = selection.mediaCollections.find(row => row.id === nextId)
    setId(next?.id || '')
    setName(next?.name || '')
    setDescription(next?.description || '')
    setChecked([])
    setQuery('')
    setConfirmDelete(false)
  }
  const action = async (work: () => Promise<void>) => {
    if (busy) return
    setBusy(true)
    try { await work() } finally { setBusy(false) }
  }
  const button = 'min-h-[44px] rounded-md border border-border px-3 text-sm disabled:opacity-50'
  const input = 'w-full rounded-md border border-border bg-surface p-2 text-text'
  return <>
    <div className="shrink-0 border-b border-border bg-surface px-4 py-2">
      <button type="button" className={button} onClick={() => { load(selection.activeCollectionId || ''); setOpen(true) }}>
        {t('mediaPage.manageCollections', { defaultValue: 'Manage collections' })}
      </button>
    </div>
    <Modal title={t('mediaPage.manageCollections', { defaultValue: 'Manage collections' })} open={open}
      onCancel={() => { if (!busy) setOpen(false) }} footer={null} destroyOnHidden>
      <div className="space-y-3" aria-busy={busy}>
        <p>{t('mediaPage.collectionStorageHelp', { defaultValue: 'Private to this account in this browser. Collections store references; removing a member or deleting a collection does not delete its sources. Sharing is not available here.' })}</p>
        <label className="block">{t('mediaPage.managedCollection', { defaultValue: 'Managed collection' })}
          <select className={input} value={id} disabled={busy} onChange={event => load(event.target.value)}>
            <option value="">{t('mediaPage.newCollection', { defaultValue: 'New collection' })}</option>
            {selection.mediaCollections.map(row => <option key={row.id} value={row.id}>{row.name} ({row.itemIds.length})</option>)}
          </select>
        </label>
        <form className="space-y-3" onSubmit={event => {
          event.preventDefault()
          void action(async () => {
            const saved = await selection.handleSaveCollection(id || null, name, description)
            if (saved) { setId(saved); setConfirmDelete(false) }
          })
        }}>
          <label className="block">{t('mediaPage.collectionName', { defaultValue: 'Collection name' })}
            <input className={input} required maxLength={120} value={name} disabled={busy} onChange={event => setName(event.target.value)} />
          </label>
          <label className="block">{t('mediaPage.collectionDescription', { defaultValue: 'Collection description' })}
            <textarea className={input} maxLength={2000} value={description} disabled={busy} onChange={event => setDescription(event.target.value)} />
          </label>
          <button type="submit" className={button} disabled={busy || !name.trim()}>{id
            ? t('mediaPage.saveCollection', { defaultValue: 'Save collection' })
            : t('mediaPage.createCollection', { defaultValue: 'Create collection' })}</button>
        </form>
        {collection && <>
          <p role="status">{t('mediaPage.collectionSavedReferences', {
            defaultValue: collection.itemIds.length === 1 ? '{{count}} saved reference' : '{{count}} saved references', count: collection.itemIds.length
          })}</p>
          <label className="block">{t('mediaPage.filterCollectionMembers', { defaultValue: 'Filter collection members' })}
            <input type="search" className={input} value={query} onChange={event => setQuery(event.target.value)} />
          </label>
          <button type="button" className={button} disabled={busy} onClick={() => void action(async () => {
            const rows = await selection.handleCheckCollection(collection.id)
            if (rows) setChecked(rows)
          })}>{t('mediaPage.checkMembers', { defaultValue: 'Check members' })}</button>
          <ul className="max-h-64 space-y-2 overflow-auto">
            {collection.itemIds.filter(memberId => `${memberId} ${checked.find(row => row.id === memberId)?.title || ''}`.toLowerCase().includes(query.toLowerCase())).map(memberId => {
              const member = checked.find(row => row.id === memberId)
              return <li key={memberId} className="flex items-center justify-between gap-2 rounded border border-border p-2">
                <span>{member?.title || memberId} <small>{member?.title ? memberId : ''} — {member
                  ? member.available ? t('mediaPage.memberAvailable', { defaultValue: 'Available' }) : t('mediaPage.memberUnavailable', { defaultValue: 'Unavailable or inaccessible; reference retained' })
                  : t('mediaPage.memberUnchecked', { defaultValue: 'Not checked' })}</small></span>
                <button type="button" className={button} disabled={busy}
                  aria-label={t('mediaPage.removeCollectionMember', { defaultValue: 'Remove {{id}} from collection', id: memberId })}
                  onClick={() => void action(async () => { await selection.handleRemoveCollectionMember(collection.id, memberId) })}>
                  {t('mediaPage.removeMember', { defaultValue: 'Remove' })}
                </button>
              </li>
            })}
          </ul>
          <p className="text-sm">{t('mediaPage.collectionResultsHelp', { defaultValue: 'The Media collection filter shows matching results on the current search page. This list contains all saved references, including those not on that page.' })}</p>
          <button type="button" className={button} disabled={busy} onClick={() => setConfirmDelete(true)}>
            {t('mediaPage.deleteCollection', { defaultValue: 'Delete collection' })}
          </button>
          {confirmDelete && <div role="alert" className="space-y-2 rounded border border-danger p-2">
            <p>{t('mediaPage.deleteCollectionConfirm', { defaultValue: 'Delete only this collection? All underlying sources will remain.' })}</p>
            <button type="button" className={button} disabled={busy} onClick={() => void action(async () => {
              if (await selection.handleDeleteCollection(collection.id)) load('')
            })}>{t('mediaPage.deleteCollectionOnly', { defaultValue: 'Delete collection only' })}</button>
            <button type="button" className={button} disabled={busy} onClick={() => setConfirmDelete(false)}>{t('mediaPage.cancel', { defaultValue: 'Cancel' })}</button>
          </div>}
        </>}
      </div>
    </Modal>
  </>
}
