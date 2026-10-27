import React from 'react'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ContentEditModal } from '../ContentEditModal'

const mocks = vi.hoisted(() => ({ confirm: vi.fn(), request: vi.fn() }))
vi.mock('@/components/Common/confirm-danger', () => ({ useConfirmDanger: () => mocks.confirm }))
vi.mock('@/services/background-proxy', () => ({ bgRequest: mocks.request }))
// The existing router prompt is a separate tested boundary; capture its activation here.
vi.mock('@/entries/shared/route-leave-prompt', () => ({ RouteLeavePrompt: ({ when }: { when: boolean }) => <output data-testid="media-draft-route-guard">{String(when)}</output> }))
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (_key: string, fallback: string) => fallback }) }))

const initialText = 'Rowan source text.'
const draftText = initialText + '\nUnsaved private correction.'
const props = { open: true, mediaId: 1, initialText, analysisContent: 'Prior analysis', prompt: 'Existing prompt' }
beforeEach(() => { mocks.confirm.mockReset().mockResolvedValue(false); mocks.request.mockReset().mockResolvedValue({}); })
afterEach(() => { cleanup(); vi.restoreAllMocks() })

describe('Media unsaved edit protection', () => {
  it('keeps the exact draft after declining dirty cancellation, then discards explicitly', async () => {
    const onClose = vi.fn()
    render(<ContentEditModal {...props} onClose={onClose} />)
    fireEvent.change(screen.getByRole('textbox'), { target: { value: draftText } })
    fireEvent.click(screen.getByRole('button', { name: 'Cancel', exact: true }))
    await waitFor(() => expect(mocks.confirm).toHaveBeenCalledOnce())
    expect(onClose).not.toHaveBeenCalled()
    expect(screen.getByRole('textbox')).toHaveValue(draftText)
    mocks.confirm.mockResolvedValue(true)
    fireEvent.click(screen.getByRole('button', { name: 'Cancel', exact: true }))
    await waitFor(() => expect(onClose).toHaveBeenCalledOnce())
    expect(mocks.request).not.toHaveBeenCalled()
  })

  it('protects document leave and activates the route guard only while dirty', () => {
    const onClose = vi.fn()
    const view = render(<ContentEditModal {...props} onClose={onClose} />)
    const cleanLeave = new Event('beforeunload', { cancelable: true })
    window.dispatchEvent(cleanLeave)
    expect(cleanLeave.defaultPrevented).toBe(false)
    fireEvent.change(screen.getByRole('textbox'), { target: { value: draftText } })
    const dirtyLeave = new Event('beforeunload', { cancelable: true })
    window.dispatchEvent(dirtyLeave)
    expect(dirtyLeave.defaultPrevented).toBe(true)
    expect(screen.getByTestId('media-draft-route-guard')).toHaveTextContent('true')
    view.unmount()
    const retiredLeave = new Event('beforeunload', { cancelable: true })
    window.dispatchEvent(retiredLeave)
    expect(retiredLeave.defaultPrevented).toBe(false)
  })

  it('closes a clean editor without asking or saving', () => {
    const onClose = vi.fn()
    render(<ContentEditModal {...props} onClose={onClose} />)
    fireEvent.click(screen.getByRole('button', { name: 'Cancel', exact: true }))
    expect(onClose).toHaveBeenCalledOnce()
    expect(mocks.confirm).not.toHaveBeenCalled()
    expect(mocks.request).not.toHaveBeenCalled()
  })

  it('does not close a replacement source after an old discard decision resolves', async () => {
    let answer!: (value: boolean) => void
    mocks.confirm.mockImplementation(() => new Promise<boolean>(resolve => { answer = resolve }))
    const onClose = vi.fn()
    const view = render(<ContentEditModal {...props} onClose={onClose} />)
    fireEvent.change(screen.getByRole('textbox'), { target: { value: draftText } })
    fireEvent.click(screen.getByRole('button', { name: 'Cancel', exact: true }))
    await waitFor(() => expect(mocks.confirm).toHaveBeenCalledOnce())
    fireEvent.click(screen.getByRole('button', { name: 'Cancel', exact: true }))
    expect(mocks.confirm).toHaveBeenCalledOnce()
    expect(answer).toBeTypeOf('function')
    view.rerender(<ContentEditModal {...props} mediaId={2} initialText="Other source" onClose={onClose} />)
    await act(async () => answer(true))
    expect(onClose).not.toHaveBeenCalled()
    expect(screen.getByRole('textbox')).toHaveValue('Other source')
  })

  it('holds cancellation during a save and closes only after the version is confirmed', async () => {
    let complete!: (value: object) => void
    mocks.request.mockImplementation(() => new Promise(resolve => { complete = resolve }))
    const onClose = vi.fn()
    render(<ContentEditModal {...props} onClose={onClose} />)
    fireEvent.change(screen.getByRole('textbox'), { target: { value: draftText } })
    fireEvent.click(screen.getByRole('button', { name: 'Save as New Version', exact: true }))
    await waitFor(() => expect(mocks.request).toHaveBeenCalledOnce())
    expect(screen.getByRole('button', { name: 'Cancel', exact: true })).toBeDisabled()
    fireEvent.click(screen.getByRole('button', { name: 'Cancel', exact: true }))
    expect(onClose).not.toHaveBeenCalled()
    expect(mocks.request).toHaveBeenCalledWith(expect.objectContaining({ path: '/api/v1/media/1/versions', method: 'POST', body: { content: draftText, prompt: 'Existing prompt', analysis_content: 'Prior analysis' } }))
    await act(async () => complete({ version_number: 3 }))
    expect(onClose).toHaveBeenCalledOnce()
    expect(mocks.confirm).not.toHaveBeenCalled()
  })

})
