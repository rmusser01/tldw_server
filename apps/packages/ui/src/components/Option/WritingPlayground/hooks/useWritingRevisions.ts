import { useCallback, useEffect, useRef, useState } from "react"
import { planRevisionApply } from "../writing-revision-utils"
import type { WritingRevisionProposal } from "../writing-revision-types"
import {
  getRevisionsFromPayload,
  getRevisionPayloadSignature,
  mergeRevisionsIntoPayload,
  type WritingSessionPayload
} from "./utils"

type ApplyEditorTextResult =
  | { applied: true }
  | { applied: false; reason: string }

type UseWritingRevisionsDeps = {
  activeSessionId: string | null
  activeSessionPayload?: Record<string, unknown> | null
  editorText: string
  applyEditorText: (nextText: string) => ApplyEditorTextResult
  applySessionPayloadPatch: (
    patcher: (payload: WritingSessionPayload) => WritingSessionPayload
  ) => void
}

const appendNote = (
  proposal: WritingRevisionProposal,
  note: string
): WritingRevisionProposal => ({
  ...proposal,
  notes: [...(proposal.notes ?? []), note]
})

export function useWritingRevisions(deps: UseWritingRevisionsDeps) {
  const {
    activeSessionId,
    activeSessionPayload,
    editorText,
    applyEditorText,
    applySessionPayloadPatch
  } = deps
  const [revisions, setRevisions] = useState<WritingRevisionProposal[]>(() =>
    getRevisionsFromPayload(activeSessionPayload)
  )
  const revisionsRef = useRef(revisions)
  const activeSessionIdRef = useRef(activeSessionId)
  const loadedSessionIdRef = useRef(activeSessionId)
  const revisionPayloadSignatureRef = useRef(
    getRevisionPayloadSignature(activeSessionPayload)
  )
  const revisionListVersionRef = useRef(0)

  activeSessionIdRef.current = activeSessionId

  useEffect(() => {
    revisionsRef.current = revisions
  }, [revisions])

  useEffect(() => {
    const nextRevisions = getRevisionsFromPayload(activeSessionPayload)
    const nextSignature = getRevisionPayloadSignature(activeSessionPayload)
    if (
      loadedSessionIdRef.current === activeSessionId &&
      revisionPayloadSignatureRef.current === nextSignature
    ) {
      return
    }

    loadedSessionIdRef.current = activeSessionId
    revisionPayloadSignatureRef.current = nextSignature
    revisionListVersionRef.current += 1
    revisionsRef.current = nextRevisions
    setRevisions(nextRevisions)
  }, [activeSessionId, activeSessionPayload])

  const persistRevisions = useCallback(
    (nextRevisions: WritingRevisionProposal[]) => {
      applySessionPayloadPatch((payload) =>
        mergeRevisionsIntoPayload(payload, nextRevisions)
      )
    },
    [applySessionPayloadPatch]
  )

  const updateRevisions = useCallback(
    (
      updater: (
        current: WritingRevisionProposal[]
      ) => WritingRevisionProposal[]
    ) => {
      const nextRevisions = updater(revisionsRef.current)
      revisionListVersionRef.current += 1
      revisionPayloadSignatureRef.current = JSON.stringify(nextRevisions)
      revisionsRef.current = nextRevisions
      setRevisions(nextRevisions)
      persistRevisions(nextRevisions)
    },
    [persistRevisions]
  )

  const addRevision = useCallback(
    (proposal: WritingRevisionProposal) => {
      updateRevisions((current) => [...current, proposal])
    },
    [updateRevisions]
  )

  const rejectRevision = useCallback(
    (proposalId: string) => {
      updateRevisions((current) =>
        current.map((proposal) =>
          proposal.id === proposalId
            ? { ...proposal, status: "rejected" }
            : proposal
        )
      )
    },
    [updateRevisions]
  )

  const applyRevision = useCallback(
    (proposalId: string) => {
      const proposal = revisionsRef.current.find(
        (revision) => revision.id === proposalId
      )
      if (!proposal) return

      const plan = planRevisionApply(editorText, proposal)
      if (plan.type === "apply" || plan.type === "retarget") {
        const result = applyEditorText(plan.nextText)
        if (result.applied === false) {
          const failedReason = result.reason || "unknown reason"
          updateRevisions((current) =>
            current.map((revision) => {
              if (revision.id !== proposalId) return revision
              return appendNote(
                { ...revision, status: "conflict" },
                `Manual apply required: ${failedReason}`
              )
            })
          )
          return
        }

        updateRevisions((current) =>
          current.map((revision) =>
            revision.id === proposalId
              ? { ...revision, status: "applied" }
              : revision
          )
        )
        return
      }

      if (plan.type === "noop" && proposal.operation === "advisory") {
        updateRevisions((current) =>
          current.map((revision) =>
            revision.id === proposalId
              ? { ...revision, status: "advisory" }
              : revision
          )
        )
        return
      }

      updateRevisions((current) =>
        current.map((revision) =>
          revision.id === proposalId
            ? appendNote({ ...revision, status: "conflict" }, plan.reason)
            : revision
        )
      )
    },
    [applyEditorText, editorText, updateRevisions]
  )

  const regenerateRevision = useCallback(
    async (
      proposalId: string,
      createReplacement: (
        source: WritingRevisionProposal
      ) => Promise<WritingRevisionProposal>
    ) => {
      const source = revisionsRef.current.find(
        (proposal) => proposal.id === proposalId
      )
      if (!source) return

      const regenerationSessionId = activeSessionIdRef.current
      const regenerationVersion = revisionListVersionRef.current
      const replacement = await createReplacement(source)
      if (activeSessionIdRef.current !== regenerationSessionId) return

      const currentSource = revisionsRef.current.find(
        (proposal) => proposal.id === source.id
      )
      if (
        revisionListVersionRef.current !== regenerationVersion ||
        currentSource !== source ||
        currentSource.sessionId !== regenerationSessionId ||
        currentSource.status !== "pending"
      ) {
        return
      }

      updateRevisions((current) => [
        ...current.map((proposal): WritingRevisionProposal =>
          proposal.id === source.id
            ? { ...proposal, status: "rejected" as const }
            : proposal
        ),
        ({
          ...replacement,
          regeneratedFromId: source.id,
          target: source.target,
          instruction: source.instruction,
          presetId: source.presetId,
          presetInstruction: source.presetInstruction,
          status: "pending"
        } satisfies WritingRevisionProposal)
      ])
    },
    [updateRevisions]
  )

  return {
    revisions,
    addRevision,
    rejectRevision,
    applyRevision,
    regenerateRevision
  }
}
