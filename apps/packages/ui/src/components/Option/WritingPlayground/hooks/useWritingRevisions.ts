import { useCallback, useEffect, useRef, useState } from "react"
import { planRevisionApply } from "../writing-revision-utils"
import type { WritingRevisionProposal } from "../writing-revision-types"
import {
  getRevisionsFromPayload,
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

  useEffect(() => {
    revisionsRef.current = revisions
  }, [revisions])

  useEffect(() => {
    const nextRevisions = getRevisionsFromPayload(activeSessionPayload)
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
        updateRevisions((current) =>
          current.map((revision) => {
            if (revision.id !== proposalId) return revision
            if (result.applied) {
              return { ...revision, status: "applied" }
            }
            return appendNote(
              { ...revision, status: "conflict" },
              `Manual apply required: ${result.reason}`
            )
          })
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

      const replacement = await createReplacement(source)
      updateRevisions((current) => [
        ...current.map((proposal) =>
          proposal.id === source.id
            ? { ...proposal, status: "rejected" }
            : proposal
        ),
        {
          ...replacement,
          regeneratedFromId: source.id,
          target: source.target,
          instruction: source.instruction,
          presetId: source.presetId,
          presetInstruction: source.presetInstruction,
          status: "pending"
        }
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
