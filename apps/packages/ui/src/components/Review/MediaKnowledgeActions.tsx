import React, { useEffect, useLayoutEffect, useRef, useState } from "react"
import { Button } from "antd"
import { useHomeMilestoneScope } from "@/hooks/useHomeMilestoneScope"
import { watchChatAccountChanges } from "@/services/chat-account-boundary"
import { useAntdMessage } from "@/hooks/useAntdMessage"
import {
  buildKnowledgeQaWorkspacePrefill,
  queueResearchWorkspacePrefill,
} from "@/utils/research-workspace-prefill"
import { buildKnowledgeMediaScopePath } from "@/utils/knowledge-scope-handoff"

type ReviewedMedia = { id: string | number; title?: string; type?: string }

/** Continue the reviewed set through the same Knowledge source contract. */
export function MediaKnowledgeActions({
  items,
  navigate,
  selection = false,
  isCurrent = () => true,
}: {
  items: ReviewedMedia[]
  navigate: (path: string) => void
  selection?: boolean
  isCurrent?: () => boolean
}) {
  const ownerScope = useHomeMilestoneScope()
  const message = useAntdMessage()
  const [pending, setPending] = useState(false)
  const mounted = useRef(true)
  const currentOwnerRef = useRef(ownerScope)
  currentOwnerRef.current = ownerScope
  const itemsKey = JSON.stringify(
    items.map(({ id, title, type }) => ({ id, title, type })),
  )
  const currentItemsRef = useRef(itemsKey)
  currentItemsRef.current = itemsKey
  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
    }
  }, [])
  const invalidated = useRef(false)
  const [ownerInvalidated, setOwnerInvalidated] = useState(false)
  useLayoutEffect(
    () =>
      watchChatAccountChanges((changed) => {
        if (!changed) return
        invalidated.current = true
        setOwnerInvalidated(true)
      }),
    [],
  )
  const mediaIds = items.map((item) => Number(item.id))
  const valid =
    items.length > 0 &&
    items.every(
      (item, index) =>
        (typeof item.id === "number" || /^\d+$/.test(item.id)) &&
        Number.isSafeInteger(mediaIds[index]) &&
        mediaIds[index] > 0,
    )
  const disabled = !valid || !ownerScope || ownerInvalidated || !isCurrent()
  const research = async () => {
    if (disabled || pending || invalidated.current || !isCurrent()) return
    setPending(true)
    try {
      const payload = buildKnowledgeQaWorkspacePrefill({
        threadId: null,
        query: "",
        answer: null,
        citations: [],
        results: items.map((item, index) => ({
          metadata: {
            media_id: mediaIds[index],
            title: item.title ?? `Media ${mediaIds[index]}`,
            source_type: item.type ?? "document",
          },
        })),
      })
      await queueResearchWorkspacePrefill(payload, ownerScope)
      if (
        !mounted.current ||
        invalidated.current ||
        currentOwnerRef.current !== ownerScope ||
        currentItemsRef.current !== itemsKey ||
        !isCurrent()
      )
        return
      navigate("/research-workspace")
    } catch {
      if (mounted.current && !invalidated.current && isCurrent())
        message.error(
          "Could not prepare these sources for Research Workspace. Please try again.",
        )
    } finally {
      if (mounted.current && !invalidated.current) setPending(false)
    }
  }
  return (
    <>
      <Button
        size="small"
        disabled={disabled}
        onClick={() => {
          if (disabled || invalidated.current || !isCurrent()) return
          navigate(buildKnowledgeMediaScopePath(mediaIds))
        }}
      >
        {selection ? "Ask selected items" : "Ask this item"}
      </Button>
      <Button
        size="small"
        disabled={disabled || pending}
        loading={pending}
        onClick={() => {
          void research()
        }}
      >
        {selection
          ? "Research with selected sources"
          : "Research with this source"}
      </Button>
    </>
  )
}
