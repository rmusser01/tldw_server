import React from "react"
import OptionLayout from "~/components/Layouts/Layout"
import { useNavigate, useParams } from "react-router-dom"
import { RouteErrorBoundary } from "@/components/Common/RouteErrorBoundary"
import { ConferenceCollectionReview } from "@/components/Review/ConferenceCollectionReview"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import type { MediaCollection } from "@/services/tldw/conference-collections"

const OptionMediaCollectionInner = () => {
  const { collectionId } = useParams<{ collectionId?: string }>()
  const navigate = useNavigate()
  const [collection, setCollection] = React.useState<MediaCollection | null>(null)
  const [loading, setLoading] = React.useState(true)
  const [error, setError] = React.useState<string | null>(null)

  React.useEffect(() => {
    const normalizedId = String(collectionId || "").trim()
    if (!normalizedId) {
      setLoading(false)
      setError("Collection id is required.")
      return
    }

    let cancelled = false
    setLoading(true)
    setError(null)
    tldwClient
      .getMediaCollection(normalizedId, { timeoutMs: 30_000 })
      .then((loaded) => {
        if (cancelled) return
        setCollection(loaded)
      })
      .catch((err) => {
        if (cancelled) return
        setError(
          err instanceof Error && err.message
            ? err.message
            : "Collection could not be loaded."
        )
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false)
        }
      })

    return () => {
      cancelled = true
    }
  }, [collectionId])

  if (loading) {
    return (
      <div className="rounded-md border border-border bg-surface px-4 py-6 text-sm text-text-muted">
        Loading collection...
      </div>
    )
  }

  if (error || !collection) {
    return (
      <div className="rounded-md border border-danger/30 bg-danger/5 px-4 py-6">
        <h1 className="text-lg font-semibold text-text">Collection unavailable</h1>
        <p className="mt-2 text-sm text-text-muted">
          {error || "Collection could not be loaded."}
        </p>
        <button
          type="button"
          onClick={() => navigate("/media-multi")}
          className="mt-4 rounded-md border border-border bg-surface px-3 py-1.5 text-sm font-medium text-text hover:bg-surface2"
        >
          Back to Media Review
        </button>
      </div>
    )
  }

  return (
    <ConferenceCollectionReview
      collection={collection}
      onOpenMedia={(mediaId) => navigate(`/media/${mediaId}/view`)}
      onAskCollection={({ collectionId: scopedCollectionId }) =>
        navigate(`/knowledge?collection_id=${encodeURIComponent(String(scopedCollectionId))}`)
      }
    />
  )
}

const OptionMediaCollection = () => {
  return (
    <OptionLayout>
      <RouteErrorBoundary
        routeId="media-collection"
        routeLabel="Conference Collection"
      >
        <OptionMediaCollectionInner />
      </RouteErrorBoundary>
    </OptionLayout>
  )
}

export default OptionMediaCollection
