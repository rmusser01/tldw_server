import OptionLayout from "~/components/Layouts/Layout"
import NotesManagerPage from "@/components/Notes/NotesManagerPage"
import { RouteErrorBoundary } from "@/components/Common/RouteErrorBoundary"
import { useLocation } from "react-router-dom"

const OptionNotes = () => {
  const { search } = useLocation()
  return (
    <OptionLayout>
      <RouteErrorBoundary routeId="notes" routeLabel="Notes">
        <NotesManagerPage sourceNoteId={new URLSearchParams(search).get("source_ref_id")} />
      </RouteErrorBoundary>
    </OptionLayout>
  )
}

export default OptionNotes
