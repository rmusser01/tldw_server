import OptionLayout from "~/components/Layouts/Layout"
import NotesManagerPage from "@/components/Notes/NotesManagerPage"
import { RouteErrorBoundary } from "@/components/Common/RouteErrorBoundary"
import { RouteLeaveGuard } from "@/entries/shared/route-leave-guard"
import { useNotesRouteAddress } from "@/routes/hooks/useNotesRouteAddress"

const OptionNotes = () => {
  // The address names the open note: /notes?note=<id> (XP-05).
  const { sourceNoteId, onNoteSelectionChange } = useNotesRouteAddress()
  return (
    <OptionLayout>
      <RouteErrorBoundary routeId="notes" routeLabel="Notes">
        <NotesManagerPage sourceNoteId={sourceNoteId} onNoteSelectionChange={onNoteSelectionChange} LeaveGuard={RouteLeaveGuard} />
      </RouteErrorBoundary>
    </OptionLayout>
  )
}

export default OptionNotes
