import OptionLayout from "@web/components/layout/WebLayout"
import NotesManagerPage from "@/components/Notes/NotesManagerPage"
import { RouteLeaveGuard } from "@/entries/shared/route-leave-guard"
import { useNotesRouteAddress } from "@/routes/hooks/useNotesRouteAddress"

const OptionNotes = () => {
  // The address names the open note: /notes?note=<id> (XP-05).
  const { sourceNoteId, onNoteSelectionChange } = useNotesRouteAddress()
  return (
    <OptionLayout>
      <NotesManagerPage sourceNoteId={sourceNoteId} onNoteSelectionChange={onNoteSelectionChange} LeaveGuard={RouteLeaveGuard} />
    </OptionLayout>
  )
}

export default OptionNotes
