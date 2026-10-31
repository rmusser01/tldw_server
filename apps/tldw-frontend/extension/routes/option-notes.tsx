import OptionLayout from "@web/components/layout/WebLayout"
import NotesManagerPage from "@/components/Notes/NotesManagerPage"
import { useLocation } from "react-router-dom"

const OptionNotes = () => {
  const { search } = useLocation()
  return (
    <OptionLayout>
      <NotesManagerPage sourceNoteId={new URLSearchParams(search).get("source_ref_id")} />
    </OptionLayout>
  )
}

export default OptionNotes
