import dynamic from "next/dynamic"
import Head from "next/head"
import { formatNotesDocumentTitle, useOpenNoteTitle } from "@/routes/route-document-title"

const Notes = dynamic(() => import("@/routes/option-notes"), { ssr: false })

/** "Notes | tldw", or "<note title> · Notes | tldw" while a note is open (XP-05). */
export default function NotesPage() {
  const noteTitle = useOpenNoteTitle()
  return (
    <>
      <Head>
        <title>{formatNotesDocumentTitle(noteTitle)}</title>
      </Head>
      <Notes />
    </>
  )
}
