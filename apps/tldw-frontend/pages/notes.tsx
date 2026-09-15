import dynamic from "next/dynamic"
import Head from "next/head"

const Notes = dynamic(() => import("@/routes/option-notes"), { ssr: false })

export default function NotesPage() {
  return <><Head><title>Notes | tldw</title></Head><Notes /></>
}
