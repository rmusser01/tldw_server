import dynamic from "next/dynamic"
import Head from "next/head"

const Flashcards = dynamic(() => import("@/routes/option-flashcards"), { ssr: false })

export default function FlashcardsPage() {
  return <><Head><title>Flashcards | tldw</title></Head><Flashcards /></>
}
