import dynamic from "next/dynamic"
import Head from "next/head"

const Characters = dynamic(() => import("@/routes/option-characters"), { ssr: false })

export default function CharactersPage() {
  return <><Head><title>Characters | tldw</title></Head><Characters /></>
}
