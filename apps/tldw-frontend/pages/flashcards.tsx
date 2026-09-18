import dynamic from "next/dynamic"
import { withPageTitle } from "@web/components/navigation/withPageTitle"

const Flashcards = dynamic(() => import("@/routes/option-flashcards"), { ssr: false })

export default withPageTitle("Flashcards", Flashcards)
