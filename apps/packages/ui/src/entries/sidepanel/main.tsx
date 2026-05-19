import React from "react"
import ReactDOM from "react-dom/client"
import "@/assets/react-pdf.css"
import IndexSidepanel from "./App"
import { checkReactInstance } from "@/utils/react-instance-check"

checkReactInstance("sidepanel")

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <IndexSidepanel />
  </React.StrictMode>
)
