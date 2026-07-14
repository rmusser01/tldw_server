export const readQuickIngestFileBytes = async (file: File): Promise<ArrayBuffer> => {
  if (typeof file.arrayBuffer === "function") {
    return file.arrayBuffer()
  }

  if (typeof FileReader !== "undefined") {
    return new Promise<ArrayBuffer>((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => {
        if (reader.result instanceof ArrayBuffer) {
          resolve(reader.result)
          return
        }
        reject(new Error("The selected file could not be read."))
      }
      reader.onerror = () =>
        reject(reader.error || new Error("The selected file could not be read."))
      reader.readAsArrayBuffer(file)
    })
  }

  if (typeof Response !== "undefined") {
    return new Response(file).arrayBuffer()
  }

  throw new Error("The selected file could not be read in this browser.")
}
