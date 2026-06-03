import net from "node:net"

export async function reservePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer()
    server.unref()
    server.once("error", reject)
    server.listen(0, "127.0.0.1", () => {
      const address = server.address()
      const port = typeof address === "object" && address ? address.port : undefined
      server.close((error) => {
        if (error) {
          reject(error)
          return
        }
        if (!port) {
          reject(new Error("Failed to reserve a loopback port"))
          return
        }
        resolve(port)
      })
    })
  })
}

export async function reservePorts(names) {
  const ports = {}
  const used = new Set()

  for (const name of names) {
    let port
    do {
      port = await reservePort()
    } while (used.has(port))

    used.add(port)
    ports[name] = port
  }

  return ports
}
