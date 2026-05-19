import Foundation

let socketPath = ProcessInfo.processInfo.environment["TLDW_SANDBOX_MACOS_HELPER_SOCKET"] ?? ""
let service = HelperService()
let server = UnixSocketServer(socketPath: socketPath, service: service)

do {
    try server.run()
} catch {
    fputs("macos-vz-helper failed to start: \(error)\n", stderr)
    exit(1)
}
