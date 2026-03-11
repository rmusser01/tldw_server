import Foundation

enum UnixSocketServerError: Error {
    case missingSocketPath
}

final class UnixSocketServer {
    private let socketPath: String
    private let service: HelperService

    init(socketPath: String, service: HelperService) {
        self.socketPath = socketPath.trimmingCharacters(in: .whitespacesAndNewlines)
        self.service = service
    }

    func start() throws {
        guard !socketPath.isEmpty else {
            throw UnixSocketServerError.missingSocketPath
        }
        _ = service.ping()
    }
}
