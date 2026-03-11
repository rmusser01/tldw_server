import Darwin
import Foundation

enum UnixSocketServerError: Error {
    case missingSocketPath
    case invalidRequest
    case unsupportedOperation(String)
    case socketPathTooLong
    case socketCreateFailed(Int32)
    case bindFailed(Int32)
    case listenFailed(Int32)
    case acceptFailed(Int32)
    case readFailed(Int32)
    case writeFailed(Int32)
}

final class UnixSocketServer {
    private let socketPath: String
    private let service: HelperService
    private var serverSocketFD: Int32 = -1
    private var isRunning = false

    init(socketPath: String, service: HelperService) {
        self.socketPath = socketPath.trimmingCharacters(in: .whitespacesAndNewlines)
        self.service = service
    }

    deinit {
        stop()
    }

    func start() throws {
        guard !socketPath.isEmpty else {
            throw UnixSocketServerError.missingSocketPath
        }
        if isRunning {
            return
        }
        try FileManager.default.createDirectory(
            at: URL(fileURLWithPath: socketPath).deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        unlink(socketPath)

        let socketFD = Darwin.socket(AF_UNIX, SOCK_STREAM, 0)
        guard socketFD >= 0 else {
            throw UnixSocketServerError.socketCreateFailed(errno)
        }

        do {
            try bindAndListen(socketFD: socketFD)
            serverSocketFD = socketFD
            isRunning = true
        } catch {
            close(socketFD)
            unlink(socketPath)
            throw error
        }
    }

    func run() throws {
        try start()
        while isRunning {
            let clientFD = accept(serverSocketFD, nil, nil)
            if clientFD < 0 {
                if !isRunning, errno == EBADF {
                    break
                }
                if errno == EINTR {
                    continue
                }
                throw UnixSocketServerError.acceptFailed(errno)
            }
            do {
                try handleConnection(clientFD: clientFD)
            } catch {
                close(clientFD)
                throw error
            }
            close(clientFD)
        }
    }

    func stop() {
        isRunning = false
        if serverSocketFD >= 0 {
            close(serverSocketFD)
            serverSocketFD = -1
        }
        unlink(socketPath)
    }

    func handleRequestData(_ data: Data) throws -> Data {
        let decoder = JSONDecoder()
        let request = try decoder.decode(HelperRequest.self, from: data)
        let encoder = JSONEncoder()

        switch request.operation {
        case "ping":
            return try encoder.encode(service.ping())
        case "validate_host":
            return try encoder.encode(
                service.validateHost(
                    runtime: request.request["runtime"]?.stringValue ?? "",
                    networkPolicy: request.request["network_policy"]?.stringValue ?? ""
                )
            )
        case "validate_template", "register_template":
            return try encoder.encode(
                service.validateTemplate(
                    runtime: request.request["runtime"]?.stringValue ?? "vz_linux",
                    templatePath: request.request["template"]?.stringValue ?? request.request["source"]?.stringValue ?? ""
                )
            )
        case "create_vm":
            return try encoder.encode(
                try service.createVM(
                    vmID: request.request["vm_name"]?.stringValue ?? request.request["run_id"]?.stringValue ?? "",
                    templatePath: request.request["template"]?.stringValue ?? "",
                    workspacePath: request.request["workspace_path"]?.stringValue ?? "",
                    readinessTimeoutSeconds: TimeInterval(request.request["timeout_sec"]?.intValue ?? 30)
                )
            )
        case "get_vm_status":
            let vmID = request.request["vm_id"]?.stringValue ?? ""
            if let status = service.getVMStatus(vmID: vmID) {
                return try encoder.encode(status)
            }
            return try encoder.encode(
                HelperVMStatusResponse(
                    protocolVersion: "1",
                    helperVersion: "0.1.0",
                    vmID: vmID,
                    state: "missing",
                    healthy: false,
                    details: ["error_code": "vm_not_found"]
                )
            )
        case "list_vms":
            return try encoder.encode(service.listVMs())
        case "terminate_vm":
            let terminated = try service.terminateVM(vmID: request.request["vm_id"]?.stringValue ?? "")
            return try encoder.encode(
                HelperTerminateResponse(
                    protocolVersion: "1",
                    helperVersion: "0.1.0",
                    terminated: terminated
                )
            )
        case "exec_guest":
            let argv = (request.request["argv"]?.arrayValue ?? []).compactMap { $0.stringValue }
            let env = request.request["env"]?.objectValue?.compactMapValues { $0.stringValue } ?? [:]
            return try encoder.encode(
                try service.execGuest(
                    vmID: request.request["vm_id"]?.stringValue ?? "",
                    argv: argv,
                    cwd: request.request["cwd"]?.stringValue ?? "",
                    env: env,
                    timeoutSeconds: TimeInterval(request.request["timeout_sec"]?.intValue ?? 30)
                )
            )
        default:
            return try encoder.encode(
                HelperErrorResponse(
                    protocolVersion: "1",
                    helperVersion: "0.1.0",
                    errorCode: "unsupported_operation",
                    message: request.operation
                )
            )
        }
    }

    private func bindAndListen(socketFD: Int32) throws {
        var address = sockaddr_un()
        address.sun_family = sa_family_t(AF_UNIX)

        let pathBytes = Array(socketPath.utf8CString)
        let maxLength = MemoryLayout.size(ofValue: address.sun_path)
        guard pathBytes.count <= maxLength else {
            throw UnixSocketServerError.socketPathTooLong
        }

        withUnsafeMutablePointer(to: &address.sun_path.0) { pointer in
            pointer.initialize(repeating: 0, count: maxLength)
            for (index, byte) in pathBytes.enumerated() {
                pointer.advanced(by: index).pointee = byte
            }
        }

        let addressLength = socklen_t(MemoryLayout<sa_family_t>.size + pathBytes.count)
        let bindResult = withUnsafePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockaddrPointer in
                Darwin.bind(socketFD, sockaddrPointer, addressLength)
            }
        }
        guard bindResult == 0 else {
            throw UnixSocketServerError.bindFailed(errno)
        }
        guard Darwin.listen(socketFD, SOMAXCONN) == 0 else {
            throw UnixSocketServerError.listenFailed(errno)
        }
    }

    private func handleConnection(clientFD: Int32) throws {
        let requestData = try readRequest(clientFD: clientFD)
        let responseData: Data
        do {
            responseData = try handleRequestData(requestData)
        } catch {
            responseData = encodeErrorResponse(for: error)
        }
        try writeResponse(clientFD: clientFD, responseData: responseData + Data([0x0A]))
    }

    private func readRequest(clientFD: Int32) throws -> Data {
        var data = Data()
        var buffer = [UInt8](repeating: 0, count: 4096)

        while true {
            let readCount = recv(clientFD, &buffer, buffer.count, 0)
            if readCount < 0 {
                throw UnixSocketServerError.readFailed(errno)
            }
            if readCount == 0 {
                break
            }
            data.append(buffer, count: readCount)
            if buffer.prefix(readCount).contains(0x0A) {
                break
            }
        }

        let trimmed = data.trimmingTrailingNewlines()
        guard !trimmed.isEmpty else {
            throw UnixSocketServerError.invalidRequest
        }
        return trimmed
    }

    private func writeResponse(clientFD: Int32, responseData: Data) throws {
        var remaining = responseData[...]
        while !remaining.isEmpty {
            let bytesWritten = remaining.withUnsafeBytes { rawBuffer in
                guard let baseAddress = rawBuffer.baseAddress else {
                    return 0
                }
                return Darwin.write(clientFD, baseAddress, rawBuffer.count)
            }
            if bytesWritten < 0 {
                throw UnixSocketServerError.writeFailed(errno)
            }
            remaining = remaining.dropFirst(bytesWritten)
        }
    }

    private func encodeErrorResponse(for error: Error) -> Data {
        let encoder = JSONEncoder()
        let response = HelperErrorResponse(
            protocolVersion: "1",
            helperVersion: "0.1.0",
            errorCode: errorCode(for: error),
            message: errorMessage(for: error)
        )
        return (try? encoder.encode(response)) ?? Data("{\"protocol_version\":\"1\",\"helper_version\":\"0.1.0\",\"error_code\":\"helper_internal_error\",\"message\":\"encoding failure\"}".utf8)
    }

    private func errorCode(for error: Error) -> String {
        switch error {
        case UnixSocketServerError.invalidRequest, is DecodingError:
            return "invalid_request"
        case UnixSocketServerError.missingSocketPath:
            return "helper_socket_unconfigured"
        case UnixSocketServerError.socketPathTooLong:
            return "helper_socket_path_too_long"
        case UnixSocketServerError.unsupportedOperation:
            return "unsupported_operation"
        case VZLinuxVMManagerError.bootNotImplemented:
            return "boot_not_implemented"
        case GuestBridgeError.guestReadinessNotImplemented:
            return "guest_readiness_not_implemented"
        case GuestBridgeError.guestExecNotImplemented:
            return "guest_exec_not_implemented"
        default:
            return "helper_internal_error"
        }
    }

    private func errorMessage(for error: Error) -> String {
        switch error {
        case UnixSocketServerError.acceptFailed(let code),
             UnixSocketServerError.bindFailed(let code),
             UnixSocketServerError.listenFailed(let code),
             UnixSocketServerError.readFailed(let code),
             UnixSocketServerError.socketCreateFailed(let code),
             UnixSocketServerError.writeFailed(let code):
            return "system_error_\(code)"
        default:
            return String(describing: error)
        }
    }
}

private extension Data {
    func trimmingTrailingNewlines() -> Data {
        var trimmed = self
        while let last = trimmed.last, last == 0x0A || last == 0x0D {
            trimmed.removeLast()
        }
        return trimmed
    }
}
