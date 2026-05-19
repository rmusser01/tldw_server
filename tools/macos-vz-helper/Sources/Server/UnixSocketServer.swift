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
    case unsafeSocketPath(String)
    case unsafeSocketDirectory(String)
    case existingSocketPathIsNotSocket(String)
    case existingSocketPathIsActive(String)
}

final class UnixSocketServer {
    private struct SocketPathIdentity {
        let device: dev_t
        let inode: ino_t

        init(_ statBuffer: stat) {
            self.device = statBuffer.st_dev
            self.inode = statBuffer.st_ino
        }
    }

    private let socketPath: String
    private let service: HelperService
    private var serverSocketFD: Int32 = -1
    private var isRunning = false
    private var ownedSocketPathIdentity: SocketPathIdentity?

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
        try prepareSocketPath()

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
            unlinkOwnedSocketPath()
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
        unlinkOwnedSocketPath()
    }

    func handleRequestData(_ data: Data) throws -> Data {
        let decoder = JSONDecoder()
        let request = try decoder.decode(HelperRequest.self, from: data)
        let encoder = JSONEncoder()

        switch request.operation {
        case "ping":
            return try encoder.encode(service.ping())
        case "validate_host":
            do {
                return try encoder.encode(
                    service.validateHost(
                        runtime: request.request["runtime"]?.stringValue ?? "",
                        networkPolicy: try optionalStringField(
                            request.request,
                            key: "network_policy",
                            defaultValue: "deny_all"
                        )
                    )
                )
            } catch {
                return encodeErrorResponse(for: error)
            }
        case "validate_template", "register_template":
            return try encoder.encode(
                service.validateTemplate(
                    runtime: request.request["runtime"]?.stringValue ?? "vz_linux",
                    templatePath: request.request["template"]?.stringValue ?? request.request["source"]?.stringValue ?? ""
                )
            )
        case "create_vm":
            do {
                let networkPolicy = try optionalStringField(
                    request.request,
                    key: "network_policy",
                    defaultValue: "deny_all"
                )
                let templatePath = try optionalStringField(
                    request.request,
                    key: "template",
                    defaultValue: try optionalStringField(
                        request.request,
                        key: "template_path",
                        defaultValue: ""
                    )
                )
                let workspacePath = try optionalStringField(request.request, key: "workspace_path", defaultValue: "")
                let metadata = VMOwnershipMetadata(
                    owner: try optionalStringField(request.request, key: "owner", defaultValue: "unknown"),
                    runtime: try optionalStringField(request.request, key: "runtime", defaultValue: "vz_linux"),
                    runID: try optionalStringField(request.request, key: "run_id", defaultValue: ""),
                    sessionID: try optionalStringField(request.request, key: "session_id", defaultValue: ""),
                    sessionMode: request.request["session_mode"]?.boolValue ?? false,
                    templateID: try optionalStringField(request.request, key: "template_id", defaultValue: ""),
                    templatePath: templatePath,
                    runManifestPath: try optionalStringField(request.request, key: "run_manifest_path", defaultValue: ""),
                    planningSource: try optionalStringField(request.request, key: "planning_source", defaultValue: ""),
                    workspacePath: workspacePath,
                    createdAt: ""
                )
                return try encoder.encode(
                    try service.createVM(
                        vmID: try optionalStringField(
                            request.request,
                            key: "vm_name",
                            defaultValue: try optionalStringField(request.request, key: "run_id", defaultValue: "")
                        ),
                        templatePath: templatePath,
                        workspacePath: workspacePath,
                        readinessTimeoutSeconds: try optionalTimeIntervalField(
                            request.request,
                            key: "timeout_sec",
                            defaultValue: 30
                        ),
                        metadata: metadata,
                        networkPolicy: networkPolicy
                    )
                )
            } catch {
                return encodeErrorResponse(for: error)
            }
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
                    metadata: .unknown,
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
            do {
                let vmID = try requiredStringField(request.request, key: "vm_id")
                let argv = try requiredStringArrayField(request.request, key: "argv")
                let cwd = try optionalStringField(
                    request.request,
                    key: "cwd",
                    defaultValue: "/workspace"
                )
                let env = try optionalStringDictionaryField(
                    request.request,
                    key: "env",
                    defaultValue: [:]
                )
                let timeoutSeconds = try optionalTimeIntervalField(
                    request.request,
                    key: "timeout_sec",
                    defaultValue: 30
                )
                let maxOutputBytes = try optionalIntField(
                    request.request,
                    key: "max_output_bytes"
                )
                return try encoder.encode(
                    try service.execGuest(
                        vmID: vmID,
                        argv: argv,
                        cwd: cwd,
                        env: env,
                        timeoutSeconds: timeoutSeconds,
                        maxOutputBytes: maxOutputBytes
                    )
                )
            } catch {
                return encodeErrorResponse(for: error)
            }
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
        let address = try socketAddress()
        let bindResult = withUnsafePointer(to: address.value) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockaddrPointer in
                Darwin.bind(socketFD, sockaddrPointer, address.length)
            }
        }
        guard bindResult == 0 else {
            throw UnixSocketServerError.bindFailed(errno)
        }
        var boundStat = stat()
        guard fstat(socketFD, &boundStat) == 0 else {
            throw UnixSocketServerError.bindFailed(errno)
        }
        ownedSocketPathIdentity = SocketPathIdentity(boundStat)
        guard Darwin.listen(socketFD, SOMAXCONN) == 0 else {
            throw UnixSocketServerError.listenFailed(errno)
        }
    }

    private func prepareSocketPath() throws {
        let socketDirectory = URL(fileURLWithPath: socketPath).deletingLastPathComponent()
        try prepareSocketDirectory(socketDirectory)

        var existing = stat()
        let statResult = lstat(socketPath, &existing)
        if statResult != 0 {
            if errno == ENOENT {
                return
            }
            throw UnixSocketServerError.unsafeSocketPath(socketPath)
        }

        let type = existing.st_mode & S_IFMT
        if type == S_IFLNK {
            throw UnixSocketServerError.unsafeSocketPath(socketPath)
        }
        if type != S_IFSOCK {
            throw UnixSocketServerError.existingSocketPathIsNotSocket(socketPath)
        }
        if try existingSocketPathHasListener() {
            throw UnixSocketServerError.existingSocketPathIsActive(socketPath)
        }
        try unlinkStaleSocketPath(expected: SocketPathIdentity(existing))
    }

    private func prepareSocketDirectory(_ socketDirectory: URL) throws {
        let directoryPath = socketDirectory.path
        guard !directoryPath.isEmpty else {
            throw UnixSocketServerError.unsafeSocketDirectory(directoryPath)
        }

        var missingDirectories: [URL] = []
        var current = socketDirectory
        while true {
            var directoryStat = stat()
            let statResult = lstat(current.path, &directoryStat)
            if statResult == 0 {
                try validateSocketDirectory(current.path, statBuffer: directoryStat)
                break
            }
            guard errno == ENOENT else {
                throw UnixSocketServerError.unsafeSocketDirectory(current.path)
            }
            missingDirectories.append(current)
            let parent = current.deletingLastPathComponent()
            guard parent.path != current.path else {
                throw UnixSocketServerError.unsafeSocketDirectory(current.path)
            }
            current = parent
        }

        for directory in missingDirectories.reversed() {
            try FileManager.default.createDirectory(
                at: directory,
                withIntermediateDirectories: false,
                attributes: [.posixPermissions: 0o700]
            )
            try validateSocketDirectory(directory.path)
        }
    }

    private func validateSocketDirectory(_ directoryPath: String, statBuffer: stat? = nil) throws {
        var directoryStat = statBuffer ?? stat()
        if statBuffer == nil {
            let statResult = lstat(directoryPath, &directoryStat)
            guard statResult == 0 else {
                throw UnixSocketServerError.unsafeSocketDirectory(directoryPath)
            }
        }

        let type = directoryStat.st_mode & S_IFMT
        guard type == S_IFDIR else {
            throw UnixSocketServerError.unsafeSocketDirectory(directoryPath)
        }
        guard directoryStat.st_uid == geteuid() else {
            throw UnixSocketServerError.unsafeSocketDirectory(directoryPath)
        }
        guard directoryStat.st_mode & 0o077 == 0 else {
            throw UnixSocketServerError.unsafeSocketDirectory(directoryPath)
        }
    }

    private func existingSocketPathHasListener() throws -> Bool {
        let socketFD = Darwin.socket(AF_UNIX, SOCK_STREAM, 0)
        guard socketFD >= 0 else {
            throw UnixSocketServerError.socketCreateFailed(errno)
        }
        defer {
            close(socketFD)
        }

        let originalFlags = fcntl(socketFD, F_GETFL, 0)
        guard originalFlags >= 0,
              fcntl(socketFD, F_SETFL, originalFlags | O_NONBLOCK) == 0 else {
            throw UnixSocketServerError.unsafeSocketPath(socketPath)
        }

        let address = try socketAddress()
        let connectResult = withUnsafePointer(to: address.value) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockaddrPointer in
                connect(socketFD, sockaddrPointer, address.length)
            }
        }
        if connectResult == 0 {
            return true
        }

        let connectError = errno
        if connectError == EINPROGRESS || connectError == EALREADY || connectError == EAGAIN || connectError == EWOULDBLOCK {
            return try waitForSocketProbe(socketFD)
        }
        if connectError == ECONNREFUSED || connectError == ENOENT {
            return false
        }
        if connectError == EISCONN {
            return true
        }
        throw UnixSocketServerError.unsafeSocketPath(socketPath)
    }

    private func waitForSocketProbe(_ socketFD: Int32) throws -> Bool {
        var descriptor = pollfd(fd: socketFD, events: Int16(POLLOUT), revents: 0)
        while true {
            let pollResult = Darwin.poll(&descriptor, 1, 200)
            if pollResult == 0 {
                return true
            }
            if pollResult < 0 {
                if errno == EINTR {
                    continue
                }
                throw UnixSocketServerError.unsafeSocketPath(socketPath)
            }

            var socketError: Int32 = 0
            var socketErrorLength = socklen_t(MemoryLayout<Int32>.size)
            let optionResult = withUnsafeMutablePointer(to: &socketError) { pointer in
                pointer.withMemoryRebound(to: UInt8.self, capacity: MemoryLayout<Int32>.size) { rawPointer in
                    getsockopt(socketFD, SOL_SOCKET, SO_ERROR, rawPointer, &socketErrorLength)
                }
            }
            guard optionResult == 0 else {
                throw UnixSocketServerError.unsafeSocketPath(socketPath)
            }

            if socketError == 0 || socketError == EISCONN {
                return true
            }
            if socketError == ECONNREFUSED || socketError == ENOENT {
                return false
            }
            if socketError == EAGAIN || socketError == EWOULDBLOCK || socketError == ETIMEDOUT {
                return true
            }
            throw UnixSocketServerError.unsafeSocketPath(socketPath)
        }
    }

    private func unlinkStaleSocketPath(expected: SocketPathIdentity) throws {
        let current = try currentSocketPathIdentity()
        guard current.device == expected.device,
              current.inode == expected.inode else {
            throw UnixSocketServerError.unsafeSocketPath(socketPath)
        }

        if unlink(socketPath) != 0 {
            throw UnixSocketServerError.unsafeSocketPath(socketPath)
        }
    }

    private func currentSocketPathIdentity() throws -> SocketPathIdentity {
        var current = stat()
        let result = lstat(socketPath, &current)
        if result != 0 {
            throw UnixSocketServerError.unsafeSocketPath(socketPath)
        }

        let currentType = current.st_mode & S_IFMT
        guard currentType == S_IFSOCK else {
            throw UnixSocketServerError.unsafeSocketPath(socketPath)
        }
        return SocketPathIdentity(current)
    }

    private func socketAddress() throws -> (value: sockaddr_un, length: socklen_t) {
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

        return (address, socklen_t(MemoryLayout<sa_family_t>.size + pathBytes.count))
    }

    private func unlinkOwnedSocketPath() {
        guard let expected = ownedSocketPathIdentity else {
            return
        }
        defer {
            ownedSocketPathIdentity = nil
        }
        guard let current = try? currentSocketPathIdentity(),
              current.device == expected.device,
              current.inode == expected.inode else {
            return
        }
        unlink(socketPath)
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

    private func optionalStringField(
        _ request: [String: JSONValue],
        key: String,
        defaultValue: String
    ) throws -> String {
        guard let value = request[key] else {
            return defaultValue
        }
        guard let stringValue = value.stringValue else {
            throw UnixSocketServerError.invalidRequest
        }
        return stringValue
    }

    private func requiredStringField(
        _ request: [String: JSONValue],
        key: String
    ) throws -> String {
        guard let value = request[key] else {
            throw UnixSocketServerError.invalidRequest
        }
        guard let stringValue = value.stringValue else {
            throw UnixSocketServerError.invalidRequest
        }
        return stringValue
    }

    private func requiredStringArrayField(
        _ request: [String: JSONValue],
        key: String
    ) throws -> [String] {
        guard let value = request[key] else {
            throw UnixSocketServerError.invalidRequest
        }
        guard let arrayValue = value.arrayValue else {
            throw UnixSocketServerError.invalidRequest
        }
        return try arrayValue.map { item in
            guard let stringValue = item.stringValue else {
                throw UnixSocketServerError.invalidRequest
            }
            return stringValue
        }
    }

    private func optionalStringDictionaryField(
        _ request: [String: JSONValue],
        key: String,
        defaultValue: [String: String]
    ) throws -> [String: String] {
        guard let value = request[key] else {
            return defaultValue
        }
        guard let objectValue = value.objectValue else {
            throw UnixSocketServerError.invalidRequest
        }
        var parsed: [String: String] = [:]
        for (envKey, envValue) in objectValue {
            guard let stringValue = envValue.stringValue else {
                throw UnixSocketServerError.invalidRequest
            }
            parsed[envKey] = stringValue
        }
        return parsed
    }

    private func optionalTimeIntervalField(
        _ request: [String: JSONValue],
        key: String,
        defaultValue: TimeInterval
    ) throws -> TimeInterval {
        guard let value = request[key] else {
            return defaultValue
        }
        switch value {
        case .int(let intValue):
            return TimeInterval(intValue)
        case .double(let doubleValue):
            return TimeInterval(doubleValue)
        default:
            throw UnixSocketServerError.invalidRequest
        }
    }

    private func optionalIntField(
        _ request: [String: JSONValue],
        key: String
    ) throws -> Int? {
        guard let value = request[key] else {
            return nil
        }
        switch value {
        case .int(let intValue):
            return intValue
        default:
            throw UnixSocketServerError.invalidRequest
        }
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
        case HelperServiceError.unsupportedRuntime:
            return "runtime_unsupported"
        case HelperServiceError.invalidCreateVMRequest:
            return "create_vm_request_invalid"
        case HelperServiceError.invalidCreateVMTimeout:
            return "create_vm_timeout_invalid"
        case HelperServiceError.unsupportedNetworkPolicy(let policy):
            return policy == "allowlist" ? "strict_allowlist_not_supported" : "unsupported_network_policy"
        case HelperServiceError.invalidExecArgv:
            return "exec_argv_invalid"
        case HelperServiceError.invalidExecCwd:
            return "exec_cwd_invalid"
        case HelperServiceError.invalidExecEnv:
            return "exec_env_invalid"
        case HelperServiceError.invalidExecTimeout:
            return "exec_timeout_invalid"
        case HelperServiceError.invalidExecOutputLimit:
            return "exec_output_limit_invalid"
        case VZLinuxVMManagerError.bootNotImplemented:
            return "boot_not_implemented"
        case GuestBridgeError.guestReadinessNotImplemented:
            return "guest_readiness_not_implemented"
        case GuestBridgeError.guestExecNotImplemented:
            return "guest_exec_not_implemented"
        case VSockSessionError.requestTimedOut:
            return "guest_transport_timeout"
        case VSockSessionError.connectionNotReady:
            return "guest_not_ready"
        case VSockSessionError.connectionRejected:
            return "guest_connection_rejected"
        case VSockSessionError.requestAlreadyInFlight:
            return "guest_request_in_flight"
        case VSockSessionError.closed:
            return "guest_transport_closed"
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
        case UnixSocketServerError.invalidRequest, is DecodingError:
            return "invalid_request"
        case HelperServiceError.unsupportedRuntime(let runtime):
            return runtime
        case HelperServiceError.invalidCreateVMRequest(let reason),
             HelperServiceError.invalidCreateVMTimeout(let reason):
            return reason
        case HelperServiceError.unsupportedNetworkPolicy(let policy):
            return policy
        case HelperServiceError.invalidExecArgv(let reason),
             HelperServiceError.invalidExecCwd(let reason),
             HelperServiceError.invalidExecEnv(let reason),
             HelperServiceError.invalidExecTimeout(let reason),
             HelperServiceError.invalidExecOutputLimit(let reason):
            return reason
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
