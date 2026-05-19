import Darwin
import Foundation
import Testing
@testable import MacOSVZHelperDaemon

@Test func unixSocketServerHandlesPingRequest() throws {
    let server = UnixSocketServer(
        socketPath: "/tmp/macos-vz-helper.sock",
        service: HelperService()
    )

    let request = """
    {"operation":"ping","protocol_version":"1","request":{}}
    """.data(using: .utf8)!

    let responseData = try server.handleRequestData(request)
    let json = try JSONSerialization.jsonObject(with: responseData) as? [String: Any]

    #expect(json?["protocol_version"] as? String == "1")
    #expect(json?["helper_version"] as? String == "0.1.0")
    #expect(json?["status"] as? String == "ok")
}

@Test func unixSocketServerHandlesCreateVMAndExecGuestRequests() throws {
    let registry = VMRegistry()
    let guestBridge = RecordingGuestBridge()
    let manager = VZLinuxVMManager(
        registry: registry,
        bootDriver: RecordingBootDriver(),
        guestBridge: guestBridge
    )
    let service = HelperService(
        hostFacts: HostFacts(isMacOS: true, isAppleSilicon: true),
        registry: registry,
        vmManager: manager
    )
    let server = UnixSocketServer(
        socketPath: "/tmp/macos-vz-helper.sock",
        service: service
    )

    let createRequest = """
    {"operation":"create_vm","protocol_version":"1","request":{"vm_name":"vm-route","template":"/tmp/template.img","workspace_path":"/workspace"}}
    """.data(using: .utf8)!
    let createResponseData = try server.handleRequestData(createRequest)
    let createJSON = try JSONSerialization.jsonObject(with: createResponseData) as? [String: Any]
    #expect(createJSON?["protocol_version"] as? String == "1")
    #expect(createJSON?["helper_version"] as? String == "0.1.0")
    #expect(createJSON?["vm_id"] as? String == "vm-route")
    #expect(createJSON?["state"] as? String == "running")

    let execRequest = """
    {"operation":"exec_guest","protocol_version":"1","request":{"vm_id":"vm-route","argv":["/bin/echo","ok"],"cwd":"/workspace","timeout_sec":15.5}}
    """.data(using: .utf8)!
    let execResponseData = try server.handleRequestData(execRequest)
    let execJSON = try JSONSerialization.jsonObject(with: execResponseData) as? [String: Any]
    #expect(execJSON?["protocol_version"] as? String == "1")
    #expect(execJSON?["helper_version"] as? String == "0.1.0")
    #expect(execJSON?["exit_code"] as? Int == 0)
    #expect(execJSON?["stdout"] as? String == "ok\n")
    #expect(guestBridge.lastExec?.vmID == "vm-route")
    #expect(guestBridge.lastExec?.timeout == 15.5)
}

@Test func unixSocketServerRejectsMalformedExecGuestRequestShape() throws {
    let registry = VMRegistry()
    let manager = VZLinuxVMManager(
        registry: registry,
        bootDriver: RecordingBootDriver(),
        guestBridge: RecordingGuestBridge()
    )
    let service = HelperService(
        hostFacts: HostFacts(isMacOS: true, isAppleSilicon: true),
        registry: registry,
        vmManager: manager
    )
    let server = UnixSocketServer(
        socketPath: "/tmp/macos-vz-helper.sock",
        service: service
    )

    let createRequest = """
    {"operation":"create_vm","protocol_version":"1","request":{"vm_name":"vm-exec-shape","template":"/tmp/template.img","workspace_path":"/workspace"}}
    """.data(using: .utf8)!
    _ = try server.handleRequestData(createRequest)

    let malformedRequests = [
        """
        {"operation":"exec_guest","protocol_version":"1","request":{"vm_id":"vm-exec-shape","argv":"/bin/echo","cwd":"/workspace","timeout_sec":15}}
        """,
        """
        {"operation":"exec_guest","protocol_version":"1","request":{"vm_id":"vm-exec-shape","argv":["/bin/echo",1],"cwd":"/workspace","timeout_sec":15}}
        """,
        """
        {"operation":"exec_guest","protocol_version":"1","request":{"vm_id":"vm-exec-shape","argv":["/bin/echo"],"cwd":["/workspace"],"timeout_sec":15}}
        """,
        """
        {"operation":"exec_guest","protocol_version":"1","request":{"vm_id":"vm-exec-shape","argv":["/bin/echo"],"cwd":"/workspace","env":{"OK":1},"timeout_sec":15}}
        """,
        """
        {"operation":"exec_guest","protocol_version":"1","request":{"vm_id":"vm-exec-shape","argv":["/bin/echo"],"cwd":"/workspace","timeout_sec":"15"}}
        """,
    ]

    for request in malformedRequests {
        let responseData = try server.handleRequestData(request.data(using: .utf8)!)
        let json = try JSONSerialization.jsonObject(with: responseData) as? [String: Any]
        #expect(json?["error_code"] as? String == "invalid_request")
    }
}

@Test func unixSocketServerRejectsInvalidExecGuestContract() throws {
    let registry = VMRegistry()
    let manager = VZLinuxVMManager(
        registry: registry,
        bootDriver: RecordingBootDriver(),
        guestBridge: RecordingGuestBridge()
    )
    let service = HelperService(
        hostFacts: HostFacts(isMacOS: true, isAppleSilicon: true),
        registry: registry,
        vmManager: manager
    )
    let server = UnixSocketServer(
        socketPath: "/tmp/macos-vz-helper.sock",
        service: service
    )

    let createRequest = """
    {"operation":"create_vm","protocol_version":"1","request":{"vm_name":"vm-exec-contract","template":"/tmp/template.img","workspace_path":"/workspace"}}
    """.data(using: .utf8)!
    _ = try server.handleRequestData(createRequest)

    let invalidRequestsAndCodes = [
        (
            """
            {"operation":"exec_guest","protocol_version":"1","request":{"vm_id":"vm-exec-contract","argv":[],"cwd":"/workspace","timeout_sec":15}}
            """,
            "exec_argv_invalid"
        ),
        (
            """
            {"operation":"exec_guest","protocol_version":"1","request":{"vm_id":"vm-exec-contract","argv":["/bin/echo"],"cwd":"/workspace/../tmp","timeout_sec":15}}
            """,
            "exec_cwd_invalid"
        ),
        (
            """
            {"operation":"exec_guest","protocol_version":"1","request":{"vm_id":"vm-exec-contract","argv":["/bin/echo"],"cwd":"/workspace","env":{"BAD=KEY":"1"},"timeout_sec":15}}
            """,
            "exec_env_invalid"
        ),
        (
            """
            {"operation":"exec_guest","protocol_version":"1","request":{"vm_id":"vm-exec-contract","argv":["/bin/echo"],"cwd":"/workspace","timeout_sec":0}}
            """,
            "exec_timeout_invalid"
        ),
    ]

    for (request, expectedCode) in invalidRequestsAndCodes {
        let responseData = try server.handleRequestData(request.data(using: .utf8)!)
        let json = try JSONSerialization.jsonObject(with: responseData) as? [String: Any]
        #expect(json?["error_code"] as? String == expectedCode)
    }
}

@Test func unixSocketServerExecGuestAcceptsMaxOutputBytes() throws {
    let registry = VMRegistry()
    let manager = VZLinuxVMManager(
        registry: registry,
        bootDriver: RecordingBootDriver(),
        guestBridge: StaticOutputGuestBridge(stdout: String(repeating: "o", count: 100), stderr: String(repeating: "e", count: 100))
    )
    let service = HelperService(
        hostFacts: HostFacts(isMacOS: true, isAppleSilicon: true),
        registry: registry,
        vmManager: manager
    )
    let server = UnixSocketServer(
        socketPath: "/tmp/macos-vz-helper.sock",
        service: service
    )

    let createRequest = """
    {"operation":"create_vm","protocol_version":"1","request":{"vm_name":"vm-exec-output-route","template":"/tmp/template.img","workspace_path":"/workspace"}}
    """.data(using: .utf8)!
    _ = try server.handleRequestData(createRequest)

    let execRequest = """
    {"operation":"exec_guest","protocol_version":"1","request":{"vm_id":"vm-exec-output-route","argv":["/bin/echo","ok"],"cwd":"/workspace","timeout_sec":15,"max_output_bytes":10}}
    """.data(using: .utf8)!
    let responseData = try server.handleRequestData(execRequest)
    let json = try JSONSerialization.jsonObject(with: responseData) as? [String: Any]
    let details = json?["details"] as? [String: Any]

    let stdout = json?["stdout"] as? String ?? ""
    let stderr = json?["stderr"] as? String ?? ""
    #expect(Data(stdout.utf8).count + Data(stderr.utf8).count <= 10)
    #expect(stdout.isEmpty == false)
    #expect(stderr.isEmpty == false)
    #expect(details?["output_limit_bytes"] as? String == "10")
    #expect(details?["stdout_truncated"] as? String == "true")
    #expect(details?["stderr_truncated"] as? String == "true")
}

@Test func unixSocketServerRejectsMalformedExecGuestMaxOutputBytes() throws {
    let registry = VMRegistry()
    let manager = VZLinuxVMManager(
        registry: registry,
        bootDriver: RecordingBootDriver(),
        guestBridge: RecordingGuestBridge()
    )
    let service = HelperService(
        hostFacts: HostFacts(isMacOS: true, isAppleSilicon: true),
        registry: registry,
        vmManager: manager
    )
    let server = UnixSocketServer(
        socketPath: "/tmp/macos-vz-helper.sock",
        service: service
    )

    let createRequest = """
    {"operation":"create_vm","protocol_version":"1","request":{"vm_name":"vm-exec-output-shape","template":"/tmp/template.img","workspace_path":"/workspace"}}
    """.data(using: .utf8)!
    _ = try server.handleRequestData(createRequest)

    let execRequest = """
    {"operation":"exec_guest","protocol_version":"1","request":{"vm_id":"vm-exec-output-shape","argv":["/bin/echo","ok"],"cwd":"/workspace","timeout_sec":15,"max_output_bytes":"10"}}
    """.data(using: .utf8)!
    let responseData = try server.handleRequestData(execRequest)
    let json = try JSONSerialization.jsonObject(with: responseData) as? [String: Any]

    #expect(json?["error_code"] as? String == "invalid_request")
}

@Test func unixSocketServerRejectsInvalidExecGuestMaxOutputBytes() throws {
    let registry = VMRegistry()
    let manager = VZLinuxVMManager(
        registry: registry,
        bootDriver: RecordingBootDriver(),
        guestBridge: RecordingGuestBridge()
    )
    let service = HelperService(
        hostFacts: HostFacts(isMacOS: true, isAppleSilicon: true),
        registry: registry,
        vmManager: manager
    )
    let server = UnixSocketServer(
        socketPath: "/tmp/macos-vz-helper.sock",
        service: service
    )

    let createRequest = """
    {"operation":"create_vm","protocol_version":"1","request":{"vm_name":"vm-exec-output-invalid","template":"/tmp/template.img","workspace_path":"/workspace"}}
    """.data(using: .utf8)!
    _ = try server.handleRequestData(createRequest)

    let execRequest = """
    {"operation":"exec_guest","protocol_version":"1","request":{"vm_id":"vm-exec-output-invalid","argv":["/bin/echo","ok"],"cwd":"/workspace","timeout_sec":15,"max_output_bytes":0}}
    """.data(using: .utf8)!
    let responseData = try server.handleRequestData(execRequest)
    let json = try JSONSerialization.jsonObject(with: responseData) as? [String: Any]

    #expect(json?["error_code"] as? String == "exec_output_limit_invalid")
    #expect(json?["message"] as? String == "output_limit_out_of_range")
}

@Test func unixSocketServerForwardsCreateVMOwnershipMetadata() throws {
    let registry = VMRegistry()
    let manager = VZLinuxVMManager(
        registry: registry,
        bootDriver: RecordingBootDriver(),
        guestBridge: ReadyGuestBridge()
    )
    let service = HelperService(
        hostFacts: HostFacts(isMacOS: true, isAppleSilicon: true),
        registry: registry,
        vmManager: manager
    )
    let server = UnixSocketServer(
        socketPath: "/tmp/macos-vz-helper.sock",
        service: service
    )

    let createRequest = """
    {"operation":"create_vm","protocol_version":"1","request":{"vm_name":"vm-owned","owner":"tldw","runtime":"vz_linux","run_id":"run-owned","session_id":"session-owned","session_mode":true,"template_id":"vz_linux:template-owned","template":"/tmp/template.img","run_manifest_path":"/tmp/image-store/runs/run-owned/manifest.json","planning_source":"image_store","workspace_path":"/workspace"}}
    """.data(using: .utf8)!
    let createResponseData = try server.handleRequestData(createRequest)
    let createJSON = try JSONSerialization.jsonObject(with: createResponseData) as? [String: Any]
    let metadata = createJSON?["metadata"] as? [String: Any]

    #expect(createJSON?["vm_id"] as? String == "vm-owned")
    #expect(metadata?["owner"] as? String == "tldw")
    #expect(metadata?["runtime"] as? String == "vz_linux")
    #expect(metadata?["run_id"] as? String == "run-owned")
    #expect(metadata?["session_id"] as? String == "session-owned")
    #expect(metadata?["session_mode"] as? Bool == true)
    #expect(metadata?["template_id"] as? String == "vz_linux:template-owned")
    #expect(metadata?["template_path"] as? String == "/tmp/template.img")
    #expect(metadata?["run_manifest_path"] as? String == "/tmp/image-store/runs/run-owned/manifest.json")
    #expect(metadata?["planning_source"] as? String == "image_store")
    #expect(metadata?["workspace_path"] as? String == "/workspace")
    #expect(metadata?["network_policy"] as? String == "deny_all")
    #expect((metadata?["created_at"] as? String)?.isEmpty == false)
}

@Test func unixSocketServerRejectsCreateVMUnsupportedNetworkPolicy() throws {
    let registry = VMRegistry()
    let manager = VZLinuxVMManager(
        registry: registry,
        bootDriver: RecordingBootDriver(),
        guestBridge: ReadyGuestBridge()
    )
    let service = HelperService(
        hostFacts: HostFacts(isMacOS: true, isAppleSilicon: true),
        registry: registry,
        vmManager: manager
    )
    let server = UnixSocketServer(
        socketPath: "/tmp/macos-vz-helper.sock",
        service: service
    )

    let createRequest = """
    {"operation":"create_vm","protocol_version":"1","request":{"vm_name":"vm-allowlist","template":"/tmp/template.img","workspace_path":"/workspace","network_policy":"allowlist"}}
    """.data(using: .utf8)!
    let responseData = try server.handleRequestData(createRequest)
    let json = try JSONSerialization.jsonObject(with: responseData) as? [String: Any]

    #expect(json?["error_code"] as? String == "strict_allowlist_not_supported")
    #expect(json?["message"] as? String == "allowlist")
    #expect(registry.status(vmID: "vm-allowlist") == nil)
}

@Test func unixSocketServerRejectsCreateVMNonStringNetworkPolicy() throws {
    let registry = VMRegistry()
    let manager = VZLinuxVMManager(
        registry: registry,
        bootDriver: RecordingBootDriver(),
        guestBridge: ReadyGuestBridge()
    )
    let service = HelperService(
        hostFacts: HostFacts(isMacOS: true, isAppleSilicon: true),
        registry: registry,
        vmManager: manager
    )
    let server = UnixSocketServer(
        socketPath: "/tmp/macos-vz-helper.sock",
        service: service
    )

    let createRequest = """
    {"operation":"create_vm","protocol_version":"1","request":{"vm_name":"vm-malformed-policy","template":"/tmp/template.img","workspace_path":"/workspace","network_policy":["deny_all"]}}
    """.data(using: .utf8)!
    let responseData = try server.handleRequestData(createRequest)
    let json = try JSONSerialization.jsonObject(with: responseData) as? [String: Any]

    #expect(json?["error_code"] as? String == "invalid_request")
    #expect(registry.status(vmID: "vm-malformed-policy") == nil)
}

@Test func unixSocketServerRejectsInvalidCreateVMContractBeforeBoot() throws {
    let registry = VMRegistry()
    let manager = VZLinuxVMManager(
        registry: registry,
        bootDriver: RecordingBootDriver(),
        guestBridge: ReadyGuestBridge()
    )
    let service = HelperService(
        hostFacts: HostFacts(isMacOS: true, isAppleSilicon: true),
        registry: registry,
        vmManager: manager
    )
    let server = UnixSocketServer(
        socketPath: "/tmp/macos-vz-helper.sock",
        service: service
    )

    let cases = [
        (
            """
            {"operation":"create_vm","protocol_version":"1","request":{"vm_name":"vm-runtime","runtime":"vz_macos","template":"/tmp/template.img","workspace_path":"/workspace"}}
            """,
            "runtime_unsupported",
            "vz_macos",
            "vm-runtime"
        ),
        (
            """
            {"operation":"create_vm","protocol_version":"1","request":{"vm_name":"bad/name","template":"/tmp/template.img","workspace_path":"/workspace"}}
            """,
            "create_vm_request_invalid",
            "vm_id_invalid",
            "bad/name"
        ),
        (
            """
            {"operation":"create_vm","protocol_version":"1","request":{"vm_name":"vm-template","template":"relative.img","workspace_path":"/workspace"}}
            """,
            "create_vm_request_invalid",
            "template_path_invalid",
            "vm-template"
        ),
        (
            """
            {"operation":"create_vm","protocol_version":"1","request":{"vm_name":"vm-workspace","template":"/tmp/template.img","workspace_path":"workspace"}}
            """,
            "create_vm_request_invalid",
            "workspace_path_invalid",
            "vm-workspace"
        ),
        (
            """
            {"operation":"create_vm","protocol_version":"1","request":{"vm_name":"vm-timeout","template":"/tmp/template.img","workspace_path":"/workspace","timeout_sec":0}}
            """,
            "create_vm_timeout_invalid",
            "timeout_out_of_range",
            "vm-timeout"
        ),
        (
            """
            {"operation":"create_vm","protocol_version":"1","request":{"vm_name":"vm-timeout-shape","template":"/tmp/template.img","workspace_path":"/workspace","timeout_sec":"30"}}
            """,
            "invalid_request",
            "invalid_request",
            "vm-timeout-shape"
        ),
    ]

    for (request, expectedCode, expectedMessage, vmID) in cases {
        let responseData = try server.handleRequestData(request.data(using: .utf8)!)
        let json = try JSONSerialization.jsonObject(with: responseData) as? [String: Any]
        #expect(json?["error_code"] as? String == expectedCode)
        #expect(json?["message"] as? String == expectedMessage)
        #expect(registry.status(vmID: vmID) == nil)
    }
}

@Test func unixSocketServerRejectsSymlinkParentCreateVMPathBeforeBoot() throws {
    let root = URL(fileURLWithPath: NSTemporaryDirectory())
        .appendingPathComponent("macos-vz-create-vm-\(UUID().uuidString)")
    let target = root.appendingPathComponent("target", isDirectory: true)
    let link = root.appendingPathComponent("workspace-link", isDirectory: true)
    try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
    try FileManager.default.createSymbolicLink(atPath: link.path, withDestinationPath: target.path)
    defer { try? FileManager.default.removeItem(at: root) }

    let registry = VMRegistry()
    let manager = VZLinuxVMManager(
        registry: registry,
        bootDriver: RecordingBootDriver(),
        guestBridge: ReadyGuestBridge()
    )
    let service = HelperService(
        hostFacts: HostFacts(isMacOS: true, isAppleSilicon: true),
        registry: registry,
        vmManager: manager
    )
    let server = UnixSocketServer(
        socketPath: "/tmp/macos-vz-helper.sock",
        service: service
    )
    let workspacePath = link.appendingPathComponent("nested-workspace").path
    let request = """
    {"operation":"create_vm","protocol_version":"1","request":{"vm_name":"vm-symlink-parent","template":"/tmp/template.img","workspace_path":"\(workspacePath)"}}
    """

    let responseData = try server.handleRequestData(request.data(using: .utf8)!)
    let json = try JSONSerialization.jsonObject(with: responseData) as? [String: Any]

    #expect(json?["error_code"] as? String == "create_vm_request_invalid")
    #expect(json?["message"] as? String == "workspace_path_invalid")
    #expect(registry.status(vmID: "vm-symlink-parent") == nil)
}

@Test func unixSocketServerRejectsValidateHostNonStringNetworkPolicy() throws {
    let server = UnixSocketServer(
        socketPath: "/tmp/macos-vz-helper.sock",
        service: HelperService(hostFacts: HostFacts(isMacOS: true, isAppleSilicon: true))
    )

    let validateRequest = """
    {"operation":"validate_host","protocol_version":"1","request":{"runtime":"vz_linux","network_policy":{"mode":"deny_all"}}}
    """.data(using: .utf8)!
    let responseData = try server.handleRequestData(validateRequest)
    let json = try JSONSerialization.jsonObject(with: responseData) as? [String: Any]

    #expect(json?["error_code"] as? String == "invalid_request")
}

@Test func unixSocketServerServesPingOverRealSocket() throws {
    let socketDirectory = try makePrivateTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: socketDirectory) }
    let socketPath = socketDirectory.appendingPathComponent("helper.sock").path
    let server = UnixSocketServer(
        socketPath: socketPath,
        service: HelperService()
    )

    let queue = DispatchQueue(label: "macos-vz-helper-test-server")
    queue.async {
        try? server.run()
    }
    defer {
        server.stop()
    }

    try waitForSocket(at: socketPath)

    let responseData = try sendSocketRequest(
        socketPath: socketPath,
        payload: """
        {"operation":"ping","protocol_version":"1","request":{}}
        """ + "\n"
    )
    let responseJSON = try JSONSerialization.jsonObject(with: responseData) as? [String: Any]

    #expect(responseJSON?["protocol_version"] as? String == "1")
    #expect(responseJSON?["helper_version"] as? String == "0.1.0")
    #expect(responseJSON?["status"] as? String == "ok")
}

@Test func unixSocketServerRefusesExistingRegularFileSocketPath() throws {
    let dir = try makePrivateTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: dir) }

    let socket = dir.appendingPathComponent("helper.sock")
    try "do not remove".write(to: socket, atomically: true, encoding: .utf8)
    let server = UnixSocketServer(socketPath: socket.path, service: HelperService())
    defer { server.stop() }

    #expect(throws: UnixSocketServerError.self) {
        try server.start()
    }
    #expect(FileManager.default.fileExists(atPath: socket.path))
}

@Test func unixSocketServerRefusesSymlinkSocketPath() throws {
    let dir = try makePrivateTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: dir) }

    let target = dir.appendingPathComponent("target")
    let socket = dir.appendingPathComponent("helper.sock")
    try "target".write(to: target, atomically: true, encoding: .utf8)
    try FileManager.default.createSymbolicLink(atPath: socket.path, withDestinationPath: target.path)

    let server = UnixSocketServer(socketPath: socket.path, service: HelperService())
    defer { server.stop() }

    #expect(throws: UnixSocketServerError.self) {
        try server.start()
    }
    #expect(FileManager.default.fileExists(atPath: socket.path))
}

@Test func unixSocketServerRemovesExistingSocketPath() throws {
    let socketDirectory = try makePrivateTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: socketDirectory) }
    let socketPath = socketDirectory.appendingPathComponent("helper.sock").path
    let fd = Darwin.socket(AF_UNIX, SOCK_STREAM, 0)
    guard fd >= 0 else { return }
    defer {
        close(fd)
    }

    try bindSocketForTest(fd: fd, path: socketPath)
    let server = UnixSocketServer(socketPath: socketPath, service: HelperService())
    try server.start()
    defer { server.stop() }

    #expect(FileManager.default.fileExists(atPath: socketPath))
}

@Test func unixSocketServerRefusesActiveSocketPath() throws {
    let socketDirectory = try makePrivateTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: socketDirectory) }
    let socketPath = socketDirectory.appendingPathComponent("helper.sock").path
    let fd = Darwin.socket(AF_UNIX, SOCK_STREAM, 0)
    guard fd >= 0 else { return }
    defer {
        close(fd)
    }

    try bindSocketForTest(fd: fd, path: socketPath)
    guard Darwin.listen(fd, SOMAXCONN) == 0 else {
        throw TestFailure.socketListenFailed
    }

    do {
        let server = UnixSocketServer(socketPath: socketPath, service: HelperService())
        #expect(throws: UnixSocketServerError.self) {
            try server.start()
        }
    }

    let acceptedFD = try acceptConnectionForTest(fd: fd)
    defer { close(acceptedFD) }
    setSocketReceiveTimeoutForTest(acceptedFD)
    var buffer = [UInt8](repeating: 0, count: 512)
    let readCount = recv(acceptedFD, &buffer, buffer.count, 0)
    #expect(readCount <= 0)

    let clientFD = Darwin.socket(AF_UNIX, SOCK_STREAM, 0)
    guard clientFD >= 0 else { return }
    defer { close(clientFD) }

    let address = try unixSocketAddress(path: socketPath)
    let connectResult = withUnsafePointer(to: address.value) { pointer in
        pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockaddrPointer in
            connect(clientFD, sockaddrPointer, address.length)
        }
    }
    #expect(connectResult == 0)
}

@Test func unixSocketServerDoesNotRemoveReplacementPathOnStop() throws {
    let socketDirectory = try makePrivateTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: socketDirectory) }
    let socketPath = socketDirectory.appendingPathComponent("helper.sock").path
    let replacementURL = URL(fileURLWithPath: socketPath)
    let server = UnixSocketServer(socketPath: socketPath, service: HelperService())
    try server.start()
    defer {
        server.stop()
    }

    unlink(socketPath)
    try "replacement".write(to: replacementURL, atomically: true, encoding: .utf8)

    server.stop()

    #expect(FileManager.default.fileExists(atPath: socketPath))
    #expect(try String(contentsOf: replacementURL, encoding: .utf8) == "replacement")
}

@Test func unixSocketServerCreatesMissingSocketParentAsOwnerOnly() throws {
    let root = try makePrivateTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let runtimeDirectory = root.appendingPathComponent("runtime")
    let socketPath = runtimeDirectory.appendingPathComponent("helper.sock").path
    let server = UnixSocketServer(socketPath: socketPath, service: HelperService())

    try server.start()
    defer { server.stop() }

    let permissions = try socketParentPermissions(at: runtimeDirectory)
    #expect(permissions == 0o700)
}

@Test func unixSocketServerRefusesGroupAccessibleSocketParent() throws {
    let dir = URL(fileURLWithPath: NSTemporaryDirectory())
        .appendingPathComponent("macos-vz-helper-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    chmod(dir.path, 0o755)
    defer { try? FileManager.default.removeItem(at: dir) }

    let socket = dir.appendingPathComponent("helper.sock")
    let server = UnixSocketServer(socketPath: socket.path, service: HelperService())
    defer { server.stop() }

    #expect(throws: UnixSocketServerError.self) {
        try server.start()
    }
    #expect(!FileManager.default.fileExists(atPath: socket.path))
}

@Test func unixSocketServerRefusesMissingSocketParentUnderGroupAccessibleParent() throws {
    let root = try makePrivateTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let sharedParent = root.appendingPathComponent("shared")
    try FileManager.default.createDirectory(at: sharedParent, withIntermediateDirectories: true)
    chmod(sharedParent.path, 0o755)

    let runtimeDirectory = sharedParent.appendingPathComponent("runtime")
    let socket = runtimeDirectory.appendingPathComponent("helper.sock")
    let server = UnixSocketServer(socketPath: socket.path, service: HelperService())
    defer { server.stop() }

    #expect(throws: UnixSocketServerError.self) {
        try server.start()
    }
    #expect(!FileManager.default.fileExists(atPath: runtimeDirectory.path))
}

@Test func unixSocketServerRefusesMissingSocketParentThroughSymlinkAncestor() throws {
    let root = try makePrivateTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let target = root.appendingPathComponent("target")
    let link = root.appendingPathComponent("link")
    try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
    chmod(target.path, 0o700)
    try FileManager.default.createSymbolicLink(atPath: link.path, withDestinationPath: target.path)

    let runtimeDirectory = link.appendingPathComponent("runtime")
    let socket = runtimeDirectory.appendingPathComponent("helper.sock")
    let server = UnixSocketServer(socketPath: socket.path, service: HelperService())
    defer { server.stop() }

    #expect(throws: UnixSocketServerError.self) {
        try server.start()
    }
    #expect(!FileManager.default.fileExists(atPath: target.appendingPathComponent("runtime").path))
}

private func makePrivateTemporaryDirectory() throws -> URL {
    let directory = URL(fileURLWithPath: "/tmp")
        .appendingPathComponent("macos-vz-helper-\(UUID().uuidString)")
    try FileManager.default.createDirectory(
        at: directory,
        withIntermediateDirectories: true,
        attributes: [.posixPermissions: 0o700]
    )
    chmod(directory.path, 0o700)
    return directory
}

private func socketParentPermissions(at url: URL) throws -> Int {
    let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
    let permissions = attributes[.posixPermissions] as? NSNumber
    return permissions?.intValue ?? -1
}

private func acceptConnectionForTest(fd: Int32) throws -> Int32 {
    let acceptedFD = Darwin.accept(fd, nil, nil)
    guard acceptedFD >= 0 else {
        throw TestFailure.socketAcceptFailed
    }
    return acceptedFD
}

private func setSocketReceiveTimeoutForTest(_ fd: Int32) {
    var timeout = timeval(tv_sec: 0, tv_usec: 200_000)
    withUnsafePointer(to: &timeout) { pointer in
        pointer.withMemoryRebound(to: UInt8.self, capacity: MemoryLayout<timeval>.size) { rawPointer in
            _ = setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, rawPointer, socklen_t(MemoryLayout<timeval>.size))
        }
    }
}

private func waitForSocket(at path: String, timeoutSeconds: TimeInterval = 2.0) throws {
    let deadline = Date().addingTimeInterval(timeoutSeconds)
    while Date() < deadline {
        if FileManager.default.fileExists(atPath: path) {
            return
        }
        Thread.sleep(forTimeInterval: 0.05)
    }
    throw TestFailure.socketNotReady
}

private func bindSocketForTest(fd: Int32, path: String) throws {
    let address = try unixSocketAddress(path: path)
    let bindResult = withUnsafePointer(to: address.value) { pointer in
        pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockaddrPointer in
            Darwin.bind(fd, sockaddrPointer, address.length)
        }
    }
    guard bindResult == 0 else {
        throw TestFailure.socketBindFailed
    }
}

private func sendSocketRequest(socketPath: String, payload: String) throws -> Data {
    let clientFD = Darwin.socket(AF_UNIX, SOCK_STREAM, 0)
    guard clientFD >= 0 else {
        throw TestFailure.clientSocketUnavailable
    }
    defer {
        close(clientFD)
    }

    let address = try unixSocketAddress(path: socketPath)
    let connectResult = withUnsafePointer(to: address.value) { pointer in
        pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockaddrPointer in
            connect(clientFD, sockaddrPointer, address.length)
        }
    }
    guard connectResult == 0 else {
        throw TestFailure.socketConnectFailed
    }

    let payloadData = Data(payload.utf8)
    try payloadData.withUnsafeBytes { rawBuffer in
        guard let baseAddress = rawBuffer.baseAddress else {
            return
        }
        if Darwin.write(clientFD, baseAddress, rawBuffer.count) < 0 {
            throw TestFailure.socketWriteFailed
        }
    }

    var buffer = [UInt8](repeating: 0, count: 4096)
    let readCount = recv(clientFD, &buffer, buffer.count, 0)
    guard readCount > 0 else {
        throw TestFailure.socketReadFailed
    }
    return Data(buffer.prefix(readCount)).trimmingTrailingNewlines()
}

private func unixSocketAddress(path: String) throws -> (value: sockaddr_un, length: socklen_t) {
    var address = sockaddr_un()
    address.sun_family = sa_family_t(AF_UNIX)
    let pathBytes = Array(path.utf8CString)
    let maxLength = MemoryLayout.size(ofValue: address.sun_path)
    if pathBytes.count > maxLength {
        throw TestFailure.socketPathTooLong
    }
    withUnsafeMutablePointer(to: &address.sun_path.0) { pointer in
        pointer.initialize(repeating: 0, count: maxLength)
        for (index, byte) in pathBytes.enumerated() {
            pointer.advanced(by: index).pointee = byte
        }
    }
    return (address, socklen_t(MemoryLayout<sa_family_t>.size + pathBytes.count))
}

private enum TestFailure: Error {
    case socketAcceptFailed
    case socketBindFailed
    case clientSocketUnavailable
    case socketConnectFailed
    case socketListenFailed
    case socketNotReady
    case socketPathTooLong
    case socketReadFailed
    case socketWriteFailed
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
