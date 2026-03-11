import Foundation
import Virtualization

final class VSockListener: NSObject, VZVirtioSocketListenerDelegate {
    let vmID: String
    let listener: VZVirtioSocketListener
    private weak var manager: VSockSessionManager?

    init(
        vmID: String,
        manager: VSockSessionManager,
        listener: VZVirtioSocketListener = VZVirtioSocketListener()
    ) {
        self.vmID = vmID
        self.manager = manager
        self.listener = listener
        super.init()
        self.listener.delegate = self
    }

    func listener(
        _ listener: VZVirtioSocketListener,
        shouldAcceptNewConnection connection: VZVirtioSocketConnection,
        from socketDevice: VZVirtioSocketDevice
    ) -> Bool {
        guard let manager else {
            connection.close()
            return false
        }
        return manager.accept(
            channel: FileHandleVSockChannel(connection: connection),
            for: vmID
        )
    }
}

final class FileHandleVSockChannel: VSockChanneling {
    private let connection: VZVirtioSocketConnection
    private let fileHandle: FileHandle
    private let queue: DispatchQueue
    private var readSource: DispatchSourceRead?
    private var buffer = Data()

    init(connection: VZVirtioSocketConnection) {
        self.connection = connection
        self.fileHandle = FileHandle(fileDescriptor: connection.fileDescriptor, closeOnDealloc: false)
        self.queue = DispatchQueue(label: "vz.helper.vsock.\(UUID().uuidString)")
    }

    func startReading(_ handler: @escaping (Result<Data, Error>) -> Void) {
        let source = DispatchSource.makeReadSource(fileDescriptor: connection.fileDescriptor, queue: queue)
        source.setEventHandler { [weak self] in
            guard let self else { return }
            do {
                let chunk = try self.fileHandle.read(upToCount: 4096) ?? Data()
                if chunk.isEmpty {
                    handler(.failure(VSockSessionError.closed))
                    return
                }
                self.buffer.append(chunk)
                while let newlineRange = self.buffer.firstRange(of: Data([0x0A])) {
                    let line = self.buffer.subdata(in: 0..<newlineRange.lowerBound)
                    self.buffer.removeSubrange(0...newlineRange.lowerBound)
                    handler(.success(line))
                }
            } catch {
                handler(.failure(error))
            }
        }
        source.setCancelHandler { [fileHandle, connection] in
            try? fileHandle.close()
            connection.close()
        }
        readSource = source
        source.resume()
    }

    func writeLine(_ data: Data) throws {
        var line = data
        line.append(0x0A)
        try fileHandle.write(contentsOf: line)
    }

    func close() {
        readSource?.cancel()
        readSource = nil
        connection.close()
    }
}
