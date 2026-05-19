import Darwin
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
    private let queue: DispatchQueue
    private var readSource: DispatchSourceRead?
    private var buffer = Data()

    init(connection: VZVirtioSocketConnection) {
        self.connection = connection
        self.queue = DispatchQueue(label: "vz.helper.vsock.\(UUID().uuidString)")
    }

    func startReading(_ handler: @escaping (Result<Data, Error>) -> Void) {
        let source = DispatchSource.makeReadSource(fileDescriptor: connection.fileDescriptor, queue: queue)
        source.setEventHandler { [weak self] in
            guard let self else { return }
            var chunk = [UInt8](repeating: 0, count: 4096)
            let readCount = Darwin.read(self.connection.fileDescriptor, &chunk, chunk.count)
            if readCount < 0 {
                if errno == EINTR || errno == EAGAIN {
                    return
                }
                handler(.failure(VSockSessionError.invalidMessage("read_error_\(errno)")))
                return
            }
            if readCount == 0 {
                handler(.failure(VSockSessionError.closed))
                return
            }

            chunk.withUnsafeBufferPointer { pointer in
                if let baseAddress = pointer.baseAddress {
                    self.buffer.append(baseAddress, count: readCount)
                }
            }
            while let newlineRange = self.buffer.firstRange(of: Data([0x0A])) {
                let line = self.buffer.subdata(in: 0..<newlineRange.lowerBound)
                self.buffer.removeSubrange(0...newlineRange.lowerBound)
                handler(.success(line))
            }
        }
        source.setCancelHandler { [connection] in
            connection.close()
        }
        readSource = source
        source.resume()
    }

    func writeLine(_ data: Data) throws {
        var line = data
        line.append(0x0A)
        try line.withUnsafeBytes { rawBuffer in
            guard let baseAddress = rawBuffer.baseAddress else {
                return
            }
            var bytesWritten = 0
            while bytesWritten < line.count {
                let result = Darwin.write(
                    connection.fileDescriptor,
                    baseAddress.advanced(by: bytesWritten),
                    line.count - bytesWritten
                )
                if result < 0 {
                    if errno == EINTR || errno == EAGAIN {
                        continue
                    }
                    throw VSockSessionError.invalidMessage("write_error_\(errno)")
                }
                if result == 0 {
                    throw VSockSessionError.closed
                }
                bytesWritten += result
            }
        }
    }

    func close() {
        readSource?.cancel()
        readSource = nil
        connection.close()
    }
}
