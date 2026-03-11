//go:build linux

package guest

import (
	"context"
	"fmt"
	"io"
	"net"
	"os"
	"syscall"
	"unsafe"
)

const vmAddrCIDHost = 2

type rawSockaddrVM struct {
	Family    uint16
	Reserved1 uint16
	Port      uint32
	CID       uint32
	Zero      [4]byte
}

type fileBackedConn struct {
	net.Conn
	file *os.File
}

func (c *fileBackedConn) Close() error {
	connErr := c.Conn.Close()
	fileErr := c.file.Close()
	if connErr != nil {
		return connErr
	}
	return fileErr
}

func dialVSockConnection(ctx context.Context, port uint32) (io.ReadWriteCloser, error) {
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}

	fd, err := syscall.Socket(syscall.AF_VSOCK, syscall.SOCK_STREAM, 0)
	if err != nil {
		return nil, fmt.Errorf("create vsock socket: %w", err)
	}

	addr := rawSockaddrVM{
		Family: syscall.AF_VSOCK,
		Port:   port,
		CID:    vmAddrCIDHost,
	}

	_, _, errno := syscall.Syscall(
		syscall.SYS_CONNECT,
		uintptr(fd),
		uintptr(unsafe.Pointer(&addr)),
		unsafe.Sizeof(addr),
	)
	if errno != 0 {
		_ = syscall.Close(fd)
		return nil, os.NewSyscallError("connect", errno)
	}

	file := os.NewFile(uintptr(fd), fmt.Sprintf("vsock:%d", port))
	conn, err := net.FileConn(file)
	if err != nil {
		_ = file.Close()
		return nil, fmt.Errorf("wrap vsock connection: %w", err)
	}

	return &fileBackedConn{
		Conn: conn,
		file: file,
	}, nil
}
