//go:build !linux

package guest

import (
	"context"
	"errors"
	"io"
)

func dialVSockConnection(_ context.Context, _ uint32) (io.ReadWriteCloser, error) {
	return nil, errors.New("vsock guest transport is only supported on linux guests")
}
