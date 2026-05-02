//go:build !linux

package guest

import "os/exec"

func configureCommandProcessGroup(_ *exec.Cmd) {}

func terminateCommandProcess(cmd *exec.Cmd) bool {
	if cmd != nil && cmd.Process != nil {
		return cmd.Process.Kill() == nil
	}
	return false
}
