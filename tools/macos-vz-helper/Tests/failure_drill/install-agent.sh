# Test-only: /workspace is an OFFLINE disposable clone, never the boot disk.
e2fsck -fn /workspace/rootfs.img
debugfs -w -R 'rm /usr/local/bin/tldw-agent-guest' /workspace/rootfs.img
debugfs -w -R 'write /workspace/fault-agent /usr/local/bin/tldw-agent-guest' /workspace/rootfs.img
debugfs -w -R 'set_inode_field /usr/local/bin/tldw-agent-guest mode 0100755' /workspace/rootfs.img
debugfs -R 'dump /usr/local/bin/tldw-agent-guest /workspace/installed-agent' /workspace/rootfs.img
cmp /workspace/fault-agent /workspace/installed-agent
e2fsck -fn /workspace/rootfs.img
sha256sum /workspace/rootfs.img /workspace/installed-agent
