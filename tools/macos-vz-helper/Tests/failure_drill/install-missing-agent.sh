# Test-only: /workspace is an OFFLINE disposable clone, never the boot disk.
# Replay the clone's journal before raw writes, so boot cannot undo new inodes.
status=0
e2fsck -p /workspace/rootfs.img || status=$?
case "$status" in 0|1) ;; *) exit "$status" ;; esac
e2fsck -fn /workspace/rootfs.img
debugfs -R 'dump /usr/local/bin/tldw-agent-guest /workspace/original-agent' /workspace/rootfs.img
debugfs -w -R 'write /workspace/original-agent /usr/local/bin/tldw-agent-guest-original' /workspace/rootfs.img
debugfs -w -R 'set_inode_field /usr/local/bin/tldw-agent-guest-original mode 0100755' /workspace/rootfs.img
debugfs -R 'dump /usr/local/bin/tldw-agent-guest-original /workspace/preserved-agent' /workspace/rootfs.img
cmp /workspace/original-agent /workspace/preserved-agent
debugfs -w -R 'rm /usr/local/bin/tldw-agent-guest' /workspace/rootfs.img
debugfs -w -R 'write /workspace/fault-agent /usr/local/bin/tldw-agent-guest' /workspace/rootfs.img
debugfs -w -R 'set_inode_field /usr/local/bin/tldw-agent-guest mode 0100755' /workspace/rootfs.img
debugfs -R 'dump /usr/local/bin/tldw-agent-guest /workspace/installed-agent' /workspace/rootfs.img
cmp /workspace/fault-agent /workspace/installed-agent
e2fsck -fn /workspace/rootfs.img
sha256sum /workspace/rootfs.img /workspace/installed-agent /workspace/preserved-agent
