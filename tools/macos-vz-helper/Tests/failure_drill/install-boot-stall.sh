# Test-only: inspect the OFFLINE disposable initrd, without modifying rootfs.
name=$(cat /workspace/initrd-name)
case "$name" in
    ''|*/*|.|..) exit 1 ;;
esac
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
unmkinitramfs "/workspace/$name" "$scratch"
tree=$scratch/main
if [ ! -d "$tree" ]; then tree=$scratch; fi
test -f "$tree/init"
test ! -L "$tree/init"
cp "$tree/init" /workspace/original-init
# The existing preparer contract verifies the submitted fixture independently.
cp /workspace/fault-agent /workspace/installed-agent
cmp /workspace/fault-agent /workspace/installed-agent
