#!/bin/sh
# Test-only agent replacement inside a disposable Debian rootfs.
set -eu

workspace="${TLDW_AGENT_GUEST_WORKSPACE_ROOT:-/workspace}"
challenge="$workspace/.tldw-missing-agent-challenge"
proof="$workspace/.tldw-missing-agent-proof.json"
proof_tmp="$proof.$$"
IFS= read -r nonce < "$challenge"
mode="$(sed -n '2p' "$challenge")"
vm_id="${TLDW_AGENT_GUEST_VM_ID:-}"
case "$nonce" in
    ""|*[!0123456789abcdef]*) exit 1 ;;
esac
case "$vm_id" in
    ""|*[!ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-]*) exit 1 ;;
esac
case "$mode" in
    no-agent|start-original) ;;
    *) exit 1 ;;
esac
printf '{"nonce":"%s","vm_id":"%s","mode":"%s"}\n' "$nonce" "$vm_id" "$mode" > "$proof_tmp"
mv "$proof_tmp" "$proof"
if [ "$mode" = start-original ]; then
    exec /usr/local/bin/tldw-agent-guest-original
fi
exec /bin/sleep 120
