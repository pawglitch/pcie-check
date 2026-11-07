#!/usr/bin/env bash
# pcie-check.sh v0.1.0 — test a GPU's PCIe link (riser / M.2 adapter / slot) for speed AND integrity.
#
# Usage:   sudo bash pcie-check.sh [gpu_index] [minutes]
#          sudo bash pcie-check.sh 1 15      # GPU 1, 15-minute soak
#
# Run it with the box UNRENTED (unlist it on Vast first) — it fully loads the GPU's link.
# Best practice: run once with the card straight in a slot, once on the adapter, compare.
#
# What it checks:
#   1. Negotiated link (gen + width) under load vs. what the card supports
#   2. Real host<->GPU bandwidth (pinned-memory copies)
#   3. Data integrity: random data GPU -> host -> GPU, compared bit-for-bit, looped for N minutes
#   4. PCIe error counters before/after: NVIDIA replay counter + kernel AER counters + Xid/AER log lines
#
# PCIe has CRC + automatic retry, so a bad link almost never corrupts data silently — it shows up
# as replays / AER errors first. Those counters climbing is the real "this adapter is marginal" signal.

set -u
VERSION=0.1.0
GPU=${1:-0}
MIN=${2:-10}
IMG=${PCIE_CHECK_IMAGE:-pytorch/pytorch:latest}

command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found"; exit 1; }
command -v docker >/dev/null || { echo "docker not found (Vast hosts have it; otherwise install it)"; exit 1; }

q() { nvidia-smi -i "$GPU" --query-gpu="$1" --format=csv,noheader,nounits | tr -d ' '; }

NAME=$(nvidia-smi -i "$GPU" --query-gpu=name --format=csv,noheader)
BUSID=$(q pci.bus_id)                       # e.g. 00000000:01:00.0
BDF=$(echo "${BUSID: -12}" | tr 'A-F' 'a-f') # -> 0000:01:00.0
DEV=/sys/bus/pci/devices/$BDF
PARENT=$(basename "$(dirname "$(readlink -f "$DEV")")")   # upstream port the card hangs off

replays() { nvidia-smi -q -i "$GPU" | awk -F: '/Replays Since Reset/{gsub(/ /,"",$2);print $2; exit}'; }
aer_total() { # $1 = BDF; prints TOTAL_ERR_COR or "n/a"
  local f=/sys/bus/pci/devices/$1/aer_dev_correctable
  [ -r "$f" ] && awk '/TOTAL_ERR_COR/{print $2}' "$f" || echo "n/a"
}

echo "pcie-check v$VERSION"
echo "=== $NAME  (GPU $GPU, $BDF, upstream port $PARENT) ==="
MAXGEN=$(q pcie.link.gen.max); MAXW=$(q pcie.link.width.max)
echo "Card/slot supports: Gen$MAXGEN x$MAXW"

R0=$(replays); A0=$(aer_total "$BDF"); P0=$(aer_total "$PARENT")
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "Before: replays=$R0  AER(card)=$A0  AER(port)=$P0"
echo
echo "Running ${MIN}-min test in $IMG (first run downloads the image)..."

docker run --rm --gpus "device=$GPU" -i "$IMG" python - "$MIN" <<'PY' &
import torch, time, sys
mins = float(sys.argv[1])
dev = "cuda"
size = 512 * 1024**2                      # 512 MiB per transfer
host = torch.empty(size, dtype=torch.uint8).pin_memory()
a = torch.empty(size, dtype=torch.uint8, device=dev)
b = torch.empty(size, dtype=torch.uint8, device=dev)
a.random_(0, 256)

def bw(direction, iters=30):
    torch.cuda.synchronize(); t = time.time()
    for _ in range(iters):
        if direction == "h2d": a.copy_(host, non_blocking=True)
        else:                  host.copy_(a, non_blocking=True)
    torch.cuda.synchronize()
    return size * iters / (time.time() - t) / 1e9

for _ in range(3): bw("h2d", 5); bw("d2h", 5)   # warm-up
print(f"Bandwidth  host->GPU: {bw('h2d'):5.1f} GB/s   GPU->host: {bw('d2h'):5.1f} GB/s", flush=True)
print("  (typical: Gen3x8 ~6.5 | Gen3x16/Gen4x8/Gen5x4 ~12-13 | Gen4x16/Gen5x8 ~25 | Gen5x16 ~50)", flush=True)

end = time.time() + mins * 60
rounds = bad = 0
while time.time() < end:
    a.random_(0, 256)
    host.copy_(a)          # GPU -> host
    b.copy_(host)          # host -> GPU
    if not torch.equal(a, b):
        bad += 1
    rounds += 1
moved = rounds * 2 * size / 1e12
print(f"Integrity: {rounds} round trips, {moved:.2f} TB moved, {bad} mismatched transfers", flush=True)
sys.exit(2 if bad else 0)
PY
DPID=$!

sleep 20   # sample link state while it's under load (GPUs drop to Gen1 at idle)
CURGEN=$(q pcie.link.gen.current); CURW=$(q pcie.link.width.current)
echo "Link under load: Gen$CURGEN x$CURW  (max Gen$MAXGEN x$MAXW)"

wait $DPID; RC=$?

R1=$(replays); A1=$(aer_total "$BDF"); P1=$(aer_total "$PARENT")
echo
echo "After:  replays=$R1  AER(card)=$A1  AER(port)=$P1"
echo "Kernel log since start (Xid / AER / PCIe errors):"
journalctl -k --since "$START" 2>/dev/null | grep -iE "xid|aer|pcie bus error|corrected error" | tail -n 20 || true

echo
echo "=== VERDICT ==="
FAIL=0; WARN=0
[ "$RC" = 2 ] && { echo "FAIL  data mismatches seen — do not use this adapter/riser"; FAIL=1; }
[ "$RC" != 0 ] && [ "$RC" != 2 ] && { echo "FAIL  test crashed (GPU may have dropped off the bus — check Xid 79 above)"; FAIL=1; }
if [ -n "$R0" ] && [ -n "$R1" ] && [ "$R1" -gt "$R0" ] 2>/dev/null; then
  echo "WARN  PCIe replays went up by $((R1-R0)) — link is marginal; try forcing one gen lower in BIOS"; WARN=1; fi
for pair in "card:$A0:$A1" "port:$P0:$P1"; do
  IFS=: read -r who x y <<<"$pair"
  if [ "$x" != "n/a" ] && [ "$y" != "n/a" ] && [ "$y" -gt "$x" ] 2>/dev/null; then
    echo "WARN  AER correctable errors on $who went up by $((y-x))"; WARN=1; fi
done
if [ "$CURGEN" -lt "$MAXGEN" ] 2>/dev/null || [ "$CURW" -lt "$MAXW" ] 2>/dev/null; then
  echo "WARN  link trained below max (Gen$CURGEN x$CURW vs Gen$MAXGEN x$MAXW) — expected if the slot/adapter is narrower, otherwise a bad connection"; WARN=1; fi
[ "$FAIL" = 0 ] && [ "$WARN" = 0 ] && echo "PASS  full speed, zero errors, zero corruption"
[ "$FAIL" = 0 ] && [ "$WARN" = 1 ] && echo "Data was intact, but see warnings above"
exit $FAIL
