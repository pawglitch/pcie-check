"""
pcie-check.py v0.1.0 - Windows version. Tests a GPU's PCIe link (slot, riser or M.2 adapter)
for speed AND integrity.

Usage (or just double-click RUN-pcie-check.cmd):
    py pcie-check.py [gpu_index] [minutes]

Checks:
  1. Negotiated link (gen + width) while under load vs. the max for this card + slot
  2. Real host<->GPU bandwidth (pinned-memory copies)
  3. Data integrity: random data GPU -> host -> GPU compared bit-for-bit, looped for N minutes
  4. Error counters before/after: NVIDIA PCIe replay counter, plus Windows Event Log
     (WHEA-Logger = PCIe bus errors, nvlddmkm = NVIDIA driver errors / Windows "Xid")

PCIe has CRC + automatic retry, so a bad link almost never corrupts data silently - it shows
up as replays / WHEA errors first. Those climbing is the real "this adapter is marginal" sign.
Results are also written to pcie-check-result.txt next to this script.
"""
import os, sys, time, subprocess, threading, datetime, json

VERSION = "0.1.0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # make torch's GPU numbering match nvidia-smi

GPU = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].strip() else 0
MIN = float(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].strip() else 10.0
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "pcie-check-result.txt")
logf = open(LOG, "w", encoding="utf-8")


def out(s=""):
    print(s, flush=True)
    logf.write(s + "\n")
    logf.flush()


def smi(fields):
    r = subprocess.run(
        ["nvidia-smi", "-i", str(GPU), f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True)
    return [x.strip() for x in r.stdout.strip().split(",")]


def replays():
    r = subprocess.run(["nvidia-smi", "-q", "-i", str(GPU)], capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if "Replays Since Reset" in line:
            v = line.split(":", 1)[1].strip()
            return int(v) if v.isdigit() else None
    return None


def win_events(since):
    ps = (
        f"$s=[datetime]'{since.strftime('%Y-%m-%d %H:%M:%S')}'; "
        "Get-WinEvent -FilterHashtable @{LogName='System'; "
        "ProviderName='Microsoft-Windows-WHEA-Logger','nvlddmkm'; StartTime=$s} "
        "-ErrorAction SilentlyContinue | "
        "Select-Object ProviderName,Id,@{n='Msg';e={($_.Message -split \"`n\")[0].Trim()}} | "
        "ConvertTo-Json -Compress"
    )
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True)
    s = r.stdout.strip()
    if not s:
        return []
    try:
        d = json.loads(s)
        return d if isinstance(d, list) else [d]
    except Exception:
        return [{"ProviderName": "?", "Id": "?", "Msg": s[:300]}]


def main():
    try:
        import torch
    except ImportError:
        out(f"PyTorch isn't installed for this Python ({sys.executable}).")
        out("Install:  py -m pip install torch --index-url https://download.pytorch.org/whl/cu128")
        return 1
    if not torch.cuda.is_available():
        out("PyTorch can't see the GPU (CPU-only build?). Reinstall with the cu128 command above.")
        return 1

    name, maxgen, maxw = smi("name,pcie.link.gen.max,pcie.link.width.max")
    bus = smi("pci.bus_id")[0]
    maxgen, maxw = int(maxgen), int(maxw)
    out(f"pcie-check v{VERSION}")
    out(f"=== {name}  (GPU {GPU}, {bus}) ===")
    out(f"Max for this card + slot: Gen{maxgen} x{maxw}")

    r0 = replays()
    start = datetime.datetime.now()
    out(f"Before: PCIe replays = {r0}")
    out(f"Running {MIN:g}-minute test...\n")

    samples = []
    stop = threading.Event()

    def sampler():
        time.sleep(3)
        while not stop.is_set():
            try:
                g, w = smi("pcie.link.gen.current,pcie.link.width.current")
                samples.append((int(g), int(w)))
            except Exception:
                pass
            stop.wait(5)

    threading.Thread(target=sampler, daemon=True).start()

    crashed = None
    rounds = bad = 0
    size = 512 * 1024**2
    try:
        dev = torch.device(f"cuda:{GPU}")
        host = torch.empty(size, dtype=torch.uint8).pin_memory()
        a = torch.empty(size, dtype=torch.uint8, device=dev)
        b = torch.empty(size, dtype=torch.uint8, device=dev)
        a.random_(0, 256)

        def bw(direction, iters=30):
            torch.cuda.synchronize(dev)
            t = time.time()
            for _ in range(iters):
                if direction == "h2d":
                    a.copy_(host, non_blocking=True)
                else:
                    host.copy_(a, non_blocking=True)
            torch.cuda.synchronize(dev)
            return size * iters / (time.time() - t) / 1e9

        for _ in range(3):
            bw("h2d", 5); bw("d2h", 5)
        out(f"Bandwidth  host->GPU: {bw('h2d'):5.1f} GB/s   GPU->host: {bw('d2h'):5.1f} GB/s")
        out("  (typical: Gen3x8/Gen4x4 ~6.5 | Gen3x16/Gen4x8/Gen5x4 ~12-13 | "
            "Gen4x16/Gen5x8 ~25 | Gen5x16 ~50)")

        end = time.time() + MIN * 60
        next_report = time.time() + 60
        while time.time() < end:
            a.random_(0, 256)
            host.copy_(a)      # GPU -> host
            b.copy_(host)      # host -> GPU
            if not torch.equal(a, b):
                bad += 1
            rounds += 1
            if time.time() >= next_report:
                left = max(0, int((end - time.time()) / 60 + 0.5))
                out(f"  ...{rounds} round trips, {bad} mismatches, ~{left} min left")
                next_report += 60
    except Exception as e:      # device lost / driver reset shows up here
        crashed = repr(e)
    finally:
        stop.set()

    moved = rounds * 2 * size / 1e12
    out(f"Integrity: {rounds} round trips, {moved:.2f} TB moved, {bad} mismatched transfers")

    time.sleep(1)
    r1 = replays()
    events = win_events(start)
    cur = max(samples) if samples else (None, None)
    out(f"Link under load: Gen{cur[0]} x{cur[1]}  (max Gen{maxgen} x{maxw})")
    out(f"After:  PCIe replays = {r1}")
    if events:
        out(f"Windows event log since start ({len(events)} WHEA / NVIDIA driver events):")
        for ev in events[:15]:
            out(f"  [{ev.get('ProviderName')} {ev.get('Id')}] {ev.get('Msg')}")
    else:
        out("Windows event log since start: no WHEA or NVIDIA driver errors")

    out("\n=== VERDICT ===")
    fail = warn = False
    if crashed:
        out(f"FAIL  test crashed - GPU likely dropped off the bus: {crashed}"); fail = True
    if bad:
        out("FAIL  data mismatches seen - do not use this adapter/riser"); fail = True
    if r0 is not None and r1 is not None and r1 > r0:
        out(f"WARN  PCIe replays went up by {r1 - r0} - link is marginal; "
            "force one gen lower in BIOS and retest"); warn = True
    whea = [e for e in events if "WHEA" in str(e.get("ProviderName"))]
    nv = [e for e in events if "nvlddmkm" in str(e.get("ProviderName"))]
    if whea:
        out(f"WARN  {len(whea)} WHEA hardware error event(s) logged (PCIe bus errors)"); warn = True
    if nv:
        out(f"WARN  {len(nv)} NVIDIA driver error event(s) logged"); warn = True
    if cur[0] is not None and (cur[0] < maxgen or cur[1] < maxw):
        out(f"WARN  link trained below max (Gen{cur[0]} x{cur[1]} vs Gen{maxgen} x{maxw}) - "
            "expected if the adapter is narrower (M.2 = x4), otherwise a bad connection"); warn = True
    if not fail and not warn:
        out("PASS  full speed, zero errors, zero corruption")
    elif not fail:
        out("Data was intact, but see warnings above")
    out(f"\nSaved to {LOG}")
    return 1 if fail else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        logf.close()
