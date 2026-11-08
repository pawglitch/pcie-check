#!/usr/bin/env python3
"""
pcie-check v0.2.0 - test a GPU's PCIe link (slot, riser, M.2 adapter, OCuLink...) for
speed AND integrity. One script for Linux and Windows.

Usage:
    python3 pcie-check.py [gpu|all] [minutes] [options]      (Linux: run with sudo)
    py pcie-check.py 0 10                                    (Windows, or RUN-pcie-check.cmd)

    python3 pcie-check.py --list            show GPUs and their current/max link
    python3 pcie-check.py all 5             test every GPU, one after another
    python3 pcie-check.py 1 15 --json r.json

Checks:
  1. Negotiated link (gen + width) under load vs. the max for this card + slot
  2. Real host<->GPU bandwidth (pinned-memory copies), and whether it fits the link
  3. Data integrity: random data GPU -> host -> GPU compared bit-for-bit, looped for N minutes
  4. Error counters before/after: NVIDIA PCIe replay counter, plus
       Linux:   kernel AER counters on the card and its upstream port, Xid/AER kernel log lines
       Windows: WHEA-Logger (PCIe bus errors) and nvlddmkm (NVIDIA driver) System-log events

PCIe has CRC + automatic retry, so a bad link almost never corrupts data silently - it shows
up as replays / AER / WHEA errors first. Those counters climbing is the real "marginal" signal.

The transfer test needs PyTorch with CUDA. It uses the PyTorch of the Python running this
script if it has one; on Linux it otherwise runs inside a pytorch/pytorch Docker container.
Everything else uses only the standard library.
"""
import argparse
import datetime
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time

VERSION = "0.2.0"
IS_WIN = os.name == "nt"
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMAGE = "pytorch/pytorch:latest"

# Usable GB/s per lane, per direction, after line encoding (8b/10b for Gen1-2, 128b/130b after).
PER_LANE = {1: 0.25, 2: 0.5, 3: 0.985, 4: 1.969, 5: 3.938, 6: 7.563}
# Real pinned copies reach ~75-85% of that. Below this fraction we call the bandwidth low.
BW_FLOOR = 0.60


# --------------------------------------------------------------------------------------------
# Worker: the only part that needs torch. Runs as a child process (host or Docker) with the
# GPU under test as its only visible device, and reports JSON lines on stdout. Keeping it in
# a child means a GPU that falls off the bus kills the worker, not the run.
# --------------------------------------------------------------------------------------------
def worker(minutes, size_mb):
    def emit(**kw):
        print(json.dumps(kw), flush=True)

    try:
        import torch
    except ImportError:
        emit(type="error", msg="PyTorch is not installed for " + sys.executable)
        return 3
    if not torch.cuda.is_available():
        emit(type="error", msg="PyTorch can't see the GPU (CPU-only build, or driver problem)")
        return 3

    dev = torch.device("cuda:0")
    size = size_mb * 1024**2
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
        bw("h2d", 5)
        bw("d2h", 5)
    emit(type="bw", h2d=round(bw("h2d"), 2), d2h=round(bw("d2h"), 2))

    end = time.time() + minutes * 60
    next_report = time.time() + 60
    rounds = bad = bad_bytes = 0
    while time.time() < end:
        a.random_(0, 256)
        host.copy_(a)       # GPU -> host
        b.copy_(host)       # host -> GPU
        if not torch.equal(a, b):
            bad += 1
            bad_bytes += int((a != b).sum().item())
        rounds += 1
        if time.time() >= next_report:
            emit(type="progress", rounds=rounds, bad=bad,
                 left_min=max(0, round((end - time.time()) / 60)))
            next_report += 60
    emit(type="done", rounds=rounds, bad=bad, bad_bytes=bad_bytes, bytes=rounds * 2 * size)
    return 2 if bad else 0


# --------------------------------------------------------------------------------------------
# Controller helpers (standard library only)
# --------------------------------------------------------------------------------------------
class Log:
    def __init__(self, path):
        self.path = path
        self.f = open(path, "w", encoding="utf-8") if path else None

    def __call__(self, s=""):
        print(s, flush=True)
        if self.f:
            self.f.write(s + "\n")
            self.f.flush()

    def close(self):
        if self.f:
            self.f.close()


def run(cmd, timeout=60):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def as_int(v):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def smi(gpu, fields):
    """nvidia-smi query for one GPU -> dict field -> str ('' if unsupported)."""
    names = fields.split(",")
    out = run(["nvidia-smi", "-i", str(gpu), "--query-gpu=" + fields,
               "--format=csv,noheader,nounits"])
    vals = [x.strip() for x in out.strip().split(",")] if out.strip() else []
    if len(vals) != len(names):
        # An unsupported field (older driver) fails the whole query; retry one by one.
        if len(names) == 1:
            return {names[0]: ""}
        d = {}
        for n in names:
            d.update(smi(gpu, n))
        return d
    return {n: ("" if v.startswith("[") else v) for n, v in zip(names, vals)}


def list_gpus():
    out = run(["nvidia-smi", "--query-gpu=index,name,uuid,pci.bus_id,"
               "pcie.link.gen.current,pcie.link.width.current,"
               "pcie.link.gen.max,pcie.link.width.max", "--format=csv,noheader,nounits"])
    gpus = []
    for line in out.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) == 8:
            gpus.append(dict(index=int(p[0]), name=p[1], uuid=p[2], bus=p[3],
                             gen=as_int(p[4]), width=as_int(p[5]),
                             maxgen=as_int(p[6]), maxwidth=as_int(p[7])))
    return gpus


def replays(gpu):
    for line in run(["nvidia-smi", "-q", "-i", str(gpu)]).splitlines():
        if "Replays Since Reset" in line:
            return as_int(line.split(":", 1)[1])
    return None


def bdf_of(bus):
    # nvidia-smi gives 00000000:01:00.0 ; sysfs wants 0000:01:00.0
    return bus[-12:].lower()


def linux_parent(bdf):
    try:
        return os.path.basename(os.path.dirname(os.path.realpath("/sys/bus/pci/devices/" + bdf)))
    except OSError:
        return None


def linux_aer(bdf):
    try:
        with open(f"/sys/bus/pci/devices/{bdf}/aer_dev_correctable") as f:
            for line in f:
                if line.startswith("TOTAL_ERR_COR"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError, TypeError):
        pass
    return None


def linux_kernel_log(since):
    out = run(["journalctl", "-k", "--no-pager", "--since",
               since.strftime("%Y-%m-%d %H:%M:%S")])
    keys = ("xid", "aer", "pcie bus error", "corrected error", "uncorrected", "fallen off")
    return [l for l in out.splitlines() if any(k in l.lower() for k in keys)]


def win_events(since):
    ps = (
        f"$s=[datetime]'{since.strftime('%Y-%m-%d %H:%M:%S')}'; "
        "Get-WinEvent -FilterHashtable @{LogName='System'; "
        "ProviderName='Microsoft-Windows-WHEA-Logger','nvlddmkm'; StartTime=$s} "
        "-ErrorAction SilentlyContinue | "
        "Select-Object ProviderName,Id,@{n='Msg';e={($_.Message -split \"`n\")[0].Trim()}} | "
        "ConvertTo-Json -Compress"
    )
    s = run(["powershell", "-NoProfile", "-Command", ps]).strip()
    if not s:
        return []
    try:
        d = json.loads(s)
        return d if isinstance(d, list) else [d]
    except ValueError:
        return [{"ProviderName": "?", "Id": "?", "Msg": s[:300]}]


def pick_runner(want):
    if want in ("auto", "host") and importlib.util.find_spec("torch") is not None:
        return "host"
    if want == "host":
        return None
    if not IS_WIN and want in ("auto", "docker") and shutil.which("docker"):
        return "docker"
    return None


def worker_command(runner, uuid, minutes, size_mb, image):
    """-> (argv, env, stdin_text)"""
    wargs = ["--worker", str(minutes), "--size-mb", str(size_mb)]
    if runner == "host":
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid)
        return [sys.executable, os.path.abspath(__file__)] + wargs, env, None
    with open(os.path.abspath(__file__), encoding="utf-8") as f:
        src = f.read()
    argv = ["docker", "run", "--rm", "--gpus", f"device={uuid}", "-i", image, "python", "-"]
    return argv + wargs, None, src


# --------------------------------------------------------------------------------------------
# One GPU
# --------------------------------------------------------------------------------------------
def test_gpu(g, args, runner, out):
    gpu = g["index"]
    res = dict(gpu=gpu, name=g["name"], bus=g["bus"], uuid=g["uuid"], minutes=args.minutes,
               runner=runner, os="windows" if IS_WIN else "linux", version=VERSION)
    caps = smi(gpu, "pcie.link.gen.max,pcie.link.width.max,pcie.link.gen.gpumax,"
                    "pcie.link.gen.hostmax")
    maxgen, maxw = as_int(caps["pcie.link.gen.max"]), as_int(caps["pcie.link.width.max"])
    cardgen, hostgen = as_int(caps["pcie.link.gen.gpumax"]), as_int(caps["pcie.link.gen.hostmax"])
    res.update(max_gen=maxgen, max_width=maxw, card_gen=cardgen, host_gen=hostgen)

    bdf = bdf_of(g["bus"])
    parent = None if IS_WIN else linux_parent(bdf)
    out(f"=== {g['name']}  (GPU {gpu}, {g['bus']}"
        + (f", upstream port {parent}" if parent else "") + ") ===")
    line = f"Max for this card + slot: Gen{maxgen} x{maxw}"
    if cardgen and hostgen and cardgen != hostgen:
        line += f"   (card Gen{cardgen}, slot/host Gen{hostgen})"
    out(line)

    r0 = replays(gpu)
    a0 = None if IS_WIN else linux_aer(bdf)
    p0 = linux_aer(parent) if parent else None
    start = datetime.datetime.now()
    before = f"Before: replays={r0}"
    if not IS_WIN:
        before += f"  AER(card)={a0 if a0 is not None else 'n/a'}" \
                  f"  AER(port)={p0 if p0 is not None else 'n/a'}"
    out(before)
    note = " (first Docker run downloads the image, can take a while)" if runner == "docker" else ""
    out(f"Running {args.minutes:g}-minute test via {runner}{note}...")
    out()

    # Sample the link while the worker runs. GPUs drop to Gen1 at idle, so only samples taken
    # under load mean anything; we mark samples taken during the soak phase.
    samples = []          # (phase, gen, width)
    phase = ["start"]
    stop = threading.Event()

    def sample():
        d = smi(gpu, "pcie.link.gen.current,pcie.link.width.current")
        gg, ww = as_int(d["pcie.link.gen.current"]), as_int(d["pcie.link.width.current"])
        if gg and ww:
            samples.append((phase[0], gg, ww))

    def sampler():
        while not stop.wait(3):
            sample()

    argv, env, stdin_text = worker_command(runner, g["uuid"], args.minutes, args.size_mb,
                                           args.image)
    stderr_tail = []
    bw = done = err = None
    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    try:
        # Docker: let stderr through so image-pull progress is visible.
        p = subprocess.Popen(argv, env=env, stdin=subprocess.PIPE if stdin_text else None,
                             stdout=subprocess.PIPE,
                             stderr=None if runner == "docker" else subprocess.PIPE,
                             text=True, encoding="utf-8", errors="replace")

        def drain_err():
            for l in p.stderr:
                stderr_tail.append(l.rstrip())
                del stderr_tail[:-15]

        if p.stderr:
            threading.Thread(target=drain_err, daemon=True).start()
        if stdin_text:
            p.stdin.write(stdin_text)
            p.stdin.close()
        for raw in p.stdout:
            try:
                m = json.loads(raw)
            except ValueError:
                continue
            t = m.get("type")
            if t == "bw":
                bw = m
                phase[0] = "soak"
                sample()          # the soak starts right now, so the link is loaded
                out(f"Bandwidth  host->GPU: {m['h2d']:5.1f} GB/s   GPU->host: {m['d2h']:5.1f} GB/s")
            elif t == "progress":
                out(f"  ...{m['rounds']} round trips, {m['bad']} mismatches, "
                    f"~{m['left_min']} min left")
            elif t == "done":
                done = m
            elif t == "error":
                err = m["msg"]
        rc = p.wait()
    except OSError as e:
        rc, err = -1, f"couldn't start worker: {e}"
    finally:
        stop.set()
        th.join(timeout=10)

    if done:
        out(f"Integrity: {done['rounds']} round trips, {done['bytes'] / 1e12:.2f} TB moved, "
            f"{done['bad']} mismatched transfers"
            + (f" ({done['bad_bytes']} bytes wrong)" if done["bad"] else ""))

    time.sleep(1)
    r1 = replays(gpu)
    a1 = None if IS_WIN else linux_aer(bdf)
    p1 = linux_aer(parent) if parent else None
    loaded = [(gg, ww) for ph, gg, ww in samples if ph == "soak"] or [(gg, ww) for _, gg, ww in samples]
    cur = max(loaded) if loaded else (None, None)
    soak_widths = [ww for ph, _, ww in samples if ph == "soak"]
    out(f"Link under load: Gen{cur[0]} x{cur[1]}  (max Gen{maxgen} x{maxw})")
    after = f"After:  replays={r1}"
    if not IS_WIN:
        after += f"  AER(card)={a1 if a1 is not None else 'n/a'}" \
                 f"  AER(port)={p1 if p1 is not None else 'n/a'}"
    out(after)

    events, klog = [], []
    if IS_WIN:
        events = win_events(start)
        if events:
            out(f"Windows event log since start ({len(events)} WHEA / NVIDIA driver events):")
            for ev in events[:15]:
                out(f"  [{ev.get('ProviderName')} {ev.get('Id')}] {ev.get('Msg')}")
        else:
            out("Windows event log since start: no WHEA or NVIDIA driver errors")
    else:
        klog = linux_kernel_log(start)
        out("Kernel log since start (Xid / AER / PCIe errors): "
            + ("none" if not klog else ""))
        for l in klog[-20:]:
            out("  " + l)

    # ---- verdict ----
    fails, warns = [], []
    if err and not done:
        fails.append(f"test couldn't run: {err}")
    elif not done:
        tail = (" | last output: " + stderr_tail[-1]) if stderr_tail else ""
        fails.append("test crashed - GPU may have dropped off the bus "
                     f"(Xid 79 / WHEA events above){tail}")
    if done and done["bad"]:
        fails.append("data mismatches seen - do not use this adapter/riser")
    if r0 is not None and r1 is not None and r1 > r0:
        warns.append(f"PCIe replays went up by {r1 - r0} - link is marginal; "
                     "force one gen lower in BIOS and retest")
    for who, x, y in (("card", a0, a1), ("upstream port", p0, p1)):
        if x is not None and y is not None and y > x:
            warns.append(f"AER correctable errors on {who} went up by {y - x}")
    whea = [e for e in events if "WHEA" in str(e.get("ProviderName"))]
    nv = [e for e in events if "nvlddmkm" in str(e.get("ProviderName"))]
    if whea:
        warns.append(f"{len(whea)} WHEA hardware error event(s) logged (PCIe bus errors)")
    if nv:
        warns.append(f"{len(nv)} NVIDIA driver error event(s) logged")
    if klog:
        warns.append(f"{len(klog)} Xid/AER/PCIe line(s) in the kernel log")
    if cur[0] and maxgen and maxw and (cur[0] < maxgen or cur[1] < maxw):
        warns.append(f"link trained below max (Gen{cur[0]} x{cur[1]} vs Gen{maxgen} x{maxw}) - "
                     "expected if the adapter is narrower (M.2 = x4), otherwise a bad connection")
    if soak_widths and min(soak_widths) < max(soak_widths):
        warns.append(f"link width changed during the test (x{min(soak_widths)}..x{max(soak_widths)})"
                     " - lanes dropping out under load is a bad-connection sign")
    expected = None
    if bw and cur[0] in PER_LANE and cur[1]:
        expected = PER_LANE[cur[0]] * cur[1]
        low = min(bw["h2d"], bw["d2h"])
        if low < BW_FLOOR * expected:
            warns.append(f"bandwidth {low:.1f} GB/s is only {100 * low / expected:.0f}% of what "
                         f"Gen{cur[0]} x{cur[1]} can carry (~{expected:.1f}) - expect ~75-85%; "
                         "check for a shared chipset link, CPU/NUMA placement, or a bad adapter")

    out("")
    out("=== VERDICT ===")
    for f_ in fails:
        out("FAIL  " + f_)
    for w in warns:
        out("WARN  " + w)
    verdict = "FAIL" if fails else "WARN" if warns else "PASS"
    if verdict == "PASS":
        out("PASS  full speed, zero errors, zero corruption")
    elif verdict == "WARN":
        out("Data was intact, but see warnings above")

    res.update(verdict=verdict, fails=fails, warns=warns,
               link_gen=cur[0], link_width=cur[1],
               h2d_gbs=bw and bw["h2d"], d2h_gbs=bw and bw["d2h"],
               expected_gbs=expected and round(expected, 1),
               rounds=done and done["rounds"], mismatches=done and done["bad"],
               mismatched_bytes=done and done["bad_bytes"],
               bytes_moved=done and done["bytes"],
               replays_before=r0, replays_after=r1,
               aer_card=[a0, a1], aer_port=[p0, p1], port=parent,
               win_events=events, kernel_log=klog[-50:],
               worker_rc=rc, started=start.isoformat(timespec="seconds"))
    return res


# --------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="pcie-check",
        description="Test a GPU's PCIe link (slot / riser / adapter) for speed and integrity.")
    ap.add_argument("gpu", nargs="?", default="0", help="GPU index (nvidia-smi order) or 'all'")
    ap.add_argument("minutes", nargs="?", type=float, default=10.0, help="soak length (default 10)")
    ap.add_argument("--list", action="store_true", help="list GPUs and their links, then exit")
    ap.add_argument("--json", metavar="FILE", help="also write results as JSON")
    ap.add_argument("--log", metavar="FILE", default=os.path.join(HERE, "pcie-check-result.txt"),
                    help="text log (default: pcie-check-result.txt next to the script; '' = none)")
    ap.add_argument("--runner", choices=("auto", "host", "docker"), default="auto",
                    help="where the transfer test runs (default: host PyTorch, else Docker)")
    ap.add_argument("--image", default=os.environ.get("PCIE_CHECK_IMAGE", DEFAULT_IMAGE),
                    help=f"Docker image for --runner docker (default {DEFAULT_IMAGE})")
    ap.add_argument("--size-mb", type=int, default=512, help="transfer size in MiB (default 512)")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--version", action="version", version=f"pcie-check {VERSION}")
    args = ap.parse_args(argv)

    if args.worker:
        return worker(float(args.gpu), args.size_mb)   # gpu slot carries minutes here

    if not shutil.which("nvidia-smi"):
        print("nvidia-smi not found - install the NVIDIA driver first.")
        return 1
    gpus = list_gpus()
    if not gpus:
        print("nvidia-smi found no GPUs.")
        return 1

    if args.list:
        for g in gpus:
            print(f"GPU {g['index']}: {g['name']:<32} {g['bus']}  "
                  f"now Gen{g['gen']} x{g['width']}  max Gen{g['maxgen']} x{g['maxwidth']}")
        print("(idle GPUs drop to Gen1; the test reads the link under load)")
        return 0

    if args.gpu.lower() == "all":
        chosen = gpus
    else:
        idx = as_int(args.gpu)
        chosen = [g for g in gpus if g["index"] == idx]
        if not chosen:
            print(f"No GPU {args.gpu}. Available: " + ", ".join(str(g["index"]) for g in gpus))
            return 1

    runner = pick_runner(args.runner)
    if not runner:
        print(f"Can't run the transfer test: no PyTorch for {sys.executable}"
              + ("" if IS_WIN else " and no Docker") + ".")
        print("Install PyTorch with CUDA, e.g.:")
        print("  py -m pip install torch --index-url https://download.pytorch.org/whl/cu128"
              if IS_WIN else
              "  python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu128"
              "\n  (or install Docker + the NVIDIA container toolkit)")
        return 1

    out = Log(args.log or None)
    out(f"pcie-check v{VERSION}  ({'Windows' if IS_WIN else 'Linux'}, "
        f"{datetime.datetime.now():%Y-%m-%d %H:%M})")
    if not IS_WIN and hasattr(os, "geteuid") and os.geteuid() != 0:
        out("(not root: kernel log / AER counters may be unreadable - sudo gives the full picture)")
    results = []
    try:
        for i, g in enumerate(chosen):
            if i:
                out("\n")
            results.append(test_gpu(g, args, runner, out))
        if len(results) > 1:
            out("\n=== SUMMARY ===")
            for r in results:
                link = f"Gen{r['link_gen']} x{r['link_width']}" if r["link_gen"] else "?"
                bwtxt = f"{r['h2d_gbs']:.1f}/{r['d2h_gbs']:.1f} GB/s" if r["h2d_gbs"] else "-"
                out(f"GPU {r['gpu']}  {r['verdict']:<4}  {link:<9} {bwtxt:<16} {r['name']}")
        if args.json:
            with open(args.json, "w", encoding="utf-8") as f:
                json.dump(results if len(results) > 1 else results[0], f, indent=2)
            out(f"\nJSON written to {args.json}")
        if args.log:
            out(f"Saved to {args.log}")
    finally:
        out.close()
    return 1 if any(r["verdict"] == "FAIL" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
