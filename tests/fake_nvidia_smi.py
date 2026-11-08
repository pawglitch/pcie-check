"""Stand-in for nvidia-smi so the controller can be tested without a GPU.

Behaviour is driven by env vars:
  FAKE_STATE     dir for counters (replay counter increments per -q call if FAKE_REPLAY_STEP)
  FAKE_GEN       current gen under load (default 4)
  FAKE_WIDTH     current width under load (default 16); FAKE_WIDTH_DROP=1 alternates x16/x8
"""
import os
import sys

GPUS = [
    dict(index=0, name="NVIDIA GeForce RTX 5080", uuid="GPU-aaaa", bus="00000000:01:00.0"),
    dict(index=1, name="NVIDIA GeForce RTX 3080", uuid="GPU-bbbb", bus="00000000:05:00.0"),
]
state = os.environ.get("FAKE_STATE", "/tmp")
gen = os.environ.get("FAKE_GEN", "4")


def bump(name):
    p = os.path.join(state, name)
    n = int(open(p).read()) if os.path.exists(p) else 0
    open(p, "w").write(str(n + 1))
    return n


def width():
    if os.environ.get("FAKE_WIDTH_DROP") and bump("width_calls") % 2:
        return "8"
    return os.environ.get("FAKE_WIDTH", "16")


args = sys.argv[1:]
idx = None
if "-i" in args:
    idx = int(args[args.index("-i") + 1])
sel = [g for g in GPUS if idx is None or g["index"] == idx]

if "-q" in args:
    step = int(os.environ.get("FAKE_REPLAY_STEP", "0"))
    n = bump("replay_calls")
    print("    PCIe\n        Replays Since Reset          : %d" % (n * step))
    sys.exit(0)

q = next(a for a in args if a.startswith("--query-gpu=")).split("=", 1)[1].split(",")
vals = {
    "index": lambda g: str(g["index"]), "name": lambda g: g["name"],
    "uuid": lambda g: g["uuid"], "pci.bus_id": lambda g: g["bus"],
    "pcie.link.gen.current": lambda g: gen, "pcie.link.width.current": lambda g: width(),
    "pcie.link.gen.max": lambda g: "4", "pcie.link.width.max": lambda g: "16",
    "pcie.link.gen.gpumax": lambda g: "5", "pcie.link.gen.hostmax": lambda g: "4",
}
for f in q:
    if f not in vals:
        print(f'Field "{f}" is not a valid field to query.', file=sys.stderr)
        sys.exit(2)
for g in sel:
    print(", ".join(vals[f](g) for f in q))
