# pcie-check

**Is this riser / M.2 adapter / slot actually safe to run a GPU on?**

One command that tells you whether a GPU's PCIe link is running at full speed *and* not quietly erroring — for people building multi-GPU rigs on risers, bifurcation cards and M.2-to-PCIe adapters (rental hosts on Vast/Salad, local-LLM boxes, ex-mining frames).

It checks four things in one run:

1. **Link training** — what gen and width the card actually negotiated *under load* vs. the max the card + slot support (GPUs drop to Gen1 at idle, so idle readings lie).
2. **Bandwidth** — real host↔GPU throughput with pinned-memory copies.
3. **Integrity** — random data goes GPU → host → GPU and is compared bit-for-bit, on a loop, for as long as you tell it.
4. **Error counters before/after** — NVIDIA's PCIe replay counter, plus kernel AER counters (Linux) or WHEA-Logger / `nvlddmkm` events (Windows).

### Why the counters matter more than the checksum

PCIe has CRC and automatic retransmit on every packet, so a bad cable almost never corrupts data *silently*. What a marginal link does instead is retry — the replay counter and AER/WHEA error counts climb — then it trains down to a lower speed, and in the worst case the GPU drops off the bus (Xid 79 on Linux). The bit-for-bit soak is the belt; the counters are the suspenders that actually catch the problem first.

---

## Install

One file: `pcie-check.py`. It needs `nvidia-smi` (comes with the driver) and Python 3. The transfer test needs PyTorch with CUDA. It uses the PyTorch of the Python you run it with; on Linux it falls back to a `pytorch/pytorch` Docker container (any Vast.ai host already has Docker and the NVIDIA toolkit). Nothing else needs installing.

```
pip install torch --index-url https://download.pytorch.org/whl/cu128   # cu128+ for RTX 50-series
```

## Run

```bash
sudo python3 pcie-check.py --list          # GPUs and their links
sudo python3 pcie-check.py 1 15            # GPU 1, 15-minute soak
sudo python3 pcie-check.py all 5           # every GPU, one after another, with a summary
sudo python3 pcie-check.py 0 10 --json r.json
```

On Windows use `py` instead of `sudo python3`, or double-click `RUN-pcie-check.cmd`. It lists the GPUs, asks which one (or `all`) and how many minutes.

`sudo` on Linux is only needed for the kernel log and AER counters. Results always go to `pcie-check-result.txt` next to the script (`--log ''` turns that off), and `--json` writes a machine-readable copy.

| Option | |
|---|---|
| `--runner host\|docker` | force where the transfer test runs (default: host PyTorch, else Docker) |
| `--image IMG` | Docker image (default `pytorch/pytorch:latest`, or `$PCIE_CHECK_IMAGE`) |
| `--size-mb N` | transfer size, default 512 MiB |

The transfer test runs in a child process pinned to the GPU under test. If the GPU falls off the bus, the child dies but the script still reads the counters and logs and reports FAIL.

---

## Reading the result

| Verdict | Meaning | What to do |
|---|---|---|
| **PASS** | Full speed, zero replays, zero logged errors, zero mismatches | Use it |
| **WARN** — replays / AER / WHEA went up | Link is marginal: data got through, but only because of retries | Force that slot one gen lower in BIOS (e.g. Gen5 → Gen4) and retest. If it still warns, replace the riser/adapter |
| **WARN** — trained below max | Link came up narrower or slower than the card supports | Expected for M.2 adapters (always x4) and narrower slots. Otherwise: reseat, or the riser is bad |
| **WARN** — width changed during test | Lanes dropped out under load | Bad connection. Reseat or replace the riser |
| **WARN** — bandwidth low for the link | Under 60% of what the negotiated link carries (75–85% is normal) | Shared chipset uplink, a CPU/NUMA placement issue, or a bad adapter |
| **FAIL** — mismatches | Corrupted data made it through | Don't use it |
| **FAIL** — crashed | GPU dropped off the bus mid-test | Don't use it; check `dmesg` / Event Viewer |

Run it **once with the card straight in a slot and once on the adapter** — the difference is the adapter.

**Run it on an idle box.** It saturates the GPU's link — unlist rental machines first.

### Expected bandwidth (pinned copies, each direction)

| Link | Theoretical | Typical measured |
|---|---|---|
| Gen3 x8 / Gen4 x4 / Gen5 x2 | ~8 GB/s | ~6.5 GB/s |
| Gen3 x16 / Gen4 x8 / Gen5 x4 | ~16 GB/s | ~12–13 GB/s |
| Gen4 x16 / Gen5 x8 | ~32 GB/s | ~24–26 GB/s |
| Gen5 x16 | ~63 GB/s | ~50 GB/s |

The card limits this too: RTX 20-series is Gen3, 30/40-series is Gen4, 50-series is Gen5. A Gen4 card in a Gen5 slot runs at Gen4.

## Example

RTX 5080 straight in a Gen4 x16 slot, 2-minute run ([full output](examples/rtx5080-gen4x16-slot-windows-v0.2.txt)):

```
Max for this card + slot: Gen4 x16   (card Gen5, slot/host Gen4)
Bandwidth  host->GPU:  28.4 GB/s   GPU->host:  27.2 GB/s
Integrity: 2774 round trips, 2.98 TB moved, 0 mismatched transfers
Link under load: Gen4 x16  (max Gen4 x16)
After:  replays=0
Windows event log since start: no WHEA or NVIDIA driver errors

=== VERDICT ===
PASS  full speed, zero errors, zero corruption
```

## Limits

- NVIDIA GPUs only (uses `nvidia-smi` and CUDA).
- Linux AER counters only appear if the kernel and firmware expose AER; if they don't, the script says `n/a` and relies on the replay counter.
- It tests the link, not the GPU itself — use gpu-burn / OCCT / MATS for VRAM and core stability.

## Tests

`python3 -m unittest discover -s tests -v` runs the controller against a fake `nvidia-smi` and a stub worker (pass, replays, corruption, crash, slow bandwidth, narrow link, lanes dropping, all-GPU mode). No GPU needed; CI runs it on every push.

## License

MIT
