# Changelog

## 0.2.0 — 2026-11-07

One script for both platforms: `pcie-check.py` replaces `linux/pcie-check.sh` and `windows/pcie-check.py`.

- Linux now uses host PyTorch when it's installed and falls back to Docker only when it isn't (`--runner` to force either).
- The transfer test runs in a child process pinned to the GPU by UUID. A GPU dropping off the bus kills the child, not the run, so counters and logs are still collected.
- `all` tests every GPU in turn and prints a summary table. `--list` shows GPUs and their links.
- `--json FILE` writes machine-readable results.
- New WARN: bandwidth under 60% of what the negotiated link carries.
- New WARN: link width changed during the soak (lanes dropping out under load).
- Mismatches now report how many bytes were wrong.
- Shows when the card and slot differ (e.g. "card Gen5, slot/host Gen4").
- Link is sampled throughout the run on Linux too, not once at 20 s.
- Smoke tests with a fake `nvidia-smi`, plus GitHub Actions CI.

## 0.1.0 — 2026-11-07

First release.

- Linux script (`linux/pcie-check.sh`): runs the test in a `pytorch/pytorch` container; reads NVIDIA replay counter and kernel AER counters on the card and its upstream port; scans the kernel log for Xid / AER lines.
- Windows script (`windows/pcie-check.py` + `RUN-pcie-check.cmd`): same test with host PyTorch; reads the replay counter and WHEA-Logger / nvlddmkm events from the System log.
- Both: link gen/width sampled under load, pinned-memory bandwidth, bit-for-bit GPU→host→GPU soak for N minutes, PASS / WARN / FAIL verdict.
