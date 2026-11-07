# Changelog

## 0.1.0 — 2026-11-07

First release.

- Linux script (`linux/pcie-check.sh`): runs the test in a `pytorch/pytorch` container; reads NVIDIA replay counter and kernel AER counters on the card and its upstream port; scans the kernel log for Xid / AER lines.
- Windows script (`windows/pcie-check.py` + `RUN-pcie-check.cmd`): same test with host PyTorch; reads the replay counter and WHEA-Logger / nvlddmkm events from the System log.
- Both: link gen/width sampled under load, pinned-memory bandwidth, bit-for-bit GPU→host→GPU soak for N minutes, PASS / WARN / FAIL verdict.
