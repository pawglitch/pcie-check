"""Smoke tests for the pcie-check controller, no GPU needed.

nvidia-smi is replaced by tests/fake_nvidia_smi.py and the torch worker by a stub that
prints the same JSON lines. Run:  python3 -m unittest discover -s tests -v
"""
import importlib.util
import io
import json
import os
import stat
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("pcie_check", os.path.join(ROOT, "pcie-check.py"))
pc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pc)

STUB = r'''
import json, sys, os, time
mode = os.environ.get("STUB_MODE", "pass")
def emit(**k): print(json.dumps(k), flush=True)
h2d = 9.0 if mode == "slowbw" else 24.4
emit(type="bw", h2d=h2d, d2h=24.9)
time.sleep(float(os.environ.get("STUB_SLEEP", "0.2")))
if mode == "crash":
    print("RuntimeError: CUDA error: unspecified launch failure", file=sys.stderr); sys.exit(1)
bad = 3 if mode == "corrupt" else 0
emit(type="done", rounds=100, bad=bad, bad_bytes=bad * 7, bytes=100 * 2 * 512 * 1024**2)
sys.exit(2 if bad else 0)
'''


@unittest.skipIf(os.name == "nt", "fake nvidia-smi shim is POSIX-only")
class Smoke(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        shim = os.path.join(self.tmp, "nvidia-smi")
        with open(shim, "w") as f:
            f.write(f'#!/bin/sh\nexec "{sys.executable}" "{ROOT}/tests/fake_nvidia_smi.py" "$@"\n')
        os.chmod(shim, os.stat(shim).st_mode | stat.S_IEXEC)
        self.stub = os.path.join(self.tmp, "stub.py")
        open(self.stub, "w").write(STUB)
        self.env_keep = dict(os.environ)
        os.environ["PATH"] = self.tmp + os.pathsep + os.environ["PATH"]
        os.environ["FAKE_STATE"] = self.tmp
        self.orig_wc, self.orig_pr = pc.worker_command, pc.pick_runner
        pc.worker_command = lambda *a, **k: ([sys.executable, self.stub], None, None)
        pc.pick_runner = lambda want: "host"

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.env_keep)
        pc.worker_command, pc.pick_runner = self.orig_wc, self.orig_pr

    def go(self, *argv, **env):
        os.environ.update({k: str(v) for k, v in env.items()})
        js = os.path.join(self.tmp, "r.json")
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = pc.main(list(argv) + ["--json", js, "--log", ""])
        with open(js) as f:
            return rc, json.load(f), buf.getvalue()

    def test_pass(self):
        rc, r, text = self.go("0", "0.01")
        self.assertEqual((rc, r["verdict"]), (0, "PASS"), text)
        self.assertEqual((r["link_gen"], r["link_width"]), (4, 16))
        self.assertIn("card Gen5, slot/host Gen4", text)

    def test_replays_warn(self):
        rc, r, text = self.go("0", "0.01", FAKE_REPLAY_STEP=5)
        self.assertEqual((rc, r["verdict"]), (0, "WARN"), text)
        self.assertTrue(any("replays went up" in w for w in r["warns"]))

    def test_corruption_fails(self):
        rc, r, _ = self.go("0", "0.01", STUB_MODE="corrupt")
        self.assertEqual((rc, r["verdict"], r["mismatches"]), (1, "FAIL", 3))

    def test_crash_fails(self):
        rc, r, _ = self.go("0", "0.01", STUB_MODE="crash")
        self.assertEqual((rc, r["verdict"]), (1, "FAIL"))
        self.assertIn("unspecified launch failure", r["fails"][0])

    def test_low_bandwidth_warns(self):
        _, r, _ = self.go("0", "0.01", STUB_MODE="slowbw")
        self.assertTrue(any("bandwidth" in w for w in r["warns"]), r["warns"])

    def test_narrow_link_warns(self):
        _, r, _ = self.go("0", "0.01", FAKE_WIDTH=4, STUB_SLEEP=4)
        self.assertTrue(any("trained below max" in w for w in r["warns"]), r["warns"])

    def test_width_drop_warns(self):
        _, r, _ = self.go("0", "0.01", FAKE_WIDTH_DROP=1, STUB_SLEEP=7)
        self.assertTrue(any("width changed" in w for w in r["warns"]), r["warns"])

    def test_all_gpus(self):
        rc, r, text = self.go("all", "0.01")
        self.assertEqual([x["gpu"] for x in r], [0, 1])
        self.assertIn("=== SUMMARY ===", text)

    def test_list_and_bad_index(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(pc.main(["--list"]), 0)
            self.assertEqual(pc.main(["7", "--log", ""]), 1)
        self.assertIn("RTX 3080", buf.getvalue())
        self.assertIn("No GPU 7", buf.getvalue())


class Pure(unittest.TestCase):
    def test_bdf(self):
        self.assertEqual(pc.bdf_of("00000000:0A:00.0"), "0000:0a:00.0")

    def test_docker_command_ships_script(self):
        argv, env, src = pc.worker_command("docker", "GPU-x", 1.0, 64, "img")
        self.assertEqual(argv[:6], ["docker", "run", "--rm", "--gpus", "device=GPU-x", "-i"])
        self.assertIn("def worker(", src)
        self.assertIsNone(env)


if __name__ == "__main__":
    unittest.main()
