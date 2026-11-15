"""Macro inference on CPU vs GPU (2026-11-14): time macro_s3.Ensemble.serve on real rows (a run chunk *.s3r) in batches
like the live macro server's (~30 rows), with T CPU threads, and compare the outputs with the same batches on CUDA.
usage: ./.venv/bin/python -I src/s3/py/macro_cpu_bench.py MACRO.pt CHUNK.s3r [--threads 1,2,4] [--batch 32] [--iters 300]
"""
import argparse, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import torch  # noqa: E402
import macro_s3 as ms  # noqa: E402
from train_macro_s3 import read_chunk  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("macro"); ap.add_argument("chunk")
    ap.add_argument("--threads", default="1,2,4"); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--iters", type=int, default=300)
    a = ap.parse_args()
    _, ints, floats, _ = read_chunk(a.chunk)
    rng = np.random.default_rng(0)
    batches = [rng.choice(len(ints), a.batch, replace=False) for _ in range(a.iters)]
    dec = (ints[:, ms.OFF["opt_kind"][0]] != 0)
    print(f"{len(ints)} rows, {dec.mean():.0%} decision rows; batches of {a.batch}, {a.iters} iterations", flush=True)
    ref = None
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True     # as s3_server.py
        g = ms.Ensemble.load(a.macro, "cuda")
        with torch.inference_mode():
            for b in batches[:20]:
                g.serve(ints[b], floats[b], "cuda")
            torch.cuda.synchronize(); t = time.perf_counter()
            ref = [g.serve(ints[b], floats[b], "cuda").cpu().numpy() for b in batches]
            dt = time.perf_counter() - t
        print(f"cuda      : {1e3 * dt / a.iters:6.2f} ms/batch = {a.iters * a.batch / dt:7.0f} rows/s (GPU shared with the live run)", flush=True)
        del g
    for th in [int(x) for x in a.threads.split(",")]:
        torch.set_num_threads(th)
        c = ms.Ensemble.load(a.macro, "cpu")
        with torch.inference_mode():
            for b in batches[:20]:
                c.serve(ints[b], floats[b], "cpu")
            t = time.perf_counter()
            out = [c.serve(ints[b], floats[b], "cpu").numpy() for b in batches]
            dt = time.perf_counter() - t
        diff = max(float(np.abs(o - r).max()) for o, r in zip(out, ref)) if ref is not None else float("nan")
        print(f"cpu {th} thr: {1e3 * dt / a.iters:6.2f} ms/batch = {a.iters * a.batch / dt:7.0f} rows/s; max |cpu - cuda| {diff:.2e}", flush=True)


if __name__ == "__main__":
    main()
