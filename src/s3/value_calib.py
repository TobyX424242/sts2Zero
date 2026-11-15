"""Offline value calibration of a macro ensemble on stored macro chunks (*.s3r, decompressed), per member, per ascension
group (A0-8, A9, A10) and act: mean predicted value vs mean outcome on floor rows (the trainer's V1 diagnostic, offline).

usage: python -I src/s3/value_calib.py MACRO.pt ALPHA CHUNK.s3r [CHUNK.s3r ...] [--cpu] [--split K]
  ALPHA   the outcome's alpha (macro_train_log.jsonl "alpha"; or a path to that file: its last alpha)
  --split K  also report chunks with sequence number < K and >= K separately (a<actor>_<seq>.s3r: early vs late)
"""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "py"))
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import torch  # noqa: E402
import macro_s3 as ms  # noqa: E402
import train_macro_s3 as tr  # noqa: E402


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    split = None
    if "--split" in sys.argv:
        split = int(sys.argv[sys.argv.index("--split") + 1]); args.remove(str(split))
    dev = "cpu" if "--cpu" in sys.argv or not torch.cuda.is_available() else "cuda"
    macro, alpha, chunks = args[0], args[1], args[2:]
    if os.path.exists(alpha):
        alpha = [json.loads(l)["alpha"] for l in open(alpha) if '"alpha"' in l][-1]
    alpha = float(alpha)
    ens = ms.Ensemble.load(macro, dev)
    for n in ens.nets:
        n.eval()
    groups = {"all": chunks}
    if split is not None:
        seq = lambda p: int(re.search(r"_(\d+)\.s3r$", p).group(1))
        groups = {f"seq<{split}": [c for c in chunks if seq(c) < split], f"seq>={split}": [c for c in chunks if seq(c) >= split], "all": chunks}
    print(f"ensemble {macro} (version {ens.version}, {len(ens.nets)} members), alpha {alpha}, device {dev}")
    for gname, cs in groups.items():
        runs = []
        for c in cs:
            R, ints, floats, meta = tr.read_chunk(c)
            for r in R:
                a, n = int(r[9]), int(r[10])
                runs.append({"ints": ints[a:a + n], "floats": floats[a:a + n], "meta": meta[a:a + n], "won": float(r[3]),
                             "perturbed": bool(r[8]), "asc": int(r[2]), "act": int(r[5]),
                             "progress": 1.0 if r[3] else min(int(r[5]), tr.N_ACTS - 1) / tr.N_ACTS})
        if not runs:
            continue
        ni, nf, nm = tr.cat(runs, "ints"), tr.cat(runs, "floats"), tr.cat(runs, "meta")
        z = np.concatenate([np.full(len(r["meta"]), tr.outcome(r, alpha), np.float32) for r in runs])
        L = []
        with torch.no_grad():
            for i in range(0, len(ni), 8192):
                b = ms.rows_to_batch(ni[i:i + 8192], nf[i:i + 8192], dev)
                L.append(torch.stack([n.forward_rows(b, policy=False)[2] for n in ens.nets]).cpu())
        L = torch.cat(L, 1)
        v = torch.sigmoid(L.mean(0)).numpy(); vm = torch.sigmoid(L).numpy()
        cal = tr.calibration(runs, nm, z, v, vm)
        print(f"--- {gname}: {len(cs)} chunks, {len(runs)} runs, {len(z)} rows")
        print(f"  {'group':10s} {'rows':>6s} {'outcome':>8s} {'ens':>7s} {'ens_p':>7s}  " + " ".join(f"{'m' + str(j):>7s}" for j in range(len(ens.nets)))
              + f" {'logit_sd':>8s}   (bias = value - outcome; ens = sigmoid(mean logit) as in T_f, ens_p = mean probability)")
        for k, x in cal.items():
            print(f"  {k:10s} {x['rows']:6d} {x['outcome']:8.4f} {x['ens'] - x['outcome']:+7.4f} {x['ens_p'] - x['outcome']:+7.4f}  "
                  + " ".join(f"{m - x['outcome']:+7.4f}" for m in x["members"]) + f" {x['logit_sd']:8.3f}")
        # which member drives the spread: mean logit of each member minus the mean of the others, per group
        asc = np.concatenate([np.full(len(r["meta"]), r["asc"]) for r in runs]); fl = nm[:, tr.M_KIND] == 0
        Ln = L.numpy()
        for name, f in tr.ASC_GROUPS:
            m = fl & np.vectorize(f, otypes=[bool])(asc)
            if m.any():
                dev_ = [float((Ln[j, m] - np.delete(Ln[:, m], j, 0).mean(0)).mean()) for j in range(len(Ln))]
                absd = [float(np.abs(Ln[j, m] - np.delete(Ln[:, m], j, 0).mean(0)).mean()) for j in range(len(Ln))]
                print(f"  {name:10s} member logit minus the others' mean: " + " ".join(f"m{j} {d:+.3f} (|.| {a:.3f})" for j, (d, a) in enumerate(zip(dev_, absd))))


if __name__ == "__main__":
    main()
