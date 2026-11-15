"""Offline data extraction for the A10 run-value analysis (2026-11-14). No network is loaded or run.

Reads stored macro chunks (*.s3r or *.s3r.zst, the files the macro trainer consumed) and the run logs of the same
segment and writes one compact .npz: per run (ascension, outcome, bosses won, killer ...) and per row (the value the
published ensemble gave that row at play time, the scalar state features, room/encounter ids), plus deck / relic /
potion ids of the floor rows. The recorded value is out of sample: it was computed before the trainer saw the run.

Usage: python -I research/s3/a10val_extract.py <chunk dir> <segment meta dir (with runs/*.jsonl)> <out.npz>
"""
import glob, json, os, subprocess, sys
import numpy as np

IW, FW, RM, RW = 460, 208, 16, 16
OD_IDS, OD_UPG, OR_IDS, OP_IDS, OCTX = 0, 64, 320, 400, 405


def fnv(s):
    h = 14695981039346656037
    for ch in s:
        h ^= ord(ch); h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def read_chunk(path):
    b = subprocess.run(["zstd", "-dc"], stdin=open(path, "rb"), capture_output=True, check=True).stdout if path.endswith(".zst") else open(path, "rb").read()
    assert b[:8] == b"S3RUNS01", path
    ver, R, N, iw, fw, rm, rw = np.frombuffer(b, "<i4", 7, 8)
    assert (ver, iw, fw, rm, rw) == (1, IW, FW, RM, RW), path
    off = 64
    runs = np.frombuffer(b, "<i4", R * RM, off).reshape(R, RM); off += runs.nbytes
    ints = np.frombuffer(b, "<i2", N * IW, off).reshape(N, IW); off += ints.nbytes
    floats = np.frombuffer(b, "<f4", N * FW, off).reshape(N, FW); off += floats.nbytes
    meta = np.frombuffer(b, "<f4", N * RW, off).reshape(N, RW)
    return runs, ints, floats, meta


def main():
    cdir, mdir, out = sys.argv[1:4]
    files = sorted(glob.glob(os.path.join(cdir, "*.s3r")) + glob.glob(os.path.join(cdir, "*.s3r.zst")))
    logs = {}
    for p in glob.glob(os.path.join(mdir, "runs", "*.jsonl")):
        for l in open(p):
            try:
                d = json.loads(l)
            except Exception:
                continue
            logs[fnv("S3:" + d["seed"])] = d
    R_, ROW, IDS = [], [], []
    g = 0
    for i, p in enumerate(files):
        try:
            runs, ints, floats, meta = read_chunk(p)
        except Exception as e:
            print("skip", p, e); continue
        first = sum(len(x) for x in ROW)
        for r in runs:
            h = (int(r[0]) & 0xFFFFFFFF) | ((int(r[1]) & 0xFFFFFFFF) << 32)
            d = logs.get(h, {})
            kb = d.get("killed_by") or ""
            R_.append([r[2], r[3], r[4], r[5], r[6], r[7], r[8], first + r[9], r[10], r[11], r[12], r[13], r[15],
                       d.get("bosses", -1), d.get("bosses_won", -1), d.get("rests", -1), d.get("smiths", -1), d.get("shops", -1),
                       d.get("potions_used", -1), d.get("deck", -1), d.get("relics", -1), 1 if d else 0])
            R_[-1].append(kb)
        s = floats[:, :16]
        row = np.stack([meta[:, 0] + g, meta[:, 1], meta[:, 2], meta[:, 3], meta[:, 6], meta[:, 7], meta[:, 15], meta[:, 14],
                        s[:, 2] * 100, s[:, 3] * 100, s[:, 4], s[:, 5] * 500, s[:, 8] * 40, s[:, 9] * 20, s[:, 10] * 10, s[:, 11] * 20,
                        s[:, 12] * 5, s[:, 14], s[:, 15] * 20, ints[:, OCTX + 4], ints[:, OCTX + 5], meta[:, 4], meta[:, 5], meta[:, 8]], 1).astype(np.float32)
        ROW.append(row)
        fl = meta[:, 1] == 0
        IDS.append(np.concatenate([ints[fl, OD_IDS:OD_IDS + 64], ints[fl, OD_UPG:OD_UPG + 64], ints[fl, OR_IDS:OR_IDS + 40],
                                   ints[fl, OP_IDS:OP_IDS + 5]], 1).astype(np.int16))
        g += len(runs)
        if i % 100 == 0:
            print(i, len(files), g, flush=True)
    killers = sorted({r[-1] for r in R_})
    kidx = {k: i for i, k in enumerate(killers)}
    runs = np.array([r[:-1] + [kidx[r[-1]]] for r in R_], dtype=np.int64)
    np.savez(out, runs=runs, rows=np.concatenate(ROW), ids=np.concatenate(IDS), killers=np.array(killers),
             run_cols=np.array("asc won floors act final_hp max_hp perturbed first_row n_rows fights fights_won elites mver bosses bosses_won "
                               "rests smiths shops potions_used deck relics has_log killer".split()),
             row_cols=np.array("run kind floor dtype value vstd vpol act hp max_hp hp_frac gold deck n_upg n_ench n_relics n_potions is_combat "
                               "floor_in_act room enc chosen logp mver".split()))
    print("runs", len(runs), "rows", sum(len(x) for x in ROW), "with log", int(runs[:, 21].sum()), "->", out)


if __name__ == "__main__":
    main()
