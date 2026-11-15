"""S2 battle-start pool: every combat floor of every Defect v0.111.0 single-player run at the given ascension in the
full Spire Codex dump (src/s0/export_battle_starts.py battle_starts(), same JSON format), split by run:
runs whose hash falls in the held-out 5% (sha1(run_hash) mod 20 == 0) go to the evaluation file (gate G2), the rest
to the training file.

Usage: ./.venv/bin/python -I src/s2/export_starts.py [--runs PATH] [--ascension 10] [--out-dir src/s2/data]
"""
import argparse, gzip, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(ROOT, "src", "s0"))
import orjson  # noqa: E402
from export_battle_starts import battle_starts  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(ROOT, "data/spire_codex/runs-latest.jsonl.gz"))
    ap.add_argument("--build", default="v0.111.0")
    ap.add_argument("--character", default="DEFECT")
    ap.add_argument("--ascension", type=int, default=10)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "data"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    want = "CHARACTER." + a.character
    tag = f"{a.build}_{a.character.lower()}_a{a.ascension}"
    n_runs = {"train": 0, "eval": 0}; n_out = {"train": 0, "eval": 0}; seen = set()
    outs = {k: open(os.path.join(a.out_dir, f"starts_{tag}_{k}.jsonl"), "w") for k in n_out}
    with gzip.open(a.runs, "rb") as f:
        for line in f:
            if a.build.encode() not in line or want.encode() not in line:
                continue
            r = orjson.loads(line)
            if (r.get("build_id") != a.build or len(r.get("players") or []) != 1 or r["players"][0].get("character") != want
                    or int(r.get("ascension") or 0) != a.ascension):
                continue
            h = r.get("run_hash") or r.get("seed")
            if h in seen:
                continue
            seen.add(h)
            starts = battle_starts(r)
            if not starts:
                continue
            split = "eval" if int(hashlib.sha1(str(h).encode()).hexdigest(), 16) % 20 == 0 else "train"
            n_runs[split] += 1
            for s in starts:
                outs[split].write(json.dumps(s) + "\n")
                n_out[split] += 1
    for w in outs.values():
        w.close()
    print(json.dumps({"runs": n_runs, "battle_starts": n_out, "tag": tag}))


if __name__ == "__main__":
    main()
