"""cmp.py A B: compare two test outputs (runs.jsonl, fights.jsonl, macro/combat data files) ignoring timing fields."""
import sys, json, os, hashlib, glob

TIME = {"seconds", "practice_seconds", "ms", "sec", "time", "wall", "elapsed", "t"}


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items() if k not in TIME and not k.endswith("_seconds") and not k.endswith("_ms")}
    if isinstance(o, list):
        return [clean(v) for v in o]
    return o


def load(p, key):
    rows = [clean(json.loads(l)) for l in open(p) if l.strip()]
    return sorted(rows, key=lambda r: json.dumps(r.get(key) if key else r, sort_keys=True))


a, b = sys.argv[1], sys.argv[2]
ok = True
for f, key in (("runs.jsonl", "seed"), ("fights.jsonl", None)):
    ra, rb = load(os.path.join(a, f), key), load(os.path.join(b, f), key)
    same = ra == rb
    ok &= same
    print(f"{f}: {len(ra)} vs {len(rb)} rows, identical={same}")
    if not same:
        for x, y in zip(ra, rb):
            if x != y:
                diff = {k: (x.get(k), y.get(k)) for k in set(x) | set(y) if x.get(k) != y.get(k)}
                print("  first diff:", json.dumps(diff)[:600])
                break


def digest(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "**", "*"), recursive=True)):
        if os.path.isfile(p):
            out[os.path.relpath(p, d)] = (os.path.getsize(p), hashlib.md5(open(p, "rb").read()).hexdigest())
    return out


for sub in ("mdata", "cdata"):
    da, db = digest(os.path.join(a, sub)), digest(os.path.join(b, sub))
    same = da == db
    ok &= same
    print(f"{sub}: {len(da)} vs {len(db)} files, {sum(s for s, _ in da.values())} vs {sum(s for s, _ in db.values())} bytes, identical={same}")
print("ALL IDENTICAL" if ok else "DIFFERENT")
