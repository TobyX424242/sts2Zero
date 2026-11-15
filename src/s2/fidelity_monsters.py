"""Boss/monster fidelity check (laptop only; reads the locally decompiled game assembly in .cache/decomp_v0111, which
never leaves the laptop): for every monster class present in both the game and iRyougi, compare the multiset of integer
literals (HP, damage, block, power amounts, hit counts; float/animation literals ignored). Prints classes whose
literals differ, with the literals only one side has, boss-encounter monsters first.
Usage: python3 src/s2/fidelity_monsters.py"""
import collections, glob, os, re
G = ".cache/decomp_v0111/MegaCrit.Sts2.Core.Models.Monsters"
S = "src/sims/iRyougi-sts2-sim/src/Sts2Sim.Core/Models/Monsters"
num = re.compile(r"(?<![\w.])(-?\d+)(m)?(?![\w.]|f\b)")
def lits(path):
    t = open(path, encoding="utf-8", errors="replace").read()
    t = re.sub(r"//.*", "", t); t = re.sub(r'"[^"]*"', '""', t)
    t = re.sub(r"\d+\.\d+f?", "", t)                       # floats (animation timings)
    out = collections.Counter()
    for m in num.finditer(t):
        v = int(m.group(1))
        if v in (0, 1): continue                            # loop/index noise
        out[v] += 1
    return out
boss_src = open(glob.glob("src/sims/iRyougi-sts2-sim/src/Sts2Sim.Core/Content/Acts/*Bosses*.cs")[0]).read() if glob.glob("src/sims/iRyougi-sts2-sim/src/Sts2Sim.Core/Content/Acts/*Bosses*.cs") else ""
boss_files = set()
for f in glob.glob("src/sims/iRyougi-sts2-sim/src/Sts2Sim.Core/Content/Acts/*Bosses*.cs"):
    boss_files |= set(re.findall(r"Monster<(\w+)>|typeof\((\w+)\)|new (\w+)\(", open(f).read()) and
                      [x for tup in re.findall(r"Monster<(\w+)>|typeof\((\w+)\)", open(f).read()) for x in tup if x])
gm = {os.path.basename(p)[:-3]: p for p in glob.glob(f"{G}/*.cs")}
sm = {os.path.basename(p)[:-3]: p for p in glob.glob(f"{S}/*.cs")}
both = sorted(set(gm) & set(sm))
print(f"monster classes: game {len(gm)}, sim {len(sm)}, both {len(both)}; only in game: {sorted(set(gm) - set(sm))[:20]}")
rows = []
for name in both:
    a, b = lits(gm[name]), lits(sm[name])
    if a != b:
        rows.append((name in boss_files, name, dict(a - b), dict(b - a)))
rows.sort(key=lambda r: (not r[0], r[1]))
print(f"classes with differing literals: {len(rows)} of {len(both)} (boss-encounter monsters marked *)")
for boss, name, ga, sa in rows:
    print(f"{'*' if boss else ' '} {name:28s} game-only {ga}  sim-only {sa}")
