# src/ic113/STALEMATE.md: fights, stalemates and fight length per replay dir (trace.txt of stalemate_rep.sh).
# usage: python3 stalemate_summary.py OUTDIR...
import re,sys,statistics
for d in sys.argv[1:]:
    rounds=[]; stale=0; hp=[]
    for l in open(d+"/trace.txt"):
        if not l.startswith("==="): continue
        if "STALEMATE" in l: stale+=1; continue
        m=re.search(r"won end hp (\d+) rounds (\d+)", l)
        if m: rounds.append(int(m.group(2))); hp.append(int(m.group(1)))
    n=len(rounds)+stale
    print(f"{d}: SLIMES_WEAK fights {n}, stalemates {stale}, won {len(rounds)}: median rounds {statistics.median(rounds)}, >20 rounds {sum(r>20 for r in rounds)}, max {max(rounds)}, mean end hp {statistics.mean(hp):.1f}")
