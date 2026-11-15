"""A20: outcome head vs scalar head, per evaluation arm, paired on the same held-out battle starts (same chance seeds)."""
import json, math, sys
a_dir, b_dir = sys.argv[1], sys.argv[2]
arms = sys.argv[3:] or ["net", "s16", "s64", "ot64", "ex64"]
def load(d, arm):
    return {r["id"]: r for r in map(json.loads, open(f"{d}/{arm}.jsonl")) if r["result"] != "error"}
print(f"{'arm':6s} {'A won/rel/pot':>24s} {'B won/rel/pot':>24s}   paired rel diff (A - B), 95% CI   [A = first dir, B = second dir]")
for arm in arms:
    A, B = load(a_dir, arm), load(b_dir, arm)
    ids = sorted(set(A) & set(B)); n = len(ids)
    def stats(R):
        won = sum(R[i]["result"] == "won" for i in ids) / n; rel = sum(R[i]["rel"] for i in ids) / n
        pot = sum(R[i]["potions_used"] for i in ids) / max(1, sum(R[i]["potions_start"] for i in ids))
        return f"{100*won:5.1f}% {rel:.4f} {100*pot:4.1f}%"
    d = [A[i]["rel"] - B[i]["rel"] for i in ids]; m = sum(d) / n; sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    print(f"{arm:6s} {stats(A):>24s} {stats(B):>24s}   {m:+.4f} ({m - 1.96*sd/math.sqrt(n):+.4f} to {m + 1.96*sd/math.sqrt(n):+.4f}), n {n}")
