"""Summary of an eval_ckpt.sh directory: per arm, fights, win rate, mean relative T_f and mean T_f with 95% CIs, and
paired differences against the first arm on the battle starts both arms finished."""
import json, math, sys

d, arms = sys.argv[1], [a.split(":")[0] for a in sys.argv[2:]]
res = {}
for a in arms:
    rs = [json.loads(l) for l in open(f"{d}/{a}.jsonl")]
    res[a] = {r["id"]: r for r in rs if r["result"] != "error"}
    n = len(res[a]); errs = len(rs) - n
    rel = [r["rel"] for r in res[a].values()]; sc = [r["score"] for r in res[a].values()]
    m = sum(rel) / n; sd = math.sqrt(sum((x - m) ** 2 for x in rel) / max(1, n - 1))
    won = sum(r["result"] == "won" for r in res[a].values())
    pot = sum(r["potions_used"] for r in res[a].values()) / max(1, sum(r["potions_start"] for r in res[a].values()))
    print(f"{a:8s} fights {n} (errors {errs}), won {100 * won / n:.1f}%, mean rel T_f {m:.4f} ± {1.96 * sd / math.sqrt(n):.4f}, "
          f"mean T_f {sum(sc) / n:.4f}, potions used {100 * pot:.1f}%, decisions/fight {sum(r['decisions'] for r in res[a].values()) / n:.1f}")
base = arms[0]
for a in arms[1:]:
    ids = sorted(set(res[a]) & set(res[base]))
    dif = [res[a][i]["rel"] - res[base][i]["rel"] for i in ids]
    m = sum(dif) / len(dif); sd = math.sqrt(sum((x - m) ** 2 for x in dif) / max(1, len(dif) - 1))
    print(f"{a} - {base}: paired mean rel diff {m:+.4f} (95% CI {m - 1.96 * sd / math.sqrt(len(dif)):+.4f} to {m + 1.96 * sd / math.sqrt(len(dif)):+.4f}), n {len(dif)}")
