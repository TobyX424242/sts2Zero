"""ic113 combat warm-up (src/ic113/WARMUP.md): coverage-mutation pool and evaluation subsets.

  python3 -I src/ic113/warmup_pool.py pool CARD_INFO TRAIN_STARTS OUT_POOL_JSON [--eps 0.01]
      Ironclad pool cards except the Basic and the multiplayer-only ones, each weighted 1 / (f + eps), f = share of the
      training starts whose deck holds the card: never-seen cards get 1/eps, a card in 30% of decks ~3.
  python3 -I src/ic113/warmup_pool.py evalset EVAL_STARTS TABLES OUT_ALL OUT_SUB N_SUB [--seed 113]
      held-out elite and boss starts that have a table (OUT_ALL, file order), and a fixed random subset of N_SUB of
      them (OUT_SUB) for the hourly progress check.
  python3 -I src/ic113/warmup_pool.py counts CARD_COUNTS_JSON... [--pool OUT_POOL_JSON]
      sums the actors' card_counts files: per Ironclad card the fights whose deck held it and the plays, least-seen first.
"""
import json, random, sys


def pool(card_info, train, out, eps=0.01):
    ci = json.load(open(card_info))["card"]
    # multiplayer-only cards (ally targets, e.g. Blaze, Demonic Shield, Tank, Midnight, Outrage) never occur in a
    # single-player run; card_info files written before 2026-11-18 10:00 lack the flag (the w2 pool included those 5)
    ic = sorted(k for k, v in ci.items() if "IRONCLAD" in v["pools"] and v["rarity"] != "Basic" and not v.get("multiplayer_only", False))
    n, N = {k: 0 for k in ic}, 0
    for line in open(train):
        s = json.loads(line); N += 1
        for c in {x["id"] for x in s["deck"]}:
            if c in n:
                n[c] += 1
    w = {k: round(1.0 / (n[k] / N + eps), 4) for k in ic}
    json.dump(w, open(out, "w"), indent=0, sort_keys=True)
    tot = sum(w.values())
    print(f"{len(ic)} Ironclad cards (Basic excluded) over {N} training starts; draw share of the 10 rarest:")
    for k in sorted(ic, key=lambda k: n[k])[:10]:
        print(f"  {k:28s} in {n[k] / N:7.4f} of decks, weight {w[k]:7.2f}, share of swaps {w[k] / tot:.3f}")


def evalset(starts, tables, out_all, out_sub, n_sub, seed=113):
    have = {json.loads(l)["id"] for l in open(tables)}
    keep = [l for l in open(starts) if json.loads(l)["room_type"] in ("elite", "boss") and json.loads(l)["id"] in have]
    open(out_all, "w").writelines(keep)
    sub = random.Random(seed).sample(keep, min(n_sub, len(keep)))
    open(out_sub, "w").writelines(sub)
    print(f"{len(keep)} held-out elite / boss starts with tables -> {out_all}; subset {len(sub)} -> {out_sub}")


def counts(files, pool_path=None):
    tot, fights, mut = {}, 0, 0
    for p in files:
        d = json.load(open(p))
        fights += d["fights"]; mut += d["mutated_starts"]
        for k, v in d["cards"].items():
            t = tot.setdefault(k, [0, 0, 0, 0])
            for i, f in enumerate(("fights_in_deck", "plays", "mutated_fights", "mutated_plays")):
                t[i] += v[f]
    ids = sorted(json.load(open(pool_path))) if pool_path else sorted(k for k in tot if k.startswith("CARD."))
    print(f"{fights} fights, {mut} mutated starts; {len(ids)} cards; least played first:")
    print(f"{'card':28s} {'fights in deck':>14s} {'plays':>9s} {'mut. fights':>11s} {'mut. plays':>10s}")
    for k in sorted(ids, key=lambda k: tot.get(k, [0, 0])[1]):
        t = tot.get(k, [0, 0, 0, 0])
        print(f"{k:28s} {t[0]:14d} {t[1]:9d} {t[2]:11d} {t[3]:10d}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "pool":
        eps = float(a[a.index("--eps") + 1]) if "--eps" in a else 0.01
        pool(a[1], a[2], a[3], eps)
    elif a[0] == "evalset":
        seed = int(a[a.index("--seed") + 1]) if "--seed" in a else 113
        evalset(a[1], a[2], a[3], a[4], int(a[5]), seed)
    elif a[0] == "counts":
        pp = a[a.index("--pool") + 1] if "--pool" in a else None
        fs = [x for i, x in enumerate(a[1:], 1) if x != "--pool" and (i < 2 or a[i - 1] != "--pool")]
        counts(fs, pp)
