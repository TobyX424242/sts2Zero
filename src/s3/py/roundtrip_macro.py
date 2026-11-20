"""Round trip of the macro encoder (S3_HANDOVER §9.1): the rows the C# encoder produced for the state after every floor
of simulated runs (S3 encode ...: rows.jsonl) against the examples the S1 dataset builder
(src/value_v1/build_floor_dataset.py) rebuilds from the same runs written as Spire Codex style run records
(runs.jsonl). Decks, relics and potions are compared as multisets (the networks are set encoders), context and scalars
exactly, and the S1 network's value on both encodings.

Usage: ./.venv/bin/python -I src/s3/py/roundtrip_macro.py src/s3/out/encode [MACRO.pt] [MVOCAB.json]
ic113: the macro file must use the vocabulary the C# encoder used (default src/s3/data/macro_vocab_v113.json; a
v0.111 file is converted by src/ic113/convert_v113.py); character and era come from the run records.
"""
import collections, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import torch  # noqa: E402
import macro_s3 as m  # noqa: E402

sys.path.insert(0, os.path.join(HERE, "..", "..", "value_v1"))
import build_floor_dataset as bfd  # noqa: E402


def main():
    d = sys.argv[1]
    mp = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "data", "macro_s1.pt")
    vpath = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "..", "data", "macro_vocab_v113.json")
    ens = m.Ensemble.load(mp, "cpu")
    voc = json.load(open(vpath))
    bad = [k for k in ("card", "ench", "relic", "potion", "encounter") if list(ens.vocab[k]) != list(voc[k])]
    assert not bad, f"{mp} and {vpath} differ in {bad}: convert the macro file (src/ic113/convert_v113.py)"
    bfd.init_worker(ens.vocab)
    rows = collections.defaultdict(dict)
    for line in open(os.path.join(d, "rows.jsonl")):
        r = json.loads(line)
        rows[r["run"]][r["floor"]] = (np.array(r["ints"], np.int64), np.array(r["floats"], np.float32))
    o = m.OFF
    stats = collections.Counter(); bad = collections.Counter(); examples = []
    pi, pf, ci, cf = [], [], [], []
    for line in open(os.path.join(d, "runs.jsonl"), "rb"):
        run = json.loads(line)
        meta = {"sample_id": 0, "character": run["players"][0]["character"].replace("CHARACTER.", ""), "ascension": run["ascension"],
                "era": int(run.get("era", m.ERA_V0111))}
        _, arr, miss, _, dropped, F, insane = bfd.build_run((line, meta))
        stats["runs"] += 1; stats["undo_misses"] += sum(miss.values()); stats["future_dropped"] += dropped
        if insane or not arr:
            stats["runs_rejected_" + (insane or "empty")] += 1
            continue
        mine = rows[run["seed"]]
        # floors before a logged relic removal or card downgrade: the log-based rebuild is ambiguous there (a removed
        # relic gets age 63 on earlier floors; a downgrade is undone on an arbitrary copy of the card), so relic ages /
        # upgrade levels are not compared on those floors
        g_ = 0; last_relic_removed = last_downgrade = 0; removed_toks = set()
        ids_ = [c["id"] for c in run["players"][0]["deck"]] + [c["id"] for act in run["map_point_history"] for fl in act
                                                              for c in fl["player_stats"][0].get("cards_removed", [])]
        run_dup = {bfd.tok("card", i) for i, n in collections.Counter(ids_).items() if n > 1}
        for act in run["map_point_history"]:
            for fl in act:
                g_ += 1
                ps_ = fl["player_stats"][0]
                if ps_.get("relics_removed"):
                    last_relic_removed = g_; removed_toks |= {bfd.tok("relic", x) for x in ps_["relics_removed"]}
                if ps_.get("downgraded_cards"): last_downgrade = g_
        py_floors = set(int(g) for g in arr["floor"])
        stats["floors_python"] += len(py_floors); stats["floors_csharp"] += len(mine)
        stats["floors_only_python"] += len(py_floors - set(mine)); stats["floors_only_csharp"] += len(set(mine) - py_floors)
        for i, g in enumerate(arr["floor"]):
            g = int(g)
            if g not in mine:
                continue
            I, Fl = mine[g]
            stats["floors_compared"] += 1
            ok = True

            def cmp(name, a, b):
                nonlocal ok
                if a != b:
                    ok = False; bad[name] += 1
                    if len(examples) < 12:
                        examples.append((run["seed"], g, name, str(a)[:300], str(b)[:300]))
            cs = lambda *keys: collections.Counter(zip(*[I[o[k][0]:o[k][1]].tolist() for k in keys]))
            ps = lambda *keys: collections.Counter(zip(*[arr[k][i].astype(np.int64).tolist() for k in keys]))
            dk = ("d_ids", "d_upg", "d_ench", "d_eamt", "d_age")
            if g < last_downgrade:
                dk = ("d_ids", "d_ench", "d_eamt", "d_age"); stats["floors_upgrade_levels_not_compared"] += 1
            a, b = cs(*dk), ps(*dk); a.pop((0,) * len(dk), None); b.pop((0,) * len(dk), None)
            upg_ambiguous = False
            if a != b and "d_upg" in dk:
                # the rebuild undoes an upgrade on an arbitrary copy when a card id has several copies: accept decks that
                # are equal up to which copy of a duplicated card is upgraded (same ids, same number of upgrade levels per id)
                dk2 = ("d_ids", "d_ench", "d_eamt", "d_age")
                a2, b2 = cs(*dk2), ps(*dk2); a2.pop((0,) * 4, None); b2.pop((0,) * 4, None)
                diff_ids = {k[0] for k in (a - b) + (b - a)}
                if a2 == b2 and diff_ids <= run_dup:
                    upg_ambiguous = True; stats["floors_duplicate_upgrade_ambiguity"] += 1
            if not upg_ambiguous and a != b:
                # the run history logs an enchantment once, without its amount: the rebuild shows the final amount on every
                # floor since, the live encoder the amount at that floor. Accept decks equal up to enchantment amounts (and
                # the duplicate-upgrade ambiguity above)
                dk3 = ("d_ids", "d_ench", "d_age")
                a3, b3 = cs(*dk3), ps(*dk3); a3.pop((0,) * 3, None); b3.pop((0,) * 3, None)
                if a3 == b3 and all(k[2] != 0 or k[0] in run_dup for k in (a - b) + (b - a)):
                    upg_ambiguous = True; stats["floors_enchant_amount_drift"] += 1
            # a deck of more than MAX_DECK cards (e.g. the Reflections event doubles it): both sides encode the first 64
            # cards of their own list (the live deck order vs the rebuilt order), so which cards are cut differs; only the
            # scalars (full deck size, upgrades) are compared on such floors (ic113)
            trunc = bool(arr["deck_trunc"][i])
            if trunc:
                stats["floors_deck_over_64_not_compared"] += 1
            elif not upg_ambiguous:
                cmp("deck", a, b)
            rk = ("r_ids", "r_age")
            if g < last_relic_removed:
                rk = ("r_ids",); stats["floors_relic_ages_not_compared"] += 1
            a, b = cs(*rk), ps(*rk); a.pop((0,) * len(rk), None); b.pop((0,) * len(rk), None)
            if g < last_relic_removed:   # the rebuild puts a later-removed relic on every earlier floor
                a = collections.Counter({k: v for k, v in a.items() if k[0] not in removed_toks})
                b = collections.Counter({k: v for k, v in b.items() if k[0] not in removed_toks})
            cmp("relics", a, b)
            a, b = cs("p_ids"), ps("p_ids"); a.pop((0,), None); b.pop((0,), None); cmp("potions", a, b)
            cmp("ctx", I[o["ctx"][0]:o["ctx"][1]].tolist(), arr["ctx"][i].astype(np.int64).tolist())
            sc_c, sc_p = Fl[:16], arr["scal"][i]
            for name, j in m.S.items():
                if (name == "n_upgraded" and (g < last_downgrade or upg_ambiguous)) or (name == "n_relics" and g < last_relic_removed):
                    continue
                if abs(float(sc_c[j]) - float(sc_p[j])) > 1e-6:
                    cmp("scal." + name, round(float(sc_c[j]), 5), round(float(sc_p[j]), 5))
            stats["floors_equal"] += ok
            # the Python example as a row, for the value comparison
            J = np.zeros(m.IW, np.int64); G = np.zeros(m.FW, np.float32)
            for k in ("d_ids", "d_upg", "d_ench", "d_eamt", "d_age", "r_ids", "r_age", "p_ids", "ctx"):
                J[o[k][0]:o[k][1]] = arr[k][i]
            G[:16] = arr["scal"][i]
            if ok and not upg_ambiguous and not trunc and g >= last_downgrade and g >= last_relic_removed:
                stats["floors_identical_inputs"] += 1
                pi.append(J); pf.append(G); ci.append(I); cf.append(Fl)
    with torch.no_grad():
        vp = ens.serve(np.stack(pi), np.stack(pf), "cpu")[:, m.O_V].numpy()
        vc = ens.serve(np.stack(ci), np.stack(cf), "cpu")[:, m.O_V].numpy()
    dv = np.abs(1 / (1 + np.exp(-vp)) - 1 / (1 + np.exp(-vc)))
    out = {**stats, "mismatches_by_field": dict(bad), "value_abs_diff_max": float(dv.max()), "value_abs_diff_mean": float(dv.mean()),
           "mean_value": float((1 / (1 + np.exp(-vc))).mean())}
    # potions differ only when more than five are held (the dataset keeps the last five acquired, the encoder the last
    # five slots); decks over 64 cards are not compared card by card (counted above); anything else must stay below 0.1% of floors
    other = sum(v for k, v in bad.items() if k not in ("potions", "scal.n_potions"))
    out["PASS"] = bool(other <= 0.001 * stats["floors_compared"] and out["value_abs_diff_max"] < 1e-5
                       and stats["floors_only_python"] == 0 and stats["floors_only_csharp"] == 0)
    print(json.dumps(out, indent=1))
    for e in examples:
        print("MISMATCH", e)


if __name__ == "__main__":
    main()
