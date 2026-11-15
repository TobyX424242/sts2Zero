"""Round trip C# encoder -> Python decoder (layout v3 or v4: features are compared as written, every slot), plus a
CombatNet forward pass.

Usage: ./.venv/bin/python -I src/s0/py/test_roundtrip.py [obs_dir] [vocab.json]
Checks, per state: the decoded token multisets (per kind: id, id2, features) equal the C# originals in check.jsonl;
the action rows match the C# action list (count, kind, source/target token kinds). Then runs CombatNet (CPU) on
all states and checks shapes, masking and finiteness.
"""
import json, os, sys, time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import combat_obs as co  # noqa: E402

root = os.path.dirname(os.path.dirname(HERE))
d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "s0/data/obs_v3_1000")
vp = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "s0/data/combat_vocab_v1.json")
vocab = co.load_vocab(vp)
obs, act = co.load_arrays(d)
bad_tok = bad_act = 0
n_prompt_rows = [0]
n_tok = 0
kinds = Counter()
for line in open(os.path.join(d, "check.jsonl")):
    r = json.loads(line)
    i = r["i"]
    want = co.multisets([(k, a, b, tuple(f)) for k, a, b, f in r["tokens"]])
    got_tokens = co.decode_state(obs[i], vocab)
    got = co.multisets(got_tokens)
    n_tok += len(got_tokens)
    kinds.update(k for k, *_ in got_tokens)
    if want != got:
        bad_tok += 1
        if bad_tok <= 3:
            print("token mismatch in state", i, {k: (want.get(k), got.get(k)) for k in set(want) | set(got) if want.get(k) != got.get(k)})
    rows = act[i][act[i][:, 0] > 0]
    ok = len(rows) == len(r["actions"])
    for row, (kind, key, src) in zip(rows, r["actions"]):
        ok &= row[0] == kind + 1
        if kind == 1:  # card: source must be a hand token, target an enemy/global token or none
            ok &= obs[i][row[1], 0] == 2 and (row[2] == -1 or obs[i][row[2], 0] in (1, 6))
        if kind == 2:
            ok &= obs[i][row[1], 0] == 10
        if "|S:" in key:  # prompt answer: code 1 + chosen cards; first chosen card's id and upgrade in columns 4-5
            chosen = [c for part in key.split("|S:")[1:] for c in part.split("&") if c]
            ok &= row[3] == 1 + len(chosen)
            n_prompt_rows[0] += 1
            if chosen:
                c0 = chosen[0][len("bundle:"):].split("+")[0] if chosen[0].startswith("bundle:") else chosen[0]
                cid, up = (c0.split("#") + ["0"])[:2] if "#" in c0 else (c0, "0")
                want_id = vocab["card"].index(cid) if cid in vocab["card"] else 1
                ok &= row[4] == want_id and row[5] == int(up) + 1
        else:
            ok &= row[3] == 0
    bad_act += not ok
n = len(obs)
print(f"states {n}, tokens {n_tok} (per kind {dict(sorted((co.KINDS[k], v) for k, v in kinds.items()))})")
print(f"round trip: token multiset mismatches {bad_tok}/{n}; action-row mismatches {bad_act}/{n}; prompt-answer rows checked {n_prompt_rows[0]}")

import torch  # noqa: E402
torch.manual_seed(0)
net = co.CombatNet(vocab)
params = sum(p.numel() for p in net.parameters())
ds = co.CombatObsDataset(d)
dl = torch.utils.data.DataLoader(ds, batch_size=256)
t0 = time.time()
finite = masked_ok = True
with torch.no_grad():
    for b in dl:
        logits, v = net(b["obs"], b["act"])
        surv = v[:, 0]
        assert logits.shape == (b["obs"].shape[0], co.MAX_ACTIONS) and v.shape == (b["obs"].shape[0], 2)
        valid = b["act"][..., 0] > 0
        finite &= bool(torch.isfinite(logits[valid]).all() and torch.isfinite(surv).all())
        masked_ok &= bool(torch.isneginf(logits[~valid]).all())
        probs = torch.softmax(logits, -1)
        assert torch.allclose(probs.sum(-1), torch.ones(len(probs)), atol=1e-5)
print(f"CombatNet: {params:,} parameters; forward on {n} states (CPU) {time.time() - t0:.2f} s; finite {finite}; padding masked {masked_ok}")
sys.exit(0 if bad_tok == 0 and bad_act == 0 and finite and masked_ok else 1)
