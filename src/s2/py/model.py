"""Combat network construction and checkpoints for S2 (model class: src/s0/py/combat_obs.py CombatNet).

Checkpoint = {"state_dict", "config": {"d", "layers", "heads", "head"[, "ext"]}, "version": int (bumped by the trainer on
every save)}.
head: "outcome" (41 end-HP bins + 6 potion-kept logits, priced by the T_f table in the search; default since 2026-11-11)
or "scalar" (P(survive) logit + relative T_f, the first design; kept for comparison).
ext: 1 = the network also reads the layout-v4/v5 feature slots (card damage / block, hidden power counters, relic counters) through
`ext_feat` (combat_obs.CombatNet); absent / 0 = the v3 network (masks those slots). A checkpoint without ext weights
(any v3 checkpoint) loads into an ext network: ext_feat keeps its fresh init (zero last layer), so the outputs equal the
checkpoint's. Everything else loads strictly.
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "s0", "py"))
import combat_obs as co  # noqa: E402

DEFAULT = {"d": 128, "layers": 2, "heads": 4, "head": "outcome"}
N_VALUE = {"outcome": 41 + 6, "scalar": 2}
HP_BINS, POTION_SLOTS = 41, 6
ARCH_KEYS = ("d", "layers", "heads", "head", "ext")   # config keys that change the module structure


def config_of(ck=None, config=None):
    """The network configuration a checkpoint (and optional overrides) describes."""
    cfg = dict(DEFAULT)
    if ck is not None and "config" in ck:
        cfg["head"] = "scalar"  # checkpoints written before the outcome head carry no "head" key
        cfg.update(ck["config"])
    if config:
        cfg.update(config)
    return cfg


def arch(cfg):
    """Structure-defining part of a config (equal arch = the state dicts are interchangeable)."""
    return tuple(int(cfg.get(k, 0)) if k == "ext" else cfg.get(k) for k in ARCH_KEYS)


def load_weights(net, sd):
    """Loads a state dict; missing ext_feat weights are allowed (the network's ext_feat is left as it is: a fresh one
    from construction, zero last layer); any other missing or unexpected key raises."""
    missing, unexpected = net.load_state_dict(sd, strict=False)
    ext_missing = [k for k in missing if k.startswith("ext_feat.")]
    other = [k for k in missing if not k.startswith("ext_feat.")]
    if unexpected or other or (ext_missing and len(ext_missing) != len([k for k in net.state_dict() if k.startswith("ext_feat.")])):
        raise RuntimeError(f"checkpoint does not fit the network: missing {missing}, unexpected {unexpected}")
    return bool(ext_missing)


def load_net(vocab, ck=None, config=None):
    cfg = config_of(ck, config)
    cfg["ext"] = int(cfg.get("ext", 0))
    if not cfg["ext"]:
        cfg.pop("ext")      # v3 networks keep the old config (no new key in their checkpoints)
    net = co.CombatNet(vocab, d=cfg["d"], layers=cfg["layers"], heads=cfg["heads"], n_value=N_VALUE[cfg["head"]],
                       dropout=cfg.get("dropout", 0.0), vdrop=cfg.get("vdrop", 0.0), ext=cfg.get("ext", 0))
    net.config = cfg
    if ck is not None:
        load_weights(net, ck["state_dict"] if "state_dict" in ck else ck)
    return net
