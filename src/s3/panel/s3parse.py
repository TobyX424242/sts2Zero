"""Parse the local copy of an S3 chain (src/server_backup/<root>/) into the series the watch panel shows.

A chain is one supervisor run: s3a (2026-11-11..15, layout v1/v2), s3c (2026-11-16, continues s3a segment 49), s3v3
(from 2026-11-16 14:00 UTC, layout v3 macro network, 52 actors, continues s3c segment 0), the experiment arms lsa / lsb,
s3ic (Ironclad A10 on game v0.113, from 2026-11-18 12:19 UTC; its own Nextcloud root, no parent chain), and whatever
comes next. build(run) parses one chain: <run>/ (status.txt, curve.txt, supervisor.log, seg_N.log, eval_N.log,
eval/seg_NNN/, and for s3ic also curriculum_stage, hook_status.txt, segment.env, vmix/vbar_sNNN.json, eval_report.txt)
and <run>_sNNN/meta/ (macro / combat trainer logs, run and fight records, actor and server logs). It also walks the
chain's lineage (parent_of: the first "start weights .../<chain>_sNNN/..." in supervisor.log, else KNOWN_PARENT; chains
in NO_PARENT have none) and puts the fixed-seed evaluations of all ancestors up to the branch point into
payload["lineage"], with runs made cumulative over the chains (curve.txt counts runs from each chain's own start).

Nextcloud roots: PANEL_REMOTE lists them (default nextcloud:sts2_backup/s3 and nextcloud:sts2_backup/s3ic). Each root
has its own local directory: the first one PANEL_DATA (default src/server_backup/s3), every other one a sibling
named after the root (src/server_backup/s3ic). base_of(chain) finds a chain's local directory.
Which chain is shown: PANEL_RUN pins one; otherwise the panel picks the chain whose status.txt is the newest (on
Nextcloud, see s3sync.remote_chains; newest_local_chain() is the offline fallback) unless one is chosen in the page.

Everything is read from local files; nothing here touches the network. Per-file aggregates of the larger files (run
records, fight records, actor and server logs) are cached under src/s3/panel/cache/ keyed by file size + mtime,
so a reload reads only what changed. Missing files, a segment without meta, cut-off last lines and segments that
failed or were stopped early are all normal inputs: what is missing is reported in the payload, never guessed.

Interruptible servers (2026-11-19): parse_supervisor reads the restarts of s3_resume.sh ("downtime N min", "no supervisor
for X"), the adoption of the interrupted segment ("segment K recovered after interruption" -> result "recovered", a real
partial segment; "interrupted and NOT adopted" -> "discarded", a gap), evaluations run again, carry backups, pauses and
stops, and end-time moves; build() adds the interruption list (build_interruptions), the hosts (build_hosts: status.txt
names the host; KNOWN_HOSTS), and shifts the run counts of a chain restored on another host (the supervisor there counts
only the run records that reached Nextcloud).

Robustness: a line that does not parse, a record with fields of an unexpected type, or a segment whose files break a
parser is counted / reported in the payload (segment "missing" list, notes) and never stops the page.

CLI (prints a short text summary of what was parsed):  python3 src/s3/panel/s3parse.py [chain]
"""
import calendar, glob, hashlib, json, math, os, re, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
PINNED = os.environ.get("PANEL_RUN") or None      # PANEL_RUN pins the chain; otherwise the panel follows the newest
RUN = PINNED or "s3a"                               # default of build() / CLI when nothing else is known
# lineage when supervisor.log does not name the parent segment (s3c started from a copy: /root/exp/s049init/)
KNOWN_PARENT = {"s3c": ("s3a", 49), "s3v3": ("s3c", 0)}
# chains that continue no other chain (s3ic starts from converted s3v3_s005 weights + an Ironclad warm-up: other
# character, other game version; its curve stands alone)
NO_PARENT = {"s3ic"}
# segments before this one are restarts / tests and are hidden by the page's "counted" filter
COUNTED_FROM = {"s3a": 4}
# hard-coded run changes (see build(): the server's files show them only later)
EXTENDED_END = {"s3a": 1794650908}          # 2026-11-13 22:08:28 UTC: s3a extended by 24 h (owner, 2026-11-13)
RUN_NOTES = {"s3a": [
    "Run extended by another 12 hours (owner, 2026-11-13): it now ends 2026-11-14 10:08 UTC (the waiter restarts the chain at 2026-11-13 22:08). "
    "Moved to the new server (TR 9980X + RTX PRO 6000, 60 actors) from segment 19 (rerun), 2026-11-13 05:51 UTC. "
    "Earlier: extended by 24 hours (owner, 2026-11-12) to 2026-11-13 22:08 UTC. The segment running at the original "
    "end (22:08 UTC; segment 15, about 46 minutes) is cut short and not evaluated; the chain resumes right after.",
    "Settings changes: tree reuse off from segment 13; practice copies at the normal 64/16 search mix and 29 actors "
    "(was 26) from segment 14.",
    "From segment 19: 60% of runs at A10 (was 50%), the rest at A5-A9 only, weighted toward A7-A9 (owner). Fight-scoring "
    "options F1/F2 are built but off: no gain in a 1,000-seed A/B.",
    "New training methods from segment 12 on (started 2026-11-12 14:40 UTC; segment 11 was stopped early): hallway search "
    "halved, every boss move searched at 64, one practice copy of each act 2/3 boss fight, tree reuse, surprise-weighted "
    "combat training (0.5), averaged acting weights (0.999), replay and optimizer state carried across segments. "
    "Compare segments 12+ with 4-11.",
], "s3ic": [
    "Ironclad A10 self-play on game v0.113.1 (src/ic113/CHAIN.md, PLAN steps 7 and 8). Started 2026-11-18 12:19 UTC "
    "for 168 hours, 2-hour segments, 60 actors. Nextcloud root sts2_backup/s3ic. It continues no chain: the macro network is "
    "s3v3 segment 5 converted to v0.113 (character=ironclad, era 6), the combat network is the 4.1-hour Ironclad combat "
    "warm-up (harness 59.8 -> 85.1%, full runs A10 3.1%, A0 52.9%). Fresh Adam state, empty replay windows.",
    "Recipe = the live s3v3 one, with: tf_heal=1; outcome target with alpha fixed at 0.70 (0.7 won + 0.3 acts cleared / 3); "
    "KL to S1 off; combat weight decay on weight matrices only; Fake Merchant purchases by the shop head (fake_merchant=1).",
    "Ascension curriculum, never down: stage 1 = 20% of runs at A10, the rest A0-A9; stage 2 when A10 self-play win >= 5% or "
    "A0-A4 >= 50% (over >= 300 runs) = 30% A10, the rest A5-A9; stage 3 when A10 >= 12% = A8 32%, A9 41%, A10 27%. "
    "Belt rule = mixed potion valuation (potion_mix, replaces the Defect potion table): Vbar per act from the previous "
    "segment's runs, scored with the new macro; segment 0 has no Vbar (state term alone).",
    "Evaluation: A10 on seeds Y100000-Y100299 (300) with the deployment rules (greedy map, look 20, potion_mix with the "
    "segment's Vbar, fake_merchant): not comparable with the s3v3 curve (Defect, v0.111, plain tf64). Self-play training "
    "chunks: 1 in 4 kept on Nextcloud (never synced here); fight logs in full; actor memory guard at 8 GB.",
]}
BASE = os.environ.get("PANEL_DATA") or os.path.join(ROOT, "src", "server_backup", "s3")
REMOTES = [r.rstrip("/") for r in re.split(r"[,\s]+", os.environ.get("PANEL_REMOTE") or "nextcloud:sts2_backup/s3,nextcloud:sts2_backup/s3ic") if r]


def _roots():
    """[(remote root, local directory)]: the first root is stored in BASE, the others next to it, named after the root."""
    out = []; used = set()
    for i, r in enumerate(REMOTES):
        loc = BASE if i == 0 else os.path.join(os.path.dirname(BASE), re.sub(r"[^\w.-]", "_", r.rsplit("/", 1)[-1].rsplit(":", 1)[-1]) or "root%d" % i)
        if loc in used:
            loc += "_%d" % i
        used.add(loc); out.append((r, loc))
    return out


ROOTS = _roots()
CHAIN_ROOT = {}        # chain -> (remote root, local directory), from the last Nextcloud listing (s3sync.remote_chains)


def base_of(run):
    """Local directory that holds <run>/ and <run>_sNNN/: where the chain is on the laptop, else where Nextcloud has it."""
    for _, loc in ROOTS:
        if os.path.isdir(os.path.join(loc, run)):
            return loc
    if run in CHAIN_ROOT:
        return CHAIN_ROOT[run][1]
    return BASE


def remote_of(run):
    """Nextcloud root of a chain: from the last listing, else the root whose local directory has it, else the first."""
    if run in CHAIN_ROOT:
        return CHAIN_ROOT[run][0]
    b = base_of(run)
    return next((r for r, loc in ROOTS if loc == b), ROOTS[0][0])
# with PANEL_DATA (a test tree) the cache and state live next to that tree, not with the real run's
_SIDE = os.path.dirname(os.path.abspath(os.environ["PANEL_DATA"])) if os.environ.get("PANEL_DATA") else HERE
CACHE = os.path.join(_SIDE, "cache")
STATE = os.path.join(_SIDE, "state")
VERSION = 9            # bump when an aggregate's shape changes: old cache entries are then ignored
BIN = 300              # seconds per throughput bin
FRESH = 15             # a large file modified in the last FRESH seconds is probably still being downloaded


# ---------------------------------------------------------------- small helpers
def utc_ts(s):
    """'2026-11-12 10:42:26' (UTC) -> unix time, or None."""
    try:
        return calendar.timegm(time.strptime(s.strip(), "%Y-%m-%d %H:%M:%S"))
    except (ValueError, AttributeError):
        return None


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def sig(path):
    try:
        st = os.stat(path); return [st.st_size, st.st_mtime_ns]
    except OSError:
        return None


def read_text(path):
    try:
        with open(path, "r", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def jsonl(path):
    """Yields the JSON objects of a file; lines that do not parse (cut off by a crash or a download) are counted."""
    bad = 0
    try:
        with open(path, "rb") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    o = json.loads(line)
                except ValueError:
                    bad += 1; continue
                if isinstance(o, dict):
                    yield o
                else:
                    bad += 1
    except OSError:
        pass
    jsonl.bad = bad


jsonl.bad = 0
_mem = {}
_mem_lock = threading.Lock()


def cached(kind, paths, fn, disk=True):
    """fn(*paths) -> JSON-able aggregate, cached by the size + mtime of every path."""
    s = [sig(p) for p in paths]
    key = kind + "|" + "|".join(os.path.relpath(p, BASE) for p in paths)
    with _mem_lock:
        hit = _mem.get(key)
    if hit and hit[0] == s:
        return hit[1]
    cf = os.path.join(CACHE, kind, hashlib.sha1(key.encode()).hexdigest()[:20] + ".json")
    if disk:
        try:
            with open(cf) as f:
                c = json.load(f)
            if c.get("v") == VERSION and c.get("sig") == s and c.get("key") == key:
                with _mem_lock:
                    _mem[key] = (s, c["data"])
                return c["data"]
        except (OSError, ValueError):
            pass
    data = fn(*paths)
    if [sig(p) for p in paths] != s:       # changed while it was read: use it now, do not cache it
        return data
    with _mem_lock:
        _mem[key] = (s, data)
    if disk:
        try:
            os.makedirs(os.path.dirname(cf), exist_ok=True)
            tmp = cf + ".tmp%d" % threading.get_ident()
            with open(tmp, "w") as f:
                json.dump({"v": VERSION, "sig": s, "key": key, "data": data}, f)
            os.replace(tmp, cf)
        except OSError:
            pass
    return data


def room_kind(enc):
    e = enc or ""
    return "boss" if e.endswith("_BOSS") else "elite" if e.endswith("_ELITE") else "hallway" if e else "none"


def short_enc(e):
    return (e or "").replace("ENCOUNTER.", "")


# ---------------------------------------------------------------- status.txt
MACRO_HEAD = ("t", "iter", "runs", "new_runs", "win_rate", "mean_floors", "ascension", "acts_cleared_2plus", "act1_cleared")


def partial_macro(line):
    """Leading scalar fields of a macro-trainer JSON line that may be cut off (status.txt cuts lines at 700 characters)."""
    try:
        o = json.loads(line)
        return o if isinstance(o, dict) else None
    except ValueError:
        pass
    o = {}
    for k in MACRO_HEAD:
        m = re.search(r'"%s": (-?[0-9.]+(?:e-?[0-9]+)?)[,}]' % k, line)
        if m:
            o[k] = float(m.group(1))
    if "runs" not in o or "t" not in o:
        return None
    by = {}
    for m in re.finditer(r'"(\d+)": (\{"runs": [^{}]*\})', line):
        try:
            by[m.group(1)] = json.loads(m.group(2))
        except ValueError:
            pass
    if by:
        o["by_ascension"] = by
    o["_partial"] = True
    return o


STAGE_FIELDS = (("runs", r"runs (\d+)", int), ("act1", r"act 1 cleared ([0-9.]+)%", float), ("act2_of", r"act 2 cleared of those ([0-9.]+)%", float),
                ("boss_reach", r"act-3 runs reaching the boss ([0-9.]+)%", float), ("boss1", r"boss 1 won ([0-9.]+)%", float),
                ("boss2", r"boss 2 won ([0-9.]+)%", float), ("won", r"runs won ([0-9.]+)%", float))


def parse_stage_line(l):
    """One line of status.txt's "stages per directory" block (curve_runs.py): segment, ascension, act funnel."""
    m = re.match(r"\s+(\S+) A(\d+): (.*)$", l)
    if not m:
        return None
    o = {"dir": m.group(1), "asc": int(m.group(2))}
    head = m.group(3).split("act-3 runs lost at:")[0]
    for k, pat, cast in STAGE_FIELDS:
        mm = re.search(pat, head)
        o[k] = cast(mm.group(1)) if mm else None
    mm = re.search(r"act-3 runs lost at: (.*?)(?:\||$)", m.group(3))
    if mm:
        o["lost_at"] = [[a.strip(), float(b)] for a, b in re.findall(r"([a-z][a-z ]*?\d?) ([0-9.]+)%", mm.group(1))]
    return o


def parse_status(path):
    text = read_text(path)
    if text is None:
        return {"present": False}
    st = {"present": True, "mtime": os.path.getmtime(path), "raw": text[:20000]}
    lines = text.splitlines()
    if lines:
        m = re.match(r"S3 run (\S+) on (\S+): (.*)$", lines[0])
        if m:
            st["host"] = m.group(2); st["state"] = m.group(3).strip()
    s = st.get("state", "")
    st["kind"] = ("finished" if s.startswith("FINISHED") else "stopped" if s.startswith("STOPPED") else
                  "paused" if s.startswith("PAUSED") else
                  "evaluating" if s.startswith("evaluating") else "between" if s.startswith("between") else
                  "running" if "running" in s else "unknown")
    m = re.search(r"running \((\d+) of (\d+) s\)", s)
    if m:
        st["seg_elapsed"] = int(m.group(1)); st["seg_planned"] = int(m.group(2))
    m = re.search(r"updated ([0-9-]+ [0-9:]+) UTC", text)
    if m: st["updated"] = utc_ts(m.group(1))
    m = re.search(r"ends ([0-9-]+ [0-9:]+) UTC \((-?\d+) min left\)", text)
    if m: st["ends"] = utc_ts(m.group(1)); st["min_left"] = int(m.group(2))
    m = re.search(r"; segment (\d+); runs so far (\d+)", text)
    if m: st["segment"] = int(m.group(1)); st["runs_so_far"] = int(m.group(2))
    m = re.search(r"^weights in use: (.*)$", text, re.M)
    if m: st["weights"] = m.group(1).strip()
    m = re.search(r"^chain settings: (.*)$", text, re.M)       # s3ic: curriculum stage and Vbar file (hook_status.txt)
    if m: st["chain_settings"] = m.group(1).strip()
    m = re.search(r"^disk (\S*), load (\S*), gpu ?(.*)$", text, re.M)
    if m: st["disk"] = m.group(1).rstrip(","); st["load"] = m.group(2).rstrip(","); st["gpu"] = m.group(3).strip() or None
    # sections
    sec = {}; cur = None
    for l in lines:
        if l.startswith("--- "):
            cur = l[4:].split(" (")[0].split(",")[0].strip(); sec[cur] = []
        elif cur is not None:
            sec[cur].append(l)
    # self-play table (curve_runs.py output)
    sp = {"head": None, "groups": []}; stages = []; in_stages = False
    for l in sec.get("self-play by training order", []):
        if l.startswith("-- "):
            in_stages = l[3:].startswith("stages per directory")
            if not in_stages:
                sp["groups"].append({"name": l[3:].strip(), "rows": []})
        elif in_stages:
            x = parse_stage_line(l)
            if x:
                stages.append(x)
        elif sp["groups"] and re.match(r"\s+\d+: runs", l):
            g = lambda pat, cast=float: (lambda mm: cast(mm.group(1)) if mm else None)(re.search(pat, l))
            sp["groups"][-1]["rows"].append({
                "bucket": g(r"^\s+(\d+):", int), "runs": g(r"runs\s+(\d+)", int), "won": g(r"won\s+([0-9.]+)%"),
                "acts": g(r"acts cleared ([0-9.]+)"), "act1": g(r"act1\s+([0-9.]+)%"), "act2": g(r"act2\s+([0-9.]+)%"),
                "floor": g(r"floor\s+([0-9.]+)"), "elites": g(r"elites ([0-9.]+)"), "rest_share": g(r"rest share\s+([0-9.]+)%"), "deck": g(r"deck\s+([0-9.]+)"),
                "hallway": g(r"hallway ([0-9.]+)%"), "elite": g(r"elite ([0-9.]+)%"), "boss": g(r"boss ([0-9.]+)%"),
                "versions": g(r"versions (\S+)", str)})
        elif "runs (errors" in l:
            sp["head"] = l.strip()
    st["selfplay"] = sp; st["stages_dir"] = stages
    st["live_macro"] = [o for o in (partial_macro(l) for l in sec.get("last macro-trainer lines", []) if l.startswith("{")) if o]
    return st


# ---------------------------------------------------------------- supervisor.log, curve.txt, eval logs
EVAL_RE = re.compile(r"== (\S+): (\d+) runs \((\d+) errors\), won (\d+) = ([0-9.]+)% \(95% CI ([0-9.]+)-([0-9.]+)\), "
                     r"mean floor ([0-9.]+), act 1 cleared ([0-9.]+)%, act 2 cleared ([0-9.]+)%")


def parse_eval_line(text):
    m = EVAL_RE.search(text or "")
    if not m:
        return None
    n, err, won = int(m.group(2)), int(m.group(3)), int(m.group(4))
    ok = n - err; lo, hi = wilson(won, ok)
    return {"arm": m.group(1), "n": n, "errors": err, "won": won, "ok": ok, "pct": 100 * won / ok if ok else None,
            "lo": 100 * lo, "hi": 100 * hi, "ci_logged": [float(m.group(6)), float(m.group(7))],
            "floor": float(m.group(8)), "act1": float(m.group(9)), "act2": float(m.group(10))}


def parse_eval_log(text):
    """The four summary lines of summarize_runs.py (eval_N.log or eval/seg_NNN/summary.txt)."""
    e = parse_eval_line(text)
    if not e:
        return None
    m = re.search(r"^\s+deaths: (.*)$", text, re.M)
    if m:
        e["deaths"] = [[a.rsplit(" ", 1)[0], int(a.rsplit(" ", 1)[1])] for a in m.group(1).split(", ") if re.search(r" \d+$", a)]
    m = re.search(r"^\s+fights: (.*)$", text, re.M)
    if m:
        e["fights"] = {}
        for part in m.group(1).split("; "):
            mm = re.match(r"(\w+) (\d+) won ([0-9.]+)%(?: rel T_f ([0-9.]+))?(?: HP lost ([0-9.]+))?(?: potions ([0-9.]+))?", part)
            if mm:
                e["fights"][mm.group(1)] = {"n": int(mm.group(2)), "won": float(mm.group(3)),
                                            "rel": float(mm.group(4)) if mm.group(4) else None,
                                            "hp_lost": float(mm.group(5)) if mm.group(5) else None}
    m = re.search(r"bosses won (\d+)/(\d+)", text)
    if m:
        e["bosses_won"] = int(m.group(1)); e["bosses"] = int(m.group(2))
    return e


CURRIC_RE = re.compile(r"(\S+): self-play .*?A10 (\d+)/(\d+) = [0-9.]+%, A0-A4 (\d+)/(\d+) = [0-9.]+%, A5-A9 (\d+)/(\d+) = [0-9.]+%, "
                       r"all (\d+)/(\d+) = [0-9.]+%; errors (\d+), stalemates (\d+) \((.*?)\); (.*)$")


def parse_curric_line(text):
    """A curriculum reading (src/ic113/chain_stats.py curriculum: curriculum.log, or supervisor.log "hook: curriculum: ")."""
    m = CURRIC_RE.search(text or "")
    if not m:
        mm = re.search(r"chain start, stage (\d+)", text or "")
        return {"start": True, "to": int(mm.group(1))} if mm else None
    g = [int(m.group(i)) for i in range(2, 12)]
    o = {"dir": m.group(1), "a10": g[0:2], "a0_4": g[2:4], "a5_9": g[4:6], "all": g[6:8], "errors": g[8], "stalemates": g[9],
         "stale_text": m.group(12) if m.group(12) != "-" else "", "verdict": m.group(13).strip()}
    mm = re.match(r"STAGE (\d+) -> (\d+) \((.*)\)", o["verdict"])
    if mm:
        o["from"] = int(mm.group(1)); o["to"] = int(mm.group(2)); o["why"] = mm.group(3)
    else:
        mm = re.match(r"stage stays (\d+)", o["verdict"])
        if mm:
            o["from"] = o["to"] = int(mm.group(1))
    mm = re.search(r"_s(\d{3})$", o["dir"])
    o["seg"] = int(mm.group(1)) if mm else None
    return o


def _seg_no(text):
    """Segment number in "segment 12", "seg 12" or "<chain>_s012", else None."""
    m = re.search(r"(?:segment|seg) (\d+)\b|_s(\d{3})\b", text or "")
    return int(m.group(1) or m.group(2)) if m else None


def parse_supervisor(path):
    """supervisor.log of a chain. Besides segments, evaluations and problems it collects the interruptions of a chain on
    an interruptible server (2026-11-19, s3_supervise.sh / s3_resume.sh headers): "interruptions" = one entry per
    "no supervisor for X" restart (with the "downtime N min" line before it, the segment that was running and whether
    the restarted supervisor adopted it: "recovered" or "discarded") and per pause / stop by hand (until the chain was
    resumed); "end_moves" = every change of the scheduled end (downtime, resume, end_time set, supervisor start);
    "carry" = the off-host carry backups."""
    text = read_text(path)
    out = {"present": text is not None, "segments": {}, "problems": [], "starts": [], "end": None, "config": {}, "lines": 0,
           "pause": None, "events": [], "curric": [], "hook_segs": {}, "seg_settings": {}, "memguard": {}, "eval_reports": {},
           "interruptions": [], "end_moves": [], "carry": []}
    if text is None:
        return out
    segs = out["segments"]; last_ts = None; open_seg = None
    pend_down = None          # the last "downtime" line: s3_resume.sh writes it just before its "no supervisor" line
    cur_end = [None]          # scheduled end as last logged
    open_pause = None         # interruption entry of a pause / stop by hand, until the chain is resumed
    ints = out["interruptions"]

    def end_move(ts, to, why):
        if to is not None and to != cur_end[0]:
            if cur_end[0] is not None:
                out["end_moves"].append({"ts": ts, "from": cur_end[0], "to": to, "why": why})
            cur_end[0] = to

    def last_restart(ts, within=900):
        """The restart this adoption / evaluation / carry line belongs to (logged by the supervisor it started)."""
        for e in reversed(ints):
            if e["kind"] == "restart" and ts - e["ts"] <= within:
                return e
            if e["kind"] == "restart":
                return None
        return None

    def close_pause(ts, how):
        nonlocal open_pause
        if open_pause is not None:
            open_pause["until"] = ts; open_pause["resumed_by"] = how; open_pause = None

    for raw in text.splitlines():
        out["lines"] += 1
        m = re.match(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] (.*)$", raw)
        if not m:
            if raw.strip() and out["problems"] and out["problems"][-1].get("detail") is not None and len(out["problems"][-1]["detail"]) < 12:
                out["problems"][-1]["detail"].append(raw.strip()[:300])
            elif raw.strip():
                out["problems"].append({"ts": last_ts, "kind": "note", "seg": open_seg, "text": raw.strip()[:300], "detail": []})
            continue
        ts = utc_ts(m.group(1)); msg = m.group(2); last_ts = ts
        mm = re.match(r"supervisor start: (\S+), ends ([0-9-]+ [0-9:]+) UTC, actors (\d+), segment (\d+) s, ascension (\d+) \(share ([0-9.]+)\), conc (\d+)", msg)
        if mm:
            if open_seg is not None and segs[open_seg]["end"] is None:      # the previous supervisor died in this segment
                segs[open_seg].update(end=ts, result="interrupted", why="supervisor restarted")
                # (no s3_resume.sh line before it: started by hand, e.g. resume_chain.sh after a restore on another host)
                ints.append({"kind": "restart", "ts": ts, "since": None, "downtime_min": None, "seg": open_seg, "name": segs[open_seg]["name"],
                             "outcome": None, "runs": None, "why": None, "evals_rerun": [], "carry_reupload": None, "end_to": None,
                             "text": msg[:300], "how": "supervisor start"})
            out["starts"].append(ts)
            out.pop("end", None)     # a supervisor started after an end line continues the run (extend_run.sh)
            out["config"] = {"ends": utc_ts(mm.group(2)), "actors": int(mm.group(3)), "segment_s": int(mm.group(4)),
                             "ascension": int(mm.group(5)), "asc_share": float(mm.group(6)), "conc": int(mm.group(7))}
            end_move(ts, out["config"]["ends"], "supervisor start")
            close_pause(ts, "supervisor start")
            if (out.get("pause") or {}).get("state") in ("paused", "stopped"):
                out["pause"] = {"state": None, "ts": ts, "text": msg}
            continue
        mm = re.match(r"segment (\d+) \((\S+)\): (-?\d+) s, from (.*), runs so far (\d+)", msg)
        if mm:
            k = int(mm.group(1))
            if open_seg is not None and open_seg != k and segs[open_seg]["end"] is None:
                segs[open_seg].update(end=ts, result="interrupted", why="no end line in the log")
            segs[k] = {"seg": k, "name": mm.group(2), "start": ts, "planned": int(mm.group(3)), "from": mm.group(4),
                       "runs_at_start": int(mm.group(5)), "end": None, "result": None, "why": None, "runs_logged": None,
                       "eval": None, "eval_ts": None, "last_macro": None}
            open_seg = k; continue
        mm = re.match(r"downtime (\d+) min \(last heartbeat ([0-9-]+ [0-9:]+) UTC\): end moved to ([0-9-]+ [0-9:]+) UTC", msg)
        if mm:          # s3_resume.sh: the chain had no supervisor; the time since the last heartbeat does not count
            out["adoptive"] = True
            pend_down = {"ts": ts, "min": int(mm.group(1)), "hb": utc_ts(mm.group(2)), "to": utc_ts(mm.group(3))}
            end_move(ts, pend_down["to"], "downtime %s min" % mm.group(1))
            out["events"].append({"ts": ts, "kind": "downtime", "seg": open_seg, "text": msg[:300]})
            out["problems"].append({"ts": ts, "kind": "downtime", "seg": open_seg, "text": msg, "detail": None}); continue
        mm = re.match(r"no supervisor for (\S+)(?::| and its pause flag is set:) (.*)", msg)
        if mm:
            down = pend_down if pend_down and ts - pend_down["ts"] <= 300 else None; pend_down = None
            paused = "pause flag is set" in msg
            iseg = None
            if open_seg is not None and segs[open_seg]["end"] is None:
                # the segment ran until the host stopped: the last heartbeat (a minute apart), else this line (cron, every minute)
                cut = max(down["hb"], segs[open_seg]["start"] or 0) if down and down.get("hb") else ts
                segs[open_seg].update(end=cut, result="interrupted", why="supervisor was gone; resumed by s3_resume.sh")
                iseg = open_seg
            ints.append({"kind": "restart", "ts": ts, "since": down["hb"] if down else None, "downtime_min": down["min"] if down else None,
                         "seg": iseg, "name": segs[iseg]["name"] if iseg is not None else None, "outcome": None, "runs": None, "why": None,
                         "evals_rerun": [], "carry_reupload": None, "end_to": down["to"] if down else None, "text": msg[:300]})
            if paused:      # s3_resume.sh completed a pause flag the dead supervisor had not reached
                ints[-1]["kind"] = "pause"; open_pause = ints[-1]; open_pause["until"] = None
                out["pause"] = {"state": "paused", "ts": ts, "text": msg}
            out["problems"].append({"ts": ts, "kind": "pause" if paused else "restart", "seg": open_seg, "text": msg, "detail": None}); continue
        mm = re.match(r"segment (\d+) recovered after interruption: (\d+) runs,? ?([^;{]*?);? ?(\{.*)?$", msg)
        if mm:          # the restarted supervisor adopted the interrupted segment: its weights are last_good, it is evaluated
            k = int(mm.group(1)); s = segs.setdefault(k, {"seg": k})
            out["adoptive"] = True
            s.update(result="recovered", why=mm.group(3).strip() or None, runs_logged=int(mm.group(2)), recovered_at=ts,
                     last_macro=partial_macro(mm.group(4)) if mm.group(4) else s.get("last_macro"))
            if s.get("end") is None:
                s["end"] = ts
            e = last_restart(ts)
            if e is not None:
                e.update(seg=k, name=s.get("name") or e.get("name"), outcome="recovered", runs=int(mm.group(2)))
            out["problems"].append({"ts": ts, "kind": "recovered", "seg": k, "text": msg[:300], "detail": None}); continue
        mm = re.match(r"segment (\d+) \((\S+)\) interrupted and NOT adopted: (.*?)(?:; the chain continues from (.*))?$", msg)
        if mm:          # not adopted: the chain goes on from the previous last_good; the segment's training is lost
            k = int(mm.group(1)); s = segs.setdefault(k, {"seg": k, "name": mm.group(2)})
            out["adoptive"] = True
            s.update(result="discarded", why=mm.group(3), discarded_at=ts)
            if s.get("end") is None:
                s["end"] = ts
            e = last_restart(ts)
            if e is not None:
                e.update(seg=k, name=mm.group(2), outcome="discarded", why=mm.group(3))
            out["problems"].append({"ts": ts, "kind": "discarded", "seg": k, "text": msg[:300], "detail": None}); continue
        mm = re.match(r"segment (\d+) \((\S+)\) has live processes: not adopted", msg)
        if mm:
            out["problems"].append({"ts": ts, "kind": "note", "seg": int(mm.group(1)), "text": msg[:300], "detail": None}); continue
        mm = re.match(r"evaluation of segment\(s\) ?([\d ]+) missing or incomplete", msg)
        if mm:          # a restart found an evaluation that the interruption cut off: run again
            ks = [int(x) for x in mm.group(1).split()]
            e = last_restart(ts)
            if e is not None:
                e["evals_rerun"] = sorted(set(e["evals_rerun"]) | set(ks))
            out["events"].append({"ts": ts, "kind": "eval_rerun", "seg": ks[0] if ks else open_seg, "text": msg[:300]}); continue
        if msg.startswith("carry backup"):       # off-host copy of the replay / optimizer carries (CARRY_BACKUP_EVERY)
            body = re.sub(r"^carry backup: (?:\[[^\]]*\] )?", "", msg)
            k = _seg_no(body); ok = body.startswith("ok ")
            if msg.startswith("carry backup: "):
                out["carry"].append({"ts": ts, "seg": k, "ok": ok, "text": body[:300]})
            mm = re.match(r"carry backup of segment (\d+) not on Nextcloud", msg)
            if mm:
                e = last_restart(ts)
                if e is not None:
                    e["carry_reupload"] = int(mm.group(1))
            out["events"].append({"ts": ts, "kind": "carry", "seg": k, "text": msg[:300]})
            if msg.startswith("carry backup: ") and not ok:          # the upload failed (carry_backup.sh "FAILED ...")
                out["problems"].append({"ts": ts, "kind": "carry", "seg": k, "text": msg[:300], "detail": None})
            elif " skipped: " in msg:                                     # carry files missing, previous upload still running
                out["problems"].append({"ts": ts, "kind": "note", "seg": k, "text": msg[:300], "detail": None})
            continue
        mm = re.match(r"restore (\S+): (.*)", msg)
        if mm:          # restore_from_nextcloud.sh (its log lines, if they were added to supervisor.log)
            out["events"].append({"ts": ts, "kind": "restore", "seg": None, "text": msg[:300]})
            if re.search(r"FAILED|nothing done|not answering|refus", msg):
                out["problems"].append({"ts": ts, "kind": "note", "seg": None, "text": msg[:300], "detail": None})
            continue
        mm = re.match(r"hook: (.*)", msg)          # chain hook (s3ic): curriculum, Vbar job, segment settings, eval report
        if mm:
            h = mm.group(1)
            sm = re.search(r"(?:segment|seg) (\d+)\b|_s(\d{3})\b", h)
            hk = int(sm.group(1) or sm.group(2)) if sm else (open_seg + 1 if open_seg is not None else 0)
            out["events"].append({"ts": ts, "kind": "hook", "seg": hk, "text": h[:400]})
            if h.startswith("curriculum: "):
                c = parse_curric_line(h[12:])
                if c:
                    c["ts"] = ts; out["curric"].append(c)
            hm = re.match(r"segment (\d+): stage (\d+) \(asc_mix ([0-9.]+), asc_weights (.*?)\), potion_mix (\S+)", h)
            if hm:
                out["hook_segs"][int(hm.group(1))] = {"ts": ts, "stage": int(hm.group(2)), "asc_mix": float(hm.group(3)),
                                                     "asc_weights": hm.group(4), "vmix": hm.group(5)}
            hm = re.match(r"eval report seg (\d+): (.*)", h)
            if hm:
                out["eval_reports"][int(hm.group(1))] = hm.group(2)[:300]
            if re.search(r"FAILED|Traceback|Exception|\bERROR\b", h):        # a failed Vbar job, a hook that crashed
                out["problems"].append({"ts": ts, "kind": "note", "seg": open_seg, "text": msg[:300], "detail": None})
            continue
        mm = re.match(r"segment (\d+) settings from segment\.env \(written for segment (\d+)\): ASC_MIX (\S+), actor args '(.*)'", msg)
        if mm:
            out["seg_settings"][int(mm.group(1))] = {"ts": ts, "for_seg": int(mm.group(2)), "asc_mix": mm.group(3), "actor_args": mm.group(4)[:300]}
            continue
        mm = re.match(r"segment (\d+): memory guard killed actor (\d+) \((\d+) GB > (\d+) GB\)", msg)
        if mm:
            k = int(mm.group(1)); out["memguard"][k] = out["memguard"].get(k, 0) + 1
            out["problems"].append({"ts": ts, "kind": "memguard", "seg": k, "text": msg, "detail": None}); continue
        mm = re.match(r"(not )?pruned (\S+)", msg)
        if mm:
            sm = re.search(r"_s(\d{3})$", mm.group(2)); k = int(sm.group(1)) if sm else open_seg
            out["events"].append({"ts": ts, "kind": "prune", "seg": k, "text": msg[:300]})
            if mm.group(1):
                out["problems"].append({"ts": ts, "kind": "note", "seg": k, "text": msg[:300], "detail": None})
            continue
        if msg.startswith("combat network switched"):
            out["events"].append({"ts": ts, "kind": "switch", "seg": open_seg, "text": msg[:300]})
            out["problems"].append({"ts": ts, "kind": "note", "seg": open_seg, "text": msg[:300], "detail": None}); continue
        mm = re.match(r"segment (\d+) stopped by the guard: (.*)", msg)
        if mm:
            k = int(mm.group(1)); segs.setdefault(k, {"seg": k})["guard"] = mm.group(2)
            out["problems"].append({"ts": ts, "kind": "guard", "seg": k, "text": msg, "detail": None}); continue
        mm = re.match(r"segment (\d+) ok: (\d+) runs(?: \(stopped early: (.*?)\))?; ?(.*)", msg)
        if mm:
            k = int(mm.group(1)); s = segs.setdefault(k, {"seg": k})
            s.update(end=ts, result="guard" if mm.group(3) else "ok", why=mm.group(3), runs_logged=int(mm.group(2)),
                     last_macro=partial_macro(mm.group(4)))
            continue
        mm = re.match(r"segment (\d+) FAILED \((\d+) runs(?:, (.*?))?\); weights not advanced; consecutive failures (\d+)", msg)
        if mm:
            k = int(mm.group(1)); s = segs.setdefault(k, {"seg": k})
            s.update(end=ts, result="failed", why=mm.group(3), runs_logged=int(mm.group(2)))
            out["problems"].append({"ts": ts, "kind": "failed", "seg": k, "text": msg, "detail": []}); continue
        mm = re.match(r"eval: seg (\d+) runs (\d+) \| ?(.*)", msg)
        if mm:
            k = int(mm.group(1)); s = segs.setdefault(k, {"seg": k})
            s["eval"] = parse_eval_line(mm.group(3)); s["eval_ts"] = ts; s["eval_runs"] = int(mm.group(2)); s["eval_text"] = mm.group(3)[:260]
            if s["eval"] is None:
                out["problems"].append({"ts": ts, "kind": "eval", "seg": k, "text": "evaluation of segment %d produced no summary line" % k, "detail": None})
            else:        # evaluated again (after an interruption): the earlier failed attempt is no longer a problem
                out["problems"] = [p for p in out["problems"] if not (p["kind"] == "eval" and p["seg"] == k)]
            continue
        mm = re.match(r"supervisor end: (.*?); total runs (\d+); last good weights (.*)", msg)
        if mm:
            out["end"] = {"ts": ts, "final": mm.group(1), "total_runs": int(mm.group(2)), "weights": mm.group(3)}
            if not mm.group(1).startswith("FINISHED"):
                out["problems"].append({"ts": ts, "kind": "stopped", "seg": None, "text": msg, "detail": None})
            continue
        mm = re.match(r"(pause flag set|pause flag removed|chain paused|chain resumed|chain stopped|end_time set|WARNING)\b|.*stopped by hand", msg)
        if mm:          # pauses between segments (owner's tests on the server), resumes, stops for a move, end-time changes
            kind = "pause" if mm.group(1) in ("pause flag set", "pause flag removed", "chain paused", "chain resumed", "chain stopped") else "note"
            out["problems"].append({"ts": ts, "kind": kind, "seg": open_seg, "text": msg, "detail": None})
            if kind == "pause":
                st = {"pause flag set": "pending", "chain paused": "paused", "chain stopped": "stopped"}.get(mm.group(1))
                out["pause"] = {"state": st, "ts": ts, "text": msg}
                if st in ("paused", "stopped"):
                    iseg = None
                    if st == "stopped" and open_seg is not None and segs[open_seg]["end"] is None:     # stopped mid-segment
                        segs[open_seg].update(end=ts, result="interrupted", why="chain stopped by hand")
                        iseg = open_seg
                    close_pause(ts, "paused again")
                    open_pause = {"kind": "pause", "ts": ts, "until": None, "seg": iseg, "name": segs[iseg]["name"] if iseg is not None else None,
                                  "outcome": None, "runs": None, "why": None, "evals_rerun": [], "carry_reupload": None, "text": msg[:300]}
                    ints.append(open_pause)
                if mm.group(1) == "chain resumed":
                    close_pause(ts, "chain resumed")
            mm2 = re.search(r"(?:chain resumed \(ends|end_time set to) ([0-9-]+ [0-9:]+) UTC", msg)
            if mm2:
                end_move(ts, utc_ts(mm2.group(1)), "chain resumed" if msg.startswith("chain resumed") else "end_time set")
            continue
        if re.search(r"evaluation of segment \d+ over|three failed segments|dropped \d+ unuploaded|stopping the instance|disk \d+%", msg):
            mm = re.search(r"segment (\d+)", msg)
            out["problems"].append({"ts": ts, "kind": "note", "seg": int(mm.group(1)) if mm else open_seg, "text": msg, "detail": None})
    return out


def parse_curve(path):
    text = read_text(path)
    rows = []
    if text is None:
        return None
    for l in text.splitlines():
        m = re.match(r"seg (\d+) runs (\d+) \| ?(.*)", l)
        if m:
            rows.append({"seg": int(m.group(1)), "runs": int(m.group(2)), "eval": parse_eval_line(m.group(3)), "text": m.group(3)[:260]})
    # an evaluation cut off by an interruption and run again (or run twice) can leave two lines for one segment: the last counts
    last = {r["seg"]: i for i, r in enumerate(rows)}
    return [r for i, r in enumerate(rows) if last[r["seg"]] == i]


# ---------------------------------------------------------------- per-file aggregates (cached)
def new_group():
    return {"n": 0, "won": 0, "a1": 0, "a2": 0, "floors": 0, "deaths": {}, "killed": {}, "fights": 0, "fights_won": 0,
            "elites": 0, "bosses": 0, "bosses_won": 0, "sims": 0, "seconds": 0.0, "rests": 0, "smiths": 0,
            "lost": {"hallway": 0, "elite": 0, "boss": 0, "none": 0}}


STALE_RE = re.compile(r"stalemate: (\S+?)[,;:]?(?:\s|$)")


def agg_runs(path, target=10):
    """One run-record file. Groups: "t" = target ascension, "l" = lower levels; error and perturbed runs are counted
    separately and left out of the groups (as in curve_runs.py and diag_runs.py). Also: error runs without perturbed
    starts per ascension ("asc_err": the curriculum counts them as lost), combat stalemates by encounter ("stale") and
    the run records' look counters summed ("look": potion_mix, fake_merchant, ...)."""
    a = {"lines": 0, "bad": 0, "errors": {}, "error_n": 0, "perturbed": 0, "g": {"t": new_group(), "l": new_group()}, "asc": {},
         "versions": [None, None], "asc_err": {}, "stale": {}, "look": {}, "look_runs": 0}
    for r in jsonl(path):
        a["lines"] += 1
        if r.get("error"):
            a["error_n"] += 1
            key = re.sub(r"\d+", "#", str(r["error"]))[:160]
            e = a["errors"].setdefault(key, {"n": 0, "example": str(r["error"])[:300], "seed": r.get("seed"), "floor": r.get("floors")})
            e["n"] += 1
            m = STALE_RE.search(str(r["error"]))
            if m:
                k = short_enc(m.group(1)); a["stale"][k] = a["stale"].get(k, 0) + 1
            if not r.get("perturbed"):
                try:
                    k = str(int(r["ascension"])); a["asc_err"][k] = a["asc_err"].get(k, 0) + 1
                except (KeyError, TypeError, ValueError):
                    pass
            continue
        if r.get("perturbed"):
            a["perturbed"] += 1; continue
        try:          # every field is converted before anything is counted: a record with odd types is skipped whole
            asc = int(r["ascension"]); won = bool(r["won"]); act = int(r.get("act") or 0); floors = int(r.get("floors") or 0)
            extra = {f: int(r.get(f) or 0) for f in ("fights", "fights_won", "elites", "bosses", "bosses_won", "rests", "smiths")}
            sims = int(r.get("simulations") or 0); secs = float(r.get("seconds") or 0)
            killed = r.get("killed_by"); killed = killed if isinstance(killed, str) else None
        except (KeyError, TypeError, ValueError):
            a["bad"] += 1; continue
        g = a["g"]["t" if asc == target else "l"]
        g["n"] += 1; g["won"] += won; g["a1"] += (act >= 1 or won); g["a2"] += (act >= 2 or won); g["floors"] += floors
        k = room_kind(killed)
        if not won:
            dk = "%d|%s" % (max(0, min(act, 2)) + 1, k); g["deaths"][dk] = g["deaths"].get(dk, 0) + 1
            g["lost"][k] += 1
            if k == "boss":
                e = short_enc(killed); g["killed"][e] = g["killed"].get(e, 0) + 1
        for f, v in extra.items():
            g[f] += v
        g["sims"] += sims; g["seconds"] += secs
        x = a["asc"].setdefault(str(asc), [0, 0, 0, 0, 0])
        x[0] += 1; x[1] += won; x[2] += (act >= 1 or won); x[3] += (act >= 2 or won); x[4] += floors
        lk = r.get("look")
        if isinstance(lk, dict):
            for k, v in lk.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    a["look"][k] = a["look"].get(k, 0) + v
        a["look_runs"] += 1
        mv = r.get("macro_version")
        if isinstance(mv, int):
            v = a["versions"]; v[0] = mv if v[0] is None else min(v[0], mv); v[1] = mv if v[1] is None else max(v[1], mv)
    a["bad"] += jsonl.bad
    return a


def agg_fights(fpath, rpath, target=10):
    """One fight-record file joined to the same actor's run records by run seed (fights of unfinished, errored and
    perturbed runs are skipped, as in diag_runs.py)."""
    group = {}
    for r in jsonl(rpath):
        if r.get("error") or r.get("perturbed") or "seed" not in r:
            continue
        try:
            group[r["seed"]] = "t" if r.get("ascension") == target else "l"
        except TypeError:          # an unhashable seed
            pass
    a = {"lines": 0, "bad": 0, "skipped": 0, "g": {}}
    for g in ("t", "l"):
        a["g"][g] = {"room": {"monster": [0, 0], "elite": [0, 0], "boss": [0, 0]}, "enc": {}}
    for x in jsonl(fpath):
        a["lines"] += 1
        try:
            g = group.get(x.get("run_seed"))
        except TypeError:
            g = None
        if g is None or x.get("error"):
            a["skipped"] += 1; continue
        k = str(x.get("room_type")); won = x.get("result") == "won"
        G = a["g"][g]
        rm = G["room"].setdefault(k, [0, 0]); rm[0] += 1; rm[1] += won
        if k not in ("boss", "elite"):
            continue
        try:
            hpf = x["start_hp"] / max(1, x["max_hp"])
            e = G["enc"].setdefault(short_enc(x.get("encounter")), [k, 0, 0, 0.0, 0, 0, 0, 0.0])
            # kind, fights, won, start-HP share sum, potions at start, turns, floor sum, HP share lost (won fights)
            e[1] += 1; e[2] += won; e[3] += hpf; e[4] += int(x.get("potions_start") or 0); e[5] += int(x.get("turns") or 0)
            e[6] += int(x.get("floor") or 0)
            if won:
                e[7] += (x["start_hp"] - x["end_hp"]) / max(1, x["max_hp"])
        except (KeyError, TypeError):
            a["bad"] += 1
    a["bad"] += jsonl.bad
    for g in a["g"].values():
        for e in g["enc"].values():
            e[3] = round(e[3], 3); e[7] = round(e[7], 3)
    return a


ACTOR_RE = re.compile(r"\[(a\d+)\] (\d+)s runs (\d+) \(won (\d+), errors (\d+), mean floors ([0-9.]+)\) = (\d+)/h, fights (\d+) \(won (\d+)\), "
                      r"combat decisions (\d+), sims (\d+) = (\d+)/s")


PM_RE = re.compile(r"potion_mix decisions (\d+)\+(\d+) swaps (\d+)\+(\d+)")


def agg_actor_log(path):
    """Progress lines of one actor, thinned to the last line of every BIN seconds."""
    a = {"bins": {}, "last_t": None, "final": None, "stalls": [], "run_errors": [], "notes": None, "lines": 0, "pm": None, "pm_end": {}}
    text = read_text(path) or ""
    for l in text.splitlines():
        m = ACTOR_RE.match(l)
        if m:
            t = int(m.group(2)); w = re.search(r"bridge wait (\d+)%", l)
            row = [t, int(m.group(3)), int(m.group(4)), int(m.group(5)), int(m.group(11)), int(w.group(1)) if w else None,
                   int(m.group(8)), int(m.group(9))]            # t, runs, won, errors, sims, bridge wait %, fights, fights won
            a["bins"][str(t // BIN)] = row; a["last_t"] = t; a["final"] = row; a["lines"] += 1
            pm = PM_RE.search(l)          # s3ic: potion_mix decisions reward+extra, swaps reward+extra (cumulative)
            if pm:
                a["pm"] = [int(pm.group(i)) for i in range(1, 5)]
        elif "] potion_mix reward: " in l or "] potion_mix extra: " in l:
            src = "reward" if "] potion_mix reward: " in l else "extra"; o = {}
            for k, pat in (("decisions", r"decisions (\d+)"), ("swaps", r"swaps (\d+)"), ("kinds", r"kinds scored (\d+)"),
                           ("nomean", r"without Vbar (\d+)"), ("rows", r"macro rows (\d+)"), ("dev", r"mean \|dV - Vbar\| ([0-9.]+)"),
                           ("dev_n", r"over (\d+)")):
                mm = re.search(pat, l)
                if mm:
                    o[k] = float(mm.group(1)) if k == "dev" else int(mm.group(1))
            a["pm_end"][src] = o
        elif "no progress for" in l:
            a["stalls"].append(l.strip()[:300])
        elif " error at floor " in l and len(a["run_errors"]) < 5:
            a["run_errors"].append(l.strip()[:300])
        elif "] notes: " in l:
            a["notes"] = l.split("] notes: ", 1)[1].strip()[:400] or None
        elif re.search(r"Unhandled exception|Traceback|FATAL", l) and len(a["stalls"]) < 5:
            a["stalls"].append(l.strip()[:300])
    return a


SRV_RE = re.compile(r"\[s3srv\] (\d+)s combat (\d+) states in (\d+) batches \(mean (\d+)\), (\d+)/s, busy (\d+)%; macro (\d+) rows in (\d+) batches "
                    r"\(mean (\d+)\), busy (\d+)%")


def agg_server_log(path):
    a = {"bins": {}, "problems": [], "last_t": None}
    for l in (read_text(path) or "").splitlines():
        m = SRV_RE.match(l)
        if m:
            t = int(m.group(1))
            a["bins"][str(t // BIN)] = [t, int(m.group(2)), int(m.group(6)), int(m.group(7)), int(m.group(10)), int(m.group(4)), int(m.group(9))]
            a["last_t"] = t        # t, combat states, combat busy %, macro rows, macro busy %, mean combat batch, mean macro batch
        elif re.search(r"Traceback|Error|reload failed|exiting", l) and "UserWarning" not in l and len(a["problems"]) < 5:
            a["problems"].append(l.strip()[:300])
    return a


CT_FIELDS = ("t", "step", "ingested", "policy", "value", "hp", "pol_acc", "kl", "hp_mae", "grad_norm", "keep", "survive", "version")


def agg_ctrain(path):
    """Combat-trainer records (train_log.jsonl, or logs/ctrain.txt where the JSON follows '[train] '), averaged in groups
    so that a segment gives at most about 120 points."""
    recs = []; bad = 0
    if path.endswith(".jsonl"):
        recs = list(jsonl(path)); bad = jsonl.bad
    else:
        for l in (read_text(path) or "").splitlines():
            if l.startswith("[train] {"):
                try:
                    recs.append(json.loads(l[8:]))
                except ValueError:
                    bad += 1
    recs = [r for r in recs if isinstance(r.get("t"), (int, float))]
    k = max(1, math.ceil(len(recs) / 120)); out = []
    for i in range(0, len(recs), k):
        grp = recs[i:i + k]; o = {}
        for f in CT_FIELDS:
            v = [r[f] for r in grp if isinstance(r.get(f), (int, float))]
            if v:
                o[f] = grp[-1][f] if f in ("t", "step", "ingested", "version") and isinstance(grp[-1].get(f), (int, float)) else round(sum(v) / len(v), 5)
        out.append(o)
    return {"rows": out, "n": len(recs), "bad": bad, "per_point": k}


MACRO_KEEP = ("t", "iter", "runs", "new_runs", "win_rate", "mean_floors", "ascension", "acts_cleared_2plus", "act1_cleared", "alpha",
              "mean_outcome", "value_mean_floor_rows", "outcome_mean_floor_rows", "value_logloss", "value_loss", "behaviour_lag_kl",
              "policy_steps", "kl_weight", "ent", "kl_bc", "kl_step", "clip", "adv_std", "early_stop", "decisions", "version",
              "target_window_win_rate", "window_win_rate", "value_steps", "value_logloss_table", "value_logloss_const", "value_explained",
              "value_logloss_a10", "value_logloss_table_a10", "value_logloss_a0_9", "value_logloss_table_a0_9")


def agg_macro(path):
    rows = []
    for r in jsonl(path):
        if not isinstance(r.get("runs"), (int, float)) or not isinstance(r.get("t"), (int, float)):
            continue
        o = {k: r[k] for k in MACRO_KEEP if k in r and isinstance(r[k], (int, float, bool, str))}
        by = r.get("by_ascension")
        if isinstance(by, dict):
            o["by"] = {a: [v.get("runs"), v.get("won"), v.get("acts_cleared"), v.get("floors")] for a, v in by.items() if isinstance(v, dict)}
        cal = r.get("calibration")      # value calibration per ascension group (A0-8, A9, A10) and act: rows, outcome, ens, logit sd
        if isinstance(cal, dict):
            o["cal"] = {c: [v.get("rows"), v.get("outcome"), v.get("ens"), v.get("logit_sd")] for c, v in cal.items() if isinstance(v, dict)}
        rows.append(o)
    return {"rows": rows, "bad": jsonl.bad}


def agg_eval_runs(path, target=10):
    a = agg_runs(path, target)
    g = a["g"]["t"]; l = a["g"]["l"]
    for k in ("n", "won", "a1", "a2", "floors"):
        g[k] += l[k]
    for k, v in l["deaths"].items():
        g["deaths"][k] = g["deaths"].get(k, 0) + v
    return {"n": g["n"], "won": g["won"], "a1": g["a1"], "a2": g["a2"], "floors": g["floors"], "deaths": g["deaths"],
            "errors": a["error_n"], "bad": a["bad"], "stale": a["stale"], "look": a["look"], "look_runs": a["look_runs"]}


# ---------------------------------------------------------------- s3ic: chain info, evaluation report, curriculum, Vbar
ERA_GAME = {6: "v0.113"}          # actor option era=N -> game version of the era row (src/ic113/PORT.md)
# src/ic113/chain_stats.py STAGES (PLAN step 7)
CURRIC_STAGES = {1: "20% of runs at A10, the rest A0-A9 uniform", 2: "30% at A10, the rest A5-A9 uniform",
                 3: "A8 32%, A9 41%, A10 27% (Defect's final mix)"}


def chain_info(out):
    """Character, game version and evaluation set-up of a chain, from config.env / extra.env / segment.env."""
    env = (out.get("config") or {}).get("env") or {}; ex = out.get("extra_env") or {}; se = out.get("segment_env") or {}
    args = " ".join([env.get("EVAL_BASE_ARGS", ""), ex.get("EXTRA_ACTOR_ARGS", "")])
    ch = re.search(r"\bcharacter=(\w+)", args); era = re.search(r"\bera=(\d+)", args)
    era = int(era.group(1)) if era else None
    al = re.search(r"--alpha ([0-9.]+)", ex.get("EXTRA_MTRAIN_ARGS", ""))
    info = {"character": ch.group(1).capitalize() if ch else "Defect", "era": era,
            "game": ERA_GAME.get(era, "era %d" % era) if era is not None else "v0.111",
            "remote": out.get("remote"), "alpha_fixed": float(al.group(1)) if al else None,
            "eval_args": se.get("EVAL_ARGS") or env.get("EVAL_BASE_ARGS") or None,
            "kl_runs": env.get("KL_RUNS"), "runs_per_iter": env.get("RUNS_PER_ITER")}
    try:
        n = int(env.get("EVAL_COUNT") or 0); first = int(env.get("EVAL_FIRST") or 0); pre = env.get("EVAL_SEEDS")
        if n and pre:
            info["eval_seeds"] = "%s%06d-%s%06d" % (pre, first, pre, first + n - 1)
    except ValueError:
        pass
    return info


def split_eval_report(text):
    """eval_report.txt (s3ic hook): {segment: report text} from its "== segment N (...)" headers."""
    out = {}; cur = None
    for l in (text or "").splitlines():
        m = re.match(r"== segment (\d+) ", l)
        if m:
            cur = int(m.group(1)); out[cur] = ""
        elif cur is not None:
            out[cur] += l + "\n"
    return out


def parse_eval_report(text):
    """src/ic113/chain_stats.py eval output: head line, act funnel, HP share entering bosses, stalemates, potion_mix."""
    o = {"head": (text.strip().splitlines() or [""])[0][:300]}
    m = re.search(r"^funnel[^:]*: (.*)$", text, re.M)
    if m:
        o["funnel"] = [[a, int(b), int(c)] for a, b, c in re.findall(r"(act-\d boss(?: \d)?) (\d+) \([0-9]+%\) won (\d+)", m.group(1))]
    m = re.search(r"^HP share entering: (.*)$", text, re.M)
    if m:
        o["hp"] = [[a, float(b), float(c)] for a, b, c in re.findall(r"(act-\d boss(?: \d)?) ([0-9.]+) \(potions ([0-9.]+)\)", m.group(1))]
    m = re.search(r"^stalemates (\d+): (.*?)(?:; other errors: (.*))?$", text, re.M)
    if m:
        o["stale_n"] = int(m.group(1))
        o["stale"] = {a: int(b) for a, b in re.findall(r"(\S+) (\d+)(?:,|$)", m.group(2))} if m.group(2) != "-" else {}
        o["other_errors"] = (m.group(3) or "")[:300]
    m = re.search(r"^deaths: (.*)$", text, re.M)
    if m:
        o["deaths"] = m.group(1)[:300]
    m = re.search(r"^potion_mix per run: (.*)$", text, re.M)
    if m:
        o["potion_mix"] = {a: float(b) for a, b in re.findall(r"(\w+) ([0-9.]+)", m.group(1))}
    return o


def curric_counts(runs):
    """Curriculum groups of one segment's run records, counted as chain_stats.py does: runs with perturbed starts left
    out, error runs counted as lost. {group: [won, runs]}."""
    g = {"a10": [0, 0], "a0_4": [0, 0], "a5_9": [0, 0], "all": [0, 0]}
    for a, v in (runs.get("asc") or {}).items():
        a = int(a); n = v[0] + (runs.get("asc_err") or {}).get(str(a), 0)
        for k in ("all", "a10" if a >= 10 else "a0_4" if a <= 4 else "a5_9"):
            g[k][0] += v[1]; g[k][1] += n
    for a, e in (runs.get("asc_err") or {}).items():      # error runs at a level without a finished run
        if str(a) not in (runs.get("asc") or {}):
            a = int(a)
            for k in ("all", "a10" if a >= 10 else "a0_4" if a <= 4 else "a5_9"):
                g[k][1] += e
    return g


def build_curriculum(run_dir, env, sup, seg_out, status, macro_rows):
    """s3ic ascension curriculum: stage now, stage per segment, readings of the hook, measured rates vs thresholds."""
    stage_file = (read_text(os.path.join(run_dir, "curriculum_stage")) or "").strip()
    hook_status = (read_text(os.path.join(run_dir, "hook_status.txt")) or "").strip()
    clog = read_text(os.path.join(run_dir, "curriculum.log"))
    if not (stage_file or hook_status or sup.get("curric") or sup.get("hook_segs")):
        return None
    th = {}
    for k, name, d in (("t1_a10", "CURRIC_T1_A10", 0.05), ("t1_low", "CURRIC_T1_LOW", 0.50), ("t2_a10", "CURRIC_T2_A10", 0.12), ("min_runs", "CURRIC_MIN_RUNS", 300)):
        try:
            th[k] = float(env.get(name, d))
        except ValueError:
            th[k] = d
    stage = None
    m = re.match(r"(\d+)", stage_file) or re.match(r"stage (\d+)", hook_status)
    if m:
        stage = int(m.group(1))
    readings = []
    if clog:
        for l in clog.splitlines():
            c = parse_curric_line(l)
            if c and not c.get("start"):
                mm = re.match(r"\[([0-9-]+ [0-9:]+)\]", l); c["ts"] = utc_ts(mm.group(1)) if mm else None; c["src"] = "curriculum.log"
                readings.append(c)
    have = {r["dir"] for r in readings}
    for c in sup.get("curric", []):
        if not c.get("start") and c["dir"] not in have:
            c = dict(c, src="supervisor.log"); readings.append(c)
    readings.sort(key=lambda r: (r.get("seg") if r.get("seg") is not None else -1, r.get("ts") or 0))
    # a restart whose segment was not adopted reads the same last good segment again: one reading per segment (the last)
    last = {r["dir"]: i for i, r in enumerate(readings)}
    readings = [r for i, r in enumerate(readings) if last[r["dir"]] == i]
    if stage is None:
        st = [r["to"] for r in readings if r.get("to")] + [h["stage"] for _, h in sorted(sup.get("hook_segs", {}).items())]
        stage = st[-1] if st else None
    per = []
    for so in seg_out:
        k = so["seg"]; hs = sup.get("hook_segs", {}).get(k) or {}; ss = sup.get("seg_settings", {}).get(k) or {}
        row = {"seg": k, "stage": hs.get("stage"), "asc_mix": hs.get("asc_mix") if hs else ss.get("asc_mix"), "asc_weights": hs.get("asc_weights"),
               "vmix": hs.get("vmix"), "result": so.get("result")}
        if so.get("runs"):
            row["measured"] = curric_counts(so["runs"])
        rd = [r for r in readings if r.get("seg") == k]
        if rd:
            row["reading"] = rd[-1]
        per.append(row)
    # the running segment: A10 from status.txt's per-directory block (errors and perturbed runs left out there), A0-A4
    # from the macro-trainer lines of the segment seen in status.txt (by_ascension, perturbed runs left out)
    live = None
    if status.get("present") and status.get("kind") == "running" and status.get("segment") is not None:
        k = status["segment"]; live = {"seg": k}
        for x in status.get("stages_dir", []):
            if x["dir"].endswith("_s%03d" % k) and x["asc"] == 10 and x.get("won") is not None:
                live["a10"] = [round(x["won"] * x["runs"] / 100), x["runs"]]
        by = {}
        for r in macro_rows:
            if r.get("seg") == k and isinstance(r.get("by"), dict):
                for a, v in r["by"].items():
                    if v and v[0] and v[1] is not None:
                        b = by.setdefault(a, [0.0, 0]); b[0] += v[0] * v[1]; b[1] += v[0]
        for key, lv in (("a0_4", range(0, 5)), ("a5_9", range(5, 10)), ("a10_macro", (10,))):
            w = sum(by.get(str(a), [0, 0])[0] for a in lv); n = sum(by.get(str(a), [0, 0])[1] for a in lv)
            if n:
                live[key] = [round(w), n]
        live["updates"] = len([r for r in macro_rows if r.get("seg") == k and r.get("by")])
    return {"stage": stage, "stage_file": stage_file, "hook_status": hook_status, "thresholds": th, "stages": CURRIC_STAGES,
            "readings": readings, "per_segment": per, "live": live}


def build_vmix(run_dir):
    """Vbar files of the potion_mix rule (s3ic vmix/vbar_sNNN.json): per segment and act, states, V(empty belt), Vbar per kind."""
    out = []
    for p in sorted(glob.glob(os.path.join(run_dir, "vmix", "vbar_s[0-9][0-9][0-9].json"))):
        try:
            d = json.load(open(p))
        except (OSError, ValueError):
            out.append({"seg": int(p[-8:-5]), "error": "unreadable"}); continue
        o = {"seg": int(p[-8:-5]), "made": d.get("made"), "source": d.get("source"), "phase2": "phase2" in d, "acts": {}}
        for act, kinds in sorted((d.get("acts") or {}).items()):
            if not isinstance(kinds, dict) or not kinds:
                continue
            vals = {k.split(".", 1)[-1]: round(float(v), 5) for k, v in kinds.items() if isinstance(v, (int, float))}
            xs = list(vals.values()); mu = sum(xs) / len(xs)
            o["acts"][act] = {"n": (d.get("n") or {}).get(act), "v_empty": (d.get("v_empty") or {}).get(act), "kinds": len(vals),
                              "mean": round(mu, 5), "sd": round(math.sqrt(sum((x - mu) ** 2 for x in xs) / len(xs)), 5), "values": vals}
        o["info"] = {k: v for k, v in (d.get("info") or {}).items() if isinstance(v, (int, float, str))}
        out.append(o)
    return out


# ---------------------------------------------------------------- the whole payload
def merge_group(dst, src):
    for k, v in src.items():
        if isinstance(v, dict):
            d = dst.setdefault(k, {})
            for kk, vv in v.items():
                d[kk] = d.get(kk, 0) + vv
        else:
            dst[k] = dst.get(k, 0) + v


def is_fresh(path):
    try:
        return time.time() - os.path.getmtime(path) < FRESH
    except OSError:
        return False


def actor_key(p):
    m = re.search(r"a(\d+)\.[a-z]+$", p)
    return int(m.group(1)) if m else 0


def history_path(run):
    """Live points seen in status.txt, one file per chain (s3a keeps the name it had before chains were added)."""
    return os.path.join(STATE, "live_history.jsonl" if run == "s3a" else "live_history_%s.jsonl" % re.sub(r"[^\w.-]", "_", run))


def load_history(run):
    rows = []
    try:
        with open(history_path(run)) as f:
            for l in f:
                try:
                    rows.append(json.loads(l))
                except ValueError:
                    pass
    except OSError:
        pass
    return rows


def add_history(st, run):
    """Remembers the macro-trainer lines seen in status.txt, so that the running segment (which has no meta on Nextcloud
    until it ends) gets its points on the self-play curves as syncs come in."""
    if not st.get("present") or not st.get("updated"):
        return load_history(run)
    rows = load_history(run)
    if any(r.get("updated") == st["updated"] for r in rows):
        return rows
    row = {"updated": st["updated"], "segment": st.get("segment"), "runs_so_far": st.get("runs_so_far"), "state": st.get("state"), "host": st.get("host"),
           "macro": [dict({k: o.get(k) for k in MACRO_HEAD if k in o}, **({"by": {a: [v.get("runs"), v.get("won"), v.get("acts_cleared"), v.get("floors")]
                                                                                for a, v in o["by_ascension"].items() if isinstance(v, dict)}}
                                                                          if isinstance(o.get("by_ascension"), dict) else {}))
                     for o in st.get("live_macro", [])]}
    try:
        os.makedirs(STATE, exist_ok=True)
        with open(history_path(run), "a") as f:
            f.write(json.dumps(row) + "\n")
    except OSError:
        pass
    rows.append(row)
    return rows


# ---------------------------------------------------------------- interruptions, restores, hosts
def restore_lines(run_dir, run):
    """Log lines of src/s3/restore_from_nextcloud.sh ("[ts] restore <chain>: ..."), if a restore log was put in the
    chain directory (restore*.log); supervisor.log does not get them. [{ts, kind: restore, seg: None, text}]"""
    out = []
    for p in sorted(glob.glob(os.path.join(run_dir, "restore*.log"))):
        for l in (read_text(p) or "").splitlines():
            m = re.match(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] (restore \S+: .*)$", l)
            if m:
                out.append({"ts": utc_ts(m.group(1)), "kind": "restore", "seg": None, "text": m.group(2)[:300]})
    return out


def build_interruptions(sup, restore, seg_out):
    """Restarts by s3_resume.sh (or by hand), pauses / stops by hand and restores, oldest first, with a summary: restarts,
    the downtime s3_resume.sh logged (only gaps over 3 minutes are logged; a restart by hand after a lost host gets an
    estimate from the last status.txt that showed the segment running), segments recovered / discarded / lost, time
    paused, end moves."""
    ents = [dict(e) for e in sup.get("interruptions", [])]
    so = {s["seg"]: s for s in seg_out}
    for e in ents:
        s = so.get(e.get("seg"))
        if e["kind"] != "restart" or s is None:
            continue
        if e.get("outcome") is None and s.get("lost"):
            e["outcome"] = "lost"; e["why"] = s.get("why")
        if e.get("since") is None and s.get("end_est") and s.get("end") and e["ts"] - s["end"] > 180:
            e["since"] = s["end"]; e["since_est"] = s["end_est"]; e["downtime_est_min"] = round((e["ts"] - s["end"]) / 60)
        if e.get("outcome") in ("discarded", "lost") and e.get("runs") is None:
            e["runs"] = s.get("runs_total")
    rs = sorted([e for e in restore if e.get("ts")], key=lambda e: e["ts"]) + [
        e for e in sup.get("events", []) if e["kind"] == "restore" and e.get("ts")]
    rs.sort(key=lambda e: e["ts"])
    grp = []
    for e in rs:                  # one entry per restore run (its lines are minutes apart)
        if grp and e["ts"] - grp[-1]["until"] <= 1800:
            grp[-1]["until"] = e["ts"]; grp[-1]["lines"] += 1; grp[-1]["text"] = e["text"]
        else:
            grp.append({"kind": "restore", "ts": e["ts"], "until": e["ts"], "lines": 1, "text": e["text"], "first": e["text"]})
    ents += grp
    ents.sort(key=lambda e: e["ts"] or 0)
    r = [e for e in ents if e["kind"] == "restart"]; p = [e for e in ents if e["kind"] == "pause"]
    summ = {"restarts": len(r), "with_downtime": len([e for e in r if e.get("downtime_min") is not None]),
            "downtime_min": sum(e.get("downtime_min") or 0 for e in r),
            "recovered": len([e for e in r if e.get("outcome") == "recovered"]), "discarded": len([e for e in r if e.get("outcome") == "discarded"]), "lost": len([e for e in r if e.get("outcome") == "lost"]),
            "downtime_est_min": sum(e.get("downtime_est_min") or 0 for e in r),
            "between_segments": len([e for e in r if e.get("seg") is None]), "pauses": len(p),
            "paused_s": sum(e["until"] - e["ts"] for e in p if e.get("until")), "open_pause": any(e.get("until") is None for e in p),
            "restores": len(grp), "end_moves": sup.get("end_moves", []), "carry": sup.get("carry", [])[-12:],
            "adoptive": bool(sup.get("adoptive"))}      # heartbeat / adoption supervisor (else downtime is unknown)
    return {"entries": ents, "summary": summ}


# Hosts a chain ran on that no status.txt seen by this panel names (main session, 2026-11-19): s3ic ran on the old RTX 5090
# box until it was paused for the move, then on the interruptible box (Vast.ai, RTX PRO 6000) from the restore on.
KNOWN_HOSTS = {"s3ic": [("359e176b111f", "2026-11-18 12:19:53", "2026-11-19 03:11:34", "old RTX 5090 box"),
                        ("2c4dcc67332d", "2026-11-19 06:07:31", "2026-11-19 08:53:20", "interruptible box, RTX PRO 6000")]}


def build_hosts(run, sup, status, hist):
    """Which host ran the chain when. status.txt names the host ("S3 run X on HOST: ..."); supervisor.log does not, but a
    host can only change at a supervisor start, so a status.txt written at t by HOST says HOST ran the chain from the
    last supervisor start before t until t. Observations: status.txt now and the ones remembered at earlier syncs
    (state/live_history_*.jsonl), plus KNOWN_HOSTS. Returns {current, intervals: [{host, from, to, src, note}]}."""
    starts = sorted(sup.get("starts") or [])
    obs = [(h["updated"], h["host"]) for h in hist if h.get("host") and h.get("updated")]
    if status.get("present") and status.get("host") and status.get("updated"):
        obs.append((status["updated"], status["host"]))
    ivs = []
    for host, a, b, note in KNOWN_HOSTS.get(run, []):
        ivs.append({"host": host, "from": utc_ts(a), "to": utc_ts(b), "src": "known", "note": note})
    for t, host in sorted(set(obs)):
        s0 = max([s for s in starts if s <= t] or [t])
        ivs.append({"host": host, "from": s0, "to": t, "src": "status.txt", "note": None})
    ivs.sort(key=lambda x: (x["from"], x["to"]))
    out = []
    for iv in ivs:           # merge the observations of one host that follow each other
        if out and out[-1]["host"] == iv["host"] and iv["from"] <= max(out[-1]["to"], out[-1]["from"]) + 6 * 3600:
            out[-1]["to"] = max(out[-1]["to"], iv["to"]); out[-1]["from"] = min(out[-1]["from"], iv["from"])
            if iv["src"] != out[-1]["src"]:
                out[-1]["src"] = "known + status.txt"
            out[-1]["note"] = out[-1]["note"] or iv["note"]
        else:
            out.append(dict(iv))
    return {"current": status.get("host"), "current_at": status.get("updated"), "intervals": out}


def host_at(intervals, ts):
    """Host that ran the chain at ts, if an interval covers it (else None: not derivable)."""
    if ts is None:
        return None
    for iv in intervals:
        if iv["from"] - 5 <= ts <= iv["to"]:
            return iv["host"]
    return None


# ---------------------------------------------------------------- chains and lineage
def local_chains():
    """Chains on the laptop = directories under the roots' local directories with a status.txt or curve.txt:
    [(chain, status.txt mtime)], newest first (rclone copies keep Nextcloud's modification time)."""
    out = []; seen = set()
    for _, base in ROOTS:
        try:
            names = os.listdir(base)
        except OSError:
            continue
        for n in names:
            d = os.path.join(base, n)
            if n in seen or re.search(r"_s\d{3}($|_)", n) or not os.path.isdir(d):
                continue
            st = sig(os.path.join(d, "status.txt"))
            if st or os.path.exists(os.path.join(d, "curve.txt")):
                out.append((n, st[1] / 1e9 if st else 0.0)); seen.add(n)
    return sorted(out, key=lambda x: -x[1])


def newest_local_chain():
    ch = local_chains()
    return ch[0][0] if ch else RUN


def parent_of(run):
    """(parent chain, segment) whose weights the chain started from, or None. From the first supervisor start line
    ("start weights .../<chain>_sNNN/weights.pt"), else KNOWN_PARENT. None for the chains in NO_PARENT."""
    if run in NO_PARENT:
        return None
    text = read_text(os.path.join(base_of(run), run, "supervisor.log")) or ""
    m = re.search(r"supervisor start: .*?start weights (\S+)", text)
    if m:
        mm = re.search(r"([^/\s]+)_s(\d{3})/weights\.pt$", m.group(1))
        if mm and mm.group(1) != run:
            return (mm.group(1), int(mm.group(2)))
    return KNOWN_PARENT.get(run)


def lineage(run):
    """[(chain, last segment of it in the lineage)] from the oldest ancestor to run (whose entry is (run, None))."""
    out = [(run, None)]; seen = {run}
    p = parent_of(run)
    while p and p[0] not in seen and len(out) < 12:
        out.insert(0, p); seen.add(p[0]); p = parent_of(p[0])
    return out


def chain_evals(run):
    """{seg: {seg, runs, e, ts}} of a chain: curve.txt, plus supervisor.log eval lines for segments curve.txt lacks."""
    d = os.path.join(base_of(run), run)
    curve = cached("curve", [os.path.join(d, "curve.txt")], lambda p: {"rows": parse_curve(p)}, disk=False)["rows"] or []
    sup = cached("supervisor", [os.path.join(d, "supervisor.log")], parse_supervisor, disk=False)
    ev = {}
    for k, sg in sup.get("segments", {}).items():
        if sg.get("eval_ts") is not None:
            ev[int(k)] = {"seg": int(k), "runs": sg.get("eval_runs"), "e": sg.get("eval"), "ts": sg.get("eval_ts")}
    for r in curve:
        ts = ev.get(r["seg"], {}).get("ts")
        ev[r["seg"]] = {"seg": r["seg"], "runs": r["runs"], "e": r["eval"], "ts": ts}
    return ev


def build_lineage(run, own_eval=None):
    """Fixed-seed evaluations along the lineage (e.g. s3a 4..49 -> s3c 0 -> s3v3 0..), runs cumulative over chains."""
    lin = lineage(run); chains = []; points = []; offset = 0
    for i, (c, upto) in enumerate(lin):
        if c == run and own_eval is not None:
            ev = {e["seg"]: {"seg": e["seg"], "runs": e.get("runs"), "e": e.get("e"), "ts": e.get("ts")} for e in own_eval}
        else:
            ev = chain_evals(c)
        segs = sorted(k for k in ev if upto is None or k <= upto)
        chains.append({"run": c, "upto": upto, "offset": offset, "parent": list(lin[i - 1]) if i else None,
                       "evals": len([k for k in segs if ev[k]["e"]]), "have_curve": os.path.exists(os.path.join(base_of(c), c, "curve.txt"))})
        for k in segs:
            x = ev[k]; e = x["e"]
            if not e or x["runs"] is None:
                continue
            points.append({"chain": c, "seg": k, "runs": x["runs"], "cum": offset + x["runs"], "ts": x["ts"], "pct": e["pct"], "lo": e["lo"],
                           "hi": e["hi"], "won": e["won"], "ok": e["ok"], "floor": e["floor"], "act1": e["act1"], "act2": e["act2"]})
        if upto is not None:     # the next chain starts where this one was branched off
            at = [ev[k]["runs"] for k in segs if ev[k]["runs"] is not None]
            offset += at[-1] if at else 0
    return {"chains": chains, "points": points}


def build(run=None, progress=lambda s: None):
    t_build = time.time()
    RUN = run or globals()["RUN"]
    B = base_of(RUN); run_dir = os.path.join(B, RUN)
    out = {"run": RUN, "base": os.path.relpath(B, ROOT), "remote": remote_of(RUN), "generated_at": time.time(), "notes": [], "bin_seconds": BIN,
           "counted_from": COUNTED_FROM.get(RUN, 0)}
    if not os.path.isdir(B):
        out["notes"].append("No local data yet (%s does not exist). Run a light sync." % out["base"])
    status = dict(cached("status", [os.path.join(run_dir, "status.txt")], parse_status, disk=False))    # a copy: run counts are corrected below
    sup = cached("supervisor", [os.path.join(run_dir, "supervisor.log")], parse_supervisor, disk=False)
    curve = cached("curve", [os.path.join(run_dir, "curve.txt")], lambda p: {"rows": parse_curve(p)}, disk=False)["rows"]
    out["status"] = status
    target = (sup.get("config") or {}).get("ascension", 10)
    out["target_asc"] = target; out["config"] = sup.get("config") or {}
    cfg = read_text(os.path.join(run_dir, "config.env"))
    if cfg:
        out["config"]["env"] = dict(re.findall(r"export (\w+)='([^']*)'", cfg))
    # actors actually started: extra.env (EXTRA_N) and the newest seg_N.log ("run <name>: 52 actors") override the
    # supervisor's start line, which shows config.env's N
    ex = read_text(os.path.join(run_dir, "extra.env")) or ""
    m = re.search(r"^EXTRA_N=['\"]?(\d+)", ex, re.M)
    if m:
        out["config"]["actors"] = int(m.group(1))
    sl = sorted(glob.glob(os.path.join(run_dir, "seg_*.log")), key=lambda p: int(re.sub(r"\D", "", os.path.basename(p)) or 0))
    m = re.search(r"\] run \S+: (\d+) actors", read_text(sl[-1]) or "") if sl else None
    if m:
        out["config"]["actors"] = int(m.group(1))
    end_file = read_text(os.path.join(run_dir, "end_time"))
    try:
        out["end_time"] = int(end_file.strip())
    except (AttributeError, ValueError):
        out["end_time"] = out["config"].get("ends") or status.get("ends")
    # the supervisor moves the end by the time a chain spent paused (end_time.paused, paused_at): status.txt's "ends",
    # when it is newer than the end_time file, is the end the supervisor is working to
    ef = sig(os.path.join(run_dir, "end_time"))
    if status.get("ends") and (ef is None or status.get("mtime", 0) >= ef[1] / 1e9):
        out["end_time"] = status["ends"]
    # Hard-coded changes made by hand on the server (owner, 2026-11-13). The server's end_time file only moves when
    # extend_run.sh restarts the supervisor at the original end, so the panel uses the extended end from now on.
    if RUN in EXTENDED_END and (out["end_time"] or 0) < EXTENDED_END[RUN]:
        out["end_time"] = EXTENDED_END[RUN]
    out["info"] = list(RUN_NOTES.get(RUN, []))
    out["extra_env"] = dict(re.findall(r"^(\w+)=\"?(.*?)\"?\s*$", ex, re.M))
    out["segment_env"] = dict(re.findall(r"^(\w+)=\"?(.*?)\"?\s*$", read_text(os.path.join(run_dir, "segment.env")) or "", re.M))
    out["chain_info"] = chain_info(out)
    out["run_start"] = sup["starts"][0] if sup.get("starts") else None
    out["supervisor_end"] = sup.get("end"); out["supervisor_present"] = sup.get("present", False)
    out["pause"] = sup.get("pause") if (sup.get("pause") or {}).get("state") else None
    if not sup.get("present"):
        out["notes"].append("supervisor.log is not on the laptop: segment start times, results and guard stops are unknown. Run a light sync.")

    # segments: union of the supervisor log and the local segment directories
    segs = {int(k): dict(v) for k, v in sup.get("segments", {}).items()}
    for d in sorted(glob.glob(os.path.join(B, RUN + "_s[0-9][0-9][0-9]"))):
        k = int(d[-3:]); segs.setdefault(k, {"seg": k})
    if status.get("segment") is not None and status.get("kind") in ("running", "evaluating", "between"):
        segs.setdefault(status["segment"], {"seg": status["segment"]})
    order = sorted(segs)
    hist = add_history(status, RUN)
    seen = {}          # per segment: the last status.txt (now or remembered) that showed it running, and its run count
    for h in hist + ([{"updated": status.get("updated"), "segment": status.get("segment"), "runs_so_far": status.get("runs_so_far"),
                       "state": status.get("state")}] if status.get("present") else []):
        if h.get("segment") is not None and h.get("updated") and "running" in (h.get("state") or ""):
            x = seen.setdefault(h["segment"], {"last": 0, "runs": None})
            x["last"] = max(x["last"], h["updated"])
            if isinstance(h.get("runs_so_far"), int):
                x["runs"] = max(x["runs"] or 0, h["runs_so_far"])
    for i, k in enumerate(order):
        s = segs[k]
        for f, v in (("name", "%s_s%03d" % (RUN, k)), ("start", None), ("end", None), ("planned", None), ("runs_at_start", None),
                     ("result", None), ("why", None), ("runs_logged", None), ("eval", None), ("eval_ts", None), ("last_macro", None)):
            s.setdefault(f, v)
        if s["result"] is None and s["start"] is not None:
            live = status.get("present") and status.get("segment") == k and status.get("kind") in ("running", "evaluating", "between")
            s["result"] = "running" if live and status.get("kind") == "running" else "open"
        if s["runs_at_start"] is not None and seen.get(k, {}).get("runs") is not None and s["result"] not in ("ok", "guard", "recovered"):
            s["runs_seen"] = max(0, seen[k]["runs"] - s["runs_at_start"])
        # a segment cut off by a supervisor start (host lost, chain restored elsewhere): it ran at least until the last
        # status.txt that showed it running, not until the restart (the time between is downtime). Only for a chain whose
        # supervisor adopts interrupted segments (see "lost" below): the run count used then is the one seen at that time.
        if s["result"] == "interrupted" and s.get("why") == "supervisor restarted" and sup.get("adoptive") and seen.get(k) and s["end"] \
                and s["start"] and s["start"] <= seen[k]["last"] < s["end"] - 900:
            s["end"] = seen[k]["last"]; s["end_est"] = "last seen running in status.txt"
    # The supervisor counts runs from the run records on its disk. A chain restored on another host gets back only the
    # records that reached Nextcloud, so the count can go back (s3ic 2026-11-19: 314,598 -> 294,723). From such a
    # segment on, counts logged after its start are shifted so that the run axis keeps going forward.
    adj = 0; breaks = []; prev = None
    for k in order:
        s = segs[k]; r = s["runs_at_start"]
        if r is None:
            continue
        if prev is not None and r + adj < segs[prev]["runs_at_start"]:
            p = segs[prev]
            n = p["runs_logged"] if p["runs_logged"] is not None else p.get("runs_seen") or 0
            adj = p["runs_at_start"] + n - r
            t0 = max([x for x in sup.get("starts", []) if s["start"] and x <= s["start"]] or [s["start"] or 0])
            breaks.append({"ts": t0, "seg": k, "adj": adj, "logged": r})
        if adj:
            s["runs_at_start_logged"] = r; s["runs_at_start"] = r + adj
        prev = k
    adj_at = lambda ts: ([b["adj"] for b in breaks if ts is not None and ts >= b["ts"]] or [0])[-1]
    out["runs_breaks"] = breaks
    if breaks and isinstance(status.get("runs_so_far"), int) and adj_at(status.get("updated")):
        status["runs_so_far_logged"] = status["runs_so_far"]; status["runs_so_far"] += adj_at(status.get("updated"))
    for i, k in enumerate(order):
        s = segs[k]
        nxt = segs[order[i + 1]] if i + 1 < len(order) else None
        if s["runs_logged"] is None and nxt and nxt.get("runs_at_start") is not None and s.get("runs_at_start") is not None:
            s["runs_from_next"] = nxt["runs_at_start"] - s["runs_at_start"]
        # A supervisor that adopts interrupted segments (s3ic since 2026-11-19) logs "recovered" or "NOT adopted" for each.
        # A segment interrupted without either line (host lost, chain restored elsewhere: the restore gives such a segment
        # an empty directory and a verdict) whose successor started from other weights was lost: shown like a discarded one.
        if s["result"] == "interrupted" and sup.get("adoptive") and nxt and nxt.get("from"):
            fm = re.search(r"([\w.-]+)/weights\.pt", nxt["from"])
            if fm and fm.group(1) != s["name"]:
                s["result"] = "discarded"; s["lost"] = True
                s["why"] = "lost with the host: no adoption after the restart, the chain went on from %s" % fm.group(1)

    # evaluation curve
    eval_report_sections = split_eval_report(read_text(os.path.join(run_dir, "eval_report.txt")))
    evals = {}
    for r in (curve or []):
        evals[r["seg"]] = {"seg": r["seg"], "runs": r["runs"], "e": r["eval"], "text": r["text"], "src": "curve.txt"}
    for k, s in segs.items():
        if k not in evals and s.get("eval_ts") is not None:
            evals[k] = {"seg": k, "runs": s.get("eval_runs"), "e": s.get("eval"), "text": s.get("eval_text"), "src": "supervisor.log"}
    for k, ev in evals.items():
        ev["ts"] = segs.get(k, {}).get("eval_ts")
        if breaks and ev.get("runs") is not None:       # counted on a restored chain: see the run-count shift above
            a = adj_at(ev["ts"] if ev["ts"] is not None else (segs.get(k) or {}).get("start"))
            if a:
                ev["runs_logged"] = ev["runs"]; ev["runs"] += a
        for p in (os.path.join(run_dir, "eval_%d.log" % k), os.path.join(run_dir, "eval", "seg_%03d" % k, "summary.txt")):
            t = read_text(p)
            d = parse_eval_log(t) if t else None
            if d:
                ev["detail"] = {"fights": d.get("fights"), "deaths": d.get("deaths"), "bosses": d.get("bosses"), "bosses_won": d.get("bosses_won")}
                if ev["e"] is None:
                    ev["e"] = {x: d[x] for x in ("arm", "n", "errors", "won", "ok", "pct", "lo", "hi", "ci_logged", "floor", "act1", "act2")}
                break
        rp = glob.glob(os.path.join(run_dir, "eval", "seg_%03d" % k, "tf64.jsonl"))
        if rp:
            ev["records"] = cached("evalruns", [rp[0]], lambda p: agg_eval_runs(p, target))
        rep_t = read_text(os.path.join(run_dir, "eval", "seg_%03d" % k, "report.txt")) or eval_report_sections.get(k)
        if rep_t:
            ev["report"] = parse_eval_report(rep_t)
    out["eval"] = [evals[k] for k in sorted(evals)]
    if curve is None:
        out["notes"].append("curve.txt is not on the laptop: no fixed-seed evaluation curve. Run a light sync.")

    # per segment: meta
    macro_rows = []; ctrain_rows = []; thr = []; seg_out = []; problems_actor = []
    for k in order:
        s = segs[k]; name = s["name"]; meta = os.path.join(B, name, "meta")
        has_meta = os.path.isdir(meta)
        so = {"seg": k, "name": name, "start": s["start"], "end": s["end"], "planned": s["planned"], "runs_at_start": s["runs_at_start"],
              "result": s["result"], "why": s["why"], "guard": s.get("guard"), "runs_logged": s["runs_logged"], "runs_from_next": s.get("runs_from_next"),
              "eval": evals.get(k, {}).get("e"), "eval_text": evals.get(k, {}).get("text"), "has_meta": has_meta, "counted": k >= out["counted_from"],
              "missing": []}
        so.update(runs_total=None, runs_counted=None, duration=None, runs_per_hour=None)
        if s["result"] in ("recovered", "discarded"):
            so[s["result"] + "_at"] = s.get(s["result"] + "_at")
        for f in ("lost", "end_est", "runs_seen", "runs_at_start_logged"):      # interruptions, lost hosts, restored run counts
            if s.get(f) is not None:
                so[f] = s[f]
        silent_all = 0
        try:
            progress("segment %d" % k)
            if s["result"] != "discarded":      # a segment not adopted after an interruption is a gap: its training was thrown away
                # macro trainer
                mp = os.path.join(meta, "macro_train_log.jsonl")
                if os.path.exists(mp):
                    m = cached("macro", [mp], agg_macro, disk=False)
                    for r in m["rows"]:
                        r2 = dict(r); r2["seg"] = k; r2["ts"] = s["start"] + r["t"] if s["start"] else None; macro_rows.append(r2)
                    so["macro_updates"] = len(m["rows"]); so["macro_bad_lines"] = m["bad"]
                else:
                    so["missing"].append("macro_train_log.jsonl")
                # combat trainer
                cp = next((p for p in (os.path.join(meta, "train_log.jsonl"), os.path.join(meta, "logs", "ctrain.txt")) if os.path.exists(p)), None)
                if cp:
                    c = cached("ctrain", [cp], agg_ctrain, disk=False)
                    mt = [(r["t"], r["runs"]) for r in macro_rows if r["seg"] == k]
                    if s["runs_at_start"] is not None:
                        mt = [(0.0, s["runs_at_start"])] + mt
                    for r in c["rows"]:
                        r2 = dict(r); r2["seg"] = k; r2["ts"] = s["start"] + r["t"] if s["start"] else None
                        r2["runs"] = interp(mt, r["t"]); ctrain_rows.append(r2)
                    so["ctrain_records"] = c["n"]
                else:
                    so["missing"].append("train_log.jsonl / logs/ctrain.txt")
                # run records
                rfiles = sorted(glob.glob(os.path.join(meta, "runs", "a*.jsonl")), key=actor_key)
                if rfiles:
                    tot = {"lines": 0, "bad": 0, "error_n": 0, "perturbed": 0, "errors": {}, "g": {"t": {}, "l": {}}, "asc": {}, "files": len(rfiles),
                           "asc_err": {}, "stale": {}, "look": {}, "look_runs": 0}
                    for i, p in enumerate(rfiles):
                        if is_fresh(p):
                            so["missing"].append("runs/%s (still downloading)" % os.path.basename(p)); continue
                        a = cached("runs", [p], lambda q: agg_runs(q, target))
                        for f in ("lines", "bad", "error_n", "perturbed"):
                            tot[f] += a[f]
                        for g in ("t", "l"):
                            merge_group(tot["g"][g], a["g"][g])
                        for asc, v in a["asc"].items():
                            x = tot["asc"].setdefault(asc, [0, 0, 0, 0, 0])
                            for j in range(5):
                                x[j] += v[j]
                        for f in ("asc_err", "stale", "look"):
                            for kk, v in (a.get(f) or {}).items():
                                tot[f][kk] = tot[f].get(kk, 0) + v
                        tot["look_runs"] += a.get("look_runs", 0)
                        for key, e in a["errors"].items():
                            x = tot["errors"].setdefault(key, {"n": 0, "example": e["example"], "seed": e.get("seed"), "floor": e.get("floor"), "actors": []})
                            x["n"] += e["n"]; x["actors"].append(os.path.basename(p)[:-6])
                    so["runs"] = tot
                else:
                    so["missing"].append("runs/*.jsonl")
                # fight records
                ffiles = sorted(glob.glob(os.path.join(meta, "fights", "a*.jsonl")), key=actor_key)
                if ffiles:
                    ft = {"files": 0, "lines": 0, "bad": 0, "skipped": 0, "g": {"t": {"room": {}, "enc": {}}, "l": {"room": {}, "enc": {}}}}
                    for i, p in enumerate(ffiles):
                        rp = os.path.join(meta, "runs", os.path.basename(p))
                        if is_fresh(p) or not os.path.exists(rp):
                            if is_fresh(p):
                                so["missing"].append("fights/%s (still downloading)" % os.path.basename(p))
                            continue
                        progress("segment %d: fight records %d/%d" % (k, i + 1, len(ffiles)))
                        a = cached("fights", [p, rp], lambda q, r: agg_fights(q, r, target))
                        ft["files"] += 1
                        for f in ("lines", "bad", "skipped"):
                            ft[f] += a[f]
                        for g in ("t", "l"):
                            for room, v in a["g"][g]["room"].items():
                                x = ft["g"][g]["room"].setdefault(room, [0, 0]); x[0] += v[0]; x[1] += v[1]
                            for enc, v in a["g"][g]["enc"].items():
                                x = ft["g"][g]["enc"].setdefault(enc, [v[0], 0, 0, 0.0, 0, 0, 0, 0.0])
                                for j in range(1, 8):
                                    x[j] += v[j]
                    ft["of_files"] = len(rfiles) or len(ffiles)
                    so["fights"] = ft
                # actor logs -> throughput
                afiles = sorted(glob.glob(os.path.join(meta, "logs", "a[0-9]*.txt")), key=actor_key)
                if afiles:
                    acts = [(os.path.basename(p)[:-4], cached("actorlog", [p], agg_actor_log)) for p in afiles]
                    srv = [cached("serverlog", [p], agg_server_log) for p in sorted(glob.glob(os.path.join(meta, "logs", "server[0-9]*.txt")))]
                    msp = os.path.join(meta, "logs", "mserver.txt")
                    msrv = cached("serverlog", [msp], agg_server_log) if os.path.exists(msp) else None
                    rows, summ = throughput(acts, srv, msrv)
                    for r in rows:
                        r["seg"] = k; r["ts"] = s["start"] + r["t"] if s["start"] else None
                        r["runs"] = (s["runs_at_start"] or 0) + r.pop("cum_runs") if s["runs_at_start"] is not None else None
                        thr.append(r)
                    so["actors"] = summ
                    last_all = max([a["last_t"] or 0 for _, a in acts] or [0])
                    if all(a["last_t"] is None for _, a in acts):
                        # no actor got to its first progress line (a segment cut short in its first minutes): one line, not
                        # one per actor, and none when the segment has no runs (judged below, once the run count is known)
                        silent_all = len(acts)
                    for nm, a in acts:
                        if silent_all:
                            continue
                        for l in a["stalls"]:
                            problems_actor.append({"seg": k, "actor": nm, "kind": "actor", "text": l})
                        if a["last_t"] is not None and last_all - a["last_t"] > 600:
                            problems_actor.append({"seg": k, "actor": nm, "kind": "silent", "text": "%s printed its last progress line at %d s; the other actors "
                                                   "went on to %d s (silent for the last %d min of the segment)" % (nm, a["last_t"], last_all, (last_all - a["last_t"]) // 60),
                                                   "ts": s["start"] + a["last_t"] if s["start"] else None})
                        elif a["last_t"] is None:
                            problems_actor.append({"seg": k, "actor": nm, "kind": "silent", "text": "%s printed no progress line at all" % nm})
                    notes = {}
                    for nm, a in acts:
                        for part in (a["notes"] or "").split(", "):
                            mm = re.match(r"(.*) (\d+)$", part.strip())
                            if mm:
                                notes[mm.group(1)] = notes.get(mm.group(1), 0) + int(mm.group(2))
                    so["actor_notes"] = sorted(notes.items(), key=lambda x: -x[1])[:20]
                    pm = [a["pm"] for _, a in acts if a.get("pm")]
                    if pm or any(a.get("pm_end") for _, a in acts):
                        end = {}
                        for _, a in acts:
                            for src, o in (a.get("pm_end") or {}).items():
                                e = end.setdefault(src, {"actors": 0})
                                e["actors"] += 1
                                for kk, v in o.items():
                                    if kk == "dev":
                                        e["dev_sum"] = e.get("dev_sum", 0.0) + v * o.get("dev_n", 0)
                                    else:
                                        e[kk] = e.get(kk, 0) + v
                        for e in end.values():
                            if e.get("dev_n"):
                                e["dev"] = round(e.pop("dev_sum", 0.0) / e["dev_n"], 5)
                            e.pop("dev_sum", None)
                        so["potion_mix"] = {"actors": len(pm), "decisions": [sum(x[0] for x in pm), sum(x[1] for x in pm)],
                                            "swaps": [sum(x[2] for x in pm), sum(x[3] for x in pm)], "end": end}
                else:
                    so["missing"].append("logs/a*.txt")
            if s["result"] == "discarded":
                so["missing"] = []          # nothing of it is shown: not adopted, the chain went on from the segment before
            elif not has_meta:
                so["missing"] = ["meta/ (uploaded when the segment ends)"] if s["result"] in ("running", "open") else \
                    ["meta/ never reached Nextcloud: the segment was recovered after an interruption, and the upload of its "
                     "meta/ was cut off (it may still come with a later retry of that upload; then a light sync brings it)"] if s["result"] == "recovered" else \
                    ["meta/ (run a light sync; if it stays missing, Nextcloud has none for this segment, as for segment 0)"]
            # run count: supervisor log first, then records, then the next segment's offset (a discarded segment: the
            # supervisor's count, i.e. the next segment's offset, since its uploaded records may be cut off)
            counted = so.get("runs", {}).get("lines")
            if s["result"] == "discarded":
                so["runs_total"] = s.get("runs_from_next") if s.get("runs_from_next") is not None else counted
            else:
                so["runs_total"] = s["runs_logged"] if s["runs_logged"] is not None else counted if counted is not None else s.get("runs_from_next")
            so["runs_counted"] = counted
            if so["result"] == "running" and status.get("runs_so_far") is not None and s["runs_at_start"] is not None:
                so["runs_total"] = status["runs_so_far"] - s["runs_at_start"]; so["runs_live"] = True
            # duration: start to end line; for an interrupted / recovered / discarded segment the end is when the host
            # stopped (last heartbeat, else the restart line), so runs per hour are over the time it really ran
            dur = (s["end"] - s["start"]) if s["start"] and s["end"] else status.get("seg_elapsed") if so["result"] == "running" else None
            so["duration"] = dur
            so["runs_per_hour"] = round(3600 * so["runs_total"] / dur) if so["runs_total"] and dur and dur > 60 and s["result"] != "discarded" else None
            if silent_all and so["runs_total"]:
                problems_actor.append({"seg": k, "actor": "all", "kind": "silent", "text": "none of the %d actors printed a progress line, "
                                       "although the segment has %d runs" % (silent_all, so["runs_total"])})
        except Exception as e:      # one unreadable segment must not hide the others
            so["missing"].append("parser error: %s: %s" % (type(e).__name__, str(e)[:200]))
            out["notes"].append("Segment %d: its files could not be parsed (%s: %s); the other segments are shown." % (k, type(e).__name__, str(e)[:200]))
        seg_out.append(so)

    # live macro points of segments without a macro log (from the status.txt history)
    have = {k for k in order if any(r["seg"] == k for r in macro_rows)}
    live = {}
    for h in hist:
        for o in h.get("macro", []):
            if not isinstance(o.get("runs"), (int, float)) or not isinstance(o.get("t"), (int, float)):
                continue
            k = None
            for kk in order:          # discarded segments are gaps: their points belong to no segment
                r0 = segs[kk].get("runs_at_start_logged", segs[kk]["runs_at_start"])     # the count as logged (see the shift above)
                if r0 is not None and o["runs"] > r0 and segs[kk]["result"] != "discarded" and \
                        (h.get("segment") is None or kk <= h["segment"]):
                    k = kk
            if k is None or k in have:
                continue
            # the trainer's t counts from the segment start: a line older than that is another segment's (after a
            # restore, status.txt shows the tail of a restored segment's log while the new one has no line yet)
            if segs[k]["start"] and h.get("updated") and o["t"] > h["updated"] - segs[k]["start"] + 120:
                continue
            live[(k, o["runs"])] = dict(o, seg=k, live=True, ts=segs[k]["start"] + o["t"] if segs[k]["start"] else None)
    macro_rows += [live[x] for x in sorted(live)]
    macro_rows.sort(key=lambda r: (r["seg"], r["runs"]))
    out["macro"] = macro_rows; out["ctrain"] = ctrain_rows; out["throughput"] = thr; out["segments"] = seg_out
    out["live_macro_points"] = len(live)
    for so in seg_out:
        so["memguard"] = sup.get("memguard", {}).get(so["seg"]) or sup.get("memguard", {}).get(str(so["seg"]))
    try:
        out["curriculum"] = build_curriculum(run_dir, out["config"].get("env") or {}, sup, seg_out, status, macro_rows)
    except Exception as e:
        out["curriculum"] = None; out["notes"].append("Curriculum could not be read: %s: %s" % (type(e).__name__, e))
    try:
        out["vmix"] = build_vmix(run_dir)
    except Exception as e:
        out["vmix"] = []; out["notes"].append("Vbar files could not be read: %s: %s" % (type(e).__name__, e))
    out["events"] = sup.get("events", [])
    try:
        rl = restore_lines(run_dir, RUN)
        have_ev = {(e["ts"], e["text"]) for e in out["events"]}
        out["events"] = sorted(out["events"] + [e for e in rl if (e["ts"], e["text"]) not in have_ev], key=lambda e: e["ts"] or 0)
        out["interruptions"] = build_interruptions(sup, rl, seg_out)
        out["hosts"] = build_hosts(RUN, sup, status, hist)
        for so in seg_out:
            so["host"] = host_at(out["hosts"]["intervals"], so["start"])
    except Exception as e:
        out["interruptions"] = None; out["hosts"] = None
        out["notes"].append("Interruptions / hosts could not be read: %s: %s" % (type(e).__name__, e))

    # problems
    probs = [dict(p) for p in sup.get("problems", [])]
    out["problems"] = {"supervisor": probs, "actors": problems_actor}
    out["have"] = {"status": bool(status.get("present")), "curve": curve is not None, "supervisor": bool(sup.get("present")),
                   "meta_segments": [s["seg"] for s in seg_out if s["has_meta"]],
                   "runs_segments": [s["seg"] for s in seg_out if "runs" in s],
                   "fights_segments": [s["seg"] for s in seg_out if s.get("fights", {}).get("files")]}
    out["pending_fresh"] = any("still downloading" in m for s in seg_out for m in s["missing"])
    try:
        out["lineage"] = build_lineage(RUN, out["eval"])
    except Exception as e:
        out["lineage"] = None; out["notes"].append("Lineage curve could not be built: %s: %s" % (type(e).__name__, e))
    out["build_seconds"] = round(time.time() - t_build, 2)
    return out


def interp(pairs, t):
    """runs at time t from (t, runs) pairs of one segment (linear, clamped)."""
    if not pairs:
        return None
    if t <= pairs[0][0]:
        return pairs[0][1]
    for (t0, r0), (t1, r1) in zip(pairs, pairs[1:]):
        if t <= t1:
            return round(r0 + (r1 - r0) * (t - t0) / max(1e-9, t1 - t0))
    if len(pairs) >= 2:        # after the last update: continue at the last rate
        (t0, r0), (t1, r1) = pairs[-2], pairs[-1]
        return round(r1 + (r1 - r0) * (t - t1) / max(1e-9, t1 - t0))
    return pairs[-1][1]


def throughput(acts, srv, msrv):
    """Per BIN: runs/h and simulations/s summed over actors (from the differences of their cumulative counters), actors
    reporting, mean bridge wait (cumulative since the segment start, as the actors print it), combat states/s and the
    servers' busy share (cumulative as well)."""
    bins = sorted({int(b) for _, a in acts for b in a["bins"]})
    rows = []
    for b in bins:
        dr = ds = 0; dts = []; rep = 0; waits = []; cum = 0
        for _, a in acts:
            cur = a["bins"].get(str(b))
            last = None
            for bb in range(b, -1, -1):
                if str(bb) in a["bins"]:
                    last = a["bins"][str(bb)]; break
            if last:
                cum += last[1]
            if not cur:
                continue
            rep += 1
            if cur[5] is not None:
                waits.append(cur[5])
            prev = a["bins"].get(str(b - 1)) or ([0, 0, 0, 0, 0, None, 0, 0] if b == 0 else None)
            if prev and cur[0] > prev[0]:
                dr += cur[1] - prev[1]; ds += cur[4] - prev[4]; dts.append(cur[0] - prev[0])
        dt = sum(dts) / len(dts) if dts else None
        row = {"t": (b + 1) * BIN, "actors": rep, "cum_runs": cum,
               "runs_h": round(3600 * dr / dt) if dt else None, "sims_s": round(ds / dt) if dt else None,
               "wait": round(sum(waits) / len(waits), 1) if waits else None}
        cs = []; dst = 0; sdt = []
        for sv in srv:
            cur = sv["bins"].get(str(b))
            if not cur:
                continue
            cs.append(cur[2]); prev = sv["bins"].get(str(b - 1)) or ([0, 0, 0, 0, 0, 0, 0] if b == 0 else None)
            if prev and cur[0] > prev[0]:
                dst += cur[1] - prev[1]; sdt.append(cur[0] - prev[0])
        row["srv_busy"] = round(sum(cs) / len(cs), 1) if cs else None
        row["srv_busy_max"] = max(cs) if cs else None
        row["states_s"] = round(dst / (sum(sdt) / len(sdt))) if sdt else None
        if msrv:
            cur = msrv["bins"].get(str(b))
            row["msrv_busy"] = cur[4] if cur else None
        row["_dt"] = dt
        rows.append(row)
    # the last step of a segment is usually a stub (a few actors print once more while they finish their runs)
    while len(rows) >= 2 and (rows[-1]["actors"] < 0.5 * rows[-2]["actors"] or (rows[-1]["_dt"] or 0) < BIN / 2):
        rows.pop()
    for r in rows:
        r.pop("_dt", None)
    fin = [a["final"] for _, a in acts if a["final"]]
    summ = {"n": len(acts), "runs": sum(f[1] for f in fin), "errors": sum(f[3] for f in fin),
            "sims_s": round(sum(f[4] / max(1, f[0]) for f in fin)) if fin else None,
            "wait": round(sum(f[5] for f in fin if f[5] is not None) / max(1, len([f for f in fin if f[5] is not None])), 1) if fin else None,
            "srv_busy": round(sum(s["bins"][max(s["bins"], key=int)][2] for s in srv if s["bins"]) / max(1, len([s for s in srv if s["bins"]])), 1) if srv else None,
            "msrv_busy": msrv["bins"][max(msrv["bins"], key=int)][4] if msrv and msrv["bins"] else None,
            "last_t": {nm: a["last_t"] for nm, a in acts}}
    return rows, summ


class Store:
    """Keeps the latest payload of the chosen chain; rebuilds in one background thread when asked, when the chain
    changes (set_run) or when its local files changed."""

    def __init__(self, run=None):
        self.lock = threading.Lock(); self.wake = threading.Event()
        self.run = run or RUN
        self.payload = None; self.json = b"{}"; self.state = {"busy": True, "what": "starting", "built_at": None, "error": None, "run": None}
        self._last_sig = None
        threading.Thread(target=self._loop, daemon=True).start()

    def request(self):
        self._last_sig = None; self.wake.set()

    def set_run(self, run):
        with self.lock:
            changed = run != self.run; self.run = run
        if changed:
            self.request()

    def _tree_sig(self, run):
        """Files of the chain, its segments and the top level of its ancestors (not the whole backup tree)."""
        n = 0; h = 0
        B = base_of(run)
        dirs = [os.path.join(B, run)] + glob.glob(os.path.join(B, run + "_s[0-9][0-9][0-9]"))
        tops = [os.path.join(base_of(c), c) for c, _ in lineage(run)[:-1]]
        for top in dirs:
            for root, _, files in os.walk(top):
                for f in files:
                    try:
                        st = os.stat(os.path.join(root, f))
                    except OSError:
                        continue
                    n += 1; h ^= hash((root, f, st.st_size, st.st_mtime_ns))
        for top in tops:
            for f in ("curve.txt", "supervisor.log"):
                x = sig(os.path.join(top, f))
                if x:
                    n += 1; h ^= hash((top, f, x[0], x[1]))
        return (run, n, h)

    def _loop(self):
        while True:
            try:
                run = self.run
                s = self._tree_sig(run)
                if s != self._last_sig or self.payload is None or self.payload.get("run") != run:
                    with self.lock:
                        self.state.update(busy=True, what="reading local files of " + run)
                    p = build(run, lambda w: self.state.update(what=run + ": " + w))
                    b = json.dumps(p, separators=(",", ":"), allow_nan=False, default=str).encode()
                    with self.lock:
                        self.payload = p; self.json = b
                        self.state.update(busy=False, what="", built_at=time.time(), error=None, run=run)
                    self._last_sig = None if p.get("pending_fresh") else s     # files in mid-download: look again
                    if self.run != run:          # the chain was switched while this one was being read
                        continue
            except Exception as e:          # a parsing bug must not stop the panel: show it
                import traceback
                with self.lock:
                    self.state.update(busy=False, what="", error="%s: %s | %s" % (type(e).__name__, e, traceback.format_exc()[-600:]))
                self._last_sig = None
            self.wake.wait(20); self.wake.clear()


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else PINNED or newest_local_chain()
    p = build(run, lambda w: print("  ..", w, file=sys.stderr))
    print("chain", run, "| local chains:", " ".join(c for c, _ in local_chains()))
    st = p["status"]
    print("status:", st.get("state"), "| segment", st.get("segment"), "| runs so far", st.get("runs_so_far"), "| updated",
          time.strftime("%F %T", time.gmtime(st["updated"])) if st.get("updated") else None)
    for e in p["eval"]:
        x = e["e"] or {}
        print("eval seg %d runs %s: won %s/%s (%.1f-%.1f%%) floor %s act1 %s act2 %s" % (e["seg"], e["runs"], x.get("won"), x.get("ok"), x.get("lo", 0), x.get("hi", 0),
                                                                                    x.get("floor"), x.get("act1"), x.get("act2")))
    for x in (p.get("lineage") or {}).get("points", []):
        print("lineage %-6s seg %3d cum runs %9d: won %s/%s = %.1f%%" % (x["chain"], x["seg"], x["cum"], x["won"], x["ok"], x["pct"]))
    for s in p["segments"]:
        r = s.get("runs") or {}
        print("seg %d %-9s runs %s (records %s, errors %s, perturbed %s) dur %s runs/h %s macro updates %s fights files %s missing %s" % (
            s["seg"], s["result"], s.get("runs_total"), s.get("runs_counted"), r.get("error_n"), r.get("perturbed"), s.get("duration"), s.get("runs_per_hour"),
            s.get("macro_updates"), s.get("fights", {}).get("files"), s["missing"]))
    print("macro rows %d, ctrain rows %d, throughput rows %d, problems %d + %d, build %.2f s" % (
        len(p["macro"]), len(p["ctrain"]), len(p["throughput"]), len(p["problems"]["supervisor"]), len(p["problems"]["actors"]), p["build_seconds"]))
    for n in p["notes"]:
        print("note:", n)


if __name__ == "__main__":
    main()
