"""Pull a chain's logs from Nextcloud into src/server_backup/<root>/ (read-only on the Nextcloud side).

A chain is one supervisor run (s3a, s3c, s3v3, s3ic, ...): its files are <chain>/ (status.txt, curve.txt, supervisor.log,
seg_N.log, eval_N.log, eval/seg_NNN/, config.env, extra.env, rclone_tail.log; s3ic also curriculum_stage, hook.log,
hook_status.txt, segment.env, vmix/, eval_report.txt) and one <chain>_sNNN/ per segment (meta/ uploaded when the segment
ends; data/ = training chunks).

Nextcloud roots (PANEL_REMOTE, comma-separated; default nextcloud:sts2_backup/s3,nextcloud:sts2_backup/s3ic): chains are
listed in all of them; each root is copied into its own local directory (s3parse.ROOTS: src/server_backup/s3 and
src/server_backup/s3ic), so the two never mix.

Two levels, for the chain(s) being synced (by default the one the panel shows):
  light  everything under <chain>/ and <chain>_sNNN/meta/ except fights/: status, curve, supervisor / segment /
         evaluation logs, evaluation outputs, trainer / actor / server logs, run records (about 35 MiB per v3 segment)
  full   light + the fight records (<chain>_sNNN/meta/fights/*.jsonl, about 800 MiB per v3 segment)
Both levels also take the top-level files of the chain's ancestors (<parent>/*: curve.txt, supervisor.log, status.txt;
not their eval/ or segments), so that the lineage curve (s3a -> s3c -> s3v3) can be drawn.
Never downloaded: weights and checkpoints (*.pt, meta/ckpt, meta/mckpt, <chain>/latest), the training chunks
(<chain>_sNNN/data/), the off-host carry backup (<chain>/carry/: replay windows and optimizer states, ~0.3 GB per set,
needed only by restore_from_nextcloud.sh; excluded since 2026-11-19, older local copies can be deleted) and stray directories such as <chain>_x001 or <chain>_smoke_s000.
Only `rclone lsf` / `rclone lsjson` and `rclone copy --update` are used, always with Nextcloud as the source.

CLI:  python3 src/s3/panel/s3sync.py chains                    chains on Nextcloud, newest status.txt first
      python3 src/s3/panel/s3sync.py preview [chain ...]       what light and full would download now
      python3 src/s3/panel/s3sync.py light | full [chain ...]  run one sync in the foreground
      (no chain given: PANEL_RUN if set, else the chain whose status.txt on Nextcloud is the newest)
env:  PANEL_REMOTE (Nextcloud roots, comma-separated), PANEL_RUN (pins the chain), PANEL_DATA (local directory of the
      first root; the others go next to it)
"""
import calendar, json, os, re, signal, subprocess, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s3parse   # noqa: E402   (roots and chain locations; s3parse touches no network)

ROOT = s3parse.ROOT
ROOTS = s3parse.ROOTS                  # [(remote root, local directory)]
REMOTE = ", ".join(r for r, _ in ROOTS)
PINNED = os.environ.get("PANEL_RUN") or None
DEST = ROOTS[0][1]
LEVELS = ("light", "full")
TIME_LIMIT = {"light": 30 * 60, "full": 3 * 3600, "preview": 5 * 60}
NET = ["--contimeout", "15s", "--timeout", "60s", "--retries", "2", "--low-level-retries", "3"]


SEGDIR = "_s[0-9][0-9][0-9]"     # rclone glob of a segment directory: s3c_s001 yes, s3c_smoke_s000 / s3c_x001 no


def filters(level, runs, ancestors=()):
    """rclone filter rules (first match wins), relative to REMOTE. runs: chains synced in full (light or full level);
    ancestors: chains of which only the top-level files are taken (lineage curve)."""
    # carry/ = the off-host copy of the replay / optimizer carries (carry_backup.sh, ~0.3 GB per set): for a restore only
    f = ["- *.pt", "- /*/data/**", "- /*/meta/ckpt/**", "- /*/meta/mckpt/**", "- /*/latest/**", "- /*/carry/**"]
    f += [f"+ /{r}/**" for r in runs]
    f += [f"+ /{a}/*" for a in ancestors if a not in runs]
    if level == "light":
        f.append("- /*/meta/fights/**")
    f += [f"+ /{r}{SEGDIR}/meta/**" for r in runs] + ["- *"]
    out = []
    for x in f:
        out += ["--filter", x]
    return out


def _list_root(remote, timeout):
    cmd = (["rclone", "lsjson", "--max-depth", "2", "--files-only", "--filter", f"- /*{SEGDIR}/**", "--filter", f"- /*{SEGDIR}_*/**",
            "--filter", "- /*_x[0-9]*/**", "--filter", "+ /*/status.txt", "--filter", "- *", remote] + NET)
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError("listing %s took longer than %d s" % (remote, timeout))
    except FileNotFoundError:
        raise RuntimeError("rclone is not installed")
    if p.returncode != 0:
        raise RuntimeError(remote + ": " + _rclone_error(p.stderr))
    out = []
    for o in json.loads(p.stdout or "[]"):
        path = o.get("Path", "")
        m = re.match(r"([^/]+)/status\.txt$", path)
        t = re.match(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)", o.get("ModTime", ""))
        if m and t:
            out.append((m.group(1), calendar.timegm(time.strptime(t.group(1), "%Y-%m-%dT%H:%M:%S"))))   # ModTime is UTC ("Z")
    return out


def remote_chains(timeout=120):
    """Chains on Nextcloud = top-level directories with a status.txt in any root, newest first: [(chain, unix mtime,
    remote root)]. Segment and stray directories are not listed (directory filters keep a listing to a few seconds).
    The roots are listed side by side; one that fails is reported in remote_chains.errors (all failing raises).
    Also records where each chain lives (s3parse.CHAIN_ROOT); a name in two roots counts in the first one."""
    res = {}; errs = []

    def one(i, remote):
        try:
            res[i] = _list_root(remote, timeout)
        except RuntimeError as e:
            errs.append(str(e))
    th = [threading.Thread(target=one, args=(i, r), daemon=True) for i, (r, _) in enumerate(ROOTS)]
    for t in th:
        t.start()
    for t in th:
        t.join(timeout + 5)
    remote_chains.errors = errs
    if not res:
        raise RuntimeError(" | ".join(errs) or "no Nextcloud root could be listed")
    out = []; seen = set()
    for i, (remote, loc) in enumerate(ROOTS):
        for c, t in res.get(i, []):
            if c in seen:
                continue
            seen.add(c); out.append((c, t, remote)); s3parse.CHAIN_ROOT[c] = (remote, loc)
    return sorted(out, key=lambda x: -x[1])


remote_chains.errors = []


def by_root(runs):
    """Chains grouped by Nextcloud root: [(remote, local directory, [chains])] in ROOTS order."""
    g = {}
    for r in runs:
        g.setdefault(s3parse.remote_of(r), []).append(r)
    return [(remote, loc, g[remote]) for remote, loc in ROOTS if remote in g] + \
           [(remote, s3parse.base_of(rs[0]), rs) for remote, rs in g.items() if remote not in [x for x, _ in ROOTS]]


def ancestors_of(runs):
    """Chains the given ones continue from (s3parse.lineage: supervisor.log, else the known lineage)."""
    try:
        import s3parse
        anc = []
        for r in runs:
            anc += [c for c, _ in s3parse.lineage(r)[:-1] if c not in anc and c not in runs]
        return anc
    except Exception:
        return []


def is_fight(path):
    return "/meta/fights/" in path


def _short(text, n=400):
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n] + " ..."


def _rclone_error(stderr):
    """The useful part of rclone's stderr (never the config: rclone does not print it)."""
    lines = [l for l in stderr.splitlines() if l.strip()]
    msgs = []
    for l in lines:
        try:
            j = json.loads(l)
            if j.get("level") in ("error", "fatal", "critical"):
                msgs.append(j.get("msg", ""))
        except ValueError:
            if "ERROR" in l or "Failed" in l or "error" in l.lower():
                msgs.append(l)
    return _short(" | ".join(msgs[-3:]) or (lines[-1] if lines else "rclone failed without a message"))


def preview(runs):
    """One recursive listing per Nextcloud root (what a full sync of these chains covers), compared with the local
    files. Returns {level: {remote_files, remote_bytes, todo_files, todo_bytes}} or raises RuntimeError."""
    runs = list(runs); anc = ancestors_of(runs)
    res = {l: {"remote_files": 0, "remote_bytes": 0, "todo_files": 0, "todo_bytes": 0} for l in LEVELS}
    segs = set()
    for remote, loc, rr in by_root(runs + anc):
        _preview_root(remote, loc, [r for r in rr if r in runs], [a for a in rr if a in anc], runs, anc, res, segs)
    res["segments_with_meta"] = sorted(segs)
    res["runs"] = runs; res["ancestors"] = anc
    res["checked_at"] = time.time()
    return res


def _preview_root(remote, loc, runs, anc, all_runs, all_anc, res, segs):
    cmd = ["rclone", "lsf", "-R", "--files-only", "--format", "tsp", "--separator", "\t", remote] + filters("full", runs, anc) + NET
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=TIME_LIMIT["preview"])
    except subprocess.TimeoutExpired:
        raise RuntimeError("listing %s took longer than 5 minutes" % remote)
    except FileNotFoundError:
        raise RuntimeError("rclone is not installed")
    if p.returncode != 0:
        raise RuntimeError(remote + ": " + _rclone_error(p.stderr))
    for line in p.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        mt, size, path = parts
        try:
            size = int(size); rt = time.mktime(time.strptime(mt, "%Y-%m-%d %H:%M:%S"))   # lsf prints local time
        except ValueError:
            continue
        first = path.split("/", 1)[0]
        if first not in all_runs and first not in all_anc:
            segs.add(first)
        lp = os.path.join(loc, path)
        try:
            st = os.stat(lp); todo = st.st_size != size or st.st_mtime + 2 < rt
        except OSError:
            todo = True
        for l in LEVELS:
            if l == "light" and is_fight(path):
                continue
            res[l]["remote_files"] += 1; res[l]["remote_bytes"] += size
            if todo:
                res[l]["todo_files"] += 1; res[l]["todo_bytes"] += size


class Syncer:
    """At most one rclone process at a time; state is readable at any moment through snapshot()."""

    def __init__(self):
        self.lock = threading.Lock()
        self.proc = None; self.timed_out = False; self.stopping = False
        self.on_done = None           # called (no arguments) after every sync that ended
        self.targets = lambda: [PINNED or "s3a"]     # chains to sync; the panel sets this to the chain it shows
        self.state = {"running": False, "level": None, "started_at": None, "progress": None,
                      "last_level": None, "last_started_at": None, "last_finished_at": None, "last_ok": None, "last_error": None,
                      "last_ok_at": None, "last_seconds": None, "last_transferred": None,
                      "preview": None, "preview_running": False, "preview_error": None,
                      "auto_minutes": 0, "auto_level": "light", "next_auto_at": None,
                      "remote": REMOTE, "runs": None, "last_runs": None, "dest": ", ".join(os.path.relpath(loc, ROOT) for _, loc in ROOTS)}
        threading.Thread(target=self._auto_loop, daemon=True).start()

    def snapshot(self):
        with self.lock:
            s = dict(self.state)
            s["progress"] = dict(s["progress"]) if s["progress"] else None
            return s

    # --- sync
    def start(self, level, source="button", runs=None):
        if level not in LEVELS:
            return False, f"unknown level {level!r}"
        with self.lock:
            if self.state["running"]:
                return False, f"a {self.state['level']} sync is already running"
            self.state.update(running=True, level=level, started_at=time.time(), progress={"source": source})
        threading.Thread(target=self._run, args=(level, runs), daemon=True).start()
        return True, f"{level} sync started"

    def _run(self, level, runs=None):
        t0 = time.time(); err = None; done = {"files": 0, "bytes": 0}
        try:
            runs = [r for r in (runs or self.targets()) if r]
        except Exception as e:
            runs = []; err = f"no chain to sync: {e}"
        anc = ancestors_of(runs)
        with self.lock:
            self.state["runs"] = runs
        self.timed_out = False
        deadline = time.time() + TIME_LIMIT[level]
        if not runs:
            err = err or "no chain to sync"
        for remote, dest, rr in (by_root(runs + anc) if runs else []):      # one rclone copy per Nextcloud root
            if self.stopping or time.time() >= deadline:
                break
            e, last = self._copy(level, remote, dest, [r for r in rr if r in runs], [a for a in rr if a in anc], deadline, done)
            done["files"] += last.get("files", 0); done["bytes"] += last.get("bytes", 0)
            if e:
                err = (err + " | " if err else "") + (remote + ": " + e if len(ROOTS) > 1 else e)
        now = time.time()
        with self.lock:
            self.state.update(running=False, level=None, started_at=None, progress=None, last_level=level, last_started_at=t0, last_runs=runs,
                              last_finished_at=now, last_ok=err is None, last_error=err, last_seconds=round(now - t0, 1),
                              last_transferred=done)
            if err is None:
                self.state["last_ok_at"] = now
                self.state["preview"] = None      # what was listed before the sync is no longer true
        self.stopping = False
        if self.on_done:
            try:
                self.on_done()
            except Exception:
                pass

    def _copy(self, level, remote, dest, runs, anc, deadline, done):
        """One `rclone copy --update` from a Nextcloud root into its local directory. Returns (error or None, last stats)."""
        os.makedirs(dest, exist_ok=True)
        cmd = (["rclone", "copy", remote, dest, "--update", "--transfers", "8", "--checkers", "8"] + filters(level, runs, anc) + NET +
               ["--use-json-log", "--stats", "2s", "--stats-log-level", "NOTICE"])
        last = {}
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, start_new_session=True)
        except OSError as e:
            return f"could not start rclone: {e}", last
        self.proc = proc
        killer = threading.Timer(max(1, deadline - time.time()), lambda: self._kill(proc, timed_out=True))
        killer.daemon = True; killer.start()
        errs = []
        for line in proc.stderr:
            try:
                j = json.loads(line)
            except ValueError:
                if line.strip():
                    errs.append(line.strip())
                continue
            st = j.get("stats")
            if st:
                last = {"bytes": st.get("bytes", 0), "total_bytes": st.get("totalBytes", 0), "files": st.get("transfers", 0),
                        "total_files": st.get("totalTransfers", 0), "checks": st.get("checks", 0), "speed": st.get("speed", 0),
                        "eta": st.get("eta"), "errors": st.get("errors", 0)}
                shown = dict(last, files=last["files"] + done["files"], bytes=last["bytes"] + done["bytes"],
                             total_files=last["total_files"] + done["files"], total_bytes=last["total_bytes"] + done["bytes"],
                             root=remote)
                with self.lock:
                    if self.state["progress"] is not None:
                        self.state["progress"].update(shown)
            elif j.get("level") in ("error", "fatal", "critical"):
                errs.append(j.get("msg", ""))
        rc = proc.wait(); killer.cancel(); self.proc = None
        if rc != 0:
            why = _short(" | ".join(errs[-3:])) if errs else "no message"
            return f"rclone exit {rc}" + (" (time limit reached)" if self.timed_out else " (stopped)" if rc < 0 else "") + f": {why}", last
        return None, last

    def _kill(self, proc, timed_out=False):
        self.timed_out = self.timed_out or timed_out
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except OSError:
            pass

    def stop(self):
        """Stops a running rclone (used when the panel exits); a sync over several roots does not go on to the next one."""
        self.stopping = True
        p = self.proc
        if p is not None and p.poll() is None:
            self._kill(p)

    # --- preview
    def start_preview(self):
        with self.lock:
            if self.state["preview_running"]:
                return False, "already checking"
            if self.state["running"]:
                return False, "a sync is running; check again when it has finished"
            self.state["preview_running"] = True
        threading.Thread(target=self._preview, daemon=True).start()
        return True, "checking Nextcloud"

    def _preview(self):
        try:
            res = preview(self.targets()); err = None
        except Exception as e:
            res = None; err = str(e)
        with self.lock:
            self.state["preview_running"] = False; self.state["preview_error"] = err
            if res is not None:
                self.state["preview"] = res

    # --- auto sync
    def set_auto(self, minutes, level="light"):
        minutes = max(0, min(int(minutes), 24 * 60))
        if minutes and minutes < 2:
            minutes = 2
        if level not in LEVELS:
            level = "light"
        with self.lock:
            self.state["auto_minutes"] = minutes; self.state["auto_level"] = level
            self.state["next_auto_at"] = time.time() + 60 * minutes if minutes else None
        return minutes

    def _auto_loop(self):
        while True:
            time.sleep(5)
            with self.lock:
                due = self.state["auto_minutes"] and self.state["next_auto_at"] and time.time() >= self.state["next_auto_at"]
                level = self.state["auto_level"]
                if due:
                    self.state["next_auto_at"] = time.time() + 60 * self.state["auto_minutes"]
            if due:
                self.start(level, source="auto")     # refused (and skipped) while another sync is running


def _fmt(n):
    return f"{n / 1048576:.1f} MiB"


def default_runs():
    """PANEL_RUN, else the chain with the newest status.txt on Nextcloud."""
    if PINNED:
        return [PINNED]
    ch = remote_chains()
    if not ch:
        raise RuntimeError("no <chain>/status.txt on Nextcloud")
    return [ch[0][0]]


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "preview"
    if what not in LEVELS + ("preview", "chains"):
        print(__doc__); return 2
    try:
        if what == "chains":
            for c, t, r in remote_chains():
                print(f"{c:14s} status.txt {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(t))}  {r} -> {os.path.relpath(s3parse.CHAIN_ROOT[c][1], ROOT)}")
            for e in remote_chains.errors:
                print("not listed:", e)
            return 0
        runs = sys.argv[2:] or default_runs()
        if sys.argv[2:]:
            try:
                remote_chains()          # where the named chains live (a chain not yet on the laptop)
            except RuntimeError:
                pass
        print("chains:", " ".join(f"{r} ({s3parse.remote_of(r)})" for r in runs), "| ancestors (top-level files only):", " ".join(ancestors_of(runs)) or "none", flush=True)
        if what == "preview":
            r = preview(runs)
            for l in LEVELS:
                print(f"{l:5s}: on Nextcloud {r[l]['remote_files']} files, {_fmt(r[l]['remote_bytes'])}; "
                      f"to download now {r[l]['todo_files']} files, {_fmt(r[l]['todo_bytes'])}")
            print("segments with meta:", " ".join(r["segments_with_meta"]))
            return 0
    except RuntimeError as e:
        print(f"Nextcloud not reachable: {e}"); return 1
    s = Syncer(); ok, msg = s.start(what, source="cli", runs=runs); print(msg)
    try:
        while s.snapshot()["running"]:
            time.sleep(2); p = s.snapshot()["progress"] or {}
            if "bytes" in p:
                print(f"  {p['files']}/{p['total_files']} files, {_fmt(p['bytes'])} of {_fmt(p['total_bytes'])}, {p['speed'] / 1048576:.1f} MiB/s", flush=True)
    except KeyboardInterrupt:
        s.stop(); print("stopped"); return 130
    st = s.snapshot()
    print(f"finished in {st['last_seconds']} s: " + ("ok" if st["last_ok"] else st["last_error"]) +
          f"; transferred {st['last_transferred']['files']} files, {_fmt(st['last_transferred']['bytes'])} -> {st['dest']}")
    return 0 if st["last_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
