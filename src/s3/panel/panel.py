#!/usr/bin/env python3
"""Watch panel for the S3 training chains: a local web page built from the log files on this laptop.

Start:   src/s3/panel/run_panel.sh [port]        (default port 8733; open http://127.0.0.1:8733/)
Stop:    Ctrl-C in that terminal (a running sync is stopped with it).

What it does
  - Serves one page (page.html, everything inline: no CDN, no request leaves the laptop except rclone to Nextcloud).
  - Shows one chain at a time (s3a, s3c, s3v3, s3ic, ...). Chains are listed in every Nextcloud root (PANEL_REMOTE;
    default nextcloud:sts2_backup/s3 for the Defect chains and nextcloud:sts2_backup/s3ic for the Ironclad chain), and
    each root has its own local directory (src/server_backup/s3, src/server_backup/s3ic). By default the panel
    follows the newest chain: the one whose status.txt on Nextcloud was written last (one `rclone lsjson` of the
    top-level directories per root, ~8 s, at start and before every sync), or on the laptop if Nextcloud cannot be listed. The "Chain" selector in the page pins another chain (kept in
    state/panel_state.json; "auto" follows the newest again). PANEL_RUN=<chain> pins one at start.
  - Reads src/server_backup/<root>/<chain>/ and <chain>_sNNN/meta/ (s3parse.py; aggregates cached in panel/cache/),
    plus the curve.txt / supervisor.log of the chain's ancestors for the lineage curve (s3a -> s3c -> s3v3; s3ic has none).
  - For s3ic (Ironclad, v0.113) the page adds: the ascension curriculum (stage now, stage per segment, the hook's
    readings and the measured rates against the next stage's thresholds), results by ascension level (per macro update
    and per segment), the mixed potion valuation (Vbar per act from vmix/vbar_sNNN.json, potion_mix counters from the
    actor logs and run records), the evaluation report of the chain hook (act funnel, stalemates), and chain events
    (hook, pruning, memory guard). Value calibration by ascension group and act is shown for every chain that logs it.
  - Interruptible servers (s3ic from 2026-11-19; s3_supervise.sh / s3_resume.sh headers): segments the restarted
    supervisor adopted are shown as "recovered" (partial segments: runs, duration to the interruption, curves), ones it did
    not adopt as "discarded" (gaps; a segment lost with its host and never adopted counts as discarded too); the run
    header has an "Interruptions" card (restarts with downtime and the segment affected, pauses, restores, end-time
    moves, carry backups) and the ribbon marks the downtime. The host comes from status.txt ("S3 run X on HOST"): the
    current one in the header, per segment from the status.txt seen at earlier syncs (state/live_history_*.jsonl) and
    KNOWN_HOSTS in s3parse.py. A status.txt older than 15 min is flagged "server unreachable or chain stopped" and its
    running segment is shown as "no news". A restored chain counts only the run records that reached Nextcloud: the
    panel shifts the later run counts so that the run axis goes on.
  - The page polls the panel every 3 s and redraws when the local files changed; ?theme=dark|light, ?x=time, ?segs=all
    preset the theme, the horizontal axis and the segment filter.
  - "Sync" buttons run `rclone copy --update` from the chain's Nextcloud root in the background (s3sync.py) for the
    chain shown (and the top-level files of its ancestors):
      light = status, curve, logs, evaluations, trainer / actor / server logs, run records (~35 MiB per v3 segment);
      full  = light + fight records (~800 MiB per v3 segment). Weights (*.pt), training chunks (data/) and the off-host
      carry backup (<chain>/carry/, ~0.3 GB per set, for a restore only) never. Syncs before 2026-11-19 did copy carry/:
      src/server_backup/s3ic/s3ic/carry/ (about 1.4 GB on 2026-11-19) is not read by the panel and can be deleted.
    One sync at a time; the page shows progress, the last finish time and the last error. Auto-sync is off until it is
    switched on in the page (the choice is remembered in state/sync_state.json).
  - Nextcloud is only ever read. The training server is never contacted.

API (127.0.0.1 only): GET /api/state (sync, parse and chain state), GET /api/data (state + the chain's payload),
POST /api/run {"run": "auto" | chain}, /api/sync {"level": "light"|"full"}, /api/preview, /api/auto, /api/reparse
(POSTs need the header X-Panel: 1).

Files written by the panel: src/server_backup/s3/** and src/server_backup/s3ic/** (synced copies),
src/s3/panel/cache/** (parsed aggregates, safe to delete), src/s3/panel/state/** (last sync result, chosen
chain, live points seen in status.txt).
Environment: PANEL_REMOTE (Nextcloud roots, comma-separated; default nextcloud:sts2_backup/s3,nextcloud:sts2_backup/s3ic),
PANEL_RUN (pins a chain; default: follow the newest), PANEL_DATA (local directory of the first root; the other roots go
next to it, named after the root).
Python standard library only.
"""
import json, os, re, signal, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s3parse, s3sync   # noqa: E402

SYNC_STATE = os.path.join(s3parse.STATE, "sync_state.json")
PANEL_STATE = os.path.join(s3parse.STATE, "panel_state.json")
KEEP = ("last_level", "last_runs", "last_started_at", "last_finished_at", "last_ok", "last_error", "last_ok_at", "last_seconds", "last_transferred",
        "auto_minutes", "auto_level")


class Handler(BaseHTTPRequestHandler):
    server_version = "s3panel"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype + ("; charset=utf-8" if ctype.startswith("text/") else ""))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        # the page may talk to this server only: nothing can be loaded from or sent to another host
        self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                         "connect-src 'self'; img-src data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def _local(self):
        """Only requests addressed to this machine by name are answered (blocks DNS-rebinding pages)."""
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
        return host in ("127.0.0.1", "localhost", "[::1]")

    def do_GET(self):
        if not self._local():
            return self._send(403, {"error": "local requests only"})
        path = self.path.split("?", 1)[0]
        app = self.server.app
        if path in ("/", "/index.html"):
            try:
                with open(os.path.join(HERE, "page.html"), "rb") as f:
                    return self._send(200, f.read(), "text/html")
            except OSError as e:
                return self._send(500, ("page.html is missing: %s" % e).encode(), "text/plain")
        if path == "/api/state":
            return self._send(200, app.state())
        if path == "/api/data":
            with app.store.lock:
                data = app.store.json
            head = json.dumps(app.state()).encode()
            return self._send(200, head[:-1] + b',"data":' + data + b"}")
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self._local() or self.headers.get("X-Panel") != "1":
            return self._send(403, {"error": "local requests from the panel page only"})
        try:
            n = min(int(self.headers.get("Content-Length") or 0), 4096)
            body = json.loads(self.rfile.read(n) or b"{}")
            if not isinstance(body, dict):
                raise ValueError
        except ValueError:
            return self._send(400, {"error": "bad request body"})
        app = self.server.app; path = self.path.split("?", 1)[0]
        if path == "/api/sync":
            ok, msg = app.syncer.start(str(body.get("level", "light")))
            return self._send(200 if ok else 409, {"ok": ok, "message": msg, **app.state()})
        if path == "/api/preview":
            ok, msg = app.syncer.start_preview()
            return self._send(200 if ok else 409, {"ok": ok, "message": msg, **app.state()})
        if path == "/api/auto":
            try:
                m = app.syncer.set_auto(int(body.get("minutes", 0)), str(body.get("level", "light")))
            except (TypeError, ValueError):
                return self._send(400, {"error": "minutes must be a number"})
            app.save_state()
            return self._send(200, {"ok": True, "message": "auto-sync every %d min" % m if m else "auto-sync off", **app.state()})
        if path == "/api/run":
            ok, msg = app.choose(str(body.get("run", "")))
            return self._send(200 if ok else 400, {"ok": ok, "message": msg, **app.state()})
        if path == "/api/reparse":
            app.store.request()
            return self._send(200, {"ok": True, "message": "reading local files", **app.state()})
        return self._send(404, {"error": "not found"})


class App:
    def __init__(self):
        self.lock = threading.RLock()      # re-entrant: state() -> current_run() -> auto_run() nest
        self.choice = "auto"; self.remote = None; self.remote_at = None; self.remote_error = None
        try:
            with open(PANEL_STATE) as f:
                c = json.load(f).get("run")
            if isinstance(c, str) and re.fullmatch(r"[\w.-]+", c):
                self.choice = c
        except (OSError, ValueError, AttributeError):
            pass
        if s3parse.PINNED:
            self.choice = s3parse.PINNED
        self.store = s3parse.Store(self.current_run())
        self.syncer = s3sync.Syncer()
        self.syncer.targets = self._sync_targets
        threading.Thread(target=self.detect, daemon=True).start()
        try:                                   # the last sync result survives a restart of the panel
            with open(SYNC_STATE) as f:
                old = json.load(f)
            self.syncer.state.update({k: old[k] for k in KEEP if k in old and not k.startswith("auto_")})
            if old.get("auto_minutes"):            # auto-sync is off until switched on in the page; that choice is kept
                self.syncer.set_auto(old["auto_minutes"], old.get("auto_level", "light"))
        except (OSError, ValueError, TypeError):
            pass
        self.syncer.on_done = self._sync_done

    # --- which chain
    def detect(self):
        """Newest chain on Nextcloud (status.txt modification time, all roots); on failure the laptop's newest is used."""
        try:
            ch = s3sync.remote_chains(timeout=90); err = " | ".join(s3sync.remote_chains.errors)[:300] or None
        except Exception as e:
            ch = None; err = str(e)[:300]
        with self.lock:
            if ch is not None:
                self.remote = ch; self.remote_at = time.time()
            self.remote_error = err
        self.store.set_run(self.current_run())

    def auto_run(self):
        with self.lock:
            if self.remote:
                return self.remote[0][0], "Nextcloud"
        return s3parse.newest_local_chain(), "laptop"

    def current_run(self):
        return self.auto_run()[0] if self.choice == "auto" else self.choice

    def _sync_targets(self):
        if self.choice == "auto":
            self.detect()            # a chain started since the last look is picked up by this sync
        return [self.current_run()]

    def chains(self):
        loc = dict(s3parse.local_chains())
        with self.lock:
            rem = {c: t for c, t, _ in (self.remote or [])}
        names = sorted(set(loc) | set(rem), key=lambda n: -max(loc.get(n, 0), rem.get(n, 0)))
        return [{"name": n, "status_at": rem.get(n) or loc.get(n) or None, "local": n in loc, "remote": n in rem,
                 "root": s3parse.remote_of(n).rsplit("/", 1)[-1]} for n in names]

    def choose(self, run):
        if run != "auto" and not (re.fullmatch(r"[\w.-]{1,64}", run) and any(c["name"] == run for c in self.chains())):
            return False, "unknown chain %r" % run
        self.choice = run
        try:
            os.makedirs(s3parse.STATE, exist_ok=True)
            tmp = PANEL_STATE + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"run": run}, f)
            os.replace(tmp, PANEL_STATE)
        except OSError:
            pass
        cur = self.current_run(); self.store.set_run(cur)
        return True, "showing " + cur + (" (auto: newest chain)" if run == "auto" else "")

    def _sync_done(self):
        self.save_state()
        self.store.request()

    def save_state(self):
        s = self.syncer.snapshot()
        try:
            os.makedirs(s3parse.STATE, exist_ok=True)
            tmp = SYNC_STATE + ".tmp"
            with open(tmp, "w") as f:
                json.dump({k: s[k] for k in KEEP}, f)
            os.replace(tmp, SYNC_STATE)
        except OSError:
            pass

    def state(self):
        with self.store.lock:
            parse = dict(self.store.state)
        auto, src = self.auto_run()
        with self.lock:
            rinfo = {"choice": self.choice, "current": self.current_run(), "auto": auto, "auto_source": src,
                     "remote_checked_at": self.remote_at, "remote_error": self.remote_error, "pinned_env": s3parse.PINNED}
        rinfo["chains"] = self.chains()
        return {"now": time.time(), "parse": parse, "sync": self.syncer.snapshot(), "run": rinfo}


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8733
    app = App()
    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError as e:
        print("cannot listen on 127.0.0.1:%d (%s). Is the panel already running? Try another port: run_panel.sh %d" % (port, e, port + 1))
        return 1
    httpd.daemon_threads = True; httpd.app = app

    def stop(*_):
        app.syncer.stop()
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    print("S3 watch panel: http://127.0.0.1:%d/   (chain %s, data %s, remote %s; Ctrl-C stops it)" % (
        port, "%s (PANEL_RUN)" % app.choice if s3parse.PINNED else app.choice if app.choice != "auto" else
        "auto: newest status.txt, now " + app.current_run(), ", ".join(os.path.relpath(l, s3parse.ROOT) for _, l in s3parse.ROOTS), s3sync.REMOTE), flush=True)
    auto = app.syncer.snapshot()["auto_minutes"]
    print("auto-sync: " + ("every %d min (as last set in the page)" % auto if auto else "off (switch it on in the page)"), flush=True)
    httpd.serve_forever()
    httpd.server_close()
    print("panel stopped", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
