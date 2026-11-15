"""One JSON line of throughput numbers for a bench_tr.sh configuration (window from T0 = 240 s to the configured end).
Usage: python -I src/s3/bench_report.py RUN_DIR SYS_FILE TAG N CSERVERS MACRO_PERIOD SECS "CTRAIN_ARGS" "GUARD"
Actor numbers: deltas of the cumulative progress / timing lines between the first line at t >= T0 and the last at t <= end.
"""
import glob, json, os, re, sys

R, SYSF, TAG, N, CS, MP, SECS, CT, WHY = sys.argv[1:10]
N, CS, SECS = int(N), int(CS), int(SECS)
T0 = float(os.environ.get("T0", 240))
END = SECS
num = r"([\d.]+)"
prog = re.compile(r"\] (\d+)s runs (\d+) .*?combat decisions (\d+), sims (\d+) = .*?bridge wait (\d+)%.*?practice decisions (\d+), practice sims (\d+)")
tim = re.compile(r"\] timing (\d+)s: process CPU (\d+)s")


def window(points):
    """points: list of (t, values...) ascending; returns (first at t>=T0, last at t<=END) or None."""
    a = [p for p in points if T0 <= p[0] <= END]
    return (a[0], a[-1]) if len(a) >= 2 and a[-1][0] - a[0][0] > 60 else None


out = {"tag": TAG, "N": N, "cservers": CS, "macro_period": float(MP), "ctrain_args": CT, "guard": WHY or None}
runs_h = sims_s = prac_s = wait_s = span = cpu_cores = dec_s = 0.0
n_ok = 0; per_actor_sims = []; runs_total = 0; errors = 0
for f in sorted(glob.glob(os.path.join(R, "logs", "a*.txt"))):
    P, T = [], []
    for line in open(f, errors="replace"):
        m = prog.search(line)
        if m:
            t, runs, cd, sims, bw, pd, ps = map(int, m.groups()); P.append((t, runs, sims, bw * t / 100.0, ps, cd + pd)); continue
        m = tim.search(line)
        if m: T.append((int(m.group(1)), int(m.group(2))))
        if "error at floor" in line: errors += 1
    w = window(P)
    if not w: continue
    (ta, ra, sa, wa, pa, da), (tb, rb, sb, wb, pb, db) = w
    dt = tb - ta; n_ok += 1; dec_s += (db - da) / dt
    runs_h += 3600 * (rb - ra) / dt; sims_s += (sb - sa) / dt; prac_s += (pb - pa) / dt; wait_s += (wb - wa) / dt
    per_actor_sims.append((sb - sa) / dt)
    wt = window(T)
    if wt: cpu_cores += (wt[1][1] - wt[0][1]) / (wt[1][0] - wt[0][0])
out.update(actors_measured=n_ok, runs_per_h=round(runs_h), live_sims_s=round(sims_s), practice_sims_s=round(prac_s),
           live_sims_s_per_actor=round(sims_s / max(1, n_ok)), all_sims_s_per_actor=round((sims_s + prac_s) / max(1, n_ok)),
           bridge_wait_pct=round(100 * wait_s / max(1, n_ok), 1), actor_cpu_cores=round(cpu_cores / max(1, n_ok), 2),
           run_errors=errors, decisions_s=round(dec_s), records_s_est=round(0.91 * dec_s))
out["runs_finished_total"] = sum(1 for f in glob.glob(os.path.join(R, "runs", "*.jsonl")) for _ in open(f))

srv = re.compile(r"\[s3srv\] (\d+)s combat (\d+) states .*?busy (\d+)%; macro (\d+) rows .*?busy (\d+)%")
busy = {}
for f in sorted(glob.glob(os.path.join(R, "logs", "server*.txt"))) + [os.path.join(R, "logs", "mserver.txt")]:
    pts = []
    for line in open(f, errors="replace") if os.path.exists(f) else []:
        m = srv.search(line)
        if m: pts.append(tuple(map(int, m.groups())))
    a = [p for p in pts if T0 <= p[0] <= END]
    if not a: continue
    k = os.path.basename(f)[:-4]
    if k == "mserver": busy[k] = {"busy": round(sum(p[4] for p in a) / len(a))}
    else:
        rate = (a[-1][1] - a[0][1]) / max(1, a[-1][0] - a[0][0]) if len(a) > 1 else 0
        busy[k] = {"busy": round(sum(p[2] for p in a) / len(a)), "states_s": round(rate)}
out["servers"] = busy
out["combat_states_s"] = sum(v.get("states_s", 0) for v in busy.values())

batch = 512
m = re.search(r"--batch (\d+)", CT)
if m: batch = int(m.group(1))
reuse = 4.0
m = re.search(r"--reuse ([\d.]+)", CT)
if m: reuse = float(m.group(1))
C = []
cf = os.path.join(R, "logs", "ctrain.txt")
for line in open(cf, errors="replace") if os.path.exists(cf) else []:
    if line.startswith("[train] {"):
        d = json.loads(line[8:]); C.append((d["t"], d["step"], d["ingested"]))
    if "start budget" in line: out["ctrain_start"] = line.strip()[:200]
if C:
    a = [c for c in C if T0 <= c[0] <= END + 30]
    last = C[-1]
    out["ctrain_last"] = {"t": last[0], "step": last[1], "ingested": last[2], "reuse_reached": round(last[1] * batch / max(1, last[2]), 2),
                          "budget_gap_samples": round(reuse * last[2] - last[1] * batch)}
    if len(a) >= 2:
        dt = a[-1][0] - a[0][0]
        out["ctrain_steps_s"] = round((a[-1][1] - a[0][1]) / dt, 2)
        out["ctrain_ingest_s"] = round((a[-1][2] - a[0][2]) / dt)
        out["ctrain_window_reuse"] = round((a[-1][1] - a[0][1]) * batch / max(1, a[-1][2] - a[0][2]), 2)
    # capacity: steps/s over intervals after which budget was still left (the trainer was not waiting for data)
    st = dt_ = 0.0
    for x, y in zip(C, C[1:]):
        if T0 <= x[0] and y[0] <= END + 30 and reuse * y[2] - y[1] * batch > batch * 200:
            st += y[1] - x[1]; dt_ += y[0] - x[0]
    if dt_ > 30: out["ctrain_capacity_steps_s"] = round(st / dt_, 1)
    out["ctrain_needed_steps_s"] = round(reuse * out["records_s_est"] / batch, 1)
mt = os.path.join(R, "macro_train_log.jsonl")
its = [json.loads(l) for l in open(mt)] if os.path.exists(mt) else []
out["macro_iters"] = len(its)
out["macro_iters_in_window"] = sum(1 for d in its if T0 <= d.get("t", 0) <= END)

S = []
for line in (open(SYSF) if os.path.exists(SYSF) else []):
    p = line.split(); d = {"t": int(p[0])}
    for kv in p[1:]:
        k, v = kv.split("=", 1); d[k] = v
    S.append(d)
w = [d for d in S if T0 <= d["t"] <= END]
if w:
    g = [tuple(map(float, d["gpu"].split(","))) for d in w]
    out.update(cpu_idle_pct=round(sum(float(d["idle"]) for d in w) / len(w), 1),
               gpu_util_pct=round(sum(x[0] for x in g) / len(g), 1), gpu_mem_mb_max=int(max(x[1] for x in g)),
               ram_mb_max=max(int(d["ram_mb"]) for d in S), load_mean=round(sum(float(d["load"]) for d in w) / len(w), 1))
out["disk_growth_mb_max"] = max(int(d["disk_mb"]) for d in S) if S else None
out["disk_growth_mb_end"] = int(S[-1]["disk_mb"]) if S else None
print(json.dumps(out))
