# Simulator patches

These patches are against `iRyougi/sts2-sim` at commit `db844ad` (2026-11-10). The source tree in
`../iRyougi-sts2-sim/` already has them applied.

| File | What |
|---|---|
| `iryougi_local.patch` | Compile fix for .NET SDK 9.0.118 (parentheses only) |
| `iryougi_s0.patch` | Draw-pile position knowledge and clone support needed by the resampling search |
| `iryougi_s3.patch` | `CardSelectCmd.CurrentDeckPurpose`, so the agent knows whether a deck selection is a transform, upgrade or removal |
| `iryougi_perf.patch` | Throughput: hook-listener caching / skip index, exact-size lists, `ModelDb` cache (1.31x measured) |

Apply on a fresh clone in the order listed. The tree in `../iRyougi-sts2-sim/` is the authoritative combined state.
