# Wiki10M GPU-Subtree Measurement

- Run ID: 20260614_025045_persistent_init_watchdog_fix
- Date: 2026-06-14 Asia/Shanghai
- Build command changed from runbook: yes, added `timeout --signal=INT --kill-after=60s 2700s`; manually interrupted after stable hang evidence.
- Code changes under test: init wait no longer ticks persistent watchdog; root publication marks progress before `P_INIT_DONE`.

## Build

- Exit code: 130, manual SIGINT to the measurement process group.
- Wall time: 5:21.83
- Max RSS: 10,717,600 KB
- `_mem.index` bytes: 0
- `profile.json`: absent
- Last completed stage: root GPU-subtree partition `[rbc/par-spool] total_ms=19926`.
- Observed GPU state while inside first persistent subtree kernel: GPU util 100%, memory util 0%, device memory grew from about 14.2 GiB to 14.8 GiB.
- stdout/stderr after entering kernel: no new stdout lines, stderr empty.

## Verdict

Fail. The init watchdog false positive is removed, but the first persistent subtree kernel does not return on the observed Wiki10M root child (`resident_rows=148098`). The current profile path writes counters only after kernel return, so no `profile.json` or persistent counter snapshot is available from this run.
