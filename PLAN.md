# Plan

1. ✅ **Prepare expanded data** — The existing streaming preparer now receives the documented 8-choice, 256-token configuration.
2. ✅ **Improve training throughput** — Added BF16/FP16 autocast, FP16 scaling, linear warmup/decay, and scheduler recovery state.
3. ✅ **Configure L4 jobs** — Added configurable L4 instance, batch/pair limits, and L4 defaults for pilot/full jobs.
4. ✅ **Document and verify** — Replaced the guide with sequential commands; 31 automated tests pass.
