# Baseline execution plan

1. Validate the exact 22-method registry and eight frozen dataset identities:
   `python benchmark/run_baseline.py --validate`.
2. Run the label-isolation/data-contract smoke:
   `python benchmark/run_baseline.py --smoke`.
3. Inspect a single dry-run plan, for example:
   `python benchmark/run_baseline.py --method DOMINANT --dataset Amazon --seed 0`.
4. Review `reports/BASELINE_TRANSDUCTIVE_PROTOCOL_CLOSURE_20260809.md`, resolve
   every native bridge, environment, source, and native sanity blocker, then
   explicitly set `execute_enabled` only for methods that pass those gates.
5. Only after user approval, run seed 0 with an explicit `--execute`.

`benchmark/run_all_baselines_seed0.sh` is dry-run by default. It does not train
unless invoked with `--execute`; the current registry intentionally blocks all
formal execution until method-level native reproduction sanity is recorded.

The main protocol is `transductive`. `strict_oof` is a secondary robustness
option and is refused unless the selected method has a separately audited
strict adapter. CoReGAD's internal cross-fitting is unchanged and is never
injected into baseline training.
