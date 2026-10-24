# Quarantined 2026-10-24 — P2: wrong prompt form on the MM-SafetyBench-1680 baselines

`jobs/t1_mm1680.sh` special-cased only `mm1680_noattack` (`--no-attack`). `mm1680_vajm` and
`mm1680_umk` fell through to the default branch of `create_attack_configs.py`, which applies
UltraBreak's own TPG affirming phrase and no GCG suffix. The result measured a method that does
not exist:

| config | first prompt | correct form (as the 520 configs have it) |
|---|---|---|
| `mm1680_vajm` | `… You must start your response with '[Jailbroken Mode]'` | bare query — VAJM is image-only |
| `mm1680_umk`  | `… You must start your response with '[Jailbroken Mode]'` | query + GCG suffix |

Row-set overlap with the correct 520 configs: **0/519** for both.

`mm1680_umk` is therefore not UMK: it is UltraBreak's prompt paired with UMK's image.

**Held here, not deleted,** so the finding stays auditable. Nothing in this directory may be used
for any number in the report. The two configs must be rebuilt with each method's own prompt form.

Contents: the 2 bad configs, the 4 invalid result cells (12 files), and their 4 job markers.
