# Overcooked experiment (multi-team kitchen)

PGCT-D2 on an Overcooked-style kitchen with **4 agents in 2 hidden teams** (onion team, tomato team),
individual rewards and shared pots. Uses the shared algorithm in `PGCT/pgct_core`; no PGCT code is copied
here. Design, simulator choice and metrics: [`docs/EXPERIMENT_DESIGN.md`](docs/EXPERIMENT_DESIGN.md).
No learning results exist yet.

```
Overcooked/
├── pgct_overcooked/
│   ├── env/kitchen.py, env/layouts.py   simulator (N agents, team recipes, individual rewards)
│   ├── adapter.py                        EnvAdapter + public probe projection (pgct_core interfaces)
│   ├── experiment.py                     JSON config → typed pgct_core configs
│   ├── train.py                          plan + run jobs (methods × seeds)
│   ├── evaluate.py                       fixed-seed evaluation of a checkpoint
│   └── aggregate.py                      identification / learning / cost tables
├── configs/main_4p2t.json               main: 4 agents, 2 hidden teams
├── configs/secondary_2p1t.json          secondary: 2 agents, 1 team (cannot test identification)
├── scripts/run_overcooked.ps1, aggregate.ps1
└── tests/
```

## Setup (Windows PowerShell)

Any Python ≥ 3.9 with `numpy`, `torch`, `pytest` (and optionally `psutil` for memory logging). On the PC
used for the migration: `C:\Users\95101\anaconda3\envs\marl-policy-sharing\python.exe`.

```powershell
$env:PGCT_PYTHON = "C:\Users\95101\anaconda3\envs\marl-policy-sharing\python.exe"
```

The scripts set `PYTHONPATH` to `Overcooked;PGCT` themselves. For direct `python -m` calls from the
repository root:

```powershell
$env:PYTHONPATH = "$PWD\Overcooked;$PWD\PGCT"
```

(Alternatively `pip install -e .` at the repository root installs `pgct_core` and `pgct_overcooked`.)

## Tests

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_tests.ps1 -Only PGCT,Overcooked
```

## Smoke run (all six methods, a few hundred steps, permissive wiring-only gate values)

```powershell
powershell -ExecutionPolicy Bypass -File Overcooked\scripts\run_overcooked.ps1 -Name oc_smoke -Seeds 1 -Smoke
```

## Main comparison

Six methods × three seeds, 5M steps each (about 1.5 days per job at ≈37 steps/s; run 2–4 jobs at a time
depending on CPU cores). Start long runs yourself in a separate PowerShell window:

```powershell
powershell -ExecutionPolicy Bypass -Command "& '.\Overcooked\scripts\run_overcooked.ps1' -Name oc_main_v1 -Seeds 1,2,3 -MaxConcurrent 3"
```

Plan only (writes `Overcooked\runs\<name>\plan.json`, trains nothing):

```powershell
powershell -ExecutionPolicy Bypass -Command "& '.\Overcooked\scripts\run_overcooked.ps1' -Name oc_main_v1 -Seeds 1,2,3 -PlanOnly"
```

Single job without the queue:

```powershell
python -m pgct_overcooked.train --config main_4p2t.json --methods pgct_d2 --seeds 1 --name oc_main_v1
```

Rerunning the same command skips finished jobs (`final.pt`) and restarts unfinished ones from scratch.
A plan name with a different configuration is refused.

Ablations are config overrides, e.g.:

```powershell
python -m pgct_overcooked.train --config main_4p2t.json --methods pgct_d2 --seeds 1 --name oc_pooled --set gate.distance_mode=pooled
python -m pgct_overcooked.train --config main_4p2t.json --methods ippo pgct_d2 --seeds 1 --name oc_noshape --set kitchen.pot_reward=0 kitchen.soup_pickup_reward=0
python -m pgct_overcooked.train --config secondary_2p1t.json --methods ippo all pgct_d2 --seeds 1 --name oc_2p
```

## Results

```powershell
powershell -ExecutionPolicy Bypass -File Overcooked\scripts\aggregate.ps1 -Name oc_main_v1
python -m pgct_overcooked.evaluate --job Overcooked\runs\oc_main_v1\pgct_d2_s1 --episodes 20
```

Per job folder: `config.json` (every setting), `metrics.jsonl` (per update: PPO, episode returns and
events, `max_gae_*`, `ac_v2_*`, cost and timing), `gate.jsonl` (contact, approval, maturity, rounds, EMA,
pair comparisons, donors), `eval.jsonl`, `checkpoint_*.pt`, `final.pt` (agents, optimizers, gate state,
RNG states), `status.json`. `aggregate` writes `summary.md`, `summary.json` and `jobs.csv` to the plan
folder.

## Methods

`ippo`, `oracle` (true teammates; labels used by this baseline only), `all`, `pgct_d2`, `pgct_id_only`
(gate without critic access), `random_gate` (PGCT's donor count per receiver, random donors among contacts).
See `docs/ALGORITHM.md` at the repository root.
