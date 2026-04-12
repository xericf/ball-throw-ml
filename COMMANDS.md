# Command Reference

All commands assume you are in the project root:

```
cd /Users/ericfu/School/CPSC440/project
source proj_env/bin/activate
```

---

## 1. Environment

```bash
# needa make proj_env with python -m venv proj_env
# Activate venv (every new shell)
source proj_env/bin/activate

# One-time install (only if proj_env is missing or requirements changed)
python -m venv proj_env
source proj_env/bin/activate
pip install -r requirements.txt
```

---

## 2. Sanity checks

```bash
# Regression tests — 22 tests covering all 3 curriculum phases + Magnus
python -m pytest test_world.py -v

# Quick pass/fail only
python -m pytest test_world.py -q
```

---

## 3. Exploring the world (enjoy.py)

Requires `mjpython` (shipped with MuJoCo) so the GUI runs on the main thread.

```bash
# Interactive viewer — use GUI sliders, ENTER throws, R resets, 1/2/3 switches phase
mjpython enjoy.py                # phase 1 (no wall)
mjpython enjoy.py --phase 2      # phase 2 (wall present)
mjpython enjoy.py --phase 3      # phase 3 (wall + tilted gravity)

# Scripted demo — cycles through straight / +spin / -spin / high arc / hard throw
mjpython enjoy.py --demo
mjpython enjoy.py --phase 3 --demo
```

Viewer keybindings:

|         key | action                                              |
| ----------: | --------------------------------------------------- |
|     `ENTER` | throw with current slider values                    |
|  `R` / `F5` | reset scene (new target/wall/gravity randomisation) |
| `1`/`2`/`3` | switch curriculum phase                             |

Slider ranges (from `action_space`):

| slider | range     | physical meaning                   |
| ------ | --------- | ---------------------------------- |
| pitch  | `[-1, 1]` | launch elevation `[-30°, 30°]`     |
| yaw    | `[-1, 1]` | azimuth `[-180°, 180°]`            |
| thrust | `[-1, 1]` | initial speed `[2, 22] m/s`        |
| spin   | `[-1, 1]` | angular velocity `[-20, 20] rad/s` |

---

## 4. Training PPO

```bash
# Fast smoke test — a few thousand steps, confirms wiring
python train_ppo.py --timesteps 8000 --n-envs 8 --run-name smoke

# Full curriculum run — 1.5M steps, 8 parallel envs
python train_ppo.py --timesteps 1500000 --n-envs 12 --run-name ppo_full

# Skip ahead: start directly at phase 2 or 3 (bypass earlier phases)
python train_ppo.py --start-phase 2 --run-name ppo_from_p2
python train_ppo.py --start-phase 3 --run-name ppo_from_p3

# Tune the curriculum promotion threshold (default 0.70)
python train_ppo.py --threshold 0.80 --run-name ppo_strict

# Custom seed / longer run
python train_ppo.py --timesteps 3000000 --seed 42 --run-name ppo_long
```

All `train_ppo.py` flags:

| flag            | default     | description                                 |
| --------------- | ----------- | ------------------------------------------- |
| `--timesteps`   | `1_500_000` | total env steps                             |
| `--n-envs`      | `8`         | parallel `SubprocVecEnv` workers            |
| `--start-phase` | `1`         | curriculum starting phase (1, 2, or 3)      |
| `--eval-phase`  | `3`         | phase used by `EvalCallback`                |
| `--seed`        | `0`         | base RNG seed                               |
| `--run-name`    | `ppo`       | prefix for logs/ and models/ artifacts      |
| `--threshold`   | `0.70`      | curriculum promotion success-rate threshold |
| `--n-steps`     | `2048`      | PPO rollout length                          |
| `--batch-size`  | `256`       | PPO minibatch size                          |
| `--lr`          | `3e-4`      | Adam learning rate                          |

Artifacts produced per run (with `--run-name NAME`):

- `models/NAME_final.zip` — final policy
- `models/best_model.zip` — best policy from eval rollouts (overwritten across runs; rename if you want to keep)
- `models/NAME_<step>_steps.zip` — periodic checkpoints
- `logs/NAME_<n>/` — TensorBoard event files
- `logs/NAME_eval/` — eval callback logs

---

## 5. TensorBoard

```bash
tensorboard --logdir logs/
# then open http://localhost:6006
```

Curves to watch:

- `rollout/ep_rew_mean` — should rise (less negative)
- `curriculum/success_rate` — rolling fraction of `dist < 1m` episodes
- `curriculum/phase` — step function: 1 → 2 → 3
- `curriculum/mean_landing_dist` — rolling L2 error in metres
- `eval/mean_reward` — held-out phase-3 evaluation

---

## 6. Visualising a trained policy

```bash
# Best model (from EvalCallback)
mjpython enjoy.py --phase 3 --model models/best_model.zip

# A specific run's final model
mjpython enjoy.py --phase 3 --model models/ppo_full_final.zip

# Run more/fewer episodes
mjpython enjoy.py --phase 2 --model models/ppo_full_final.zip --episodes 50
```

---

## 7. Cleaning up

```bash
# Remove a specific run's artifacts
rm -rf logs/smoke_* models/smoke_*

# Nuke all logs and models (careful — destructive)
rm -rf logs/* models/*
```
