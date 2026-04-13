# Command Reference

All commands assume you are in the project root:

```bash
cd /Users/dzyl/Desktop/code/ball-throw-ml
source venv/bin/activate
```

## 1. Environment

```bash
# Create the local virtualenv if it does not exist yet
python -m venv venv

# Activate it in each new shell
source venv/bin/activate

# Install runtime dependencies
pip install -r requirements.txt

# Install pytest if you want to run the test suite
pip install pytest
```

## 2. Sanity Checks

```bash
# Full regression pass
python -m pytest test_world.py -v

# Quick pass/fail only
python -m pytest test_world.py -q
```

The current test file covers:

- observation shape and finiteness
- curriculum-specific target, wall, and gravity behavior
- Magnus-force behavior
- episode termination and terminal info
- the one-shot training wrapper

## 3. Exploring The World

`enjoy.py` requires `mjpython` so the MuJoCo GUI runs on the main thread.

```bash
# Interactive viewer
mjpython enjoy.py

# Start in a specific phase
mjpython enjoy.py --phase 2
mjpython enjoy.py --phase 3
mjpython enjoy.py --phase 4

# Scripted demo
mjpython enjoy.py --demo

# Watch a trained PPO policy
mjpython enjoy.py --phase 4 --model models/ppo_full_final.zip
```

Viewer keys:

| key | action |
| --- | ------ |
| `ENTER` | throw with the current slider values |
| `R` / `F5` | reset with a new random episode |
| `0` / `1` / `2` / `3` / `4` | switch curriculum phase |

Slider ranges:

| slider | range | physical meaning |
| ------ | ----- | ---------------- |
| `pitch` | `[-1, 1]` | launch elevation `[-45 deg, 45 deg]` |
| `yaw` | `[-1, 1]` | target-relative yaw offset `[-90 deg, 90 deg]` |
| `thrust` | `[-1, 1]` | initial speed `[2, 22] m/s` |
| `spin` | `[-1, 1]` | angular velocity `[-20, 20] rad/s` |

During training, spin is disabled before phase 2 by default so PPO does not waste exploration on an irrelevant action dimension. The viewer uses the raw environment, so spin works unless you explicitly construct the env with `disable_spin_before_phase > current_phase`.

## 4. Training PPO

```bash
# Smoke test
python train_ppo.py --timesteps 20000 --n-envs 4 --run-name smoke

# Default curriculum run
python train_ppo.py --run-name ppo_full

# Start from a later phase
python train_ppo.py --start-phase 2 --run-name ppo_from_p2
python train_ppo.py --start-phase 4 --run-name ppo_from_p4

# Stricter curriculum promotion
python train_ppo.py --threshold 0.80 --run-name ppo_strict

# Resume from a saved checkpoint
python train_ppo.py \
  --resume-from models/ppo_full_p2_300000_steps.zip \
  --timesteps 300000 \
  --run-name ppo_resume
```

Key CLI flags:

| flag | default | description |
| ---- | ------- | ----------- |
| `--timesteps` | `1_500_000` | total env steps for this run |
| `--n-envs` | `12` | number of parallel workers |
| `--start-phase` | `0` | curriculum phase to start training from |
| `--eval-phase` | `0` | phase used by `EvalCallback` |
| `--seed` | `0` | base RNG seed |
| `--run-name` | `ppo` | prefix for logs and saved models |
| `--threshold` | `0.75` | success-rate threshold for phase promotion |
| `--n-steps` | `2048` | PPO rollout length |
| `--batch-size` | `256` | PPO minibatch size |
| `--lr` | `3e-4` | Adam learning rate |
| `--resume-from` | `None` | checkpoint `.zip` to resume from |

Saved artifacts for `--run-name NAME`:

- `models/NAME_final.zip`
- `models/NAME_final_vecnormalize.pkl`
- `models/NAME_p{PHASE}_best.zip`
- `models/NAME_p{PHASE}_best_vecnormalize.pkl`
- `models/NAME_p{PHASE}_{STEP}_steps.zip`
- `models/NAME_p{PHASE}_{STEP}_steps_vecnormalize.pkl`
- `logs/NAME_*`
- `logs/NAME_eval/`

`VecNormalize` stats are always saved alongside checkpoints/finals because the policy expects normalized observations.

## 5. Current Curriculum

| phase | target distance | wall | gravity | notes |
| ----- | --------------- | ---- | ------- | ----- |
| `0` | `5-10 m` | hidden | standard | shortest-range warmup |
| `1` | `9-20 m` | hidden | standard | no wall yet |
| `2` | `9-20 m` | easy wall | standard | narrow wall near halfway |
| `3` | `9-20 m` | hard wall | standard | wider wall / broader placement |
| `4` | `9-20 m` | hard wall | tilted | gravity tilt up to `+/-15 deg` |

The training env is wrapped in `OneShotFlightWrapper`, so PPO makes one decision per episode: choose a throw, simulate the full flight internally, and learn from the terminal reward.

## 6. TensorBoard

```bash
tensorboard --logdir logs/
```

Useful curves:

- `rollout/ep_rew_mean`
- `curriculum/success_rate`
- `curriculum/phase`
- `curriculum/mean_landing_dist`
- `eval/mean_reward`

## 7. PPO Evaluation In The Viewer

```bash
# Final model
mjpython enjoy.py --phase 4 --model models/ppo_full_final.zip

# Phase-tagged best model
mjpython enjoy.py --phase 3 --model models/ppo_full_p3_best.zip

# Explicit VecNormalize stats if auto-detection is not enough
mjpython enjoy.py \
  --phase 4 \
  --model models/ppo_full_p4_best.zip \
  --vecnormalize models/ppo_full_p4_best_vecnormalize.pkl
```

## 8. GA Status

`train_ga.py` is currently a placeholder. The implemented training pipeline in this repo is the PPO path in `train_ppo.py`.
