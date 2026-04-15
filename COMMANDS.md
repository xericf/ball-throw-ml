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
| `--eval-phase` | `4` | phase used by `EvalCallback`; defaults to the hardest phase |
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

## 6. Training GA

The GA trains a neuroevolution policy: a 7→64→64→4 MLP whose weights are evolved
directly, with no gradient. Each genome is a flat array of ~4 932 floats. Fitness is
mean reward over `--n-eval-episodes` rollouts. The same five-phase curriculum as PPO
is used, with promotion gated on the **elite** success rate (top-K genomes only).

```bash
# Smoke test — tiny population, fast
python train_ga.py --pop-size 20 --num-generations 10 --n-eval-episodes 5 --n-workers 4 --run-name smoke_ga

# Default curriculum run
python train_ga.py --run-name ga_full

# Start from a later phase
python train_ga.py --start-phase 2 --run-name ga_from_p2

# Resume from a population checkpoint
python train_ga.py --resume-from models/ga_full_p1_gen200_checkpoint.npy --run-name ga_resume

# Enable W&B logging
python train_ga.py --run-name ga_full --wandb --wandb-project ball-throw-ml
```

Key CLI flags:

| flag | default | description |
| ---- | ------- | ----------- |
| `--pop-size` | `100` | number of genomes per generation |
| `--num-generations` | `500` | total generations |
| `--n-eval-episodes` | `30` | rollouts per genome per generation |
| `--num-parents-mating` | `20` | elites kept unchanged each generation |
| `--mutation-percent-genes` | `10.0` | % of weights perturbed per offspring |
| `--mutation-std` | `0.1` | Gaussian noise std for mutation |
| `--start-phase` | `0` | curriculum phase to begin from |
| `--n-workers` | `8` | parallel worker processes |
| `--run-name` | `ga` | prefix for logs and saved models |
| `--seed` | `0` | RNG seed |
| `--threshold` | `0.75` | elite success-rate threshold for phase promotion |
| `--checkpoint-freq` | `50` | save population snapshot every N generations |
| `--resume-from` | `None` | population checkpoint `.npy` to resume from |
| `--crossover` | off | enable uniform crossover (off by default — hurts neuroevolution) |
| `--wandb` | off | enable Weights & Biases logging |
| `--wandb-project` | `ball-throw-ml` | W&B project name |
| `--wandb-entity` | `None` | W&B entity / team |

Saved artifacts for `--run-name NAME`:

- `models/NAME_final.pt` — PyTorch state dict of the best genome overall
- `models/NAME_final_genome.npy` — raw genome array (for resuming or analysis)
- `models/NAME_p{PHASE}_best.pt` — best genome seen in each phase
- `models/NAME_p{PHASE}_best_genome.npy`
- `models/NAME_p{PHASE}_gen{N}_checkpoint.npy` — full population snapshot (for resuming)
- `logs/NAME/` — TensorBoard event files

To load a saved GA policy in Python:

```python
import torch
from train_ga import PolicyNet

net = PolicyNet()
net.load_state_dict(torch.load("models/ga_full_final.pt", weights_only=True))
net.eval()
# obs is a (7,) float32 numpy array
action = net(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).detach().numpy()
```

### GA design notes

**No crossover by default.** Uniform crossover between neural net parents produces
incoherent weights (competing conventions problem) — offspring consistently score ~0%
even when both parents are excellent. Pure clone-and-mutate is used instead.
Pass `--crossover` to re-enable for ablation.

**Elite-only curriculum.** The phase-promotion buffer tracks only the top-K elites'
success rates, not the full population mean. Offspring start with random-ish weights
relative to each other, dragging the population mean far below the elites; using the
full mean as the promotion signal would permanently stall the curriculum.

## 7. TensorBoard

```bash
tensorboard --logdir logs/
```

### PPO curves

| tag | description |
| --- | ----------- |
| `rollout/ep_rew_mean` | mean episode reward across workers |
| `curriculum/phase` | current curriculum phase |
| `curriculum/success_rate` | rolling success rate (all envs) |
| `curriculum/mean_landing_dist` | mean landing distance in meters |
| `eval/mean_reward` | EvalCallback reward on phase 4 |

### GA curves

| tag | description |
| --- | ----------- |
| `ga/mean_reward` | mean reward across all genomes this generation |
| `ga/best_reward` | reward of the single best genome this generation |
| `ga/mean_success_rate` | mean success rate across the full population |
| `ga/elite_success_rate` | mean success rate of the top-K elites only |
| `ga/best_success_rate` | success rate of the single best genome |
| `ga/mean_landing_dist` | mean landing distance across all genomes |
| `ga/reward_std` | std of rewards across the population (diversity signal) |
| `curriculum/phase` | current curriculum phase |
| `curriculum/elite_success_rate` | same as `ga/elite_success_rate` (curriculum context) |
| `curriculum/rolling_success` | trailing mean of elite success rates in the promotion buffer |
| `perf/eval_time_s` | wall-clock seconds spent on fitness evaluation this generation |

`ga/elite_success_rate` and `curriculum/rolling_success` are the most useful pair:
the former is the raw per-generation elite reading, the latter is the smoothed signal
the curriculum actually uses to decide phase promotion. A healthy run shows
`elite_success_rate` rising first, with `rolling_success` trailing behind and
eventually crossing `threshold` (default 0.75) to trigger promotion.

## 8. Evaluating Trained Policies

`evaluate.py` runs a saved checkpoint through each curriculum phase and prints a
comparison table (success rate, landing distance, reward). Use `--output` to save
JSON for plotting.

```bash
# Evaluate PPO across all phases (VecNormalize stats auto-detected)
python evaluate.py --model models/ppo_full_final.zip --algo ppo

# Evaluate GA across all phases
python evaluate.py --model models/ga_full_final.pt --algo ga

# More episodes, specific phases, save JSON
python evaluate.py --model models/ppo_full_final.zip --algo ppo \
    --phases 2 3 4 --n-episodes 500 --output results/ppo_eval.json

python evaluate.py --model models/ga_full_final.pt --algo ga \
    --phases 2 3 4 --n-episodes 500 --output results/ga_eval.json

# Explicit VecNormalize path (if auto-detection fails)
python evaluate.py --model models/ppo_full_p4_best.zip --algo ppo \
    --vecnormalize models/ppo_full_p4_best_vecnormalize.pkl
```

Key CLI flags:

| flag | default | description |
| ---- | ------- | ----------- |
| `--model` | *(required)* | `.zip` for PPO, `.pt` for GA |
| `--algo` | *(required)* | `ppo` or `ga` |
| `--vecnormalize` | auto-detected | VecNormalize `.pkl` for PPO |
| `--phases` | `0 1 2 3 4` | curriculum phases to evaluate |
| `--n-episodes` | `200` | episodes per phase |
| `--seed` | `42` | base RNG seed |
| `--output` | `None` | save results as JSON to this path |

The JSON output has this shape (useful for plotting):
```json
{
  "algo": "ppo",
  "model": "models/ppo_full_final.zip",
  "n_episodes": 200,
  "seed": 42,
  "phases": {
    "0": { "success_rate": 0.95, "mean_landing_dist": 0.41, "mean_reward": 4.12, ... },
    "4": { "success_rate": 0.82, "mean_landing_dist": 0.63, "mean_reward": 3.71, ... }
  }
}
```

## 9. PPO Viewer

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
