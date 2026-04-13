# Current Implementation Status

This file reflects the current repo state rather than the original proposal checklist.

## Implemented

- MuJoCo world with target, projectile, floor, and optional wall
- `AerodynamicEnv` with egocentric observations and target-relative yaw
- Magnus-force dynamics via `F = k * (omega x v)`
- five-stage curriculum from short-range warmup through tilted gravity
- PPO training with parallel workers, normalization, checkpoints, eval, and curriculum promotion
- `OneShotFlightWrapper` so PPO learns from one throw per episode instead of many counterfactual no-op steps
- interactive `enjoy.py` viewer for manual throws or watching trained policies
- regression tests for the current environment behavior

## Not Yet Implemented

- genetic algorithm baseline in `train_ga.py`
- ablation runs and reporting scripts
- result plotting / benchmark comparison utilities

## Near-Term Next Steps

1. Implement the GA baseline so PPO has a real comparison point.
2. Add a small evaluation script that measures success rate and landing distance for saved checkpoints.
3. Decide whether `EvalCallback` should default to the hardest phase instead of phase 0.
4. Add a short README that points new contributors at `COMMANDS.md`, `train_ppo.py`, and `enjoy.py`.
