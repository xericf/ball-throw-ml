"""evaluate.py — policy evaluation script for ball-throw-ml.

Runs a trained PPO or GA policy through each curriculum phase and reports
success rate, mean landing distance, and mean reward. Designed to produce
the comparison table and raw data for plots.

Usage:
    python evaluate.py --model models/ppo_full_final.zip --algo ppo
    python evaluate.py --model models/ga_full_final.pt   --algo ga

    # Specific phases, more episodes, save JSON for plotting:
    python evaluate.py --model models/ppo_full_final.zip --algo ppo \\
        --phases 2 3 4 --n-episodes 500 --output results/ppo_eval.json

    # PPO with explicit VecNormalize path (auto-detected if omitted):
    python evaluate.py --model models/ppo_full_p4_best.zip --algo ppo \\
        --vecnormalize models/ppo_full_p4_best_vecnormalize.pkl
"""

import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from envs.aerodynamic_env import AerodynamicEnv, OneShotFlightWrapper

PHASE_NAMES = {
    0: "Phase 0: Short-range, no wall",
    1: "Phase 1: Long-range, no wall",
    2: "Phase 2: Easy wall",
    3: "Phase 3: Hard wall",
    4: "Phase 4: Hard wall + gravity tilt",
}


# ─── Per-algorithm eval functions ─────────────────────────────────────────────


def _eval_ppo(model_path: str, vecnorm_path: str | None, phase: int, n_episodes: int, seed: int) -> dict:
    """Evaluate a PPO checkpoint on one curriculum phase."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

    model = PPO.load(model_path, device="cpu")

    def _make():
        env = AerodynamicEnv(curriculum_phase=phase)
        env.reset(seed=seed)
        return OneShotFlightWrapper(env)

    raw = VecMonitor(DummyVecEnv([_make]))

    if vecnorm_path and os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, raw)
        env.training = False
        env.norm_reward = False
    else:
        if vecnorm_path:
            print(f"\n  Warning: VecNormalize stats not found at {vecnorm_path!r}.")
            print("  PPO performance will be degraded without obs normalisation.\n")
        env = VecNormalize(raw, norm_obs=False, norm_reward=False, training=False)

    rewards, dists, successes, wall_hits = [], [], [], []
    obs = env.reset()

    for _ in range(n_episodes):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = env.step(action)
        # OneShotFlightWrapper always terminates in 1 step; DummyVecEnv
        # auto-resets and returns new obs, terminal info is in infos[0].
        info = infos[0]
        rewards.append(float(reward[0]))
        dists.append(float(info["landing_dist"]))
        successes.append(float(info["success"]))
        wall_hits.append(float(info.get("hit_wall", False)))

    env.close()
    return _summarise(rewards, dists, successes, wall_hits)


def _eval_ga(model_path: str, phase: int, n_episodes: int, seed: int) -> dict:
    """Evaluate a GA checkpoint on one curriculum phase."""
    import torch
    from train_ga import PolicyNet

    net = PolicyNet()
    net.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    net.eval()

    env = OneShotFlightWrapper(AerodynamicEnv(curriculum_phase=phase))
    rewards, dists, successes, wall_hits = [], [], [], []

    with torch.no_grad():
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed + ep)
            obs_t = torch.from_numpy(np.array(obs, dtype=np.float32)).unsqueeze(0)
            action = net(obs_t).squeeze(0).numpy()
            _, reward, _, _, info = env.step(action)
            rewards.append(float(reward))
            dists.append(float(info["landing_dist"]))
            successes.append(float(info["success"]))
            wall_hits.append(float(info.get("hit_wall", False)))

    env.close()
    return _summarise(rewards, dists, successes, wall_hits)


def _summarise(rewards, dists, successes, wall_hits) -> dict:
    rewards = np.array(rewards, dtype=np.float32)
    dists = np.array(dists, dtype=np.float32)
    successes = np.array(successes, dtype=np.float32)
    wall_hits = np.array(wall_hits, dtype=np.float32)
    n = len(rewards)
    return {
        "n_episodes": n,
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std()),
        "mean_landing_dist": float(dists.mean()),
        "std_landing_dist": float(dists.std()),
        "median_landing_dist": float(np.median(dists)),
        "success_rate": float(successes.mean()),
        "success_rate_se": float(successes.std() / np.sqrt(n)),
        "wall_hit_rate": float(wall_hits.mean()),
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a trained PPO or GA policy on each curriculum phase.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1] if "Usage:" in __doc__ else "",
    )
    p.add_argument("--model", required=True, help="Model file (.zip for PPO, .pt for GA)")
    p.add_argument("--algo", required=True, choices=["ppo", "ga"])
    p.add_argument(
        "--vecnormalize",
        default=None,
        help="VecNormalize stats .pkl for PPO. Auto-detected from model path if omitted.",
    )
    p.add_argument(
        "--phases",
        nargs="+",
        type=int,
        default=list(range(5)),
        metavar="N",
        help="Phases to evaluate (default: 0 1 2 3 4)",
    )
    p.add_argument(
        "--n-episodes",
        type=int,
        default=200,
        help="Episodes per phase (default: 200)",
    )
    p.add_argument("--seed", type=int, default=42, help="Base RNG seed (default: 42)")
    p.add_argument(
        "--output",
        default=None,
        help="Save full results as JSON to this path (for plotting)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.model):
        print(f"Error: model file not found: {args.model}")
        sys.exit(1)

    vecnorm_path = args.vecnormalize
    if args.algo == "ppo" and vecnorm_path is None:
        vecnorm_path = re.sub(r"\.zip$", "_vecnormalize.pkl", args.model)

    print(f"Algorithm  : {args.algo.upper()}")
    print(f"Model      : {args.model}")
    if args.algo == "ppo":
        status = "found" if (vecnorm_path and os.path.exists(vecnorm_path)) else "MISSING"
        print(f"VecNorm    : {vecnorm_path}  [{status}]")
    print(f"Phases     : {args.phases}")
    print(f"Episodes   : {args.n_episodes} per phase")
    print(f"Seed       : {args.seed}")
    print()

    results: dict[int, dict] = {}
    for phase in args.phases:
        label = PHASE_NAMES.get(phase, f"Phase {phase}")
        print(f"  {label} ...", end="", flush=True)

        if args.algo == "ppo":
            metrics = _eval_ppo(args.model, vecnorm_path, phase, args.n_episodes, args.seed)
        else:
            metrics = _eval_ga(args.model, phase, args.n_episodes, args.seed)

        results[phase] = metrics
        sr = metrics["success_rate"] * 100
        se = metrics["success_rate_se"] * 100
        print(
            f"  success={sr:.1f}±{se:.1f}%"
            f"  dist={metrics['mean_landing_dist']:.2f}m"
            f"  reward={metrics['mean_reward']:.2f}"
        )

    # ── Results table ─────────────────────────────────────────────────────────
    col_w = 34
    print()
    print(
        f"{'Phase':<{col_w}}  {'Success %':>10}  {'±SE':>5}  "
        f"{'Dist (m)':>9}  {'Reward':>8}  {'Wall Hit %':>10}"
    )
    print("─" * (col_w + 50))
    for phase in args.phases:
        m = results[phase]
        name = PHASE_NAMES.get(phase, f"Phase {phase}")
        wh = f"{m['wall_hit_rate']*100:.1f}" if m["wall_hit_rate"] > 0 else "  —"
        print(
            f"{name:<{col_w}}  "
            f"{m['success_rate']*100:>10.1f}  "
            f"{m['success_rate_se']*100:>5.1f}  "
            f"{m['mean_landing_dist']:>9.3f}  "
            f"{m['mean_reward']:>8.2f}  "
            f"{wh:>10}"
        )
    print()

    # ── JSON output ───────────────────────────────────────────────────────────
    if args.output:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        payload = {
            "algo": args.algo,
            "model": args.model,
            "n_episodes": args.n_episodes,
            "seed": args.seed,
            "phases": {str(k): v for k, v in results.items()},
        }
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
