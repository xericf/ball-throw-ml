"""Genetic algorithm training via neuroevolution for ball-throw-ml.

Evolves the weights of a 7→64→64→4 MLP policy using rank-based elitism,
uniform crossover, and Gaussian mutation. Mirrors train_ppo.py structure:
same curriculum phases, TensorBoard metrics, checkpoint saving, and CLI interface.

Usage:
    python train_ga.py --run-name ga_full
    python train_ga.py --pop-size 50 --num-generations 100 --n-eval-episodes 5 --run-name smoke
    python train_ga.py --resume-from models/ga_p1_gen200_checkpoint.npy --start-phase 1

The task is a contextual bandit after OneShotFlightWrapper: the agent sees a
7-dim observation once per episode and outputs a 4-dim action. The ball then
flies under MuJoCo physics and returns a terminal reward. No gradient needed.
"""

import argparse
import multiprocessing as mp
import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from envs.aerodynamic_env import AerodynamicEnv, OneShotFlightWrapper

LOG_DIR = "logs/"
MODEL_DIR = "models/"
MAX_CURRICULUM_PHASE = 4


# ─── Policy Network ────────────────────────────────────────────────────────────


class PolicyNet(nn.Module):
    """7 → 64 → 64 → 4 MLP policy. Same architecture as PPO MlpPolicy.

    Tanh activations match SB3's default and keep outputs in [-1, 1]
    without any explicit clamping — the action space bounds are satisfied
    by construction.
    """

    def __init__(self, obs_dim: int = 7, act_dim: int = 4, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim),
            nn.Tanh(),  # Output in [-1, 1] matching action space bounds
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def set_params(net: nn.Module, genome: np.ndarray) -> None:
    """Load a flat genome array into network parameter tensors in-place."""
    idx = 0
    with torch.no_grad():
        for p in net.parameters():
            n = p.numel()
            p.copy_(
                torch.from_numpy(genome[idx : idx + n].reshape(p.shape)).float()
            )
            idx += n


def get_params(net: nn.Module) -> np.ndarray:
    """Flatten network parameters into a genome array."""
    return np.concatenate(
        [p.detach().cpu().numpy().ravel() for p in net.parameters()]
    )


# ─── Fitness Evaluation ────────────────────────────────────────────────────────

# Persistent per-worker state — lives in each subprocess, not the main process.
# Avoids re-initialising MuJoCo on every genome evaluation.
_worker_env: "OneShotFlightWrapper | None" = None


def _evaluate_single(args: tuple) -> tuple:
    """Worker: evaluate one genome for n_episodes, return (reward, success_rate, mean_dist, wall_hit_rate).

    Runs in a subprocess spawned by multiprocessing.Pool. Each worker owns its
    own MuJoCo environment instance — no shared state, spawn-safe.

    The environment is created once per worker on the first call and reused
    thereafter. reset() calls mujoco.mj_resetData() in-place, so no state
    bleeds between genome evaluations. On a curriculum phase change,
    set_curriculum_phase() updates the live instance without recreating it.
    """
    global _worker_env
    genome, n_episodes, phase, base_seed = args

    if _worker_env is None:
        _worker_env = OneShotFlightWrapper(
            AerodynamicEnv(curriculum_phase=phase, disable_spin_before_phase=2)
        )
    elif _worker_env.env.curriculum_phase != phase:
        # Phase changed — update the live instance; no MuJoCo reinit needed.
        _worker_env.env.set_curriculum_phase(phase)

    net = PolicyNet()
    set_params(net, genome)
    net.eval()

    total_reward = 0.0
    success_count = 0
    wall_hit_count = 0
    landing_dists = []

    with torch.no_grad():
        for ep in range(n_episodes):
            seed = (base_seed + ep) if base_seed is not None else None
            obs, _ = _worker_env.reset(seed=seed)
            obs_t = torch.from_numpy(np.array(obs, dtype=np.float32)).unsqueeze(0)
            action = net(obs_t).squeeze(0).numpy()
            _, reward, _, _, info = _worker_env.step(action)
            total_reward += float(reward)
            success_count += int(info.get("success", False))
            wall_hit_count += int(info.get("hit_wall", False))
            landing_dists.append(float(info.get("landing_dist", 999.0)))

    # No env.close() — _worker_env persists for the lifetime of this worker process.
    mean_reward = total_reward / n_episodes
    success_rate = success_count / n_episodes
    mean_dist = float(np.mean(landing_dists))
    wall_hit_rate = wall_hit_count / n_episodes
    return mean_reward, success_rate, mean_dist, wall_hit_rate


def evaluate_population(
    population: np.ndarray,
    n_episodes: int,
    phase: int,
    base_seed: int,
    n_workers: int,
    pool=None,
) -> tuple:
    """Evaluate all genomes in parallel. Returns (rewards, success_rates, mean_dists, wall_hit_rates).

    If *pool* is provided (a persistent multiprocessing.Pool), it is reused
    directly and not closed here. Otherwise a fresh pool is created and torn
    down each call (original behaviour, kept for standalone use).
    """
    args = [
        (genome, n_episodes, phase, base_seed + i * 1000)
        for i, genome in enumerate(population)
    ]
    if pool is not None:
        results = pool.map(_evaluate_single, args)
    else:
        ctx = mp.get_context("spawn")  # spawn is MuJoCo-safe (matches train_ppo.py)
        with ctx.Pool(processes=n_workers) as p:
            results = p.map(_evaluate_single, args)

    rewards = np.array([r[0] for r in results], dtype=np.float32)
    success_rates = np.array([r[1] for r in results], dtype=np.float32)
    mean_dists = np.array([r[2] for r in results], dtype=np.float32)
    wall_hit_rates = np.array([r[3] for r in results], dtype=np.float32)
    return rewards, success_rates, mean_dists, wall_hit_rates


# ─── GA Operators ──────────────────────────────────────────────────────────────


def generate_offspring(
    parents: np.ndarray,
    n_offspring: int,
    mutation_std: float,
    mutation_frac: float,
    rng: np.random.Generator,
    crossover: bool = False,
) -> np.ndarray:
    """Produce n_offspring from parents via mutation (and optionally crossover).

    Crossover is OFF by default. Uniform crossover between neural network parents
    produces incoherent weights (competing conventions problem) — offspring
    consistently score 0% even when both parents are excellent. Pure clone+mutate
    propagates the parent's structure intact and lets mutation explore locally.

    Set crossover=True only if you have a specific reason (e.g. ablation study).
    """
    n_parents, genome_len = parents.shape
    offspring = np.empty((n_offspring, genome_len), dtype=np.float32)

    for i in range(n_offspring):
        if crossover and n_parents > 1:
            # Uniform crossover between two parents (kept for ablation)
            p1, p2 = rng.integers(0, n_parents, size=2)
            if p1 == p2:
                p2 = (p1 + 1) % n_parents
            mask = rng.random(genome_len) < 0.5
            child = np.where(mask, parents[p1], parents[p2]).astype(np.float32)
        else:
            # Pure clone: pick one parent, inherit weights intact
            p1 = rng.integers(0, n_parents)
            child = parents[p1].copy()

        # Gaussian mutation on a random fraction of genes
        n_mut = max(1, int(mutation_frac * genome_len))
        mut_idx = rng.choice(genome_len, size=n_mut, replace=False)
        child[mut_idx] += rng.standard_normal(n_mut).astype(np.float32) * mutation_std

        offspring[i] = child

    return offspring


# ─── Curriculum ────────────────────────────────────────────────────────────────


class CurriculumState:
    """Tracks rolling success rate across population evaluations.

    Mirrors CurriculumManagerCallback from train_ppo.py — same threshold (0.75)
    and window (200) defaults, same per-phase logic.
    """

    def __init__(
        self, start_phase: int = 0, threshold: float = 0.75, window: int = 200
    ):
        self.phase = start_phase
        self.threshold = threshold
        self._buf: deque = deque(maxlen=window)

    def update(self, success_rates: np.ndarray) -> None:
        for sr in success_rates:
            self._buf.append(float(sr))

    def mean_success(self) -> float:
        return float(np.mean(self._buf)) if self._buf else 0.0

    def should_promote(self, min_samples: int) -> bool:
        return (
            len(self._buf) >= min_samples
            and self.mean_success() >= self.threshold
            and self.phase < MAX_CURRICULUM_PHASE
        )

    def promote(self) -> None:
        self.phase = min(self.phase + 1, MAX_CURRICULUM_PHASE)
        self._buf.clear()


# ─── Checkpoint I/O ────────────────────────────────────────────────────────────


def save_checkpoint(
    population: np.ndarray, generation: int, phase: int, run_name: str
) -> str:
    path = os.path.join(
        MODEL_DIR, f"{run_name}_p{phase}_gen{generation}_checkpoint.npy"
    )
    np.save(path, {"population": population, "generation": generation, "phase": phase})
    return path


def save_best(
    genome: np.ndarray, phase: int, run_name: str, suffix: str = "best"
) -> None:
    """Save a genome both as a PyTorch state dict (.pt) and raw array (.npy)."""
    net = PolicyNet()
    set_params(net, genome)
    torch.save(
        net.state_dict(),
        os.path.join(MODEL_DIR, f"{run_name}_p{phase}_{suffix}.pt"),
    )
    np.save(
        os.path.join(MODEL_DIR, f"{run_name}_p{phase}_{suffix}_genome.npy"), genome
    )


# ─── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Neuroevolution (GA) training for ball-throw-ml. "
            "Evolves weights of a 7→64→64→4 MLP policy with curriculum learning."
        )
    )
    p.add_argument(
        "--pop-size", type=int, default=100, help="Population size (default: 100)"
    )
    p.add_argument(
        "--num-generations",
        type=int,
        default=500,
        help="Total GA generations (default: 500)",
    )
    p.add_argument(
        "--n-eval-episodes",
        type=int,
        default=30,
        help="Episodes per genome per generation (default: 30)",
    )
    p.add_argument(
        "--num-parents-mating",
        type=int,
        default=20,
        help="Elite parents kept unchanged per generation (default: 20)",
    )
    p.add_argument(
        "--mutation-percent-genes",
        type=float,
        default=10.0,
        help="Percent of weights mutated per offspring (default: 10.0)",
    )
    p.add_argument(
        "--mutation-std",
        type=float,
        default=0.1,
        help="Gaussian noise std for mutation (default: 0.1)",
    )
    p.add_argument(
        "--start-phase",
        type=int,
        default=0,
        choices=range(5),
        metavar="{0..4}",
        help="Initial curriculum phase 0-4 (default: 0)",
    )
    p.add_argument(
        "--n-workers",
        type=int,
        default=8,
        help="Parallel worker processes for fitness evaluation (default: 8)",
    )
    p.add_argument(
        "--run-name",
        type=str,
        default="ga",
        help="Log/model name prefix (default: ga)",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0)")
    p.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Phase promotion success rate threshold (default: 0.75)",
    )
    p.add_argument(
        "--checkpoint-freq",
        type=int,
        default=50,
        help="Save population checkpoint every N generations (default: 50)",
    )
    p.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to a population checkpoint .npy file to resume from",
    )
    p.add_argument(
        "--crossover",
        action="store_true",
        default=False,
        help="Enable uniform crossover between parents (off by default — hurts neuroevolution)",
    )
    p.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Enable Weights & Biases logging",
    )
    p.add_argument(
        "--wandb-project",
        type=str,
        default="ball-throw-ml",
        help="W&B project name (default: ball-throw-ml)",
    )
    p.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="W&B entity/team (default: your default entity)",
    )
    p.add_argument(
        "--wandb-resume-id",
        type=str,
        default=None,
        help="W&B run ID to resume (found in the run's URL)",
    )
    return p.parse_args()


# ─── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, args.run_name))

    # W&B — optional, enabled via --wandb
    use_wandb = args.wandb
    if use_wandb:
        if not _WANDB_AVAILABLE:
            print("Warning: wandb not installed. Run `pip install wandb`. Disabling W&B.")
            use_wandb = False
        else:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.run_name,
                id=args.wandb_resume_id,
                resume="must" if args.wandb_resume_id else ("allow" if args.resume_from else False),
                config={
                    "pop_size": args.pop_size,
                    "num_generations": args.num_generations,
                    "n_eval_episodes": args.n_eval_episodes,
                    "num_parents_mating": args.num_parents_mating,
                    "mutation_percent_genes": args.mutation_percent_genes,
                    "mutation_std": args.mutation_std,
                    "start_phase": args.start_phase,
                    "n_workers": args.n_workers,
                    "seed": args.seed,
                    "threshold": args.threshold,
                    "genome_len": None,  # filled in after template is built
                    "algorithm": "neuroevolution",
                    "policy": "7-64-64-4-tanh",
                },
                tags=["ga", f"phase{args.start_phase}"],
            )

    # Genome shape from a template network
    template = PolicyNet()
    genome_len = template.n_params
    print(
        f"Policy: 7→64→64→4 MLP  |  Genome: {genome_len} params  |  "
        f"Pop: {args.pop_size}  |  Gens: {args.num_generations}  |  Workers: {args.n_workers}"
    )
    if use_wandb:
        wandb.config.update({"genome_len": genome_len}, allow_val_change=True)

    # Initialize or resume from checkpoint
    if args.resume_from:
        loaded = np.load(args.resume_from, allow_pickle=True)
        if loaded.ndim == 0:
            # Population checkpoint: 0-d object array containing a dict
            state = loaded.item()
            population = state["population"].astype(np.float32)
            start_generation = int(state["generation"])
            start_phase = int(state.get("phase", args.start_phase))
            print(
                f"Resumed checkpoint: {args.resume_from}  |  Gen {start_generation}  |  Phase {start_phase}"
            )
        else:
            # Best-genome file: plain 1-d float array — seed the whole population
            # from this genome with small noise so evolution has a warm start.
            seed_genome = loaded.astype(np.float32)
            rng_init = np.random.default_rng(args.seed)
            noise = rng_init.normal(0, 0.05, size=(args.pop_size, len(seed_genome))).astype(np.float32)
            population = np.tile(seed_genome, (args.pop_size, 1)) + noise
            start_generation = 0
            start_phase = args.start_phase
            print(
                f"Warm-started from genome: {args.resume_from}  |  Phase {start_phase}"
            )
        curriculum = CurriculumState(
            start_phase=start_phase, threshold=args.threshold
        )
    else:
        rng_init = np.random.default_rng(args.seed)
        population = rng_init.uniform(
            -0.5, 0.5, size=(args.pop_size, genome_len)
        ).astype(np.float32)
        start_generation = 0
        curriculum = CurriculumState(
            start_phase=args.start_phase, threshold=args.threshold
        )
        print(f"Starting fresh  |  Phase {curriculum.phase}")

    # Clamp parents to always leave room for at least one offspring
    if args.num_parents_mating >= args.pop_size:
        args.num_parents_mating = max(1, args.pop_size // 2)
        print(
            f"Warning: --num-parents-mating clamped to {args.num_parents_mating} "
            f"(must be < pop_size={args.pop_size})"
        )

    mutation_frac = args.mutation_percent_genes / 100.0
    # Require at least 2 full generations of data before promoting a phase
    min_promotion_samples = args.pop_size * 2

    best_reward_overall = -np.inf
    best_genome_overall: np.ndarray = population[0].copy()
    best_reward_this_phase = -np.inf

    print(f"\nTensorBoard: tensorboard --logdir {LOG_DIR}\n")

    n_workers_actual = min(args.n_workers, args.pop_size)
    _ctx = mp.get_context("spawn")
    pool = _ctx.Pool(processes=n_workers_actual)

    for generation in range(start_generation, args.num_generations):
        t0 = time.monotonic()
        phase = curriculum.phase

        # ── Evaluate all genomes ───────────────────────────────────────────────
        rewards, success_rates, mean_dists, wall_hit_rates = evaluate_population(
            population,
            n_episodes=args.n_eval_episodes,
            phase=phase,
            base_seed=args.seed + generation * 10_000,
            n_workers=n_workers_actual,
            pool=pool,
        )
        elapsed = time.monotonic() - t0

        # ── Compute metrics ────────────────────────────────────────────────────
        best_idx = int(np.argmax(rewards))
        gen_best_reward = float(rewards[best_idx])
        mean_reward = float(rewards.mean())
        mean_success = float(success_rates.mean())
        mean_dist = float(mean_dists.mean())
        best_success = float(success_rates[best_idx])
        best_landing_dist = float(mean_dists[best_idx])
        mean_wall_hit_rate = float(wall_hit_rates.mean())
        population_diversity = float(np.mean(np.std(population, axis=0)))

        # ── Rank population (needed for both curriculum and evolution) ─────────
        ranked_idx = np.argsort(rewards)[::-1]
        elite_mean_success = float(
            success_rates[ranked_idx[: args.num_parents_mating]].mean()
        )

        metrics = {
            "ga/mean_reward": mean_reward,
            "ga/best_reward": gen_best_reward,
            "ga/mean_success_rate": mean_success,
            "ga/elite_success_rate": elite_mean_success,
            "ga/best_success_rate": best_success,
            "ga/mean_landing_dist": mean_dist,
            "ga/best_landing_dist": best_landing_dist,
            "ga/wall_hit_rate": mean_wall_hit_rate * 100,
            "ga/population_diversity": population_diversity,
            "ga/reward_std": float(rewards.std()),
            "curriculum/phase": phase,
            "curriculum/success_rate": mean_success,
            "curriculum/elite_success_rate": elite_mean_success,
            "curriculum/rolling_success": curriculum.mean_success(),
            "perf/eval_time_s": elapsed,
        }
        for k, v in metrics.items():
            writer.add_scalar(k, v, generation)
        if use_wandb:
            wandb.log(metrics, step=generation)

        # ── Save per-phase and overall bests ──────────────────────────────────
        if gen_best_reward > best_reward_this_phase:
            best_reward_this_phase = gen_best_reward
            save_best(population[best_idx], phase, args.run_name, suffix="best")

        if gen_best_reward > best_reward_overall:
            best_reward_overall = gen_best_reward
            best_genome_overall = population[best_idx].copy()

        # ── Console output ─────────────────────────────────────────────────────
        wall_str = f" | Wall {mean_wall_hit_rate * 100:4.1f}%" if phase >= 2 else ""
        print(
            f"Gen {generation:4d}/{args.num_generations} | Ph {phase} | "
            f"Rew mean={mean_reward:+7.2f} best={gen_best_reward:+7.2f} | "
            f"Succ {mean_success * 100:5.1f}% | Dist {mean_dist:.2f}m"
            f"{wall_str} | {elapsed:.1f}s"
        )

        # ── Curriculum update (elites only) ───────────────────────────────────
        # Feed only the top-K genomes' success rates into the curriculum buffer.
        # Using the full population mean is a bug: the 80% of offspring that
        # crossover destroys vote the mean to ~17% even when elites are at 100%.
        elite_success = success_rates[ranked_idx[: args.num_parents_mating]]
        curriculum.update(elite_success)
        if curriculum.should_promote(min_samples=min_promotion_samples):
            old_phase = curriculum.phase
            rolling_success_at_promotion = curriculum.mean_success()
            curriculum.promote()
            best_reward_this_phase = -np.inf  # reset per-phase best
            print(
                f"  *** Curriculum: Phase {old_phase} → {curriculum.phase} "
                f"(rolling success {rolling_success_at_promotion * 100:.0f}%) ***"
            )
            writer.add_scalar("curriculum/phase", curriculum.phase, generation)
            if use_wandb:
                wandb.log({"curriculum/phase": curriculum.phase}, step=generation)
                wandb.alert(
                    title=f"Phase {old_phase} → {curriculum.phase}",
                    text=(
                        f"Promoted at generation {generation}. "
                        f"Rolling success: {rolling_success_at_promotion * 100:.0f}%"
                    ),
                    level=wandb.AlertLevel.INFO,
                )
            ckpt_path = save_checkpoint(
                population, generation, curriculum.phase, args.run_name
            )
            print(f"  Checkpoint: {ckpt_path}")

        # ── Evolution: rank-selection + mutation ──────────────────────────────
        elite = population[ranked_idx[: args.num_parents_mating]].copy()

        n_offspring = args.pop_size - args.num_parents_mating
        offspring = generate_offspring(
            parents=elite,
            n_offspring=n_offspring,
            mutation_std=args.mutation_std,
            mutation_frac=mutation_frac,
            rng=np.random.default_rng(args.seed + generation),
            crossover=args.crossover,
        )
        # Elites pass unchanged; offspring fill the rest
        population = np.vstack([elite, offspring])

        # ── Periodic checkpoint ────────────────────────────────────────────────
        if (generation + 1) % args.checkpoint_freq == 0:
            ckpt_path = save_checkpoint(
                population, generation + 1, curriculum.phase, args.run_name
            )
            print(f"  Checkpoint: {ckpt_path}")

    pool.close()
    pool.join()

    # ── Final save ────────────────────────────────────────────────────────────
    print(f"\nTraining complete. Best reward: {best_reward_overall:+.2f}")
    net = PolicyNet()
    set_params(net, best_genome_overall)
    torch.save(
        net.state_dict(), os.path.join(MODEL_DIR, f"{args.run_name}_final.pt")
    )
    np.save(
        os.path.join(MODEL_DIR, f"{args.run_name}_final_genome.npy"),
        best_genome_overall,
    )
    print(f"Saved: models/{args.run_name}_final.pt")
    print(f"Saved: models/{args.run_name}_final_genome.npy")
    writer.close()
    if use_wandb:
        artifact = wandb.Artifact(
            name=f"{args.run_name}-final",
            type="model",
            description="Final best genome and PyTorch state dict from GA training",
            metadata={"best_reward": best_reward_overall, "genome_len": genome_len},
        )
        artifact.add_file(os.path.join(MODEL_DIR, f"{args.run_name}_final.pt"))
        artifact.add_file(os.path.join(MODEL_DIR, f"{args.run_name}_final_genome.npy"))
        wandb.log_artifact(artifact)
        wandb.finish()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
