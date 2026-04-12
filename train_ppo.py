"""
train_ppo.py — PPO trainer with curriculum learning for AerodynamicEnv.

Usage:
    python train_ppo.py --run-name ppo_full
    python train_ppo.py --timesteps 20000 --n-envs 4 --run-name smoke

The CurriculumManagerCallback watches the rolling success rate (from the info
dicts emitted on terminal steps) and promotes all workers from phase 1 -> 2 -> 3
as soon as the agent hits a configurable threshold.
"""

import argparse
import os
import re
import sys
from collections import deque

import torch

torch.set_num_threads(1)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    EvalCallback,
)
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
    VecNormalize,
)

sys.path.insert(0, os.path.dirname(__file__))
from envs.aerodynamic_env import AerodynamicEnv, OneShotFlightWrapper

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


def make_env(rank: int, start_phase: int, base_seed: int):
    """Factory returning a thunk that builds a single one-shot worker env."""

    def _thunk():
        env = AerodynamicEnv(curriculum_phase=start_phase)
        env.reset(seed=base_seed + rank)
        return OneShotFlightWrapper(env)

    return _thunk


class CurriculumManagerCallback(BaseCallback):
    """
    Monitor rolling success rate and advance curriculum phase when threshold hit.

    Reads terminal-step info dicts directly from self.locals["infos"] rather
    than ep_info_buffer: VecMonitor only stores {"r","l","t"} in that buffer
    and silently drops custom keys like "success" and "landing_dist".
    """

    def __init__(
        self,
        start_phase: int = 0,
        max_phase: int = 3,
        threshold: float = 0.70,
        window: int = 100,
        min_new_episodes: int = 50,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.current_phase = start_phase
        self.max_phase = max_phase
        self.threshold = threshold
        self.window = window
        self.min_new_episodes = min_new_episodes
        self._episodes_since_promotion = 0
        self._success_buf: deque = deque(maxlen=window)
        self._dist_buf: deque = deque(maxlen=window)
        self._wall_hit_buf: deque = deque(maxlen=window)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for info, done in zip(infos, dones):
            if not done:
                continue
            self._episodes_since_promotion += 1
            self._success_buf.append(float(info.get("success", False)))
            dist = info.get("landing_dist")
            if dist is not None:
                self._dist_buf.append(dist)
            if "hit_wall" in info:
                self._wall_hit_buf.append(float(info["hit_wall"]))

        if (
            len(self._success_buf) >= self.window
            and self._episodes_since_promotion >= self.min_new_episodes
            and self.current_phase < self.max_phase
        ):
            success_rate = float(np.mean(self._success_buf))
            if success_rate > self.threshold:
                new_phase = self.current_phase + 1
                self.training_env.env_method("set_curriculum_phase", new_phase)
                if self.verbose:
                    print(
                        f"\n[Curriculum] success_rate={success_rate:.3f} > "
                        f"{self.threshold:.2f} — promoting phase "
                        f"{self.current_phase} -> {new_phase}\n"
                    )
                self.current_phase = new_phase
                self._episodes_since_promotion = 0
                self._success_buf.clear()
                self._dist_buf.clear()
                self._wall_hit_buf.clear()
                self.logger.record("curriculum/phase", float(new_phase))

        return True

    def _on_rollout_end(self) -> None:
        self.logger.record("curriculum/phase", float(self.current_phase))

        if self._success_buf:
            success_rate = float(np.mean(self._success_buf))
            self.logger.record("curriculum/success_rate", success_rate)

        if self._dist_buf:
            dists = np.array(self._dist_buf)
            mean_dist = float(dists.mean())
            self.logger.record("curriculum/mean_landing_dist", mean_dist)
            self.logger.record("curriculum/median_landing_dist", float(np.median(dists)))
            self.logger.record("curriculum/pct_within_1m", float((dists < 1.0).mean() * 100))
            self.logger.record("curriculum/pct_within_2m", float((dists < 2.0).mean() * 100))
            self.logger.record("curriculum/pct_within_6m", float((dists < 6.0).mean() * 100))
            wall_str = ""
            if self._wall_hit_buf:
                wall_str = f"  wall_hit={(np.mean(self._wall_hit_buf)*100):.1f}%"
                self.logger.record(
                    "curriculum/wall_hit_rate",
                    float(np.mean(self._wall_hit_buf) * 100),
                )
            if self.verbose >= 1:
                print(
                    f"  [dist] mean={mean_dist:.2f}m  "
                    f"<1m={(dists < 1.0).mean()*100:.1f}%  "
                    f"<2m={(dists < 2.0).mean()*100:.1f}%  "
                    f"<6m={(dists < 6.0).mean()*100:.1f}%"
                    f"{wall_str}  | phase={self.current_phase}"
                )


class SaveBestModelCallback(BaseCallback):
    """Saves {name_prefix}_best.zip + {name_prefix}_vecnormalize.pkl on every new best.

    Pass as callback_on_new_best to EvalCallback (with best_model_save_path=None).
    enjoy.py auto-detects the vecnorm by stripping '_best' from the model stem.
    """

    def __init__(self, save_path: str, name_prefix: str, verbose: int = 1):
        super().__init__(verbose)
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        model_path = os.path.join(self.save_path, f"{self.name_prefix}_best")
        vecnorm_path = os.path.join(self.save_path, f"{self.name_prefix}_best_vecnormalize.pkl")
        self.model.save(model_path)
        self.training_env.save(vecnorm_path)
        if self.verbose >= 1:
            print(f"[Best] {model_path}.zip  (vecnorm → {os.path.basename(vecnorm_path)})")
        return True


class PhaseCheckpointCallback(BaseCallback):
    """
    Saves periodic checkpoints with the current curriculum phase in the filename.
    Also saves paired VecNormalize stats so resume is always reliable.

    Produces:
        models/{name_prefix}_p{phase}_{timesteps}_steps.zip
        models/{name_prefix}_p{phase}_{timesteps}_steps_vecnormalize.pkl
    """

    def __init__(self, save_freq: int, save_path: str, name_prefix: str,
                 curriculum_cb: "CurriculumManagerCallback", verbose: int = 1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.curriculum_cb = curriculum_cb

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            phase = self.curriculum_cb.current_phase
            stem = f"{self.name_prefix}_p{phase}_{self.num_timesteps}_steps"
            model_path = os.path.join(self.save_path, stem)
            vecnorm_path = os.path.join(self.save_path, f"{stem}_vecnormalize.pkl")
            self.model.save(model_path)
            self.training_env.save(vecnorm_path)
            if self.verbose >= 1:
                print(f"[Checkpoint] Saved {model_path}.zip  (phase {phase})")
        return True


def build_vec_env(n_envs: int, start_phase: int, base_seed: int, vecnorm_path: str = None):
    if n_envs == 1:
        vec = DummyVecEnv([make_env(0, start_phase, base_seed)])
    else:
        vec = SubprocVecEnv(
            [make_env(i, start_phase, base_seed) for i in range(n_envs)],
            start_method="spawn",
        )
    vec = VecMonitor(vec)
    if vecnorm_path and os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, vec)
        env.training = True
        env.norm_reward = True
        print(f"Loaded VecNormalize stats from {vecnorm_path}")
    else:
        env = VecNormalize(vec, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return env


def build_eval_env(start_phase: int, seed: int):
    # Evaluate on the *target* phase so we track final capability.
    def _thunk():
        env = AerodynamicEnv(curriculum_phase=start_phase)
        env.reset(seed=seed)
        return OneShotFlightWrapper(env)

    vec = DummyVecEnv([_thunk])
    vec = VecMonitor(vec)
    # training=False + norm_reward=False: eval env borrows obs stats from the
    # training env (EvalCallback calls sync_envs_normalization automatically
    # when both envs are VecNormalize-wrapped).
    vec = VecNormalize(
        vec, norm_obs=True, norm_reward=False, training=False, clip_obs=10.0
    )
    return vec


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=1_500_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--start-phase", type=int, default=0, choices=[0, 1, 2, 3])
    p.add_argument("--eval-phase", type=int, default=0, choices=[0, 1, 2, 3])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", type=str, default="ppo")
    p.add_argument("--threshold", type=float, default=0.70)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--resume-from", type=str, default=None,
                   help="Path to a checkpoint .zip to resume from "
                        "(e.g. models/ppo_full_p1_50000_steps.zip)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"Building {args.n_envs} parallel envs (phase {args.start_phase})...")
    eval_env = build_eval_env(args.eval_phase, args.seed + 10_000)

    if args.resume_from:
        vecnorm_path = re.sub(r'\.zip$', '_vecnormalize.pkl', args.resume_from)
        if not os.path.exists(vecnorm_path):
            print(f"Warning: no VecNormalize stats at {vecnorm_path}; starting fresh normalization")
            vecnorm_path = None
        train_env = build_vec_env(args.n_envs, args.start_phase, args.seed,
                                  vecnorm_path=vecnorm_path)
        model = PPO.load(args.resume_from, env=train_env, device="cpu")
        print(f"Resumed from {args.resume_from}")
    else:
        train_env = build_vec_env(args.n_envs, args.start_phase, args.seed)
        model = PPO(
            "MlpPolicy",
            train_env,
            device="cpu",
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.05,
            vf_coef=0.5,
            target_kl=0.05,
            max_grad_norm=0.5,
            tensorboard_log=LOG_DIR,
            seed=args.seed,
            verbose=1,
        )

    curriculum_cb = CurriculumManagerCallback(
        start_phase=args.start_phase,
        threshold=args.threshold,
    )
    checkpoint_cb = PhaseCheckpointCallback(
        save_freq=max(50_000 // args.n_envs, 1),
        save_path=MODEL_DIR,
        name_prefix=args.run_name,
        curriculum_cb=curriculum_cb,
    )
    save_best_cb = SaveBestModelCallback(
        save_path=MODEL_DIR,
        name_prefix=args.run_name,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=None,       # we save ourselves via callback_on_new_best
        callback_on_new_best=save_best_cb,
        log_path=os.path.join(LOG_DIR, f"{args.run_name}_eval"),
        eval_freq=max(25_000 // args.n_envs, 1),
        n_eval_episodes=20,
        deterministic=True,
        render=False,
    )

    callbacks = CallbackList([curriculum_cb, checkpoint_cb, eval_cb])

    print(f"Training for {args.timesteps} timesteps — run name: {args.run_name}")
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            tb_log_name=args.run_name,
            progress_bar=True,
        )
    finally:
        final_path = os.path.join(MODEL_DIR, f"{args.run_name}_final.zip")
        model.save(final_path)
        vecnorm_path = os.path.join(MODEL_DIR, f"{args.run_name}_final_vecnormalize.pkl")
        train_env.save(vecnorm_path)
        print(f"Saved final model to {final_path}")
        print(f"Saved VecNormalize stats to {vecnorm_path}")
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
