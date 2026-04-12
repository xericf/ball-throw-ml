"""
enjoy.py — Interactive MuJoCo viewer for AerodynamicEnv.

Use the "Control" sliders on the right-hand GUI panel to set throw parameters:
  pitch   – launch elevation   [-1, 1]  →  [-30°, 30°]
  yaw     – horizontal azimuth [-1, 1]  →  [-180°, 180°]
  thrust  – initial speed      [-1, 1]  →  [2, 22] m/s
  spin    – angular velocity   [-1, 1]  →  [-20, 20] rad/s

Keys (viewer window must be focused):
  ENTER      – throw the ball with the current slider values
  R / F5     – reset / new random episode (slider values preserved)
  1 / 2 / 3  – switch curriculum phase

Usage:
    mjpython enjoy.py                  # phase 1, no wall
    mjpython enjoy.py --phase 2        # phase 2, wall present
    mjpython enjoy.py --phase 3        # phase 3, tilted gravity
    mjpython enjoy.py --demo           # auto-demo (no interaction needed)
    mjpython enjoy.py --phase 3 --model models/ppo_full_best.zip  # watch trained policy
"""

import argparse
import queue
import time
import numpy as np
import mujoco  # type: ignore[import-untyped]
import mujoco.viewer  # type: ignore[import-untyped]

import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from envs.aerodynamic_env import AerodynamicEnv

# ---------------------------------------------------------------------------
# Viewer key callback — only throw, reset, and phase switches remain.
# Throw parameters are set via the "Control" sliders in the GUI panel.
# ---------------------------------------------------------------------------
_key_queue = queue.Queue()

_KEY_MAP = {
    257: " ",  # ENTER → throw
    294: "r",  # F5    → reset (new scene)
    82: "r",  # R     → reset (new scene)
    49: "1",  # 1
    50: "2",  # 2
    51: "3",  # 3
}


def _key_callback(keycode):
    key = _KEY_MAP.get(keycode)
    if key:
        _key_queue.put(key)


def _poll_key():
    try:
        return _key_queue.get_nowait()
    except queue.Empty:
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def throw_and_run(env, viewer, action):
    """Apply one throw and simulate until landing, syncing the viewer."""
    env.reset()
    terminated = truncated = False
    info: dict = {}
    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(action)
        viewer.sync()
        time.sleep(env.model.opt.timestep)
    dist = info.get("landing_dist", float("nan"))
    success = info.get("success", False)
    print(f"  Landing dist: {dist:.2f} m  {'SUCCESS' if success else 'miss'}")


def run_policy(env, viewer, model, n_episodes=20, vec_normalize=None):
    """Query a trained policy for the throw action and visualise each episode."""
    for i in range(n_episodes):
        obs, _ = env.reset()
        policy_obs = vec_normalize.normalize_obs(obs) if vec_normalize is not None else obs
        action, _ = model.predict(policy_obs, deterministic=True)
        label = (
            f"policy throw {i + 1}/{n_episodes} -> "
            f"pitch={action[0]:+.2f} yaw={action[1]:+.2f} "
            f"thrust={action[2]:+.2f} spin={action[3]:+.2f}"
        )
        print(f"\n{label}")
        terminated = truncated = False
        info: dict = {}
        while not (terminated or truncated):
            _, _, terminated, truncated, info = env.step(action)
            viewer.sync()
            time.sleep(env.model.opt.timestep)
        dist = info.get("landing_dist", float("nan"))
        success = info.get("success", False)
        print(f"  Landing dist: {dist:.2f} m  {'SUCCESS' if success else 'miss'}")
        time.sleep(0.6)


def run_demo(env, viewer):
    """Cycle through a handful of preset throws."""
    demos = [
        ("straight throw", [0.30, 0.0, 0.5, 0.0]),
        ("+spin (curves right)", [0.30, 0.0, 0.5, 1.0]),
        ("-spin (curves left)", [0.30, 0.0, 0.5, -1.0]),
        ("high arc", [0.80, 0.0, 0.3, 0.0]),
        ("hard throw", [0.20, 0.0, 1.0, 0.0]),
    ]
    for label, action in demos:
        print(f"\nDemo: {label}")
        throw_and_run(env, viewer, np.array(action, dtype=np.float32))
        time.sleep(1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=1, choices=[0, 1, 2, 3])
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run automatic demo throws instead of interactive mode",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to a trained SB3 PPO .zip; runs policy rollouts instead of interactive mode",
    )
    parser.add_argument(
        "--vecnormalize",
        type=str,
        default=None,
        help="Path to the matching VecNormalize .pkl (auto-detected as "
        "models/<run_name>_vecnormalize.pkl if --model path ends in _final/_best)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of episodes to run when --model is provided",
    )
    args = parser.parse_args()

    policy_model = None
    vec_normalize = None
    if args.model is not None:
        from stable_baselines3 import PPO  # type: ignore[import-untyped]

        print(f"Loading policy from {args.model}")
        policy_model = PPO.load(args.model, device="cpu")

        vn_path = args.vecnormalize
        if vn_path is None:
            # Auto-detect sibling pkl: strip _final / _best from model stem.
            stem = os.path.splitext(os.path.basename(args.model))[0]
            for suffix in ("_final", "_best"):
                if stem.endswith(suffix):
                    candidate = os.path.join(
                        os.path.dirname(args.model), f"{stem[:-len(suffix)]}_vecnormalize.pkl"
                    )
                    if os.path.exists(candidate):
                        vn_path = candidate
                        break
        if vn_path is not None:
            import pickle  # noqa: WPS433

            print(f"Loading VecNormalize stats from {vn_path}")
            with open(vn_path, "rb") as f:
                vec_normalize = pickle.load(f)
        else:
            print(
                "WARNING: no VecNormalize stats found. Policy will see raw (unnormalized) "
                "observations — expect garbage behaviour."
            )

    env = AerodynamicEnv(curriculum_phase=args.phase)
    obs, _ = env.reset(seed=42)

    print(__doc__)
    print(f"Curriculum phase: {args.phase}")

    with mujoco.viewer.launch_passive(
        env.model,
        env.data,
        key_callback=_key_callback,
        show_left_ui=False,
        show_right_ui=True,
    ) as viewer:
        viewer.cam.distance = 25
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 135

        mujoco.mj_forward(env.model, env.data)
        viewer.sync()

        # Print initial world configuration
        _target = env.data.mocap_pos[env.target_mocap_id]
        print(f"\n--- World reset (phase {env.curriculum_phase}) ---")
        print(
            f"  Target : ({_target[0]:+.1f}, {_target[1]:+.1f})  dist={float(np.linalg.norm(_target[:2])):.1f}m"
        )
        if env.curriculum_phase >= 2:
            _wall = env.data.mocap_pos[env.wall_mocap_id]
            _ww = float(env.model.geom_size[env.wall_geom_id][1]) * 2.0
            print(
                f"  Wall   : ({_wall[0]:+.1f}, {_wall[1]:+.1f})  dist={float(np.linalg.norm(_wall[:2])):.1f}m  width={_ww:.1f}m"
            )
        _g = env.model.opt.gravity
        print(f"  Gravity: [{_g[0]:+.2f}, {_g[1]:+.2f}, {_g[2]:+.2f}]")

        if policy_model is not None:
            run_policy(
                env,
                viewer,
                policy_model,
                n_episodes=args.episodes,
                vec_normalize=vec_normalize,
            )
            print("\nPolicy rollout complete. Close the window to exit.")
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.05)
            return

        if args.demo:
            run_demo(env, viewer)
            print("\nDemo complete. Close the window to exit.")
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.05)
            return

        # ---- Interactive mode ----
        print("\nInteractive mode. Adjust sliders then press ENTER to throw.\n")
        thrown = False
        terminated = truncated = False
        last_action = np.zeros(4, dtype=np.float32)

        def reset_scene(phase=None):
            """Full scene reset: randomises target/wall. Preserves slider values."""
            saved_ctrl = env.data.ctrl.copy()
            if phase is not None:
                env.curriculum_phase = phase
            with viewer.lock():
                env.reset()
                env.data.ctrl[:] = saved_ctrl

            # ---- Print world configuration ----
            target = env.data.mocap_pos[env.target_mocap_id]
            target_dist = float(np.linalg.norm(target[:2]))
            print(f"\n--- World reset (phase {env.curriculum_phase}) ---")
            print(
                f"  Target : ({target[0]:+.1f}, {target[1]:+.1f})  dist={target_dist:.1f}m"
            )
            if env.curriculum_phase >= 2:
                wall = env.data.mocap_pos[env.wall_mocap_id]
                wall_dist = float(np.linalg.norm(wall[:2]))
                wall_width = float(env.model.geom_size[env.wall_geom_id][1]) * 2.0
                print(
                    f"  Wall   : ({wall[0]:+.1f}, {wall[1]:+.1f})  dist={wall_dist:.1f}m  width={wall_width:.1f}m"
                )
            g = env.model.opt.gravity
            print(f"  Gravity: [{g[0]:+.2f}, {g[1]:+.2f}, {g[2]:+.2f}]")

        def reset_ball():
            """Reset only the ball, keeping the current target/wall in place."""
            saved_mocap_pos = env.data.mocap_pos.copy()
            saved_mocap_quat = env.data.mocap_quat.copy()
            saved_ctrl = env.data.ctrl.copy()
            with viewer.lock():
                mujoco.mj_resetData(env.model, env.data)
                env.data.mocap_pos[:] = saved_mocap_pos
                env.data.mocap_quat[:] = saved_mocap_quat
                env.data.ctrl[:] = saved_ctrl
            env._thrown = False
            env._step_count = 0
            env._spin = 0.0

        def drain_key_queue():
            """Discard backlogged keys (e.g. GLFW key-repeat events)."""
            while True:
                try:
                    _key_queue.get_nowait()
                except queue.Empty:
                    break

        while viewer.is_running():
            if thrown and not (terminated or truncated):
                _, _, terminated, truncated, info = env.step(last_action)
                if np.any(~np.isfinite(env.data.qpos)):
                    print(
                        "  Simulation unstable — press ENTER to retry or R to reset scene."
                    )
                    terminated = True
                viewer.sync()
                time.sleep(env.model.opt.timestep)
                if terminated or truncated:
                    dist = info.get("landing_dist", float("nan"))
                    if np.isfinite(dist):
                        print(
                            f"  Landed: {dist:.2f} m from target  "
                            f"{'SUCCESS' if info.get('success') else 'miss'}"
                        )
                    drain_key_queue()
            else:
                viewer.sync()
                time.sleep(0.02)

                key = _poll_key()
                if key == " ":
                    last_action = env.data.ctrl[:4].copy().astype(np.float32)
                    print(
                        f"\nThrow -> pitch={last_action[0]:+.2f}  "
                        f"yaw={last_action[1]:+.2f}  "
                        f"thrust={last_action[2]:+.2f}  "
                        f"spin={last_action[3]:+.2f}"
                    )
                    reset_ball()  # keep scene, only reset ball position
                    env._apply_throw(last_action)
                    env._thrown = True
                    mujoco.mj_forward(env.model, env.data)
                    viewer.sync()
                    thrown = True
                    terminated = truncated = False
                    drain_key_queue()  # discard any repeat keypresses

                elif key == "r":
                    reset_scene()
                    mujoco.mj_forward(env.model, env.data)
                    viewer.sync()
                    thrown = False
                    terminated = truncated = False
                    print("Reset (new scene).")

                elif key == "1":
                    reset_scene(phase=1)
                    mujoco.mj_forward(env.model, env.data)
                    viewer.sync()
                    print("Phase 1 (no wall).")
                elif key == "2":
                    reset_scene(phase=2)
                    mujoco.mj_forward(env.model, env.data)
                    viewer.sync()
                    print("Phase 2 (wall added).")
                elif key == "3":
                    reset_scene(phase=3)
                    mujoco.mj_forward(env.model, env.data)
                    viewer.sync()
                    print("Phase 3 (gravity tilt).")


if __name__ == "__main__":
    main()
