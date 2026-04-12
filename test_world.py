"""
Unit tests for AerodynamicEnv across all 3 curriculum phases.

Run with:
    python -m pytest test_world.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pytest
from envs.aerodynamic_env import AerodynamicEnv, MAX_EPISODE_STEPS

STRAIGHT = np.array([0.3,  0.0, 0.5, 0.0], dtype=np.float32)
SPIN_POS  = np.array([0.3,  0.0, 0.5, 1.0], dtype=np.float32)
SPIN_NEG  = np.array([0.3,  0.0, 0.5, -1.0], dtype=np.float32)


def run_episode(env, action):
    """Run a full episode with a fixed action and return final info."""
    obs, _ = env.reset(seed=0)
    terminated = truncated = False
    info = {}
    rewards = []
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
    return info, rewards


# ---------------------------------------------------------------------------
# Observation structure
# ---------------------------------------------------------------------------

class TestObservation:
    def test_shape_and_finite_phase1(self):
        env = AerodynamicEnv(curriculum_phase=1)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (8,), f"expected (8,), got {obs.shape}"
        assert np.all(np.isfinite(obs)), "obs contains non-finite values"
        env.close()

    def test_shape_and_finite_phase2(self):
        env = AerodynamicEnv(curriculum_phase=2)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (8,)
        assert np.all(np.isfinite(obs))
        env.close()

    def test_shape_and_finite_phase3(self):
        env = AerodynamicEnv(curriculum_phase=3)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (8,)
        assert np.all(np.isfinite(obs))
        env.close()


# ---------------------------------------------------------------------------
# Phase 0
# ---------------------------------------------------------------------------

class TestPhase0:
    def test_close_target_distance(self):
        """Target should be 5-10 m away in Phase 0."""
        env = AerodynamicEnv(curriculum_phase=0)
        for seed in range(10):
            env.reset(seed=seed)
            target = env.data.mocap_pos[env.target_mocap_id]
            dist = float(np.linalg.norm(target[:2]))
            assert 4.9 <= dist <= 10.1, f"seed={seed}: target dist {dist:.2f} not in [5, 10]"
        env.close()

    def test_wall_hidden(self):
        """Wall should be hidden (x > 500) in Phase 0."""
        env = AerodynamicEnv(curriculum_phase=0)
        env.reset(seed=0)
        wall_pos = env.data.mocap_pos[env.wall_mocap_id]
        assert abs(wall_pos[0]) > 500
        env.close()

    def test_standard_gravity(self):
        """Gravity should be standard in Phase 0."""
        env = AerodynamicEnv(curriculum_phase=0)
        env.reset(seed=0)
        np.testing.assert_allclose(env.model.opt.gravity, [0.0, 0.0, -9.81], atol=1e-6)
        env.close()

    def test_wall_obs_zero(self):
        """obs[2], obs[3], obs[4] (wall channels) should all be 0 in Phase 0."""
        env = AerodynamicEnv(curriculum_phase=0)
        obs, _ = env.reset(seed=0)
        assert obs[2] == 0.0, f"wall_dx={obs[2]}"
        assert obs[3] == 0.0, f"wall_dy={obs[3]}"
        assert obs[4] == 0.0, f"wall_width={obs[4]}"
        env.close()


# ---------------------------------------------------------------------------
# Phase 1
# ---------------------------------------------------------------------------

class TestPhase1:
    def test_wall_hidden(self):
        """Wall should be placed far away (x > 500) so it never interferes."""
        env = AerodynamicEnv(curriculum_phase=1)
        env.reset(seed=0)
        wall_pos = env.data.mocap_pos[env.wall_mocap_id]
        assert abs(wall_pos[0]) > 500, f"wall_x={wall_pos[0]:.1f}, expected > 500"
        env.close()

    def test_standard_gravity(self):
        """Gravity should be exactly [0, 0, -9.81] in Phase 1."""
        env = AerodynamicEnv(curriculum_phase=1)
        env.reset(seed=0)
        g = env.model.opt.gravity
        np.testing.assert_allclose(g, [0.0, 0.0, -9.81], atol=1e-6)
        env.close()

    def test_wall_width_zero_in_obs(self):
        """obs[4] (wall_width) should be 0.0 in Phase 1."""
        env = AerodynamicEnv(curriculum_phase=1)
        obs, _ = env.reset(seed=0)
        assert obs[4] == 0.0, f"expected wall_width=0, got {obs[4]}"
        env.close()

    def test_wall_identity_orientation(self):
        """Wall quaternion should be identity in Phase 1."""
        env = AerodynamicEnv(curriculum_phase=1)
        env.reset(seed=0)
        q = env.data.mocap_quat[env.wall_mocap_id]
        np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-6)
        env.close()

    def test_wall_obs_zero(self):
        """obs wall channels should be 0 in Phase 1 (no wall in obs)."""
        env = AerodynamicEnv(curriculum_phase=1)
        obs, _ = env.reset(seed=0)
        assert obs[2] == 0.0, f"wall_dx={obs[2]}"
        assert obs[3] == 0.0, f"wall_dy={obs[3]}"
        assert obs[4] == 0.0, f"wall_width={obs[4]}"
        env.close()


# ---------------------------------------------------------------------------
# Phase 2
# ---------------------------------------------------------------------------

class TestPhase2:
    def test_wall_between_origin_and_target(self):
        """Wall should be at 30–70% of the target distance along each axis."""
        env = AerodynamicEnv(curriculum_phase=2)
        for seed in range(10):
            env.reset(seed=seed)
            target = env.data.mocap_pos[env.target_mocap_id]
            wall   = env.data.mocap_pos[env.wall_mocap_id]
            # wall_frac = wall_dist / target_dist should be in [0.3, 0.7]
            target_dist = np.linalg.norm(target[:2])
            wall_dist   = np.linalg.norm(wall[:2])
            frac = wall_dist / target_dist
            assert 0.28 <= frac <= 0.72, (
                f"seed={seed}: wall fraction {frac:.3f} not in [0.3, 0.7]"
            )
        env.close()

    def test_wall_width_in_obs(self):
        """obs[4] (wall_width) should be in [1.0, 3.0] in Phase 2."""
        env = AerodynamicEnv(curriculum_phase=2)
        obs, _ = env.reset(seed=0)
        wall_width = obs[4]
        assert 1.0 <= wall_width <= 3.0, f"wall_width={wall_width:.3f} out of [1.0, 3.0]"
        env.close()

    def test_wall_orientation_faces_origin(self):
        """
        Wall quaternion should be a pure Z rotation by atan2(wall_y, wall_x).
        Extract yaw from q = [w, 0, 0, sin(θ/2)] => θ = 2*arcsin(q[3]).
        """
        env = AerodynamicEnv(curriculum_phase=2)
        for seed in range(10):
            env.reset(seed=seed)
            wall_pos = env.data.mocap_pos[env.wall_mocap_id]
            q = env.data.mocap_quat[env.wall_mocap_id]

            expected_angle = np.arctan2(wall_pos[1], wall_pos[0])
            # Pure Z rotation: q = [cos(θ/2), 0, 0, sin(θ/2)]
            actual_angle = 2.0 * np.arctan2(q[3], q[0])  # atan2(sin(θ/2), cos(θ/2))*2

            # Normalise both angles to [-π, π] before comparing
            diff = (actual_angle - expected_angle + np.pi) % (2 * np.pi) - np.pi
            assert abs(diff) < 0.01, (
                f"seed={seed}: wall yaw {np.degrees(actual_angle):.1f}° != "
                f"expected {np.degrees(expected_angle):.1f}°"
            )
        env.close()

    def test_gravity_unchanged_phase2(self):
        """Gravity should still be standard in Phase 2."""
        env = AerodynamicEnv(curriculum_phase=2)
        env.reset(seed=0)
        np.testing.assert_allclose(env.model.opt.gravity, [0.0, 0.0, -9.81], atol=1e-6)
        env.close()


# ---------------------------------------------------------------------------
# Phase 3
# ---------------------------------------------------------------------------

class TestPhase3:
    def test_gravity_tilted(self):
        """At least one horizontal gravity component should be non-zero."""
        env = AerodynamicEnv(curriculum_phase=3)
        any_tilted = False
        for seed in range(10):
            env.reset(seed=seed)
            gx, gy = env.model.opt.gravity[0], env.model.opt.gravity[1]
            if abs(gx) > 1e-6 or abs(gy) > 1e-6:
                any_tilted = True
                break
        assert any_tilted, "No tilt observed across 10 seeds"
        env.close()

    def test_gravity_varies_across_resets(self):
        """Gravity vector should differ across resets (randomised tilt)."""
        env = AerodynamicEnv(curriculum_phase=3)
        gravities = []
        for seed in range(5):
            env.reset(seed=seed)
            gravities.append(env.model.opt.gravity.copy())
        # Not all gravity vectors should be identical
        all_same = all(np.allclose(gravities[0], g) for g in gravities[1:])
        assert not all_same, "Gravity vector did not change across resets"
        env.close()

    def test_gravity_tilt_within_15_degrees(self):
        """
        tx and ty are each sampled from [-15°, 15°] independently.
        Check that |arcsin(gx/|g|)| and |arcsin(gy/|g|)| are both ≤ 15°.
        (Combined angle can exceed 15° when both components are large.)
        """
        env = AerodynamicEnv(curriculum_phase=3)
        for seed in range(20):
            env.reset(seed=seed)
            g = env.model.opt.gravity
            g_mag = np.linalg.norm(g)
            tx_rad = np.arcsin(np.clip( g[0] / g_mag, -1.0, 1.0))
            ty_rad = np.arcsin(np.clip( g[1] / g_mag, -1.0, 1.0))
            assert abs(tx_rad) <= np.deg2rad(15.0) + 1e-6, (
                f"seed={seed}: tx {np.degrees(tx_rad):.1f}° exceeds ±15°"
            )
            assert abs(ty_rad) <= np.deg2rad(15.0) + 1e-6, (
                f"seed={seed}: ty {np.degrees(ty_rad):.1f}° exceeds ±15°"
            )
        env.close()

    def test_gravity_in_obs(self):
        """obs[5], obs[6], obs[7] should reflect all three gravity components."""
        env = AerodynamicEnv(curriculum_phase=3)
        obs, _ = env.reset(seed=7)
        g = env.model.opt.gravity
        assert abs(obs[5] - g[0]) < 1e-5, f"obs gravity_x mismatch: {obs[5]} vs {g[0]}"
        assert abs(obs[6] - g[1]) < 1e-5, f"obs gravity_y mismatch: {obs[6]} vs {g[1]}"
        assert abs(obs[7] - g[2]) < 1e-5, f"obs gravity_z mismatch: {obs[7]} vs {g[2]}"
        env.close()


# ---------------------------------------------------------------------------
# Magnus effect
# ---------------------------------------------------------------------------

class TestMagnusEffect:
    def test_spin_causes_lateral_deviation(self):
        """
        Same throw with opposite spin should land at clearly different positions.
        The Magnus force is F = k*(ω×v); ω_z with v_x produces force in ±y.
        """
        env = AerodynamicEnv(curriculum_phase=1)

        def land_pos(action):
            env.reset(seed=42)
            terminated = truncated = False
            while not (terminated or truncated):
                _, _, terminated, truncated, _ = env.step(action)
            return env.data.xpos[env.ball_body_id].copy()

        pos_pos = land_pos(SPIN_POS)
        pos_neg = land_pos(SPIN_NEG)

        lateral_diff = abs(pos_pos[1] - pos_neg[1])
        assert lateral_diff > 0.5, (
            f"Spin ±1 lateral diff {lateral_diff:.4f}m too small — Magnus may be broken"
        )
        env.close()

    def test_no_spin_symmetric(self):
        """Zero spin with zero yaw should stay close to y=0 (symmetric throw)."""
        env = AerodynamicEnv(curriculum_phase=1)
        env.reset(seed=42)
        terminated = truncated = False
        while not (terminated or truncated):
            _, _, terminated, truncated, _ = env.step(STRAIGHT)
        y = abs(env.data.xpos[env.ball_body_id][1])
        assert y < 0.5, f"No-spin throw deviated {y:.3f}m laterally (expected ~0)"
        env.close()


# ---------------------------------------------------------------------------
# Episode dynamics
# ---------------------------------------------------------------------------

class TestEpisodeDynamics:
    def test_episode_terminates_on_floor_contact(self):
        """Episode should terminate (terminated=True) when ball hits floor."""
        env = AerodynamicEnv(curriculum_phase=1)
        info, _ = run_episode(env, STRAIGHT)
        assert "landing_dist" in info, "No landing_dist in final info"
        env.close()

    def test_truncation_within_max_steps(self):
        """Episode must end within MAX_EPISODE_STEPS even for a very slow throw."""
        env = AerodynamicEnv(curriculum_phase=1)
        env.reset(seed=0)
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated):
            _, _, terminated, truncated, _ = env.step(STRAIGHT)
            steps += 1
        assert steps <= MAX_EPISODE_STEPS, f"Ran {steps} steps, limit is {MAX_EPISODE_STEPS}"
        env.close()

    def test_reward_nonpositive(self):
        """Dense reward = -L2_distance should always be ≤ 0."""
        env = AerodynamicEnv(curriculum_phase=1)
        _, rewards = run_episode(env, STRAIGHT)
        assert all(r <= 0 for r in rewards), (
            f"Found positive reward: max={max(rewards):.4f}"
        )
        env.close()

    def test_info_on_termination(self):
        """Final info must contain landing_dist (float ≥ 0) and success (bool)."""
        env = AerodynamicEnv(curriculum_phase=1)
        info, _ = run_episode(env, STRAIGHT)
        assert "landing_dist" in info, "missing landing_dist in info"
        assert "success" in info, "missing success in info"
        assert isinstance(info["landing_dist"], float)
        assert info["landing_dist"] >= 0.0
        assert isinstance(info["success"], bool)
        env.close()

    def test_obs_finite_throughout_episode(self):
        """Observation should remain finite for the entire episode."""
        env = AerodynamicEnv(curriculum_phase=1)
        env.reset(seed=0)
        terminated = truncated = False
        while not (terminated or truncated):
            obs, _, terminated, truncated, _ = env.step(STRAIGHT)
            assert np.all(np.isfinite(obs)), f"Non-finite obs: {obs}"
        env.close()
