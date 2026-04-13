"""
Unit tests for AerodynamicEnv across the current curriculum phases.

Run with:
    python -m pytest test_world.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pytest

from envs.aerodynamic_env import (
    MAX_EPISODE_STEPS,
    AerodynamicEnv,
    OneShotFlightWrapper,
)

STRAIGHT = np.array([0.3, 0.0, 0.5, 0.0], dtype=np.float32)
SPIN_POS = np.array([0.3, 0.0, 0.5, 1.0], dtype=np.float32)
SPIN_NEG = np.array([0.3, 0.0, 0.5, -1.0], dtype=np.float32)


def run_episode(env, action, seed=0):
    """Run a full episode with a fixed action and return final info + rewards."""
    obs, _ = env.reset(seed=seed)
    assert obs.shape == (7,)
    terminated = truncated = False
    info = {}
    rewards = []
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (7,)
        rewards.append(reward)
    return info, rewards


def target_frame_lateral(env, world_xy):
    """Project a world-frame xy point onto the target-relative lateral axis."""
    bearing = env._target_bearing
    cb = np.cos(bearing)
    sb = np.sin(bearing)
    x, y = float(world_xy[0]), float(world_xy[1])
    return -x * sb + y * cb


def target_frame_gravity(env):
    """Return the expected egocentric gravity channels for the current episode."""
    cb = np.cos(env._target_bearing)
    sb = np.sin(env._target_bearing)
    gx, gy, gz = env.model.opt.gravity
    return np.array(
        [
            gx * cb + gy * sb,
            -gx * sb + gy * cb,
            gz,
        ],
        dtype=np.float32,
    )


class TestObservation:
    @pytest.mark.parametrize("phase", [0, 1, 2, 3, 4])
    def test_shape_and_finite(self, phase):
        env = AerodynamicEnv(curriculum_phase=phase)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (7,), f"expected (7,), got {obs.shape}"
        assert np.all(np.isfinite(obs)), "obs contains non-finite values"
        env.close()


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

    def test_spin_disabled_during_training(self):
        """Spin action should have no effect before phase 2 when disabled for training."""
        env = AerodynamicEnv(curriculum_phase=0, disable_spin_before_phase=2)

        def land_pos(action):
            env.reset(seed=42)
            terminated = truncated = False
            while not (terminated or truncated):
                _, _, terminated, truncated, _ = env.step(action)
            return env.data.xpos[env.ball_body_id].copy()

        pos_pos = land_pos(SPIN_POS)
        pos_neg = land_pos(SPIN_NEG)
        lateral_diff = abs(pos_pos[1] - pos_neg[1])
        assert lateral_diff < 0.01, (
            f"Spin should be disabled but caused {lateral_diff:.4f}m deviation"
        )
        env.close()

    def test_spin_works_in_gui_mode(self):
        """Spin should work by default when not explicitly disabled."""
        env = AerodynamicEnv(curriculum_phase=0)

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
            f"Spin should work by default but only caused {lateral_diff:.4f}m deviation"
        )
        env.close()

    def test_wall_hidden(self):
        """Wall should be hidden far away in Phase 0."""
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
        """Wall channels should all be 0 when no wall is present."""
        env = AerodynamicEnv(curriculum_phase=0)
        obs, _ = env.reset(seed=0)
        assert obs[1] == 0.0, f"wall_forward={obs[1]}"
        assert obs[2] == 0.0, f"wall_lateral={obs[2]}"
        assert obs[3] == 0.0, f"wall_width={obs[3]}"
        env.close()


class TestPhase1:
    def test_far_target_distance(self):
        """Target should be 9-20 m away in Phase 1."""
        env = AerodynamicEnv(curriculum_phase=1)
        for seed in range(10):
            env.reset(seed=seed)
            target = env.data.mocap_pos[env.target_mocap_id]
            dist = float(np.linalg.norm(target[:2]))
            assert 8.9 <= dist <= 20.1, f"seed={seed}: target dist {dist:.2f} not in [9, 20]"
        env.close()

    def test_wall_hidden(self):
        """Wall should stay hidden in Phase 1."""
        env = AerodynamicEnv(curriculum_phase=1)
        env.reset(seed=0)
        wall_pos = env.data.mocap_pos[env.wall_mocap_id]
        assert abs(wall_pos[0]) > 500, f"wall_x={wall_pos[0]:.1f}, expected > 500"
        env.close()

    def test_standard_gravity(self):
        """Gravity should remain standard in Phase 1."""
        env = AerodynamicEnv(curriculum_phase=1)
        env.reset(seed=0)
        np.testing.assert_allclose(env.model.opt.gravity, [0.0, 0.0, -9.81], atol=1e-6)
        env.close()

    def test_wall_obs_zero(self):
        """Wall channels should remain zero in Phase 1."""
        env = AerodynamicEnv(curriculum_phase=1)
        obs, _ = env.reset(seed=0)
        assert obs[1] == 0.0, f"wall_forward={obs[1]}"
        assert obs[2] == 0.0, f"wall_lateral={obs[2]}"
        assert obs[3] == 0.0, f"wall_width={obs[3]}"
        env.close()


class TestPhase2:
    def test_wall_between_origin_and_target(self):
        """Easy-wall phase should place the wall about halfway to the target."""
        env = AerodynamicEnv(curriculum_phase=2)
        for seed in range(10):
            env.reset(seed=seed)
            target = env.data.mocap_pos[env.target_mocap_id]
            wall = env.data.mocap_pos[env.wall_mocap_id]
            target_dist = np.linalg.norm(target[:2])
            wall_dist = np.linalg.norm(wall[:2])
            frac = wall_dist / target_dist
            assert 0.44 <= frac <= 0.56, (
                f"seed={seed}: wall fraction {frac:.3f} not in [0.45, 0.55]"
            )
        env.close()

    def test_wall_width_range(self):
        """Phase 2 uses the narrow easy-wall width range [1.5, 2.0]."""
        env = AerodynamicEnv(curriculum_phase=2)
        obs, _ = env.reset(seed=0)
        wall_width = obs[3]
        assert 1.5 <= wall_width <= 2.0, f"wall_width={wall_width:.3f} out of [1.5, 2.0]"
        env.close()

    def test_wall_orientation_faces_origin(self):
        """Wall yaw should match the origin->wall bearing."""
        env = AerodynamicEnv(curriculum_phase=2)
        for seed in range(10):
            env.reset(seed=seed)
            wall_pos = env.data.mocap_pos[env.wall_mocap_id]
            q = env.data.mocap_quat[env.wall_mocap_id]
            expected_angle = np.arctan2(wall_pos[1], wall_pos[0])
            actual_angle = 2.0 * np.arctan2(q[3], q[0])
            diff = (actual_angle - expected_angle + np.pi) % (2 * np.pi) - np.pi
            assert abs(diff) < 0.01, (
                f"seed={seed}: wall yaw {np.degrees(actual_angle):.1f} deg != "
                f"expected {np.degrees(expected_angle):.1f} deg"
            )
        env.close()

    def test_gravity_unchanged(self):
        """Gravity should still be standard in Phase 2."""
        env = AerodynamicEnv(curriculum_phase=2)
        env.reset(seed=0)
        np.testing.assert_allclose(env.model.opt.gravity, [0.0, 0.0, -9.81], atol=1e-6)
        env.close()


class TestPhase3:
    def test_wall_between_origin_and_target(self):
        """Hard-wall phase expands wall placement to the [0.4, 0.6] range."""
        env = AerodynamicEnv(curriculum_phase=3)
        for seed in range(10):
            env.reset(seed=seed)
            target = env.data.mocap_pos[env.target_mocap_id]
            wall = env.data.mocap_pos[env.wall_mocap_id]
            target_dist = np.linalg.norm(target[:2])
            wall_dist = np.linalg.norm(wall[:2])
            frac = wall_dist / target_dist
            assert 0.39 <= frac <= 0.61, (
                f"seed={seed}: wall fraction {frac:.3f} not in [0.4, 0.6]"
            )
        env.close()

    def test_wall_width_range(self):
        """Phase 3 widens the wall range to [1.5, 4.0]."""
        env = AerodynamicEnv(curriculum_phase=3)
        for seed in range(10):
            obs, _ = env.reset(seed=seed)
            wall_width = obs[3]
            assert 1.5 <= wall_width <= 4.0, (
                f"seed={seed}: wall_width={wall_width:.3f} out of [1.5, 4.0]"
            )
        env.close()

    def test_gravity_still_standard(self):
        """Phase 3 is hard wall only; gravity tilt starts in Phase 4."""
        env = AerodynamicEnv(curriculum_phase=3)
        env.reset(seed=0)
        np.testing.assert_allclose(env.model.opt.gravity, [0.0, 0.0, -9.81], atol=1e-6)
        env.close()


class TestPhase4:
    def test_gravity_tilted(self):
        """At least one horizontal gravity component should be non-zero in Phase 4."""
        env = AerodynamicEnv(curriculum_phase=4)
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
        """Gravity vector should differ across resets in the tilt phase."""
        env = AerodynamicEnv(curriculum_phase=4)
        gravities = []
        for seed in range(5):
            env.reset(seed=seed)
            gravities.append(env.model.opt.gravity.copy())
        all_same = all(np.allclose(gravities[0], g) for g in gravities[1:])
        assert not all_same, "Gravity vector did not change across resets"
        env.close()

    def test_gravity_tilt_within_15_degrees(self):
        """Each sampled tilt component should stay within +/-15 degrees."""
        env = AerodynamicEnv(curriculum_phase=4)
        for seed in range(20):
            env.reset(seed=seed)
            g = env.model.opt.gravity
            g_mag = np.linalg.norm(g)
            tx_rad = np.arcsin(np.clip(g[0] / g_mag, -1.0, 1.0))
            ty_rad = np.arcsin(np.clip(g[1] / g_mag, -1.0, 1.0))
            assert abs(tx_rad) <= np.deg2rad(15.0) + 1e-6, (
                f"seed={seed}: tx {np.degrees(tx_rad):.1f} deg exceeds +/-15 deg"
            )
            assert abs(ty_rad) <= np.deg2rad(15.0) + 1e-6, (
                f"seed={seed}: ty {np.degrees(ty_rad):.1f} deg exceeds +/-15 deg"
            )
        env.close()

    def test_gravity_channels_are_egocentric(self):
        """Obs gravity channels should match the target-relative gravity projection."""
        env = AerodynamicEnv(curriculum_phase=4)
        obs, _ = env.reset(seed=7)
        expected = target_frame_gravity(env)
        np.testing.assert_allclose(obs[4:], expected, atol=1e-5)
        env.close()


class TestMagnusEffect:
    def test_spin_causes_lateral_deviation(self):
        """Opposite spin should produce noticeably different target-frame lateral landing."""
        env = AerodynamicEnv(curriculum_phase=2)

        def lateral_landing(action):
            env.reset(seed=42)
            terminated = truncated = False
            while not (terminated or truncated):
                _, _, terminated, truncated, _ = env.step(action)
            pos = env.data.xpos[env.ball_body_id].copy()
            return target_frame_lateral(env, pos[:2])

        lat_pos = lateral_landing(SPIN_POS)
        lat_neg = lateral_landing(SPIN_NEG)
        lateral_diff = abs(lat_pos - lat_neg)
        assert lateral_diff > 0.5, (
            f"Spin +/-1 lateral diff {lateral_diff:.4f}m too small; Magnus may be broken"
        )
        env.close()

    def test_no_spin_stays_on_target_line(self):
        """Zero-spin, zero-yaw throws should stay close to the target ray on flat ground."""
        env = AerodynamicEnv(curriculum_phase=1)
        env.reset(seed=42)
        terminated = truncated = False
        while not (terminated or truncated):
            _, _, terminated, truncated, _ = env.step(STRAIGHT)
        pos = env.data.xpos[env.ball_body_id].copy()
        lateral = abs(target_frame_lateral(env, pos[:2]))
        assert lateral < 0.5, (
            f"No-spin throw deviated {lateral:.3f}m from the target line (expected ~0)"
        )
        env.close()


class TestEpisodeDynamics:
    def test_episode_terminates_on_floor_contact(self):
        """Episode should terminate with landing info when the ball hits the floor."""
        env = AerodynamicEnv(curriculum_phase=1)
        info, _ = run_episode(env, STRAIGHT)
        assert "landing_dist" in info, "No landing_dist in final info"
        env.close()

    def test_truncation_within_max_steps(self):
        """Episode must end within MAX_EPISODE_STEPS."""
        env = AerodynamicEnv(curriculum_phase=1)
        env.reset(seed=0)
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated):
            _, _, terminated, truncated, _ = env.step(STRAIGHT)
            steps += 1
        assert steps <= MAX_EPISODE_STEPS, f"Ran {steps} steps, limit is {MAX_EPISODE_STEPS}"
        env.close()

    def test_rewards_are_finite(self):
        """Rewards should remain finite throughout the episode."""
        env = AerodynamicEnv(curriculum_phase=1)
        _, rewards = run_episode(env, STRAIGHT)
        assert rewards, "Expected at least one reward value"
        assert all(np.isfinite(r) for r in rewards), "Found non-finite reward"
        env.close()

    def test_info_on_termination(self):
        """Final info should contain the current terminal metrics."""
        env = AerodynamicEnv(curriculum_phase=2)
        info, _ = run_episode(env, STRAIGHT)
        assert "landing_dist" in info, "missing landing_dist in info"
        assert "min_dist" in info, "missing min_dist in info"
        assert "success" in info, "missing success in info"
        assert "hit_wall" in info, "missing hit_wall in info"
        assert isinstance(info["landing_dist"], float)
        assert info["landing_dist"] >= 0.0
        assert isinstance(info["min_dist"], float)
        assert info["min_dist"] >= 0.0
        assert isinstance(info["success"], bool)
        assert isinstance(info["hit_wall"], bool)
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


class TestOneShotWrapper:
    def test_wrapper_collapses_full_flight_to_one_outer_step(self):
        """The training wrapper should emit exactly one transition per throw."""
        wrapped = OneShotFlightWrapper(AerodynamicEnv(curriculum_phase=1))
        obs, _ = wrapped.reset(seed=0)
        assert obs.shape == (7,)
        obs, reward, terminated, truncated, info = wrapped.step(STRAIGHT)
        assert obs.shape == (7,)
        assert np.isfinite(reward)
        assert terminated is True
        assert truncated is False
        assert "landing_dist" in info
        wrapped.close()
