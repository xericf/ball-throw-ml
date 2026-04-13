import os
import numpy as np
import gymnasium as gym
import mujoco  # type: ignore[import-untyped]

MAGNUS_K = 0.012  # Magnus force coefficient; F = k * cross(omega, v)
MAX_EPISODE_STEPS = 1000  # 5 s at 0.005 s/step
YAW_OFFSET_MAX = np.pi / 2  # ±90° relative to target bearing


class AerodynamicEnv(gym.Env):
    """
    3D aerodynamic targeting environment (egocentric, target-facing frame).

    The agent (fixed point at the origin) throws a sphere at a randomised
    target disc.  Optionally, a wall blocks the direct line of sight and the
    ground plane is tilted so that gravity has a horizontal component.

    All observations and the yaw action are expressed in a frame whose +x axis
    points at the target — rotationally-equivalent scenarios collapse to the
    same observation, and yaw=0 means "throw straight at the target".

    Observation (7-dim):
        target_distance          – scalar distance from agent to target
        wall_forward_distance    – wall projected onto the agent→target axis
        wall_lateral_offset      – wall perpendicular component (signed)
        wall_width               – full width of the wall
        gravity_forward          – gravity along agent→target axis
        gravity_lateral          – gravity perpendicular to that axis
        gravity_z                – vertical gravity component

    Action (4-dim, [-1, 1]):
        pitch   – launch elevation,        mapped to [-45°, 45°]
        yaw     – offset from target bearing, mapped to [-90°, 90°]
        thrust  – initial speed,           mapped to [2, 22] m/s
        spin    – angular velocity (ω_z),  mapped to [-20, 20] rad/s
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def set_curriculum_phase(self, phase: int) -> None:
        """Runtime phase switch, invoked via VecEnv.env_method from the curriculum callback.

        Using a method (rather than set_attr) is required because wrappers like
        OneShotFlightWrapper don't forward attribute *writes* to the inner env;
        env_method correctly walks to the wrapped AerodynamicEnv via gym.Wrapper's
        __getattr__.
        """
        self.curriculum_phase = int(phase)

    def __init__(
        self,
        curriculum_phase: int = 1,
        render_mode=None,
        disable_spin_before_phase: int = 0,
    ):
        super().__init__()
        self.curriculum_phase = curriculum_phase
        self.render_mode = render_mode
        self.disable_spin_before_phase = disable_spin_before_phase

        # Load MuJoCo model
        xml_path = os.path.join(os.path.dirname(__file__), "scene.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # ---- Cache object IDs for fast per-step lookup ----
        self.ball_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "ball"
        )
        self.ball_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom"
        )
        self.floor_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        self.target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target"
        )
        self.wall_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "wall"
        )
        self.wall_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "wall_geom"
        )
        self.ball_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free"
        )

        # Index into qvel for the ball's freejoint (6 DOF: vx vy vz wx wy wz)
        self.ball_qvel_start = self.model.jnt_dofadr[self.ball_joint_id]

        # Mocap row indices (mocap_pos / mocap_quat arrays are indexed separately)
        self.target_mocap_id = self.model.body_mocapid[self.target_body_id]
        self.wall_mocap_id = self.model.body_mocapid[self.wall_body_id]

        # ---- Gymnasium spaces ----
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # Episode state
        self._thrown = False
        self._step_count = 0
        self._spin = 0.0
        self._hit_wall = False
        self._min_dist_seen = float("inf")

        # Cached per-episode quantities (populated in reset). The observation
        # is constant across the flight, so we compute it once and reuse.
        self._target_bearing = 0.0
        self._target_dist = 0.0
        self._wall_forward = 0.0
        self._wall_lateral = 0.0
        self._wall_width = 0.0
        self._cached_obs = np.zeros(7, dtype=np.float32)

        # Viewer (lazy-initialised for human rendering)
        self._viewer = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random  # seeded numpy RNG from gym.Env

        # Reset physics in-place so the passive viewer keeps a valid reference
        mujoco.mj_resetData(self.model, self.data)

        # Reset gravity to standard
        self.model.opt.gravity[:] = [0.0, 0.0, -9.81]

        # ---- Randomise target ----
        if self.curriculum_phase == 0:
            target_dist = rng.uniform(5.0, 10.0)
        else:
            target_dist = rng.uniform(9.0, 20.0)

        target_angle = rng.uniform(-np.pi, np.pi)
        target_x = target_dist * np.cos(target_angle)
        target_y = target_dist * np.sin(target_angle)
        self.data.mocap_pos[self.target_mocap_id] = [target_x, target_y, 0.01]

        # Cache the agent→target bearing and distance for the egocentric frame.
        self._target_bearing = float(target_angle)
        self._target_dist = float(target_dist)

        # ---- Wall placement ----
        # Phase 2 (easy wall): narrow, close to target, easy to dodge.
        # Phase 3+ (was phase 2): full range.
        wall_frac = 0.0
        wall_width = 0.0
        place_wall = False
        if self.curriculum_phase == 2:
            wall_frac = float(rng.uniform(0.45, 0.55))
            wall_width = float(rng.uniform(1.5, 2))
            place_wall = True
        elif self.curriculum_phase >= 3:
            wall_frac = float(rng.uniform(0.4, 0.6))
            wall_width = float(rng.uniform(1.5, 4))
            place_wall = True

        if place_wall:
            wall_x = target_x * wall_frac
            wall_y = target_y * wall_frac
            self.data.mocap_pos[self.wall_mocap_id] = [wall_x, wall_y, 5.0]
            # geom_size stores half-extents: [half_x, half_y, half_z]
            self.model.geom_size[self.wall_geom_id] = [0.1, wall_width / 2.0, 7.0]
            # Rotate wall so its wide face is perpendicular to the origin→wall direction
            angle = np.arctan2(wall_y, wall_x)
            self.data.mocap_quat[self.wall_mocap_id] = [
                np.cos(angle / 2),
                0.0,
                0.0,
                np.sin(angle / 2),
            ]
            # Egocentric wall coords (constant per episode)
            cb = np.cos(self._target_bearing)
            sb = np.sin(self._target_bearing)
            self._wall_forward = float(wall_x * cb + wall_y * sb)
            self._wall_lateral = float(-wall_x * sb + wall_y * cb)
            self._wall_width = float(wall_width)
        else:
            # Hide wall far away and reset orientation
            self.data.mocap_pos[self.wall_mocap_id] = [1000.0, 0.0, 5.0]
            self.model.geom_size[self.wall_geom_id] = [0.1, 1.0, 5.0]
            self.data.mocap_quat[self.wall_mocap_id] = [1.0, 0.0, 0.0, 0.0]
            self._wall_forward = 0.0
            self._wall_lateral = 0.0
            self._wall_width = 0.0

        # ---- Curriculum phase 4+: tilt gravity ----
        if self.curriculum_phase >= 4:
            tx_deg, ty_deg = rng.uniform(-15.0, 15.0, size=2)
            tx = np.deg2rad(tx_deg)
            ty = np.deg2rad(ty_deg)
            g = 9.81
            self.model.opt.gravity[:] = [
                g * np.sin(tx),
                g * np.sin(ty),
                -g * np.cos(tx) * np.cos(ty),
            ]

        # Run forward kinematics to apply mocap positions
        mujoco.mj_forward(self.model, self.data)

        self._thrown = False
        self._step_count = 0
        self._spin = 0.0
        self._hit_wall = False
        self._min_dist_seen = float("inf")

        # Build the (constant) egocentric observation once per episode.
        cb = np.cos(self._target_bearing)
        sb = np.sin(self._target_bearing)
        gx, gy, gz = (
            float(self.model.opt.gravity[0]),
            float(self.model.opt.gravity[1]),
            float(self.model.opt.gravity[2]),
        )
        self._cached_obs = np.array(
            [
                self._target_dist,
                self._wall_forward,
                self._wall_lateral,
                self._wall_width,
                gx * cb + gy * sb,  # gravity_forward
                -gx * sb + gy * cb,  # gravity_lateral
                gz,
            ],
            dtype=np.float32,
        )

        return self._cached_obs.copy(), {}

    def step(self, action):
        # Apply throw on the very first step of each episode
        if not self._thrown:
            self._apply_throw(action)
            self._thrown = True

        # Apply Magnus effect before advancing physics
        self._apply_magnus()

        mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        obs = self._get_obs()
        terminated = self._check_floor_contact()
        truncated = self._step_count >= MAX_EPISODE_STEPS

        # Track wall contact across the whole flight (phase 2+ only)
        if self.curriculum_phase >= 2 and not self._hit_wall:
            self._hit_wall = self._check_wall_contact()

        _bp = self.data.xpos[self.ball_body_id]
        _tp = self.data.mocap_pos[self.target_mocap_id]
        _dx = _bp[0] - _tp[0]
        _dy = _bp[1] - _tp[1]
        dist = float((_dx * _dx + _dy * _dy) ** 0.5)
        if dist < self._min_dist_seen:
            self._min_dist_seen = dist
        # Quadratic proximity bonus: ramps from 0 at 6 m to +3 at 0 m
        prox = max(0.0, (5.0 - dist) / 5.0)
        # Closest-approach shaping, scaled by how close the ball actually
        # landed to the target. Throws that miss by more than 4 m get zero
        # credit, killing the "throw short of the wall" and "spin past the
        # target" exploits where the trajectory grazed the target but the
        # ball never landed near it.
        prox_min = max(0.0, (6.0 - self._min_dist_seen) / 6.0)
        landing_factor = max(0.0, (3.0 - dist) / 3.0)
        # Wall-hit penalty only applies at terminal step so it sums once per episode
        wall_penalty = (
            -5.0 * float(self._hit_wall) if (terminated or truncated) else 0.0
        )
        reward = (
            -dist
            + 2.0 * (prox**2)  # bonus within 5 m of landing
            + 1.5 * float(dist < 2.0)  # stepping-stone bonus inside 2 m
            + 5.0 * float(dist < 1.0)  # success bonus inside 1 m
            + 2.0 * (prox_min**2) * landing_factor  # gated closest-approach
            + wall_penalty  # -5 if ball hit the wall at any point
        )

        info = {}
        if terminated or truncated:
            info["landing_dist"] = dist
            info["min_dist"] = float(self._min_dist_seen)
            info["success"] = dist < 1.0
            info["hit_wall"] = self._hit_wall

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
        elif self.render_mode == "rgb_array":
            renderer = mujoco.Renderer(self.model, height=480, width=640)
            renderer.update_scene(self.data)
            return renderer.render()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_throw(self, action):
        """Convert normalised action to initial ball velocity.

        Yaw is target-relative: yaw_norm=0 throws straight at the target,
        yaw_norm=±1 corresponds to ±YAW_OFFSET_MAX (±90°) off the bearing.
        This removes the wraparound discontinuity that the world-frame yaw
        had at ±π, and gives the policy a sensible identity init.
        """
        pitch_norm, yaw_norm, thrust_norm, spin_norm = action

        pitch_rad = float(pitch_norm) * (np.pi / 4)  # [-45°, 45°]
        yaw_rad = self._target_bearing + float(yaw_norm) * YAW_OFFSET_MAX
        speed = (float(thrust_norm) + 1.0) / 2.0 * 20.0 + 2.0  # [2, 22] m/s
        spin_radps = float(spin_norm) * 20.0  # [-20, 20] rad/s

        # During training, zero spin before the wall phase to prevent the
        # Gaussian policy from collapsing to ±1 on an irrelevant dimension.
        if self.curriculum_phase < self.disable_spin_before_phase:
            spin_radps = 0.0

        vx = speed * np.cos(pitch_rad) * np.cos(yaw_rad)
        vy = speed * np.cos(pitch_rad) * np.sin(yaw_rad)
        vz = speed * np.sin(pitch_rad)

        s = self.ball_qvel_start
        self.data.qvel[s : s + 3] = [vx, vy, vz]
        self.data.qvel[s + 3 : s + 6] = [0.0, 0.0, spin_radps]  # ω_z only

        self._spin = spin_radps

    def _apply_magnus(self):
        """Apply Magnus lift force: F = MAGNUS_K * (ω × v).

        Uses xfrc_applied (SET, not ADD) to avoid force accumulation across
        steps.  mj_resetData clears xfrc_applied at the start of each episode.
        Scalar cross product avoids numpy dispatch overhead and array copies.
        """
        s = self.ball_qvel_start
        vx = self.data.qvel[s]
        vy = self.data.qvel[s + 1]
        vz = self.data.qvel[s + 2]
        ox = self.data.qvel[s + 3]
        oy = self.data.qvel[s + 4]
        oz = self.data.qvel[s + 5]
        # xfrc_applied shape: (nbody, 6) — [fx, fy, fz, tx, ty, tz] in world frame
        xfrc = self.data.xfrc_applied[self.ball_body_id]
        xfrc[0] = MAGNUS_K * (oy * vz - oz * vy)
        xfrc[1] = MAGNUS_K * (oz * vx - ox * vz)
        xfrc[2] = MAGNUS_K * (ox * vy - oy * vx)

    def _get_obs(self):
        """Return the cached 7-dim egocentric observation.

        The observation is constant across an episode (target/wall/gravity are
        fixed at reset and the agent is a static point at the origin), so we
        compute it once in reset and return a copy here.
        """
        return self._cached_obs.copy()

    def _check_floor_contact(self) -> bool:
        """Return True if the ball geom is in contact with the floor geom."""
        # Fast path: ball is still airborne (ball radius = 0.1 m, guard at 0.15 m)
        if self.data.xpos[self.ball_body_id, 2] > 0.15:
            return False
        ball = self.ball_geom_id
        floor = self.floor_geom_id
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if (g1 == ball or g2 == ball) and (g1 == floor or g2 == floor):
                return True
        return False

    def _check_wall_contact(self) -> bool:
        """Return True if the ball geom is currently touching the wall geom."""
        ball = self.ball_geom_id
        wall = self.wall_geom_id
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if (g1 == ball or g2 == ball) and (g1 == wall or g2 == wall):
                return True
        return False


class OneShotFlightWrapper(gym.Wrapper):
    """Collapse a full ball flight into one agent decision.

    The raw env applies the throw on its first internal step and ignores the
    action on every subsequent step — so ~130 of the ~131 transitions per
    episode are counterfactual noise from PPO's perspective, corrupting credit
    assignment. This wrapper drives the inner physics to completion inside a
    single outer step() call and returns the terminal reward from the inner env
    (which already includes distance penalties, proximity bonuses, and success
    bonuses), turning the task into a clean contextual bandit.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, True, False, info
