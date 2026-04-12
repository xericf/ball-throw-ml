# Project Plan: RL Physics Sandbox & Curriculum Learning

## Phase 1: Infrastructure & Dependencies (Week 1)

Establish the software pipeline on the M3 Max.

- **Set up the Virtual Environment:** Create a clean Python environment (using `conda` or `venv`) specifically for this project to prevent package conflicts.
- **Install Core Libraries:**
    ```bash
    pip install gymnasium mujoco stable-baselines3[extra]
    ```
- **Create the Project Structure:**
    - `envs/`: Folder for custom Gym environment.
    - `models/`: Folder to save trained weights.
    - `logs/`: Folder for TensorBoard data.
    - `train_ppo.py`: Gradient-based training script.
    - `train_ga.py`: Gradient-free baseline script.
    - `enjoy.py`: Script to launch MuJoCo GUI and visualize the agent.

---

## Phase 2: Building the MuJoCo World (Weeks 2-3)

Engineering the physics sandbox before implementing algorithms.

- **The XML Scene (`scene.xml`):** Define a static floor, a $1 \times 1 \times 1$ cube (agent), a sphere (ball), a cylinder (target disc), and a tall box (wall).
- **The Gymnasium Class (`aerodynamic_env.py`):** Inherit from `gymnasium.Env`.
    - **`__init__`**: Define `observation_space` (Target $\Delta x/\Delta z$, Wall $\Delta x/\Delta z$, Wall Width, Gravity vector) and `action_space` (Pitch, Yaw, Thrust, Spin).
    - **`reset()`**: Logic to randomize target/wall positions and wall width based on the curriculum phase.
    - **`step(action)`**:
        - Map actions to physical forces.
        - **The Magnus Math:** Calculate velocity vector $\vec{v}$. Calculate the cross product of velocity and chosen "spin." Apply as lateral force using `mujoco.mj_applyFT`.
        - **Reward:** Calculate dense reward (Euclidean distance to target).
        - **Truncation:** End episode if ball hits ground or 3 seconds pass.

---

## Phase 3: Training Pipeline & Curriculum (Week 4)

Hooking the environment to the "brain."

- **The Base PPO Script:** Use `SubprocVecEnv` to run 32 parallel copies on the M3 Max and initialize Stable Baselines3 PPO.
- **The Curriculum Callback:** Write a `CurriculumManagerCallback` to monitor the rolling average success rate.
    - If success rate $> 70\%$, update environment variables to unlock the next phase (e.g., spawning the wall).
- **The "Fake Tilt" Trick:** In `reset()`, use a rotation matrix on the MuJoCo gravity vector (`model.opt.gravity`) to simulate $\pm 15^\circ$ tilt without rotating the floor mesh.

---

## Phase 4: The Application Bake-Off (Week 5)

Implement the Genetic Algorithm (GA) baseline for comparison.

- **The GA Script:** Use `PyGAD` or a basic Evolutionary Strategy (ES) loop in PyTorch.
- **The GA Training Loop:** Instantiate 100 neural networks with random weights.
    - Run through environment.
    - Select top 10, add Gaussian noise (mutation), and spawn the next generation.
- **Data Logging:** Log "Mean L2 Distance Error" and "Success Rate" to CSV or TensorBoard.

---

## Phase 5: Evaluation & Report Writing (Week 6)

Analysis and final documentation.

- **The Ablation Study:** Run a PPO session where the agent is "blind" to the explicit gravity vector and wall width.
- **Generate Graphs:** Use `matplotlib` to plot:
    - PPO vs. GA learning curves.
    - Full Observation vs. Blind Observation curves.
- **Draft the 6-Page Report:** Follow course requirements, emphasizing mathematical rigor and comparative results.
