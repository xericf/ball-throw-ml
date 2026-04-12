# CPSC 440 Project Proposal: Deep RL Benchmark for Non-linear Aerodynamic Targeting

**Authors:** Eric Fu, Dylan Lau [cite: 2, 3]  
**Date:** March 29, 2026

---

## 1 Problem Focus

This project focuses on the challenge of generalized trajectory planning and optimal control in environments governed by non-linear aerodynamics and dynamic gravitational reference frames. While standard projectile motion is analytically solvable, introducing aerodynamic drag, the Magnus effect (spin-dependent lateral forces), and dynamic terrain slopes renders closed-form kinematic equations computationally intractable for real-time targeting.

Furthermore, existing Reinforcement Learning (RL) physics benchmarks often rely on static targets or fixed obstacles. This static geometry frequently leads to "policy collapse," where the neural network achieves high rewards through positional memorization of a single trajectory rather than learning the underlying continuous physical manifolds. The core problem we are investigating is whether continuous-control deep reinforcement learning algorithms can learn a generalized, robust heuristic for non-linear ballistics when forced to dynamically calculate spin and thrust to bypass randomized line-of-sight obstructions on shifting planes.

## 2 Proposed Plan

We plan to build a novel 3D continuous-control environment using MuJoCo to benchmark RL and genetic algorithms on complex, non-linear aerodynamic tasks. The environment is a 2-dimensional flat plane that consists of:

- An agent with initial randomized orientation and position.
- A randomized target disc.
- An infinitely tall obstacle of randomized width blocking direct line-of-sight.

Furthermore, the ground plane will be subjected to randomized biaxial tilts $(\pm15^{\circ})$, converting standard gravity into a dynamic drift vector. The agent will be trained to throw a small rigid body sphere directly on the target disc; it will be scored based on the initial landing spot's distance towards the center of the disc[cite: 15, 16].

For controls, the agent will be a fixed point on the plane. It will have access to rotating its body in any direction and throwing the sphere with a set of parameters. To hit the target, the agent must output continuous actions for pitch, yaw, thrust, and angular velocity (spin) to utilize the Magnus effect, curving the projectile around the obstacle while compensating for the slope's gravitational pull[cite: 19, 20]. We will use **Stable Baselines3** to train an agent via **Proximal Policy Optimization (PPO)** alongside a gradient-free **Genetic Algorithm (GA)**, utilizing a curriculum learning approach.

## 3 Experimental Design

To ensure structured learning and rigorous benchmarking, the environment and training loop will be parameterized as follows:

### State and Action Spaces

To prevent policy collapse and encourage generalized learning, the agent will be provided with explicit, relative features rather than absolute global coordinates.

**Observation Space:**

- Target Delta: $(\Delta x_{t},\Delta z_{t})$
- Obstacle Delta: $(\Delta x_{w},\Delta z_{w})$
- Obstacle Width: $w$
- Local Gravity Vector: $(g_{x},g_{z})$

**Action Space:**
The continuous action space $\mathcal{A}\in[-1,1]^{4}$ consists of pitch, yaw, forward thrust, and lateral angular velocity (spin).

### Curriculum Schedule

We will implement a 3-phase curriculum to ensure the policy receives positive initial reward signals[cite: 34, 35]:

1.  **Phase 1 (Basic Targeting):** Flat plane, no obstacle. The agent learns the parabolic mapping of distance to pitch and thrust.
2.  **Phase 2 (Aerodynamic Bypassing):** Flat plane, randomized obstacle introduced. The agent learns to utilize the Magnus effect to curve around the wall[cite: 37, 38].
3.  **Phase 3 (Dynamic Drift):** Tilted plane $(\pm15^{\circ})$, randomized obstacle. The agent learns to compensate for gravitational drift.

### Algorithm Comparison and Benchmarking

We will conduct a comparative analysis between **PPO** and a **Genetic Algorithm (GA)**. GAs may prove more robust in Phase 3 where dynamic gravity introduces severe non-linearities. Performance will be evaluated by:

- Mean episodic reward.
- $L_{2}$ distance error from the target center.
- Success rate (percentage of shots landing within a 1-meter radius).

## 4 Expected Contribution

The primary contribution will be the open-source release of this environment as a mathematically rigorous benchmark for optimal control under non-linear aerodynamic conditions. Secondarily, we will provide an empirical evaluation of PPO's sample efficiency against a GA baseline. We aim to determine whether providing explicit gravitational vectors and relative obstacle width prevents policy collapse compared to a baseline trained without these features. Failure modes or negative results under extreme tilt parameters will be documented with full analysis.
