# CPSC 440 Project Proposal: Deep RL Benchmark for Non-linear Aerodynamic Targeting

**Authors:** Eric Fu, Dylan Lau [cite: 2, 3]  
**Date:** March 29, 2026

---

## 1 Problem Focus

This project focuses on the challenge of generalized trajectory planning and optimal control in environments governed by non-linear aerodynamics and dynamic gravitational reference frames. While standard projectile motion is analytically solvable, introducing aerodynamic drag, the Magnus effect (spin-dependent lateral forces), and dynamic terrain slopes renders closed-form kinematic equations computationally intractable for real-time targeting.

Furthermore, existing Reinforcement Learning (RL) physics benchmarks often rely on static targets or fixed obstacles. This static geometry frequently leads to "policy collapse," where the neural network achieves high rewards through positional memorization of a single trajectory rather than learning the underlying continuous physical manifolds. The core problem we are investigating is whether continuous-control deep reinforcement learning algorithms can learn a generalized, robust heuristic for non-linear ballistics when forced to dynamically calculate spin and thrust to bypass randomized line-of-sight obstructions on shifting planes.

## 2 Proposed Plan

We built a novel 3D continuous-control environment using MuJoCo to benchmark RL and genetic algorithms on complex, non-linear aerodynamic tasks. The environment is a 2-dimensional flat plane that consists of:

- An agent with initial randomized orientation and position.
- A randomized target disc.
- An infinitely tall obstacle of randomized width blocking direct line-of-sight.

Furthermore, the ground plane will be subjected to randomized biaxial tilts $(\pm15^{\circ})$, converting standard gravity into a dynamic drift vector. The agent will be trained to throw a small rigid body sphere directly on the target disc; it will be scored based on the initial landing spot's distance towards the center of the disc[cite: 15, 16].

For controls, the agent is a fixed point on the plane. It outputs continuous actions for pitch, yaw, thrust, and angular velocity (spin) to utilize the Magnus effect, curving the projectile around the obstacle while compensating for the slope's gravitational pull[cite: 19, 20]. We use **Stable Baselines3** to train an agent via **Proximal Policy Optimization (PPO)** alongside a gradient-free **Genetic Algorithm (GA)**, both using a 5-phase curriculum learning approach.

To simplify credit assignment, we wrap the environment in a `OneShotFlightWrapper` that converts each episode into a single-step contextual bandit: the agent outputs one action, the full ball flight is simulated internally by MuJoCo, and a terminal reward is returned. This eliminates hundreds of no-op steps and makes the task tractable for both PPO and GA without reward shaping across time.

## 3 Experimental Design

To ensure structured learning and rigorous benchmarking, the environment and training loop will be parameterized as follows:

### State and Action Spaces

To prevent policy collapse and encourage generalized learning, the agent will be provided with explicit, relative features rather than absolute global coordinates.

**Observation Space (7-dimensional, egocentric):**

All observations are expressed in an egocentric frame with the agent at the origin and $+x$ pointing toward the target, preventing positional memorization.

1. Target distance (scalar)
2. Wall forward distance (projection of wall position along agent→target axis)
3. Wall lateral offset (signed perpendicular component)
4. Wall width $w$
5. Gravity forward $g_x$ (gravity component along agent→target axis)
6. Gravity lateral $g_y$ (gravity component perpendicular to agent→target axis)
7. Gravity vertical $g_z$

**Action Space:**
The continuous action space $\mathcal{A}\in[-1,1]^{4}$ consists of pitch, yaw, forward thrust, and lateral angular velocity (spin).

### Curriculum Schedule

We implemented a 5-phase curriculum to ensure the policy receives positive initial reward signals[cite: 34, 35]. Phase promotion occurs when the rolling success rate exceeds 75% for at least 50 consecutive episodes (PPO) or when elite-only success exceeds 75% (GA).

| Phase | Target Range | Wall | Gravity |
|-------|-------------|------|---------|
| 0 | 5–10 m | None | Standard |
| 1 | 9–20 m | None | Standard |
| 2 | 9–20 m | Narrow (1.5–2 m wide) | Standard |
| 3 | 9–20 m | Wide (1.5–4 m wide) | Standard |
| 4 | 9–20 m | Wide (1.5–4 m wide) | Tilted $\pm15^{\circ}$ |

Phases 0–1 build basic targeting and range estimation. Phase 2–3 introduce the obstacle and require spin to curve around the wall. Phase 4 adds biaxial gravity tilt, requiring simultaneous obstacle avoidance and gravitational drift compensation[cite: 37, 38].

### Algorithm Comparison and Benchmarking

We conducted a comparative analysis between **PPO** and a **Genetic Algorithm (GA)**. The GA evolves a 7→64→64→4 MLP (Tanh activations, ~4,932 parameters per genome) using rank-based elitism: a population of 100 genomes is evaluated per generation, the top-20 elites survive unchanged, and offspring are produced by Gaussian mutation (10% of weights perturbed, $\sigma=0.1$). Crossover is disabled as weight-space crossover harms neuroevolution due to permutation symmetry. PPO uses the same MLP architecture (64→64 hidden layers) via Stable Baselines3's MlpPolicy with VecNormalize across 14 parallel environments. Performance is evaluated by:

- Mean episodic reward.
- $L_{2}$ distance error from the target center.
- Success rate (percentage of shots landing within a 1-meter radius).

## 4 Contribution

The primary contribution is the open-source release of this environment as a mathematically rigorous benchmark for optimal control under non-linear aerodynamic conditions. Secondarily, we provide an empirical evaluation of PPO's sample efficiency against a GA baseline across all 5 curriculum phases. We find that providing explicit egocentric gravitational vectors and relative obstacle observations is critical to preventing policy collapse. Failure modes and negative results — particularly the GA's prohibitive sample cost on Phase 2+ — are documented with full analysis.
