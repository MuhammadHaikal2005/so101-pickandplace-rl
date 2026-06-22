# v3 and v4 Reward Tweaks — Policy Still Refuses To Release

> Date: 2026-05-14
> Project / Stage: SO-101 sim-to-real for GR00T N1 fine-tuning — reward iteration on the hold-over-target attractor
> Topic: Two reward additions (descent gradient, release-at-target bonus) failed to break the policy out of the "hold cube above target" local optimum

## What we accomplished

- Designed, implemented, trained, evaluated, and diagnosed reward v3 (descent reward when grasped over target).
- Designed, implemented, trained, evaluated, and diagnosed reward v4 (release-at-target one-time bonus).
- Confirmed that both interventions leave the policy in the same local optimum that beat v2: full grasp + lift + transport, but never release. Across both runs, success rate stayed at 0 percent despite eval reward holding at 935 to 950.
- Established that the limiting factor is **exploration**, not reward shape: small reward additions in the regime where the policy already operates cannot pull it out of an attractor it has trained itself into.
- Added a new state variable `_release_bonus_fired` to the env and a new info key `bonus_release_at_target`. Kept the env interface stable for any consumers.
- Maintained 13/13 unit tests passing across both reward iterations.

## Walkthrough — what we did and why

### v3 training and post-mortem

After pt03 we committed reward v3 (additive `reward_descent` term that fires when grasped and within 5 cm XY of the target, scaling with `1 - tanh(20 * cube_lift)`) and started a 500k SAC training run. It completed in 72 minutes wall time.

Result: eval reward 948.73 ± 20.54, success rate 0 percent. Almost identical to v2's 949.92.

Diagnostic on the deterministic best model showed exactly the same failure as v2 — the cube is grasped, lifted to ~17 cm above table, transported to within 4-6 cm of the target XY, and held there for the rest of the episode. Reward composition:

- reach 0.24 + grasp 1.0 + lift 4.0 + transport 0.73 + descent **~0.01** = 5.98 per step

The descent reward at z=0.17 evaluates to `2 × (1 - tanh(20 × 0.16)) ≈ 2 × 0.005 ≈ 0.01`. The tanh ramp was far too steep — descent reward only meaningfully fires for `cube_lift < 0.05 m`, but the policy holds the cube at `cube_lift ≈ 0.16 m`. The new term never activated and therefore never pulled the policy lower. v3 trained against essentially the v2 reward function.

### v4 design — release-at-target bonus

The descent gradient approach assumed the policy might gradually lower the cube if rewarded for being lower. But the policy never enters the regime where the descent term has any value. A different approach: reward the **discrete event** of releasing the cube, regardless of how it got there.

Added a one-time bonus that fires the instant the policy transitions from grasping to not-grasping, provided the cube is over the target zone and still in the air:

```python
bonus_release_at_target = 0.0
if (
    not self._release_bonus_fired
    and self._has_grasped
    and not in_contact
    and cube_target_xy < 0.05
    and cube_z > (TABLE_Z + 0.005)
):
    bonus_release_at_target = 20.0
    self._release_bonus_fired = True
```

Reasoning: if the policy stochastically explores `action_jaw > 0` (opening) once, even briefly, this bonus fires immediately for +20. After release the cube falls to the table; if it lands within the place threshold (which it should given the policy positions over target), the existing place bonus (+5) and the 5-frame success criterion fire, awarding +1000 and terminating. Total release outcome ≈ +1025 versus continuing to hold for ~6 per step × N remaining steps.

Added `_release_bonus_fired` to both `__init__` and `reset` so the bonus state resets per episode. Added `bonus_release_at_target` to the info dict and to the reward sum. Verified all 13 unit tests still pass.

### v4 training and post-mortem

500k step SAC run completed in ~71 minutes.

Eval progression matched v2 and v3 closely:

| Timesteps | Mean reward |
| --- | --- |
| 25k | 36 |
| 100k | 44 |
| **150k** | **288** ← discovered grasping |
| 175k | 905 |
| 200k | 215 (regression) |
| 275k | 912 |
| 425k | 931 |
| 475k | 960 |
| 500k | 904 |

Final mean reward 904, std 30. **Success rate 0 percent.**

Render of 5 deterministic episodes:

| Episode | Reward | cube_target_xy | cube_z |
| --- | --- | --- | --- |
| 0 | 973.20 | 0.043 m | 0.110 m |
| 1 | 950.54 | 0.086 m | 0.162 m |
| 2 | 860.22 | 0.221 m | 0.090 m |
| 3 | 975.04 | 0.046 m | 0.113 m |
| 4 | 916.23 | 0.090 m | 0.167 m |

The cube ends each episode in the air, mostly over the target zone (3 episodes within 5 cm, one at 22 cm — episode 2 missed the target XY). The policy never released. The release bonus condition `not in_contact and self._has_grasped and cube over target` was never satisfied in the deterministic eval rollout, so the bonus had no effect on the gradient — the policy never learned what releasing pays because it never released even once during deterministic playback.

### The pattern across v2, v3, v4

All three reward iterations produced an eval reward in the 905-950 band with 0 percent success. The underlying behavior is identical: reach, grasp, lift, transport, then hold the cube high over the target indefinitely.

Each iteration added more reward shaping in the regime the policy was already operating in, without changing the regime. The exploration constraint is the binding one: SAC's policy distribution at convergence does not include "open the gripper above the target" as a probable action. Without ever sampling that action, no positive evidence for release can enter the replay buffer, and the value function cannot learn that release is preferable.

Three remaining options for the next attempt, in increasing order of intervention:

- **Force higher entropy** by setting `target_entropy` manually (or `ent_coef="auto_X"` with a larger X) so SAC keeps exploring all action dimensions.
- **Reduce dense-reward magnitudes** dramatically so the success bonus (+1000) absolutely dominates the math, making the policy's value function strongly prefer release.
- **Seed the replay buffer with demonstrations** — hand-script a few successful trajectories (reach, grasp, lift, transport, release) and inject them into SAC's buffer at the start of training. This is closer to what the field calls "demonstration-bootstrapped RL".

Not yet decided which direction to take; pending the user's call after watching the v4 demo.

## Problems hit and how we fixed them

### v3 descent reward never activated

- **What happened:** SAC v3 trained to 949 reward, identical behavior to v2. Diagnostic showed the policy holds the cube at z ≈ 0.17 m for the entire post-transport phase; descent reward at that height evaluates to ~0.01 per step.
- **Why it happened:** The descent formula `2.0 * (1.0 - tanh(20.0 * cube_lift))` saturates rapidly. At `cube_lift = 0.05 m`, `tanh(1.0) ≈ 0.76`, descent reward ≈ 0.48. At `cube_lift = 0.10 m`, descent reward ≈ 0.1. At `cube_lift = 0.17 m`, ≈ 0.01. The activation region of the term does not overlap the region where the policy operates.
- **How we fixed it (or did not):** Left v3 in place as a documented step. For v4 the approach changed entirely — bonus on the release event rather than a continuous gradient.
- **Lesson learned:** When adding a new reward term as a shaping signal, verify its value across the actual state distribution the policy occupies, not just at the boundary cases. A term whose activation region does not intersect the policy's working region is functionally a no-op.

### v4 release bonus also failed

- **What happened:** SAC v4 trained to 904 reward average, 0 percent success. Deterministic eval policy never releases the cube; the release bonus condition (`grasped` → `not grasped` over target) is never met.
- **Why it happened:** The release bonus only fires when release happens. The policy has converged on a deterministic strategy that never opens the gripper over the target. SAC's `ent_coef` auto-tunes downward as the policy commits to its strategy; by convergence the action distribution has very low variance and "open gripper" actions are sampled too rarely to land a release that gets stored in the replay buffer with the bonus.
- **How we fixed it (or did not):** Acknowledged the limit of pure reward shaping for this case. Three escalation paths identified for the next session.
- **Lesson learned:** Reward design alone cannot teach a behavior the policy never executes. For rare events at convergence (release-while-grasping for a policy that prefers holding), the exploration mechanism must be addressed directly — through entropy, demonstrations, or reward magnitudes that make the alternative strategy untenable.

## Concepts clarified

### Why a continuous shaping reward cannot rescue an out-of-distribution region

A dense shaping term provides gradient via its derivative with respect to state. If the policy already operates in a region where that derivative is near zero (because the term has saturated, like a tanh with steep slope), the policy receives no useful gradient from the term. Adding the term is mathematically the same as adding a constant in that region. v3 illustrated this concretely: at `cube_lift = 0.17 m` the descent term contributes 0.01 to reward and ~0 to ∂reward/∂cube_z. The policy correctly ignored it.

### Why discrete-event bonuses also fail without exploration of the event

A bonus that fires on a state transition (here, contact → no-contact while over target) can only enter the value function via samples in the replay buffer. SAC bootstraps the Q-function from sampled transitions. If the transition is never sampled (because the policy's deterministic action never releases, and its stochastic samples have low enough variance that release is rare), the bonus has no effect. v4 illustrated this: the bonus is correctly defined and correctly wired, but never triggered during the runs that populated the replay buffer with high-reward holding patterns.

### Local optima in continuous-control RL

A local optimum in this setting is not a single point but a *region* of policy space where any small parameter change reduces expected return. The hold-over-target attractor is locally stable because:

- Any small change to the action that reduces grip force risks losing contact (penalty drop, loss of grasp/lift/transport rewards).
- Any small change to arm motion that moves the cube risks reducing the transport term (cube moves away from target).
- Any small change that opens the gripper transitions to a region (cube falling) where the value function has not been trained, so its predicted value is approximately zero, which is worse than the certain ~5.5 reward per step the policy is currently collecting.

The policy's local linearization sees no profitable direction. Escape requires either large-step exploration that lands in a different region, or a redrawing of the value function via off-policy samples from a different policy distribution (the demonstration-bootstrapping idea).

## Where things stand now

```
so101-rl-poc/runs/
├── ppo_v1..v7/                 (historical PPO failures)
├── sac_lift_v1/                (gym-lowcostrobot failure)
├── sac_cart_v1/                (reach trap, 160 reward, 0%)
├── sac_cart_v2/                (close to success, 950 reward, 0%)
├── sac_cart_v3/                (descent reward, 949 reward, 0%)
└── sac_cart_v4/                (release bonus, 904 reward, 0%)
    ├── eval/best_model.zip
    └── demo.mp4
```

Latest pushed git commit: the v3 commit (`3cba62f`). v4 changes are committed locally but I have not yet pushed them. v4 training artefacts (final.zip, best_model.zip, demo.mp4, train log) are on disk but not yet selectively added to git.

Conda env `so101-rl` (Python 3.11) unchanged. All 13 unit tests passing.

Three iterations of reward additions on top of the v2 baseline have not increased success rate above 0 percent. The policy reliably picks, lifts, and transports — but cannot be cajoled into releasing via reward shaping in the regime it now operates in.

## What's next

The reward-shaping path is exhausted for this task with this exploration setup. Three concrete next directions, listed in order of how aggressively they intervene on the policy's exploration:

- **Force higher SAC entropy.** Replace `ent_coef="auto"` with `ent_coef="auto_1.0"` (larger target entropy) or set a manual floor. Goal: keep the action distribution wide enough that release events get sampled and enter the replay buffer.
- **Slash all dense-reward magnitudes** so the success bonus dominates the cumulative-return math. For example, divide reach/grasp/lift/transport coefficients by 10 — holding for 200 steps becomes worth ~120, success bonus stays at 1000. The value function should then strongly favor releasing.
- **Demonstration-bootstrapped SAC**: write a short scripted trajectory that performs the full pick-and-place (waypoints driven by IK, then `action_jaw > 0` over target), record it as transitions, inject into SAC's replay buffer before training begins. This teaches the policy what a successful episode looks like before it has to discover it.

Pending the user's call on which to try first. Personally I lean toward direction 1 (lowest engineering cost, highest probability of working) followed by direction 3 if entropy alone is insufficient. Direction 2 is the riskiest because it discards much of the dense shaping that made v2 onward find the grasp at all.

Additional housekeeping for the next session:

- Push the v4 commit + v4 artefacts to GitHub.
- Possibly extract `_release_bonus_fired` and `_first_lift_fired` etc. into a single bookkeeping struct if the count of one-time event flags keeps growing.
- If the next iteration succeeds, update the README to reflect the working recipe and write a final progress note declaring the proof of concept complete.
