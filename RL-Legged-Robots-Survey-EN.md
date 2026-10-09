# Reinforcement Learning for Motion Planning and Control of Legged Robots: Research Survey, 2024–2026

**Research topic:** Motion planning and control of legged robots based on reinforcement learning (RL).

**Literature search cutoff: October 10, 2026.** This report reviews selected representative results from 2024–2026, focusing on humanoid robots and including quadruped research. It is not an exhaustive systematic review. Publication years primarily refer to the first preprint release; conference or journal publication years are stated separately. Some recent works remain preprints, so their availability should not be interpreted as evidence of conference or journal acceptance. The report draws on papers, official project pages, and repository documentation. No reproduction experiments were conducted, and results from different papers are not ranked directly against one another.

## 1. Scope and Core Concepts

Legged robots include bipeds, humanoids, and quadrupeds. Motion planning determines paths, footholds, contact timing, or whole-body reference trajectories. Motion control generates commands from the robot's current state to achieve the requested behavior while maintaining stability.

| Level | Typical inputs | Typical outputs | Main question |
|---|---|---|---|
| Task and path planning | Destination, scene, task requirements | Path or local goals | Where should the robot go? |
| Motion and contact planning | Terrain, state, local goals | Footholds, contact sequences, whole-body references | How should it traverse or interact with the scene? |
| Motion control | Proprioception, reference, optional environmental perception | Joint targets or torques | How should the robot execute the behavior stably? |

These levels can be explicitly separated or partly integrated into a single policy. An end-to-end visual policy does not necessarily contain a separate planner, and a general motion tracker does not automatically provide autonomous navigation. In learning-based systems, whole-body control (WBC) sometimes broadly refers to a policy that coordinates the entire body. This usage should be distinguished from conventional WBC based on inverse dynamics and quadratic programming (QP).

## 2. Overview of Representative Works

| Work | Date | Main direction | Key method | Evaluation scope and reading focus |
|---|---|---|---|---|
| SoloParkour [2] | 2024 / CoRL 2024 | Constrained visual RL | Privileged-policy experience bootstraps off-policy visual learning | Physical Solo-12; constraints and sample efficiency |
| PIE [3] | 2024 | Visual quadruped parkour | Implicit–explicit estimation | Zero-shot hardware transfer; perception errors |
| ASAP [4] | 2025 | Sim-to-real transfer | Learn a residual action model from real data, then fine-tune the policy | Dynamic hardware motions; compensation for dynamics mismatch |
| GMT [5] | 2025 | General motion tracking | Adaptive sampling and Motion Mixture of Experts (MoE) | Multiple motions with one policy; balancing the training distribution |
| BeyondMimic [6] | First released in 2025 | Tracking and task composition | RL tracking, diffusion distillation, inference-time guidance | Dynamic hardware motions and downstream goal control |
| BFM [7] | 2025 | Behavior foundation models | Conditional variational autoencoder (CVAE) and masked online distillation | Reusable behavior representations and multiple control interfaces |
| SONIC [8] | First released in 2025; journal version in 2026 | Scaling general control | Scale data, models, and compute; unified motion interface | Hardware whole-body control and planner/VLA integration |
| RGMT [9] | January 2026 | Robust general tracking | Dynamics-conditioned attention and recovery curriculum | Unseen motions, disturbances, and hardware recovery |
| Walk the PLANC [10] | January 2026 | Foothold planning and RL | Reduced-order planner and control Lyapunov function (CLF) rewards | Hardware walking over constrained footholds |
| ULTRA [15] | First released in March 2026; IROS 2026 | Perception- and goal-driven whole-body manipulation | Physics-based retargeting, tracking teacher, multimodal distillation, RL fine-tuning | G1 transport and sparse-goal execution; one set of policy parameters |
| Scaling BFM [11] | July 2026 | Scaling general control | Global tracking, expanded interaction data, Humanoid Transformer | Simulation and hardware; mechanisms for scaling training |
| PGMT [12] | September 2026 | Terrain-aware tracking | Motion-conditioned terrain encoding and tracking relaxation | G1 on complex terrain, recovery, and teleoperation |
| ViBe [16] | September 2026 | Pretrained motor backbone and visual adaptation | Freeze SONIC and the vision encoder; train an extractor and LoRA with RL | G1 locomotion, manipulation, and dodgeball; separate task adaptations |
| PASSAGE [13] | September 2026 | Whole-body traversal planning | Scene-aligned data, flow-matching planning, RL post-training | Onboard perception and traversal of cluttered scenes |
| Generate, Track, Improve [14] | September 2026 | Generative planning and RL | Depth-conditioned flow matching, tracking policy, off-policy fine-tuning | Outdoor G1 locomotion with multiple skills |
| iGPC [17] | October 2026 | Motor priors and perceptive contact interaction | Retain the GPC backbone; task adaptation, perceptive distillation, input calibration | G1 target contact, support-assisted stabilization, and box pushing; no execution-time motion reference |

## 3. From Agile Motion to General Whole-Body Control

### 3.1 SoloParkour and PIE

SoloParkour [2] formulates quadruped parkour as constrained RL. A policy first learns with privileged environmental information. Its experience then bootstraps an off-policy algorithm trained from depth images, which is deployed on Solo-12. Its value lies in incorporating physical limits and the cost of visual learning into the training design. PIE [3] uses implicit–explicit estimation to improve the interpretation of proprioceptive state and environmental information, and demonstrates hardware transfer after a single simulation training stage.

### 3.2 ASAP: Using Real Data to Improve Simulation Training

ASAP [4] first learns a tracking policy in simulation using retargeted human motions. It then collects real execution data and trains a residual action model to compensate for dynamics mismatch. This model is incorporated into simulation before the control policy is fine-tuned.

The compensation incorporates information learned from real executions, which distinguishes it from standard domain randomization. This approach is relevant to tracking errors caused by actuators, contacts, and other modeling inaccuracies. Reproduction requires attention to real-data collection and hardware availability.

### 3.3 Distinguishing GMT, BeyondMimic, BFM, and SONIC

| Work | Main focus | Important distinction |
|---|---|---|
| GMT [5] | Unify tracking of multiple motions using adaptive sampling and MoE | Motion coverage and environmental adaptability are different capabilities |
| BeyondMimic [6] | High-quality tracking, with diffusion-based skill composition and task guidance | The tracking repository does not necessarily implement the complete diffusion-control pipeline |
| BFM [7] | Learn a reusable behavior latent space and multiple conditioning interfaces | A foundation model is not synonymous with a particular network architecture |
| SONIC [8] | Systematically scale data, models, and compute for general tracking | The tracker, kinematic planner, and vision-language-action (VLA) model serve different functions |

BeyondMimic addresses both motion acquisition and downstream task control. The initial BFM work uses a CVAE and masked online distillation. SONIC combines scaled motion tracking with a unified interface; its official page also demonstrates kinematic planning and integration with higher-level models. Together, these works advance reusable whole-body control. Demonstration videos alone, however, do not establish reliability across terrain or during long-horizon tasks.

## 4. Key Developments in 2026

### 4.1 RGMT: Robust Tracking with Imperfect References

RGMT [9] uses recent proprioceptive information, a causal temporal encoder, and cross-attention to aggregate a reference-motion window. This conditions the use of reference information on the current dynamics. Training also includes a recovery curriculum with randomly initialized unstable states and gradually reduced assistance forces.

The authors report approximately 3.5 hours of motion data, single-stage training without distillation, and transfer to unseen motions and physical hardware. The reported data volume describes the reference-motion collection; it does not imply low total interaction volume or GPU cost.

### 4.2 Walk the PLANC: Giving RL a Physical Planning Structure

PLANC [10] targets constrained footholds such as stepping stones and narrow boards. A reduced-order gait planner supplies dynamically consistent targets, while CLF rewards guide RL training, followed by hardware validation.

This offers a research entry point for someone with a control background: retain an interpretable planning structure and use RL to improve adaptation to errors and complex dynamics. The use of CLF rewards alone does not justify a claim of global stability for the complete learned system. Such a claim requires examining the theoretical assumptions and scope of the proofs.

### 4.3 Scaling BFM: Jointly Scaling Motion Diversity, Interaction Data, and Model Structure

Scaling BFM [11] emphasizes three coordinated design choices: global-frame behavior tracking, matching on-policy rollout volume to reference-motion diversity, and the Humanoid Transformer. It highlights the distinction between the size of the reference dataset and the amount of actual RL interaction data.

### 4.4 PGMT: Allowing Motion Adjustments for Terrain

PGMT [12] receives proprioceptive state and history, future reference motions, and a local terrain elevation map, and outputs whole-body joint targets. It first learns flat-ground tracking and recovery, then adds motion-conditioned terrain encoding to adapt references that would otherwise be unsuitable for the current terrain. Physical G1 demonstrations include obstacle conditions up to 37 cm high.

Part of the permitted adaptation is explicitly defined: on slopes, stairs, and boxes, lower-body link-position, link-orientation, and joint-position tracking objectives can be relaxed, while upper-body and other objectives retain their original requirements. This is implemented through rewards rather than hard constraint guarantees. Multiple critic heads separately evaluate upper-body, lower-body, terrain, and other returns, but deployment uses one actor. Locomotion, teleoperation, and recovery do not require switching among control policies.

The controller's scope is to coordinate terrain-aware execution of a supplied motion intent. The paper already demonstrates stair climbing with simultaneous upper-body teleoperation, so adding that combination alone is insufficient novelty. Elevation maps mainly represent geometry and provide limited understanding of object semantics and interaction purposes. Evaluation should measure both traversal and motion preservation.

### 4.5 PASSAGE and Generate, Track, Improve: Training Planners with Closed-Loop RL Feedback

PASSAGE [13] uses 100 hours of scene-aligned human motion from 1,500 scenes to train a conditional flow-matching planner, combined with a perceptive tracker and planner RL post-training. Its onboard system is tested in 50 unseen real layouts, emphasizing geometry-dependent selection and composition of whole-body traversal behaviors.

Generate, Track, Improve [14] generates whole-body references from depth images, executes them with a perceptive RL tracking policy, and improves the generator through off-policy RL. The authors report increases in traversal success of up to 25 percentage points under some test conditions and demonstrate outdoor G1 locomotion with multiple skills. This is an internal experimental result and should not be directly compared with success rates from different papers and test sets.

Both works point to a research direction in which planners learn from execution feedback which motions the current controller can reliably complete, in addition to fitting demonstrations.

### 4.6 ULTRA: One Policy for Motion Tracking and Visual Goal Execution

ULTRA [15] distinguishes two task inputs. Dense references specify intermediate motions; sparse goals primarily specify target poses for the robot's root or an object, leaving intermediate motion underdetermined. For box transport, the controller can receive a complete demonstration trajectory or only the box's target position.

Training proceeds through physically consistent retargeting, a tracking teacher with full state, multimodal student distillation, and RL fine-tuning for goal execution. Random input masking teaches the student to operate with missing references or observations. One set of student parameters supports reference tracking, sparse-goal control, and depth point-cloud perception. The teacher–student relationship belongs to training and does not constitute deployment-time policy switching.

Experiments include changes in object scale and goal position, as well as composed motions. These results do not substitute for a test that explicitly excludes selected combinations of simultaneous constraints from training. ULTRA combines learned motor capabilities with perception more broadly; it should not be characterized as ViBe-style adaptation with a frozen motor backbone and only trainable adapters.

### 4.7 ViBe: Freezing the Motor Backbone and Learning Visual Corrections

ViBe [16] reuses the SONIC whole-body tracker and a pretrained vision encoder, freezes both, and trains a visual extractor and LoRA adapters with PPO. The controller receives proprioceptive history, a reference-motion window, onboard RGB images, and an optional task command. Visual feedback directly affects actions, enabling adjustments to footholds, contact placement, or avoidance behavior.

The architecture and training recipe are shared, but separate adapted policies are trained for different tasks. Locomotion, transport, cube reorientation, and dodgeball demonstrations do not establish that one shared set of weights covers all tasks. In cube reorientation, a simple planner selects reference clips in the routine “right flip, right flip, front flip” until the requested color is on top. This changes the motion reference and should not be interpreted as switching among multiple control policies.

ViBe directly establishes a precedent for combining a frozen pretrained motor backbone, pretrained vision, and RL adaptation of a small number of modules. Sharing an adaptation module across tasks and generalizing to unseen task combinations require separate validation.

### 4.8 iGPC: Retaining a Generative Motor Backbone for Perceptive Interaction

iGPC [17] learns task adapters around GPC's motor-token representation, generative prior, and action decoder, adding conditioning on object-surface geometry, contact, and stability. Interaction experts learn target contact, environmental support, and object pushing without prescribed motion references during execution.

A perceptive student reconstructs the expert's inputs from depth, LiDAR, and proprioceptive history. Distillation supervises motor tokens and joint targets, followed by bounded input calibration and RL fine-tuning of task adapters. The motor backbone and decoder are retained rather than replaced by a new action network.

Tasks still use task adapters; sharing a backbone does not establish unseen-combination generalization with a unified policy. The paper uses Isaac Lab and MuJoCo in some student box-pushing evaluations, which does not mean all training is conducted in MuJoCo. Broader generalization and contact-safety evaluation remain future directions identified by the authors.

### 4.9 Interfaces and Generalization Boundaries of the Four Works

| Work | Perception input | Task interface | Policy organization | Boundary relevant to research novelty |
|---|---|---|---|---|
| PGMT [12] | Terrain elevation map | Motion references or motion commands converted by an upstream system | One whole-body actor; multiple critic heads | Terrain adaptation and upper-body control during stair climbing already exist; these differ from explicitly holding out combinations of sparse task constraints |
| ULTRA [15] | Depth point clouds, object state, and other modalities | Dense references or sparse root/object goals | One set of multimodal student parameters | Goal execution and composed motions already exist; using a single policy alone is not novel |
| ViBe [16] | RGB and pretrained visual features | Reference window and optional task command | Separate adapted policies for different tasks | A shared architecture and backbone do not imply shared parameters across all tasks; selecting a reference clip is not policy switching |
| iGPC [17] | Depth, LiDAR, proprioceptive history | Goals and interaction conditions; no execution-time motion reference | Retained GPC backbone with task adapters | Backbone reuse, perceptive interaction, and reference-free execution already exist; compositional generalization requires additional evidence |

Combining a motor backbone with perception is an established technical direction. A task definition should specify the controller's inputs, responsibilities, and success criteria. A generalization claim must additionally identify which factors or combinations were present during training. The absence of paired demonstrations does not establish that a combination was absent from RL rollouts. Demonstrations of multiple tasks also do not automatically establish generalization to unseen task combinations.

## 5. Open-Source Availability and Reproduction

The following table records official entry points confirmed during this review. A repository link does not establish that code, weights, data, training, and hardware deployment assets are all available. None of these repositories was installed or executed for this report.

| Work | Confirmed entry point | Scope or items requiring verification |
|---|---|---|
| ASAP | [Official repository](https://github.com/LeCAR-Lab/ASAP) | Implementation documentation is available; check hardware and real-data collection requirements |
| BeyondMimic | [Tracking training repository](https://github.com/HybridRobotics/whole_body_tracking) | Explicitly covers motion-tracking training; diffusion components require separate verification |
| Scaling BFM | [Official repository](https://github.com/zengweishuai/ScaleBFM) | Official entry point confirmed; release completeness was not checked component by component |
| SONIC | [Official project](https://nvlabs.github.io/GEAR-SONIC/) | The page provides Code and Docs links; verify individual models and training assets |
| SoloParkour | [Official project](https://gepetto.github.io/SoloParkour/) | A code entry point is provided; check installation and data dependencies |
| PLANC | [Official project](https://caltech-amber.github.io/planc/) | A code link is provided, but the target repository was not successfully read during this review |
| PGMT | [Official project](https://luyili.github.io/pgmt/) | The confirmed [GitHub repository](https://github.com/LuyiLi/pgmt) hosts the project website; complete training code was not confirmed |
| ULTRA | [Official repository](https://github.com/Sirui-Xu/ULTRA) | Retargeting, teacher, student, and fine-tuning scripts; IsaacGym training and a MuJoCo sim-to-sim entry point. Released teacher weights include data added after the paper, so check correspondence with the reported experiments |
| ViBe | [Official repository](https://github.com/lok-i/vibe) | Based on mjlab/MuJoCo-Warp; training, playback, model, and export entry points. Released models are task-specific and require SONIC, vision-model, and motion-data dependencies |
| iGPC | [Paper](https://arxiv.org/abs/2610.08120) | A complete training repository released by the authors was not confirmed; this review does not label it fully open source |
| Other entries | Papers and projects listed in the references | Complete runnable implementations were not confirmed during this review |

Before reproduction, check the simulator and version, robot model and joint ordering, motion-retargeting pipeline, observation and action scaling, training resources, access to weights and data, and sim-to-sim or hardware deployment interfaces. Using the same robot name does not imply interchangeable control configurations.

## 6. Candidate Research Questions

The following are research hypotheses motivated by the literature. They are neither established conclusions nor confirmed unaddressed gaps.

### Direction A: Adapt Tracking Objectives to Terrain and Current State

**Problem:** Fixed tracking weights may cause collisions when a reference is unsuitable for the terrain. Excessive relaxation may instead discard the intended motion.

**Hypothesis:** Dynamically adjusting local goals or weights from terrain, proprioceptive history, and tracking error can improve traversal while preserving acceptable motion similarity.

**Minimal experiment:** Compare fixed weights, hand-designed terrain rules, and learned adjustments on a common baseline. Test unseen stairs, gaps, and noise levels. Report traversal success, falls, contact violations, pose error, and completion time across multiple training seeds.

**Overlap with existing work:** PGMT [12] already uses terrain-dependent relaxation of lower-body tracking. Merely proposing to relax a reference according to terrain is insufficient novelty. A contribution must specify the new mechanism, applicable constraints, and differences from PGMT.

### Direction B: Physical Foothold Planning with Residual RL

**Problem:** Explicit foothold planning is sensitive to model and perception errors, while direct velocity control lacks explicit foothold constraints.

**Hypothesis:** Retaining reduced-order planning targets and allowing RL to learn bounded compensation can improve reliability on constrained footholds.

**Minimal experiment:** Compare velocity-tracking RL, foothold-conditioned RL, and planning with residual RL under identical training budgets and disturbances. Report foothold error, constraint violations, traversal success, action smoothness, and control computation time. Ablate residual magnitude to determine whether the policy bypasses the planner.

### Direction C: Improve Motion Generators with Execution Feedback

**Problem:** A generator may produce statistically natural motions that the current robot cannot execute.

**Hypothesis:** Fine-tuning a generator using execution feedback from a fixed tracker can improve closed-loop success.

**Minimal experiment:** Compare the original generator, tracker-only adaptation, and generator-only fine-tuning. Distinguish generation quality, dynamic executability, and task success, and evaluate unseen scene geometry to determine whether improvements extend beyond training scenes.

### Direction D: Perception-Driven Generalization to Unseen Combinations of Task Constraints

**Problem:** Can a reusable motor backbone and one shared perceptive adaptation module satisfy simultaneous task constraints that never occurred together during training, without retraining for each combination? For example, train on empty-handed stair climbing and level-ground transport while keeping an object level, then test stair climbing while transporting the object and keeping it level. This is a hypothesis to investigate; its novelty relative to the broader literature has not been established.

**Hypothesis:** Explicit representations of composable body/object goals and scene constraints, coupled through a shared whole-body policy, can improve success on unseen combinations. PGMT [12], ULTRA [15], ViBe [16], and iGPC [17] are necessary comparison points. A single policy, perception, or reference-free execution alone is insufficient as a contribution.

**Minimal experiment:** Fix the robot and perception inputs, and define a training/test matrix of environmental conditions and task constraints. Exclude test combinations from demonstrations, backbone pretraining, RL rollouts, teacher training, and distillation, while exposing each individual factor during training. Compare a policy with directly concatenated inputs, a shared adaptation policy using the same inputs and data, and a model trained on all combinations as an empirical upper-bound baseline. Report success requiring all constraints to be satisfied simultaneously, falls and collisions, object orientation error, and the gap between seen and unseen combinations. Compare ground-truth geometry with visual inputs to separate coordination failures from perception failures.

## 7. Reading and Implementation Recommendations

For a general progression, read BeyondMimic → ASAP → PLANC → PGMT → Generate, Track, Improve. This moves from motion tracking to real-world dynamics mismatch, foothold planning, terrain adaptation, and planner optimization.

For motor backbones, perception, and task generalization, read SONIC → ViBe → PGMT → ULTRA → iGPC. Record task inputs, reference dependence, shared parameters, modules trained separately for each task, and whether the tested variations were genuinely excluded from training.

With limited compute, begin with a baseline whose code is available. Fix the robot, training budget, and task scope; reproduce its basic behavior and record failure cases. Then study one specific mechanism through ablations. Reproducing a large behavior foundation model from scratch requires a separate assessment of data and compute requirements and should not be chosen solely from demonstrations.

## 8. References and Official Links

2. **SoloParkour: Constrained Reinforcement Learning for Visual Locomotion from Privileged Experience.** CoRL 2024. [Paper](https://arxiv.org/abs/2409.13678) · [Project](https://gepetto.github.io/SoloParkour/). The PMLR volume was published in 2025; the conference was CoRL 2024.
3. **PIE: Parkour with Implicit-Explicit Learning Framework for Legged Robots.** 2024. [Paper](https://arxiv.org/abs/2408.13740)
4. **ASAP: Aligning Simulation and Real-World Physics for Learning Agile Humanoid Whole-Body Skills.** 2025. [Paper](https://arxiv.org/abs/2502.01143) · [Project](https://agile.human2humanoid.com/)
5. **GMT: General Motion Tracking for Humanoid Whole-Body Control.** 2025. [Paper](https://arxiv.org/abs/2506.14770) · [Project](https://gmt-humanoid.github.io/)
6. **BeyondMimic: From Motion Tracking to Versatile Humanoid Control via Guided Diffusion.** First released in 2025. [Paper](https://arxiv.org/abs/2508.08241) · [Project](https://beyondmimic.github.io/)
7. **Behavior Foundation Model for Humanoid Robots.** 2025. [Paper](https://arxiv.org/abs/2509.13780) · [Project](https://bfm4humanoid.github.io/)
8. **SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control.** First released in 2025; Science Robotics, 2026. [Preprint](https://arxiv.org/abs/2511.07820) · [Project](https://nvlabs.github.io/GEAR-SONIC/)
9. **Robust and Generalized Humanoid Motion Tracking.** 2026. [Paper](https://arxiv.org/abs/2601.23080) · [Project](https://zeonsunlightyu.github.io/RGMT.github.io/)
10. **Walk the PLANC: Physics-Guided RL for Agile Humanoid Locomotion on Constrained Footholds.** 2026. [Paper](https://arxiv.org/abs/2601.06286) · [Project](https://caltech-amber.github.io/planc/)
11. **Scaling Behavior Foundation Model for Humanoid Robots.** 2026. [Paper](https://arxiv.org/abs/2607.15163) · [Project](https://scalebfm.github.io/)
12. **PGMT: Perceptive General Motion Tracking for Humanoid Robots.** 2026. [Paper](https://arxiv.org/abs/2609.08511) · [Project](https://luyili.github.io/pgmt/)
13. **PASSAGE: Scaling Scene-Aligned Motion Learning for Perceptive Humanoid Traversal in Cluttered Environments.** 2026. [Paper](https://arxiv.org/abs/2609.18732)
14. **Generate, Track, Improve: Perceptive Multi-Skill Humanoid Locomotion with RL-Fine-Tuned Motion Generators.** 2026. [Paper](https://arxiv.org/abs/2609.31577) · [Project](https://zolkin1.github.io/generate-track-improve/)
15. **ULTRA: Unified Multimodal Control for Autonomous Humanoid Whole-Body Loco-Manipulation.** First released in March 2026; IROS 2026. This report uses v2, updated in September 2026. [Paper](https://arxiv.org/abs/2603.03279) · [Full text, v2](https://arxiv.org/html/2603.03279v2) · [Official repository](https://github.com/Sirui-Xu/ULTRA)
16. **ViBe: Visual Behavior Adaptation for Perceptive Humanoid Whole-Body Control.** September 2026. [Paper](https://arxiv.org/abs/2609.09918) · [Project](https://lok-i.github.io/vibe-control/) · [Official repository](https://github.com/lok-i/vibe)
17. **iGPC: Generative Motion Priors for Object-Aware Humanoid Interaction.** October 2026. [Paper](https://arxiv.org/abs/2610.08120) · [Full text](https://arxiv.org/html/2610.08120v1)
