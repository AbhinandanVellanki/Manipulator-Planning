# HW2 — Sampling-Based Motion Planning (RRT & PRM)

RRT and PRM implementations with collision checking for moving a Franka Panda manipulator around a workspace with obstacles. Assignment for 16-662 (Robot Autonomy), CMU.

## Contents

- `Code/Franka.py` — Panda kinematics/Jacobian helper (same role as in HW1).
- `Code/RobotUtil.py` — collision-checking and geometry utilities (block obstacles, point-in-hull tests, transforms).
- `Code/PRMGenerator.py` — builds a probabilistic roadmap over the robot's configuration space against the obstacle set and pickles it to `myPRM.p`.
- `Code/PRMQuery.py` — loads `myPRM.p` and queries it for a collision-free path between a start and goal configuration.
- `Code/RRTQuery.py` — builds and queries an RRT (with path shortening) directly between a start and goal, then replays the path in the MuJoCo viewer.
- `franka_emika_panda/` — MuJoCo MJCF model of the Panda arm used for execution/visualization.
- `Media/` — recorded visualizations of the planners running.
- `HW2.pdf` — assignment spec.
- `abhinanv_HW2.pdf` / `.docx` — submitted write-up.

## MuJoCo Visualizations

[RRT without Path Shortening](Media/RRT.mov)

[RRT with Path Shortening](Media/RRT_shortened.mov)

[PRM](Media/PRM.mov)

## Running

`PRMGenerator.py`/`PRMQuery.py`/`RRTQuery.py` hardcode an absolute `xml_filepath` to the MJCF model — update it to point at the local `franka_emika_panda/panda_with_hand_torque.xml` before running.

```bash
pip install mujoco numpy
python Code/PRMGenerator.py   # builds myPRM.p (only needed once)
python Code/PRMQuery.py       # query the roadmap
python Code/RRTQuery.py       # build + query an RRT
```
