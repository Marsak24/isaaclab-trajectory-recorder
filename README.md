# Isaac Lab Trajectory Recorder

A lightweight trajectory-recording utility for **Isaac Lab** and **RSL-RL** policy playback.

This repository contains a generic recording extension maintained by **Marwah Al-Sakkaf**. It extends the standard Isaac Lab policy-playback workflow with structured export of robot state for offline analysis.

## What it records

For one selected vectorized environment, the recorder can export:

- joint position, velocity, and acceleration
- base position and orientation
- base linear/angular velocity and acceleration
- applied and computed actuator torque
- policy action and current velocity command
- foot pose, velocity, and acceleration
- contact-sensor vectors and a thresholded contact mask

Recordings are written as:

- **NPZ** — lossless structured NumPy arrays
- **JSON** — metadata and frame conventions
- **CSV** — optional flattened human-readable export

The script performs **recording only**. It contains no task-specific analysis, scoring, or evaluation method.

## Requirements

Use this inside an Isaac Lab installation with the RSL-RL workflow available. The script follows the layout of Isaac Lab's standard `scripts/reinforcement_learning/rsl_rl/` examples.

Tested with a Unitree Go1 velocity-locomotion task. The default foot body names are:

```text
FL_foot FR_foot RL_foot RR_foot
```

If your robot or task uses different scene keys or body names, adapt the corresponding constants/CLI arguments.

## Usage

Place `play_record_state.py` beside the standard Isaac Lab RSL-RL playback scripts, then run for example:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_record_state.py \
  --task Isaac-Velocity-Flat-Unitree-Go1-v0 \
  --checkpoint /path/to/model.pt \
  --num_envs 1 \
  --headless \
  --record_start_s 2.0 \
  --record_duration_s 10.0 \
  --record_prefix go1_rollout
```

Outputs are written by default to:

```text
outputs/robot_state/
```

To skip the wide CSV and keep only NPZ + JSON:

```bash
--no_record_csv
```

If your scene uses different asset/sensor keys:

```bash
--robot_key robot
--contact_sensor_key contact_forces
```

## Data conventions

The metadata JSON records the environment index, timestep, joint/body names, quaternion ordering, contact threshold, and frame conventions used by the recorder.

Always use the exported joint/body-name metadata rather than assuming an ordering in downstream code.

## Attribution

Trajectory-recording extension maintained by **Marwa Al-Sakaf**.

This utility is built on the [Isaac Lab](https://github.com/isaac-sim/IsaacLab) RSL-RL playback workflow. Original Isaac Lab copyright and BSD-3-Clause notices are preserved.

If this repository is useful in academic work, please cite it using the included `CITATION.cff`.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
