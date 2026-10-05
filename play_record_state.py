# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Play an RSL-RL checkpoint and record raw robot state from one Isaac Lab environment.

The recorder writes NPZ + JSON, and optionally a wide CSV.  It intentionally
contains only generic state logging; no task-specific analysis or scoring is
performed here.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher

# local imports from the standard Isaac Lab RSL-RL example folder
import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Play an RSL-RL policy and record raw robot state.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--use_pretrained_checkpoint", action="store_true")
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument("--record_output_dir", type=str, default="outputs/robot_state")
parser.add_argument("--record_prefix", type=str, default=None)
parser.add_argument("--record_start_s", type=float, default=2.0)
parser.add_argument("--record_duration_s", type=float, default=10.0)
parser.add_argument("--record_env_id", type=int, default=0)
parser.add_argument("--contact_threshold_n", type=float, default=5.0)
parser.add_argument("--robot_key", type=str, default="robot")
parser.add_argument("--contact_sensor_key", type=str, default="contact_forces")
parser.add_argument("--no_record_csv", action="store_true", default=False)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.record_start_s < 0.0:
    parser.error("--record_start_s must be non-negative")
if args_cli.record_duration_s <= 0.0:
    parser.error("--record_duration_s must be positive")
if args_cli.record_env_id < 0:
    parser.error("--record_env_id must be non-negative")

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


FOOT_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
FOOT_LABELS = ["FL", "FR", "RL", "RR"]


def _cpu_np(x, env_id=None):
    """Detach a torch tensor, optionally select one vectorized environment, and return NumPy."""
    if env_id is not None:
        x = x[env_id]
    return x.detach().cpu().numpy().copy()


class IsaacLabStateRecorder:
    """Collect raw articulated-body state from one vectorized Isaac Lab environment."""

    def __init__(
        self,
        env,
        output_dir,
        prefix,
        env_id=0,
        robot_key="robot",
        contact_sensor_key="contact_forces",
        contact_threshold_n=5.0,
    ):
        self.env = env.unwrapped
        self.env_id = int(env_id)
        self.output_dir = Path(output_dir)
        self.prefix = prefix
        self.contact_threshold_n = float(contact_threshold_n)

        self.robot = self.env.scene[robot_key]
        self.contact_sensor = self.env.scene.sensors[contact_sensor_key]

        self.joint_names = list(self.robot.data.joint_names)
        self.body_names = list(self.robot.data.body_names)
        self.foot_body_names = FOOT_NAMES.copy()
        self.foot_labels = FOOT_LABELS.copy()

        self.foot_ids_robot = [self.body_names.index(name) for name in self.foot_body_names]
        sensor_body_names = list(self.contact_sensor.body_names)
        self.foot_ids_sensor = [sensor_body_names.index(name) for name in self.foot_body_names]

        self.samples = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _command(self):
        """Return current base-velocity command when the task exposes one; otherwise NaNs."""
        try:
            cmd = self.env.command_manager.get_command("base_velocity")
            return _cpu_np(cmd, self.env_id).astype(np.float32)
        except Exception:
            return np.full(3, np.nan, dtype=np.float32)

    def record(self, action, sim_time):
        d = self.robot.data
        e = self.env_id

        contact_vec = _cpu_np(
            self.contact_sensor.data.net_forces_w[:, self.foot_ids_sensor, :], e
        )
        contact_mag = np.linalg.norm(contact_vec, axis=-1)
        contact_mask = (contact_mag >= self.contact_threshold_n).astype(np.uint8)

        sample = {
            "step": len(self.samples),
            "time_s": float(sim_time),
            "root_pos_w": _cpu_np(d.root_pos_w, e),
            "root_quat_wxyz": _cpu_np(d.root_quat_w, e),
            "root_lin_vel_w": _cpu_np(d.root_lin_vel_w, e),
            "root_ang_vel_w": _cpu_np(d.root_ang_vel_w, e),
            "root_lin_acc_w": _cpu_np(d.root_lin_acc_w, e),
            "root_ang_acc_w": _cpu_np(d.root_ang_acc_w, e),
            "joint_pos": _cpu_np(d.joint_pos, e),
            "joint_vel": _cpu_np(d.joint_vel, e),
            "joint_acc": _cpu_np(d.joint_acc, e),
            "applied_torque": _cpu_np(d.applied_torque, e),
            "computed_torque": _cpu_np(d.computed_torque, e),
            "action": _cpu_np(action, e),
            "command": self._command(),
            "foot_pos_w": _cpu_np(d.body_pos_w[:, self.foot_ids_robot, :], e),
            "foot_quat_wxyz": _cpu_np(d.body_quat_w[:, self.foot_ids_robot, :], e),
            "foot_lin_vel_w": _cpu_np(d.body_lin_vel_w[:, self.foot_ids_robot, :], e),
            "foot_ang_vel_w": _cpu_np(d.body_ang_vel_w[:, self.foot_ids_robot, :], e),
            "foot_lin_acc_w": _cpu_np(d.body_lin_acc_w[:, self.foot_ids_robot, :], e),
            "foot_ang_acc_w": _cpu_np(d.body_ang_acc_w[:, self.foot_ids_robot, :], e),
            "contact_sensor_net_force_w": contact_vec,
            "contact_mask": contact_mask,
        }
        self.samples.append(sample)

    def _arrays(self):
        if not self.samples:
            raise RuntimeError("No samples were recorded.")
        keys = self.samples[0].keys()
        arrays = {}
        for key in keys:
            values = [s[key] for s in self.samples]
            arrays[key] = np.asarray(values)
        arrays["joint_names"] = np.asarray(self.joint_names)
        arrays["foot_body_names"] = np.asarray(self.foot_body_names)
        arrays["foot_labels"] = np.asarray(self.foot_labels)
        return arrays

    def close(self, write_csv=True):
        arrays = self._arrays()
        npz_path = self.output_dir / f"{self.prefix}.npz"
        json_path = self.output_dir / f"{self.prefix}.json"
        csv_path = self.output_dir / f"{self.prefix}.csv"

        np.savez_compressed(npz_path, **arrays)

        metadata = {
            "format_version": 1,
            "samples": int(len(self.samples)),
            "step_dt_s": float(self.env.step_dt),
            "env_id": self.env_id,
            "joint_names": self.joint_names,
            "body_names": self.body_names,
            "foot_body_names": self.foot_body_names,
            "foot_labels": self.foot_labels,
            "quaternion_order": "wxyz",
            "position_frame": "selected environment local world frame",
            "velocity_acceleration_frame": "world frame",
            "contact_threshold_n": self.contact_threshold_n,
            "contact_sensor_note": (
                "Isaac Lab ContactSensor net_forces_w is logged verbatim. "
                "Its exact physical interpretation depends on the sensor configuration."
            ),
            "torque_note": (
                "computed_torque is the actuator-model torque before final clipping; "
                "applied_torque is the torque applied after clipping."
            ),
        }
        json_path.write_text(json.dumps(metadata, indent=2) + "\n")

        paths = {"npz": str(npz_path), "json": str(json_path)}
        if write_csv:
            self._write_csv(csv_path, arrays)
            paths["csv"] = str(csv_path)
        return paths

    def _write_csv(self, path, arrays):
        """Write a human-readable flattened CSV in addition to the lossless NPZ."""
        dynamic_keys = [
            "step", "time_s",
            "root_pos_w", "root_quat_wxyz",
            "root_lin_vel_w", "root_ang_vel_w",
            "root_lin_acc_w", "root_ang_acc_w",
            "joint_pos", "joint_vel", "joint_acc",
            "applied_torque", "computed_torque",
            "action", "command",
            "foot_pos_w", "foot_quat_wxyz",
            "foot_lin_vel_w", "foot_ang_vel_w",
            "foot_lin_acc_w", "foot_ang_acc_w",
            "contact_sensor_net_force_w", "contact_mask",
        ]

        header = []
        slices = []
        for key in dynamic_keys:
            a = np.asarray(arrays[key])
            if a.ndim == 1:
                names = [key]
                flat = a.reshape(len(a), 1)
            else:
                flat = a.reshape(a.shape[0], -1)
                names = [f"{key}_{i}" for i in range(flat.shape[1])]
            header.extend(names)
            slices.append(flat)

        matrix = np.concatenate(slices, axis=1)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(matrix.tolist())


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
):
    """Play an RSL-RL agent and record raw state from one selected environment."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    if args_cli.record_env_id >= env_cfg.scene.num_envs:
        raise ValueError(
            f"--record_env_id={args_cli.record_env_id} is invalid for {env_cfg.scene.num_envs} environment(s)."
        )

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] A published pre-trained checkpoint is unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    dt = env.unwrapped.step_dt
    output_dir = os.path.abspath(args_cli.record_output_dir)
    prefix = args_cli.record_prefix or time.strftime("robot_state_%Y%m%d_%H%M%S")
    recorder = IsaacLabStateRecorder(
        env,
        output_dir=output_dir,
        prefix=prefix,
        env_id=args_cli.record_env_id,
        robot_key=args_cli.robot_key,
        contact_sensor_key=args_cli.contact_sensor_key,
        contact_threshold_n=args_cli.contact_threshold_n,
    )

    start_step = max(1, int(round(args_cli.record_start_s / dt)))
    n_steps = max(1, int(round(args_cli.record_duration_s / dt)))
    end_step = start_step + n_steps
    final_step = max(end_step, args_cli.video_length if args_cli.video else 0)

    print(
        f"[INFO] Recording from {args_cli.record_start_s:.3f} s for "
        f"{args_cli.record_duration_s:.3f} s ({n_steps} samples at dt={dt:.6f} s)."
    )

    obs = env.get_observations()
    rollout_step = 0
    recorded_steps = 0

    try:
        while simulation_app.is_running():
            t0 = time.time()
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                policy_nn.reset(dones)

            rollout_step += 1
            if start_step <= rollout_step < end_step:
                recorder.record(actions, sim_time=recorded_steps * dt)
                recorded_steps += 1

            if rollout_step >= final_step:
                break

            sleep_time = dt - (time.time() - t0)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        if recorded_steps:
            paths = recorder.close(write_csv=not args_cli.no_record_csv)
            print("[INFO] Saved recording:")
            for name, path in paths.items():
                print(f"  {name}: {path}")
        else:
            print("[WARNING] No samples were recorded.")
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
