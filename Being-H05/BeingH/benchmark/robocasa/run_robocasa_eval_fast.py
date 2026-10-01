# Copyright (c) 2026 BeingBeyond Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import concurrent.futures
import logging
import os
os.environ["MUJOCO_GL"] = "egl"
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any

import draccus
import numpy as np
from scipy.spatial.transform import Rotation
import tqdm
import imageio

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.controllers import load_composite_controller_config
from robocasa.utils.dataset_registry import SINGLE_STAGE_TASK_DATASETS, MULTI_STAGE_TASK_DATASETS

from BeingH.benchmark.utils.policy import Policy_Robocasa as Policy

# ==============================================================================
# Your correct create_env function (100% original)
# ==============================================================================
def create_env(
    env_name,
    # robosuite-related configs
    robots="PandaOmron",
    camera_names=[
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
    ],
    camera_widths=512,
    camera_heights=512,
    seed=None,
    render_onscreen=False,
    # robocasa-related configs
    obj_instance_split=None,
    generative_textures=None,
    randomize_cameras=False,
    layout_and_style_ids=None,
    layout_ids=None,
    style_ids=None,
):
    controller_config = load_composite_controller_config(
        controller=None,
        robot=robots if isinstance(robots, str) else robots[0],
    )

    env_kwargs = dict(
        env_name=env_name,
        robots=robots,
        controller_configs=controller_config,
        camera_names=camera_names,
        camera_widths=camera_widths,
        camera_heights=camera_heights,
        has_renderer=render_onscreen,
        has_offscreen_renderer=(not render_onscreen),
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=(not render_onscreen),
        camera_depths=False,
        seed=seed,
        obj_instance_split=obj_instance_split,
        generative_textures=generative_textures,
        randomize_cameras=randomize_cameras,
        layout_and_style_ids=layout_and_style_ids,
        layout_ids=layout_ids,
        style_ids=style_ids,
        translucent_robot=False,
    )

    env = robosuite.make(**env_kwargs)
    return env

# ==============================================================================
# Helper Functions and Parallel Evaluation Logic
# ==============================================================================
def robocasa_obses_to_policy_obs_dict(obs: Dict[str, Any], task_description: str) -> Dict[str, Any]:
    """Convert environment observation to the format required by the policy model."""

    eef_pos = obs["robot0_base_to_eef_pos"].astype(np.float32)
    eef_quat = obs["robot0_base_to_eef_quat"] # [x, y, z, w]
    rot = Rotation.from_quat(eef_quat)
    eef_axis_angle = rot.as_rotvec().astype(np.float32)

    # print("task_description", task_description)
    return {
        'state.eef_position': eef_pos.reshape(1,-1),
        'state.eef_rotation': eef_axis_angle.reshape(1,-1),
        'state.gripper_qpos': obs["robot0_gripper_qpos"].astype(np.float32).reshape(1,-1),
        'state.base_position': obs["robot0_base_pos"].astype(np.float32).reshape(1,-1),
        'state.base_rotation': obs["robot0_base_quat"].astype(np.float32).reshape(1,-1),

        'video.left_view': np.expand_dims(np.ascontiguousarray(obs["robot0_agentview_left_image"][::-1, :]), axis=0),
        'video.right_view': np.expand_dims(np.ascontiguousarray(obs["robot0_agentview_right_image"][::-1, :]), axis=0),
        'video.wrist_view': np.expand_dims(np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, :]), axis=0),

        'language.instruction': [task_description],
    }

def save_rollout_video(images: List[np.ndarray], save_path: str):
    """Save rollout video."""
    with imageio.get_writer(save_path, fps=20) as writer:
        for img in images:
            writer.append_data(img)

def set_seed_everywhere(seed: int):
    """Set random seed."""
    np.random.seed(seed)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

@dataclass
class EvalRobocasaConfig:
    """Evaluation configuration."""
    port: int = 5555
    num_open_loop_steps: int = 8
    max_workers: Optional[int] = 1
    num_trials_per_task: int = 10
    max_steps_per_episode: int = 720
    task_type: str = "single"
    task_names_to_eval: Optional[str] = None
    local_log_dir: str = "./experiments/robocasa_savevideos_logs"
    seed: int = 42
    action_type: str = ""
    data_config_name: str = ""

def setup_logging(cfg: EvalRobocasaConfig):
    """Setup log file."""
    run_id = f"EVAL-ROBOCASA-{cfg.task_type}-{cfg.seed}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    log_file_path = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(log_file_path, "w")
    logger.info(f"Logging to: {log_file_path}")
    return log_file, run_id

def log_message(message: str, log_file=None):
    """Print and log message."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()

def run_robocasa_episode(cfg, env, task_description, policy):
    """Run a single episode."""
    obs = env.reset()
    policy.reset()
    replay_images = []
    success = False

    policy.policy.action_dim = env.action_spec[0].shape[0]

    for _ in range(cfg.max_steps_per_episode):
        obs_dict = robocasa_obses_to_policy_obs_dict(obs, task_description)
        replay_images.append(obs_dict["video.left_view"][0])

        # print("obs_dict")
        action = policy.get_action(obs_dict)
        obs, _, _, _ = env.step(action)
        
        if env._check_success():
            success = True
            break

    return success, replay_images

def run_robocasa_trial_worker(args):
    """Function executed by parallel worker process."""
    cfg, env_name, trial_idx, seed = args
    set_seed_everywhere(seed)

    policy = Policy(port=cfg.port, exec_chunk_size=cfg.num_open_loop_steps)
    env = create_env(env_name=env_name, seed=seed)
    
    task_description = env.get_ep_meta()["lang"]
    # print("start", trial_idx)
    success, replay_images = run_robocasa_episode(cfg, env, task_description, policy)
    # print(f"{env_name} success", success)

    if trial_idx < 3:
        video_dir = Path(cfg.local_log_dir) / "videos"
        video_dir.mkdir(exist_ok=True)
        if success:
            video_path = video_dir / f"task_{env_name}_{trial_idx}_success.mp4"
        else:
            video_path = video_dir / f"task_{env_name}_{trial_idx}_fail.mp4"
        save_rollout_video(replay_images, video_path)
    
    env.close()
    return {"env_name": env_name, "success": success}

@draccus.wrap()
def eval_robocasa(cfg: EvalRobocasaConfig):
    """Main evaluation function."""
    set_seed_everywhere(cfg.seed)
    log_file, _ = setup_logging(cfg)

    if cfg.task_names_to_eval:
        tasks_to_run = cfg.task_names_to_eval.split()
    else:
        tasks_to_run = list(SINGLE_STAGE_TASK_DATASETS if cfg.task_type == "single" else MULTI_STAGE_TASK_DATASETS)
        if cfg.task_type == "single":
            tasks_to_run.remove('NavigateKitchen')
    print("tasks_to_run", tasks_to_run)
    jobs = []
    job_counter = 0
    for env_name in tasks_to_run:
        for trial_idx in range(cfg.num_trials_per_task):
            job_seed = cfg.seed + job_counter
            jobs.append((cfg, env_name, trial_idx, job_seed))
            job_counter += 1

    log_message(f"Created {len(jobs)} trials to run.", log_file)
    
    results = defaultdict(lambda: {"successes": 0, "episodes": 0})
    total_episodes = 0
    
    # with concurrent.futures.ProcessPoolExecutor(max_workers=cfg.max_workers) as executor:
    #     future_to_job = {executor.submit(run_robocasa_trial_worker, job): job for job in jobs}
        
    #     for future in tqdm.tqdm(concurrent.futures.as_completed(future_to_job), total=len(jobs)):
    #         result = future.result()
    #         if result:
    #             env_name = result["env_name"]
    #             results[env_name]["episodes"] += 1
    #             if result["success"]:
    #                 results[env_name]["successes"] += 1
                
    #             total_episodes += 1
    #             if total_episodes > 0 and total_episodes % 10 == 0:
    #                 successes = sum(r["successes"] for r in results.values())
    #                 log_message(f"  Intermediate rate @ {total_episodes} eps: {successes/total_episodes:.2%}", log_file)
    
    # === Debug Mode: Serial Execution ===
    for job in tqdm.tqdm(jobs):
        result = run_robocasa_trial_worker(job)
        
        env_name = result["env_name"]
        results[env_name]["episodes"] += 1
        if result["success"]:
            results[env_name]["successes"] += 1
        total_episodes += 1
        # print(f"Finished episode {total_episodes}")

        if total_episodes > 0 and total_episodes % 10 == 0:
            successes = sum(r["successes"] for r in results.values())
            log_message(f"  Intermediate rate @ {total_episodes} eps: {successes/total_episodes:.2%}", log_file)

    # === ADDED: Per-Task Success Rate Report ===
    log_message("\n" + "="*70, log_file)
    log_message(f"{'Task Name':<45} | {'Success Rate':<12} | {'Count'}", log_file)
    log_message("-" * 70, log_file)

    # Sort keys to make the log readable and deterministic
    sorted_tasks = sorted(results.keys())

    for env_name in sorted_tasks:
        stats = results[env_name]
        n_success = stats["successes"]
        n_episodes = stats["episodes"]
        
        if n_episodes > 0:
            rate = n_success / n_episodes
            log_message(f"{env_name:<45} | {rate:10.2%}   | {n_success}/{n_episodes}", log_file)
        else:
            log_message(f"{env_name:<45} | {'N/A':>10}   | 0/0", log_file)

    log_message("="*70 + "\n", log_file)
    # ===========================================

    successes = sum(r["successes"] for r in results.values())
    total_eps = sum(r["episodes"] for r in results.values())
    
    final_rate = successes / total_eps if total_eps > 0 else 0.0
    log_message(f"FINAL AGGREGATE RATE: {final_rate:.2%} ({successes}/{total_eps})", log_file)
    
    log_file.close()

    # successes = sum(r["successes"] for r in results.values())
    # total_eps = sum(r["episodes"] for r in results.values())
    
    # log_message(f"\nFINAL RATE: {successes/total_eps:.2%} ({successes}/{total_eps})", log_file)
    # log_file.close()

if __name__ == "__main__":
    # import multiprocessing
    # multiprocessing.set_start_method("spawn", force=True)
    eval_robocasa()