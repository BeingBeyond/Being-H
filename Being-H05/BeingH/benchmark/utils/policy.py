# Copyright (c) 2026 BeingBeyond Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R

import torch
from BeingH.benchmark.utils.service import ExternalRobotInferenceClient
from typing import Dict, Any
from tqdm import trange
import robosuite.utils.transform_utils as T

def hardware_obses_to_policy_obs_dict(arm_qpos, hand_qpos, camera_obs):
    obs_dict = {
        'state.state': np.concatenate([arm_qpos, hand_qpos]).reshape(1,-1), # (1,13)
        'video.camera_1_rgb': np.expand_dims(camera_obs['camera_1.rgb'], axis=0), # (1,256,256,3)
        'video.camera_2_rgb': np.expand_dims(camera_obs['camera_2.rgb'], axis=0), # (1,256,256,3)
        'annotation.human.action.task_description': ['Put the small cube on the big cube.'],
    }
    return obs_dict

def libero_orig_to_policy(obs, task_description):
    state = np.concatenate([
        obs["robot0_eef_pos"],
        T.quat2axisangle(obs["robot0_eef_quat"]),
        obs["robot0_gripper_qpos"],
    ]).astype(np.float32)
    obs_dict = {
        'state.state': state.reshape(1,-1),
        'video.top_view': np.expand_dims(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]), axis=0), # (1,256,256,3)
        'video.wrist_view': np.expand_dims(np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]), axis=0), # (1,256,256,3)
        'language.instruction': [task_description],
    }
    return obs_dict

# import math

# def my_quat2euler(quat):
#     """
#     Convert quaternion [x, y, z, w] to Euler angles [roll, pitch, yaw] (xyz order)
#     """
#     # Extract x, y, z, w
#     x, y, z, w = quat[0], quat[1], quat[2], quat[3]
    
#     # Roll (x-axis rotation)
#     t0 = +2.0 * (w * x + y * z)
#     t1 = +1.0 - 2.0 * (x * x + y * y)
#     roll = math.atan2(t0, t1)
    
#     # Pitch (y-axis rotation)
#     t2 = +2.0 * (w * y - z * x)
#     t2 = +1.0 if t2 > +1.0 else t2
#     t2 = -1.0 if t2 < -1.0 else t2
#     pitch = math.asin(t2)
    
#     # Yaw (z-axis rotation)
#     t3 = +2.0 * (w * z + x * y)
#     t4 = +1.0 - 2.0 * (y * y + z * z)
#     yaw = math.atan2(t3, t4)
    
#     return np.array([roll, pitch, yaw])

# def libero_to_policy(obs, task_description):
#     obs_dict = {
#         'state.state': np.concatenate([obs["robot0_eef_pos"], my_quat2euler(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]]).reshape(1,-1),
#         'state.eef_position': obs["robot0_eef_pos"].reshape(1,-1),
#         'state.eef_rotation': my_quat2euler(obs["robot0_eef_quat"]).reshape(1,-1),
#         'state.libero_gripper_position': obs["robot0_gripper_qpos"].reshape(1,-1),
#         'video.top_view': np.expand_dims(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]), axis=0), # (1,256,256,3)
#         'video.wrist_view': np.expand_dims(np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]), axis=0), # (1,256,256,3)
#         'language.instruction': [task_description],
#     }
#     return obs_dict

def libero_to_policy(obs, task_description):
    eef_position = obs["robot0_eef_pos"].astype(np.float32)
    eef_rotation = T.quat2axisangle(obs["robot0_eef_quat"]).astype(np.float32)
    gripper_position = obs["robot0_gripper_qpos"].astype(np.float32)
    state = np.concatenate([eef_position, eef_rotation, gripper_position]).astype(np.float32)
    obs_dict = {
        'state.state': state.reshape(1,-1),
        'state.eef_position': eef_position.reshape(1,-1),
        'state.eef_rotation': eef_rotation.reshape(1,-1),
        'state.libero_gripper_position': gripper_position.reshape(1,-1),
        'video.top_view': np.expand_dims(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]), axis=0), # (1,256,256,3)
        'video.wrist_view': np.expand_dims(np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]), axis=0), # (1,256,256,3)
        'language.instruction': [task_description],
    }
    return obs_dict


Obses_to_Policy_Obs_Dict = {
    "libero_orig": libero_orig_to_policy,
    "libero": libero_to_policy,
}

import numpy as np
from scipy.spatial.transform import Rotation as R

class Policy_Libero:
    def __init__(self, host="localhost", port=5555, exec_chunk_size=16, action_type="world_delta"):
        self.policy = ExternalRobotInferenceClient(host=host, port=port)
        self.exec_chunk_size = exec_chunk_size
        self.action_type = action_type

        self.reset()

    def reset(self):
        self.t = 0
        self.action_chunk = []

    def get_action(self, obs_dict):
        if self.t == self.exec_chunk_size or self.t >= len(self.action_chunk):
            self.t = 0

        # breakpoint()

        if self.t == 0:
            action_chunk = self.policy.get_action(obs_dict)

            self.action_chunk = np.concatenate([np.array(v) for v in action_chunk.values()], axis=-1)

            # Squeeze batch dimension: (1, execute_horizon, action_dim) -> (execute_horizon, action_dim)
            # This handles the case where server returns batched output for single-sample input
            # Only squeeze when batch_size == 1 to preserve true batched data
            if self.action_chunk.ndim == 3 and self.action_chunk.shape[0] == 1:
                self.action_chunk = self.action_chunk.squeeze(0)

            # self.action_chunk = action_chunk['action.action']
            self.ref_state = obs_dict['state.state'].flatten()

            if "abstract" in self.action_type:
                # 1. Define time axis
                # Model output: 16 steps, covering 2 seconds (normalized time 0 -> 1)
                original_len = len(self.action_chunk)  # 16
                t_old = np.linspace(0, 1, original_len)

                # Target execution: 2 seconds @ 20Hz = 40 steps
                target_len = 40
                t_new = np.linspace(0, 1, target_len)

                # 2. Separate action components
                # Assume action structure: [Pos (3), Rot (3, axis-angle), Gripper (1+)]
                curr_pos = self.action_chunk[:, :3]
                curr_rot_vec = self.action_chunk[:, 3:6]
                curr_gripper = self.action_chunk[:, 6:]

                # 3. Position and Gripper -> Linear interpolation
                # axis=0 is the time dimension
                f_pos = interp1d(t_old, curr_pos, axis=0, kind='linear')
                f_gripper = interp1d(t_old, curr_gripper, axis=0, kind='linear') # Gripper can also use nearest

                new_pos = f_pos(t_new)
                new_gripper = f_gripper(t_new)

                # 4. Rotation -> SLERP (Spherical Linear Interpolation)
                # Must convert to quaternion first for correct pose interpolation
                rotations = R.from_rotvec(curr_rot_vec)
                slerp = Slerp(t_old, rotations)

                new_rotations = slerp(t_new)
                new_rot_vec = new_rotations.as_rotvec()

                # 5. Merge and slice
                # Combine into 40-frame dense trajectory
                dense_action_chunk = np.concatenate([new_pos, new_rot_vec, new_gripper], axis=-1)

                # User requirement: Take first 16 frames after interpolation (covering first 0.8 seconds)
                self.action_chunk = dense_action_chunk[:original_len]

        current_state = obs_dict['state.state'].flatten()
        # breakpoint()
        action = self.action_chunk[self.t]

        if "eef_delta" in self.action_type:
            # 1. Parse current state (assume state[3:6] is axis-angle)
            current_rot_vec = current_state[3:6]
            current_rotation = R.from_rotvec(current_rot_vec)

            # 2. Parse Action (assume action[3:6] is local axis-angle delta)
            delta_pos_local = action[:3]
            delta_rot_vec_local = action[3:6]

            # 3. Calculate Position Delta (transform from EEF to World)
            # World_delta = R_current * Local_delta
            world_delta_pos = current_rotation.apply(delta_pos_local)

            # 4. Calculate Rotation Delta (output as axis-angle delta in World frame)
            # Logic: R_new = R_curr * R_delta_local
            # We need to find R_delta_world such that R_new = R_delta_world * R_curr
            # Derivation: R_delta_world = R_curr * R_delta_local * R_curr.inv()
            delta_rotation_local = R.from_rotvec(delta_rot_vec_local)

            # Calculate new world frame pose
            new_world_rotation = current_rotation * delta_rotation_local

            # Calculate world frame Delta from current pose to new pose
            # diff = R_new * R_curr^(-1)
            world_delta_rotation = new_world_rotation * current_rotation.inv()
            world_delta_rotvec = world_delta_rotation.as_rotvec()

            gripper_action = action[6:] if len(action) > 6 else []
            transformed_action = np.concatenate([world_delta_pos, world_delta_rotvec, gripper_action])

        elif "eef_relative" in self.action_type:
            # 1. Parse reference state (Chunk Start)
            ref_pos = self.ref_state[:3]
            ref_rot_vec = self.ref_state[3:6]
            ref_rotation = R.from_rotvec(ref_rot_vec)

            # 2. Parse Action (transformation relative to reference frame)
            relative_pos = action[:3]
            relative_rot_vec = action[3:6]
            relative_rotation = R.from_rotvec(relative_rot_vec)

            # 3. Calculate target world coordinates (Target World Pose)
            target_world_pos = ref_pos + ref_rotation.apply(relative_pos)
            target_world_rotation = ref_rotation * relative_rotation

            # 4. Parse current real-time state
            current_pos = current_state[:3]
            current_rot_vec = current_state[3:6]
            current_rotation = R.from_rotvec(current_rot_vec)

            # 5. Calculate World Delta from current state to target state
            delta_pos_world = target_world_pos - current_pos

            # R_target = R_delta_world * R_curr  =>  R_delta_world = R_target * R_curr^(-1)
            delta_rotation_world = target_world_rotation * current_rotation.inv()
            delta_rotvec_world = delta_rotation_world.as_rotvec()

            gripper_action = action[6:] if len(action) > 6 else []
            transformed_action = np.concatenate([delta_pos_world, delta_rotvec_world, gripper_action])

        elif "world_delta" in self.action_type:
            transformed_action = action

        self.t += 1
        return transformed_action

class Policy_Robocasa:
    def __init__(self, host="localhost", port=5555, exec_chunk_size=16):
        self.policy = ExternalRobotInferenceClient(host=host, port=port)
        self.exec_chunk_size = exec_chunk_size
        self.reset()

    def reset(self):
        self.t = 0
        self.action_chunk = []

    def get_action(self, obs_dict):
        if self.t == self.exec_chunk_size or self.t >= len(self.action_chunk):
            self.t = 0

        if self.t == 0:
            action_dict = self.policy.get_action(obs_dict)
            
            raw_gripper = 1. - np.array(action_dict["action.gripper_position"]).reshape(15, -1)
            self.action_chunk = np.concatenate([
                np.array(action_dict["action.eef_position"]).reshape(15, -1),
                np.array(action_dict["action.eef_rotation"]).reshape(15, -1),
                2. * raw_gripper - 1.,
                np.array(action_dict["action.base_motion"]).reshape(15, -1),
                2. * np.array(action_dict["action.control_mode"]).reshape(15, -1) - 1.,
            ], axis=-1)
        
        self.t += 1
        return self.action_chunk[self.t-1]


class Policy_GR00T:
    def __init__(self, host="localhost", port=5555, exec_chunk_size=16):
        self.policy = ExternalRobotInferenceClient(host=host, port=port)
        self.exec_chunk_size = exec_chunk_size
        self.reset()

    def reset(self):
        self.t = 0
        self.action_chunk = []

    def get_action(self, obs_dict):
        if self.t == self.exec_chunk_size or self.t >= len(self.action_chunk):
            self.t = 0

        if self.t == 0:
            self.action_chunk = self.policy.get_action(obs_dict)
        
        self.t += 1

        tmp = {}
        for k,v in self.action_chunk.items():
            tmp[k] = np.array(v[self.t-1])

        return tmp

if __name__ == "__main__":
    policy = Policy()
    dataset = ZarrDatasetReader("./place_cube_cube_ee")
    pred_action = []
    gt_action = []
    for t in range(0,100):
        obs_dict_dataset = hardware_obses_to_policy_obs_dict(
            # arm_qpos=np.round(dataset[t]["right_arm_qpos"], 3),
            arm_qpos=dataset[t]["right_arm_qpos"],
            hand_qpos=dataset[t]["right_hand_qpos"],
            camera_obs=dataset[t],
        )
        print(obs_dict_dataset)
        breakpoint()
        action = policy.get_action(obs_dict_dataset)
        print('Action prediction error:', action - dataset[t]["action"])
