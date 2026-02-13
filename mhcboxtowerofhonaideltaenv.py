import torch
import math
import numpy as np
import time
import random
import json

from util.quaternion import *
from env.genericenv import GenericEnv
from util.colors import FAIL, ENDC
from env.util.hierachical_load_policy import MHCController, PosDeltaController, PickUpController, PutDownController, PickUpMHCController, PutDownMHCController
from env.util.interactivecommandsmixin import BoxManipulationCmd
from scipy.spatial.transform import Rotation as R 

wrap_to_pi = lambda x: (x + np.pi) % (2 * np.pi) - np.pi

class MHCBoxTowerOfHonaiDeltaEnv(GenericEnv, BoxManipulationCmd):

    def __init__(
        self,
        reward_name: str,
        policy_rate: int,
        simulator_type: str,
        dynamics_randomization: bool,
        state_noise: float,
        integral_action: bool = False,
        **kwargs,
    ):
    
        super().__init__(
            robot_name="digit",
            reward_name=reward_name,
            simulator_type=simulator_type,
            terrain="",
            policy_rate=policy_rate,
            dynamics_randomization=dynamics_randomization,
            state_noise=state_noise,
            state_est=False,
            integral_action=integral_action,
            **kwargs,
        )

        self.use_diffusion = False
        self.use_point_cloud = False

        BoxManipulationCmd.__init__(self)

        self.pos_delta_controller = PosDeltaController(self)
        self.pick_up_controller = PickUpController()
        self.put_down_controller = PutDownController()
        self.mhc_controller = MHCController(self)

        # self.pick_up_mhc_controller = PickUpMHCController(self)
        # self.put_down_mhc_controller = PutDownMHCController(self)

        self.initial_variables_for_delta_env()
        

        # Only check obs if this envs is inited, not when it is parent:
        if self.__class__.__name__ == "PosDeltaEnv" and self.simulator_type not in ["ar_async", "real"]:
            self.check_observation_action_size()

        self.initialize_variables()

        np.random.seed(24)

        self.dynamics_randomization = True

    @property
    def action_size(self):
        return 11

    @property
    def observation_size(self):
        observation_size = 67 # robot proprioceptive obs
        observation_size += 4 # stand bit, del_x, del_y, del_yaw
        observation_size += 7 # box pose
        observation_size += 3 # box size
        return observation_size

    @property
    def extra_input_names(self):
        extra_input_names = ['stand-bit', 'del-x', 'del-y', 'del-yaw']
        extra_input_names += ['box-pose-x', 'box-pose-y', 'box-pose-z',
                         'box-pose-qw', 'box-pose-qx', 'box-pose-qy', 'box-pose-qz',
                         'box-size-x', 'box-size-y', 'box-size-z']
        return extra_input_names

    def _get_state(self):

        if self.box_finish_count != 7:
            box_pose = self.get_box_world_pose()
        else:
            box_pose = self.get_box_world_pose(box_finish_count = 6)

        base_pose = self.sim.get_body_pose(self.sim.base_body_name)
        box_pos_rel = self.sim.get_relative_pose(base_pose, box_pose)

        box_quat = box_pos_rel[3:]
        box_euler = R.from_quat(mj2scipy(box_quat)).as_euler('xyz')
        # Add small noise to orientation (pitch gets more noise as it's harder to estimate)
        box_quat_noisy = scipy2mj(R.from_euler('xyz', box_euler).as_quat())
        box_pos_rel[3:] = box_quat_noisy


        return np.concatenate((
            self.get_robot_state(),
            [self.stand_cmd_bit, self.del_x,self.del_y, self.del_yaw],
            box_pos_rel,
            self.box_size,
        ))

    def reset(self, interactive_evaluation=False):

        if self.round_count >= 0:
            self.log_benchmark_data()

        if self.round_count == self.total_evaluation_number:
            self.save_log_benchmark_data()

        self.reset_simulation()

        self.mhc_controller.reset_hidden_state()

        self.reset_variables()

        self.rand_target_position()

        self.rand_box_size()

        self.set_box_poses()

        self.rand_box_mass()

        self.rand_box_friction()

        self.load_plan()

        self.time_step = 0
        self.last_action = None

        self.log_init_status()

        return self.get_state()

    def update_local_position(self):
        base_pose = self.sim.get_body_pose(self.sim.base_body_name)
        self.local_x, self.local_y = base_pose[0], base_pose[1]
        self.local_yaw = R.from_quat(mj2scipy(base_pose[3:])).as_euler('xyz')[2]
        self.local_yaw = wrap_to_pi(self.local_yaw)


    def update_delta_commands(self):
        # Update the delta commands based on the current local position and the command
        diff_x = self.command_x - self.local_x
        diff_y = self.command_y - self.local_y
        if self.current_skill == "walk_without_box" or self.current_skill == "walk_with_box":
            if np.linalg.norm(np.array([diff_x, diff_y])) > 0.4 or np.abs(self.command_yaw - self.local_yaw) > np.pi / 18:
                rotation_offset = R.from_euler(seq = 'z', angles = [self.target_rotation], degrees = False).as_matrix()
                target_position_offset = np.array([-0.2, 0.0, 0.0])
                target_position_offset = np.dot(rotation_offset, target_position_offset)
                diff_x += target_position_offset[0][0]
                diff_y += target_position_offset[0][1]

        orient_cos = np.cos(self.local_yaw)
        orient_sin = np.sin(self.local_yaw)
        local_delta = np.array([
            diff_x * orient_cos + diff_y * orient_sin,
            -diff_x * orient_sin + diff_y * orient_cos,
        ])

        norm = np.linalg.norm(local_delta)
        self.local_goal_distance = float(norm)

        if self._clip_commands and norm > self._clip_norm:
            local_delta = local_delta / (norm + 1e-8) * self._clip_norm

        self.del_x, self.del_y = local_delta
        self.del_yaw = self.command_yaw - self.local_yaw
        self.del_yaw = wrap_to_pi(self.del_yaw)

        # Clip to avoid useless commands
        vel_cmd = np.array([self.del_x, self.del_y, self.del_yaw])

        if self.current_skill == "walk_without_box" or self.current_skill == "walk_with_box":
            self.stand_cmd_bit = int(np.linalg.norm(vel_cmd) < 0.01)
            if np.linalg.norm(np.array([self.command_x - self.local_x, self.command_y - self.local_y])) < 0.06 and np.abs(self.del_yaw) < 0.3:
                self.stand_cmd_bit = 1

    def step(self, action: np.ndarray):

        self.show_progress()
        self.select_target_pose()

        self.draw_markers()

        self.policy_rate = self.default_policy_rate
        if self.dynamics_randomization:
            self.policy_rate += np.random.randint(0, 6)

        if self.pick_up_bit:
            command = self.get_pick_up_state()
            mhc_state = self.mhc_controller.make_mhc_state(command)
            self.stand_bit = 0
            self.turn_rate = 0.0
            self.with_box = True
            # llc_cmd = self.pick_up_mhc_controller.get_action(mhc_state)
        elif self.put_down_bit:
            command = self.get_put_down_state()
            mhc_state = self.mhc_controller.make_mhc_state(command)
            self.stand_bit = 0
            self.turn_rate = 0.0
            self.with_box = False
            # llc_cmd = self.put_down_mhc_controller.get_action(mhc_state)
        else:
            if not self.freeze_upper_body:
                command = self.sim.reset_qpos[self.sim.arm_motor_position_inds].copy() 
                mhc_state = self.mhc_controller.make_mhc_state(np.concatenate((command, [0.0, 0.85])))
            else:

                self.update_local_position()
                self.update_delta_commands()

                if not self.with_box:
                    delta_pose_cmd = np.array([0.0, self.del_x, self.del_y, self.del_yaw])

                    command = self.pos_delta_controller.get_action(self.get_robot_state(), delta_pose_cmd)

                    mhc_state = self.pos_delta_controller.make_mhc_state_pos_delta(command)
                else:

                    act = action.copy()

                    mhc_state = self.make_mhc_state_pos_delta_target(act)
        llc_cmd = self.mhc_controller.get_action(mhc_state)
        self.mhc_controller.set_offset(mhc_state)

        # Step simulation by n steps. This call will update self.tracker_fn.
        simulator_repeat_steps = int(self.sim.simulator_rate / self.policy_rate)
        self.step_simulation(llc_cmd, simulator_repeat_steps, integral_action=False)

        self.new_llc_cmd = llc_cmd

        # Reward for taking current action before changing quantities for new state
        act = action.copy()
        self.compute_reward(action)
        self.last_llc_cmd = llc_cmd
        self.traj_idx += 1
        self.last_action = act.copy()

        # energy penalty: torque . joint velocities
        torque = self.sim.get_torque()
        joint_vel = self.sim.get_motor_velocity()
        self.energy_sum += np.mean(np.abs(torque * joint_vel))

        return self.get_state(), self.reward, self.compute_done(), {'rewards': self.reward_dict}

    def make_mhc_state_pos_delta_target(self, action):
        mhc_state = torch.zeros(213)
        mhc_state[:67] = torch.tensor(self.get_robot_state(), dtype=torch.float32)

        vx, vy, vyaw = action[8], action[9], action[10]

        vx = max(min(vx, 0.3), -0.3)
        vy = max(min(vy, 0.2), -0.2)
        vyaw = max(min(vyaw, 0.3), -0.3)

        if self.stand_bit == 0:
            vx, vy, vyaw = 0.0, 0.0, 0.0
            self.stand_cmd_bit = 1

        if self.stand_cmd_bit == 1:
            vx, vy, vyaw = 0.0, 0.0, 0.0
            self.stand_bit = 0
        else:
            self.stand_bit = 1

        self.orient_add += vyaw / self.default_policy_rate


        mhc_state[67:74] = torch.tensor([self.stand_bit, vx, vy, vyaw, 0.0, -0.15, 0.85], dtype=torch.float32)
        mhc_state[128:132] = torch.tensor(action[:4], dtype=torch.float32)
        mhc_state[155:159] = torch.tensor(action[4:8], dtype=torch.float32)
        
        self.set_offset(mhc_state)
        
        return mhc_state

    def set_offset(self, mhc_state):
        motor_pos_id = [7, 8, 9, 14, 18, 23, 30, 31, 32, 33, 34, 35, 36, 41, 45, 50, 57, 58, 59, 60]
        offset = mhc_state[(74+24):(74+24+61)][motor_pos_id]
        offset_actr_inds = [6, 7, 8, 9, 16, 17, 18, 19]
        self.robot._offset[offset_actr_inds] = offset[offset_actr_inds]

    def hw_step(self):
        base_translate = self.robot.llapi_obs.base.translation
        orientation = self.robot.llapi_obs.base.orientation
        self.local_x, self.local_y = base_translate[0], base_translate[1]
        q = np.array([orientation.x, orientation.y, orientation.z, orientation.w])
        self.local_yaw = R.from_quat(q).as_euler('xyz')[2]
        self.local_yaw = wrap_to_pi(self.local_yaw)
        print(f"local_x: {self.local_x:.3f}, local_y: {self.local_y:.3f}, local_yaw: {self.local_yaw:.3f}", end="\r")
        self.update_delta_commands()

    def get_action_mirror_indices(self):
        return [4, -5, -6, 7,      # left arm (0-3) -> right arm (4-7) with negations
                -0.1, -1, -2, 3,      # right arm (4-7) -> left arm (0-3) with negations  
                8, -9, -10]        # velocity: vx (8) stays, vy (9) negates, vyaw (10) negates

    def get_observation_mirror_indices(self):
        mirror_inds = list(self.robot.robot_state_mirror_indices)  # 67 elements
        
        # Velocity commands: stand_bit, del_x, del_y, del_yaw (4 elements)
        # stand_bit stays, del_x stays, del_y negates, del_yaw negates
        start = len(mirror_inds)  # 67
        mirror_inds += [start, start + 1, -(start + 2), -(start + 3)]
        
        # Box pose relative to base: [x, y, z, qw, qx, qy, qz] (7 elements)
        # Mirror: x stays, y negates, z stays, quaternion components need mirroring
        start = len(mirror_inds)  # 71
        box_pose = np.arange(start, start + 7, dtype=float)
        box_pose[1] *= -1   # y position negates
        box_pose[4] *= -1   # qx negates
        box_pose[6] *= -1   # qz negates
        mirror_inds += box_pose.tolist()
        
        # Box size: [x, y, z] (3 elements) - symmetric, keep as is
        start = len(mirror_inds)  # 78
        box_size = np.arange(start, start + 3, dtype=float)
        mirror_inds += box_size.tolist()
        
        assert len(mirror_inds) == self.observation_size, \
            f"Mirror obs indices length {len(mirror_inds)} != obs_size {self.observation_size}"
        return mirror_inds

    def _init_interactive_key_bindings(self):

        self.input_key_dict = {}

    def _init_interactive_xbox_bindings(self):
        pass

    def _update_control_commands_dict(self):
        self.control_commands_dict["delta x"] = self.del_x
        self.control_commands_dict["delta y"] = self.del_y
        self.control_commands_dict["delta yaw"] = self.del_yaw
        self.control_commands_dict["stand"] = self.stand_bit
        self.control_commands_dict["user_del_x"] = self.user_del_x
        self.control_commands_dict["user_del_y"] = self.user_del_y
        self.control_commands_dict["user_del_yaw"] = self.user_del_yaw
        self.control_commands_dict["command x"] = self.command_x
        self.control_commands_dict["command y"] = self.command_y
        self.control_commands_dict["command yaw"] = self.command_yaw

    @staticmethod
    def get_env_args():
        return {
            "simulator-type"     : ("box", "Which simulator to use (\"mujoco\" or \"libcassie\" or \"ar\")"),
            "policy-rate"        : (50, "Rate at which policy runs in Hz"),
            "dynamics-randomization" : (True, "Whether to use dynamics randomization or not (default is True)"),
            "state-noise"        : ([0,0,0,0,0,0], "Amount of noise to add to proprioceptive state."),
            "reward-name"        : ("pos_delta", "Which reward to use"),
        }

    def get_pick_up_state(self):
        
        out = np.concatenate((self.get_robot_state(),
                                self.box_start_pose,
                                self.box_size))

        command = self.pick_up_controller.get_action(out)

        return command

    def get_put_down_state(self):
        update_rate = np.random.randint(8, 12)
        if self.time_step - self.last_update >= 50/update_rate:

            base_pose = self.sim.get_body_pose(self.sim.base_body_name)

            if self.box_pick_order[self.box_finish_count] == 2:
                box_pose = self.sim.data.qpos[-7:].copy()
                self.box_world_rotation = R.from_quat(mj2scipy(box_pose[3:])).as_euler('xyz')[2]
                self.box_height = box_pose[2].copy()
                self.box_rotation_all = box_pose[3:].copy()
            elif self.box_pick_order[self.box_finish_count] == 1:
                box_pose = self.sim.data.qpos[-14:-7].copy()
                self.box_world_rotation = R.from_quat(mj2scipy(box_pose[3:])).as_euler('xyz')[2]
                self.box_height = box_pose[2].copy()
                self.box_rotation_all = box_pose[3:].copy()
            elif self.box_pick_order[self.box_finish_count] == 0:
                box_pose = self.sim.data.qpos[-21:-14].copy()
                self.box_world_rotation = R.from_quat(mj2scipy(box_pose[3:])).as_euler('xyz')[2]
                self.box_height = box_pose[2].copy()
                self.box_rotation_all = box_pose[3:].copy()
            
            first_frame_relative_pose = box_pose[:3] - self.first_frame_pose[:3]

            base_euler = R.from_quat(mj2scipy(base_pose[3:])).as_euler('xyz')

            Rz = np.array([
                [math.cos(-base_euler[2]), -math.sin(-base_euler[2]), 0.0],
                [math.sin(-base_euler[2]),  math.cos(-base_euler[2]), 0.0],
                [0.0,            0.0,           1.0]
            ])

            debug_box_relative_pose = np.dot(Rz, first_frame_relative_pose)
            box_euler = R.from_quat(mj2scipy(box_pose[3:])).as_euler('xyz')
            target_rotation = wrap_to_pi(box_euler[2] - base_euler[2])
            box_euler[2] = target_rotation
            target_quat = scipy2mj(R.from_euler(seq = 'xyz', angles = box_euler, degrees = False).as_quat())
            self.box_pose = np.concatenate(([debug_box_relative_pose[0] + 0.047], [debug_box_relative_pose[1]], [self.box_height], target_quat))
            self.box_pose[2] += 0.05
            self.last_update = self.time_step

        out = np.concatenate((  self.box_target,
                                self.box_target_quat,
                                self.box_pose,
                                self.box_size))

        command = self.put_down_controller.get_action(self.get_robot_state(), out)

        return command

    def check_put_down_status(self):
        box_pose = self.get_box_world_pose()
        base_pose = self.sim.get_body_pose(self.sim.base_body_name)
        box_relate_pose = self.sim.get_relative_pose(base_pose, box_pose)
        box_relate_pose[2] += base_pose[2]
        box_height_diff = np.linalg.norm(box_relate_pose[:3] - self.box_target[:3])
        box_pitch_rotation = R.from_quat(mj2scipy(box_relate_pose[3:])).as_euler('xyz')

        if box_height_diff > 0.3 or abs(box_pitch_rotation[1]) > 0.5:
            self.current_skill = "put_down_box"
            self.finish_cycle = True

    def select_target_pose(self):
        self.time_step += 1

        if self.pre_stand_bit != self.stand_bit and (self.time_step - self.pre_change_time > 50 or self.renew):
            self.pre_stand_bit = self.stand_bit
            self.pre_change_time = self.time_step
            self.record_time = True
            self.renew = False

        if self.time_step - self.pre_change_time <= 50:
            self.pre_stand_bit = self.stand_bit
        
        
        if self.current_skill == "pick_up_box" and self.time_step - self.pre_change_time > 350:
            self.unlock_change_mode = True

            self.action_process_index += 1
            if self.action_process_index == 4:
                self.action_process_index = 0
            
            self.pre_change_time = self.time_step
            self.record_time = False

        if self.current_skill == "put_down_box" and self.time_step - self.pre_change_time > self.wait_time:

            self.unlock_change_mode = True

            self.action_process_index += 1
            if self.action_process_index == 4:
                self.action_process_index = 0
            self.pre_change_time = self.time_step
            self.record_time = False
        
        if self.time_step - self.pre_change_time >= 50 and self.record_time:
            self.unlock_change_mode = True

            self.action_process_index += 1
            if self.action_process_index == 4:
                self.action_process_index = 0
            self.pre_change_time = self.time_step
            self.record_time = False

        if self.unlock_change_mode:
            self.current_skill = self.action_process[self.action_process_index]
            if self.current_skill == "walk_without_box":
                self.pick_up_bit = False
                self.put_down_bit = False
                self.freeze_upper_body = True

                if self.box_finish_count != -1:
                    self.check_put_down_status()

                self.box_finish_count += 1
                    
                if self.box_finish_count != 7 and self.finish_cycle != True:
                    self.box_number = self.box_pick_order[self.box_finish_count]

                    box_pose = self.get_box_world_pose()

                    Rz = np.array([
                        [math.cos(self.box_world_rotation), -math.sin(self.box_world_rotation), 0.0],
                        [math.sin(self.box_world_rotation),  math.cos(self.box_world_rotation), 0.0],
                        [0.0,            0.0,           1.0]
                    ])

                    self.box_world_pose = box_pose[:3].copy()
                    local_offset = np.array([-0.4, 0.0, 0.0])
                    global_offset = np.dot(Rz, local_offset)
                    self.target_position = self.box_world_pose + global_offset
                    self.command_x = self.target_position[0]
                    self.command_y = self.target_position[1]
                    self.command_yaw = self.box_world_rotation
                    self.target_rotation = self.box_world_rotation
                    self.last_command = self.sim.reset_qpos[self.sim.arm_motor_position_inds].copy()
                    self.pos_delta_controller.reset_hidden_state()
                else:
                    box_finish_count = -1
                    box_pose = self.get_box_world_pose(box_finish_count)
                    if box_pose[2] > 0.65:
                        self.success_count += 1
                    self.finish_cycle = True

                self.stand_bit = 1

            elif self.current_skill == "pick_up_box":
                self.pick_up_bit = True
                self.put_down_bit = False
                self.freeze_upper_body = False

                self.current_hold_box_number = self.box_pick_order[self.box_finish_count]

                base_pose = self.sim.get_body_pose(self.sim.base_body_name)
                if np.linalg.norm(base_pose[:2] - self.desk_position[0][:2]) < 0.8:
                    if len(self.stack_1_box_count) == 0:
                        self.unusual_status = True
                    else:
                        self.current_hold_box_number = self.stack_1_box_count.pop(0)
                elif np.linalg.norm(base_pose[:2] - self.desk_position[1][:2]) < 0.8:
                    if len(self.stack_2_box_count) == 0:
                        self.unusual_status = True
                    else:
                        self.current_hold_box_number = self.stack_2_box_count.pop(0)
                elif np.linalg.norm(base_pose[:2] - self.desk_position[2][:2]) < 0.8:
                    if len(self.stack_3_box_count) == 0:
                        self.unusual_status = True
                    else:
                        self.current_hold_box_number = self.stack_3_box_count.pop(0)
                
                target_pose_rel = self.box_world_pose[:3] - base_pose[:3]

                base_euler = R.from_quat(mj2scipy(base_pose[3:])).as_euler('xyz')

                Rz = np.array([
                    [math.cos(-base_euler[2]), -math.sin(-base_euler[2]), 0.0],
                    [math.sin(-base_euler[2]),  math.cos(-base_euler[2]), 0.0],
                    [0.0,            0.0,           1.0]
                ])

                box_relative_pose = np.dot(Rz, target_pose_rel)
                target_rotation = wrap_to_pi(self.target_rotation - base_euler[2])
                box_quat = scipy2mj(R.from_euler(seq = 'xyz', angles = [0, 0, target_rotation], degrees = False).as_quat())

                box_relative_pose[2] = self.box_height

                self.box_start_pose = np.concatenate(([box_relative_pose[0] + 0.04735], [box_relative_pose[1]], [box_relative_pose[2]], box_quat))
                self.box_number = self.box_pick_order[self.box_finish_count]
                self.box_size = self.box_size_list[self.box_number]

                self.pick_up_controller.reset_hidden_state()
                # self.pick_up_mhc_controller.reset_hidden_state()

            elif self.current_skill == "walk_with_box":
                self.pick_up_bit = False
                self.put_down_bit = False
                self.freeze_upper_body = True
                self.target_position = self.desk_position[self.target_position_order[self.box_finish_count]]
                self.target_rotation = self.desk_rotation[self.target_position_order[self.box_finish_count]]
                Rz = np.array([
                        [math.cos(self.target_rotation), -math.sin(self.target_rotation), 0.0],
                        [math.sin(self.target_rotation),  math.cos(self.target_rotation), 0.0],
                        [0.0,            0.0,           1.0]
                    ])

                local_offset = np.array([-0.4, 0.0, 0.0])
                global_offset = np.dot(Rz, local_offset)
                self.target_position = self.target_position + global_offset
                
                self.command_x = self.target_position[0]
                self.command_y = self.target_position[1]
                self.command_yaw = wrap_to_pi(self.target_rotation)

                self.stand_bit = 1

            elif self.current_skill == "put_down_box":
                self.pick_up_bit = False
                self.put_down_bit = True
                self.freeze_upper_body = False
                self.only_drop_box_once = False

                base_pose = self.sim.get_body_pose(self.sim.base_body_name)
                self.first_frame_pose = base_pose.copy()
                base_euler = R.from_quat(mj2scipy(base_pose[3:])).as_euler('xyz')
                
                target_position = self.desk_position[self.target_position_order[self.box_finish_count]]
                target_rotation = self.desk_rotation[self.target_position_order[self.box_finish_count]]
                target_quat = scipy2mj(R.from_euler(seq = 'xyz', angles = [0, 0, target_rotation], degrees = False).as_quat())
                target_pose = np.concatenate((target_position, target_quat))

                base_pose = self.sim.get_body_pose(self.sim.base_body_name)
                target_pose_rel = self.sim.get_relative_pose(base_pose, target_pose)

                target_rotation = wrap_to_pi(self.target_rotation - base_euler[2])
                target_quat = scipy2mj(R.from_euler(seq = 'xyz', angles = [0, 0, target_rotation], degrees = False).as_quat())

                self.update_put_down_target_and_stack_count(target_pose_rel[0], target_pose_rel[1], target_quat)


                if self.box_target[2] < 0.3:
                    self.wait_time = 350
                else:
                    self.wait_time = 250
                self.box_number = self.current_hold_box_number
                self.put_down_controller.reset_hidden_state()
                # self.put_down_mhc_controller.reset_hidden_state()

            
            self.log_benchmark_data()
            self.pre_change_time = self.time_step
            self.unlock_change_mode = False

    def update_put_down_target_and_stack_count(self, box_x_local, box_y_local, box_quat):
        base_pose = self.sim.get_body_pose(self.sim.base_body_name)[:2]

        stack_count_lists = [self.stack_1_box_count, self.stack_2_box_count, self.stack_3_box_count]
        stack_index = None
        for i, desk_pos in enumerate(self.desk_position):
            if np.linalg.norm(base_pose - desk_pos[:2]) < 0.8:
                stack_index = i
                break
        
        current_stack = stack_count_lists[stack_index]
        box_height = sum(self.box_height_list[box_number] * 2 for box_number in current_stack)
        box_height += self.box_height_list[self.current_hold_box_number]
        
        self.box_target_quat = box_quat
        self.box_target = np.concatenate(([box_x_local + 0.047], [box_y_local], [box_height]))
        current_stack.insert(0, self.current_hold_box_number)

    def get_box_world_pose(self, box_finish_count = None):

        if box_finish_count is not None:
            temp_box_finish_count = self.box_finish_count
            self.box_finish_count = box_finish_count

        if self.box_pick_order[self.box_finish_count] == 2:
            box_pose = self.sim.data.qpos[-7:].copy()
            self.box_world_rotation = R.from_quat(mj2scipy(box_pose[3:])).as_euler('xyz')[2]
            self.box_height = box_pose[2].copy()
            self.box_rotation_all = box_pose[3:].copy()
        elif self.box_pick_order[self.box_finish_count] == 1:
            box_pose = self.sim.data.qpos[-14:-7].copy()
            self.box_world_rotation = R.from_quat(mj2scipy(box_pose[3:])).as_euler('xyz')[2]
            self.box_height = box_pose[2].copy()
            self.box_rotation_all = box_pose[3:].copy()
        elif self.box_pick_order[self.box_finish_count] == 0:
            box_pose = self.sim.data.qpos[-21:-14].copy()
            self.box_world_rotation = R.from_quat(mj2scipy(box_pose[3:])).as_euler('xyz')[2]
            self.box_height = box_pose[2].copy()
            self.box_rotation_all = box_pose[3:].copy()

        if box_finish_count is not None:
            self.box_finish_count = temp_box_finish_count

        return box_pose

    def rand_target_position(self):

        radius = np.random.uniform(1.5, 2.5)
        self.area_radius = radius

        angle_0 = np.random.uniform(0.0, 2*np.pi)
        target_position_0 = np.array([radius * np.cos(angle_0), radius * np.sin(angle_0), 0.0])
        while True:
            angle_1 = np.random.uniform(0.0, 2*np.pi)
            target_position_1 = np.array([radius * np.cos(angle_1), radius * np.sin(angle_1), 0.0])
            if np.linalg.norm(target_position_0[:2] - target_position_1[:2]) > 0.9:
                break
        while True:
            angle_2 = np.random.uniform(0.0, 2*np.pi)
            target_position_2 = np.array([radius * np.cos(angle_2), radius * np.sin(angle_2), 0.0])
            if np.linalg.norm(target_position_0[:2] - target_position_2[:2]) > 0.9 and np.linalg.norm(target_position_1[:2] - target_position_2[:2]) > 0.9:
                break
        self.desk_position = np.array([target_position_0, target_position_1, target_position_2])
        self.desk_rotation = np.array([angle_0, angle_1, angle_2])

    def rand_box_size(self):

        box_size_0 = 0.1 + np.array([np.random.uniform(0.03, 0.045), np.random.uniform(0.03, 0.045), np.random.uniform(0.03, 0.045)])
        self.sim.set_geom_size("box0", box_size_0)
        box_size_1 = 0.1 + np.array([np.random.uniform(0.045, 0.06), np.random.uniform(0.045, 0.06), np.random.uniform(0.045, 0.06)])
        self.sim.set_geom_size("box1", box_size_1)
        box_size_2 = 0.1 + np.array([np.random.uniform(0.06, 0.075), np.random.uniform(0.06, 0.075), np.random.uniform(0.06, 0.075)])
        self.sim.set_geom_size("box2", box_size_2)

        self.box_height_list = [box_size_0[2], box_size_1[2], box_size_2[2]]
        self.box_size_list = [box_size_0, box_size_1, box_size_2]

    def rand_box_mass(self):
        box_mass_0 = 0.2 + np.random.uniform(0, 2.5)
        box_mass_1 = 0.2 + np.random.uniform(0, 2.5)
        box_mass_2 = 0.2 + np.random.uniform(0, 2.5)
        self.sim.model.body("box").mass = box_mass_0
        self.sim.model.body("box1").mass = box_mass_1
        self.sim.model.body("box2").mass = box_mass_2
        box_size = self.box_size_list[0]
        box_size_1 = self.box_size_list[1]
        box_size_2 = self.box_size_list[2]
        self.sim.model.body_inertia[self.sim.box_body_id, 0] = 0.6 * box_mass_0 * (box_size[1]**2 + box_size[2]**2) / 3
        self.sim.model.body_inertia[self.sim.box_body_id, 1] = box_mass_0 * (box_size[0]**2 + box_size[2]**2) / 3
        self.sim.model.body_inertia[self.sim.box_body_id, 2] = box_mass_0 * (box_size[0]**2 + box_size[1]**2) / 3
        self.sim.model.body_inertia[self.sim.box1_body_id, 0] = 0.6 * box_mass_1 * (box_size_1[1]**2 + box_size_1[2]**2) / 3
        self.sim.model.body_inertia[self.sim.box1_body_id, 1] = box_mass_1 * (box_size_1[0]**2 + box_size_1[2]**2) / 3
        self.sim.model.body_inertia[self.sim.box1_body_id, 2] = box_mass_1 * (box_size_1[0]**2 + box_size_1[1]**2) / 3
        self.sim.model.body_inertia[self.sim.box2_body_id, 0] = 0.6 * box_mass_2 * (box_size_2[1]**2 + box_size_2[2]**2) / 3
        self.sim.model.body_inertia[self.sim.box2_body_id, 1] = box_mass_2 * (box_size_2[0]**2 + box_size_2[2]**2) / 3
        self.sim.model.body_inertia[self.sim.box2_body_id, 2] = box_mass_2 * (box_size_2[0]**2 + box_size_2[1]**2) / 3

    def rand_box_friction(self):
        
        sliding_friction = np.random.uniform(0.5, 0.7)
        rolling_friction = 0.02
        spinning_friction = 0.005

        self.sim.model.geom_friction[self.sim.box_geom_id] = np.array([sliding_friction, rolling_friction, spinning_friction])

        sliding_friction = np.random.uniform(0.5, 0.7)
        self.sim.model.geom_friction[self.sim.box1_geom_id] = np.array([sliding_friction, rolling_friction, spinning_friction])

        sliding_friction = np.random.uniform(0.5, 0.7)
        self.sim.model.geom_friction[self.sim.box2_geom_id] = np.array([sliding_friction, rolling_friction, spinning_friction])

    def set_box_poses(self):

        initial_box_x = self.desk_position[0][0]
        initial_box_y = self.desk_position[0][1]

        # set box0 position
        box0_position = np.array([initial_box_x, initial_box_y, np.random.uniform(0.78, 0.9)])
        box0_quat = scipy2mj(R.from_euler(seq = 'xyz', angles = [0, 0, self.desk_rotation[0]], degrees = False).as_quat())
        box0_pose = np.concatenate([box0_position, box0_quat])
        self.sim.data.qpos[-21:-14] = box0_pose
        # set box1 position
        box1_position = np.array([initial_box_x, initial_box_y, np.random.uniform(0.44, 0.5)])
        box1_quat = scipy2mj(R.from_euler(seq = 'xyz', angles = [0, 0, self.desk_rotation[0]], degrees = False).as_quat())
        box1_pose = np.concatenate([box1_position, box1_quat])
        self.sim.data.qpos[-14:-7] = box1_pose
        # set box2 position
        box2_position = np.array([initial_box_x, initial_box_y, np.random.uniform(0.14, 0.15)])
        box2_quat = scipy2mj(R.from_euler(seq = 'xyz', angles = [0, 0, self.desk_rotation[0]], degrees = False).as_quat())
        box2_pose = np.concatenate([box2_position, box2_quat])
        self.sim.data.qpos[-7:] = box2_pose

        # sort box position by distance
        box_positions = np.array([box0_position, box1_position, box2_position])
        box_distances = np.linalg.norm(box_positions, axis=1)
        box_order = np.argsort(box_distances)
        self.box_pick_order = box_order

    def load_plan(self):

        box_order = []
        pos_order = []
        act_order = []

        box_map = {"b1": 0, "b2": 1, "b3": 2}
        loc_map = {"l1": 0, "l2": 1, "l3": 2}


        # load hanoi forward plan
        with open("/home/star-paladin/box-world-json/hanoi.json", "r") as f:
            plan_dict = json.load(f)

            even_counter = 0

            for step in plan_dict["plan"]:
                action, args = next(iter(step.items()))

                act_order.append(action)

                if action == "pickup" or action == "unstack":
                    box = args[0]
                    box_order.append(box_map[box])
                elif action == "locomotion" and even_counter % 2 == 0:
                    loc = args[-1]
                    pos_order.append(loc_map[loc])
                
                if action == "locomotion":
                    even_counter += 1

        self.box_pick_order = box_order
        self.target_position_order = pos_order
        self.act_order = act_order

    def initialize_variables(self):

        self.time = 0

        self.target_marker = None
        self.target_marker_inside = None

        self.area_marker = None
        self.desk_0_marker = None
        self.desk_1_marker = None
        self.desk_2_marker = None
        self.label_0_marker = None
        self.label_1_marker = None
        self.label_2_marker = None
        self.arraw_0_height = 1.75
        self.arraw_1_height = 1.8
        self.arraw_2_height = 1.85
        self.arraw_0_change = 0.002
        self.arraw_1_change = 0.002
        self.arraw_2_change = -0.002
        self.target_marker_0 = None
        self.target_marker_1 = None
        self.target_marker_2 = None

        # benchmark data variables
        self.success_count = 0
        self.finish_count = 0
        self.round_count = -1
        
        self.benchmark_data = []
        self.total_evaluation_number = 100
        self.start_time = time.time()

    def reset_variables(self):

        self.pick_up_bit = False
        self.put_down_bit = False
        self.box_number = 0
        self.box_start_pose = np.array([0.45, 0.0, 0.557, 1.0, 0.0, 0.0, 0.0])
        self.box_size = np.array([0.157, 0.157, 0.157])
        self.box_target = np.array([0.4, 0.0, 0.0])
        self.target_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.box_pose = np.array([0.45, 0.0, 0.687, 1.0, 0.0, 0.0, 0.0])
        self.last_update = 0

        self.x_velocity = 0.0
        self.y_velocity = 0.0
        self.turn_rate = 0.0
        self.orient_add = 0.0
        self.height = 0.85
        self.stand_bit = 0
        self.stand_cmd_bit = 0
        self.stand_mode = True
        self.freeze_upper_body = True
        self.put_down_box = False

        self.time_step = 0
        self.pre_change_time = 0
        self.locomotion_timer = 0
        self.use_correct_time = False
        self.current_time = False
        self.unlock_change_mode = True
        self.only_drop_box_once = False
        self.wait_time = 350
        self.pre_stand_bit = 0
        self.hand_force_reset_count = 0

        self.yaw = -0.1
        self.box_angle_local = 0.0
        self.box_x_local = 0.0
        self.box_y_local = 0.0
        self.target_position = np.array([1.5, 1.0])
        self.target_rotation = 0.0
        self.box_target_quat = np.array([1.0, 0.0, 0.0, 0.0])

        self.desk_position = np.array([[1.5, 0.0, 0.0], [0.0, -1.0, 0.0], [-1.5, 0.0, 0.0]])
        self.desk_rotation = np.array([0.0, -np.pi/2, np.pi])
        self.area_radius = 0.0

        self.with_box = False
        self.record_time = False
        self.disable_once = False
 
        self.box_number = 0
        self.box_world_pose = np.array([0.0, 0.0, 0.0])
        self.box_height = 0.0
        self.box_world_rotation = 0.0
        self.box_rotation_all = None
        self.int_to_change_box_number = 1

        self.box_finish_count = -1
        self.box_pick_order = [0, 1, 2]
        self.target_position_order = [0, 0, 0]
        self.action_process = ["walk_without_box", "pick_up_box", "walk_with_box", "put_down_box"]
        self.action_process_index = 0
        self.current_skill = "walk_without_box"
        self.last_command = self.sim.reset_qpos[self.sim.arm_motor_position_inds].copy()

        self.stack_1_box_count = [0,1,2]
        self.stack_2_box_count = []
        self.stack_3_box_count = []
        self.current_hold_box_number = None

        self.renew = False
        
        self.finish_cycle = False
        
        self.walk_marker_change = False

        # Reset env counter variables
        self.interactive_evaluation = False
        self.traj_idx = 0

        self.last_action = None
        self.last_llc_cmd = None
        self.new_llc_cmd = None
        self.feet_air_time = np.array([0, 0])

        self.round_count += 1
        

        # benchmark data variables
        self.round_status = []
        self.stack_finish_status = []
        self.success_status = []
        self.time_status = []
        self.current_skill_status = []
        self.box_pose_status = []
        self.box_target_position_status = []
        self.box_target_rotation_status = []

        self.robot_position_status = []
        self.robot_rotation_status = []
        self.robot_target_position_status = []
        self.robot_target_rotation_status = []

        self.dr_status = []
        self.energy_status = []
        self.energy_sum = 0.0

        self.first_frame_pose = None

    def log_init_status(self):
        initial_dyn_params = {"damping": self.sim.get_dof_damping().copy(),
                                   "mass": self.sim.get_body_mass().copy(),
                                   "ipos": self.sim.get_body_ipos().copy(),
                                   "spring": self.sim.get_joint_stiffness().copy(),
                                   "friction": self.sim.get_geom_friction().copy(),
                                   "solref": self.sim.get_geom_solref().copy()}

        self.dr_status.append(initial_dyn_params)
        

    def log_benchmark_data(self):

        base_pose = self.sim.get_body_pose(self.sim.base_body_name)

        all_box_pose = self.get_all_box_pose()

        self.time_status.append(self.time_step)
        self.stack_finish_status.append(self.box_finish_count)
        self.current_skill_status.append(self.current_skill)
        self.box_pose_status.append(all_box_pose)
        self.box_target_position_status.append(self.box_target[:3])
        self.box_target_rotation_status.append(R.from_quat(mj2scipy(self.box_target_quat)).as_euler('xyz'))
        self.robot_position_status.append(base_pose[:3])
        self.robot_rotation_status.append(R.from_quat(mj2scipy(base_pose[3:])).as_euler('xyz'))
        self.robot_target_position_status.append(self.target_position)
        self.robot_target_rotation_status.append(self.target_rotation)
        self.energy_status.append(self.energy_sum)
        self.energy_sum = 0.0

        self.log_current_data()

    def log_current_data(self):

        benchmark_data = {
            "round_status": self.round_count,
            "success_status": self.success_count,
            "stack_finish_status": self.stack_finish_status,
            "time_status": self.time_status,
            "current_skill_status": self.current_skill_status,
            "box_pose_status": self.box_pose_status,
            "box_target_position_status": self.box_target_position_status,
            "box_target_rotation_status": self.box_target_rotation_status,
            "robot_position_status": self.robot_position_status,
            "robot_rotation_status": self.robot_rotation_status,
            "robot_target_position_status": self.robot_target_position_status,
            "robot_target_rotation_status": self.robot_target_rotation_status,
            "dr_status": self.dr_status,
            "energy_status": self.energy_status
        }

        self.benchmark_data.append(benchmark_data)

    def save_log_benchmark_data(self):
        #save as npz file
        filename = f"Tower_of_hanoi_benchmark/TOH_base_0.3_weight_dr_benchmark.npz"
        np.savez(filename, benchmark_data=self.benchmark_data)

        print()
        print(f"\nSaved benchmark data to {filename}")

        exit()

    def initial_variables_for_delta_env(self):
        # Re-initialize interactive key bindings to ensure PosDelta controls take precedence
        # over PickUpBoxCmd controls where there are conflicts.
        self._init_interactive_key_bindings()

        # Command randomization ranges
        self._x_delta_bounds = [-2.0, 2.0]
        self._y_delta_bounds = [-2.0, 2.0]
        self._yaw_delta_bounds = [-3.14, 3.14] # rad/s
        self._randomize_commands_bounds = [300, 500] # in episode length
        self._clip_commands = True
        self._clip_norm = 2.0

        self.del_x = 0
        self.del_y = 0
        self.del_yaw = 0
        self.stand_bit = 0

        self.user_del_x = 0
        self.user_del_y = 0
        self.user_del_yaw = 0
        # self.update_orient = True

        self.local_x, self.local_y, self.local_yaw = 0, 0, 0
        self.command_x, self.command_y, self.command_yaw = 0, 0, 0

        # feet air time tracking
        self.feet_air_time = np.array([0, 0]) # 2 feet
        # reward variables
        self.feet_contact_buffer = []
        self.robot.nominal_height = 0.98 if self.robot.robot_name == "digit" else 0.81
        self.robot._min_base_height = 0.6
        self.pos_marker = None

    def get_all_box_pose(self):
        all_box_pose = []
        box_pose = self.sim.data.qpos[-21:-14].copy()
        all_box_pose.append(box_pose)
        box_pose = self.sim.data.qpos[-14:-7].copy()
        all_box_pose.append(box_pose)
        box_pose = self.sim.data.qpos[-7:].copy()
        all_box_pose.append(box_pose)
        return all_box_pose

    def show_progress(self):
        progress = (self.round_count / self.total_evaluation_number) * 100
        bar_length = 20
        filled_length = int(bar_length * self.round_count // self.total_evaluation_number)

        elapsed_time = time.time() - self.start_time
        remaining_samples = self.total_evaluation_number - self.round_count
        if self.round_count > 0:
            estimated_time_remaining = remaining_samples * elapsed_time / self.round_count
        else:
            estimated_time_remaining = 0
        hours = estimated_time_remaining // 3600
        minutes = (estimated_time_remaining % 3600) // 60
        seconds = estimated_time_remaining % 60

        bar = "🔋" * (filled_length - 1) + "🤖" + "-" * (bar_length - filled_length - 1)
        print(f"Progress: [{bar}] {progress:.2f}% Estimated time remaining: {hours:.0f} h {minutes:.0f} m {seconds:.0f} s   ")
        sim_time = self.time_step/50
        print(f"Current round: {self.round_count}/{self.total_evaluation_number} success {self.success_count} times  simulation time: {sim_time//60:.0f}m {sim_time%60:.0f}s ")

        self.show_robot_animation()

    def show_robot_animation(self):

        if not hasattr(self, "_robot_anim"):
            SCENE_WIDTH = 50
            box_x = random.randint(5, SCENE_WIDTH - 5)
            drop_x = random.randint(5, SCENE_WIDTH - 5)
            self._robot_anim = {
                "phase": 0,
                "robot_x": 2,
                "box_x": box_x,
                "drop_x": drop_x,
                "last_update": time.time(),
                "step": 0,
                "phase_start": time.time(),
                "SCENE_WIDTH": SCENE_WIDTH,
            }

        st = self._robot_anim
        now = time.time()

        # Update animation state
        if now - st["last_update"] >= 0.3:
            st["last_update"] = now
            st["step"] += 1

            if st["phase"] == 0:
                if st["robot_x"] < st["box_x"]:
                    st["robot_x"] += 1
                else:
                    st["phase"] = 1

            elif st["phase"] == 1:
                if st["robot_x"] < st["drop_x"]:
                    st["robot_x"] += 1
                else:
                    st["phase"] = 2
                    st["phase_start"] = now

            elif st["phase"] == 2:
                if now - st["phase_start"] > 2.0:
                    SCENE_WIDTH = st["SCENE_WIDTH"]
                    box_x = random.randint(5, SCENE_WIDTH - 5)
                    drop_x = random.randint(5, SCENE_WIDTH - 5)
                    st.update({
                        "phase": 0,
                        "robot_x": 2,
                        "box_x": box_x,
                        "drop_x": drop_x,
                        "step": 0,
                        "phase_start": now,
                    })

        # Build 10-line scene
        W = st["SCENE_WIDTH"]
        lines = [" " * W for _ in range(10)]

        def place(row_chars, x, s):
            """place string s into list row_chars at position x"""
            for i, ch in enumerate(s):
                idx = x + i
                if 0 <= idx < W:
                    row_chars[idx] = ch

        # which status text
        if st["phase"] == 0:
            status = "Robot searching for box..."
        elif st["phase"] == 1:
            status = "Robot carrying box on head..."
        else:
            status = "Robot dropped the box."

        lines[0] = status.ljust(W)[:W]

        robot_x = st["robot_x"]
        box_ground_x = None
        box_on_head = False

        if st["phase"] == 0:
            box_ground_x = st["box_x"]
            pose = "[┐∵]┘" if (st["step"] % 2 == 0) else "└[∵┌]"

        elif st["phase"] == 1:
            box_on_head = True
            pose = "└|∵|┘"

        else:
            box_ground_x = st["drop_x"]
            pose = "└|∵|┘"

        # Row indexes
        BOX_ROW = 5
        HEAD_BOX_ROW = 3
        ROBOT_ROW = 4
        FLOOR_ROW = 7

        # draw robot row
        robot_row = list(" " * W)
        place(robot_row, robot_x, pose)
        lines[ROBOT_ROW] = "".join(robot_row)

        # draw box depending on phase
        if box_on_head:
            # box on head above robot
            head_row = list(" " * W)
            place(head_row, robot_x + 1, "📦")
            lines[HEAD_BOX_ROW] = "".join(head_row)
        elif box_ground_x is not None:
            box_row = list(" " * W)
            place(box_row, box_ground_x, "📦")
            lines[BOX_ROW] = "".join(box_row)

        # floor
        lines[FLOOR_ROW] = ("_" * W)

        for line in lines:
            print(line)

        print("\033[12A", end="")

    def draw_markers(self):

        so3 = euler2so3(z=0, x=0, y=0)
        size = [self.area_radius, self.area_radius, 0.1]
        rgba =  [0.1, 0.9, 0.9, 0.01]
        marker_params = ["sphere", "", [0, 0, 0], size, rgba, so3]
        if self.area_marker is None:
            self.area_marker = self.sim.viewer.add_marker(*marker_params)
            self.sim.viewer.update_marker_position(self.area_marker, [0, 0, 0.005])
        self.sim.viewer.update_marker_size(self.area_marker, [self.area_radius, self.area_radius, 0.1])


        so3 = euler2so3(z=0, x=np.pi, y=0)
        size = [0.03, 0.03, 0.5]
        rgba =  [0.1, 0.8, 0.8, 0.5]
        marker_params = ["arrow", "", [0, 0, 0], size, rgba, so3]
        if self.desk_0_marker is None:
            self.desk_0_marker = self.sim.viewer.add_marker(*marker_params)

        self.arraw_0_height += self.arraw_0_change
        if self.arraw_0_height >= 1.85 or self.arraw_0_height <= 1.75:
            self.arraw_0_change = -self.arraw_0_change
                
        self.sim.viewer.update_marker_position(self.desk_0_marker, [self.desk_position[0][0], self.desk_position[0][1], self.arraw_0_height])

        so3 = euler2so3(z=0, x=np.pi, y=0)
        size = [0.03, 0.03, 0.5]
        rgba =  [0.1, 0.8, 0.8, 0.5]
        marker_params = ["arrow", "", [0, 0, 0], size, rgba, so3]
        if self.desk_1_marker is None:
            self.desk_1_marker = self.sim.viewer.add_marker(*marker_params)    
  
        self.arraw_1_height += self.arraw_1_change
        if self.arraw_1_height >= 1.85 or self.arraw_1_height <= 1.75:
            self.arraw_1_change = -self.arraw_1_change
                
        self.sim.viewer.update_marker_position(self.desk_1_marker, [self.desk_position[1][0], self.desk_position[1][1], self.arraw_1_height])

        so3 = euler2so3(z=0, x=np.pi, y=0)
        size = [0.03, 0.03, 0.5]
        rgba =  [0.1, 0.8, 0.8, 0.5]
        marker_params = ["arrow", "", [0, 0, 0], size, rgba, so3]
        if self.desk_2_marker is None:
            self.desk_2_marker = self.sim.viewer.add_marker(*marker_params)    

        self.arraw_2_height += self.arraw_2_change
        if self.arraw_2_height >= 1.85 or self.arraw_2_height <= 1.75:
            self.arraw_2_change = -self.arraw_2_change
                
        self.sim.viewer.update_marker_position(self.desk_2_marker, [self.desk_position[2][0], self.desk_position[2][1], self.arraw_2_height])

        so3 = euler2so3(z=0, x=0, y=0)
        size = [0.015, 0.015, 0.4]
        rgba =  [0.1, 0.8, 0.8, 0.6]
        marker_params = ["label", "", [0, 0, 0], size, rgba, so3]
        if self.label_0_marker is None:
            self.label_0_marker = self.sim.viewer.add_marker(*marker_params)
            self.sim.viewer.update_marker_name(self.label_0_marker, "Tower 0")
        self.sim.viewer.update_marker_position(self.label_0_marker, [self.desk_position[0][0], self.desk_position[0][1], self.arraw_0_height])

        so3 = euler2so3(z=0, x=0, y=0)
        size = [0.015, 0.015, 0.4]
        rgba =  [0.1, 0.8, 0.8, 0.6]
        marker_params = ["label", "", [0, 0, 0], size, rgba, so3]
        if self.label_1_marker is None:
            self.label_1_marker = self.sim.viewer.add_marker(*marker_params)
            self.sim.viewer.update_marker_name(self.label_1_marker, "Tower 1")
        self.sim.viewer.update_marker_position(self.label_1_marker, [self.desk_position[1][0], self.desk_position[1][1], self.arraw_1_height])

        so3 = euler2so3(z=0, x=0, y=0)
        size = [0.015, 0.015, 0.4]
        rgba =  [0.1, 0.8, 0.8, 0.6]
        marker_params = ["label", "", [0, 0, 0], size, rgba, so3]
        if self.label_2_marker is None:
            self.label_2_marker = self.sim.viewer.add_marker(*marker_params)
            self.sim.viewer.update_marker_name(self.label_2_marker, "Tower 2")
        self.sim.viewer.update_marker_position(self.label_2_marker, [self.desk_position[2][0], self.desk_position[2][1], self.arraw_2_height])


        target_position_0 = [self.desk_position[0][0], self.desk_position[0][1]]
        target_position_1 = [self.desk_position[1][0], self.desk_position[1][1]]
        target_position_2 = [self.desk_position[2][0], self.desk_position[2][1]]

        so3 = euler2so3(z=0, x=0, y=0)
        size = [0.2, 0.2, 0.0025]
        rgba =  [0.1, 0.9, 0.9, 0.8]
        marker_params = ["sphere", "", [target_position_0[0], target_position_0[1], 0.001], size, rgba, so3]
        if self.target_marker_0 is None:
            self.target_marker_0 = self.sim.viewer.add_marker(*marker_params)
        self.sim.viewer.update_marker_position(self.target_marker_0, [target_position_0[0], target_position_0[1], 0.005])
        so3 = euler2so3(z=0, x=0, y=0)
        size = [0.18, 0.18, 0.0025]
        rgba = [0.1, 0.9, 0.9, 0.8]
        marker_params = ["sphere", "", [target_position_1[0], target_position_1[1], 0.001], size, rgba, so3]
        if self.target_marker_1 is None:
            self.target_marker_1 = self.sim.viewer.add_marker(*marker_params)
        self.sim.viewer.update_marker_position(self.target_marker_1, [target_position_1[0], target_position_1[1], 0.005])
        so3 = euler2so3(z=0, x=0, y=0)
        size = [0.2, 0.2, 0.0025]
        rgba =  [0.1, 0.9, 0.9, 0.8]
        marker_params = ["sphere", "", [target_position_2[0], target_position_2[1], 0.001], size, rgba, so3]
        if self.target_marker_2 is None:
            self.target_marker_2 = self.sim.viewer.add_marker(*marker_params)
        self.sim.viewer.update_marker_position(self.target_marker_2, [target_position_2[0], target_position_2[1], 0.005])


        target_position = [self.target_position[0], self.target_position[1]]

        so3 = euler2so3(z=0, x=0, y=0)
        size = [0.2, 0.2, 0.0025]
        rgba =  [0.1, 0.9, 0.9, 1.0]
        marker_params = ["sphere", "", [target_position[0], target_position[1], 0.001], size, rgba, so3]
        if self.target_marker is None:
            self.target_marker = self.sim.viewer.add_marker(*marker_params)

        self.sim.viewer.update_marker_position(self.target_marker, [target_position[0], target_position[1], 0.005])

        so3 = euler2so3(z=0, x=0, y=0)
        size = [0.18, 0.18, 0.004]
        rgba = [0.6, 0.9, 0.9, 0.8]
        marker_params = ["sphere", "", [target_position[0], target_position[1], 0.001], size, rgba, so3]
        if self.target_marker_inside is None:
            self.target_marker_inside = self.sim.viewer.add_marker(*marker_params)

        self.sim.viewer.update_marker_position(self.target_marker_inside, [target_position[0], target_position[1], 0.005])

        if self.time_step % 30 == 0:
            self.walk_marker_change = not self.walk_marker_change
        
        if self.walk_marker_change:
            self.sim.viewer.update_marker_rgba(self.target_marker, [0.9, 0.2, 0.2, 1.0])
            self.sim.viewer.update_marker_rgba(self.target_marker_inside, [0.9, 0.4, 0.4, 0.85])
        else:
            self.sim.viewer.update_marker_rgba(self.target_marker, [0.9, 0.4, 0.4, 0.85])
            self.sim.viewer.update_marker_rgba(self.target_marker_inside, [0.9, 0.2, 0.2, 1.0])

        if self.stand_bit == 0:
            self.sim.viewer.update_marker_rgba(self.target_marker, [0.6, 0.9, 0.9, 0.8])
            self.sim.viewer.update_marker_rgba(self.target_marker_inside, [0.1, 0.9, 0.9, 1.0])
        
        self.sim.viewer.update_tasks_skills_menu(self.action_process[self.action_process_index], str(self.finish_count))