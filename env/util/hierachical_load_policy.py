import torch
import pickle
import os
from util.nn_factory import nn_factory, load_checkpoint
from util.quaternion import mj2scipy
import numpy as np
from scipy.spatial.transform import Rotation as R

class MHCController:

    def __init__(self, env):
        self.env = env

        worker_path = "./pretrained_models/mhc_sep25_2025/"
        previous_args_dict = pickle.load(open(os.path.join(worker_path, "experiment.pkl"), "rb"))
        worker_actor_checkpoint = torch.load(os.path.join(worker_path, 'actor.pt'), map_location='cpu')

        self.worker_actor, _ = nn_factory(args=previous_args_dict['nn_args'])
        load_checkpoint(model=self.worker_actor, model_dict=worker_actor_checkpoint)
        self.worker_actor.eval()
        self.worker_actor.training = False
        print("MHC_actor", self.worker_actor)

    def get_action(self, state):
        return self.worker_actor(state).detach().numpy()

    def make_mhc_state(self, action):

        mhc_state = torch.zeros(213)
        mhc_state[:67] = torch.tensor(self.env.get_robot_state(), dtype=torch.float32)
        mhc_state[67:74] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, action[8], action[9]], dtype=torch.float32)
        mhc_state[128:132] = torch.tensor(action[:4], dtype=torch.float32)
        mhc_state[155:159] = torch.tensor(action[4:8], dtype=torch.float32)

        self.set_offset(mhc_state)

        return mhc_state
    
    def set_offset(self, mhc_state):
        motor_pos_id = [7, 8, 9, 14, 18, 23, 30, 31, 32, 33, 34, 35, 36, 41, 45, 50, 57, 58, 59, 60]
        offset = mhc_state[(74+24):(74+24+61)][motor_pos_id]
        offset_actr_inds = [6, 7, 8, 9, 16, 17, 18, 19]
        self.env.robot._offset[offset_actr_inds] = offset[offset_actr_inds]

    def reset_hidden_state(self):
        self.worker_actor.init_hidden_state()

class PutDownMHCController:

    def __init__(self, env):
        self.env = env

        worker_path = "./pretrained_models/mhc_finetune/finetune_putdown_policy/"
        previous_args_dict = pickle.load(open(os.path.join(worker_path, "experiment.pkl"), "rb"))
        worker_actor_checkpoint = torch.load(os.path.join(worker_path, 'actor.pt'), map_location='cpu')

        self.worker_actor, _ = nn_factory(args=previous_args_dict['nn_args'])
        load_checkpoint(model=self.worker_actor, model_dict=worker_actor_checkpoint)
        self.worker_actor.eval()
        self.worker_actor.training = False
        print("MHC_actor", self.worker_actor)

    def get_action(self, state):
        return self.worker_actor(state).detach().numpy()

    def make_mhc_state(self, action):

        mhc_state = torch.zeros(213)
        mhc_state[:67] = torch.tensor(self.env.get_robot_state(), dtype=torch.float32)
        mhc_state[67:74] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, action[8], action[9]], dtype=torch.float32)
        mhc_state[128:132] = torch.tensor(action[:4], dtype=torch.float32)
        mhc_state[155:159] = torch.tensor(action[4:8], dtype=torch.float32)

        self.set_offset(mhc_state)

        return mhc_state
    
    def set_offset(self, mhc_state):
        motor_pos_id = [7, 8, 9, 14, 18, 23, 30, 31, 32, 33, 34, 35, 36, 41, 45, 50, 57, 58, 59, 60]
        offset = mhc_state[(74+24):(74+24+61)][motor_pos_id]
        offset_actr_inds = [6, 7, 8, 9, 16, 17, 18, 19]
        self.env.robot._offset[offset_actr_inds] = offset[offset_actr_inds]

    def reset_hidden_state(self):
        self.worker_actor.init_hidden_state()

class PickUpMHCController:

    def __init__(self, env):
        self.env = env

        worker_path = "./pretrained_models/mhc_finetune/finetune_pickup_policy/"
        previous_args_dict = pickle.load(open(os.path.join(worker_path, "experiment.pkl"), "rb"))
        worker_actor_checkpoint = torch.load(os.path.join(worker_path, 'actor.pt'), map_location='cpu')

        self.worker_actor, _ = nn_factory(args=previous_args_dict['nn_args'])
        load_checkpoint(model=self.worker_actor, model_dict=worker_actor_checkpoint)
        self.worker_actor.eval()
        self.worker_actor.training = False
        print("MHC_actor", self.worker_actor)

    def get_action(self, state):
        return self.worker_actor(state).detach().numpy()

    def make_mhc_state(self, action):

        mhc_state = torch.zeros(213)
        mhc_state[:67] = torch.tensor(self.env.get_robot_state(), dtype=torch.float32)
        mhc_state[67:74] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, action[8], action[9]], dtype=torch.float32)
        mhc_state[128:132] = torch.tensor(action[:4], dtype=torch.float32)
        mhc_state[155:159] = torch.tensor(action[4:8], dtype=torch.float32)

        self.set_offset(mhc_state)

        return mhc_state
    
    def set_offset(self, mhc_state):
        motor_pos_id = [7, 8, 9, 14, 18, 23, 30, 31, 32, 33, 34, 35, 36, 41, 45, 50, 57, 58, 59, 60]
        offset = mhc_state[(74+24):(74+24+61)][motor_pos_id]
        offset_actr_inds = [6, 7, 8, 9, 16, 17, 18, 19]
        self.env.robot._offset[offset_actr_inds] = offset[offset_actr_inds]

    def reset_hidden_state(self):
        self.worker_actor.init_hidden_state()

class PickUpController:

    def __init__(self):

        worker_path = "./pretrained_models/pick_up_box/11-21-21-47/"
        previous_args_dict = pickle.load(open(os.path.join(worker_path, "experiment.pkl"), "rb"))
        worker_actor_checkpoint = torch.load(os.path.join(worker_path, 'actor.pt'), map_location='cpu')

        self.worker_actor, _ = nn_factory(args=previous_args_dict['nn_args'])
        load_checkpoint(model=self.worker_actor, model_dict=worker_actor_checkpoint)
        self.worker_actor.eval()
        self.worker_actor.training = False
        print("Pick_Up_actor", self.worker_actor)

    def get_action(self, state):

        state = torch.tensor(state).float()
        return self.worker_actor(state).detach().numpy()

    def reset_hidden_state(self):
        self.worker_actor.init_hidden_state()
   
class PutDownController:

    def __init__(self):

        worker_path = "./pretrained_models/put_down_box/12-26-16-10 (copy)/"
        previous_args_dict = pickle.load(open(os.path.join(worker_path, "experiment.pkl"), "rb"))
        worker_actor_checkpoint = torch.load(os.path.join(worker_path, 'actor.pt'), map_location='cpu')

        previous_args_dict['nn_args'].arch = "lstm"
        self.worker_actor, _ = nn_factory(args=previous_args_dict['nn_args'])
        load_checkpoint(model=self.worker_actor, model_dict=worker_actor_checkpoint)
        self.worker_actor.eval()
        self.worker_actor.training = False
        print("Put_Down_actor", self.worker_actor)

    def get_action(self, robot_state, box_info):
        state = torch.tensor(np.concatenate((robot_state, box_info))).float()
        return self.worker_actor(state).detach().numpy()

    def reset_hidden_state(self):
        self.worker_actor.init_hidden_state()
   
class PosDeltaController:

    def __init__(self, env):
        self.env = env

        worker_path = "./pretrained_models/pos_delta/posdelta_0105/"
        previous_args_dict = pickle.load(open(os.path.join(worker_path, "experiment.pkl"), "rb"))
        worker_actor_checkpoint = torch.load(os.path.join(worker_path, 'actor.pt'), map_location='cpu')

        previous_args_dict['nn_args'].arch = "lstm"
        self.worker_actor, _ = nn_factory(args=previous_args_dict['nn_args'])
        load_checkpoint(model=self.worker_actor, model_dict=worker_actor_checkpoint)
        self.worker_actor.eval()
        self.worker_actor.training = False
        print("Pos_Delta_actor", self.worker_actor)

    def get_action(self, robot_state, delta_pose_cmd):
        state = torch.tensor(np.concatenate((robot_state, delta_pose_cmd))).float()
        return self.worker_actor(state).detach().numpy()

    def reset_hidden_state(self):
        self.worker_actor.init_hidden_state()

    def make_mhc_state_pos_delta(self, action, base_pose = None):
        mhc_state = torch.zeros(213)
        mhc_state[:67] = torch.tensor(self.env.get_robot_state(), dtype=torch.float32)

        vx, vy, vyaw = action[0], action[1], action[2]

        vx = max(min(vx, 0.3), -0.3)
        vy = max(min(vy, 0.2), -0.2)
        vyaw = max(min(vyaw, 0.3), -0.3)

        if self.env.stand_bit == 0:
            vx, vy, vyaw = 0.0, 0.0, 0.0
            self.env.stand_cmd_bit = 1

        if self.env.stand_cmd_bit == 1:
            vx, vy, vyaw = 0.0, 0.0, 0.0
            self.env.stand_bit = 0
        else:
            self.env.stand_bit = 1

        self.env.orient_add += vyaw / self.env.default_policy_rate

        mhc_state[67:74] = torch.tensor([self.env.stand_bit, vx, vy, vyaw, 0.0, -0.15, 0.85], dtype=torch.float32)
        mhc_state[128:132] = torch.tensor([-1.05542715e-01, 8.94852532e-01, -8.63756398e-03, 3.44780280e-01], dtype=torch.float32)
        mhc_state[155:159] = torch.tensor([1.05444698e-01, -8.94890429e-01, 8.85979401e-03, -3.44723293e-01], dtype=torch.float32)
        
        return mhc_state

    def wrap_to_pi(self, x):
        return (x + np.pi) % (2 * np.pi) - np.pi

    def update_delta_commands(self, base_pose = None):
        # Update the delta commands based on the current local position and the command
        base_pose = self.env.sim.get_body_pose(self.env.sim.base_body_name)
        local_x = base_pose[0]
        local_y = base_pose[1]
        local_yaw = R.from_quat(mj2scipy(base_pose[3:])).as_euler('xyz')[2]
        diff_x = self.env.target_position[0] - local_x
        diff_y = self.env.target_position[1] - local_y
        if np.abs(diff_y) > 1.5:
            diff_x -= 0.6
        if np.abs(diff_y) > 0.5:
            diff_x -= 0.3
        # if np.linalg.norm(np.array([diff_x, diff_y])) > 0.4 or np.abs(self.env.target_rotation - local_yaw) > np.pi / 9:
        #     diff_x -= 0.3
        orient_cos = np.cos(local_yaw)
        orient_sin = np.sin(local_yaw)
        local_delta = np.array([
            diff_x * orient_cos + diff_y * orient_sin,
            -diff_x * orient_sin + diff_y * orient_cos,
        ])
        norm = np.linalg.norm(local_delta)
        clip_norm = 2.0
        if norm > clip_norm:
            local_delta = local_delta / (norm + 1e-8) * clip_norm
        del_x, del_y = local_delta
        del_yaw = self.env.target_rotation - local_yaw
        del_yaw = self.wrap_to_pi(del_yaw)
        stand_cmd_bit = int(np.linalg.norm(np.array([del_x, del_y, del_yaw])) < 0.3)

        return stand_cmd_bit, del_x, del_y, del_yaw

class PosDeltaTargetController:

    def __init__(self, env):
        self.env = env

        worker_path = "./pretrained_models/pos_delta_target/01-07-03-06/"
        previous_args_dict = pickle.load(open(os.path.join(worker_path, "experiment.pkl"), "rb"))
        worker_actor_checkpoint = torch.load(os.path.join(worker_path, 'actor.pt'), map_location='cpu')

        self.worker_actor, _ = nn_factory(args=previous_args_dict['nn_args'])
        load_checkpoint(model=self.worker_actor, model_dict=worker_actor_checkpoint)
        self.worker_actor.eval()
        self.worker_actor.training = False
        print("Pos_Delta_Target_actor", self.worker_actor)

        self.stand_cmd_bit = 0

    def get_action(self, robot_state, total_delta_pose_cmd):
        
        state = torch.tensor(np.concatenate((robot_state, total_delta_pose_cmd)), dtype=torch.float32)
        return self.worker_actor(state).detach().numpy()

    def reset_hidden_state(self):
        self.worker_actor.init_hidden_state()

    def make_mhc_state_pos_delta_target(self, action):
        mhc_state = torch.zeros(213)
        mhc_state[:67] = torch.tensor(self.env.get_robot_state(), dtype=torch.float32)

        vx, vy, vyaw = action[8], action[9], action[10]

        vx = max(min(vx, 0.4), -0.4)
        vy = max(min(vy, 0.2), -0.2)
        vyaw = max(min(vyaw, 0.4), -0.4)

        if self.stand_cmd_bit == 1:
            stand_cmd = 0.0
        else:
            stand_cmd = 1.0

        if vx == 0.0 and vy == 0.0 and vyaw == 0.0:
            vx, vy, vyaw = 0.0, 0.0, 0.0
            self.env.stand_bit = 0
        else:
            self.env.stand_bit = 1.0
        
        self.env.orient_add += vyaw / self.env.default_policy_rate

        mhc_state[67:74] = torch.tensor([stand_cmd, vx, vy, vyaw, 0.0, 0.0, 0.85], dtype=torch.float32)
        mhc_state[128:132] = torch.tensor(action[:4], dtype=torch.float32)
        mhc_state[155:159] = torch.tensor(action[4:8], dtype=torch.float32)
        
        return mhc_state

    def wrap_to_pi(self, x):
        return (x + np.pi) % (2 * np.pi) - np.pi

    def update_delta_commands(self, base_pose = None):
        # Update the delta commands based on the current local position and the command
        base_pose = self.env.sim.get_body_pose(self.env.sim.base_body_name)
        local_x = base_pose[0]
        local_y = base_pose[1]
        local_yaw = R.from_quat(mj2scipy(base_pose[3:])).as_euler('xyz')[2]
        local_yaw = self.wrap_to_pi(local_yaw)
        
        diff_x = self.env.target_position[0] - local_x
        diff_y = self.env.target_position[1] - local_y
        if np.abs(diff_y) > 1.5:
            diff_x -= 0.6
        if np.abs(diff_y) > 0.5:
            diff_x -= 0.3
        # if np.linalg.norm(np.array([diff_x, diff_y])) > 0.2 or np.abs(self.env.target_rotation - local_yaw) > np.pi / 9:
        #     diff_x += 0.1
        orient_cos = np.cos(local_yaw)
        orient_sin = np.sin(local_yaw)
        
        local_delta = np.array([
            diff_x * orient_cos + diff_y * orient_sin,
            -diff_x * orient_sin + diff_y * orient_cos,
        ])
        
        norm = np.linalg.norm(local_delta)
        clip_norm = 2.0
        if norm > clip_norm:
            local_delta = local_delta / (norm + 1e-8) * clip_norm
        del_x, del_y = local_delta
        del_yaw = self.env.target_rotation - local_yaw
        del_yaw = self.wrap_to_pi(del_yaw)

        vel_cmd = np.array([del_x, del_y, del_yaw])
        self.stand_cmd_bit = int(np.linalg.norm(vel_cmd) < 0.1)

        return self.stand_cmd_bit, del_x, del_y, del_yaw