import numpy as np
from scipy.spatial.transform import Rotation as R

from env.tasks.locomotionenv.locomotionenv import LocomotionEnv
from util.quaternion import *

wrap_to_pi = lambda x: (x + np.pi) % (2 * np.pi) - np.pi

def _compute_constellation_points(pose, radius):
    """
    Generates eight points around a heading plus the center point.
    """
    x, y, theta = pose
    num_points = 8
    angle_offsets = np.arange(num_points) * np.pi / num_points
    angles = theta + angle_offsets

    points_x = x + radius * np.cos(angles)
    points_y = y + radius * np.sin(angles)
    points_x = np.concatenate([points_x, [x]], axis=0)
    points_y = np.concatenate([points_y, [y]], axis=0)

    return np.stack([points_x, points_y], axis=1)


def compute_rewards(self, action):
    q = {}

    l_foot_force = np.linalg.norm(self.feet_grf_tracker_avg[self.sim.feet_body_name[0]])
    r_foot_force = np.linalg.norm(self.feet_grf_tracker_avg[self.sim.feet_body_name[1]])
    l_foot_pose = self.sim.get_site_pose(self.sim.feet_site_name[0])
    r_foot_pose = self.sim.get_site_pose(self.sim.feet_site_name[1])

    base_pose = self.sim.get_body_pose(self.sim.base_body_name)

    BASE_LENGTH, BASE_WIDTH = 0.0, 0.35
    l_foot_in_base = self.sim.get_relative_pose(base_pose, l_foot_pose)
    r_foot_in_base = self.sim.get_relative_pose(base_pose, r_foot_pose)
    length = np.abs(l_foot_in_base[0] - r_foot_in_base[0])
    width = np.abs(l_foot_in_base[1] - r_foot_in_base[1])
    stance_length_error = np.abs(length - BASE_LENGTH)
    stance_width_error = np.abs(width - BASE_WIDTH)
    stance_error = stance_length_error + stance_width_error

    # Stance when standing still 
    zero_cmd = bool(self.stand_bit)
    if zero_cmd:
        q["stance_feet_pos"] = stance_error
    else:
        q["stance_feet_pos"] = 0.0

    ### Constellation rewards ###
    radius = 0.2
    current_points = _compute_constellation_points((self.local_x, self.local_y, self.local_yaw), radius)
    target_points = _compute_constellation_points((self.command_x, self.command_y, self.command_yaw), radius)
    q["constellation_exp"] = np.linalg.norm(current_points - target_points, axis=1).sum()

    ### Orientation rewards (base and feet) ###
    base_euler = R.from_quat(mj2scipy(base_pose[3:])).as_euler('xyz')

    q["yaw_error"] = np.abs(wrap_to_pi(self.command_yaw - self.local_yaw))
    pos_err = np.linalg.norm([self.command_x - self.local_x, self.command_y - self.local_y])
    q["pos_err"] = float(pos_err)

    ### Sim2real stability rewards ###
    if self.simulator_type == "libcassie" and self.state_est:
        base_acc = self.sim.robot_estimator_state.pelvis.translationalAcceleration[:]
    else:
        base_acc = self.sim.get_body_acceleration(self.sim.base_body_name)
    q["stable_base"] = np.abs(base_acc).sum()

    if self.last_action is not None:
        q["ctrl_penalty"] = np.sum(np.abs(self.last_action - action)) / len(action)
    else:
        q["ctrl_penalty"] = 0
    if self.simulator_type == "libcassie" and self.state_est:
        torque = self.sim.get_torque(state_est = self.state_est)
    else:
        torque = self.sim.get_torque()
    # Normalized by torque limit, sum worst case is 10, usually around 1 to 2
    q["trq_penalty"] = np.mean(np.abs(torque)/self.sim.output_torque_limit)

    # energy penalty: torque . joint velocities
    joint_vel = self.sim.get_motor_velocity()
    q["energy_penalty"] = np.mean(np.abs(torque * joint_vel))

    # q["llc_cmd_reward"] = sum(np.abs(self.last_llc_cmd - self.new_llc_cmd)) / len(self.new_llc_cmd) if self.last_llc_cmd is not None else 100

    # Penalize velocities exceeding 0.4 m/s
    # q["vel_penalty"] = 0.0
    # if np.abs(action[8]) > 0.4:
    #     q["vel_penalty"] -= (np.abs(action[8]) - 0.4)
    # if np.abs(action[9]) > 0.4:
    #     q["vel_penalty"] -= (np.abs(action[9]) - 0.4)
    # if np.abs(action[10]) > 1.0:
    #     q["vel_penalty"] -= (np.abs(action[10]) - 1.0)

    # Penalize large commanded planar velocities
    q["cmd_mag_penalty"] = -np.linalg.norm(action[8:11])

    ######
    l_arm_contact_frc, l_arm_contact_pt = self.sim.get_body_to_body_contact_force_point("left-arm/elbow", "box")
    r_arm_contact_frc, r_arm_contact_pt = self.sim.get_body_to_body_contact_force_point("right-arm/elbow", "box")
    l_arm_contact_mag = np.linalg.norm(l_arm_contact_frc)
    r_arm_contact_mag = np.linalg.norm(r_arm_contact_frc)
    q["box_contact_reward"] = 1.0 if (l_arm_contact_mag > 1.0 and r_arm_contact_mag > 1.0) else 0.0

    box_pose = self.sim.get_box_pose()
    if not hasattr(self, "initial_box_in_base"):
        # Cache the initial box pose in base frame once per episode
        self.initial_box_in_base = self.sim.get_relative_pose(base_pose, box_pose)

    # Keep orientation of the box same as initial box orientation (Relative to Base)
    # Using relative orientation ensures the robot isn't penalized for turning (yaw) while holding the box.
    box_in_base = self.sim.get_relative_pose(base_pose, box_pose)
    
    box_quat_curr_rel = box_in_base[3:]
    box_quat_init_rel = self.initial_box_in_base[3:]
    
    # Calculate orientation difference
    box_orient_diff = quaternion_distance(box_quat_curr_rel, box_quat_init_rel)
    q["box_orientation_reward"] = box_orient_diff

    box_pos_diff = np.linalg.norm(box_in_base[:3] - self.initial_box_in_base[:3])
    q["box_position_reward"] = box_pos_diff

    # Box Center Alignment Reward
    q["box_center_reward"] = np.abs(box_in_base[1])

    # Get Hand info
    l_hand_pose = self.sim.get_site_pose("left-hand")
    r_hand_pose = self.sim.get_site_pose("right-hand")
    l_hand_roll = R.from_quat(mj2scipy(l_hand_pose[3:])).as_euler('xyz')[0]
    r_hand_roll = R.from_quat(mj2scipy(r_hand_pose[3:])).as_euler('xyz')[0]
    q['hand_roll_penalty'] = (abs(l_hand_roll) + abs(r_hand_roll))

    # l_elbow_target = np.array([0.04, 0.25])   # (x,y) in base frame
    # r_elbow_target = np.array([0.04, -0.25])  # (x,y) in base frame
    # l_elbow_pose = self.sim.get_body_pose("left-arm/elbow")
    # r_elbow_pose = self.sim.get_body_pose("right-arm/elbow")
    # l_elbow_in_base = self.sim.get_relative_pose(base_pose, l_elbow_pose)[:3]
    # r_elbow_in_base = self.sim.get_relative_pose(base_pose, r_elbow_pose)[:3]
    # l_elbow_err = np.linalg.norm(l_elbow_in_base[0:2] - l_elbow_target)
    # r_elbow_err = np.linalg.norm(r_elbow_in_base[0:2] - r_elbow_target)
    # q["elbow_err"] = l_elbow_err + r_elbow_err

    # Height command penalty (range check)
    # q["height_cmd_penalty"] = -0.1 if action[11] < 0.7 or action[11] > 1.0 else 0.0

    return q

def compute_done(self):
    base_pose = self.sim.get_body_pose(self.sim.base_body_name)
    floor_quat = self.sim.get_geom_pose('floor')[3:]
    floor_rot = R.from_quat(mj2scipy(floor_quat))
    rotated_base_pose = floor_rot.inv().apply(base_pose[:3])
    current_height = rotated_base_pose[2]
    base_euler = R.from_quat(mj2scipy(base_pose[3:])).as_euler('xyz')

    # if hasattr(self, "local_goal_distance") and self.local_goal_distance > 5.0:
    #     return True

    # if abs(base_pose[0]) > 3.2 or abs(base_pose[1]) > 3.2:
    #     return True
    
    # for b in self.sim.knee_walking_list:
    #     if self.sim.is_body_collision(b):
    #         return True
    # if np.abs(base_euler[1]) > 60/180*np.pi or np.abs(base_euler[0]) > 60/180*np.pi or current_height < self.robot.min_base_height:
    #     return True

    # l_arm_contact_frc, l_arm_contact_pt = self.sim.get_body_to_body_contact_force_point("left-arm/elbow", "box")
    # r_arm_contact_frc, r_arm_contact_pt = self.sim.get_body_to_body_contact_force_point("right-arm/elbow", "box")
    # l_arm_contact_mag = np.linalg.norm(l_arm_contact_frc)
    # r_arm_contact_mag = np.linalg.norm(r_arm_contact_frc)

    # if self.time > 10:
    #     if l_arm_contact_mag < 1.0 and r_arm_contact_mag < 1.0:
    #         return True

    # # Timeout
    # if self.time > 599:
    #     return True

    # if hasattr(self, "knee_collide"):
    #     if self.knee_collide:
    #         return True 

    if base_pose[2] < 0.2:
        return True

    if self.current_skill == "walk_with_box":
        if self.box_number == 0:
            l_arm_contact = np.linalg.norm(self.sim.get_body_to_body_contact_force("left-arm/elbow", "box"))
            r_arm_contact = np.linalg.norm(self.sim.get_body_to_body_contact_force("right-arm/elbow", "box"))
        elif self.box_number == 1:
            l_arm_contact = np.linalg.norm(self.sim.get_body_to_body_contact_force("left-arm/elbow", "box1"))
            r_arm_contact = np.linalg.norm(self.sim.get_body_to_body_contact_force("right-arm/elbow", "box1"))
        elif self.box_number == 2:
            l_arm_contact = np.linalg.norm(self.sim.get_body_to_body_contact_force("left-arm/elbow", "box2"))
            r_arm_contact = np.linalg.norm(self.sim.get_body_to_body_contact_force("right-arm/elbow", "box2"))

        if l_arm_contact == 0 and r_arm_contact == 0:
            self.hand_force_reset_count += 1

        if self.hand_force_reset_count > 50:
            return True

    time_difference = self.time_step - self.pre_change_time
    if time_difference > 1000 and time_difference < 1006 and (self.current_skill == "walk_with_box" or self.current_skill == "walk_without_box"):
        #push robot
        self.base_adr = self.sim.get_body_adr(self.sim.base_body_name)
        self.sim.data.xfrc_applied[self.base_adr, 0:2] = np.array([150, 0.0])
    else:
        self.base_adr = self.sim.get_body_adr(self.sim.base_body_name)
        self.sim.data.xfrc_applied[self.base_adr, 0:2] = np.array([0.0, 0.0])

    if time_difference > 1500:
        return True

    
    if self.finish_cycle:
        return True
    
    return False
