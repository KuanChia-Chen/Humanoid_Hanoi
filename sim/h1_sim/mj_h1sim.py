import numpy as np
import pathlib

from sim import MujocoSim


class MjH1Sim(MujocoSim):

    """
    Wrapper for H1 Mujoco. This class only defines several specifics for H1.
    """
    def __init__(self, model_name: str = "h1-box-tower-of-hanoi.xml", terrain=None):
        model_path = pathlib.Path(__file__).parent.resolve() / model_name
        
        # Number of sim steps before commanded torque is actually applied
        self.torque_delay_cycles = 6
        self.torque_efficiency = 1.0

        # H1 19 actuators: base (7 DOF) + left leg (5) + right leg (5) + torso (1) + left arm (4) + right arm (4) = 26 qpos
        self.motor_position_inds = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
        self.motor_velocity_inds = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        self.joint_position_inds = []
        self.joint_velocity_inds = []  # H1 has all joints actuated

        # Base indices
        self.base_position_inds = [0, 1, 2]
        self.base_orientation_inds = [3, 4, 5, 6]
        self.base_linear_velocity_inds = [0, 1, 2]
        self.base_angular_velocity_inds = [3, 4, 5]
        
        # Arm indices (left arm: qpos 18-21, right arm: qpos 22-25)
        self.arm_position_inds = [18, 19, 20, 21, 22, 23, 24, 25]
        self.arm_velocity_inds = [17, 18, 19, 20, 21, 22, 23, 24]
        self.arm_actuator_inds = [11, 12, 13, 14, 15, 16, 17, 18]
        
        # Body names - H1 XML structure
        self.base_body_name = "pelvis"
        self.feet_site_name = ["left-foot-mid", "right-foot-mid"]  # pose purpose (need to add sites)
        self.feet_body_name = ["left_ankle_link", "right_ankle_link"]  # force purpose
        self.hand_body_name = ["left_elbow_link_ball_hand", "right_elbow_link_ball_hand"]  # force purpose
        self.hand_site_name = ["left-hand", "right-hand"]  # pose purpose (need to add sites)

        self.num_actuators = len(self.motor_position_inds)
        self.num_joints = len(self.joint_position_inds)
        
        # H1 default pose: base (7) + left leg (5) + right leg (5) + torso (1) + left arm (4) + right arm (4) = 26
        self.reset_qpos = np.array([0, 0, 1.1, 1, 0, 0, 0,  # Base position and orientation (7)
                                     0.0, 0.0, 0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0, 0.0, 0.0,
                                     0.0,
                                     0.0, 0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0, 0.0
                                    ])

        # NOTE: Have to call super init AFTER index arrays are defined
        super().__init__(model_path=model_path, terrain=terrain)

        self.simulator_rate = int(1 / self.model.opt.timestep)

        self.offset = self.reset_qpos[self.motor_position_inds]
        
        # PD gains for H1 (19 actuators: 5 left leg + 5 right leg + 1 torso + 4 left arm + 4 right arm)
        self.kp = np.array([80, 80, 80, 110, 40,  # Left leg (hip_yaw, hip_roll, hip_pitch, knee, ankle)
                            80, 80, 80, 110, 40,  # Right leg (hip_yaw, hip_roll, hip_pitch, knee, ankle)
                            80,  # Torso
                            80, 80, 80, 80,  # Left arm (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow)
                            80, 80, 80, 80])  # Right arm (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow)
        self.kd = np.array([8, 8, 8, 10, 6,  # Left leg
                            8, 8, 8, 10, 6,  # Right leg
                            8,  # Torso
                            8, 8, 8, 8,  # Left arm
                            8, 8, 8, 8])  # Right arm

        # List of bodies that cannot (prefer not) collide with environment
        self.body_collision_list = \
            ['left_knee_link', 'left_ankle_link',
             'right_knee_link', 'right_ankle_link']

        # minimal list of unwanted collisions to avoid knee walking
        self.knee_walking_list = \
            ['left_knee_link', 'right_knee_link']

        # Torque limits for H1 actuators (Nm) - based on XML actuatorfrcrange
        self.output_torque_limit = np.array([200, 200, 200, 300, 40,  # Left leg (hip_yaw, hip_roll, hip_pitch, knee, ankle)
                                           200, 200, 200, 300, 40,  # Right leg (hip_yaw, hip_roll, hip_pitch, knee, ankle)
                                           200,
                                           40, 40, 18, 18,
                                           40, 40, 18, 18])
        
        # Output damping limit is in Nm/(rad/s)
        self.output_damping_limit = np.ones(self.num_actuators) * 50.0
        
        # Output velocity limit is in rad/s
        self.output_motor_velocity_limit = np.ones(self.num_actuators) * 10.0
        
        # Input motor velocity limit is in RPM
        self.input_motor_velocity_max = \
            self.output_motor_velocity_limit * \
            self.model.actuator_gear[:, 0] * 60 / (2 * np.pi)

