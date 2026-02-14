import numpy as np

from env.robots.base_robot import BaseRobot
from util.colors import FAIL, WARNING, ENDC
from sim import MjH1SimBoxTowerOfHanoi


class H1(BaseRobot):
    def __init__(
        self,
        simulator_type: str,
        terrain: str,
        state_est: bool,
    ):
        """Robot class for H1 defining robot and sim.
        This class houses all bot specific stuff for H1.

        Args:
            simulator_type (str): "mujoco" or "ar_async"
            terrain (str): Type of terrain generation [stone, stair, obstacle...]. Initialize inside
                          each subenv class to support individual use case.
            state_est (bool): Whether to use state estimation
        """
        super().__init__(robot_name="h1", simulator_type=simulator_type)

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
        self._min_base_height = 0.3
        
        # Offset based on H1's default joint positions (all zeros for now)
        self._offset = np.array([0.0] * 19)

        self._motor_mirror_indices = [
            -5, -6, -7, -8, -9,  # right leg (mirror of left leg 0-4)
            -0.1, -1, -2, -3, -4,  # left leg (mirror of right leg 5-9)
            -10,  # torso (no mirror)
            -15, -16, -17, -18,  # right arm (mirror of left arm 11-14)
            -11, -12, -13, -14   # left arm (mirror of right arm 15-18)
        ]

        # Robot state mirror indices for H1
        # State order: base_orient (4) + base_ang_vel (3) + motor_pos (19) + motor_vel (19) + joint_pos (0) + joint_vel (0)
        # Total: 4 + 3 + 19 + 19 = 45
        self._robot_state_mirror_indices = [
            0.01, -1, 2, -3,              # base orientation (w, x, y, z)
            -4, 5, -6,                    # base rotational vel (roll, pitch, yaw)
            -5, -6, -7, -8, -9,           # right leg motor pos (mirror of left leg 0-4)
            -0.1, -1, -2, -3, -4,         # left leg motor pos (mirror of right leg 5-9)
            -10,                           # torso motor pos (no mirror)
            -15, -16, -17, -18,           # right arm motor pos (mirror of left arm 11-14)
            -11, -12, -13, -14,           # left arm motor pos (mirror of right arm 15-18)
            -24, -25, -26, -27, -28,      # right leg motor vel
            -19, -20, -21, -22, -23,      # left leg motor vel
            -29,                           # torso motor vel
            -34, -35, -36, -37,           # right arm motor vel
            -30, -31, -32, -33,           # left arm motor vel
        ]

        # H1 output names (19 actuators) - matching motor_position_inds order (XML body hierarchy)
        self.output_names = [
            "left-hip-yaw", "left-hip-roll", "left-hip-pitch", "left-knee", "left-ankle",
            "right-hip-yaw", "right-hip-roll", "right-hip-pitch", "right-knee", "right-ankle",
            "torso",
            "left-shoulder-pitch", "left-shoulder-roll", "left-shoulder-yaw", "left-elbow",
            "right-shoulder-pitch", "right-shoulder-roll", "right-shoulder-yaw", "right-elbow"
        ]
        
        # H1 robot state names (base_orient (4) + base_ang_vel (3) + motor_pos (19) + motor_vel (19) = 45)
        self.robot_state_names = [
            "base-orientation-w", "base-orientation-x", "base-orientation-y", "base-orientation-z",
            "base-roll-velocity", "base-pitch-velocity", "base-yaw-velocity",
            # Motor positions (20) - following qpos order: left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee, left_ankle, right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee, right_ankle, torso, left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow, right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow
            "left-hip-yaw-pos", "left-hip-roll-pos", "left-hip-pitch-pos", "left-knee-pos", "left-ankle-pos",
            "right-hip-yaw-pos", "right-hip-roll-pos", "right-hip-pitch-pos", "right-knee-pos", "right-ankle-pos",
            "torso-pos",
            "left-shoulder-pitch-pos", "left-shoulder-roll-pos", "left-shoulder-yaw-pos", "left-elbow-pos",
            "right-shoulder-pitch-pos", "right-shoulder-roll-pos", "right-shoulder-yaw-pos", "right-elbow-pos",
            # Motor velocities (20) - same order as positions
            "left-hip-yaw-vel", "left-hip-roll-vel", "left-hip-pitch-vel", "left-knee-vel", "left-ankle-vel",
            "right-hip-yaw-vel", "right-hip-roll-vel", "right-hip-pitch-vel", "right-knee-vel", "right-ankle-vel",
            "torso-vel",
            "left-shoulder-pitch-vel", "left-shoulder-roll-vel", "left-shoulder-yaw-vel", "left-elbow-vel",
            "right-shoulder-pitch-vel", "right-shoulder-roll-vel", "right-shoulder-yaw-vel", "right-elbow-vel"
        ]

        # Select simulator
        self.state_est = state_est
        if "mesh" in simulator_type:
            simulator_type = simulator_type.replace("_mesh", "")

        if simulator_type == "box_tower_of_hanoi":
            self._sim = MjH1SimBoxTowerOfHanoi()
        else:
            raise RuntimeError(f"{FAIL}Simulator type {simulator_type} not correct!"
                               "Select from 'box_tower_of_hanoi'.{ENDC}")

    def get_raw_robot_state(self):
        states = {}
        states['base_orient'] = self.sim.get_base_orientation()
        states['base_ang_vel'] = self.sim.data.sensor('pelvis/imu-gyro').data
        states['motor_pos'] = self.sim.get_motor_position()
        states['motor_vel'] = self.sim.get_motor_velocity()
        states['joint_pos'] = self.sim.get_joint_position()
        states['joint_vel'] = self.sim.get_joint_velocity()
        return states

