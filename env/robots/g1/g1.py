import numpy as np

from env.robots.base_robot import BaseRobot
from util.colors import FAIL, WARNING, ENDC
from sim import MjG1SimBoxTowerOfHanoi


class G1(BaseRobot):
    def __init__(
        self,
        simulator_type: str,
        terrain: str,
        state_est: bool,
    ):
        """Robot class for G1 defining robot and sim.
        This class houses all bot specific stuff for G1.

        Args:
            simulator_type (str): "mujoco" or "ar_async"
            terrain (str): Type of terrain generation [stone, stair, obstacle...]. Initialize inside
                          each subenv class to support individual use case.
            state_est (bool): Whether to use state estimation
        """
        super().__init__(robot_name="g1", simulator_type=simulator_type)

        # G1 29DOF: 6 left leg + 6 right leg + 3 waist + 7 left arm + 7 right arm
        self.kp = np.array([80, 80, 80, 110, 40, 40,  # Left leg
                            80, 80, 80, 110, 40, 40,  # Right leg
                            80, 40, 40,  # Waist
                            80, 80, 80, 80, 80, 40, 40,  # Left arm
                            80, 80, 80, 80, 80, 40, 40])  # Right arm
        self.kd = np.array([8, 8, 8, 10, 6, 6,  # Left leg
                            8, 8, 8, 10, 6, 6,  # Right leg
                            8, 6, 6,  # Waist
                            8, 8, 8, 8, 8, 6, 6,  # Left arm
                            8, 8, 8, 8, 8, 6, 6])  # Right arm
        self._min_base_height = 0.3
        
        # Offset based on G1's default joint positions (all zeros for now)
        self._offset = np.array([0.0] * 29)

        # Mirror indices for G1 motor actions (29 actuators)
        # Left leg (0-5) <-> Right leg (6-11), Waist (12-14) stays same, Left arm (15-21) <-> Right arm (22-28)
        self._motor_mirror_indices = [
            -6, -7, -8, -9, -10, -11,  # right leg (mirror of left leg 0-5)
            -0.1, -1, -2, -3, -4, -5,  # left leg (mirror of right leg 6-11)
            -12, -13, -14,  # waist (no mirror, but keep for consistency)
            -22, -23, -24, -25, -26, -27, -28,  # right arm (mirror of left arm 15-21)
            -15, -16, -17, -18, -19, -20, -21   # left arm (mirror of right arm 22-28)
        ]

        # Robot state mirror indices for G1
        # State order: base_orient (4) + base_ang_vel (3) + motor_pos (29) + motor_vel (29) + joint_pos (0) + joint_vel (0)
        # Total: 4 + 3 + 29 + 29 = 65
        self._robot_state_mirror_indices = [
            0.01, -1, 2, -3,              # base orientation (w, x, y, z)
            -4, 5, -6,                    # base rotational vel (roll, pitch, yaw)
            # Motor positions (29): left leg (0-5), right leg (6-11), waist (12-14), left arm (15-21), right arm (22-28)
            -6, -7, -8, -9, -10, -11,     # right leg motor pos (mirror of left leg 0-5)
            -0.1, -1, -2, -3, -4, -5,     # left leg motor pos (mirror of right leg 6-11)
            -12, -13, -14,                 # waist motor pos (no mirror)
            -22, -23, -24, -25, -26, -27, -28,  # right arm motor pos (mirror of left arm 15-21)
            -15, -16, -17, -18, -19, -20, -21,   # left arm motor pos (mirror of right arm 22-28)
            # Motor velocities (29): same pattern as positions
            -35, -36, -37, -38, -39, -40, # right leg motor vel
            -29, -30, -31, -32, -33, -34, # left leg motor vel
            -41, -42, -43,                 # waist motor vel
            -51, -52, -53, -54, -55, -56, -57,  # right arm motor vel
            -44, -45, -46, -47, -48, -49, -50,   # left arm motor vel
        ]

        # G1 output names (29 actuators)
        self.output_names = [
            "left-hip-pitch", "left-hip-roll", "left-hip-yaw", "left-knee", "left-ankle-pitch", "left-ankle-roll",
            "right-hip-pitch", "right-hip-roll", "right-hip-yaw", "right-knee", "right-ankle-pitch", "right-ankle-roll",
            "waist-yaw", "waist-roll", "waist-pitch",
            "left-shoulder-pitch", "left-shoulder-roll", "left-shoulder-yaw", "left-elbow", "left-wrist-roll", "left-wrist-pitch", "left-wrist-yaw",
            "right-shoulder-pitch", "right-shoulder-roll", "right-shoulder-yaw", "right-elbow", "right-wrist-roll", "right-wrist-pitch", "right-wrist-yaw"
        ]
        
        # G1 robot state names (base_orient (4) + base_ang_vel (3) + motor_pos (29) + motor_vel (29) = 65)
        self.robot_state_names = [
            "base-orientation-w", "base-orientation-x", "base-orientation-y", "base-orientation-z",
            "base-roll-velocity", "base-pitch-velocity", "base-yaw-velocity",
            # Motor positions (29)
            "left-hip-pitch-pos", "left-hip-roll-pos", "left-hip-yaw-pos", "left-knee-pos", "left-ankle-pitch-pos", "left-ankle-roll-pos",
            "right-hip-pitch-pos", "right-hip-roll-pos", "right-hip-yaw-pos", "right-knee-pos", "right-ankle-pitch-pos", "right-ankle-roll-pos",
            "waist-yaw-pos", "waist-roll-pos", "waist-pitch-pos",
            "left-shoulder-pitch-pos", "left-shoulder-roll-pos", "left-shoulder-yaw-pos", "left-elbow-pos", "left-wrist-roll-pos", "left-wrist-pitch-pos", "left-wrist-yaw-pos",
            "right-shoulder-pitch-pos", "right-shoulder-roll-pos", "right-shoulder-yaw-pos", "right-elbow-pos", "right-wrist-roll-pos", "right-wrist-pitch-pos", "right-wrist-yaw-pos",
            # Motor velocities (29)
            "left-hip-pitch-vel", "left-hip-roll-vel", "left-hip-yaw-vel", "left-knee-vel", "left-ankle-pitch-vel", "left-ankle-roll-vel",
            "right-hip-pitch-vel", "right-hip-roll-vel", "right-hip-yaw-vel", "right-knee-vel", "right-ankle-pitch-vel", "right-ankle-roll-vel",
            "waist-yaw-vel", "waist-roll-vel", "waist-pitch-vel",
            "left-shoulder-pitch-vel", "left-shoulder-roll-vel", "left-shoulder-yaw-vel", "left-elbow-vel", "left-wrist-roll-vel", "left-wrist-pitch-vel", "left-wrist-yaw-vel",
            "right-shoulder-pitch-vel", "right-shoulder-roll-vel", "right-shoulder-yaw-vel", "right-elbow-vel", "right-wrist-roll-vel", "right-wrist-pitch-vel", "right-wrist-yaw-vel"
        ]

        # Select simulator
        self.state_est = state_est
        if "mesh" in simulator_type:
            simulator_type = simulator_type.replace("_mesh", "")

        if simulator_type == "box_tower_of_hanoi":
            self._sim = MjG1SimBoxTowerOfHanoi()
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

