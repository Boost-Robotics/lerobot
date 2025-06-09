# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains logic to instantiate a robot, read information from its motors and cameras,
and send orders to its motors.
"""
# TODO(rcadene, aliberts): reorganize the codebase into one file per robot, with the associated
# calibration procedure, to make it easy for people to add their own robot.

import json
import logging
import time
import warnings
from pathlib import Path
import threading

import numpy as np
import torch

from lerobot.common.robot_devices.cameras.utils import make_cameras_from_configs
from lerobot.common.robot_devices.motors.utils import MotorsBus, make_motors_buses_from_configs
from lerobot.common.robot_devices.robots.configs import ManipulatorRobotConfig
from lerobot.common.robot_devices.robots.utils import get_arm_id
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError
from xarm.wrapper import XArmAPI
from scipy.spatial.transform import Rotation

from pynput import keyboard
import pynput
import os
import ast
import rerun as rr


# Move arms to initial positions
ip_left = "192.168.1.11"
ip_right = "192.168.1.223"
first_angle_left = [90, -30, -5, 5, 55, 0, 0]
first_angle_right = [-90, -30, -5, 5, 55, 0, 0]
first_angles = [first_angle_left, first_angle_right]
x_arm_ips = [ip_left, ip_right]
arms = [XArmAPI(x_arm_ips[i]) for i in range(2)]  # Create two XArmAPI instances for left and right arms
first = [True,True]
active_gripper_threads = [None, None] # To store active gripper command threads
camera_thread = None
set_init = False # To track if initial positions have been set
redo_first = False # To track if we need to redo the first move
on_right = 0


def set_to_initial_positions():
    global set_init
    for i in range(2):
        arm = arms[i]
        arm.motion_enable(enable=True)
        arm.set_mode(1)
        arm.set_state(state=0)
        code = arm.set_gripper_mode(0)
        code = arm.set_gripper_enable(True)
        code = arm.set_gripper_speed(5000)
        arm.set_mode(0)
        arm.set_state(state=0)
        print("MOVING FIRST!!!")
        arm.set_servo_angle(angle=first_angles[i], speed=50, wait=True)
        arm.set_mode(1)
        arm.set_state(state=0)
    set_init = True


 # Use a global variable to track spacebar presses
space_pressed = False
def on_press(key):
    global space_pressed
    try:
        if key.char == ' ':
            space_pressed = True
    except AttributeError:
        if key == pynput.keyboard.Key.space:
            space_pressed = True

def on_release(key):
    global space_pressed
    try:
        if key.char == ' ':
            space_pressed = False
    except AttributeError:
        if key == pynput.keyboard.Key.space:
            space_pressed = False

# Initialize keyboard listener in a separate thread
keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
keyboard_listener.start()

fixed_points = []
with open("/home/hans/curr_wps.txt", "r") as f:
    #get point from first 4 lines
    for line in f.readlines():
        if line.strip():
            try:
                # Try to parse as a Python list
                point = ast.literal_eval(line.strip())
                print(f"Parsed point: {point}")
                if isinstance(point, list) and len(point) == 6:
                    # Add default gripper position if only position coordinates given
                    point.append(150)  # Default gripper position
                    fixed_points.append(point)
                elif isinstance(point, list) and len(point) == 7:
                    # If it's a tuple with 7 elements, assume it's already in the correct format
                    fixed_points.append(point)
            except (SyntaxError, ValueError):
                print(f"Could not parse line: {line.strip()}")

# ret, pose = arms[1].get_forward_kinematics(fixed_points[4], input_is_radian=True, return_is_radian=True)
# # print(pose)
# # pose[0] += 1
# ret, angles = arms[1].get_inverse_kinematics(pose, input_is_radian=True, return_is_radian=True)
# print("IK angles!", angles)



def cable_dance():
   
    arm = arms[1] 
    arm.set_mode(0)
    arm.set_state(state=0)
  
    
    print("MOVING FIRST!!!")
    arm.set_gripper_position(150, wait=True)
    #load fixed points from file
    global fixed_points
    global on_right
    print(f"Fixed points loaded: {fixed_points}")
    
    #make copy fixed points
    points_to_follow = fixed_points.copy() 
    if on_right == 0:
        points_to_follow = points_to_follow[0:7]
        on_right = 1
    elif on_right == 1:
        points_to_follow = points_to_follow[7:14]
        on_right = 0
    elif on_right == 2:
        points_to_follow == points_to_follow[0:14]
        
    for point in points_to_follow:
        print(f"Moving to fixed point: {point}")
 
        #arm.set_position(x=point[0], y=point[1], z=point[2], roll=point[3], pitch=point[4], yaw=point[5], speed=75, wait=True, is_radian=True)
        arm.set_servo_angle(angle=point[0:6],wait=True, speed=0.4, is_radian=True)
        arm.set_gripper_position(point[6], wait=False)
        time.sleep(0.25)
    
    # last_point = fixed_points[-1]
    # last_point[0]-= 200
    # arm.set_position(x=last_point[0], y=last_point[1], z=last_point[2], roll=last_point[3], pitch=last_point[4], yaw=last_point[5], speed=75, wait=True, is_radian=True)
 
    arm.set_mode(1)
    arm.set_state(state=0)


def ensure_safe_goal_position(
    goal_pos: torch.Tensor, present_pos: torch.Tensor, max_relative_target: float | list[float]
):
    # Cap relative action target magnitude for safety.
    diff = goal_pos - present_pos
    max_relative_target = torch.tensor(max_relative_target)
    safe_diff = torch.minimum(diff, max_relative_target)
    safe_diff = torch.maximum(safe_diff, -max_relative_target)
    safe_goal_pos = present_pos + safe_diff

    if not torch.allclose(goal_pos, safe_goal_pos):
        logging.warning(
            "Relative goal position magnitude had to be clamped to be safe.\n"
            f"  requested relative goal position target: {diff}\n"
            f"    clamped relative goal position target: {safe_diff}"
        )

    return safe_goal_pos


class ManipulatorRobot:
    # TODO(rcadene): Implement force feedback
    """This class allows to control any manipulator robot of various number of motors.

    Non exhaustive list of robots:
    - [Koch v1.0](https://github.com/AlexanderKoch-Koch/low_cost_robot), with and without the wrist-to-elbow expansion, developed
    by Alexander Koch from [Tau Robotics](https://tau-robotics.com)
    - [Koch v1.1](https://github.com/jess-moss/koch-v1-1) developed by Jess Moss
    - [Aloha](https://www.trossenrobotics.com/aloha-kits) developed by Trossen Robotics

    Example of instantiation, a pre-defined robot config is required:
    ```python
    robot = ManipulatorRobot(KochRobotConfig())
    ```

    Example of overwriting motors during instantiation:
    ```python
    # Defines how to communicate with the motors of the leader and follower arms
    leader_arms = {
        "main": DynamixelMotorsBusConfig(
            port="/dev/tty.usbmodem575E0031751",
            motors={
                # name: (index, model)
                "shoulder_pan": (1, "xl330-m077"),
                "shoulder_lift": (2, "xl330-m077"),
                "elbow_flex": (3, "xl330-m077"),
                "wrist_flex": (4, "xl330-m077"),
                "wrist_roll": (5, "xl330-m077"),
                "gripper": (6, "xl330-m077"),
            },
        ),
    }
    follower_arms = {
        "main": DynamixelMotorsBusConfig(
            port="/dev/tty.usbmodem575E0032081",
            motors={
                # name: (index, model)
                "shoulder_pan": (1, "xl430-w250"),
                "shoulder_lift": (2, "xl430-w250"),
                "elbow_flex": (3, "xl330-m288"),
                "wrist_flex": (4, "xl330-m288"),
                "wrist_roll": (5, "xl330-m288"),
                "gripper": (6, "xl330-m288"),
            },
        ),
    }
    robot_config = KochRobotConfig(leader_arms=leader_arms, follower_arms=follower_arms)
    robot = ManipulatorRobot(robot_config)
    ```

    Example of overwriting cameras during instantiation:
    ```python
    # Defines how to communicate with 2 cameras connected to the computer.
    # Here, the webcam of the laptop and the phone (connected in USB to the laptop)
    # can be reached respectively using the camera indices 0 and 1. These indices can be
    # arbitrary. See the documentation of `OpenCVCamera` to find your own camera indices.
    cameras = {
        "laptop": OpenCVCamera(camera_index=0, fps=30, width=640, height=480),
        "phone": OpenCVCamera(camera_index=1, fps=30, width=640, height=480),
    }
    robot = ManipulatorRobot(KochRobotConfig(cameras=cameras))
    ```

    Once the robot is instantiated, connect motors buses and cameras if any (Required):
    ```python
    robot.connect()
    ```

    Example of highest frequency teleoperation, which doesn't require cameras:
    ```python
    while True:
        robot.teleop_step()
    ```

    Example of highest frequency data collection from motors and cameras (if any):
    ```python
    while True:
        observation, action = robot.teleop_step(record_data=True)
    ```

    Example of controlling the robot with a policy:
    ```python
    while True:
        # Uses the follower arms and cameras to capture an observation
        observation = robot.capture_observation()

        # Assumes a policy has been instantiated
        with torch.inference_mode():
            action = policy.select_action(observation)

        # Orders the robot to move
        robot.send_action(action)
    ```

    Example of disconnecting which is not mandatory since we disconnect when the object is deleted:
    ```python
    robot.disconnect()
    ```
    """

    def __init__(
        self,
        config: ManipulatorRobotConfig,
    ):
        self.config = config
        self.robot_type = self.config.type
        self.calibration_dir = Path(self.config.calibration_dir)
        print(f"Using calibration directory: {self.calibration_dir}")
        self.leader_arms = make_motors_buses_from_configs(self.config.leader_arms)
        self.follower_arms = make_motors_buses_from_configs(self.config.follower_arms)
        self.cameras = make_cameras_from_configs(self.config.cameras)
        self.is_connected = False
        self.logs = {}
        self.ee_poses = []
        self.ee_poses_lock = threading.Lock()
  

    def get_motor_names(self, arm: dict[str, MotorsBus]) -> list:
        return [f"{arm}_{motor}" for arm, bus in arm.items() for motor in bus.motors]

    @property
    def camera_features(self) -> dict:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            key = f"observation.images.{cam_key}"
            cam_ft[key] = {
                "shape": (cam.height, cam.width, cam.channels),
                "names": ["height", "width", "channels"],
                "info": None,
            }
        return cam_ft

    @property
    def motor_features(self) -> dict:
        action_names = self.get_motor_names(self.leader_arms)
        state_names = self.get_motor_names(self.leader_arms)
        return {
            "action": {
                "dtype": "float32",
                "shape": (len(action_names),),
                "names": action_names,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(state_names),),
                "names": state_names,
            },
        }

    @property
    def features(self):
        return {**self.motor_features, **self.camera_features}

    @property
    def has_camera(self):
        return len(self.cameras) > 0

    @property
    def num_cameras(self):
        return len(self.cameras)

    @property
    def available_arms(self):
        available_arms = []
        for name in self.follower_arms:
            arm_id = get_arm_id(name, "follower")
            available_arms.append(arm_id)
        for name in self.leader_arms:
            arm_id = get_arm_id(name, "leader")
            available_arms.append(arm_id)
        return available_arms

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                "ManipulatorRobot is already connected. Do not run `robot.connect()` twice."
            )

        if not self.leader_arms and not self.follower_arms and not self.cameras:
            raise ValueError(
                "ManipulatorRobot doesn't have any device to connect. See example of usage in docstring of the class."
            )

        # Connect the arms
        #for name in self.follower_arms:
            # print(f"Connecting {name} follower arm.")
            # self.follower_arms[name].connect()
        for name in self.leader_arms:
            print(f"Connecting {name} leader arm.")
            self.leader_arms[name].connect()

        if self.robot_type in ["koch", "koch_bimanual", "aloha"]:
            from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
        elif self.robot_type in ["so100", "moss", "lekiwi"]:
            from lerobot.common.robot_devices.motors.feetech import TorqueMode

        # We assume that at connection time, arms are in a rest position, and torque can
        # be safely disabled to run calibration and/or set robot preset configurations.
        # for name in self.follower_arms:
        #     self.follower_arms[name].write("Torque_Enable", TorqueMode.DISABLED.value)
        # for name in self.leader_arms:
        #     self.leader_arms[name].write("Torque_Enable", TorqueMode.DISABLED.value)

        #self.activate_calibration()

        # Set robot preset (e.g. torque in leader gripper for Koch v1.1)
        if self.robot_type in ["koch", "koch_bimanual"]:
            self.set_koch_robot_preset()
        elif self.robot_type == "aloha":
            self.set_aloha_robot_preset()
        # elif self.robot_type in ["so100", "moss", "lekiwi"]:
        #     self.set_so100_robot_preset()

        # # Enable torque on all motors of the follower arms
        # for name in self.follower_arms:
        #     print(f"Activating torque on {name} follower arm.")
        #     self.follower_arms[name].write("Torque_Enable", 1)

        if self.config.gripper_open_degree is not None:
            if self.robot_type not in ["koch", "koch_bimanual"]:
                raise NotImplementedError(
                    f"{self.robot_type} does not support position AND current control in the handle, which is require to set the gripper open."
                )
            # Set the leader arm in torque mode with the gripper motor set to an angle. This makes it possible
            # to squeeze the gripper and have it spring back to an open position on its own.
            for name in self.leader_arms:
                self.leader_arms[name].write("Torque_Enable", 1, "gripper")
                self.leader_arms[name].write("Goal_Position", self.config.gripper_open_degree, "gripper")

        # Check both arms can be read
        # for name in self.follower_arms:
        #     self.follower_arms[name].read("Present_Position")
        for name in self.leader_arms:
            self.leader_arms[name].read("Present_Position")

        # Connect the cameras
        for name in self.cameras:
            self.cameras[name].connect()

        self.is_connected = True

    def activate_calibration(self):
        """After calibration all motors function in human interpretable ranges.
        Rotations are expressed in degrees in nominal range of [-180, 180],
        and linear motions (like gripper of Aloha) in nominal range of [0, 100].
        """

        def load_or_run_calibration_(name, arm, arm_type):
            arm_id = get_arm_id(name, arm_type)
            arm_calib_path = self.calibration_dir / f"{arm_id}.json"

            if arm_calib_path.exists():
                with open(arm_calib_path) as f:
                    calibration = json.load(f)
            else:
                # TODO(rcadene): display a warning in __init__ if calibration file not available
                print(f"Missing calibration file '{arm_calib_path}'")

                if self.robot_type in ["koch", "koch_bimanual", "aloha"]:
                    from lerobot.common.robot_devices.robots.dynamixel_calibration import run_arm_calibration

                    calibration = run_arm_calibration(arm, self.robot_type, name, arm_type)

                elif self.robot_type in ["so100", "moss", "lekiwi"]:
                    from lerobot.common.robot_devices.robots.feetech_calibration import (
                        run_arm_manual_calibration,
                    )

                    calibration = run_arm_manual_calibration(arm, self.robot_type, name, arm_type)

                print(f"Calibration is done! Saving calibration file '{arm_calib_path}'")
                arm_calib_path.parent.mkdir(parents=True, exist_ok=True)
                with open(arm_calib_path, "w") as f:
                    json.dump(calibration, f)

            return calibration

        for name, arm in self.follower_arms.items():
            calibration = load_or_run_calibration_(name, arm, "follower")
            arm.set_calibration(calibration)
        for name, arm in self.leader_arms.items():
            calibration = load_or_run_calibration_(name, arm, "leader")
            arm.set_calibration(calibration)

    def set_koch_robot_preset(self):
        def set_operating_mode_(arm):
            from lerobot.common.robot_devices.motors.dynamixel import TorqueMode




            if (arm.read("Torque_Enable") != TorqueMode.DISABLED.value).any():
                raise ValueError("To run set robot preset, the torque must be disabled on all motors.")

            # Use 'extended position mode' for all motors except gripper, because in joint mode the servos can't
            # rotate more than 360 degrees (from 0 to 4095) And some mistake can happen while assembling the arm,
            # you could end up with a servo with a position 0 or 4095 at a crucial point See [
            # https://emanual.robotis.com/docs/en/dxl/x/x_series/#operating-mode11]
            all_motors_except_gripper = [name for name in arm.motor_names if name != "gripper"]
            if len(all_motors_except_gripper) > 0:
                # 4 corresponds to Extended Position on Koch motors
                arm.write("Operating_Mode", 4, all_motors_except_gripper)

            # Use 'position control current based' for gripper to be limited by the limit of the current.
            # For the follower gripper, it means it can grasp an object without forcing too much even tho,
            # it's goal position is a complete grasp (both gripper fingers are ordered to join and reach a touch).
            # For the leader gripper, it means we can use it as a physical trigger, since we can force with our finger
            # to make it move, and it will move back to its original target position when we release the force.
            # 5 corresponds to Current Controlled Position on Koch gripper motors "xl330-m077, xl330-m288"
            arm.write("Operating_Mode", 5, "gripper")

        for name in self.follower_arms:
            set_operating_mode_(self.follower_arms[name])

            # Set better PID values to close the gap between recorded states and actions
            # TODO(rcadene): Implement an automatic procedure to set optimal PID values for each motor
            self.follower_arms[name].write("Position_P_Gain", 1500, "elbow_flex")
            self.follower_arms[name].write("Position_I_Gain", 0, "elbow_flex")
            self.follower_arms[name].write("Position_D_Gain", 600, "elbow_flex")

        if self.config.gripper_open_degree is not None:
            for name in self.leader_arms:
                set_operating_mode_(self.leader_arms[name])

                # Enable torque on the gripper of the leader arms, and move it to 45 degrees,
                # so that we can use it as a trigger to close the gripper of the follower arms.
                self.leader_arms[name].write("Torque_Enable", 1, "gripper")
                self.leader_arms[name].write("Goal_Position", self.config.gripper_open_degree, "gripper")

    def set_aloha_robot_preset(self):
        def set_shadow_(arm):
            # Set secondary/shadow ID for shoulder and elbow. These joints have two motors.
            # As a result, if only one of them is required to move to a certain position,
            # the other will follow. This is to avoid breaking the motors.
            if "shoulder_shadow" in arm.motor_names:
                shoulder_idx = arm.read("ID", "shoulder")
                arm.write("Secondary_ID", shoulder_idx, "shoulder_shadow")

            if "elbow_shadow" in arm.motor_names:
                elbow_idx = arm.read("ID", "elbow")
                arm.write("Secondary_ID", elbow_idx, "elbow_shadow")

        for name in self.follower_arms:
            set_shadow_(self.follower_arms[name])

        for name in self.leader_arms:
            set_shadow_(self.leader_arms[name])

        for name in self.follower_arms:
            # Set a velocity limit of 131 as advised by Trossen Robotics
            self.follower_arms[name].write("Velocity_Limit", 131)

            # Use 'extended position mode' for all motors except gripper, because in joint mode the servos can't
            # rotate more than 360 degrees (from 0 to 4095) And some mistake can happen while assembling the arm,
            # you could end up with a servo with a position 0 or 4095 at a crucial point See [
            # https://emanual.robotis.com/docs/en/dxl/x/x_series/#operating-mode11]
            all_motors_except_gripper = [
                name for name in self.follower_arms[name].motor_names if name != "gripper"
            ]
            if len(all_motors_except_gripper) > 0:
                # 4 corresponds to Extended Position on Aloha motors
                self.follower_arms[name].write("Operating_Mode", 4, all_motors_except_gripper)

            # Use 'position control current based' for follower gripper to be limited by the limit of the current.
            # It can grasp an object without forcing too much even tho,
            # it's goal position is a complete grasp (both gripper fingers are ordered to join and reach a touch).
            # 5 corresponds to Current Controlled Position on Aloha gripper follower "xm430-w350"
            self.follower_arms[name].write("Operating_Mode", 5, "gripper")

            # Note: We can't enable torque on the leader gripper since "xc430-w150" doesn't have
            # a Current Controlled Position mode.

        if self.config.gripper_open_degree is not None:
            warnings.warn(
                f"`gripper_open_degree` is set to {self.config.gripper_open_degree}, but None is expected for Aloha instead",
                stacklevel=1,
            )

    def set_so100_robot_preset(self):
        for name in self.follower_arms:
            # Mode=0 for Position Control
            self.follower_arms[name].write("Mode", 0)
            # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
            self.follower_arms[name].write("P_Coefficient", 16)
            # Set I_Coefficient and D_Coefficient to default value 0 and 32
            self.follower_arms[name].write("I_Coefficient", 0)
            self.follower_arms[name].write("D_Coefficient", 32)
            # Close the write lock so that Maximum_Acceleration gets written to EPROM address,
            # which is mandatory for Maximum_Acceleration to take effect after rebooting.
            self.follower_arms[name].write("Lock", 0)
            # Set Maximum_Acceleration to 254 to speedup acceleration and deceleration of
            # the motors. Note: this configuration is not in the official STS3215 Memory Table
            self.follower_arms[name].write("Maximum_Acceleration", 254)
            self.follower_arms[name].write("Acceleration", 254)

    def ee_pos_to_mat(self, ee_pos):
        rot = Rotation.from_euler('xyz', ee_pos[3:6], degrees=False).as_matrix()
        # Gravity in world frame = [0, 0, -1]
        g_world = np.array([0, 0, -1])

        # Express gravity in camera frame
        g_cam = rot.T @ g_world

        # Project gravity onto image plane (image plane normal = camera Z = [0, 0, 1])
        g_proj = g_cam.copy()
        g_proj[2] = 0  # remove Z component

        norm = np.linalg.norm(g_proj)
        if norm < 1e-6:
            print("Warning: Projected gravity vector is too small, returning 0.0 degrees.")
            return 0.0  # Camera is aligned with gravity; image orientation undefined

        g_proj /= norm

        # Camera "up" in camera frame is +Y
        image_up = np.array([1, 0, 0])

        # Compute signed angle between image_up and projected gravity
        angle_rad = np.arctan2(
            g_proj[0] * image_up[1] - g_proj[1] * image_up[0],
            g_proj[0] * image_up[0] + g_proj[1] * image_up[1]
        )

        return (180-np.degrees(angle_rad), rot)


        # Try follow commands
        #return angle_deg
    def capture_images(self):
        while True:
            images = {}
            curr_ee_poses = []
            with self.ee_poses_lock:
                curr_ee_poses = self.ee_poses.copy()
                # print("curr_ee_poses", curr_ee_poses, self.ee_poses)
            # for name in self.cameras:
            #     yaw_correction = 0
            #     curr_rot = np.eye(3)
            #     if "left" in name:
            #         yaw_correction, curr_rot = self.ee_pos_to_mat(curr_ee_poses[0])
            #     elif "right" in name:
            #         yaw_correction, curr_rot = self.ee_pos_to_mat(curr_ee_poses[1])
            #     before_camread_t = time.perf_counter()
                
            #     #print(f"Reading camera {name} with yaw correction: {yaw_correction}")
            #     if ("left" in name or "right" in name):
            #         images[name] = self.cameras[name].async_read(curr_rot, yaw_correction = yaw_correction)
            #     else:
            #         images[name] = self.cameras[name].async_read()
            #     images[name] = torch.from_numpy(images[name])
            #     self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            #     self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t
                
            for i in range(10):
                try:
                    for name in self.cameras:
                        yaw_correction = 0
                        curr_rot = np.eye(3)
                        if "left" in name:
                            yaw_correction, curr_rot = self.ee_pos_to_mat(curr_ee_poses[0])
                        elif "right" in name:
                            yaw_correction, curr_rot = self.ee_pos_to_mat(curr_ee_poses[1])
                        before_camread_t = time.perf_counter()
                        
                        #print(f"Reading camera {name} with yaw correction: {yaw_correction}")
                        if ("left" in name or "right" in name):
                            images[name] = self.cameras[name].async_read(curr_rot, yaw_correction = yaw_correction)
                        else:
                            images[name] = self.cameras[name].async_read()
                        images[name] = torch.from_numpy(images[name])
                        self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
                        self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t
                    break
                except Exception as e:
                    # restart the camera if it fails to read a frame
                    logging.warning(
                        f"IntelRealSenseCamera({name}) failed to read a frame. Restarting the camera."
                    )
                    for name in self.cameras:
                        self.cameras[name].disconnect()
                    self.cameras = make_cameras_from_configs(self.config.cameras)
                    for name in self.cameras:
                        self.cameras[name].connect()
            image_keys = [key for key in images]
            for key in image_keys:
                rr.log(key, rr.Image(images[key].numpy()), static=True)

            time.sleep(1.0 / 35.0)
            
    def teleop_step(
        self, record_data=False, record_wps=False, play_wps=False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )
        
        global set_init
        if play_wps and record_wps:
            print("Playing waypoints, skipping initial position setting.")
            cable_dance()
        elif not set_init and not record_wps:
            print(f"Recording data: {record_data}, recording waypoints: {record_wps}, playing waypoints: {play_wps}")
            # If not recording or playing waypoints, we can set the initial positions of the arms
            set_to_initial_positions()
        


       
        # Prepare to assign the position of the leader to the follower
        start_time = time.perf_counter()
        leader_pos = {}
        names = []
        for name in self.leader_arms:
            before_lread_t = time.perf_counter()
            leader_pos[name] = self.leader_arms[name].read("Present_Position")
            leader_pos[name] = torch.from_numpy(leader_pos[name])
            leader_read_time = time.perf_counter() - before_lread_t
            self.logs[f"read_leader_{name}_pos_dt_s"] = leader_read_time
            #print(f"Reading leader {name} position took: {leader_read_time:.4f} seconds")
            names.append(name)
        
        leader_read_total = time.perf_counter() - start_time
        #print(f"Total leader read time: {leader_read_total:.4f} seconds")

        state = [torch.zeros(7, dtype=torch.float32) for _ in range(2)]  # Assuming two arms
        action = [torch.zeros(7, dtype=torch.float32) for _ in range(2)]  # Assuming two arms
        ee_pos = [np.zeros(6, dtype=np.float32) for _ in range(2)]  # Assuming two arms
        curr_pos_saved = [np.zeros(6, dtype=np.float32) for _ in range(2)]  # Assuming two arms

        # Define the worker function that processes a single arm
        def process_arm_in_thread(i: int):
            arm_start = time.perf_counter()
            arm = arms[i]
            name = names[i]
        
            # Get current position
            get_pos_start = time.perf_counter()
            curr_pos = np.array(arm.get_servo_angle(is_radian=True)[1]).astype(np.float32)
            ee_pos[i] = np.array(arm.get_position(is_radian=True)[1]).astype(np.float32)
            curr_pos_saved[i] = curr_pos
            if record_wps:
                return
           

            get_pos_time = time.perf_counter() - get_pos_start
            #print(f"Getting {name} current position took: {get_pos_time:.4f} seconds")
            
            # Process goal position
            goal_pos = leader_pos[name]
            goal_pos = goal_pos.numpy().astype(np.float32)

            # Calculate error and decide movement strategy
            error_degs = np.abs(goal_pos[0:6]-curr_pos[0:6])/np.pi*180
            max_error_degs = np.max(error_degs)
            max_error_idx = np.argmax(error_degs)
            
            global first
            global active_gripper_threads # Access the global list
            global redo_first
            move_start = time.perf_counter()
            if max_error_degs < 30 or not first[i] or (redo_first):

                if redo_first:
                    print(f"Redoing first move for {name} arm due to cable dance.")
                if i == 1:
                    redo_first = False
                if first[i]:
                    while True:
                        try:
                            arms[i].disconnect()
                            arms[i] = XArmAPI(x_arm_ips[i])  # Reinitialize arm to reset state
                            arms[i].disconnect()
                            arms[i] = XArmAPI(x_arm_ips[i])  # Reinitialize arm to reset state
                            arm = arms[i]
                            arm.motion_enable(enable=True)
                            arm.set_mode(1)
                            arm.set_state(state=0)
                            arm.set_mode(0)
                            arm.set_state(state=0)
                            print(f"Setting for {name} arm due to cable dance.")
                            while (True):
                                ret = arm.set_servo_angle(angle=goal_pos.tolist(), speed=0.75, wait=True, is_radian=True)
                                new_pos = np.array(arm.get_servo_angle(is_radian=True)[1]).astype(np.float32)
                                error = np.max(np.abs(goal_pos[0:6]-new_pos[0:6])/np.pi*180)
                                if ret == 0 and error < 0.1:
                                    break
                                print(f"{name} arm is not moving to init pos!!!")
                            arm.set_mode(1)
                            arm.set_state(state=0)
                            first[i] = False
                            #arm.set_collision_sensitivity(3)
                            break
                        except Exception as e:
                            print("An error occured")
                            continue


                        
                    

 


                else:
                    if play_wps:
                        ret, pose = arm.get_forward_kinematics(goal_pos.tolist(), input_is_radian=True, return_is_radian=True)
                        error_with_first_wp = (np.abs(curr_pos[0:6] - np.array(fixed_points[0][0:6]))/np.pi*180).max()
                        #print("Error with first waypoint:", error_with_first_wp)
                        if pose[0] > 600 and error_with_first_wp < 20 and i == 1:
                            #kill gripper threads
                            for k in range(2):
                                if active_gripper_threads[k] is not None:
                                    print(f"Killing active gripper thread for {name} arm (idx {k}) before cable dance.")
                                    active_gripper_threads[k].join()
                                    active_gripper_threads[k] = None  # Clear the thread reference
                            cable_dance()
                            redo_first = True
                            first[i] = True
                            first[0] = True
                        else:
                            arm.set_servo_angle_j(goal_pos.tolist(), is_radian=True)
                    else:
                        if i != 1 and i != 0:
                            ret, pose = arm.get_forward_kinematics(goal_pos.tolist(), input_is_radian=True, return_is_radian=True)
                            
                            # if pose[0] > 650:
                            #     # pose[3] = -np.pi
                            #     # pose[4] = -np.pi/2
                            #     # pose[5] = 0
                            #     pose[3] = -np.pi/2
                            #     pose[4] = 0
                            #     pose[5] = -np.pi/2
                            #print(f"Forward kinematics for {name} arm: {pose}")
                            arm.set_servo_cartesian(pose, is_radian=True)
                        else:
                            arm.set_servo_angle_j(goal_pos.tolist(), is_radian=True)
            else:
                print(f"Error too high for {name} arm, max error: {max_error_degs} degrees at index {max_error_idx}")
                move_time = time.perf_counter() - move_start
                #print(f"Moving {name} arm took: {move_time:.4f} seconds")
            
            # Set gripper
            
            gripper_start = time.perf_counter()
            goal_pos[6] = max(0,1-goal_pos[6])
            goal_pos[6] = max(goal_pos[6]*1000,10)

            if active_gripper_threads[i] is not None and active_gripper_threads[i].is_alive():
                print(f"Skipping new gripper command for {name} arm (idx {i}); previous command's thread still active.")
            else:
                # print(f"Attempting to launch new gripper command for {name} arm (idx {i}) with value {transformed_gripper_val_for_command}.")
                gripper_command_thread = threading.Thread(
                    target=arm.set_gripper_position,
                    args=(goal_pos[6],),
                    kwargs={'wait': False} # Assuming SDK expects this, even if it blocks
                )
                try:
                    gripper_command_thread.start()
                    active_gripper_threads[i] = gripper_command_thread # Store the new thread
                    # print(f"Successfully launched gripper command thread for {name} arm (idx {i}).")
                except Exception as e:
                    #print(f"Error starting gripper command thread for {name} arm (idx {i}): {e}")
                    active_gripper_threads[i] = None # Ensure it's cleared on error

            #arm.set_gripper_position(goal_pos[6], wait=False)
            gripper_time = time.perf_counter() - gripper_start
            #print(f"Setting {name} gripper took: {gripper_time:.4f} seconds")
            
            # Convert and store
            curr_pos[0:6] = curr_pos[0:6]/np.pi*180
            goal_pos[0:6] = goal_pos[0:6]/np.pi*180
            state[i] = torch.from_numpy(curr_pos)
            action[i] = torch.from_numpy(goal_pos)
            
            arm_total = time.perf_counter() - arm_start
            #print(f"Total {name} arm processing took: {arm_total:.4f} seconds")


        process_arm_in_thread(0)
        process_arm_in_thread(1)
        
        with self.ee_poses_lock:
            self.ee_poses = ee_pos
         # Inside the function:
        # Check for spacebar press and save waypoints
        global space_pressed
        if space_pressed:
            # Append current waypoint to a file
     
            waypoints_file = "/home/hans/curr_wps.txt"
            # Check if the file exists, if not create it
            if record_wps:
                with open(waypoints_file, "a") as f:
                    f.write(f"{curr_pos_saved[1].tolist()}\n") # Only save the right arm waypoint
                print(f"Waypoint saved to {waypoints_file}")
            else:   
                print(f"Waypoint not saved, recording waypoints is disabled.")
            # Reset to avoid multiple prints from one press
            space_pressed = False
           

        state = torch.cat(state)
        action = torch.cat(action)

        total_time = time.perf_counter() - start_time
        #print(f"Total teleop_step execution time: {total_time:.4f} seconds")
        #print("-" * 50)

        # Early exit when recording data is not requested
        if not record_data:
            return
 
    
        # just take j4 and j6 
        # Capture images from cameras
        global camera_thread
        
        if camera_thread == None:
            print("Starting camera thread")
            camera_thread = threading.Thread(target=self.capture_images, args=())
            camera_thread.start()

        #self.capture_images(ee_pos)
        
                  

        #print("Images captured from cameras:", images.keys())
        # Populate output dictionaries
        obs_dict, action_dict = {}, {}
        obs_dict["observation.state"] = state
        action_dict["action"] = action
        # for name in self.cameras:
        #     obs_dict[f"observation.images.{name}"] = images[name]

        return obs_dict, action_dict

    def capture_observation(self):
        """The returned observations do not have a batch dimension."""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )

        # Read follower position
        follower_pos = {}
        for name in self.follower_arms:
            i = 0 if name == "left" else 1
            arm = arms[i]

            before_fread_t = time.perf_counter()
            #print(f"Reading follower {name} position.")

            pos = np.zeros(7, dtype=np.float32)
            #print(arm.get_servo_angle(is_radian=False)[1])
            pos = np.array(arm.get_servo_angle(is_radian=False)[1]).astype(np.float32)

            #follower_pos[name] = self.follower_arms[name].read("Present_Position")
            follower_pos[name] = torch.from_numpy(pos)
            self.logs[f"read_follower_{name}_pos_dt_s"] = time.perf_counter() - before_fread_t

        # Create state by concatenating follower current position
        state = []
        for name in self.follower_arms:
            if name in follower_pos:
                state.append(follower_pos[name])
        state = torch.cat(state)

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Populate output dictionaries and format to pytorch
        obs_dict = {}
        obs_dict["observation.state"] = state
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]
        return obs_dict

    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        """Command the follower arms to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Args:
            action: tensor containing the concatenated goal positions for the follower arms.
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )
        
        set_to_initial_positions()

        from_idx = 0
        to_idx = 0
        action_sent = []
        for name in self.follower_arms:
            # Get goal position of each follower arm by splitting the action vector
            to_idx += len(self.follower_arms[name].motor_names)
            goal_pos = action[from_idx:to_idx]
            from_idx = to_idx

            if name == "left":
                arm = arms[0]
            elif name == "right":
                arm = arms[1]

            # Cap goal position when too far away from present position.
            # Slower fps expected due to reading from the follower.
            if self.config.max_relative_target is not None:
                present_pos = self.follower_arms[name].read("Present_Position")
                present_pos = torch.from_numpy(present_pos)
                goal_pos = ensure_safe_goal_position(goal_pos, present_pos, self.config.max_relative_target)

            # Save tensor to concat and return
            action_sent.append(goal_pos)

            # Send goal position to each follower
            goal_pos = goal_pos.numpy().astype(np.float32)
            arm.set_servo_angle_j(goal_pos[0:6])
            arm.set_gripper_position(goal_pos[6], wait=True)
            time.sleep(3)  # Small delay to ensure the command is sent

        return torch.cat(action_sent)

    def print_logs(self):
        pass
        # TODO(aliberts): move robot-specific logs logic here

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()` before disconnecting."
            )

        # for name in self.follower_arms:
        #     self.follower_arms[name].disconnect()

        for name in self.leader_arms:
            self.leader_arms[name].disconnect()

        for name in self.cameras:
            self.cameras[name].disconnect()

        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
