#!/usr/bin/env python

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

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.common.cameras.utils import make_cameras_from_configs
from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from xarm.wrapper import XArmAPI
from lerobot.common.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.common.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.common.cameras.configs import ColorMode, Cv2Rotation

from ..robot import Robot
from .config_bi_xarm6_follower import BiXarm6FollowerConfig
import copy
import threading

logger = logging.getLogger(__name__)


class BiXarm6Follower(Robot):
    """
    Bimanual Xarm6 Follower Arms - manages two Xarm6 arms for bimanual tasks
    """

    config_class = BiXarm6FollowerConfig
    name = "bi_xarm6_follower"

    def __init__(self, config: BiXarm6FollowerConfig):
        super().__init__(config)
        self.config = config
        self._is_connected = False  
        self._arms = []
        self.config.cameras = {"ego": RealSenseCameraConfig(serial_number_or_name="137222072104", obs_key="ego", width=640, 
                    height=480, fps=30, color_mode=ColorMode.RGB, use_depth=False, rotation=Cv2Rotation.NO_ROTATION),
                    "left_wrist": RealSenseCameraConfig(serial_number_or_name="137322070266", obs_key="left_wrist", width=640, 
                    height=480, fps=30, color_mode=ColorMode.RGB, use_depth=False, rotation=Cv2Rotation.NO_ROTATION),
                    "right_wrist": RealSenseCameraConfig(serial_number_or_name="819112071093", obs_key="right_wrist", width=640, 
                    height=480, fps=30, color_mode=ColorMode.RGB, use_depth=False, rotation=Cv2Rotation.NO_ROTATION)}
        # Create cameras from the configuration
        self.cameras = make_cameras_from_configs(self.config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Return mapping of motor feature names to their Python types (float)."""
        motors = {f"left_joint{i}.pos": float for i in range(1, 7)}
        motors["left_gripper.pos"] = float
        motors.update({f"right_joint{i}.pos": float for i in range(1, 7)})
        motors["right_gripper.pos"] = float
        return motors

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect_arm(self, arm):
        arm.connect()
        arm.motion_enable(enable=True)
        arm.set_mode(1)  # Position mode
        arm.set_state(state=0)  # Sport state
        arm.set_gripper_mode(0)
        arm.set_gripper_enable(True)
        arm.set_gripper_speed(5000)
        
    def connect(self, calibrate: bool = True) -> None:
        """
        We assume that at connection time, both arms are in rest position,
        and torque can be safely disabled to run calibration.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # Connect both buses
        left_arm = XArmAPI(self.config.left_ip, is_radian=True)
        right_arm = XArmAPI(self.config.right_ip, is_radian=True)
        self._arms = [left_arm, right_arm]
        for enum_idx, arm in enumerate(self._arms):
            self.connect_arm(arm)

            # set joint positions to jacobi, read from arm
            code, joint_positions = self._arms[enum_idx].get_servo_angle()
            if code != 0:
                name = "left" if enum_idx == 0 else "right"
                raise DeviceNotConnectedError(f"Failed to get joint angles from {self}, arm: {name}")

        
        

        if not self.is_calibrated and calibrate:
            self.calibrate()

        # Connect cameras
        for cam in self.cameras.values():
            for i in range(10):
                try:
                    cam.connect()
                    break 
                except Exception as e:
                    logger.warning(f"Failed to connect camera {cam}: {e}, iteration {i+1}/10")
                    # If camera connection fails, we can still proceed with the robot connection
                    cam.disconnect()
                    if i == 9:
                        raise DeviceNotConnectedError(f"Failed to connect camera {cam} after 10 attempts.")
            
           
        self._is_connected = True
        self.configure()
        logger.info(f"{self} connected.")

    @property
    # TODO (hkumar): Implement configuration for the robot if needed
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:  # TODO (hkumar): Implement configuration for the robot if needed
        pass

    def configure(self) -> None:  # TODO (hkumar): Implement configuration for the robot if needed
        pass        
    
    def get_left_angles(self, left_result, left_gripper_result):
        left_result[0], left_result[1] = self._arms[0].get_servo_angle(is_radian=False)
        left_gripper_result[0], left_gripper_result[1] = self._arms[0].get_gripper_position()
            
    def get_right_angles(self, right_result, right_gripper_result):
        right_result[0], right_result[1] = self._arms[1].get_servo_angle(is_radian=False)
        right_gripper_result[0], right_gripper_result[1] = self._arms[1].get_gripper_position()
            
    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm positions
        start = time.perf_counter()
        # Run get_servo_angle commands in parallel using threads
        left_result = [None, None]  # Store [code, joints]
        right_result = [None, None]  # Store [code, joints]
        left_gripper_result = [None, None]  # [code, position]
        right_gripper_result = [None, None]  # [code, position]
        
        left_thread = threading.Thread(target=self.get_left_angles, args=(left_result, left_gripper_result))
        right_thread = threading.Thread(target=self.get_right_angles, args=(right_result, right_gripper_result))

        left_thread.start()
        right_thread.start()
        
        left_thread.join()
        right_thread.join()
        
        code_left, joints_left = left_result
        code_right, joints_right = right_result
        if code_left != 0 or code_right != 0:
            raise DeviceNotConnectedError(f"Failed to get joint angles from {self}")
        if left_gripper_result[0] != 0 or right_gripper_result[0] != 0:
            raise DeviceNotConnectedError(f"{self} gripper is not connected.")
   
        # Combine observations with prefixes
        obs_dict = {}
        for i, angle in enumerate(joints_left[:6]):  # First 6 angles are joints
            obs_dict[f"left_joint{i+1}.pos"] = angle
        obs_dict["left_gripper.pos"] =  left_gripper_result[1]  # Gripper position
        for i, angle in enumerate(joints_right[:6]):  # First 6 angles are joints
            obs_dict[f"right_joint{i+1}.pos"] = angle 
        obs_dict["right_gripper.pos"] = right_gripper_result[1]  # Gripper position     

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")
        #print(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")
        dt_ms = (time.perf_counter() - start) * 1e3
        #print(f"{self} read cameras: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Command both arms to move to target joint configurations.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            the action sent to the motors, potentially clipped.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()

        # Get Left arm actions
        left_goal_pos = []
        for i in range(1, 7):
            name = f"left_joint{i}.pos"
            if name not in action:
                raise ValueError(f"Action missing required key: {name}")
            left_goal_pos.append(action[name])
        if "left_gripper.pos" in action:
            left_goal_pos.append(action["left_gripper.pos"])
        else:
            raise ValueError("Action missing required key: left_gripper.pos")
        
        # Right arm
        right_goal_pos = []
        for i in range(1, 7):
            name = f"right_joint{i}.pos"
            if name not in action:
                raise ValueError(f"Action missing required key: {name}")
            right_goal_pos.append(action[name])
        if "right_gripper.pos" in action:
            right_goal_pos.append(action["right_gripper.pos"])
        else:
            raise ValueError("Action missing required key: right_gripper.pos")

        dt_ms = (time.perf_counter() - start) * 1e3
        #print(f"{self} send action: {dt_ms:.1f}ms")
     
        #TODO:(hkumar): Do some safety checks on the goal positions
        # Use threads to set servo angles and gripper positions in parallel
        def set_left_arm():
            ret = self._arms[0].set_servo_angle_j(left_goal_pos[0:6], is_radian=False)
            if ret != 0:
                logger.error(f"Failed to set joint angles for left arm, doing restart: {ret}")
                self._arms[0].disconnect()
                self.connect_arm(self._arms[0])
                time.sleep(1)
            self._arms[0].set_gripper_position(left_goal_pos[6], wait=False)

        def set_right_arm():
            ret = self._arms[1].set_servo_angle_j(right_goal_pos[0:6], is_radian=False)
            if ret != 0:
                logger.error(f"Failed to set joint angles for right arm, doing restart: {ret}")
                self._arms[1].disconnect()
                self.connect_arm(self._arms[1])
                time.sleep(1)
            self._arms[1].set_gripper_position(right_goal_pos[6], wait=False)

        # Start threads for parallel execution
        left_thread = threading.Thread(target=set_left_arm)
        right_thread = threading.Thread(target=set_right_arm)

        left_thread.start()
        right_thread.start()

        # Wait for both actions to complete
        left_thread.join()
        right_thread.join()

        dt_ms = (time.perf_counter() - start) * 1e3
        #print(f"{self} send action to arms: {dt_ms:.1f}ms")

        # Return the action that was actually sent
        sent_action = copy.deepcopy(action)
        return sent_action

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        for arm in self._arms:
            arm.disconnect()
            self._arms = []
 
        for cam in self.cameras.values():
            cam.disconnect()

        self._is_connected = False
        logger.info(f"{self} disconnected.")

    def reset_to_rest_position(self):
        """Move both arms to their home position."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        first_angle_left = [90, -30, 0, 5, 55, -180]
        first_angle_right = [-50, -55, 0, 5, 55, -180]
        first_angles = [first_angle_left, first_angle_right]
        for i in range(2):
            arm = self._arms[i]
            arm.set_mode(0)
            arm.set_state(state=0)
            ret = arm.set_servo_angle(angle=first_angles[i], speed=50, wait=True, is_radian=False)
            if ret != 0:
                logger.error(f"Failed to set home position for arm {i+1}, with error: {ret}")
            arm.set_gripper_position(800, wait=True)
            arm.set_mode(1)
            arm.set_state(state=0)
            