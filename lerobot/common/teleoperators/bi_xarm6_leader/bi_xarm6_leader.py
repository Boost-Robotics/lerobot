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

from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.common.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.common.motors.dynamixel import (
    DriveMode,
    DynamixelMotorsBus,
    OperatingMode,
)

from ..teleoperator import Teleoperator
from .config_bi_xarm6_leader import BiXarm6LeaderConfig
import threading

logger = logging.getLogger(__name__)


class BiXarm6Leader(Teleoperator):

    config_class = BiXarm6LeaderConfig
    name = "bi_xarm6_leader"

    def __init__(self, config: BiXarm6LeaderConfig):
        super().__init__(config)
        self.config = config
 

        # Create left arm bus
        self.left_bus = DynamixelMotorsBus(
            port=self.config.left_port,
            motors={
                "joint1": Motor(1, "xl330-m288", MotorNormMode.DEGREES),
                "joint2": Motor(2, "xl330-m288", MotorNormMode.DEGREES),
                "joint3": Motor(3, "xl330-m288", MotorNormMode.DEGREES),
                "joint4": Motor(4, "xl330-m288", MotorNormMode.DEGREES),
                "joint5": Motor(5, "xl330-m288", MotorNormMode.DEGREES),
                "joint6": Motor(6, "xl330-m288", MotorNormMode.DEGREES),
                "gripper": Motor(7, "xl330-m077", MotorNormMode.RANGE_0_100),
            },
            calibration=self._get_left_calibration(),
        )

        # Create right arm bus
        self.right_bus = DynamixelMotorsBus(
            port=self.config.right_port,
            motors={
                "joint1": Motor(1, "xl330-m288", MotorNormMode.DEGREES),
                "joint2": Motor(2, "xl330-m288", MotorNormMode.DEGREES),
                "joint3": Motor(3, "xl330-m288", MotorNormMode.DEGREES),
                "joint4": Motor(4, "xl330-m288", MotorNormMode.DEGREES),
                "joint5": Motor(5, "xl330-m288", MotorNormMode.DEGREES),
                "joint6": Motor(6, "xl330-m288", MotorNormMode.DEGREES),
                "gripper": Motor(7, "xl330-m077", MotorNormMode.RANGE_0_100),
            },
            calibration=self._get_right_calibration(),
        )

    def _get_left_calibration(self):
        """Load calibration for left arm from existing xarm6_leader calibration"""
        from pathlib import Path

        import draccus

        left_calibration_path = (
            Path("/home/hans/.cache/huggingface/lerobot/calibration/teleoperators/xarm6_leader")
            / f"{self.config.left_id}.json"
        )
        if left_calibration_path.exists():
            try:
                with open(left_calibration_path) as f, draccus.config_type("json"):
                    return draccus.load(dict[str, MotorCalibration], f)
            except Exception as e:
                logger.warning(f"Failed to load left arm calibration: {e}")
        return None

    def _get_right_calibration(self):
        """Load calibration for right arm from existing xarm6_leader calibration"""
        from pathlib import Path

        import draccus


        right_calibration_path = (
            Path("/home/hans/.cache/huggingface/lerobot/calibration/teleoperators/xarm6_leader")
            / f"{self.config.right_id}.json"
        )
        if right_calibration_path.exists():
            try:
                with open(right_calibration_path) as f, draccus.config_type("json"):
                    return draccus.load(dict[str, MotorCalibration], f)
            except Exception as e:
                logger.warning(f"Failed to load right arm calibration: {e}")
        return None
    
    @property
    def action_features(self) -> dict[str, type]:
        action_ft = {}
        for motor in self.left_bus.motors:
            action_ft[f"left_{motor}.pos"] = float
        for motor in self.right_bus.motors:
            action_ft[f"right_{motor}.pos"] = float
        return action_ft

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.left_bus.is_connected and self.right_bus.is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # Connect both buses
        self.left_bus.connect()
        self.right_bus.connect()

        if not self.is_calibrated and calibrate:
            self.calibrate()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        left_calibrated = self.left_bus.is_calibrated and self._get_left_calibration() is not None
        right_calibrated = self.right_bus.is_calibrated and self._get_right_calibration() is not None
        return left_calibrated and right_calibrated

    def calibrate(self) -> None:
        logger.info(f"\nLoading existing calibrations for {self}")

        # Load calibrations from existing files
        left_calibration = self._get_left_calibration()
        right_calibration = self._get_right_calibration()

        if left_calibration is None:
            raise ValueError(
                f"No calibration found for left arm (ID: {self.config.left_id}). "
                "Please ensure calibration exists at /home/*/.cache/huggingface/lerobot/calibration/teleoperators/xarm6_leader/"
            )

        if right_calibration is None:
            raise ValueError(
                f"No calibration found for right arm (ID: {self.config.right_id}). "
                "Please ensure calibration exists at /home/*/.cache/huggingface/lerobot/calibration/teleoperators/xarm6_leader/"
            )

        # Write calibrations to both buses
        self.left_bus.write_calibration(left_calibration)
        self.right_bus.write_calibration(right_calibration)

        logger.info("Successfully loaded calibrations for both arms")

    def configure(self) -> None:
        self.left_bus.disable_torque()
        self.left_bus.configure_motors()
        for motor in self.left_bus.motors:
            self.left_bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)

        self.right_bus.disable_torque()
        self.right_bus.configure_motors()
        for motor in self.right_bus.motors:
            self.right_bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)

    def setup_motors(self) -> None:
        print("Setting up left arm motors:")
        for motor in reversed(self.left_bus.motors):
            input(f"Connect the controller board to the left '{motor}' motor only and press enter.")
            self.left_bus.setup_motor(motor)
            print(f"Left '{motor}' motor id set to {self.left_bus.motors[motor].id}")

        print("Setting up right arm motors:")
        for motor in reversed(self.right_bus.motors):
            input(f"Connect the controller board to the right '{motor}' motor only and press enter.")
            self.right_bus.setup_motor(motor)
            print(f"Right '{motor}' motor id set to {self.right_bus.motors[motor].id}")


    def get_action(self) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()

        # Read from both arms
        # Create threads for reading from both arms in parallel
        left_action_result = [None]
        right_action_result = [None]
        
        def read_left_arm():
            left_action_result[0] = self.left_bus.sync_read("Present_Position")
            
        def read_right_arm():
            right_action_result[0] = self.right_bus.sync_read("Present_Position")
        
        # Start both threads
        left_thread = threading.Thread(target=read_left_arm)
        right_thread = threading.Thread(target=read_right_arm)
        left_thread.start()
        right_thread.start()
        
        # Wait for both threads to complete
        left_thread.join()
        right_thread.join()
        
        # Get results
        left_action = left_action_result[0]
        right_action = right_action_result[0]
 
        #print(f"left action: {left_action}")
        # Combine actions with prefixes
        action = {}
        for motor, val in left_action.items():
            action[f"left_{motor}.pos"] = val
            if motor == "joint1":
                if action[f"left_{motor}.pos"] > 377579200:
                    action[f"left_{motor}.pos"] -= 377579200
            if motor == "joint3":
                action[f"left_{motor}.pos"] = -action[f"left_{motor}.pos"] - 75   
            if motor == "joint5":
                action[f"left_{motor}.pos"] += 90
            if motor == "joint6":
                action[f"left_{motor}.pos"] -= 180
            if motor == "gripper":
                action[f"left_{motor}.pos"]*=8
        for motor, val in right_action.items():
            action[f"right_{motor}.pos"] = val
            if motor == "joint3":
                action[f"right_{motor}.pos"] = -action[f"right_{motor}.pos"] - 75  
            if motor == "joint5":
                action[f"right_{motor}.pos"] += 90
            if motor == "joint6":
                action[f"right_{motor}.pos"] -= 180
            if motor == "gripper":
                action[f"right_{motor}.pos"]*=8

        dt_ms = (time.perf_counter() - start) * 1e3
        #print(f"{self} read action: {dt_ms:.1f}ms")
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.left_bus.disconnect()
        self.right_bus.disconnect()
        logger.info(f"{self} disconnected.")
