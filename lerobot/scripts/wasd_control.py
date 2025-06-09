import json
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml

from lerobot.common.robot_devices.cameras.utils import make_cameras_from_configs
from lerobot.common.robot_devices.motors.utils import MotorsBus, make_motors_buses_from_configs
from lerobot.common.robot_devices.robots.configs import ManipulatorRobotConfig
from lerobot.common.robot_devices.robots.utils import get_arm_id
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError
from lerobot.configs import parser

from pprint import pformat
from dataclasses import asdict
from lerobot.common.utils.utils import has_method, init_logging, log_say

from lerobot.common.robot_devices.control_configs import (
    CalibrateControlConfig,
    ControlPipelineConfig,
    RecordControlConfig,
    ReplayControlConfig,
    TeleoperateControlConfig,
)
import numpy as np
import genesis as gs
import cv2
from PIL import Image

from fastsam import FastSAM, FastSAMPrompt 

import argparse
from fastsam import FastSAM, FastSAMPrompt 
import ast
import torch
from PIL import Image
from utils.tools import convert_box_xywh_to_xyxy
import clip

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

from transformers import pipeline

import apriltag

LINE_LENGTH = 5
CENTER_COLOR = (0, 255, 0)
CORNER_COLOR = (255, 0, 255)

### Some utility functions to simplify drawing on the camera feed
# draw a crosshair
def plotPoint(image, center, color):
    center = (int(center[0]), int(center[1]))
    image = cv2.line(image,
                     (center[0] - LINE_LENGTH, center[1]),
                     (center[0] + LINE_LENGTH, center[1]),
                     color,
                     3)
    image = cv2.line(image,
                     (center[0], center[1] - LINE_LENGTH),
                     (center[0], center[1] + LINE_LENGTH),
                     color,
                     3)
    return image

# plot a little text
def plotText(image, center, color, text):
    center = (int(center[0]) + 4, int(center[1]) - 4)
    return cv2.putText(image, str(text), center, cv2.FONT_HERSHEY_SIMPLEX,
                       1, color, 3)
options = apriltag.DetectorOptions(families="tag16h5")
detector = apriltag.Detector(options)


def plotApril(image, camera_matrix=None, dist_coeffs=None, tag_size=0.05):
    """
    Detect AprilTags in the image and calculate their 3D pose with fisheye camera calibration.
    
    Args:
        image: Input image
        camera_matrix: 3x3 camera intrinsic matrix
        dist_coeffs: Distortion coefficients for fisheye model
        tag_size: Size of the AprilTag in meters (default 5cm)
        
    Returns:
        image: Image with AprilTag detections drawn
        poses: Dictionary mapping tag_id to (rvec, tvec) pose information
    """
    grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # look for tags
    detections = detector.detect(grayimg)
    poses = {}
    
    if not detections:
        print("No AprilTags detected")
    else:
        # found some tags, report them and update the camera image
        for detect in detections:
            tag_id = detect.tag_id
            print(f"tag_id: {tag_id}, center: {detect.center}")

            if (tag_id != 0):
                continue
            image = plotPoint(image, detect.center, CENTER_COLOR)
            image = plotText(image, detect.center, CENTER_COLOR, tag_id)
            
            for corner in detect.corners:
                image = plotPoint(image, corner, CORNER_COLOR)
            
            # Calculate 3D pose if camera parameters are provided
            if camera_matrix is not None and dist_coeffs is not None:
                try:
                    # Define 3D coordinates of tag corners in tag coordinate system
                    half_size = tag_size / 2
                    tag_corners_3d = np.array([
                        [-half_size, -half_size, 0],  # Bottom-left
                        [half_size, -half_size, 0],   # Bottom-right
                        [half_size, half_size, 0],    # Top-right
                        [-half_size, half_size, 0]    # Top-left
                    ], dtype=np.float32)
                    
                    # Get detected tag corners in image coordinates
                    tag_corners_2d = np.array(detect.corners, dtype=np.float32).reshape(-1, 1, 2)
                    
                    # Undistort the corner points for fisheye model
                    tag_corners_2d_undistorted = cv2.fisheye.undistortPoints(
                        tag_corners_2d, 
                        camera_matrix, 
                        dist_coeffs, 
                        None, 
                        camera_matrix
                    )
                    
                    # Use solvePnP with undistorted points
                    success, rvec, tvec = cv2.solvePnP(
                        tag_corners_3d, 
                        tag_corners_2d_undistorted.reshape(-1, 2), 
                        camera_matrix, 
                        None,  # No distortion since points are already undistorted
                        flags=cv2.SOLVEPNP_ITERATIVE
                    )
                        
                    if success:
                        poses[tag_id] = (rvec, tvec)
                        
                        # Draw coordinate axes
                        axis_length = tag_size
                        axis_points_3d = np.array([
                            [0, 0, 0],
                            [axis_length, 0, 0],  # X-axis
                            [0, axis_length, 0],  # Y-axis
                            [0, 0, axis_length]   # Z-axis
                        ], dtype=np.float32).reshape(-1, 1, 3)  # Correct shape for fisheye.projectPoints
                        
                        axis_points, _ = cv2.fisheye.projectPoints(
                            axis_points_3d,
                            rvec, tvec, camera_matrix, dist_coeffs
                        )
                        
                        # Convert projected points to pixel coordinates
                        origin = tuple(map(int, axis_points[0].ravel()))
                        x_axis = tuple(map(int, axis_points[1].ravel()))
                        y_axis = tuple(map(int, axis_points[2].ravel()))
                        z_axis = tuple(map(int, axis_points[3].ravel()))
                        
                        # Draw coordinate axes on the image
                        image = cv2.line(image, origin, x_axis, (0, 0, 255), 2)  # X-axis in red
                        image = cv2.line(image, origin, y_axis, (0, 255, 0), 2)  # Y-axis in green
                        image = cv2.line(image, origin, z_axis, (255, 0, 0), 2)  # Z-axis in blue
                        
                        # Add pose information to image
                        pos_text = f"Pos: ({tvec[0][0]:.2f}, {tvec[1][0]:.2f}, {tvec[2][0]:.2f})"
                        rot_text = f"Rot: ({rvec[0][0]:.2f}, {rvec[1][0]:.2f}, {rvec[2][0]:.2f})"
                        cv2.putText(image, pos_text, 
                                   (int(detect.center[0]), int(detect.center[1]) + 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        cv2.putText(image, rot_text, 
                                   (int(detect.center[0]), int(detect.center[1]) + 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        
                        print(f"Tag {tag_id} position: {tvec.ravel()}")
                        print(f"Tag {tag_id} rotation: {rvec.ravel()}")
                except Exception as e:
                    print(f"Error calculating pose for tag {tag_id}: {e}")
    
    return image, poses




########################## init ##########################
gs.init(backend=gs.gpu)
scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -2, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
            max_FPS=200,
        ),
        show_viewer=True,
        rigid_options=gs.options.RigidOptions(
            enable_joint_limit=False,
            enable_collision=False,
            gravity=(0, 0, -0),
        ),
    )

genisis_arm = scene.add_entity(gs.morphs.MJCF(
    file  = '/home/hans/projects/genesis_trial/mujoco_menagerie/trs_so_arm100/so_arm100.xml',
    pos   = (0.0, 0.0, 0.0),
    euler = (0, 0, 0),
    requires_jac_and_IK = True
))
print(genisis_arm)
scene.build()
jnt_names = [
    'Rotation',
    'Pitch',
    'Elbow',
    'Wrist_Pitch',
    'Wrist_Roll',
    'Jaw',
]
dofs_idx = [genisis_arm.get_joint(name).dof_idx_local for name in jnt_names]



 

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image
 


class wasdRobot:

    def __init__(self,config: ManipulatorRobotConfig):
        self.config = config
        self.robot_type = self.config.type
        self.calibration_dir = Path(self.config.calibration_dir)
        self.leader_arms = make_motors_buses_from_configs(self.config.leader_arms)
        self.follower_arms = make_motors_buses_from_configs(self.config.follower_arms)
        self.cameras = make_cameras_from_configs(self.config.cameras)
        self.is_connected = False
        self.logs = {}
        self.camera_calibration = None


      
    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                "ManipulatorRobot is already connected. Do not run `robot.connect()` twice."
            )

 

        # Connect the arms
        for name in self.follower_arms:
            print(f"Connecting {name} follower arm.")
            self.follower_arms[name].connect()
 

        if self.robot_type in ["koch", "koch_bimanual", "aloha"]:
            from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
        elif self.robot_type in ["so100", "moss"]:
            from lerobot.common.robot_devices.motors.feetech import TorqueMode

        # # We assume that at connection time, arms are in a rest position, and torque can
        # # be safely disabled to run calibration and/or set robot preset configurations.
        # for name in self.follower_arms:
        #     self.follower_arms[name].write("Torque_Enable", TorqueMode.DISABLED.value)
 

        print("conntetec")
        self.activate_calibration()

        # Set robot preset (e.g. torque in leader gripper for Koch v1.1)
        if self.robot_type in ["koch", "koch_bimanual"]:
            self.set_koch_robot_preset()
        elif self.robot_type == "aloha":
            self.set_aloha_robot_preset()
        elif self.robot_type in ["so100", "moss"]:
            self.set_so100_robot_preset()

        # Enable torque on all motors of the follower arms
        for name in self.follower_arms:
            print(f"Activating torque on {name} follower arm.")
            #self.follower_arms[name].write("Torque_Enable", 1)

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
        for name in self.follower_arms:
            self.follower_arms[name].read("Present_Position")
 
 

        self.is_connected = True

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

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()` before disconnecting."
            )

        if self.robot_type in ["koch", "koch_bimanual", "aloha"]:
            from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
        elif self.robot_type in ["so100", "moss"]:
            from lerobot.common.robot_devices.motors.feetech import TorqueMode

        for name in self.follower_arms:
            self.follower_arms[name].write("Torque_Enable", TorqueMode.DISABLED.value)

        for name in self.follower_arms:
            self.follower_arms[name].disconnect()

   

        self.is_connected = False

    def get_next_wp(self, curr_pos, goal):
        curr_pos_rad = np.array([-1,-1,1,1,-1,1])*curr_pos/180*np.pi
        print("pos_start", curr_pos, curr_pos_rad)
        genisis_arm.set_qpos(curr_pos_rad)
        scene.visualizer.update()
        print("genPos", genisis_arm.get_pos())

        end_effector = genisis_arm.get_link('Fixed_Jaw')
        qpos = genisis_arm.inverse_kinematics(
            link = end_effector,
            pos  = goal,
            #quat = np.array([0, 0, 0, 1]),
        )
        
        return qpos

    def load_camera_calibration(self, yaml_file):
        """
        Load camera calibration parameters from a YAML file.
        
        Expected YAML format:
        cam0:
          cam_overlaps: []
          camera_model: pinhole
          distortion_coeffs: [-0.006066991500288485, -0.03373518267597992, 0.013247729138741051, -0.0024388023912218434]
          distortion_model: equidistant
          intrinsics: [351.2932161422478, 351.51548235626177, 641.7888281254284, 312.45578613147603]
          resolution: [1280, 720]
          rostopic: /cam0/image_raw
        
        Args:
            yaml_file (str): Path to the YAML calibration file
            
        Returns:
            dict: Dictionary containing the camera calibration parameters or None if loading fails
        """
        try:
            with open(yaml_file, 'r') as f:
                calibration_data = yaml.safe_load(f)
            
            # Verify the expected structure
            if 'cam0' not in calibration_data:
                print(f"Error: Expected 'cam0' key not found in calibration file {yaml_file}")
                return None
                
            print(f"Successfully loaded camera calibration from {yaml_file}")
   
            cam_params = calibration_data['cam0']
            
            # Extract camera resolution if available
            if 'resolution' in cam_params:
                self.width, self.height = cam_params['resolution']
                print(f"Using calibrated camera resolution: {self.width}x{self.height}")
            
            # Extract other camera parameters as needed
            if 'intrinsics' in cam_params and 'distortion_coeffs' in cam_params:
                # Format: [fx, fy, cx, cy]
                intrinsics = cam_params['intrinsics']
                # Create camera matrix for undistortion
                self.camera_matrix = np.array([
                    [intrinsics[0], 0, intrinsics[2]],
                    [0, intrinsics[1], intrinsics[3]],
                    [0, 0, 1]
                ])
                
                # Get distortion coefficients
                self.dist_coeffs = np.array(cam_params['distortion_coeffs'])
                
                print(f"Camera matrix: {self.camera_matrix}")
                print(f"Distortion coefficients: {self.dist_coeffs}")

            return calibration_data
        except Exception as e:
            print(f"Error loading camera calibration file: {e}")
            return None
            
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
                print("LOADED CAL")
            else:
                # TODO(rcadene): display a warning in __init__ if calibration file not available
                print(f"Missing calibration file '{arm_calib_path}'")

                if self.robot_type in ["koch", "koch_bimanual", "aloha"]:
                    from lerobot.common.robot_devices.robots.dynamixel_calibration import run_arm_calibration

                    calibration = run_arm_calibration(arm, self.robot_type, name, arm_type)

                elif self.robot_type in ["so100", "moss"]:
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
       



@parser.wrap()
def control_robot(cfg: ControlPipelineConfig):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model = FastSAM("/home/hans/projects/robot_arm/FastSAM/weights/FastSAM-x.pt")
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options,
                                        num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    print(cfg)
    init_logging()
    logging.info(pformat(asdict(cfg)))
    print(cfg.robot)

    pygame.init()


    robot = wasdRobot(cfg.robot)
    if not robot.is_connected:
        robot.connect()

    # Load camera calibration if provided
    camera_cal_file = "/home/hans/projects/robot_arm/camera_calibration/output-camchain.yaml"
    robot.load_camera_calibration(camera_cal_file)

    main_window = pygame.display.set_mode((robot.width*2, robot.height + 30))  # Add 30 pixels for text
    pygame.display.set_caption("Camera Feeds (Press 'q' to quit)")
    font = pygame.font.Font(None, 30)

    
 
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, robot.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, robot.height)
    cap.set(cv2.CAP_PROP_FPS, 30.0)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # Read follower position
    follower_pos = {}
    for name in robot.follower_arms:
        print(name)
        before_fread_t = time.perf_counter()
        follower_pos[name] = robot.follower_arms[name].read("Present_Position")
        follower_pos[name] = torch.from_numpy(follower_pos[name])
        print(follower_pos)
        robot.logs[f"read_follower_{name}_pos_dt_s"] = time.perf_counter() - before_fread_t
    
    present_pos = robot.follower_arms[name].read("Present_Position")
    goal_pos = np.array([0.0,-0.2,0.2])
    init_pos = goal_pos.copy()  # Make a copy to avoid reference issues
    running = True
    waypoints = []
    xyzVec = np.zeros(3)

    xLims = [-0.13, 0.13]
    yLims = [-0.3,-0.12]
    zLims = [ 0.1,0.3]

    #pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Base-hf")
    #pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf")
    
 
    #another solution: https://github.com/Sousannah/hand-tracking-using-mediapipe/blob/main/hand_tracking.py
    
    wrist_pos_3d = np.array([0,0,0])
    prev_wrist_pos_3d = np.array([0,0,0])
    april_zero = np.array([0,0,0])  

    # Add variables for motion smoothing
    smoothing_factor = 0.2  # Lower value = more smoothing (0.0 to 1.0)
    filtered_goal_pos = goal_pos.copy()
    previous_tvec = None
    tvec_history = []
    tvec_history_size = 3  # Number of frames to average for smoother motion

    while running:
        

        start = time.time()
        
        ret, frame = cap.read()
        
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        og_frame = frame.copy()
        
        # Detect and get pose of AprilTags
        # Note: We pass the original camera_matrix and dist_coeffs because the image is already undistorted
        frame, april_pose = plotApril(frame, robot.camera_matrix, robot.dist_coeffs, tag_size=0.06)
        got_april_pose = False
        if (len(april_pose) > 0):
            rvec, tvec = april_pose[0][0], april_pose[0][1]
            tvec = tvec.reshape(3)
            
            # Convert rotation vector to rotation matrix
            rmat, _ = cv2.Rodrigues(rvec)
            
            # Convert rotation matrix to Euler angles in degrees
            # The order is [roll (x), pitch (y), yaw (z)]
            euler_angles = np.zeros(3)
            
            # Calculate yaw (around z-axis)
            euler_angles[2] = np.arctan2(rmat[1, 0], rmat[0, 0]) * 180 / np.pi
            
            # Calculate pitch (around y-axis)
            euler_angles[1] = np.arctan2(-rmat[2, 0], 
                                         np.sqrt(rmat[2, 1]**2 + rmat[2, 2]**2)) * 180 / np.pi
            
            # Calculate roll (around x-axis)
            euler_angles[0] = np.arctan2(rmat[2, 1], rmat[2, 2]) * 180 / np.pi
            
            print(f"Roll: {euler_angles[0]:.2f}°, Pitch: {euler_angles[1]:.2f}°, Yaw: {euler_angles[2]:.2f}°")
            
            # Add roll, pitch, yaw information to the image
            rpy_text = f"R:{euler_angles[0]:.1f} P:{euler_angles[1]:.1f} Y:{euler_angles[2]:.1f} deg"
            cv2.putText(frame, rpy_text, 
                       (30, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Apply a basic outlier rejection - ignore sudden large jumps
            if previous_tvec is not None:
                distance = np.linalg.norm(tvec - previous_tvec)
                if distance > 0.3:  # If position jumps more than 10cm, ignore this reading
                    print(f"Detected jump of {distance}m, ignoring this frame")
                    tvec = previous_tvec
            
            # Add to history for filtering
            tvec_history.append(tvec)
            if len(tvec_history) > tvec_history_size:
                tvec_history.pop(0)
                
            # Apply moving average filter
            if len(tvec_history) > 0:
                tvec = np.mean(tvec_history, axis=0)
            
            previous_tvec = tvec.copy()
            
            print(f"rvec: {rvec}, tvec: {tvec}")
            got_april_pose = True
              
         

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False 
        keys = pygame.key.get_pressed()
        increment = 0.01
        if keys[pygame.K_w]:
            xyzVec[1] -= increment
        if keys[pygame.K_s]:
            xyzVec[1] += increment
        if keys[pygame.K_d]:
            xyzVec[0] -= increment
        if keys[pygame.K_a]:
            xyzVec[0] += increment
        if keys[pygame.K_UP]:
            xyzVec[2] += increment
        if keys[pygame.K_DOWN]:
            xyzVec[2] -= increment
        if keys[pygame.K_SPACE]:
            april_zero = tvec
            filtered_goal_pos = goal_pos.copy()  # Reset filtered position when zeroing
            tvec_history = []  # Clear history when re-zeroing
 
        
         
        
        
        
        # # Fisheye undistortion requires a different API
        # h, w = frame.shape[:2]
        # # We need to adjust the camera matrix for fisheye undistortion
        # K = robot.camera_matrix.copy()
        # D = robot.dist_coeffs.copy()
        
        # print(f"K {K}, D {D}")

        # # Calculate the optimal new camera matrix
        # map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        #     K, D, np.eye(3), K, (w, h), cv2.CV_16SC2
        # )
        
       

        
        # depth_og = np.array(pipe(Image.fromarray(frame))["depth"])
        # print(f"Minimum value: {np.min(depth_og)}")
        # print(f"Maximum value: {np.max(depth_og)}")
        # print(f"type:{depth_og.dtype}")

        # # depth = cv2.normalize(depth_og, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # depth = cv2.cvtColor(depth_og, cv2.COLOR_GRAY2RGB)

        end = time.time()
        print("Time taken for block 1:", end - start, "seconds")
        # img = Image.fromarray(frame)
        # img = img.convert("RGB")
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        mp_frame_og = mp.Image(image_format=mp.ImageFormat.SRGB, data=og_frame)

        # STEP 3: Load the input image.
        #image = mp.Image.create_from_file("image.jpg")

        # STEP 4: Detect hand landmarks from the input image.
        detection_result = detector.detect(mp_frame_og)

        #https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker for good visual on what each index means

        # STEP 5: Process the classification result. In this case, visualize it.
        annotated_image = draw_landmarks_on_image(mp_img.numpy_view(), detection_result)
        # cv2_imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

        
   
        hand_landmarks_list = detection_result.hand_landmarks
       
 

        THUMB_TIP_INDEX = 4
        PINKY_TIP_INDEX = 20
        WRIST_INDEX = 0

        fov = 70.0
        w = robot.width
        h = robot.height
        f =  (w/2.0)/np.tan(fov/2.0/180.0*np.pi) 
        print(f"focal_length {f}")
        K = np.array([[f,   0.0, w/2],
                      [0.0, f,   h/2],
                      [0.0, 0.0, 1.0]])

        xy_dist = -1
        z = 0

        # ref_pix = [int(w*0.75), int(h)-1]
        # z_ref = depth_og[ref_pix[1],ref_pix[0]]
        # cv2.circle(depth, (int(ref_pix[0]), int(ref_pix[1])), 2, (0,255,0), 10)


        # Loop through the detected hands to visualize.
        if len(hand_landmarks_list) > 0:
            first_hand_list = hand_landmarks_list[0]
            if (len(first_hand_list) < 21):
                print("full hand not detected")
            else:
                pinky_tip_ldmk = first_hand_list[PINKY_TIP_INDEX]
                thumb_tip_ldmk = first_hand_list[THUMB_TIP_INDEX]
                wrist_ldmk = first_hand_list[WRIST_INDEX]

                pinky_pos = np.array([pinky_tip_ldmk.x, pinky_tip_ldmk.y, 1.0])
                thumb_pos = np.array([thumb_tip_ldmk.x, thumb_tip_ldmk.y, 1.0])
                wrist_pos = np.array([wrist_ldmk.x, wrist_ldmk.y, 1.0])
                wrist_px = np.array([wrist_ldmk.x*w, wrist_ldmk.y*h, 1.0])


                wrist_pos_3d = np.linalg.inv(K)@wrist_px
                # z = depth_og[int(wrist_px[1])][int(wrist_px[0])] - z_ref
                # print(f"depth map at pixel ({int(wrist_px[1])},{int(wrist_px[0])}) is {z}")
                
                # if (z > 100):
                #     z = 0
                z = 10

                wrist_pos_3d = wrist_pos_3d * z
                # cv2.circle(depth, (int(wrist_px[0]), int(wrist_px[1])), 2, (255,0,0), 10)


                print(f"pinky_pos {pinky_pos}, thumb_pos {thumb_pos}, wrist_pos {wrist_pos}")
                xy_dist =  np.linalg.norm(pinky_pos-thumb_pos)
               
                 
                


                
        diff = (wrist_pos_3d - prev_wrist_pos_3d)/20
        prev_wrist_pos_3d = wrist_pos_3d
        print(diff)
        # if np.linalg.norm(diff) < 0.1:
        #     diff[2] = -diff[1]
        #     diff[1] = 0
        #     xyzVec = diff
        


        smoother = 1.0
        for name in robot.follower_arms:
            if ((xyzVec != np.zeros(3)).any()):
                goal_pos =  goal_pos + xyzVec
                xyzVec = np.zeros(3)
            #goal_pos += np.array([0.0,0.01,0.0])
            if (got_april_pose and (april_zero != np.array([0,0,0])).all()):
                print(init_pos, tvec, april_zero)
                change = tvec - april_zero
                change = np.array([change[2], change[0], change[1]])
                goal_pos = (init_pos + change)*smoother + goal_pos*(1-smoother)
             
                # x = -z y = -x, z =y

            goal_pos[0] = np.clip(goal_pos[0],xLims[0],xLims[1])
            goal_pos[1] = np.clip(goal_pos[1],yLims[0],yLims[1])
            goal_pos[2] = np.clip(goal_pos[2],zLims[0],zLims[1])

            print("going to goal pos", goal_pos)
            joint = robot.get_next_wp(present_pos,goal_pos)
         
            next_pos_rad = joint.detach().cpu().numpy()/np.pi*180
            print("next_pos_rad", next_pos_rad)

            next_pos_rad[4] = present_pos[5]
            next_pos_rad[5] = present_pos[5]

            if (got_april_pose): 
                next_pos_rad[4] = 90 - euler_angles[0]
            if xy_dist >= 0:
                next_pos_rad[5] = min(90,int(xy_dist*3*170) - 5)
                if (next_pos_rad[5] < 40):
                    next_pos_rad[5] = 0
                

            robot.follower_arms[name].write("Goal_Position", next_pos_rad*np.array([-1,-1,1,1,-1,1]))
            present_pos = robot.follower_arms[name].read("Present_Position")

        

        # Clear the window
        main_window.fill((0, 0, 0))
        hand_pose_img = pygame.surfarray.make_surface(np.transpose(annotated_image, (1, 0, 2)))
        #depth_img = pygame.surfarray.make_surface(np.transpose(depth, (1, 0, 2)))
        # if (xy_dist > 0):
        text_surface = font.render(f"diff {diff}, z {z} wrist pos: {wrist_pos_3d}, PinkyToThumbDist {np.round(xy_dist,3)} ", True, (255, 255, 255))  


        main_window.blit(text_surface, (30, robot.height + 10))
        main_window.blit(hand_pose_img, (0, 0))
        #main_window.blit(depth_img, (robot.width, 0))
        


         
        pygame.display.flip()

        # Add a small delay to control update rate
        time.sleep(0.01)
     

    pygame.quit()
 
    print(waypoints)

   

    robot.disconnect()

# @parser.wrap()
# def control_robot(cfg: ControlPipelineConfig):
#     print(cfg)
#     init_logging()
#     logging.info(pformat(asdict(cfg)))
#     print(cfg.robot)


#     pygame.init()

#     # Set up the display (optional, but needed for event handling)
#     screen = pygame.display.set_mode((200, 200))

     
    


#     robot = wasdRobot(cfg.robot)

#     if not robot.is_connected:
#         robot.connect()

#      # Read follower position
#     follower_pos = {}
#     for name in robot.follower_arms:
#         print(name)
#         before_fread_t = time.perf_counter()
#         follower_pos[name] = robot.follower_arms[name].read("Present_Position")
#         follower_pos[name] = torch.from_numpy(follower_pos[name])
#         print(follower_pos)
#         robot.logs[f"read_follower_{name}_pos_dt_s"] = time.perf_counter() - before_fread_t
    
#     present_pos = robot.follower_arms[name].read("Present_Position")
#     goal_pos = present_pos
#     running = True
#     waypoints = []
#     while running:
#         controlvec = np.zeros(6)
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False

#             if event.type == pygame.KEYDOWN:
#                 if event.key == pygame.K_SPACE:
#                     waypoints.append(goal_pos)
             
#         keys = pygame.key.get_pressed()
#         if keys[pygame.K_w]:
#             controlvec[1] += 1
#         if keys[pygame.K_s]:
#             controlvec[1] -= 1
#         if keys[pygame.K_a]:
#             controlvec[0] -= 1
#             print("S key held")
#         if keys[pygame.K_d]:
#             controlvec[0] += 1
#             print("D key held")
#         if keys[pygame.K_UP]:
#             controlvec[2] -= 1
#         if keys[pygame.K_DOWN]:
#             controlvec[2] += 1
#         if keys[pygame.K_q]:
#             controlvec[5] += 1
#         if keys[pygame.K_e]:
#             controlvec[5] -= 1

        


      


        

#         for name in robot.follower_arms:
#             if ((controlvec != np.zeros(6)).any()):
#                 goal_pos =  goal_pos + controlvec
#             goal_pos[3] = 90
#             goal_pos[4] = 0
        
#             print(goal_pos)
#             robot.follower_arms[name].write("Goal_Position", goal_pos)
#             present_pos = robot.follower_arms[name].read("Present_Position")
           
#             print(present_pos)
 
#                 # present_pos = torch.from_numpy(present_pos)
#                 # goal_pos = ensure_safe_goal_position(goal_pos, present_pos, self.config.max_relative_target)
#         time.sleep(0.02)
#         #break

#     pygame.quit()
    
#     print(waypoints)

#     while True:
#         for wp in waypoints:
#             robot.follower_arms[name].write("Goal_Position", wp)
#             time.sleep(0.5)

#     robot.disconnect()


import pygame


# def start_genisis():
#     ########################## init ##########################
#     gs.init(backend=gs.gpu)

#     # ########################## create a scene ##########################
#     scene = gs.Scene(
#         show_viewer = False,
#     )

#     # ########################## entities ##########################
#     # plane = scene.add_entity(
#     #     gs.morphs.Plane(),
#     # )

#     # when loading an entity, you can specify its pose in the morph.
#     franka = scene.add_entity(gs.morphs.MJCF(
#         file  = '/home/hans/projects/genesis_trial/mujoco_menagerie/trs_so_arm100/so_arm100.xml',
#         pos   = (1.0, 1.0, 0.0),
#         euler = (0, 0, 0),
#         requires_jac_and_IK = True
#     ))

#     # ########################## build ##########################
#     scene.build()
#     jnt_names = [
#         'Rotation',
#         'Pitch',
#         'Elbow',
#         'Wrist_Pitch',
#         'Wrist_Roll',
#         'Jaw',
#     ]
#     dofs_idx = [franka.get_joint(name).dof_idx_local for name in jnt_names]
#     print(franka)

#     # get the end-effector link
#     end_effector = franka.get_link('Fixed_Jaw')

#     # move to pre-grasp pose
#     qpos = franka.inverse_kinematics(
#         link = end_effector,
#         pos  = np.array([0.65, 0.0, 0.25]),
#         quat = np.array([0, 1, 0, 0]),
#     )
#     print(qpos)
#     # # gripper open pos
#     # qpos[-2:] = 0.04
#     path = franka.plan_path(
#         qpos_goal     = qpos,
#         num_waypoints = 5, # 2s duration
#     )
#     print(path)

#     qpos = franka.inverse_kinematics(
#         link = end_effector,
#         pos  = np.array([0.65, 0.5, 0.25]),
#         quat = np.array([0, 1, 0, 0]),
#     )
#     path = franka.plan_path(
#         qpos_goal     = qpos,
#         num_waypoints = 5, # 2s duration
#     )
 
if __name__ == "__main__":
    #start_genisis()
    control_robot()







#https://moveit.picknik.ai/main/doc/examples/ikfast/ikfast_tutorial.html
# URDF https://github.com/JafarAbdi/ros2_so_arm100/blob/main/so_arm100_description/urdf/so_arm100.urdf
# https://github.com/google-deepmind/mujoco_menagerie/blob/main/trs_so_arm100/so_arm100.xml


#gensis seems way easier:
#https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/hello_genesis.html
