#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
import time
import threading
from collections import defaultdict
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from inference import get_model
import cv2
import numpy as np
import supervision as sv
from supervision.draw.color import ColorPalette
from cv_bridge import CvBridge
from lightglue import match_pair
from lightglue import LightGlue, DISK
from lightglue.utils import numpy_image_to_torch
from scipy.spatial.transform import Rotation as R
import torch
import copy

CUSTOM_COLOR_MAP = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#0082c8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#d2f53c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#aa6e28",
    "#fffac8",
    "#800000",
    "#aaffc3",
]

class ImageFramerateMonitor(Node):
    def __init__(self):
        super().__init__('image_framerate_monitor')
        
        # Image topics to monitor
        self.image_topics = [
            "/camera/ego_cam/image_raw",
            #"/camera/ego_cam2/image_raw", 
            #"/camera/left_wrist_cam/image_raw",
            "/camera/right_wrist_cam/image_raw"
        ]
        
        # Dictionary to store frame timestamps for each topic
        self.frame_timestamps = defaultdict(list)
        
        # Dictionary to store last printed time for each topic
        self.last_print_time = defaultdict(float)
        
        # Print interval (seconds)
        self.print_interval = 2.0
        
        # Create subscribers for each image topic
        self.subscribers = {}
        for topic in self.image_topics:
            self.subscribers[topic] = self.create_subscription(
                Image,
                topic,
                lambda msg, topic=topic: self.image_callback(msg, topic),
                10
            )
            self.get_logger().info(f'Subscribed to {topic}')
        self.bridge = CvBridge()

        # init publisher for each image topic
        self.segmented_image_publishers = {}
        for topic in self.image_topics:
            self.segmented_image_publishers[topic] = self.create_publisher(Image, f"{topic}/segmented", 1)

        # Create a pose publisher for switch pose
        self.pose_pub = self.create_publisher(
            PoseStamped,
            'switch_pose',
            10
        )

        # Initialize segmentation model
        # build SAM2 image predictor
        self.model_ethernet = get_model(model_id="ethernet-cable-detection/1")
        self.sam2_checkpoint = "/root/boost_ws/non_ros_pkgs/sam2/checkpoints/sam2.1_hiera_tiny.pt"
        self.model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
        self.sam2_model = build_sam2(self.model_cfg, self.sam2_checkpoint, device="cuda")
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)
        self.template_image = numpy_image_to_torch(cv2.imread("/root/boost_ws/network_switch.png")).cuda()

        # Thread-safe storage for latest images
        self.latest_images = {}
        self.latest_images_lock = threading.Lock()
 
        
        self.template_pts = [
            (705, 404),
            (768, 406),
            (825, 407),
            (881, 406),
            (937, 406),
            (993, 407),
            (1054, 407),
            (1094, 407),
            (1157, 408),
            (1211, 409),
            (1266, 409),
            (1320, 409),
            (1375, 409),
            (1434, 409),
            (1435, 509),
            (1373, 508),
            (1318, 508),
            (1265, 508),
            (1210, 508),
            (1156, 508),
            (1093, 508),
            (1053, 507),
            (991, 509),
            (937, 509),
            (880, 508),
            (824, 507),
            (767, 507),
            (704, 507)
        ]

        port_dist = 86.0/6.0/1000.0
        space = 0.01  # 1 cm space between ports
        y_dist = 0.025
        first_row = np.array([[0.0, 0.0, 0.0],
                            [port_dist, 0.0, 0.0],
                            [port_dist*2, 0.0, 0.0],
                            [port_dist*3, 0.0, 0.0],
                            [port_dist*4, 0.0, 0.0],
                            [port_dist*5, 0.0, 0.0],
                            [port_dist*6, 0.0, 0.0],
                            [port_dist*6+space, 0.0, 0.0],
                            [port_dist*7+space, 0.0, 0.0],
                            [port_dist*8+space, 0.0, 0.0],
                            [port_dist*9+space, 0.0, 0.0],
                            [port_dist*10+space, 0.0, 0.0],
                            [port_dist*11+space, 0.0, 0.0],
                            [port_dist*12+space, 0.0, 0.0]])
        second_row = first_row + np.array([0.0, y_dist, 0.0])
        # reverse second row to match template
        second_row = second_row[::-1]
        self.template_pts_3d = np.concatenate((first_row, second_row), axis=0).astype(np.float32)
        

        self.extractor = DISK(max_num_keypoints=2048).eval().cuda()  # load the extractor
        self.matcher = LightGlue(features='disk', flash=True).eval().cuda()  # load the matcher
        self.matcher.compile(mode='reduce-overhead')
        self.K = np.array([[610.6211437928561, 0.0, 327.0457322683239],
                           [0.0, 612.6500564468024, 241.83735262106],
                           [0.0, 0.0, 1.0]], dtype=np.float32)
        self.dist = np.array([0.11157808430798598, -0.24405456796572023, 0.006705012014351482, -0.0005953949546751423], dtype=np.float32)  # Distortion coefficients
        
    

        # Start segmentation thread
        self.segmentation_thread_running = True
        self.segmentation_thread = threading.Thread(target=self.segmentation_worker, daemon=True)
        self.segmentation_thread.start()

        self.get_logger().info('Image framerate monitor started')
        self.get_logger().info(f'Monitoring topics: {self.image_topics}')

    def segment_images(self, images):
        start_total = time.time()
        
        segmented_images = {}
        raw_images = list(images.values())
        segmented_masks = {}
        
        # Time model inference
        start_inference = time.time()
        results_robo = self.model_ethernet.infer(raw_images, batch=True)
        inference_time = time.time() - start_inference
        print(f"Model inference: {inference_time:.3f}s for {len(results_robo)} images")
        
        # Time detection processing
        start_detection = time.time()
        detections_list = []
        images_with_detections = []
        boxes = []
        segment_keys = []

        i = 0
        for key, image in images.items():
            detections = sv.Detections.from_inference(results_robo[i])
            detections = detections[detections.confidence > 0.4]
            if len(detections.confidence) > 0:
                detections_list.append(detections)
                images_with_detections.append(image)
                boxes.append(detections.xyxy)
                segment_keys.append(key)
            else:
                segmented_images[key] = image  # No detections, return original image
            i += 1
        detection_time = time.time() - start_detection
        print(f"Detection processing: {detection_time:.3f}s")

        # Time SAM2 prediction
        start_sam = time.time()
        if images_with_detections:  # Only run if there are images with detections
            self.sam2_predictor.set_image_batch(images_with_detections)
            masks_list, scores, logits = self.sam2_predictor.predict_batch(
                box_batch=boxes,
                multimask_output=False,
            )
        sam_time = time.time() - start_sam
        print(f"SAM2 prediction: {sam_time:.3f}s")

        # Time annotation
        start_annotation = time.time()
        CLASS_NAMES = ["ethernet cable"]  
        for i in range(len(images_with_detections)):
            image = images_with_detections[i]
            detections = detections_list[i]
            masks = masks_list[i]
            input_boxes = detections.xyxy
            if masks.ndim == 4:
                masks = masks.squeeze(1)
            confidences = detections.confidence
            class_names = CLASS_NAMES * len(confidences)
            class_ids = detections.class_id
            labels = [
                f"{class_name} {confidence:.2f}"
                for class_name, confidence
                in zip(class_names, confidences)
            ]
            detections = sv.Detections(
                xyxy=input_boxes,
                mask=masks.astype(bool),
                class_id=class_ids
            )
            segmented_masks[segment_keys[i]] = masks.astype(bool)
            # box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
            # annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections)
            # label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
            # annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
            annotated_frame = mask_annotator.annotate(scene=image.copy(), detections=detections)
            segmented_images[segment_keys[i]] = annotated_frame
        annotation_time = time.time() - start_annotation
        print(f"Annotation: {annotation_time:.3f}s")

        # Time template matching
        start_matching = time.time()
        for key, image in segmented_images.items():
            query_image = numpy_image_to_torch(image).cuda()
            with torch.no_grad():
                feats0, feats1, matches01 = match_pair(self.extractor, self.matcher, self.template_image, query_image)
            matches = matches01['matches']
            
            if len(matches) > 100 and matches.max() < min(len(feats0['keypoints']), len(feats1['keypoints'])):
                points0 = feats0['keypoints'][matches[..., 0]]
                points1 = feats1['keypoints'][matches[..., 1]]
                points0 = points0.cpu().numpy().astype(np.float32)
                points1 = points1.cpu().numpy().astype(np.float32)
                ports_plugged = np.zeros((2,12), dtype=bool)
                port_ints = []
                pt_to_go_to = np.array([0,0,0])
                pt_to_go_to_2d = np.array([0,0], dtype=np.float32)
                if len(points0) > 10 and len(points1) > 10:
                    H, mask = cv2.findHomography(points0, points1, cv2.RANSAC, 10.0)
                    # Can use template matching or optical flow to refine points.
                    if H is not None:
                        pts = np.array(self.template_pts).reshape(-1, 1, 2).astype(np.float32)
                        dst = cv2.perspectiveTransform(pts, H)
                        pts_3d = np.array(self.template_pts_3d).reshape(-1, 1, 3).astype(np.float32)
                        
                        # solve pnp to get the 3D points
                        ret, rvec, tvec = cv2.solvePnP(pts_3d, dst, self.K, self.dist, flags=cv2.SOLVEPNP_ITERATIVE)
                        if ret:
                            last_pt = None
                            port_num = 0
                            for enum, pt in enumerate(dst):
                                # draw line between last_pt and pt
                                if enum % 7 != 0:
                                    
                                    #cv2.line(image, tuple(last_pt[0].astype(int)), tuple(pt[0].astype(int)), (0, 255, 0), 2)
                                    center_pt = (last_pt[0] + pt[0]) / 2
                                    center_pt_dist = np.linalg.norm(last_pt[0] - pt[0])
                                    center_pt[1] += center_pt_dist / 4
                                    # Check if center pt is true in the image mask
                                    if key in segment_keys:
                                        masks = segmented_masks[key]
                                        for mask in masks:
                                            if center_pt[1] > 0 and center_pt[1] < mask.shape[0] and center_pt[0] > 0 and center_pt[0] < mask.shape[1]:
                                                if mask[int(center_pt[1]), int(center_pt[0])] == True:
                                                    if (pt_to_go_to == np.array([0,0,0])).all():
                                                        pt_to_go_to = (self.template_pts_3d[enum-1] + self.template_pts_3d[enum])/2
                                                        pt_to_go_to_2d = (last_pt[0] + pt[0]) / 2
                                                    # draw circle at center_pt
                                                    cv2.circle(image, tuple(center_pt.astype(int)), 2, (0, 255, 0), -1)
                                                    print(f"Port {port_num} plugged in at {center_pt}")
                                                    if port_num < 12:
                                                        ports_plugged[0, port_num] = True
                                                        port_ints.append(port_num*2 +1)
                                                    else:
                                                        ports_plugged[1, 11 - port_num % 12] = True
                                                        port_ints.append((12 - port_num % 12)*2)
                                                    break
                                    port_num += 1
                                    

                                    #cv2.circle(image, tuple(center_pt.astype(int)), 2, (0, 255, 0), -1)
                                # draw circle at pt
                                cv2.circle(image, tuple(pt[0].astype(int)), 2, (0, 0, 255), -1)
                                # add text label for the pt number
                                cv2.putText(image, f"{enum}", tuple(pt[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255), 1)
                                last_pt = pt
                            
                            # Transform pt_to_go_to using the rotation matrix
                            R_mat, _ = cv2.Rodrigues(rvec)
                            rotated_pt = R_mat @ pt_to_go_to.reshape(3, 1)
                            tvec[0][0] += rotated_pt[0, 0]
                            tvec[1][0] += rotated_pt[1, 0]
                            tvec[2][0] += rotated_pt[2, 0]

                            # get pose frpm rvec and tvec
                            pose_stamped = PoseStamped()
                            pose_stamped.header.frame_id = key
                            pose_stamped.pose.position.x = tvec[0][0]  
                            pose_stamped.pose.position.y = tvec[1][0]  
                            pose_stamped.pose.position.z = tvec[2][0]  

                            # Convert rotation vector to rotation matrix
                            R_mat, _ = cv2.Rodrigues(rvec)
                            rotation = R.from_matrix(R_mat)
                            quat = rotation.as_quat()  # [x, y, z, w]
                            pose_stamped.pose.orientation.x = quat[0]
                            pose_stamped.pose.orientation.y = quat[1]
                            pose_stamped.pose.orientation.z = quat[2]
                            pose_stamped.pose.orientation.w = quat[3]
                            self.pose_pub.publish(pose_stamped)

                            axis = np.float32([[0.086,0,0], [0,0.025,0], [0,0,0.086]]).reshape(-1,3)
                            imgpts, _ = cv2.projectPoints(axis, rvec, tvec, self.K, None)
                            # draw the axis on the image
                            def draw(img, corners, imgpts):
                                corner = tuple(corners[0].ravel().astype("int32"))
                                imgpts = imgpts.astype("int32")
                                img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (0,0,255), 3)
                                img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 3)
                                img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (255,0,0), 3)
                                return img
                            image = draw(image,[pt_to_go_to_2d],imgpts)


                cv2.putText(image, f"Ports plugged: {port_ints}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                print(f"Ports plugged: {ports_plugged}")
            

                
        matching_time = time.time() - start_matching
        print(f"Template matching: {matching_time:.3f}s")

        total_time = time.time() - start_total
        print(f"Total segmentation time: {total_time:.3f}s")

        
        
        

        return segmented_images

 
    def segmentation_worker(self):
        """Worker thread that processes segmentation in batches"""
        while self.segmentation_thread_running:
            # Time this loop to control processing rate
            start_time = time.time()
        
            # Get latest images
            with self.latest_images_lock:
                current_images = self.latest_images.copy()
            
            if current_images:
                # Process all images in batch
                segmented_results = self.segment_images(current_images)
                
                # Publish segmented images
                for topic, segmented_image in segmented_results.items():
                    if topic in self.segmented_image_publishers:
                        try:
                            msg = self.bridge.cv2_to_imgmsg(segmented_image, "bgr8")
                            self.segmented_image_publishers[topic].publish(msg)
                        except Exception as e:
                            self.get_logger().error(f"Failed to publish segmented image for {topic}: {e}")

            # Control loop rate
            elapsed_time = time.time() - start_time
            rate = 1.0 / elapsed_time if elapsed_time > 0 else 1.0
            print(f"Segmentation worker rate: {rate:.2f} Hz")

            img = cv2.imread("/root/boost_ws/network_switch.png")
            for i in range(len(self.template_pts)):
                pt = self.template_pts[i]
                cv2.circle(img, tuple(pt), 2, (0, 255, 0), -1)
            cv2.imwrite("/root/boost_ws/network_switch_template.png", img)
            print("self.template_pts_3d", self.template_pts_3d.shape, self.template_pts_3d)

    def image_callback(self, msg, topic):
        current_time = time.time()
        image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # store latest image in thread-safe manner
        with self.latest_images_lock:
            self.latest_images[topic] = image

        # store timestamp
        self.frame_timestamps[topic].append(current_time)
        
        # Keep only timestamps from the last 10 seconds for accurate calculation
        cutoff_time = current_time - 10.0
        self.frame_timestamps[topic] = [
            t for t in self.frame_timestamps[topic] if t > cutoff_time
        ]
        
        # Print framerate every print_interval seconds
        if current_time - self.last_print_time[topic] >= self.print_interval:
            self.calculate_and_print_framerate(topic, current_time)
            self.last_print_time[topic] = current_time

    def calculate_and_print_framerate(self, topic, current_time):
        timestamps = self.frame_timestamps[topic]
        
        if len(timestamps) < 2:
            self.get_logger().info(f'{topic}: Insufficient data for framerate calculation')
            return
        
        # Calculate framerate over the last few seconds
        time_window = min(5.0, current_time - timestamps[0])  # Use up to 5 seconds
        frames_in_window = len([t for t in timestamps if t > current_time - time_window])
        
        if time_window > 0:
            framerate = frames_in_window / time_window
            self.get_logger().info(f'{topic}: {framerate:.2f} fps (over {time_window:.1f}s window)')
        else:
            self.get_logger().info(f'{topic}: Unable to calculate framerate')

    def __del__(self):
        """Cleanup when node is destroyed"""
        self.segmentation_thread_running = False
        if hasattr(self, 'segmentation_thread'):
            self.segmentation_thread.join(timeout=1.0)


def main(args=None):
    rclpy.init(args=args)
    
    try:
        monitor = ImageFramerateMonitor()
        rclpy.spin(monitor)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop segmentation thread
        if hasattr(monitor, 'segmentation_thread_running'):
            monitor.segmentation_thread_running = False
        rclpy.shutdown()


if __name__ == '__main__':
    main()
