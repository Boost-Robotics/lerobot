import numpy as np
import cv2
import os
import time
import datetime

# Add date and time to dataset directory
datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
dataset_dir = f"/home/hans/projects/robot_arm/camera_calibration/dataset_{datetime_str}"

def collect_images():
    # Create directory structure
    cam_dir = os.path.join(dataset_dir, "cam0")
    os.makedirs(cam_dir, exist_ok=True)
    
    height = 720
    width = 1280

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30.0)

    count = 0
    mod = 5  # Save every 5th frame
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break
            
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        
        # Press 'q' to exit the loop
        if key == ord('q'):
            break
            
        count += 1
        if (count % mod == 0):
            # Generate filename with timestamp
            timestamp = int(time.time() * 1e9)  # Nanoseconds timestamp
            filename = f"{timestamp}.png"
            filepath = os.path.join(cam_dir, filename)
            
            # Save the image
            cv2.imwrite(filepath, frame)
            print(f"Saved image: {filepath}")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print(f"Collected {count//mod} images in {cam_dir}")
 
if __name__ == "__main__":
    collect_images()