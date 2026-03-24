import numpy as np
import time
import cv2

# The available ground truth state measurements can be accessed by calling sensor_data[item]. All values of "item" are provided as defined in main.py within the function read_sensors.
# The "item" values that you may later retrieve for the hardware project are:
# "x_global": Global X position
# "y_global": Global Y position
# "z_global": Global Z position
# 'v_x": Global X velocity
# "v_y": Global Y velocity
# "v_z": Global Z velocity
# "ax_global": Global X acceleration
# "ay_global": Global Y acceleration
# "az_global": Global Z acceleration (With gravtiational acceleration subtracted)
# "roll": Roll angle (rad)
# "pitch": Pitch angle (rad)
# "yaw": Yaw angle (rad)
# "q_x": X Quaternion value
# "q_y": Y Quaternion value
# "q_z": Z Quaternion value
# "q_w": W Quaternion value

# A link to further information on how to access the sensor data on the Crazyflie hardware for the hardware practical can be found here: https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/api/logs/#stateestimate

from enum import Enum

class Mode(Enum):
    TAKE_OFF = 1
    FIRST_LAP = 2
    FAST_LAP = 3


class MyAssignment:
    def __init__(self, ):
        # ---- INITIALISE YOUR VARIABLES HERE ----
        self.mode = Mode.FIRST_LAP
        self.has_taken_off = False
        self.gate_positions = [] # Store gates as tuples of tuples representing coords of the corners ((x, y, z), ... x 4)

    def compute_command(self, sensor_data, camera_data, dt):

        # NOTE: Displaying the camera image with cv2.imshow() will throw an error because GUI operations should be performed in the main thread.
        # If you want to display the camera image you can call it in main.py.

        lower, upper = self.rgb_to_hsv_bounds(r=211, g=144, b=222)  # using colour picker
        points = self.locate_pink_area(camera_data, lower, upper)
        if points is not None:
            cv2.polylines(camera_data, [points], isClosed=True, color=(0, 255, 0), thickness=2)

        cv2.imshow("Test", camera_data)
        cv2.waitKey(0)
        # Take off command
        if sensor_data['z_global'] < 0.49 and not self.has_taken_off:
            control_command = [sensor_data['x_global'], sensor_data['y_global'], 1.0, sensor_data['yaw']]
            return control_command
        if not self.has_taken_off and sensor_data['z_global'] > 0.49:
            self.has_taken_off = True

        # ---- YOUR CODE HERE ----
        control_command = [sensor_data['x_global'], sensor_data['y_global'], 1.0, sensor_data['yaw']]

        return control_command # Ordered as array with: [pos_x_cmd, pos_y_cmd, pos_z_cmd, yaw_cmd] in meters and radians
    
    def get_move_to_gate_command(self):
        #TODO - call locate gates, 
        return

    def locate_gates(self, camera_data):
        # TODO - call get image and locate corners in image, do check to see which is next, convert to world frame
        if (camera_data):
            print("Got an image")
        return 
    
    def locate_pink_area(self, image, lower_pink, upper_pink, min_area=100):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, lower_pink, upper_pink)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > min_area]

        if not contours:
            print("No pink region found — check your HSV bounds")
            return None

        largest = max(contours, key=cv2.contourArea)

        # Approximate contour to a polygon
        epsilon = 0.02 * cv2.arcLength(largest, True)
        polygon = cv2.approxPolyDP(largest, epsilon, True)

        # Reshape to a clean list of (x, y) points
        return polygon.reshape(-1, 2)

    def rgb_to_hsv_bounds(self, r, g, b, hue_tolerance=10, sat_min=40, val_min=80):
        """Helper to convert an RGB eyedropper reading into HSV bounds for inRange."""
        pixel = np.uint8([[[b, g, r]]])  # OpenCV is BGR
        hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

        lower = np.array([max(h - hue_tolerance, 0),   max(s - 60, 0),   max(v - 60, 0)])
        upper = np.array([min(h + hue_tolerance, 180),  255,               255])
        return lower, upper

    def locate_corners_of_gates_in_image(self, image, aspect_ratio_tolerance=0.3, min_area=500):
        """
        Detect pink rectangles/squares in an image.
        
        Args:
            image: image
            aspect_ratio_tolerance: how far from square (0=perfect square, 1=any rectangle)
            min_area: minimum contour area to filter out noise
        
        Returns:
            List of dicts with keys: bbox (x,y,w,h), contour, center, aspect_ratio
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Pink spans two hue ranges (red wraps around 0/180 in HSV)
        lower_pink1 = np.array([155, 40, 100])
        upper_pink1 = np.array([180, 255, 255])
        lower_pink2 = np.array([0,   40, 100])
        upper_pink2 = np.array([10,  255, 255])

        mask1 = cv2.inRange(hsv, lower_pink1, upper_pink1)
        mask2 = cv2.inRange(hsv, lower_pink2, upper_pink2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # fill small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # remove noise

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = min(w, h) / max(w, h)  # 1.0 = perfect square

            if aspect_ratio >= (1.0 - aspect_ratio_tolerance):
                results.append({
                    "bbox": (x, y, w, h),
                    "contour": cnt,
                    "center": (x + w // 2, y + h // 2),
                    "aspect_ratio": aspect_ratio
                })

        return results


    def draw_gates(self, img, results):
        for r in results:
            x, y, w, h = r["bbox"]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(img, r["center"], 5, (0, 255, 0), -1)
            label = f"AR: {r['aspect_ratio']:.2f}"
            cv2.putText(img, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return img

    def locate_corners_in_image_CV(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=100,      # max number of corners to return
            qualityLevel=0.01,   # minimum quality (0–1)
            minDistance=10       # min pixel distance between corners
        )
        corners = np.intp(corners)

        for c in corners:
            x, y = c.ravel()
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        return image
    
    def locate_corners_in_image_Harris(self, image):
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)

        harris = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
        harris = cv2.dilate(harris, None)  # enlarge corners for visibility

        # Threshold and mark corners
        image[harris > 0.01 * harris.max()] = [0, 0, 255]

        return image

# Module-level singleton so main.py can call assignment.get_command() unchanged
_controller = MyAssignment()

def get_command(sensor_data, camera_data, dt):
    return _controller.compute_command(sensor_data, camera_data, dt)

