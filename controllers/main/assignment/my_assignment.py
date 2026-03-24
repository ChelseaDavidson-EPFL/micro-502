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

# Module-level singleton so main.py can call assignment.get_command() unchanged
_controller = MyAssignment()

def get_command(sensor_data, camera_data, dt):
    return _controller.compute_command(sensor_data, camera_data, dt)

