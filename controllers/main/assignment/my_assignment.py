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

class Lap(Enum):
    FIRST_LAP = 1
    FAST_LAP = 2

class Mode(Enum):
    TAKE_OFF = 3
    SEARCH_GATE = 4
    CAPTURE_SECOND_PHOTO = 5
    FLY_THROUGH_GATE = 6

# Pink gates
gates_r_value = 211
gates_g_value = 144
gates_b_value = 222

# Rotation amounts
take_second_photo_rotation = 0.15
search_gate_rotation = 0.5

# Thresholds
eps = 0.02 # Used to check if yaw is correct - Approx 1 degree
edge_threshold = 10  # Used to check if the detected gate is at the edge (pixels tolerance from edge)

# Camera constants
CAM_FOV = 1.5  # radians
CAM_WIDTH = 300  # pixels
CAM_OFFSET_BODY = np.array([0.03, 0, 0.01])  # camera offset from drone body
F_PIXELS = CAM_WIDTH / (2 * np.tan(CAM_FOV / 2))  # focal length in pixels (~161)

# Rotation from camera frame to body frame (from appendix: zcam=xdrone, xcam=-ydrone, ycam=-zdrone)
R_CAM_TO_BODY = np.array([
    [ 0, -1,  0],
    [ 0,  0, -1],
    [ 1,  0,  0]
])

class MyAssignment:
    def __init__(self, ):
        # ---- INITIALISE YOUR VARIABLES HERE ----
        self.mode = Mode.TAKE_OFF
        self.has_taken_off = False
        self.gate_positions = [] # Store gates as tuples of tuples representing coords of the corners ((x, y, z), ... x 4)
        self.gate_detection_img = None
        self.target_gate_detection_img = None
        self.current_gate_number = 0 # Doing zero indexing
        self.target_yaw = None
        self.current_gate_first_pixels = None
        self.current_gate_second_pixels = None
        self.current_gate_first_sensor_data = None
        self.current_gate_second_sensor_data = None

    def compute_command(self, sensor_data, camera_data, dt):
        # NOTE: Displaying the camera image with cv2.imshow() will throw an error because GUI operations should be performed in the main thread.
        # If you want to display the camera image you can call it in main.py.

        # Default control command - TODO: remove this later 
        control_command = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']]
        
        # Take off command
        if (self.mode == Mode.TAKE_OFF):
            if sensor_data['z_global'] < 0.49 and not self.has_taken_off:
                control_command = [sensor_data['x_global'], sensor_data['y_global'], 1.0, sensor_data['yaw']]
                return control_command
            if not self.has_taken_off and sensor_data['z_global'] > 0.49:
                self.has_taken_off = True
                self.mode = Mode.SEARCH_GATE
        # Search for gate command
        elif (self.mode == Mode.SEARCH_GATE):
            control_command = self.get_search_gate_command(camera_data, sensor_data)
        # Take 2nd photo command
        elif (self.mode == Mode.CAPTURE_SECOND_PHOTO):
            control_command = self.get_capture_second_photo_command(camera_data, sensor_data)      
        elif (self.mode == Mode.FLY_THROUGH_GATE):
            control_command = self.get_fly_through_gate_command(sensor_data)

        # ---- YOUR CODE HERE ----
        # self.get_move_to_gate_command(camera_data)
        return control_command # Ordered as array with: [pos_x_cmd, pos_y_cmd, pos_z_cmd, yaw_cmd] in meters and radians


    def get_target_gate(self, camera_data):
        gates = self.locate_gates(camera_data)
        
        if gates is None:
            #Turn to left
            return None
        # Check if more than 1 gate 
        if len(gates) == 1: # Found a gate so now try and take a 2nd photo 
            target_gate = gates[0] 
        else:
            # Multiple gates — pick rightmost by center x - TODO: maybe this should be based on world coords not camera
            target_gate = max(gates, key=lambda g: g[:, 0].mean())
        
        if self.is_target_gate_too_close_to_left(camera_data, target_gate):
            return None # Need to turn to left to make sure full gate in view

        return target_gate

    def is_target_gate_too_close_to_left(self, camera_data, target_gate):
        image_width = camera_data.shape[1]

        touching_left = any(x <= edge_threshold for x, y in target_gate)
        touching_right = any(x >= image_width - edge_threshold for x, y in target_gate)

        return touching_left # TODO - also have check for touching right

    def get_search_gate_command(self, camera_data, sensor_data):
        target_gate = self.get_target_gate(camera_data) # Finds rightmost gate 
        if target_gate is None: # No gate in sight so rotate left 
            return [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw'] + search_gate_rotation] # Get it to try and go to 30 degrees

        # Store the target gate to show later 
        self.target_gate_detection_img = camera_data.copy()
        self.target_gate_detection_img = cv2.polylines(self.target_gate_detection_img, [target_gate], isClosed=True, color=(0, 0, 255), thickness=2)
        
        # Found a gate so store its pixels and try to take a 2nd photo 
        self.current_gate_first_pixels = target_gate
        self.current_gate_first_sensor_data = sensor_data
        self.mode = Mode.CAPTURE_SECOND_PHOTO
        self.target_yaw = sensor_data['yaw'] + take_second_photo_rotation
        return [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], self.target_yaw]

    def get_capture_second_photo_command(self, camera_data, sensor_data):
        if abs(sensor_data['yaw'] - self.target_yaw) < eps:
            # Capture 2nd photo of target gate:
            self.current_gate_second_pixels = self.get_target_gate(camera_data) #TODO - need check if it's not in frame
            if self.is_target_gate_too_close_to_left(camera_data, self.current_gate_second_pixels): # Rotate left more so it's completely in frame
                return [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], self.target_yaw]
            self.current_gate_second_sensor_data = sensor_data

            # Can now find global position of current gate 
            self.set_position_of_current_gate(self.current_gate_first_sensor_data, sensor_data)
            
            self.mode = Mode.FLY_THROUGH_GATE # Uses the set current gate position
        return [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], self.target_yaw]
    
    def get_fly_through_gate_command(self, sensor_data):
        # TODO - UP TO THIS!!!!!!!!!! - Calculate center of current gate (set by self.current_gate_number) and figure out the pos + yaw of what it would be to fly through this (set target a bit forward - might be okay to go through it head on no yaw??)
        # TODO - Similar to in get_capture_second_photo_command, if all x,y,z,yaw sensor data matches the center of the current gate then restart loop (set mode to search and increment gate number)
        
        # Get current gate 
        current_gate_points = self.gate_positions[self.current_gate_number]
        # Convert to numpy array
        pts = np.array(current_gate_points, dtype=float)

        # Compute centroid
        center = pts.mean(axis=0)
        print("Center of gate is: ", center[0], center[1], center[2])
        print("Current pos is: ", [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']])
        return [center[0], center[1], center[2], sensor_data['yaw']]
    
    def get_camera_position_in_world(self, sensor_data):
        """Get the world position of the camera given drone sensor data."""
        drone_pos = np.array([sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global']])
        
        # Rotate the camera offset from body frame to world frame
        R_body_to_world = self.get_rotation_matrix(sensor_data['roll'], sensor_data['pitch'], sensor_data['yaw'])
        cam_offset_world = R_body_to_world @ CAM_OFFSET_BODY
        
        return drone_pos + cam_offset_world
    
    def get_rotation_matrix(self, roll, pitch, yaw):
        """ZYX Euler angles to rotation matrix (body to world)."""
        cr, sr = np.cos(roll),  np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw),   np.sin(yaw)

        return np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,             cp*cr            ]
        ])

    def pixel_to_direction_vector(self, pixel_coords):
        """
        Convert a pixel (u, v) to a direction vector in camera frame.
        Moves origin from top-left to center, then uses focal length as z.
        pixel_coords: (u, v) in image coordinates
        """
        u, v = pixel_coords
        cx = CAM_WIDTH / 2   # principal point u0
        cy = CAM_WIDTH / 2   # principal point v0 — adjust if image is not square
        
        vx = u - cx  # x in camera frame
        vy = v - cy  # y in camera frame
        vz = F_PIXELS

        return np.array([vx, vy, vz], dtype=float)

    def triangulate_point(self, P, r, Q, s):
        """
        Triangulate a 3D point from two rays in world frame.
        P, Q: camera positions (world frame)
        r, s: direction vectors (world frame)
        Returns H: estimated 3D world position
        """
        # Solve using pseudoinverse: [r | -s] [lambda; mu] = Q - P
        A = np.column_stack([r, -s])
        b = Q - P
        params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        lam, mu = params

        F = P + lam * r
        G = Q + mu * s
        H = (F + G) / 2  # midpoint = best estimate of 3D point
        return H

    def set_position_of_current_gate(self, sensor_data_first, sensor_data_second):
        """
        Triangulate each corner of the gate using two photos.
        Stores result in self.gate_positions.
        """
        if self.current_gate_first_pixels is None or self.current_gate_second_pixels is None:
            print("Missing gate pixel data")
            return

        # Camera positions in world frame at each capture
        P = self.get_camera_position_in_world(sensor_data_first)
        Q = self.get_camera_position_in_world(sensor_data_second)

        # Rotation matrices: camera -> world, for each capture
        R1_body_to_world = self.get_rotation_matrix(sensor_data_first['roll'],  sensor_data_first['pitch'],  sensor_data_first['yaw'])
        R2_body_to_world = self.get_rotation_matrix(sensor_data_second['roll'], sensor_data_second['pitch'], sensor_data_second['yaw'])

        R1_cam_to_world = R1_body_to_world @ R_CAM_TO_BODY
        R2_cam_to_world = R2_body_to_world @ R_CAM_TO_BODY

        gate_3d_corners = []
        for px1, px2 in zip(self.current_gate_first_pixels, self.current_gate_second_pixels):
            # Direction vectors in camera frame
            v_cam1 = self.pixel_to_direction_vector(px1)
            v_cam2 = self.pixel_to_direction_vector(px2)

            # Rotate direction vectors to world frame
            r = R1_cam_to_world @ v_cam1
            s = R2_cam_to_world @ v_cam2

            # Normalize
            r = r / np.linalg.norm(r)
            s = s / np.linalg.norm(s)

            H = self.triangulate_point(P, r, Q, s)
            gate_3d_corners.append(tuple(H))
            print(f"  Corner world position: {H}")

        self.gate_positions.append(tuple(gate_3d_corners))

        # Clear stored pixel data for next gate
        self.current_gate_first_pixels = None
        self.current_gate_second_pixels = None

        print(f"Gate triangulated: {gate_3d_corners}")


    # def get_move_to_gate_command(self, camera_data): #TODO - this will be different for later laps
    #     #TODO - call locate gates, turn to left if there are none, choose rightmost if multiple, convert to world frame
    #     gates = self.locate_gates(camera_data)
        
    #     # Check to see which is next and convert to world
    #     if gates is None:
    #         #TODO - turn to left
    #         return
    #     #Check if more than 1 gate 
    #     if len(gates) == 1:
    #         target_gate = gates[0]
    #     else:
    #         # Multiple gates — pick rightmost by center x - TODO: maybe this should be based on world coords not camera
    #         target_gate = max(gates, key=lambda g: g[:, 0].mean())
        
    #     # Store the target gate to show later 
    #     self.target_gate_detection_img = camera_data.copy()
    #     self.target_gate_detection_img = cv2.polylines(self.target_gate_detection_img, [target_gate], isClosed=True, color=(0, 0, 255), thickness=2)
        
    #     # TODO - convert target_gate corners to world frame
    #     return

    def convert_pixel_to_world(self, pixel_coords):
        #TODO
        return

    def locate_gates(self, camera_data):
        # TODO - call get image and locate corners in image
        lower, upper = self.rgb_to_hsv_bounds(r=gates_r_value, g=gates_g_value, b=gates_b_value)
        gates = self.locate_pink_area(camera_data, lower, upper)

        if gates is None:
            return None

        # Filter to only gates with exactly 4 corners
        valid_gates = [g for g in gates if len(g) == 4]

        if not valid_gates:
            return None

        for points in valid_gates:
            self.gate_detection_img = camera_data.copy()
            self.gate_detection_img = cv2.polylines(self.gate_detection_img, [points], isClosed=True, color=(0, 255, 0), thickness=2)

        return valid_gates # list of polygons, or None
    
    def locate_pink_area(self, image, lower_pink, upper_pink, min_area=100):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_pink, upper_pink)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > min_area]

        if not contours:
            return None

        # Return a polygon for each contour
        gates = []
        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, epsilon, True)
            gates.append(polygon.reshape(-1, 2))

        return gates  # list of Nx2 arrays, one per gate

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

def show_detection():
    if _controller.gate_detection_img is not None:
        cv2.imshow("All Gate Detection", _controller.gate_detection_img)
        cv2.waitKey(1)
    if _controller.target_gate_detection_img is not None:
        cv2.imshow("Target Gate Detection", _controller.target_gate_detection_img)
        cv2.waitKey(1)

