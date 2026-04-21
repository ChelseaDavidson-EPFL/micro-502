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
    APPROACH_GATE = 5
    FLY_THROUGH_GATE = 6

# Pink gates
gates_r_value = 211
gates_g_value = 144
gates_b_value = 222
GATE_HEIGHT = 0.4 # In meters 

# Rotation amounts
take_second_photo_rotation = 0.05
search_gate_rotation = 0.15

# Thresholds
eps = 0.02 # Used to check if yaw is correct - Approx 1 degree
pos_eps = 0.1 # Allow 10cm variation in approaching gate
edge_threshold = 10  # Used to check if the detected gate is at the edge (pixels tolerance from edge)

# Camera constants
CAM_FOV = 1.5  # radians
CAM_WIDTH = 300  # pixels
CAM_HEIGHT = 300
CAM_OFFSET_BODY = np.array([0.03, 0, 0.01])  # camera offset from drone body
F_PIXELS = CAM_WIDTH / (2 * np.tan(CAM_FOV / 2))  # focal length in pixels (~161)

# Rotation from camera frame to body frame (from appendix: zcam=xdrone, xcam=-ydrone, ycam=-zdrone)
R_CAM_TO_BODY = np.array([
    [ 0,  0,  1],   # xbody =  zcam
    [-1,  0,  0],   # ybody = -xcam
    [ 0, -1,  0]    # zbody = -ycam
])

class MyAssignment:
    def __init__(self, ):
        # ---- INITIALISE YOUR VARIABLES HERE ----
        self.mode = Mode.TAKE_OFF
        self.has_taken_off = False
        self.gate_positions = [] # Store gates as tuples of tuples representing coords of the corners ((x, y, z), ... x 4)
        self.gate_centers = [] # Store just the centers of the gates for easy access when flying through
        self.gate_detection_img = None
        self.target_gate_detection_img = None
        self.current_gate_number = 0 # Doing zero indexing

        # Variables for approach gate state:
        self.measurement_target_pos = None
        self.measurement_target_yaw = None

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
                print("Mode: Search Gate")
        # Search for gate command
        elif (self.mode == Mode.SEARCH_GATE):
            control_command = self.get_search_gate_command(camera_data, sensor_data)  
        elif (self.mode == Mode.APPROACH_GATE):
            control_command = self.get_approach_gate_command(camera_data, sensor_data) 
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
        
        # Get initial rough estimation
        gate_corners = self.estimate_gate_position(target_gate, sensor_data)
        if gate_corners is None:
            return [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']]
            
        center = gate_corners.mean(axis=0)
        
        # 2. Calculate the vector from the drone to the gate
        drone_pos = np.array([sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global']])
        vec_to_gate = center - drone_pos
        
        # We only care about X/Y distance for the 0.8m offset
        dist_xy = np.hypot(vec_to_gate[0], vec_to_gate[1])
        
        # 3. Calculate position exactly 0.8m in front of the gate
        direction_xy = vec_to_gate[:2] / dist_xy
        target_x = center[0] - (direction_xy[0] * 0.8)
        target_y = center[1] - (direction_xy[1] * 0.8)
        
        # Aim the camera directly at the gate
        self.measurement_target_yaw = np.arctan2(vec_to_gate[1], vec_to_gate[0])
        
        # Set target pos (matching the gate's Z to ensure it's vertically centered)
        self.measurement_target_pos = np.array([target_x, target_y, center[2]])
        
        self.mode = Mode.APPROACH_GATE
        print("Mode: Approach Gate")
        return [self.measurement_target_pos[0], self.measurement_target_pos[1], self.measurement_target_pos[2], self.measurement_target_yaw]

    def get_approach_gate_command(self, camera_data, sensor_data):
        drone_pos = np.array([sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global']])
        
        # Check how far we are from our 0.8m measurement spot
        dist_to_target = np.linalg.norm(self.measurement_target_pos - drone_pos)
        
        # If we are within 10cm of the measurement point, take the new photo
        if dist_to_target < pos_eps:
            target_gate = self.get_target_gate(camera_data)
            
            if target_gate is not None:
                # Take final high-accuracy measurement
                gate_corners = self.estimate_gate_position(target_gate, sensor_data)
                
                if gate_corners is not None:
                    # Save the accurate corners and transition to fly through
                    self.gate_positions.append(gate_corners)
                    center = gate_corners.mean(axis=0)
                    self.gate_centers.append(center)
                    print(f"Final High-Accuracy Gate Center: {center}")
                    
                    self.mode = Mode.FLY_THROUGH_GATE
                    print("Mode: Fly Through Gate")
                    return [center[0], center[1], center[2], self.measurement_target_yaw]
            else:
                # Fallback: if the gate was somehow lost from frame, rotate slightly to find it
                return [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw'] + search_gate_rotation]

        # Not there yet, keep flying to the 0.8m mark
        return [self.measurement_target_pos[0], self.measurement_target_pos[1], self.measurement_target_pos[2], self.measurement_target_yaw]

    def estimate_gate_position(self, gate_pixels, sensor_data):
        """
        Estimates the global 3D position of all 4 gate corners from a single image,
        accounting for camera pitch/roll by casting rays into the world frame.
        Returns a list of 4 (x, y, z) numpy arrays matching the order of gate_pixels.
        """
        if len(gate_pixels) != 4:
            return None

        # 1. Get camera position and rotation in the world frame
        P_cam_world = self.get_camera_position_in_world(sensor_data)
        R_body_to_world = self.get_rotation_matrix(sensor_data['roll'], sensor_data['pitch'], sensor_data['yaw'])
        R_cam_to_world = R_body_to_world @ R_CAM_TO_BODY

        slopes = []
        world_rays = []
        
        # 2. Convert each pixel corner into a 3D ray in the world frame
        for px in gate_pixels:
            v_cam = self.pixel_to_direction_vector(px)
            v_world = R_cam_to_world @ v_cam
            world_rays.append(v_world)
            
            # Calculate the vertical slope of the ray (dz / dxy)
            norm_xy = np.hypot(v_world[0], v_world[1])
            if norm_xy < 1e-4:
                return None # Prevent division by zero if looking perfectly vertical
                
            slope = v_world[2] / norm_xy
            slopes.append(slope)

        if len(slopes) != 4:
            return None

        # 3. Sort a COPY of the slopes descending to find top and bottom edges.
        # (Using a copy ensures we don't lose the original corner order)
        sorted_slopes = sorted(slopes, reverse=True)
        m_top = (sorted_slopes[0] + sorted_slopes[1]) / 2.0
        m_bot = (sorted_slopes[2] + sorted_slopes[3]) / 2.0

        # 4. Calculate horizontal distance to the gate
        # Using the known 0.4m height difference between top and bottom edges
        slope_diff = m_top - m_bot
        if slope_diff < 1e-4:
            return None # Prevent division by zero 

        D_xy = GATE_HEIGHT / slope_diff

        # 5. Calculate the 3D position of EACH corner
        corner_positions = []
        for v_world in world_rays:
            norm_xy = np.hypot(v_world[0], v_world[1])
            
            # Scale the normalized ray by the computed horizontal distance
            corner_x = P_cam_world[0] + D_xy * (v_world[0] / norm_xy)
            corner_y = P_cam_world[1] + D_xy * (v_world[1] / norm_xy)
            corner_z = P_cam_world[2] + D_xy * (v_world[2] / norm_xy)
            
            corner_positions.append(np.array([corner_x, corner_y, corner_z]))

        # Returns a list of 4 points in world coordinates
        return np.array(corner_positions)

    def get_fly_through_gate_command(self, sensor_data):
        # TODO - UP TO THIS!!!!!!!!!! - Calculate center of current gate (set by self.current_gate_number) and figure out the pos + yaw of what it would be to fly through this (set target a bit forward - might be okay to go through it head on no yaw??)
        # TODO - Similar to in get_capture_second_photo_command, if all x,y,z,yaw sensor data matches the center of the current gate then restart loop (set mode to search and increment gate number)
        # Get current center
        center = self.gate_centers[self.current_gate_number]

        if (abs(sensor_data['x_global'] - center[0]) < pos_eps and
            abs(sensor_data['y_global'] - center[1]) < pos_eps and
            abs(sensor_data['z_global'] - center[2]) < pos_eps and
            abs((sensor_data['yaw'] - self.measurement_target_yaw + np.pi) % (2 * np.pi) - np.pi) < eps):
            # We are through the gate, move to next one
            self.current_gate_number += 1
            self.mode = Mode.SEARCH_GATE
            print("Mode: Search Gate")
        
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
        cy = CAM_HEIGHT  / 2   # principal point v0
        
        vx = u - cx  # x in camera frame
        vy = v - cy  # y in camera frame
        vz = F_PIXELS

        return np.array([vx, vy, vz], dtype=float)


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

